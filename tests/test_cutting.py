import mujoco
import numpy as np
import pytest

from rs3m_twin import CubeParams, FaceSim, build_mjcf, magnet_torque
from rs3m_twin.mjcf import DEFAULT_PATH, bounding_box

MM = 1e-3


def _run(sim, tq, t, track, stop=None):
    for _ in range(int(t / (10 * sim.model.opt.timestep))):
        if stop is not None and np.degrees(sim.angle(stop)) >= 90:
            return  # quarter turn done
        sim.step(tq, n=10)
        track["pen"] = max(track["pen"], sim.penetration())
        track["weld"] = max(track["weld"], sim.weld_error())


def _hold(sim, face, deg):
    """Torque that balances the magnet so ``face`` stays ``deg`` off-grid."""
    return -sim.magnet(np.radians(deg))


P = CubeParams()


def _check(track):
    assert track["pen"] <= 0.5 * MM
    assert track["weld"] <= 0.5 * MM


@pytest.mark.parametrize("off", [P.cut_fwd_deg - 5, -(P.cut_rev_deg - 5)])
def test_cut_inside_limit_completes(off):
    """U free just inside the cut limit: R's pieces catch U, push it home and R turns."""
    sim = FaceSim()
    sim.set_angle("U", np.radians(off))
    tq = np.zeros(6)
    tq[1] = 5e-3
    track = {"pen": 0.0, "weld": 0.0}
    _run(sim, tq, 0.8, track, stop="R")
    assert np.degrees(sim.angle("R")) >= 90
    assert abs(np.degrees(sim.angle("U"))) <= 2
    assert sim.blockers("R") == set()
    _check(track)


@pytest.mark.parametrize("off", [P.cut_fwd_deg + 5, -(P.cut_rev_deg + 5)])
def test_cut_past_limit_catches(off):
    """U past the limit: U is held where it is, R stalls; on release U seats in the nearest detent."""
    sim = FaceSim()
    sim.set_angle("U", np.radians(off))
    tq = np.zeros(6)
    tq[1] = 5e-3
    track = {"pen": 0.0, "weld": 0.0, "u": 0.0}
    for _ in range(40):
        _run(sim, tq, 0.01, track)
        track["u"] = max(track["u"], abs(np.degrees(sim.angle("U")) - off))
    assert abs(np.degrees(sim.angle("R"))) < 10
    assert track["u"] <= 5
    assert sim.blockers("R") == {"U"}
    tq[1] = 0.0
    _run(sim, tq, 1.0, track)
    assert abs(np.degrees(sim.angle("U")) - 90 * round(off / 90)) <= 1  # nearest detent
    _check(track)


@pytest.mark.parametrize("a, b", [("U", "R"), ("R", "F")])
def test_orthogonal_off_grid_blocks(a, b):
    sim = FaceSim()
    off = P.cut_fwd_deg + 5
    sim.set_angle(a, np.radians(off))
    ia, ib = "URFDLB".index(a), "URFDLB".index(b)
    tq = np.zeros(6)
    tq[ia] = _hold(sim, a, off)
    tq[ib] = -5e-3
    track = {"pen": 0.0, "weld": 0.0}
    _run(sim, tq, 0.4, track)
    assert abs(np.degrees(sim.angle(b))) < 2
    assert abs(np.degrees(sim.angle(a)) - off) < 3
    assert a in sim.blockers(b)
    assert track["pen"] <= 0.5 * MM
    assert track["weld"] <= 0.5 * MM


def test_catch_params_randomized():
    q = P.randomize(0)
    assert q.catch_k != P.catch_k and q.catch_damping != P.catch_damping and q.catch_friction != P.catch_friction


def test_committed_model():
    assert open(DEFAULT_PATH).read() == build_mjcf(CubeParams(), path=None)
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert abs(m.body_subtreemass[0] - 0.078) <= 1e-3
    lo, hi = bounding_box(m, d)
    assert np.all(np.abs(hi - lo - 0.056) <= 0.2 * MM), hi - lo
    vis = [g for g in range(m.ngeom) if "_vis" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    assert vis and all(m.geom_group[g] == 1 and m.geom_contype[g] == 0 for g in vis)
