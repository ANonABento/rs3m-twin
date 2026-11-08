import numpy as np
import pytest

from rs3m_twin import FaceSim, magnet_torque

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
    return -magnet_torque(np.radians(deg), sim.params.magnet_k, sim.params.magnet_peak_deg)


@pytest.mark.parametrize("off, turns", [(40, True), (50, False)])
def test_cut_angle(off, turns):
    sim = FaceSim()
    sim.set_angle("U", np.radians(off))
    tq = np.zeros(6)
    tq[0] = _hold(sim, "U", off)  # U is held off-grid
    tq[1] = 5e-3  # turn R
    track = {"pen": 0.0, "weld": 0.0}
    _run(sim, tq, 0.8, track, stop="R")
    r = np.degrees(sim.angle("R"))
    if turns:  # R cut the corner: pushed U home and took the shared pieces
        assert r >= 90, r
        assert sim.blockers("R") == set()
    else:
        assert abs(r) < 2, r
        assert abs(np.degrees(sim.angle("U")) - off) < 3
        assert sim.blockers("R") == {"U"}
    assert track["pen"] <= 0.5 * MM
    assert track["weld"] <= 0.5 * MM


@pytest.mark.parametrize("a, b", [("U", "R"), ("R", "F")])
def test_orthogonal_off_grid_blocks(a, b):
    sim = FaceSim()
    sim.set_angle(a, np.radians(50))
    ia, ib = "URFDLB".index(a), "URFDLB".index(b)
    tq = np.zeros(6)
    tq[ia] = _hold(sim, a, 50)
    tq[ib] = -5e-3
    track = {"pen": 0.0, "weld": 0.0}
    _run(sim, tq, 0.4, track)
    assert abs(np.degrees(sim.angle(b))) < 2
    assert abs(np.degrees(sim.angle(a)) - 50) < 3
    assert a in sim.blockers(b)
    assert track["pen"] <= 0.5 * MM
    assert track["weld"] <= 0.5 * MM
