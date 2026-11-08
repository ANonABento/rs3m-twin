from dataclasses import replace

import numpy as np
import pytest

from rs3m_twin import CubeParams, FaceSim
from rs3m_twin.magnets import dipole_torque, magnet_layout, magnet_moment


def test_48_magnets():
    cells, pos, mom = magnet_layout(CubeParams())
    assert len(cells) == 48
    kind = np.abs(cells).sum(1)
    assert (kind == 3).sum() == 24 and (kind == 2).sum() == 24  # 8 corners x 3, 12 edges x 2
    assert np.allclose(np.linalg.norm(mom, axis=1), 1)


def test_moment_n52():
    assert magnet_moment(CubeParams()) == pytest.approx(1.43 * np.pi * 1.5e-3**2 * 1.5e-3 / (4e-7 * np.pi))
    assert magnet_moment(replace(CubeParams(), magnet_scale=0.5)) == pytest.approx(magnet_moment(CubeParams()) / 2)


@pytest.mark.parametrize("axis", [(0, 0, 1), (1, 0, 0), (0, -1, 0)])
def test_snap_curve(axis):
    p = CubeParams()
    deg = np.arange(0.5, 45, 0.5)
    tau = dipole_torque(p, axis, np.radians(deg))
    assert dipole_torque(p, axis, 0.0) == pytest.approx(0, abs=1e-12)
    assert dipole_torque(p, axis, np.radians(45)) == pytest.approx(0, abs=1e-9)
    assert np.allclose(dipole_torque(p, axis, -np.radians(deg)), -tau, atol=1e-12)  # odd
    assert np.all(tau <= 0)  # restoring toward 0
    assert 5 <= deg[np.argmin(tau)] <= 25


def test_release_at_30_settles():
    sim = FaceSim()
    sim.set_angle("U", np.radians(30))
    sim.step(np.zeros(6), n=int(1.5 / sim.model.opt.timestep))
    assert abs(np.degrees(sim.angle("U"))) <= 1


def test_no_snap_without_magnets():
    sim = FaceSim(replace(CubeParams(), magnet_scale=0.0))
    assert sim.magnet(np.radians(20)) == 0
    sim.set_angle("U", np.radians(30))
    sim.step(np.zeros(6), n=int(1.0 / sim.model.opt.timestep))
    assert abs(np.degrees(sim.angle("U")) - 30) < 2


def test_sin4_kept():
    sim = FaceSim(replace(CubeParams(), magnet_model="sin4"))
    assert sim.magnet(np.radians(15)) == pytest.approx(-CubeParams().magnet_k)


def test_randomize_scales_magnets():
    r = CubeParams().randomize(3)
    assert r.magnet_scale != 1.0 and 0.8 <= r.magnet_scale <= 1.2
