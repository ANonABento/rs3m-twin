import numpy as np
import pytest

from rs3m_twin import FaceSim, magnet_torque


@pytest.mark.parametrize("peak", [10.0, 15.0, 30.0])
def test_magnet_shape(peak):
    k = 3e-3
    deg = np.arange(-180, 180.01, 0.25)
    tau = magnet_torque(np.radians(deg), k, peak)
    for d in (-180, -90, 0, 90, 180):
        assert abs(magnet_torque(np.radians(d), k, peak)) < 1e-12
    pos = (deg > 0) & (deg < 45)
    assert np.all(tau[pos] < 0)  # restoring toward 0
    assert deg[pos][np.argmin(tau[pos])] == pytest.approx(peak)
    assert tau.min() == pytest.approx(-k)


def _settle(sim, face, t=1.5):
    sim.step(np.zeros(6), n=int(t / sim.model.opt.timestep))
    return np.degrees(sim.angle(face))


def test_quarter_turn_under_150ms():
    sim = FaceSim()
    tq = np.zeros(6)
    tq[0] = 5e-3  # 5 mN*m on U
    t, dt = 0.0, sim.model.opt.timestep
    while np.degrees(sim.angle("U")) < 90 and t < 0.3:
        sim.step(tq, n=10)
        t += 10 * dt
    assert t < 0.15, t
    assert all(w.number == 0 for w in sim.data.warning)


@pytest.mark.parametrize("face", ["U", "R", "F"])
def test_release_at_30_snaps_back(face):
    sim = FaceSim()
    sim.set_angle(face, np.radians(30))
    assert abs(_settle(sim, face)) <= 1


@pytest.mark.parametrize("face", ["U", "R", "B"])
def test_release_at_60_snaps_forward(face):
    sim = FaceSim()
    sim.set_angle(face, np.radians(60))
    assert abs(_settle(sim, face) - 90) <= 1
    assert all(w.number == 0 for w in sim.data.warning)
