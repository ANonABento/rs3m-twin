import numpy as np
import pytest

from rs3m_twin import SOLVED, CubeParams, CubeSim, FaceSim, build_mjcf, invert_moves

MM = 1e-3
P = CubeParams(mechanism="contact")


def _push_r(off, t):
    """U free at ``off`` deg, R driven with 5 mN*m; returns the sim and the worst piece overlap."""
    sim = FaceSim(P)
    sim.set_angle("U", np.radians(off))
    tq = np.zeros(6)
    tq[1] = 5e-3
    pen = 0.0
    for _ in range(int(t / (10 * sim.model.opt.timestep))):
        if np.degrees(sim.angle("R")) >= 90:
            break
        sim.step(tq, n=10)
        pen = max(pen, sim.penetration())
        assert not sim.catch  # jamming comes from contact only
    return sim, pen


def test_contact_cut_at_40_completes():
    sim, pen = _push_r(40, 1.0)
    assert np.degrees(sim.angle("R")) >= 90
    assert abs(np.degrees(sim.angle("U"))) <= 2
    assert pen <= 0.5 * MM


def test_contact_jams_at_60():
    sim, pen = _push_r(60, 3.0)
    assert abs(np.degrees(sim.angle("R"))) < 10
    assert pen <= 0.5 * MM


def test_contact_inner_parts_built():
    xml = build_mjcf(P, path=None)
    assert xml.count("_foot") == 8 and xml.count("_tongue") == 12 and xml.count("_cap") == 6
    assert "_foot" not in build_mjcf(CubeParams(), path=None)
    with pytest.raises(ValueError):
        build_mjcf(CubeParams(mechanism="gears"), path=None)


def test_contact_scramble_and_inverse():
    rng = np.random.default_rng(0)
    moves, last = [], None
    while len(moves) < 20:
        f = "URFDLB"[rng.integers(6)]
        if f != last:
            moves.append(f + ["", "'", "2"][rng.integers(3)])
            last = f
    scramble = " ".join(moves)
    sim = CubeSim(P)
    sim.apply_moves(scramble)
    assert sim.facelets() != SOLVED
    sim.apply_moves(invert_moves(scramble))
    assert sim.is_aligned()
    assert sim.facelets() == SOLVED
