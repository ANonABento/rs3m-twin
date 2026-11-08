import numpy as np

from rs3m_twin import SOLVED, CubeSim, invert_moves

R_STATE = "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"


def test_solved_facelets():
    sim = CubeSim()
    assert sim.facelets() == SOLVED == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    assert sim.is_aligned()


def test_r_move():
    sim = CubeSim()
    sim.apply_moves("R")
    assert sim.is_aligned()
    assert sim.facelets() == R_STATE
    sim.apply_moves("R'")
    assert sim.facelets() == SOLVED


def test_scramble_and_inverse():
    rng = np.random.default_rng(0)
    moves, last = [], None
    while len(moves) < 20:
        f = "URFDLB"[rng.integers(6)]
        if f != last:
            moves.append(f + ["", "'", "2"][rng.integers(3)])
            last = f
    scramble = " ".join(moves)
    sim = CubeSim()
    sim.apply_moves(scramble)
    assert sim.is_aligned()
    assert sim.facelets() != SOLVED
    sim.apply_moves(invert_moves(scramble))
    assert sim.is_aligned()
    assert sim.facelets() == SOLVED
    sim.reset()
    assert sim.facelets() == SOLVED
