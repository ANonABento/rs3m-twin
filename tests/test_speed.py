import os
import sys
import time

from rs3m_twin import SOLVED, CubeSim, invert_moves

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from bench import scramble  # noqa: E402


def test_scramble_and_inverse_under_3s():
    sim = CubeSim()
    s = scramble()
    t0 = time.perf_counter()
    sim.apply_moves(s)
    sim.apply_moves(invert_moves(s))
    assert time.perf_counter() - t0 < 3.0
    assert sim.facelets() == SOLVED
