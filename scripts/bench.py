"""Time CubeSim on a 20-move scramble plus its inverse and profile the hot spots.

    python scripts/bench.py [--profile]
"""

import cProfile
import pstats
import sys
import time

import numpy as np

from rs3m_twin import SOLVED, CubeSim, invert_moves


def scramble(n: int = 20, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    moves, last = [], None
    while len(moves) < n:
        f = "URFDLB"[rng.integers(6)]
        if f != last:
            moves.append(f + ["", "'", "2"][rng.integers(3)])
            last = f
    return " ".join(moves)


def run() -> float:
    sim = CubeSim()
    s = scramble()
    t0 = time.perf_counter()
    sim.apply_moves(s)
    sim.apply_moves(invert_moves(s))
    dt = time.perf_counter() - t0
    assert sim.facelets() == SOLVED
    return dt


def main() -> None:
    if "--profile" in sys.argv:
        pr = cProfile.Profile()
        pr.runcall(run)
        pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
    dt = run()
    print(f"timestep {CubeSim().model.opt.timestep * 1e3:.2f} ms, 40 moves in {dt:.2f} s")


if __name__ == "__main__":
    main()
