import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rs3m_twin import CubeParams

FIELDS = ["friction", "damping", "frictionloss", "magnet_k", "mass"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", range(20))
def test_randomize_within_bounds(seed):
    base = CubeParams()
    r = base.randomize(np.random.default_rng(seed), scale=0.2)
    for f in FIELDS:
        ratio = getattr(r, f) / getattr(base, f)
        assert 0.8 <= ratio <= 1.2, (f, ratio)
    assert r.edge == base.edge and r.cut_fwd_deg == base.cut_fwd_deg


def test_randomize_seeds_differ_and_repeat():
    base = CubeParams()
    a, b = base.randomize(1), base.randomize(2)
    assert all(getattr(a, f) != getattr(b, f) for f in FIELDS)
    assert base.randomize(1) == a
    assert base.mass == CubeParams().mass  # original untouched


def test_demo_writes_mp4(tmp_path):
    out = tmp_path / "demo.mp4"
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), str(out), "--moves", "R U"],
                       capture_output=True, text=True, env=env, timeout=300)
    if r.returncode == 77:
        pytest.skip(r.stderr.strip()[-200:])
    assert r.returncode == 0, r.stderr
    av = pytest.importorskip("av")
    with av.open(str(out)) as c:
        st = c.streams.video[0]
        n = sum(1 for _ in c.decode(st))
        assert n / float(st.average_rate) >= 1.0
