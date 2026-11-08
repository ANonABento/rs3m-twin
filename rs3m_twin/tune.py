"""Fit CubeParams to measurements from a real cube (scipy.optimize.least_squares).

CSV columns: ``kind,x,value`` (header required, ``#`` lines ignored).

* ``detent``  x = face offset from aligned, deg; value = torque on the face, N*m
  (sign as measured: negative pulls back toward 0).  Fits ``magnet_k`` and ``magnet_peak_deg``.
* ``drag``    x = steady turn speed, rad/s; value = torque needed to hold that speed, N*m.
  Model ``frictionloss + damping * x``.  Fits ``frictionloss`` and ``damping``.
* ``cut_fwd`` / ``cut_rev``  value = largest neighbour offset, deg, at which a turn still
  went through (x unused).  Fits ``cut_fwd_deg`` / ``cut_rev_deg``.

Run ``python -m rs3m_twin.tune data.csv`` to print the fitted values.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .params import CubeParams
from .sim import magnet_torque

FIT_FIELDS = ("magnet_k", "magnet_peak_deg", "damping", "frictionloss", "cut_fwd_deg", "cut_rev_deg")
KINDS = ("detent", "drag", "cut_fwd", "cut_rev")
EXAMPLE_CSV = Path(__file__).with_name("data") / "synthetic_measurements.csv"


def load_csv(path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, list[tuple[float, float]]] = {k: [] for k in KINDS}
    with open(path, newline="") as f:
        for row in csv.DictReader(line for line in f if not line.lstrip().startswith("#")):
            kind = row["kind"].strip()
            if kind not in rows:
                raise ValueError(f"unknown measurement kind {kind!r}")
            x = row.get("x") or "0"
            rows[kind].append((float(x), float(row["value"])))
    return {k: (np.array([r[0] for r in v]), np.array([r[1] for r in v])) for k, v in rows.items() if v}


def predict(params: CubeParams, kind: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if kind == "detent":
        return magnet_torque(np.radians(x), params.magnet_k, params.magnet_peak_deg)
    if kind == "drag":
        return params.frictionloss + params.damping * x
    if kind == "cut_fwd":
        return np.full_like(x, params.cut_fwd_deg)
    if kind == "cut_rev":
        return np.full_like(x, params.cut_rev_deg)
    raise ValueError(kind)


def _fields_for(data) -> list[str]:
    need = {"detent": ("magnet_k", "magnet_peak_deg"), "drag": ("damping", "frictionloss"),
            "cut_fwd": ("cut_fwd_deg",), "cut_rev": ("cut_rev_deg",)}
    return [f for f in FIT_FIELDS if any(f in need[k] for k in data)]


def fit(data, init: CubeParams | None = None) -> CubeParams:
    """Return ``init`` with every param the data constrains replaced by its least-squares fit.

    ``data`` is a path or the dict from :func:`load_csv`.  Params are fitted in log space
    (all positive); each kind's residuals are scaled by its RMS so units don't matter.
    """
    if not isinstance(data, dict):
        data = load_csv(data)
    init = init or CubeParams()
    fields = _fields_for(data)
    scale = {k: max(np.sqrt(np.mean(y**2)), 1e-12) * np.sqrt(len(y)) for k, (_, y) in data.items()}

    def make(z):
        return replace(init, **{f: float(np.exp(v)) for f, v in zip(fields, z)})

    def resid(z):
        p = make(z)
        return np.concatenate([(predict(p, k, x) - y) / scale[k] for k, (x, y) in data.items()])

    z0 = np.log([getattr(init, f) for f in fields])
    lo = np.full(len(fields), -np.inf)
    hi = np.full(len(fields), np.inf)
    if "magnet_peak_deg" in fields:  # peak must sit strictly inside (0, 45) deg
        i = fields.index("magnet_peak_deg")
        lo[i], hi[i] = np.log(0.5), np.log(44.5)
        z0[i] = np.clip(z0[i], lo[i] + 1e-6, hi[i] - 1e-6)
    res = least_squares(resid, z0, bounds=(lo, hi), x_scale="jac")
    return make(res.x)


def synthetic(params: CubeParams, noise: float = 0.0, rng=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Measurements a cube with ``params`` would give, with relative Gaussian noise."""
    rng = np.random.default_rng(rng)
    xs = {"detent": np.linspace(-40, 40, 33), "drag": np.linspace(0.5, 40, 12),
          "cut_fwd": np.zeros(5), "cut_rev": np.zeros(5)}
    out = {}
    for k, x in xs.items():
        y = predict(params, k, x)
        y = y + noise * np.sqrt(np.mean(y**2)) * rng.standard_normal(y.shape)
        out[k] = (x, y)
    return out


def write_csv(data, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "x", "value"])
        for k, (x, y) in data.items():
            for a, b in zip(x, y):
                w.writerow([k, f"{a:.6g}", f"{b:.6g}"])


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    p = fit(argv[0] if argv else EXAMPLE_CSV)
    for f in FIT_FIELDS:
        print(f"{f} = {getattr(p, f):.6g}")


if __name__ == "__main__":
    main()
