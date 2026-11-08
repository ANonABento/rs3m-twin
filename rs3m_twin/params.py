"""Validating loader for params.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("params.yaml")

SCHEMA: dict[str, tuple[str, ...]] = {
    "geometry": ("edge", "gap", "edge_radius", "core_radius"),
    "mass": ("total", "core_fraction"),
    "friction": ("sliding", "torsional", "rolling"),
    "damping": ("layer",),
    "magnets": ("snap_torque", "capture_angle"),
    "corner_cut": ("forward", "reverse"),
}


class ParamsError(ValueError):
    """Raised when params are missing or invalid."""


def validate(data: Any) -> dict[str, dict[str, float]]:
    if not isinstance(data, dict):
        raise ParamsError("params must be a mapping")
    missing = []
    out: dict[str, dict[str, float]] = {}
    for section, keys in SCHEMA.items():
        sec = data.get(section)
        if not isinstance(sec, dict):
            missing.extend(f"{section}.{k}" for k in keys)
            continue
        out[section] = {}
        for k in keys:
            if k not in sec:
                missing.append(f"{section}.{k}")
                continue
            v = sec[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ParamsError(f"{section}.{k} must be a number, got {v!r}")
            if v < 0:
                raise ParamsError(f"{section}.{k} must be >= 0, got {v}")
            out[section][k] = float(v)
    if missing:
        raise ParamsError("missing keys: " + ", ".join(missing))
    g = out["geometry"]
    if g["edge"] <= 0 or out["mass"]["total"] <= 0:
        raise ParamsError("geometry.edge and mass.total must be > 0")
    pitch = (g["edge"] - 2 * g["gap"]) / 3
    if 2 * g["edge_radius"] >= pitch:
        raise ParamsError("edge_radius too large for cubie size")
    if not 0 <= out["mass"]["core_fraction"] < 1:
        raise ParamsError("mass.core_fraction must be in [0, 1)")
    return out


def load_params(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p) as f:
        return validate(yaml.safe_load(f))
