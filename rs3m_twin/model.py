"""MJCF builder: ``build(mode)`` for mode in {"fast", "contact"} (SI units)."""
from __future__ import annotations

import numpy as np

from .geometry import build_cubies
from .params import load_params

MODES = ("fast", "contact")


def _mesh_asset(name: str, mesh) -> str:
    v = " ".join("%.6g" % x for x in (np.asarray(mesh.vertices) / 1000.0).ravel())
    f = " ".join(str(int(i)) for i in np.asarray(mesh.faces).ravel())
    return f'<mesh name="{name}" vertex="{v}" face="{f}"/>'


def build(mode: str = "fast", params: dict | None = None) -> str:
    """Return an MJCF XML string. Fast: rigid cubies under one cube body.
    Contact: every cubie is a free body with convex-decomposed collision hulls."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    p = params or load_params()
    fr = p["friction"]
    friction = f'{fr["sliding"]} {fr["torsional"]} {fr["rolling"]}'
    parts = build_cubies(p, decomp=(mode == "contact"))
    assets, bodies = [], []
    for pt in parts:
        assets.append(_mesh_asset(pt.name, pt.mesh))
        mass = pt.mass / 1000.0
        if mode == "fast":
            bodies.append(
                f'<body name="{pt.name}"><geom type="mesh" mesh="{pt.name}" mass="{mass:.6g}"/></body>')
            continue
        hulls = pt.hulls or [pt.mesh.convex_hull]
        vol = sum(h.volume for h in hulls)
        geoms = [f'<geom type="mesh" mesh="{pt.name}" contype="0" conaffinity="0" mass="0"/>']
        for i, h in enumerate(hulls):
            hn = f"{pt.name}_h{i}"
            assets.append(_mesh_asset(hn, h))
            geoms.append(f'<geom type="mesh" mesh="{hn}" mass="{mass * h.volume / vol:.6g}"/>')
        joint = "" if pt.name == "core" else f'<freejoint name="{pt.name}_free"/>'
        bodies.append(f'<body name="{pt.name}">{joint}{"".join(geoms)}</body>')
    inner = "".join(bodies)
    world = (f'<body name="cube" pos="0 0 0.1"><freejoint name="cube_free"/>{inner}</body>'
             if mode == "fast" else inner)  # freejoints must sit directly under worldbody
    return (f'<mujoco model="rs3m_{mode}"><compiler angle="radian" inertiafromgeom="true"/>'
            f'<option gravity="0 0 0"/><default><geom friction="{friction}"/></default>'
            f'<asset>{"".join(assets)}</asset><worldbody>{world}</worldbody></mujoco>')
