"""Procedural cubie meshes (26 cubies + core) in mm and grams."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .params import load_params


@dataclass
class Part:
    name: str
    index: tuple[int, int, int]  # (-1..1)^3, core is (0,0,0)
    mesh: trimesh.Trimesh        # in cube frame, mm
    mass: float                  # g
    hulls: list[trimesh.Trimesh] = field(default_factory=list)


def rounded_box(size: float, radius: float) -> trimesh.Trimesh:
    """Convex hull of 8 corner spheres = box with rounded edges/corners."""
    if radius <= 0:
        return trimesh.creation.box(extents=[size] * 3)
    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    h = size / 2 - radius
    pts = [s.vertices + np.array(c) for c in itertools.product((-h, h), repeat=3)]
    return trimesh.convex.convex_hull(np.vstack(pts))


def decompose(mesh: trimesh.Trimesh, threshold: float = 0.05) -> list[trimesh.Trimesh]:
    """Convex decomposition with coacd; falls back to the convex hull if unavailable."""
    try:
        import coacd
    except ImportError:
        return [mesh.convex_hull]
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def build_cubies(params: dict | None = None, decomp: bool = False) -> list[Part]:
    p = params or load_params()
    g = p["geometry"]
    size = (g["edge"] - 2 * g["gap"]) / 3
    step = size + g["gap"]
    parts: list[Part] = []
    for idx in itertools.product((-1, 0, 1), repeat=3):
        if idx == (0, 0, 0):
            continue
        m = rounded_box(size, g["edge_radius"])
        m.apply_translation(np.array(idx) * step)
        parts.append(Part("cubie_%d_%d_%d" % idx, idx, m, 0.0))
    core = trimesh.creation.icosphere(subdivisions=3, radius=g["core_radius"])
    parts.append(Part("core", (0, 0, 0), core, 0.0))

    total = p["mass"]["total"]
    core_m = total * p["mass"]["core_fraction"]
    vol = sum(pt.mesh.volume for pt in parts[:-1])
    for pt in parts[:-1]:
        pt.mass = (total - core_m) * pt.mesh.volume / vol
    parts[-1].mass = core_m
    if decomp:
        for pt in parts:
            pt.hulls = decompose(pt.mesh)
    return parts


def bounds(parts: list[Part]) -> np.ndarray:
    return trimesh.util.concatenate([pt.mesh for pt in parts]).bounds
