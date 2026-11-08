"""Build the MJCF model of the cube."""

import itertools
from pathlib import Path

import numpy as np

from .params import CubeParams

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "rs3m.xml"


def _chamfer_box_verts(h: float, c: float) -> str:
    pts = set()
    for axis in range(3):
        for s in itertools.product((-1, 1), repeat=3):
            p = [s[i] * (h - c) for i in range(3)]
            p[axis] = s[axis] * h
            pts.add(tuple(p))
    return " ".join(f"{v:.6g}" for p in sorted(pts) for v in p)


def cubelet_positions():
    """Integer grid coords of the 26 visible cubelets, centres first."""
    cells = [c for c in itertools.product((-1, 0, 1), repeat=3) if any(c)]
    centres = [c for c in cells if sum(map(abs, c)) == 1]
    others = [c for c in cells if sum(map(abs, c)) > 1]
    return centres, others


FACES = "URFDLB"  # Kociemba order
FACE_AXES = {"U": (0, 0, 1), "R": (1, 0, 0), "F": (0, -1, 0), "D": (0, 0, -1), "L": (-1, 0, 0), "B": (0, 1, 0)}


def piece_name(kind: str, c) -> str:
    return f"{kind}_{''.join('npz'[v + 1] for v in c)}"


def faces_of(c) -> list[str]:
    """Faces (in FACES order) whose layer contains grid cell ``c``."""
    return [f for f in FACES if np.dot(FACE_AXES[f], c) == 1]


def build_mjcf(params: CubeParams | None = None, path: str | Path | None = DEFAULT_PATH) -> str:
    """Return the MJCF XML for ``params`` and write it to ``path`` (skip if None)."""
    p = params or CubeParams()
    pitch, h = p.pitch, p.pitch / 2
    centres, others = cubelet_positions()
    m_core = p.mass * p.core_mass_frac
    m_centre = p.mass * p.centre_mass_frac / len(centres)
    m_other = p.mass * (1 - p.core_mass_frac - p.centre_mass_frac) / len(others)
    i_box = lambda m: m * (2 * h) ** 2 / 6  # noqa: E731
    r_core = 0.6 * h

    bodies, welds = [], []
    for kind, cells, m in (("centre", centres, m_centre), ("cubelet", others, m_other)):
        for c in cells:
            name = piece_name(kind, c)  # n=-1, z=0, p=+1
            pos = " ".join(f"{v * pitch:.6g}" for v in c)
            bodies.append(
                f'      <body name="{name}" pos="{pos}">\n'
                f'        <freejoint name="{name}"/>\n'
                f'        <inertial pos="0 0 0" mass="{m:.6g}" diaginertia="{i_box(m):.4g} {i_box(m):.4g} {i_box(m):.4g}"/>\n'
                f'        <geom name="{name}" type="mesh" mesh="cubelet" class="piece"/>\n'
                f"      </body>"
            )
            for f in faces_of(c):  # one soft weld per candidate layer; one active at a time
                active = "true" if f == faces_of(c)[0] else "false"
                welds.append(
                    f'    <weld name="w_{name}_{f}" body1="{name}" body2="layer_{f}" active="{active}" solref="0.002 1"/>'
                )
    i_core = 0.4 * m_core * r_core**2
    i_layer = 1e-9
    layers = "\n".join(
        f'      <body name="layer_{f}">\n'
        f'        <inertial pos="0 0 0" mass="1e-6" diaginertia="{i_layer} {i_layer} {i_layer}"/>\n'
        f'        <joint name="hinge_{f}" type="hinge" axis="{" ".join(map(str, a))}" damping="{p.damping:.6g}"'
        f' armature="{p.armature:.6g}" frictionloss="{p.frictionloss:.6g}"/>\n'
        f"      </body>"
        for f, a in FACE_AXES.items()
    )
    motors = "\n".join(f'    <motor name="motor_{f}" joint="hinge_{f}" ctrlrange="-0.05 0.05"/>' for f in FACES)
    xml = f"""<mujoco model="rs3m">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 0"/>  <!-- core is held; gravity only loads the welds -->
  <asset>
    <mesh name="cubelet" vertex="{_chamfer_box_verts(h - p.gap, p.chamfer)}"/>
  </asset>
  <default>
    <default class="piece">
      <geom friction="{p.friction:.6g}" margin="0" rgba="0.1 0.1 0.1 1"/>
    </default>
  </default>
  <worldbody>
    <body name="core">
      <inertial pos="0 0 0" mass="{m_core:.6g}" diaginertia="{i_core:.4g} {i_core:.4g} {i_core:.4g}"/>
      <geom name="core" type="sphere" size="{r_core:.6g}" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
{layers}
    </body>
{chr(10).join(bodies)}
  </worldbody>
  <equality>
{chr(10).join(welds)}
  </equality>
  <actuator>
{motors}
  </actuator>
</mujoco>
"""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml)
    return xml


def bounding_box(model, data) -> np.ndarray:
    """World-frame (min, max) over all mesh vertices; call after mj_forward."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        mid = model.geom_dataid[g]
        if model.geom_type[g] != 7 or mid < 0:  # mjGEOM_MESH
            continue
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a : a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return np.stack([lo, hi])
