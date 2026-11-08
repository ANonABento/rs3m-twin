import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from rs3m_twin import CubeParams, build_mjcf
from rs3m_twin.mjcf import COLOURS, FACE_AXES, INNER, bounding_box, cubelet_positions, faces_of, piece_name, rgba


def _model(p):
    xml = build_mjcf(p, path=None)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return xml, m, d


def _vis_ids(m):
    return [g for g in range(m.ngeom) if "_vis" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]


def test_default_is_stickerless():
    assert CubeParams().visual_style == "stickerless"
    with pytest.raises(ValueError):
        build_mjcf(CubeParams(visual_style="shiny"), path=None)


def test_visual_geoms_inert_and_massless():
    xml, m, _ = _model(CubeParams())
    ids = _vis_ids(m)
    assert len(ids) == 26 + 54  # one block per piece + one tile per outward face
    for g in ids:
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] == 1
    root = ET.fromstring(xml)
    assert root.find(".//default[@class='visual']/geom").get("mass") == "0"
    for el in root.iter("geom"):
        if "_vis" in el.get("name", ""):
            assert el.get("class") == "visual" and el.get("mass") is None  # inherits mass="0"
    _, md, _ = _model(CubeParams(visual_style="debug"))
    assert np.allclose(m.body_mass, md.body_mass) and np.allclose(m.body_inertia, md.body_inertia)


def test_collision_geoms_hidden_group():
    _, m, _ = _model(CubeParams())
    _, md, _ = _model(CubeParams(visual_style="debug"))
    for model, grp in ((m, 3), (md, 0)):
        pieces = [g for g in range(model.ngeom) if model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        assert len(pieces) == 26 and all(model.geom_group[g] == grp for g in pieces)
    assert not _vis_ids(md)


@pytest.mark.parametrize("p", [CubeParams(), CubeParams(edge=0.056, mass=0.072)])
def test_mass_and_bbox_unchanged(p):
    _, m, d = _model(p)
    assert abs(mujoco.mj_getTotalmass(m) - p.mass) <= 0.001
    lo, hi = bounding_box(m, d)
    assert np.all(np.abs((hi - lo) - p.edge) <= 0.0002)


def test_54_outward_faces_coloured():
    _, m, d = _model(CubeParams())
    centres, others = cubelet_positions()
    n = 0
    for kind, cells in (("centre", centres), ("cubelet", others)):
        for c in cells:
            name = piece_name(kind, c)
            blk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_vis")
            assert np.allclose(m.geom_rgba[blk], rgba(INNER))
            for f in faces_of(c):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_vis_{f}")
                assert g >= 0
                assert np.allclose(m.geom_rgba[g], rgba(COLOURS[f]))
                out = d.geom_xpos[g] - d.xpos[m.geom_bodyid[g]]
                assert np.dot(out / np.linalg.norm(out), FACE_AXES[f]) > 0.999  # tile sits on side f
                a, nv = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                v = m.mesh_vert[a : a + nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                depth = (v - d.xpos[m.geom_bodyid[g]]) @ np.array(FACE_AXES[f], float)
                assert depth.min() > 0.0090 and np.ptp(depth) < 2e-4  # flat tile lying on the outer face
                n += 1
    assert n == 54


def test_renders_a_frame():
    os.environ.setdefault("MUJOCO_GL", "egl")
    _, m, d = _model(CubeParams())
    try:
        r = mujoco.Renderer(m, height=120, width=160)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"GL unavailable: {e}")
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.22, 45, -30
    r.update_scene(d, camera=cam)
    img = r.render()
    assert img.shape == (120, 160, 3) and img.std() > 1
