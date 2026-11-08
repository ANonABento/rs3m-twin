import mujoco
import numpy as np
import pytest

from rs3m_twin.model import build


@pytest.mark.parametrize("mode", ["fast", "contact"])
def test_compiles_mass_bbox(mode):
    m = mujoco.MjModel.from_xml_string(build(mode))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert abs(m.body_mass.sum() * 1000 - 77.0) <= 1.0
    pts = []
    for g in range(m.ngeom):
        mid = m.geom_dataid[g]
        if mid < 0:
            continue
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append(v)
    pts = np.vstack(pts)
    assert np.allclose(np.ptp(pts, axis=0) * 1000, 56.0, atol=0.1)


def test_bad_mode():
    with pytest.raises(ValueError):
        build("slow")
