import mujoco
import numpy as np
import pytest

from rs3m_twin import CubeParams, build_mjcf
from rs3m_twin.mjcf import bounding_box


@pytest.fixture(params=[CubeParams(), CubeParams(edge=0.056, mass=0.072)])
def built(request, tmp_path):
    p = request.param
    path = tmp_path / "rs3m.xml"
    build_mjcf(p, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return p, model, data


def test_structure_and_no_warnings(built):
    _, model, data = built
    assert model.nbody == 1 + 1 + 6 + 6 + 20  # world, core, layers, centres, cubelets
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert all(w.number == 0 for w in data.warning)


def test_total_mass(built):
    p, model, _ = built
    assert abs(mujoco.mj_getTotalmass(model) - p.mass) <= 0.001


def test_bounding_box(built):
    p, model, data = built
    mujoco.mj_forward(model, data)
    lo, hi = bounding_box(model, data)
    assert np.all(np.abs((hi - lo) - p.edge) <= 0.0002)


def test_default_writes_models_dir():
    from rs3m_twin.mjcf import DEFAULT_PATH

    build_mjcf()
    assert DEFAULT_PATH.exists()
    mujoco.MjModel.from_xml_path(str(DEFAULT_PATH))


def test_committed_model_matches_standard():
    from rs3m_twin.mjcf import DEFAULT_PATH

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert abs(mujoco.mj_getTotalmass(model) - 0.078) <= 0.001
    lo, hi = bounding_box(model, data)
    assert np.all(np.abs((hi - lo) - 0.056) <= 0.0002)
