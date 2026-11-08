import numpy as np
import pytest

from rs3m_twin import build_cubies
from rs3m_twin.geometry import bounds, decompose, rounded_box


@pytest.fixture(scope="module")
def parts():
    return build_cubies()


def test_count(parts):
    assert len(parts) == 27
    assert sum(p.name == "core" for p in parts) == 1


def test_bbox(parts):
    ext = np.ptp(bounds(parts), axis=0)
    assert np.allclose(ext, 56.0, atol=0.1)


def test_mass(parts):
    assert abs(sum(p.mass for p in parts) - 77.0) <= 1.0
    assert all(p.mass > 0 for p in parts)


def test_meshes_watertight(parts):
    assert all(p.mesh.is_watertight for p in parts)


def test_decompose_returns_hulls():
    hulls = decompose(rounded_box(18.0, 2.0))
    assert hulls and all(h.volume > 0 for h in hulls)
