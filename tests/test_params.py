import copy

import pytest
import yaml

from rs3m_twin import ParamsError, load_params
from rs3m_twin.params import DEFAULT_PATH, validate


def test_defaults_match_reference():
    p = load_params()
    assert p["geometry"]["edge"] == 56.0
    assert p["mass"]["total"] == 77.0
    assert p["geometry"]["gap"] == 0.25
    assert p["geometry"]["edge_radius"] == 2.0
    assert p["magnets"] == {"snap_torque": 4.0, "capture_angle": 15.0}
    assert p["corner_cut"] == {"forward": 45.0, "reverse": 15.0}


@pytest.mark.parametrize("section,key", [("mass", "total"), ("magnets", "capture_angle"), ("damping", "layer")])
def test_missing_key_rejected(tmp_path, section, key):
    data = yaml.safe_load(DEFAULT_PATH.read_text())
    del data[section][key]
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(data))
    with pytest.raises(ParamsError, match=f"{section}.{key}"):
        load_params(f)


def test_missing_section_and_bad_type():
    data = yaml.safe_load(DEFAULT_PATH.read_text())
    d = copy.deepcopy(data)
    del d["friction"]
    with pytest.raises(ParamsError, match="friction.sliding"):
        validate(d)
    d = copy.deepcopy(data)
    d["geometry"]["edge"] = "big"
    with pytest.raises(ParamsError):
        validate(d)
