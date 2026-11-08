from dataclasses import replace

import pytest

from rs3m_twin.params import CubeParams
from rs3m_twin.tune import EXAMPLE_CSV, FIT_FIELDS, fit, load_csv, synthetic, write_csv

TRUE = replace(CubeParams(), magnet_k=4.2e-3, magnet_peak_deg=19.0, damping=3.1e-5,
               frictionloss=1.4e-4, cut_fwd_deg=38.0, cut_rev_deg=24.0)


def _close(got, want, tol=0.05):
    for f in FIT_FIELDS:
        assert getattr(got, f) == pytest.approx(getattr(want, f), rel=tol), f


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_fit_recovers_known_params(tmp_path, noise):
    path = tmp_path / "m.csv"
    write_csv(synthetic(TRUE, noise=noise, rng=1), path)
    _close(fit(path), TRUE)  # starts from the defaults, which differ by up to ~50%


def test_example_csv_fits():
    data = load_csv(EXAMPLE_CSV)
    assert set(data) == {"detent", "drag", "cut_fwd", "cut_rev"}
    _close(fit(data), TRUE)


def test_partial_data_keeps_other_params():
    x, y = synthetic(TRUE)["drag"]
    p = fit({"drag": (x, y)})
    assert p.damping == pytest.approx(TRUE.damping, rel=0.05)
    assert p.magnet_k == CubeParams().magnet_k
