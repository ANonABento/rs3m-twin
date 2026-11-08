from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rs3m_twin import video_fit  # noqa: E402
from rs3m_twin.params import CubeParams  # noqa: E402


def test_tracking_follows_rendered_angle(tmp_path):
    angles = np.linspace(-30, 60, 40)  # crosses 45: needs the 90-deg unwrap
    path = tmp_path / "turn.mp4"
    roi = video_fit.render_video(path, angles, fps=100)
    got, fps = video_fit.track_angle(path, roi)
    assert fps == pytest.approx(100)
    assert np.max(np.abs(got - angles)) < 1.5


def test_fit_recovers_params_from_synthetic_video(tmp_path):
    true = replace(CubeParams(), frictionloss=2.6e-4, damping=1.6e-5, magnet_scale=1.15)
    path = tmp_path / "snap.mp4"
    roi = video_fit.make_synthetic(path, true)
    angles, fps = video_fit.track_angle(path, roi)
    fit = video_fit.fit_video(angles, fps)  # starts from the defaults
    for f in video_fit.VIDEO_FIELDS:
        assert getattr(fit, f) == pytest.approx(getattr(true, f), rel=0.1), f
