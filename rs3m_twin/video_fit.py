"""Fit turning feel (friction, damping, magnet strength) from a video of one face snapping.

Track one face's colour region with OpenCV to get its angle over time, then fit ``frictionloss``
(hinge dry friction), ``damping`` and ``magnet_scale`` so a :class:`FaceSim` snap released from the
first tracked angle matches the tracked curve.  The least-squares step is :func:`tune.fit_log`.

    python -m rs3m_twin.video_fit clip.mp4 --roi X Y W H [--face U] [--fps 960]
    python -m rs3m_twin.video_fit --make-synthetic out.mp4      # a test clip rendered from the sim
"""

import argparse
import itertools
import sys
from dataclasses import replace

import numpy as np

from .params import CubeParams
from .sim import FaceSim
from .tune import fit_log

VIDEO_FIELDS = ("frictionloss", "damping", "magnet_scale")
GRID = {"frictionloss": (1 / 3, 1, 3), "damping": (1 / 3, 1, 3), "magnet_scale": (0.7, 1, 1.4)}  # x init


def _cv2():
    import cv2  # optional dependency (opencv-python)

    return cv2


def simulate_snap(params: CubeParams, theta0_deg: float, t: np.ndarray, face: str = "U") -> np.ndarray:
    """Face angle (deg) at times ``t`` (s, from release) after letting go at rest at ``theta0_deg``."""
    sim = FaceSim(params)
    sim.set_angle(face, np.radians(theta0_deg))
    dt = params.timestep
    n = int(np.ceil(t[-1] / dt)) + 1
    out = np.empty(n + 1)
    out[0] = sim.angle(face)
    for i in range(n):
        sim.step(np.zeros(6))
        out[i + 1] = sim.angle(face)
    return np.interp(t, dt * np.arange(n + 1), np.degrees(out))


def render_video(path, angles_deg, fps: float, size: int = 240, colour=(0, 200, 255)) -> tuple[int, int, int, int]:
    """Write an MP4 of a coloured face square turning through ``angles_deg``; return the face ROI (x, y, w, h)."""
    cv2 = _cv2()
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    c, half = size / 2, size * 0.25
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    for a in angles_deg:
        r = np.radians(a)
        rot = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])  # image y points down: CCW on screen
        img = np.full((size, size, 3), 40, np.uint8)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (90, 90, 90), 6)  # a stand-in for the rest of the cube
        cv2.fillConvexPoly(img, np.round(corners @ rot.T + c).astype(np.int32), colour, lineType=cv2.LINE_AA)
        w.write(img)
    w.release()
    m = int(half * 1.5)
    return int(c - m), int(c - m), 2 * m, 2 * m


def track_angle(path, roi, tol: float = 60.0) -> tuple[np.ndarray, float]:
    """Return (angles in deg, unwrapped, CCW positive; fps) of the colour region inside ``roi``.

    The colour to follow is the median of the ROI's central 20 %, in the first frame.  The square's
    orientation is known only modulo 90 deg, so frames are unwrapped by continuity (needs slow motion).
    """
    cv2 = _cv2()
    x, y, w, h = roi
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    target, out = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y : y + h, x : x + w].astype(np.float32)
        if target is None:
            cy, cx = h // 2, w // 2
            target = np.median(crop[cy - h // 10 : cy + h // 10 + 1, cx - w // 10 : cx + w // 10 + 1].reshape(-1, 3), 0)
        mask = (np.linalg.norm(crop - target, axis=2) < tol).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            out.append(np.nan)
            continue
        (_, _, a) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        out.append(-a)  # image y is down
    cap.release()
    if not out:
        raise ValueError(f"no frames read from {path}")
    a = np.array(out, float)
    good = ~np.isnan(a)
    a = np.interp(np.arange(len(a)), np.flatnonzero(good), a[good])
    a = np.degrees(np.unwrap(np.radians(a) * 4) / 4)  # period 90 deg
    a -= 90.0 * np.round(a[0] / 90.0)  # start within 45 deg of the detent at 0
    return a, fps


def fit_video(angles_deg, fps: float, init: CubeParams | None = None, face: str = "U", fields=VIDEO_FIELDS) -> CubeParams:
    """Fit ``fields`` so a snap released at rest follows ``angles_deg`` (the release angle is fitted too).

    Friction and damping trade off, so a coarse grid around ``init`` picks the two best starts and
    :func:`tune.fit_log` refines each; the lower cost wins.
    """
    init = init or CubeParams()
    y = np.asarray(angles_deg, float)
    t = np.arange(len(y)) / fps
    fields = list(fields)

    def resid(p, theta0=y[0]):
        return simulate_snap(p, theta0, t, face) - y

    grid = []
    for f in itertools.product(*[GRID.get(k, (1.0,)) for k in fields]):
        p = replace(init, **{k: getattr(init, k) * g for k, g in zip(fields, f)})
        grid.append((float(np.mean(resid(p) ** 2)), p))
    grid.sort(key=lambda g: g[0])
    best = min((fit_log(resid, p, fields, extra=(y[0],)) for _, p in grid[:2]), key=lambda r: r[2])
    return best[0]


def make_synthetic(path, params: CubeParams | None = None, theta0_deg: float = 32.0, seconds: float = 0.6, fps: float = 500.0):
    """Render the sim's own snap to ``path``; return the ROI to track."""
    params = params or CubeParams()
    t = np.arange(int(seconds * fps)) / fps
    return render_video(path, simulate_snap(params, theta0_deg, t), fps)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--face", default="U")
    ap.add_argument("--fps", type=float, help="true capture rate (slow-motion files often report 30)")
    ap.add_argument("--make-synthetic", metavar="OUT")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.make_synthetic:
        print("roi", *make_synthetic(a.make_synthetic))
        return
    if not a.video or not a.roi:
        ap.error("need a video and --roi")
    angles, fps = track_angle(a.video, a.roi)
    p = fit_video(angles, a.fps or fps, face=a.face)
    for f in VIDEO_FIELDS:
        print(f"{f} = {getattr(p, f):.6g}")


if __name__ == "__main__":
    main()
