"""Render a scramble of the twin to MP4, headless.

    MUJOCO_GL=egl python scripts/demo.py out.mp4 --moves "R U R' U'"

Uses EGL by default and falls back to OSMesa. Exits with code 77 when no GL context can be made.
"""

import argparse
import os
import sys

GL_UNAVAILABLE = 77


def _make_renderer(width, height):
    backends = [os.environ["MUJOCO_GL"]] if "MUJOCO_GL" in os.environ else ["egl", "osmesa"]
    errors = []
    for b in backends:
        os.environ["MUJOCO_GL"] = b
        os.environ["PYOPENGL_PLATFORM"] = b
        try:
            import mujoco

            return mujoco, b, lambda model: mujoco.Renderer(model, height=height, width=width)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{b}: {e}")
    raise RuntimeError("; ".join(errors))


def _write_mp4(path, frames, fps):
    try:
        import av
    except ImportError:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1)
        return
    with av.open(path, "w") as out:
        st = out.add_stream("libx264", rate=fps)
        st.height, st.width = frames[0].shape[:2]
        st.pix_fmt = "yuv420p"
        for fr in frames:
            out.mux(st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")))
        out.mux(st.encode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="scramble.mp4")
    ap.add_argument("--moves", default="R U F' L D2 B")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=None, help="randomise params with this seed")
    a = ap.parse_args(argv)
    try:
        mujoco, backend, make = _make_renderer(a.width, a.height)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from rs3m_twin import CubeParams, CubeSim

        params = CubeParams() if a.seed is None else CubeParams().randomize(a.seed)
        sim = CubeSim(params)
        renderer = make(sim.model)
    except Exception as e:  # noqa: BLE001
        print(f"GL unavailable: {e}", file=sys.stderr)
        return GL_UNAVAILABLE

    cam = mujoco.MjvCamera()
    cam.lookat[:] = 0
    cam.distance, cam.azimuth, cam.elevation = 0.22, 45, -30
    frames, every = [], max(1, round(1.0 / (a.fps * sim.model.opt.timestep)))
    counter = [0]
    base_step = sim.step

    def grab():
        renderer.update_scene(sim.data, camera=cam)
        frames.append(renderer.render().copy())

    def step(face_torques=None, n=1):
        for _ in range(n):
            base_step(face_torques, 1)
            face_torques = None
            counter[0] += 1
            if counter[0] % every == 0:
                grab()

    sim.step = step
    grab()
    sim.apply_moves(a.moves)
    while len(frames) < a.fps:  # hold the final frame so the clip is at least 1 s
        grab()
    _write_mp4(a.out, frames, a.fps)
    print(f"wrote {a.out}: {len(frames)} frames ({len(frames) / a.fps:.2f} s) via {backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
