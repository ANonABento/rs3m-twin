"""Face turning: layer hinges, weld membership and the magnet detent."""

import mujoco
import numpy as np

from .mjcf import FACE_AXES, FACES, build_mjcf, cubelet_positions, faces_of, piece_name
from .params import CubeParams

QUARTER = np.pi / 2


def magnet_torque(theta, k: float, peak_deg: float):
    """Detent torque -k*sin(4*phi), phi = theta warped so |tau| peaks at ``peak_deg``.

    Zero at every 90 deg multiple (stable) and at 45 deg offsets (unstable).
    """
    theta = np.asarray(theta, dtype=float)
    x = np.mod(theta + np.pi / 4, QUARTER) - np.pi / 4  # offset from nearest detent, [-45, 45) deg
    a, p = np.abs(x), np.radians(peak_deg)
    q = np.pi / 8  # where sin(4*phi) peaks
    phi = np.where(a <= p, a * q / p, q + (a - p) * q / (np.pi / 4 - p))
    return -k * np.sign(x) * np.sin(4 * phi)


class FaceSim:
    """Cube with 6 motorised layers; pieces ride their layer on soft welds."""

    def __init__(self, params: CubeParams | None = None):
        self.params = params or CubeParams()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params, path=None))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hinge_dof = {f: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"hinge_{f}")] for f in FACES}
        self.hinge_qpos = {f: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"hinge_{f}")] for f in FACES}
        centres, others = cubelet_positions()
        self.cells = {piece_name("centre", c): c for c in centres} | {piece_name("cubelet", c): c for c in others}
        mujoco.mj_forward(m, self.data)

    def _eq(self, piece: str, face: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"w_{piece}_{face}")

    def claim(self, face: str) -> None:
        """Weld every piece in ``face``'s layer to that layer (cube must be aligned)."""
        for piece, c in self.cells.items():
            fs = faces_of(c)
            if face in fs:
                for f in fs:
                    self.data.eq_active[self._eq(piece, f)] = f == face

    def angle(self, face: str) -> float:
        return float(self.data.qpos[self.hinge_qpos[face]])

    def set_angle(self, face: str, theta: float) -> None:
        """Pose ``face``'s layer and its pieces at ``theta`` (from solved), at rest."""
        self.claim(face)
        axis = np.array(FACE_AXES[face], float)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, axis, theta)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        self.data.qpos[self.hinge_qpos[face]] = theta
        for piece, c in self.cells.items():
            if face in faces_of(c):
                j = self.model.body_jntadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, piece)]
                a = self.model.jnt_qposadr[j]
                self.data.qpos[a : a + 3] = rot @ (np.array(c) * self.params.pitch)
                self.data.qpos[a + 3 : a + 7] = quat
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, face_torques=None, n: int = 1) -> None:
        """Advance ``n`` steps with motor torques (N*m, FACES order) plus magnets."""
        if face_torques is not None:
            self.data.ctrl[:] = face_torques
        p = self.params
        for _ in range(n):
            self.data.qfrc_applied[:] = 0
            for f in FACES:
                self.data.qfrc_applied[self.hinge_dof[f]] = magnet_torque(self.angle(f), p.magnet_k, p.magnet_peak_deg)
            mujoco.mj_step(self.model, self.data)
