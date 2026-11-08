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
        self.owner = {pc: faces_of(c)[0] for pc, c in self.cells.items()}
        self.home = dict.fromkeys(FACES, 0.0)
        self.active: str | None = None
        self._piece_id = {pc: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pc) for pc in self.cells}
        self._layer_id = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"layer_{f}") for f in FACES}
        mujoco.mj_forward(m, self.data)

    def _eq(self, piece: str, face: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"w_{piece}_{face}")

    def _rebase(self, piece: str, face: str) -> None:
        """Set weld ``piece``->``face`` relpose to the current relative pose (no snap)."""
        d, e = self.data, self._eq(piece, face)
        b1, b2 = self._piece_id[piece], self._layer_id[face]
        r1 = d.xmat[b1].reshape(3, 3)
        self.model.eq_data[e, 3:6] = r1.T @ (d.xpos[b2] - d.xpos[b1])
        q1 = np.zeros(4)
        mujoco.mju_negQuat(q1, d.xquat[b1])
        mujoco.mju_mulQuat(self.model.eq_data[e, 6:10], q1, d.xquat[b2])

    def _give(self, piece: str, face: str) -> None:
        for f in FACES:
            self.data.eq_active[self._eq(piece, f)] = f == face
        self._rebase(piece, face)
        self.owner[piece] = face

    def claim(self, face: str) -> None:
        """Weld every piece in ``face``'s layer to that layer (cube must be aligned)."""
        mujoco.mj_forward(self.model, self.data)
        for piece, c in self.cells.items():
            if face in faces_of(c):
                self._give(piece, face)

    def offset(self, face: str) -> float:
        """Angle of ``face`` from the detent it last rested at (rad)."""
        return self.angle(face) - self.home[face]

    def within_cut(self, face: str) -> bool:
        """True if ``face`` is close enough to aligned for a neighbour to cut past it."""
        x = np.degrees(self.offset(face))
        return -self.params.cut_rev_deg <= x <= self.params.cut_fwd_deg

    def owns_layer(self, face: str) -> bool:
        return all(self.owner[pc] == face for pc, c in self.cells.items() if face in faces_of(c))

    def blockers(self, face: str) -> set[str]:
        """Faces currently holding pieces of ``face``'s layer."""
        return {self.owner[pc] for pc, c in self.cells.items() if face in faces_of(c)} - {face}

    def _membership(self, ctrl: np.ndarray) -> np.ndarray:
        """Hysteretic membership: a layer keeps its pieces until another one asks for them
        while the owner is back within ``align_tol`` of a detent. A request is refused (the
        turn stalls) unless every blocking face is within its cut angle; if so, the request
        pushes the blocker home. Returns the motor torques actually applied."""
        tol = np.radians(self.params.align_tol_deg)
        for f in FACES:
            nearest = np.round(self.angle(f) / QUARTER) * QUARTER
            if abs(self.angle(f) - nearest) < tol:
                self.home[f] = nearest
        out = ctrl.copy()
        for i, f in enumerate(FACES):
            if ctrl[i] == 0 or self.owns_layer(f):
                continue
            out[i] = 0.0  # stall until the layer is ours
            if not self._settled(f):
                continue
            bl = self.blockers(f)
            if not all(self.within_cut(g) for g in bl) or self.active in bl and not self._settled(self.active):
                continue
            ready = True
            for g in bl:
                if not self._settled(g):  # cut: our push drives the blocker home (PD, capped)
                    ready = False
                    j, u = FACES.index(g), abs(ctrl[i])
                    v_want = -30.0 * self.offset(g)  # approach home without overshoot
                    push = 2.0 * u * (v_want - self.data.qvel[self.hinge_dof[g]])
                    out[j] = np.clip(push, -u, u)
            if ready:
                for pc, c in self.cells.items():
                    if f in faces_of(c) and self.owner[pc] != f:
                        self._give(pc, f)
                self.active = f  # keeps its pieces until it settles (hysteresis)
        return out

    def _settled(self, face: str) -> bool:
        """At its detent (within ``align_tol``) and nearly still: safe to hand pieces over."""
        tol = np.radians(self.params.align_tol_deg)
        return abs(self.offset(face)) < tol and abs(self.data.qvel[self.hinge_dof[face]]) < 20 * tol

    def weld_error(self) -> float:
        """Largest position error (m) over active piece welds."""
        d, err = self.data, 0.0
        for pc, f in self.owner.items():
            e = self._eq(pc, f)
            b1, b2 = self._piece_id[pc], self._layer_id[f]
            want = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ self.model.eq_data[e, 3:6]
            err = max(err, float(np.linalg.norm(want - d.xpos[b2])))
        return err

    def penetration(self) -> float:
        """Deepest current contact penetration (m, >= 0)."""
        n = self.data.ncon
        return float(max(0.0, -self.data.contact.dist[:n].min())) if n else 0.0

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
        self.home[face] = np.round(theta / QUARTER) * QUARTER
        mujoco.mj_forward(self.model, self.data)

    def step(self, face_torques=None, n: int = 1) -> None:
        """Advance ``n`` steps with motor torques (N*m, FACES order) plus magnets."""
        if face_torques is not None:
            self.cmd = np.asarray(face_torques, float).copy()
        cmd = getattr(self, "cmd", np.zeros(6))
        p = self.params
        for _ in range(n):
            self.data.ctrl[:] = self._membership(cmd)
            self.data.qfrc_applied[:] = 0
            for f in FACES:
                self.data.qfrc_applied[self.hinge_dof[f]] = magnet_torque(self.angle(f), p.magnet_k, p.magnet_peak_deg)
            mujoco.mj_step(self.model, self.data)
