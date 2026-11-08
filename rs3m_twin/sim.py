"""Face turning: layer hinges, weld membership and the magnet detent."""

import mujoco
import numpy as np

from .mjcf import FACE_AXES, FACES, build_mjcf, cubelet_positions, faces_of, piece_name
from .magnets import DipoleDetent
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
        self.catch: dict[str, tuple[float, str | None, float]] = {}  # face -> (target, driver, drive torque)
        self._piece_id = {pc: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pc) for pc in self.cells}
        self._layer_id = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"layer_{f}") for f in FACES}
        self._hq = np.array([self.hinge_qpos[f] for f in FACES])
        self._hd = np.array([self.hinge_dof[f] for f in FACES])
        self.catch_dt = 5e-4  # max step while a piece catch is active, s
        self._detent = DipoleDetent(self.params) if self.params.magnet_model == "dipole" else None
        mujoco.mj_forward(m, self.data)

    def magnet(self, theta):
        """Magnet torque (N*m) on a layer at ``theta`` (float or array): the dipole table, or the sin4 curve."""
        if self._detent is not None:
            return self._detent(theta)
        out = magnet_torque(theta, self.params.magnet_k, self.params.magnet_peak_deg)
        return out if np.ndim(theta) else float(out)

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

    @property
    def cells(self) -> dict:
        return self._cells

    @cells.setter
    def cells(self, value: dict) -> None:  # cache each layer's pieces; recomputed when pieces move
        self._cells = value
        self._layer = {f: [pc for pc, c in value.items() if f in faces_of(c)] for f in FACES}

    def owns_layer(self, face: str) -> bool:
        return all(self.owner[pc] == face for pc in self._layer[face])

    def blockers(self, face: str) -> set[str]:
        """Faces currently holding pieces of ``face``'s layer."""
        return {self.owner[pc] for pc in self._layer[face]} - {face}

    def _membership(self, ctrl: np.ndarray) -> np.ndarray:
        """Hysteretic membership: a layer keeps its pieces until another one asks for them
        while the owner is back within ``align_tol`` of a detent. A request is refused (the
        turn stalls) unless every blocking face is within its cut angle; if so, the request
        pushes the blocker home. Returns the motor torques actually applied."""
        tol = np.radians(self.params.align_tol_deg)
        ang = self.data.qpos[self._hq]
        nearest = np.round(ang / QUARTER) * QUARTER
        for f, a, n in zip(FACES, ang, nearest):
            if abs(a - n) < tol:
                self.home[f] = float(n)
        out = ctrl.copy()
        if self.params.mechanism == "contact":
            return self._contact_membership(ctrl)
        for g, (_, drv, u0) in list(self.catch.items()):
            if drv is None:  # released: pieces guide the layer into its nearest detent
                if self._settled(g, strict=True):
                    del self.catch[g]
            elif abs(ctrl[FACES.index(drv)]) < 0.1 * u0:  # drive let go: release the catch
                self.catch[g] = (np.round(self.angle(g) / QUARTER) * QUARTER, None, 0.0)
        for i, f in enumerate(FACES):
            if ctrl[i] == 0 or self.owns_layer(f):
                continue
            out[i] = 0.0  # stall until the layer is ours
            if not self._settled(f):
                continue
            bl = self.blockers(f)
            if self.active in bl and not self._settled(self.active):
                continue
            if not all(self.within_cut(g) for g in bl):
                for g in bl:  # past the cut limit: pieces jam, hold the blocker where it is
                    if g not in self.catch:
                        self.catch[g] = (self.angle(g), f, abs(ctrl[i]))
                continue
            ready = True
            for g in bl:
                if not self._settled(g):  # cut: the pieces push the blocker home
                    ready = False
                    self.catch[g] = (self.home[g], f, abs(ctrl[i]))
            if ready:
                for g in bl:
                    self.catch.pop(g, None)
                for pc, c in self.cells.items():
                    if f in faces_of(c) and self.owner[pc] != f:
                        self._give(pc, f)
                self.active = f  # keeps its pieces until it settles (hysteresis)
        return out

    def _contact_membership(self, ctrl: np.ndarray) -> np.ndarray:
        """``mechanism="contact"``: no catch torque and no stall. The motor always drives the pieces
        the layer holds; those press on the blockers' pieces through the inner-part collision
        shapes, which either cut the blocker home or jam. Once every blocker sits settled in a
        detent its pieces in this layer are handed over."""
        for i, f in enumerate(FACES):
            if ctrl[i] == 0 or self.owns_layer(f):
                continue
            bl = self.blockers(f)
            if all(self._settled(g) for g in bl):
                for pc in self._layer[f]:
                    if self.owner[pc] != f:
                        self._give(pc, f)
                self.active = f
        return ctrl.copy()

    def _settled(self, face: str, strict: bool = False) -> bool:
        """At its detent (within ``align_tol``) and nearly still: safe to hand pieces over.
        ``strict`` (dropping a released catch) asks for a quarter of that, so the lightly damped
        magnet detent does not ring the layer back out past ``align_tol``."""
        tol = np.radians(self.params.align_tol_deg) / (4 if strict else 1)
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
        nearest = np.round(theta / QUARTER) * QUARTER
        if abs(theta - nearest) < np.radians(self.params.align_tol_deg):  # else keep the detent it left
            self.home[face] = nearest
        mujoco.mj_forward(self.model, self.data)

    def step(self, face_torques=None, n: int = 1) -> None:
        """Advance ``n`` steps with motor torques (N*m, FACES order) plus magnets."""
        if face_torques is not None:
            self.cmd = np.asarray(face_torques, float).copy()
        cmd = getattr(self, "cmd", np.zeros(6))
        p = self.params
        d, cap = self.data, 3.0 * p.magnet_k  # pieces jam harder than the magnet pulls, but not instantly
        for _ in range(n):
            d.ctrl[:] = self._membership(cmd)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[self._hd] = self.magnet(d.qpos[self._hq])
            if not self.catch:
                mujoco.mj_step(self.model, d)
                continue
            # piece catch: a stiff explicit spring-damper with friction, only stable at small steps, so
            # while a catch is active the step is split into substeps of at most ``catch_dt``
            idx = np.array([FACES.index(g) for g in self.catch])
            target = np.array([t for t, _, _ in self.catch.values()])
            dof, qa = self._hd[idx], self._hq[idx]
            dt = self.model.opt.timestep
            k = int(np.ceil(dt / self.catch_dt - 1e-9))
            self.model.opt.timestep = dt / k
            for _ in range(k):
                d.qfrc_applied[self._hd] = self.magnet(d.qpos[self._hq])
                v = d.qvel[dof]
                tau = p.catch_k * (target - d.qpos[qa]) - p.catch_damping * v - p.catch_friction * np.tanh(v / 0.05)
                d.qfrc_applied[dof] += np.clip(tau, -cap, cap)
                mujoco.mj_step(self.model, d)
            self.model.opt.timestep = dt

