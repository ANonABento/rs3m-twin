"""Cube state API: reset, step, facelets, alignment and motorised move strings."""

import mujoco
import numpy as np

from .mjcf import FACE_AXES, FACES, faces_of
from .params import CubeParams
from .sim import QUARTER, FaceSim

SOLVED = "".join(f * 9 for f in FACES)

# Kociemba sticker grid per face: (row axis, col axis) in the cube frame,
# rows top->bottom and cols left->right as seen looking at that face from outside.
_GRID = {
    "U": ((0, -1, 0), (1, 0, 0)),
    "R": ((0, 0, -1), (0, 1, 0)),
    "F": ((0, 0, -1), (1, 0, 0)),
    "D": ((0, 1, 0), (1, 0, 0)),
    "L": ((0, 0, -1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0)),
}


def _facelet_index(normal, cell) -> int:
    f = next(g for g in FACES if tuple(FACE_AXES[g]) == tuple(normal))
    row_ax, col_ax = _GRID[f]
    r, c = int(np.dot(row_ax, cell)) + 1, int(np.dot(col_ax, cell)) + 1
    return 9 * FACES.index(f) + 3 * r + c


def parse_moves(moves: str) -> list[tuple[str, int]]:
    """``"R U2 F'"`` -> [(face, clockwise quarter turns)], e.g. [("R", 1), ("U", 2), ("F", -1)]."""
    out = []
    for tok in moves.split():
        f, mod = tok[0], tok[1:]
        if f not in FACES or mod not in ("", "'", "2"):
            raise ValueError(f"bad move {tok!r}")
        out.append((f, {"": 1, "'": -1, "2": 2}[mod]))
    return out


def invert_moves(moves: str) -> str:
    return " ".join(f + {1: "'", -1: "", 2: "2"}[q] for f, q in reversed(parse_moves(moves)))


class CubeSim(FaceSim):
    """FaceSim plus cube state: Kociemba facelets and motor-driven move sequences."""

    def __init__(self, params: CubeParams | None = None, torque: float = 5e-3):
        super().__init__(params)
        self.torque = torque  # motor torque cap, N*m
        self.kp, self.kd = 0.03, 5e-4  # motor PD gains (N*m/rad, N*m*s/rad)
        self._cell0 = dict(self.cells)  # solved-state cell: fixes each piece's sticker colours
        self._rot0 = {pc: self.data.xmat[i].reshape(3, 3).copy() for pc, i in self._piece_id.items()}

    def reset(self, state: str | None = None) -> None:
        """Back to solved; ``state`` may be a move string then applied (motorised) from solved."""
        self.__init__(self.params, self.torque)
        if state:
            self.apply_moves(state)

    def facelets(self) -> str:
        """54-char Kociemba string (URFDLB faces, 9 stickers each), read from piece poses."""
        out = ["?"] * 54
        p = self.params.pitch
        for pc, c in self._cell0.items():
            i = self._piece_id[pc]
            rot = self.data.xmat[i].reshape(3, 3) @ self._rot0[pc].T
            cell = np.rint(self.data.xpos[i] / p).astype(int)
            for f in faces_of(c):
                n = np.rint(rot @ np.array(FACE_AXES[f], float)).astype(int)
                out[_facelet_index(n, cell)] = f
        return "".join(out)

    def is_aligned(self, tol_deg: float | None = None) -> bool:
        tol = np.radians(self.params.align_tol_deg if tol_deg is None else tol_deg)
        return all(abs(self.angle(f) - np.round(self.angle(f) / QUARTER) * QUARTER) < tol for f in FACES)

    def turn(self, face: str, quarters: int, timeout: float = 1.0) -> None:
        """Motor-drive ``face`` by ``quarters`` clockwise quarter turns (capped PD on the hinge)."""
        i, dt, dof = FACES.index(face), self.model.opt.timestep, self.hinge_dof[face]
        target = self.home[face] - quarters * QUARTER  # clockwise = negative about the outward axis
        tol = np.radians(self.params.align_tol_deg)
        tq, t = np.zeros(6), 0.0
        while t < timeout:
            err = target - self.angle(face)
            tq[i] = np.clip(self.kp * err - self.kd * self.data.qvel[dof], -self.torque, self.torque)
            self.step(tq, n=10)
            t += 10 * dt
            if abs(err) < tol / 4 and abs(self.data.qvel[dof]) < 0.05:
                break
        self.step(np.zeros(6), n=40)  # let the magnet seat it
        if abs(self.angle(face) - target) > tol:
            raise RuntimeError(f"{face} turn stalled at {np.degrees(self.angle(face)):.1f} deg (target {np.degrees(target):.0f})")
        self.home[face] = target
        self._snap()

    def _snap(self) -> None:
        """Seat every hinge and piece exactly on the grid (removes drift between moves), then
        move layer membership to the pieces' new cells and re-zero the welds."""
        d, p = self.data, self.params.pitch
        for f in FACES:
            d.qpos[self.hinge_qpos[f]] = self.home[f]
        cells = {}
        for pc, i in self._piece_id.items():
            cell = np.rint(d.xpos[i] / p)
            rot = np.rint(d.xmat[i].reshape(3, 3) @ self._rot0[pc].T) @ self._rot0[pc]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            a = self.model.jnt_qposadr[self.model.body_jntadr[i]]
            d.qpos[a : a + 3] = cell * p
            d.qpos[a + 3 : a + 7] = quat
            cells[pc] = tuple(int(v) for v in cell)
        d.qvel[:] = 0
        self.cells = cells
        mujoco.mj_forward(self.model, d)
        for pc, f in self.owner.items():
            self._rebase(pc, f)

    def apply_moves(self, moves: str) -> None:
        """Drive a move string like ``"R U R' U'"`` through the face motors."""
        for f, q in parse_moves(moves):
            self.turn(f, q)
