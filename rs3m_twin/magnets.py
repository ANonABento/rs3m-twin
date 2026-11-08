"""Point-dipole model of the Standard's 48 magnets (8 corners x 3 + 12 edges x 2).

Each corner carries one magnet on each face that meets an edge; each edge carries
one on each of its two faces that meet a corner.  Moments point from corner to
edge on both sides of every interface, so a lined-up pair is head-to-tail and
attracts in every cube state.  Magnets are point dipoles.  Only corner-edge pairs
interact (corner-corner and edge-edge never face each other), and pairs farther
apart than ``magnet_cutoff`` are ignored.
"""

import itertools

import numpy as np

MU0 = 4e-7 * np.pi


def magnet_moment(params) -> float:
    """Dipole moment (A*m^2) of one cylinder: Br * V / mu0, times ``magnet_scale``."""
    vol = np.pi * (params.magnet_diameter / 2) ** 2 * params.magnet_height
    return params.magnet_scale * params.magnet_br * vol / MU0


def magnet_layout(params):
    """Return (cells, positions, unit moments): one row per magnet, solved cube, metres."""
    p, c = params.pitch, params.magnet_inset * params.pitch
    half = params.magnet_gap / 2
    cells, pos, mom = [], [], []
    for cell in itertools.product((-1, 0, 1), repeat=3):
        nz = sum(map(abs, cell))
        if nz == 3:  # corner: one magnet per axis, facing the edge along -axis
            for a in range(3):
                x = np.array(cell, float) * c
                x[a] = cell[a] * (p / 2 + half)
                u = np.zeros(3)
                u[a] = -cell[a]
                cells.append(cell)
                pos.append(x)
                mom.append(u)
        elif nz == 2:  # edge: magnets on both faces along its zero axis
            a = cell.index(0)
            for s in (-1, 1):
                x = np.array(cell, float) * c
                x[a] = s * (p / 2 - half)
                u = np.zeros(3)
                u[a] = -s  # from the corner (at +s) into the edge
                cells.append(cell)
                pos.append(x)
                mom.append(u)
    return np.array(cells), np.array(pos), np.array(mom)


def _field_force(r, m1, m2):
    """Field of dipoles m1 at m2 (offset r from m1 to m2) and the force on m2, row-wise."""
    d = np.linalg.norm(r, axis=1, keepdims=True)
    n = r / d
    k = MU0 / (4 * np.pi)
    m1n = np.sum(m1 * n, 1, keepdims=True)
    m2n = np.sum(m2 * n, 1, keepdims=True)
    b = k * (3 * n * m1n - m1) / d**3
    f = 3 * k / d**4 * (m1 * m2n + m2 * m1n + n * np.sum(m1 * m2, 1, keepdims=True) - 5 * n * m1n * m2n)
    return b, f


def _rot(axis, th):
    a = np.asarray(axis, float)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def dipole_torque(params, axis, theta):
    """Torque (N*m) about ``axis`` on the layer turned by ``theta``, rest of the cube solved."""
    axis = np.asarray(axis, float)
    cells, pos, mom = magnet_layout(params)
    mom = mom * magnet_moment(params)
    moving = cells @ axis == 1
    kind = np.abs(cells).sum(1)  # 3 corner, 2 edge: only corner-edge faces ever meet
    out = []
    for th in np.atleast_1d(theta):
        r = _rot(axis, th)
        pm, mm = pos[moving] @ r.T, mom[moving] @ r.T
        pf, mf = pos[~moving], mom[~moving]
        i, j = (g.ravel() for g in np.meshgrid(np.arange(len(pm)), np.arange(len(pf)), indexing="ij"))
        rr = pm[i] - pf[j]
        keep = (np.linalg.norm(rr, axis=1) <= params.magnet_cutoff) & (kind[moving][i] != kind[~moving][j])
        i, j, rr = i[keep], j[keep], rr[keep]
        if not len(i):
            out.append(0.0)
            continue
        b, f = _field_force(rr, mf[j], mm[i])
        tau = np.cross(pm[i], f) + np.cross(mm[i], b)
        out.append(float(tau.sum(0) @ axis))
    return np.array(out) if np.ndim(theta) else out[0]


class DipoleDetent:
    """Dipole torque tabulated over one quarter turn and interpolated.

    The layout is the same around every face axis, so one table serves all six.
    """

    def __init__(self, params, n: int = 361):
        self.deg = np.linspace(-45.0, 45.0, n)
        self.tau = dipole_torque(params, (0, 0, 1), np.radians(self.deg))

    def __call__(self, theta):
        """Torque at ``theta`` (rad); a float, or an array for an array of angles."""
        x = (np.degrees(theta) + 45.0) % 90.0 - 45.0
        out = np.interp(x, self.deg, self.tau)
        return out if np.ndim(theta) else float(out)
