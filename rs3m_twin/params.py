"""Physical constants of the cube, all in one place (SI units unless noted)."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class CubeParams:
    edge: float = 0.0555  # full cube edge, m
    mass: float = 0.077  # total mass, kg (MagLev)
    core_mass_frac: float = 0.10  # share of mass in the core
    centre_mass_frac: float = 0.20  # share of mass in the 6 centres
    chamfer: float = 0.0012  # bevel on cubelet collision shapes, m
    friction: float = 0.15
    damping: float = 2e-5  # hinge damping, N*m*s/rad
    armature: float = 1e-7  # hinge armature, kg*m^2
    frictionloss: float = 2e-4  # hinge dry friction, N*m
    magnet_k: float = 3e-3  # peak detent torque, N*m
    magnet_peak_deg: float = 15.0
    cut_fwd_deg: float = 45.0
    cut_rev_deg: float = 20.0
    align_tol_deg: float = 1.0  # a layer counts as aligned (may hand over pieces) within this, deg
    gap: float = 3e-5  # per-side clearance between cubelet collision shapes, m

    @property
    def pitch(self) -> float:
        return self.edge / 3.0

    def randomize(self, rng: np.random.Generator | int | None = None, scale: float = 0.2) -> "CubeParams":
        """Return a copy with friction, damping, magnet strength and mass each scaled by an
        independent factor drawn uniformly from [1 - scale, 1 + scale] (domain randomisation)."""
        rng = np.random.default_rng(rng)
        f = rng.uniform(1.0 - scale, 1.0 + scale, size=5)
        return replace(
            self,
            friction=self.friction * f[0],
            damping=self.damping * f[1],
            frictionloss=self.frictionloss * f[2],
            magnet_k=self.magnet_k * f[3],
            mass=self.mass * f[4],
        )
