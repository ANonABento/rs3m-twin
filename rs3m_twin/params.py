"""Physical constants of the cube, all in one place (SI units unless noted)."""

from dataclasses import dataclass


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

    @property
    def pitch(self) -> float:
        return self.edge / 3.0
