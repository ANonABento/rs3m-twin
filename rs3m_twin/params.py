"""Physical constants of the cube, all in one place (SI units unless noted)."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class CubeParams:
    edge: float = 0.056  # full cube edge, m (Super RS3 M, 56 mm)
    mass: float = 0.078  # total mass, kg (Super RS3 M Standard)
    core_mass_frac: float = 0.10  # share of mass in the core
    centre_mass_frac: float = 0.20  # share of mass in the 6 centres
    chamfer: float = 0.0012  # bevel on cubelet collision shapes, m
    friction: float = 0.15
    damping: float = 2e-5  # hinge damping, N*m*s/rad
    armature: float = 1e-7  # hinge armature, kg*m^2
    frictionloss: float = 2e-4  # hinge dry friction, N*m
    magnet_k: float = 3e-3  # peak detent torque, N*m
    magnet_peak_deg: float = 15.0
    magnet_model: str = "dipole"  # "dipole": 48 point dipoles (magnets.py); "sin4": the curve above
    magnet_scale: float = 1.0  # multiplies every dipole moment (true strength unknown, fitted later)
    magnet_br: float = 1.43  # remanence, T (N52, assumed)
    magnet_diameter: float = 3e-3  # m (assumed)
    magnet_height: float = 1.5e-3  # m (assumed)
    magnet_inset: float = 0.37  # magnet centre's in-plane distance from each turn axis, in pitches
    magnet_gap: float = 6e-3  # centre-to-centre distance of a lined-up pair, m (incl. plastic walls)
    magnet_cutoff: float = 8e-3  # ignore magnet pairs farther apart than this, m
    cut_fwd_deg: float = 50.0
    cut_rev_deg: float = 35.0
    catch_k: float = 0.5  # piece-catch spring, N*m/rad (beats magnet_k within a degree)
    catch_damping: float = 3e-3  # piece-catch damper, N*m*s/rad
    catch_friction: float = 1e-3  # piece-catch dry friction, N*m
    align_tol_deg: float = 1.0  # a layer counts as aligned (may hand over pieces) within this, deg
    timestep: float = 0.001  # s; largest that keeps every test stable with implicitfast (1.5/2 ms miss the detent check)
    gap: float = 1e-4  # per-side clearance between cubelet collision shapes, m (30 um jams at a 2 ms step)
    visual_style: str = "stickerless"  # "stickerless": coloured visual meshes; "debug": bare collision shapes
    vis_size: float = 0.0182  # visual block edge, m
    vis_round: float = 0.0022  # visual block edge rounding radius, m

    @property
    def pitch(self) -> float:
        return self.edge / 3.0

    def randomize(self, rng: np.random.Generator | int | None = None, scale: float = 0.2) -> "CubeParams":
        """Return a copy with friction, damping, magnet strength (magnet_k, magnet_scale), mass and the catch
        spring/damper/friction each scaled by an
        independent factor drawn uniformly from [1 - scale, 1 + scale] (domain randomisation)."""
        rng = np.random.default_rng(rng)
        f = rng.uniform(1.0 - scale, 1.0 + scale, size=9)
        return replace(
            self,
            friction=self.friction * f[0],
            damping=self.damping * f[1],
            frictionloss=self.frictionloss * f[2],
            magnet_k=self.magnet_k * f[3],
            mass=self.mass * f[4],
            catch_k=self.catch_k * f[5],
            catch_damping=self.catch_damping * f[6],
            catch_friction=self.catch_friction * f[7],
            magnet_scale=self.magnet_scale * f[8],
        )
