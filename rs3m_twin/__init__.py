"""MuJoCo digital twin of the MoYu RS3M Super."""

from .mjcf import build_mjcf
from .params import CubeParams
from .sim import FaceSim, magnet_torque

__all__ = ["CubeParams", "FaceSim", "build_mjcf", "magnet_torque"]
