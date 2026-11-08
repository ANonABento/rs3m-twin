"""MuJoCo digital twin of the MoYu RS3M Super."""

from .mjcf import build_mjcf
from .params import CubeParams

__all__ = ["CubeParams", "build_mjcf"]
