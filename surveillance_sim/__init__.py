"""Simulate how quickly hospital- and GP-based surveillance detects an emerging outbreak."""

from .epidemic import distance_kernel, mobility_matrix, simulate_epidemic
from .geography import distance_matrix, hospital_catchments
from .surveillance import Detection, simulate_gp_surveillance, simulate_hospital_surveillance

__all__ = [
    "Detection",
    "distance_kernel",
    "distance_matrix",
    "hospital_catchments",
    "mobility_matrix",
    "simulate_epidemic",
    "simulate_gp_surveillance",
    "simulate_hospital_surveillance",
]

__version__ = "0.1.0"
