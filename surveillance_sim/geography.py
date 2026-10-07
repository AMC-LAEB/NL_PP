"""Helpers to turn coordinates into the inputs the models need."""

from typing import List

import numpy as np


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between points, in the units of ``coords``.

    Use projected coordinates (e.g. metres in a national grid), not lat/lon.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def hospital_catchments(
    location_coords: np.ndarray,
    hospital_coords: np.ndarray,
    radius: float = 10_000,
) -> List[List[int]]:
    """For each location, list the hospitals within ``radius``.

    Locations with no hospital within ``radius`` are assigned their nearest one, so
    every location has at least one hospital. The result is the ``close_hospitals``
    argument of :func:`surveillance_sim.simulate_hospital_surveillance`.

    Parameters
    ----------
    location_coords : array (n_locations, 2)
        Projected x/y coordinates of the locations.
    hospital_coords : array (n_hospitals, 2)
        Projected x/y coordinates of the hospitals, in the same units.
    radius : float
        Catchment radius in the units of the coordinates (default 10 km in metres).
    """
    loc = np.asarray(location_coords, dtype=float)
    hosp = np.asarray(hospital_coords, dtype=float)
    dist = np.sqrt(((loc[:, None, :] - hosp[None, :, :]) ** 2).sum(axis=-1))

    nearest = np.argmin(dist, axis=1)
    catchments = []
    for i, row in enumerate(dist):
        close_idx = np.where(row < radius)[0]
        catchments.append(close_idx.tolist() if len(close_idx) else [int(nearest[i])])
    return catchments
