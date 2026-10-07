"""Surveillance models: how quickly does a surveillance system detect an outbreak?

Both functions take an epidemic trajectory ``out`` of shape ``(n_days, n_locations)``
holding the number of *new* infections per day per location, simulate the
surveillance process on top of it, and report the state of the epidemic on the
first day a case is detected.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np


class Detection(NamedTuple):
    """State of the epidemic on the day of first detection.

    Indexes like a list, so ``result[1]`` is the detection day.
    """

    cumulative_infections: int
    """Total infections across all locations up to and including the detection day."""
    detection_day: int
    """Index of the first day with at least one detected case."""
    n_locations_infected: int
    """Number of locations with at least one infection by the detection day."""


def _detection_result(out: np.ndarray, first_day: int) -> Detection:
    cum_infections = np.cumsum(out, axis=0)[first_day, :]
    return Detection(
        cumulative_infections=int(cum_infections.sum()),
        detection_day=int(first_day),
        n_locations_infected=int(np.sum(cum_infections > 0)),
    )


def simulate_hospital_surveillance(
    out: np.ndarray,
    close_hospitals: Sequence[Sequence[int]],
    n_hospitals: int,
    prop_hospitals: float,
    hosp_p: float,
    p_symp: float,
    rng=None,
) -> Optional[Detection]:
    """Simulate detection through a sentinel network of hospitals.

    A random subset of ``round(n_hospitals * prop_hospitals)`` hospitals takes part.
    Each infection in location ``i`` is hospitalised and tested with probability
    ``hosp_p * p_symp``, scaled by the fraction of location ``i``'s nearby hospitals
    that participate (patients are assumed to spread evenly over nearby hospitals).

    Parameters
    ----------
    out : array (n_days, n_locations)
        New infections per day per location.
    close_hospitals : list of lists of int, length n_locations
        For each location, the indices (0..n_hospitals-1) of hospitals its residents
        may attend. Every location needs at least one; see
        :func:`surveillance_sim.hospital_catchments` to build this from coordinates.
    n_hospitals : int
        Total number of hospitals.
    prop_hospitals : float in [0, 1]
        Fraction of hospitals participating in surveillance.
    hosp_p : float
        Probability that a symptomatic infection is hospitalised (and tested).
    p_symp : float
        Probability that an infection is symptomatic.
    rng : None, int or numpy.random.Generator
        Random seed or generator.

    Returns
    -------
    Detection or None
        ``None`` if no case is detected within the simulated period.
    """
    rng = np.random.default_rng(rng)
    out = np.asarray(out)
    if len(close_hospitals) != out.shape[1]:
        raise ValueError("close_hospitals must have one entry per location (column of `out`)")

    base_p = hosp_p * p_symp

    n_include = int(round(n_hospitals * prop_hospitals))
    included_hospitals = set(rng.choice(n_hospitals, n_include, replace=False).tolist())

    n_close = np.array([len(h_list) for h_list in close_hospitals])
    n_included = np.array([sum(h in included_hospitals for h in h_list) for h_list in close_hospitals])
    p_vec = base_p * n_included / n_close

    detected_matrix = rng.binomial(out.astype(int), p_vec)
    detection_days = np.where(detected_matrix.sum(axis=1) > 0)[0]

    if len(detection_days) == 0:
        return None
    return _detection_result(out, detection_days[0])


def simulate_gp_surveillance(
    out: np.ndarray,
    population: np.ndarray,
    background_weekly_rate: np.ndarray,
    n_gps: int,
    p_symp: float,
    healthcare_seeking_p: float,
    init_week: int,
    n_weekly_samples: float,
    gp_practice_size: int = 3626,
    rng=None,
) -> Optional[Detection]:
    """Simulate detection through a sentinel network of general practitioners (GPs).

    ``n_gps`` practices are placed at random, weighted by population. Each practice
    covers ``gp_practice_size`` people of its location. Every day:

    * variant infections in covered populations present at the GP with probability
      ``healthcare_seeking_p * p_symp``;
    * background (non-variant) influenza-like-illness patients present at a rate set
      by ``background_weekly_rate`` for the current week;
    * each GP swabs ``Poisson(n_weekly_samples / 7)`` of its presenting patients,
      drawn at random from the mix of variant and background patients.

    Detection happens on the first day at least one variant patient is swabbed.

    Parameters
    ----------
    out : array (n_days, n_locations)
        New infections per day per location.
    population : array (n_locations,)
        Population of each location.
    background_weekly_rate : array (n_weeks,)
        Per-person probability of a background ILI consultation in each week of the
        year. Indexed cyclically, so typically length 52.
    n_gps : int
        Number of participating GP practices.
    p_symp : float
        Probability that an infection is symptomatic.
    healthcare_seeking_p : float
        Probability that a symptomatic infection visits the GP.
    init_week : int
        Index into ``background_weekly_rate`` at which the epidemic (day 0) starts.
    n_weekly_samples : float
        Average number of patients swabbed per GP per week.
    gp_practice_size : int
        Number of people registered with a single practice.
    rng : None, int or numpy.random.Generator
        Random seed or generator.

    Returns
    -------
    Detection or None
        ``None`` if no case is detected within the simulated period.
    """
    rng = np.random.default_rng(rng)
    out = np.asarray(out)
    population = np.asarray(population, dtype=float)
    background_weekly_rate = np.asarray(background_weekly_rate, dtype=float)
    if population.shape[0] != out.shape[1]:
        raise ValueError("population must have one entry per location (column of `out`)")

    n_days = out.shape[0]
    base_prob = healthcare_seeking_p * p_symp

    # GP placement
    gp_locations = rng.choice(len(population), size=n_gps, p=population / population.sum(), replace=True)
    gp_counts = np.bincount(gp_locations, minlength=len(population))
    active_indices = np.where(gp_counts > 0)[0]

    active_gp_sizes = gp_counts[active_indices] * gp_practice_size
    gp_coverage = np.minimum(1.0, active_gp_sizes / population[active_indices])
    prob_presenting = gp_coverage * base_prob

    # Background ILI rate for every day at once
    week_indices = ((init_week * 7 + np.arange(n_days)) // 7) % len(background_weekly_rate)
    daily_bg_rates = background_weekly_rate[week_indices] / 7.0  # (n_days,)

    # Vectorised draws across all days and active locations simultaneously
    active_out = out[:, active_indices].astype(int)                                 # (n_days, n_active)
    n_variant = rng.binomial(active_out, prob_presenting)                           # (n_days, n_active)
    n_bg = rng.poisson(daily_bg_rates[:, None] * active_gp_sizes[None, :])          # (n_days, n_active)

    # Stochastic daily sampling: Poisson(weekly_budget / 7) tests per GP per day
    daily_sample_rate = n_weekly_samples / 7.0
    samples_taken = rng.poisson(
        daily_sample_rate * gp_counts[active_indices][None, :] * np.ones((n_days, 1))
    )                                                                               # (n_days, n_active)

    tot_presenting = n_variant + n_bg
    samples_taken = np.minimum(samples_taken, tot_presenting)

    # Hypergeometric draw, only where there are patients to sample from
    valid = (tot_presenting > 0) & (samples_taken > 0)
    n_detected = np.zeros_like(n_variant)
    if np.any(valid):
        n_detected[valid] = rng.hypergeometric(n_variant[valid], n_bg[valid], samples_taken[valid])

    detection_days = np.where(n_detected.sum(axis=1) > 0)[0]
    if len(detection_days) == 0:
        return None
    return _detection_result(out, detection_days[0])
