"""Spatial metapopulation SIR model with commuting, used to generate epidemic
trajectories that the surveillance models can be run on.

You don't need this module if you already have your own trajectories: the
surveillance functions only need an array of new infections per day per location.
"""

from typing import Optional

import numpy as np


def distance_kernel(distance_km, dist_exp: float, offset: float = 7) -> np.ndarray:
    """Relative commuting propensity as a function of distance.

    Distances are binned in 3 km steps; the kernel decays with the bin index ``k`` as
    ``1 / (1 + k / (dist_exp * offset)) ** offset``.
    """
    d_idx = np.digitize(np.asarray(distance_km, dtype=float), bins=np.arange(0, 301, 3)) + 1
    return 1.0 / (1.0 + d_idx / (dist_exp * offset)) ** offset


def mobility_matrix(
    distance_km: np.ndarray,
    population: np.ndarray,
    stay_home_frac,
    dist_exp: float,
    pop_exp: float,
    offset: float = 7,
    epsilon: float = -0.5,
) -> np.ndarray:
    """Build a commuting matrix from a distance-decay model with competing destinations.

    Parameters
    ----------
    distance_km : array (n_locations, n_locations)
        Pairwise distances between locations in kilometres.
    population : array (n_locations,)
        Population of each location.
    stay_home_frac : float or array (n_locations,)
        Fraction of each location's residents that works in their own location.
    dist_exp, pop_exp, offset, epsilon : float
        Model parameters: distance decay scale, origin-population exponent,
        decay shape, and competing-destinations exponent.

    Returns
    -------
    array (n_locations, n_locations)
        Normalised flux matrix with ``stay_home_frac`` on the diagonal. To use it in
        :func:`simulate_epidemic`, pass ``matrix.T * 0.5`` (half the day spent commuting).
    """
    population = np.asarray(population, dtype=float)
    stay_home_frac = np.broadcast_to(np.asarray(stay_home_frac, dtype=float), population.shape)

    # 1. Distance decay
    p_ij_mat = distance_kernel(distance_km, dist_exp, offset)

    # 2. Origin population multiplier
    flux_mat = p_ij_mat * (population[:, np.newaxis] ** pop_exp)
    flux_mat = np.nan_to_num(flux_mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(flux_mat, 0)

    # 3. Competing destinations (influx-based competition)
    col_sums = flux_mat.sum(axis=0)
    cd_mat = col_sums[:, np.newaxis] - flux_mat.T
    with np.errstate(divide="ignore", invalid="ignore"):
        modifier = np.nan_to_num(cd_mat ** epsilon, nan=0.0, posinf=0.0)
    flux_mat = flux_mat * modifier
    np.fill_diagonal(flux_mat, 0)

    # 4. Normalisation
    col_sums_final = flux_mat.sum(axis=0)
    col_sums_final[col_sums_final == 0] = 1.0
    flux_mat = (flux_mat / col_sums_final) * (1 - stay_home_frac)
    np.fill_diagonal(flux_mat, stay_home_frac)
    return flux_mat


def simulate_epidemic(
    seed_index: int,
    alpha: float,
    beta: float,
    mobility: np.ndarray,
    n_days: int,
    population: np.ndarray,
    tau: float = 1,
    extinction_threshold: float = 1000,
    poisson: bool = False,
    n_seed: int = 10,
    max_attempts: Optional[int] = 1000,
    rng=None,
) -> np.ndarray:
    """Simulate a commuting SIR epidemic seeded in one location.

    Each time step, residents travel according to ``mobility``, are exposed at their
    destination for half a step, then return home and are exposed there for the
    other half. Simulations that die out before reaching ``extinction_threshold``
    cumulative infections are discarded and rerun.

    The simulation **stops on the day cumulative infections reach**
    ``extinction_threshold``; all later days are zero. Set the threshold to the
    largest outbreak size you care about: a surveillance system that hasn't detected
    the outbreak by then returns ``None``.

    Parameters
    ----------
    seed_index : int
        Location where the epidemic starts with ``n_seed`` infected people.
    alpha : float
        Transmission rate per day (``R0 = alpha / beta``).
    beta : float
        Recovery rate per day.
    mobility : array (n_locations, n_locations)
        ``mobility[i, j]`` is the fraction of residents of ``i`` in ``j`` during a step.
    n_days : int
        Number of days to simulate.
    population : array (n_locations,)
        Population of each location.
    tau : float
        Time step in days (``1 / tau`` steps per day).
    extinction_threshold : float
        Cumulative infections at which the run counts as an outbreak and stops.
    poisson : bool
        Use Poisson noise (stochastic) instead of deterministic expected values.
    n_seed : int
        Number of initial infections.
    max_attempts : int or None
        Give up after this many runs that don't reach the threshold (None = never).
    rng : None, int or numpy.random.Generator
        Random seed or generator.

    Returns
    -------
    array (n_days, n_locations)
        New infections per day per home location.
    """
    rng = np.random.default_rng(rng)
    mobility = np.asarray(mobility)
    n_loc = mobility.shape[0]
    steps_per_day = int(round(1 / tau))

    def get_vals(x):
        return rng.poisson(x) if poisson else x

    dtype = np.int64 if poisson else np.float64

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        infections = np.zeros((n_days, n_loc), dtype=dtype)

        I_mat = np.zeros((n_loc, n_loc), dtype=dtype)
        S_mat = np.zeros((n_loc, n_loc), dtype=dtype)
        R_mat = np.zeros((n_loc, n_loc), dtype=dtype)

        # Everyone starts at home (diagonal)
        I_mat[seed_index, seed_index] = n_seed
        np.fill_diagonal(S_mat, population)
        S_mat[seed_index, seed_index] = max(0, S_mat[seed_index, seed_index] - n_seed)

        for day in range(n_days):
            if np.sum(I_mat) == 0:
                break

            for _ in range(steps_per_day):
                # Movement phase
                I_diag = np.diag(I_mat).copy()
                S_diag = np.diag(S_mat).copy()
                R_diag = np.diag(R_mat).copy()

                out_I = get_vals(I_diag[:, None] * mobility)
                out_S = get_vals(S_diag[:, None] * mobility)
                out_R = get_vals(R_diag[:, None] * mobility)

                np.fill_diagonal(I_mat, I_diag - out_I.sum(axis=1))
                np.fill_diagonal(S_mat, S_diag - out_S.sum(axis=1))
                np.fill_diagonal(R_mat, R_diag - out_R.sum(axis=1))

                I_mat = np.maximum(I_mat + out_I, 0)
                S_mat = np.maximum(S_mat + out_S, 0)
                R_mat = np.maximum(R_mat + out_R, 0)

                # SIR update at destinations
                N = I_mat.sum(axis=0) + S_mat.sum(axis=0) + R_mat.sum(axis=0)
                N[N == 0] = 1.0
                force_of_inf = alpha * 0.5 * tau * (I_mat.sum(axis=0) / N)
                new_inf_1 = get_vals(S_mat * force_of_inf)
                rec_1 = get_vals(beta * 0.5 * tau * I_mat)

                I_mat = np.maximum(I_mat + new_inf_1 - rec_1, 0)
                S_mat = np.maximum(S_mat - new_inf_1, 0)
                R_mat = np.maximum(R_mat + rec_1, 0)

                # Return home
                I_mat = np.diag(I_mat.sum(axis=1))
                S_mat = np.diag(S_mat.sum(axis=1))
                R_mat = np.diag(R_mat.sum(axis=1))

                # SIR update at home
                N = I_mat.sum(axis=0) + S_mat.sum(axis=0) + R_mat.sum(axis=0)
                N[N == 0] = 1.0
                force_of_inf_2 = alpha * 0.5 * tau * (I_mat.sum(axis=0) / N)
                new_inf_2 = get_vals(S_mat * force_of_inf_2)
                rec_2 = get_vals(beta * 0.5 * tau * I_mat)

                I_mat = np.maximum(I_mat + new_inf_2 - rec_2, 0)
                S_mat = np.maximum(S_mat - new_inf_2, 0)
                R_mat = np.maximum(R_mat + rec_2, 0)

                infections[day] += new_inf_1.sum(axis=1) + new_inf_2.sum(axis=1)

            if np.sum(infections) >= extinction_threshold:
                return infections

    raise RuntimeError(
        f"No outbreak reached {extinction_threshold} infections in {max_attempts} attempts"
    )
