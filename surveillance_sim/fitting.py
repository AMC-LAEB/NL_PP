"""Fit the mobility model to observed commuting data with MCMC (emcee).

The model works at the level of towns, but commuting data is usually published per
region (e.g. municipality). The fit aggregates the modelled town-to-town flows to
regions and compares them with two observed tables:

* ``work_location``: rows = region of work, columns = region of residence,
  values = share of that column's residents working in each row region.
* ``home_location``: rows = region of residence, columns = region of work,
  values = share of that column's workers living in each row region.

Columns only need to cover a subset of regions (e.g. the larger cities). Shares can
be percentages or fractions; columns are renormalised after removing people who
live and work in the same region.

Requires the ``fit`` extra: ``pip install -e ".[fit]"``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .epidemic import distance_kernel, mobility_matrix

PARAM_NAMES = ["dist_exp", "pop_exp", "offset", "epsilon", "log_sigma"]

# Uniform prior bounds (lower, upper) per parameter
DEFAULT_PRIOR_BOUNDS = {
    "dist_exp": (0, 4),
    "pop_exp": (0, 2),
    "offset": (0, 10),
    "epsilon": (-2, 2),
    "log_sigma": (-8, 1),
}


def _clean_observed(table: pd.DataFrame) -> pd.DataFrame:
    """Drop within-region commuting and renormalise each column to sum to 1."""
    table = table.astype(float).copy()
    for label in table.index.intersection(table.columns):
        table.loc[label, label] = 0
    return table / table.sum(axis=0)


class MobilityLikelihood:
    """Compares the mobility model with observed regional commuting shares.

    Parameters
    ----------
    distance_km : array (n_towns, n_towns)
    population : array (n_towns,)
    town_region : array (n_towns,)
        Region label (matching the commuting tables) of each town.
    work_location, home_location : DataFrame
        Observed commuting tables, see module docstring.
    """

    def __init__(self, distance_km, population, town_region,
                 work_location: pd.DataFrame, home_location: pd.DataFrame):
        self.distance_km = np.asarray(distance_km, dtype=float)
        self.population = np.asarray(population, dtype=float)

        self.regions, region_idx = np.unique(np.asarray(town_region).astype(str), return_inverse=True)
        self.membership = np.zeros((len(self.population), len(self.regions)))
        self.membership[np.arange(len(self.population)), region_idx] = 1.0
        region_pos = {r: i for i, r in enumerate(self.regions)}

        work = _clean_observed(work_location)
        home = _clean_observed(home_location)
        cities = [c for c in home.columns if c in region_pos]
        if not cities:
            raise ValueError("No commuting-table columns match the town regions")

        def match(table):
            table = table.loc[table.index.intersection(list(region_pos)), cities]
            rows = np.array([region_pos[r] for r in table.index])
            cols = np.array([region_pos[c] for c in cities])
            return rows, cols, table.to_numpy()

        self._work_rows, self._work_cols, self.observed_work = match(work)
        self._home_rows, self._home_cols, self.observed_home = match(home)
        self.cities = cities

    def predicted(self, dist_exp, pop_exp, offset, epsilon):
        """Modelled ``(work_location, home_location)`` shares on the observed grid."""
        flux = mobility_matrix(self.distance_km, self.population, 0.0, dist_exp, pop_exp, offset, epsilon)
        np.fill_diagonal(flux, 0)
        # Extreme parameters can overflow; those give non-finite residuals and are rejected
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            flux = flux / flux.sum(axis=0) * self.population

            regional = self.membership.T @ flux @ self.membership
            np.fill_diagonal(regional, 0)
            home_model = regional / regional.sum(axis=0)
            work_model = (regional / regional.sum(axis=1)[:, None]).T

        work = work_model[np.ix_(self._work_rows, self._work_cols)]
        home = home_model[np.ix_(self._home_rows, self._home_cols)]
        return work, home

    def residuals(self, dist_exp, pop_exp, offset, epsilon):
        work, home = self.predicted(dist_exp, pop_exp, offset, epsilon)
        return np.concatenate([
            (np.log1p(self.observed_work) - np.log1p(work)).ravel(),
            (np.log1p(self.observed_home) - np.log1p(home)).ravel(),
        ])

    def log_likelihood(self, params):
        *model_params, log_sigma = params
        sigma = np.exp(log_sigma)
        res = self.residuals(*model_params)
        if np.any(~np.isfinite(res)):
            return -np.inf
        return -0.5 * np.sum((res / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


@dataclass
class MobilityFit:
    """Result of :func:`fit_mobility`."""

    chain: np.ndarray
    """Full chain, shape (n_steps, n_walkers, 5)."""
    log_prob: np.ndarray
    """Log posterior, shape (n_steps, n_walkers)."""
    burnin: int
    param_names: List[str] = field(default_factory=lambda: list(PARAM_NAMES))

    @property
    def samples(self) -> np.ndarray:
        """Flattened posterior samples after burn-in, shape (n_samples, 5)."""
        return self.chain[self.burnin:].reshape(-1, self.chain.shape[-1])

    def median(self) -> dict:
        """Posterior median of each parameter."""
        return dict(zip(self.param_names, np.median(self.samples, axis=0)))

    def to_dataframe(self) -> pd.DataFrame:
        """Posterior samples as a DataFrame (one column per parameter)."""
        return pd.DataFrame(self.samples, columns=self.param_names)


def fit_mobility(
    distance_km,
    population,
    town_region,
    work_location: pd.DataFrame,
    home_location: pd.DataFrame,
    n_walkers: int = 20,
    n_steps: int = 100,
    burnin: int = 20,
    prior_bounds: Optional[dict] = None,
    initial: Optional[np.ndarray] = None,
    rng=None,
    progress: bool = True,
) -> MobilityFit:
    """Fit ``dist_exp, pop_exp, offset, epsilon`` (and noise ``log_sigma``) with emcee.

    Uses uniform priors (``DEFAULT_PRIOR_BOUNDS`` unless overridden) and a Gaussian
    likelihood on ``log1p`` commuting shares. Walkers start uniformly over the
    prior (``log_sigma`` over [-5, 0]) unless ``initial`` (n_walkers, 5) is given.
    """
    import emcee

    rng = np.random.default_rng(rng)
    bounds = {**DEFAULT_PRIOR_BOUNDS, **(prior_bounds or {})}
    lower = np.array([bounds[p][0] for p in PARAM_NAMES])
    upper = np.array([bounds[p][1] for p in PARAM_NAMES])

    likelihood = MobilityLikelihood(distance_km, population, town_region, work_location, home_location)

    def log_posterior(params):
        if np.any(params <= lower) or np.any(params >= upper):
            return -np.inf
        return likelihood.log_likelihood(params)

    if initial is None:
        init_lower = lower.copy()
        init_upper = upper.copy()
        init_lower[-1], init_upper[-1] = -5, 0
        initial = rng.uniform(init_lower, init_upper, size=(n_walkers, len(PARAM_NAMES)))

    sampler = emcee.EnsembleSampler(n_walkers, len(PARAM_NAMES), log_posterior)
    sampler.random_state = np.random.RandomState(rng.integers(2**32 - 1)).get_state()
    sampler.run_mcmc(initial, n_steps, progress=progress)

    return MobilityFit(chain=sampler.get_chain(), log_prob=sampler.get_log_prob(), burnin=burnin)


# =============================================================================
# Diagnostic plots (require matplotlib)
# =============================================================================

def plot_traces(fit: MobilityFit):
    """Trace plot of every walker for each parameter."""
    import matplotlib.pyplot as plt

    n = fit.chain.shape[-1]
    fig, axes = plt.subplots(n, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(fit.chain[:, :, i], "k", alpha=0.3)
        ax.axvline(fit.burnin, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel(fit.param_names[i])
        ax.set_xlim(0, fit.chain.shape[0])
    axes[-1].set_xlabel("Step")
    fig.suptitle("MCMC traces (red line: end of burn-in)")
    fig.tight_layout()
    return fig


def plot_posteriors(fit: MobilityFit):
    """Marginal posterior histogram for each parameter."""
    import matplotlib.pyplot as plt

    samples = fit.samples
    fig, axes = plt.subplots(1, samples.shape[1], figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.hist(samples[:, i], bins=50, color="steelblue")
        ax.set_title(fit.param_names[i])
    fig.tight_layout()
    return fig


def plot_distance_kernel(samples: np.ndarray, max_km: float = 200, n_lines: int = 300, rng=None):
    """Posterior distance-decay kernel: random draws plus the median."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng)
    samples = np.asarray(samples)
    distances = np.arange(0, max_km, 1.0)
    kernels = np.array([distance_kernel(distances, s[0], s[2]) for s in samples])

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in rng.choice(len(samples), size=min(n_lines, len(samples)), replace=False):
        ax.plot(distances, kernels[idx], color="tab:blue", alpha=0.05)
    ax.plot(distances, np.median(kernels, axis=0), color="tab:red", linewidth=2, label="Posterior median")
    ax.set(xlabel="Distance (km)", ylabel="Relative commuting propensity", yscale="log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fit(likelihood: MobilityLikelihood, params: Sequence[float]):
    """Observed vs predicted commuting shares for one parameter set."""
    import matplotlib.pyplot as plt

    work, home = likelihood.predicted(*params[:4])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, obs, pred, title in [(axes[0], likelihood.observed_work, work, "Where residents work"),
                                 (axes[1], likelihood.observed_home, home, "Where workers live")]:
        obs, pred = obs.ravel(), pred.ravel()
        r = np.corrcoef(obs, pred)[0, 1]
        mask = (obs > 0) & (pred > 0)
        ax.scatter(np.log10(obs[mask]), np.log10(pred[mask]), s=1.5, color="black", alpha=0.3)
        ax.plot([-5, 0], [-5, 0], "r--", alpha=0.8, label="1:1")
        ax.set(xlabel="Observed share (log10)", ylabel="Predicted share (log10)", title=f"{title}\nr = {r:.3f}")
        ax.legend()
    fig.tight_layout()
    return fig
