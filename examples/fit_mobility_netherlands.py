"""Fit the mobility model to Dutch (CBS) commuting data with MCMC.

    pip install -e ".[fit]"
    python examples/fit_mobility_netherlands.py --steps 600 --walkers 32 --burnin 300

Saves the posterior samples to surveillance_sim/data/netherlands/mobility_posterior.csv
(used by ``load_netherlands().mobility_params()``) and diagnostic plots to figures/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from surveillance_sim.fitting import (
    MobilityLikelihood,
    fit_mobility,
    plot_distance_kernel,
    plot_fit,
    plot_posteriors,
    plot_traces,
)
from surveillance_sim.netherlands import DATA_DIR, load_netherlands

parser = argparse.ArgumentParser()
parser.add_argument("--walkers", type=int, default=20)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--burnin", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=Path, default=DATA_DIR / "mobility_posterior.csv")
parser.add_argument("--figures", type=Path, default=Path("figures"))
args = parser.parse_args()

nl = load_netherlands()
inputs = dict(
    distance_km=nl.distance_km(),
    population=nl.population,
    town_region=nl.towns["municipality"],
    work_location=nl.commuting_work_location,
    home_location=nl.commuting_home_location,
)

fit = fit_mobility(**inputs, n_walkers=args.walkers, n_steps=args.steps, burnin=args.burnin, rng=args.seed)

print("Posterior median:")
for name, value in fit.median().items():
    print(f"  {name:10s} {value: .4f}")

fit.to_dataframe().to_csv(args.out, index=False)
print(f"Saved {len(fit.samples)} posterior samples to {args.out}")

args.figures.mkdir(exist_ok=True)
plot_traces(fit).savefig(args.figures / "mcmc_traces.png", dpi=150)
plot_posteriors(fit).savefig(args.figures / "mcmc_posteriors.png", dpi=150)
plot_distance_kernel(fit.samples).savefig(args.figures / "distance_kernel.png", dpi=150)
plot_fit(MobilityLikelihood(**inputs), list(fit.median().values())).savefig(args.figures / "mobility_fit.png", dpi=150)
print(f"Saved plots to {args.figures}/")
