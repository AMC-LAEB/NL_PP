"""Hospital and GP surveillance for the Netherlands, using the shipped Dutch data.

    python examples/surveillance_netherlands.py --sims 20 --r0 1.5
"""

import argparse

import numpy as np

from surveillance_sim import (
    mobility_matrix,
    simulate_epidemic,
    simulate_gp_surveillance,
    simulate_hospital_surveillance,
)
from surveillance_sim.netherlands import load_netherlands

parser = argparse.ArgumentParser()
parser.add_argument("--sims", type=int, default=10, help="number of simulated outbreaks")
parser.add_argument("--r0", type=float, default=1.5)
parser.add_argument("--max-infections", type=float, default=1e5, help="outbreak size at which simulations stop")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
nl = load_netherlands()
population = nl.population

# =============================================================================
# 1. OUTBREAKS, seeded in towns proportional to population
# =============================================================================
params = nl.mobility_params()
print("Mobility parameters (posterior median): "
      + ", ".join(f"{k}={v:.3f}" for k, v in params.items() if k != "log_sigma"))
flux = mobility_matrix(nl.distance_km(), population, nl.towns["stay_home_frac"].to_numpy(),
                       params["dist_exp"], params["pop_exp"], params["offset"], params["epsilon"])
mobility = flux.T * 0.5

beta = 0.25
seeds = rng.choice(len(population), size=args.sims, p=population / population.sum())
print(f"Simulating {args.sims} outbreaks (R0={args.r0}) in {len(population)} towns...")
epidemics = [
    simulate_epidemic(s, alpha=beta * args.r0, beta=beta, mobility=mobility, n_days=200,
                      population=population, extinction_threshold=args.max_infections,
                      poisson=True, rng=rng)
    for s in seeds
]


def summarise(results):
    days = [r.detection_day for r in results if r is not None]
    infs = [r.cumulative_infections for r in results if r is not None]
    if not days:
        return "never detected"
    return (f"median day {np.median(days):5.1f} | median infections {np.median(infs):9.0f} | "
            f"detected {len(days)}/{len(results)}")


# =============================================================================
# 2. HOSPITAL SURVEILLANCE
# =============================================================================
close_hospitals = nl.close_hospitals(radius_m=10_000)
n_hospitals = len(nl.hospitals)

print(f"\nHospital surveillance ({n_hospitals} general hospitals)")
for prop in [0.1, 0.25, 0.5, 1.0]:
    res = [simulate_hospital_surveillance(out, close_hospitals, n_hospitals, prop,
                                          hosp_p=0.0125, p_symp=0.6, rng=rng)
           for out in epidemics]
    print(f"  {prop:4.0%} of hospitals: {summarise(res)}")

# =============================================================================
# 3. GP SURVEILLANCE, outbreak starting in winter vs summer
# =============================================================================
background_ili = nl.background_ili()

for label, calendar_week in [("winter, week 1", 1), ("summer, week 27", 27)]:
    print(f"\nGP surveillance, outbreak starts in {label} (2 swabs per GP per week)")
    for n_gps in [20, 50, 100, 200]:
        res = [simulate_gp_surveillance(out, population, background_ili, n_gps,
                                        p_symp=0.6, healthcare_seeking_p=0.2,
                                        init_week=calendar_week - 1, n_weekly_samples=2, rng=rng)
               for out in epidemics]
        print(f"  {n_gps:4d} GPs: {summarise(res)}")
