"""End-to-end example on synthetic data.

Run with:  python examples/run_example.py

Replace the synthetic inputs in section 1 with your own data to run the models
for your own country or region.
"""

import numpy as np

from surveillance_sim import (
    distance_matrix,
    hospital_catchments,
    mobility_matrix,
    simulate_epidemic,
    simulate_gp_surveillance,
    simulate_hospital_surveillance,
)

rng = np.random.default_rng(42)

# =============================================================================
# 1. INPUT DATA (swap these for your own)
# =============================================================================
n_locations = 150
location_coords = rng.uniform(0, 200_000, size=(n_locations, 2))       # metres
population = rng.lognormal(mean=9, sigma=1.2, size=n_locations).astype(int) + 1000
stay_home_frac = 0.37                                                   # per location, or one value

n_hospitals = 25
hospital_coords = rng.uniform(0, 200_000, size=(n_hospitals, 2))

# Per-person weekly probability of a background ILI GP visit, 52 weeks,
# here a seasonal curve peaking mid-winter (week 0 = start of the year).
weeks = np.arange(52)
background_ili = 20e-5 * (1 + 0.8 * np.cos(2 * np.pi * weeks / 52))

# =============================================================================
# 2. SIMULATE EPIDEMICS
# =============================================================================
flux = mobility_matrix(
    distance_matrix(location_coords) / 1000,  # km
    population,
    stay_home_frac,
    dist_exp=2.0,
    pop_exp=0.5,
)
mobility = flux.T * 0.5

r0, beta = 1.5, 0.25
n_sims = 20
seeds = rng.choice(n_locations, size=n_sims, p=population / population.sum())

print(f"Simulating {n_sims} epidemics (R0={r0})...")
epidemics = [
    simulate_epidemic(s, alpha=beta * r0, beta=beta, mobility=mobility, n_days=200,
                      population=population, extinction_threshold=1e4, poisson=True, rng=rng)
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
# 3. HOSPITAL SURVEILLANCE
# =============================================================================
catchments = hospital_catchments(location_coords, hospital_coords, radius=10_000)

print("\nHospital surveillance (hosp_p=0.0125, p_symp=0.6)")
for prop in [0.1, 0.25, 0.5, 1.0]:
    res = [simulate_hospital_surveillance(out, catchments, n_hospitals, prop,
                                          hosp_p=0.0125, p_symp=0.6, rng=rng)
           for out in epidemics]
    print(f"  {prop:4.0%} of hospitals: {summarise(res)}")

# =============================================================================
# 4. GP SURVEILLANCE
# =============================================================================
print("\nGP surveillance (2 swabs per GP per week, winter start)")
for n_gps in [10, 50, 100, 200]:
    res = [simulate_gp_surveillance(out, population, background_ili, n_gps,
                                    p_symp=0.6, healthcare_seeking_p=0.2,
                                    init_week=0, n_weekly_samples=2, rng=rng)
           for out in epidemics]
    print(f"  {n_gps:4d} GPs: {summarise(res)}")
