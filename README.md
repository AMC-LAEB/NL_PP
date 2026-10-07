# surveillance-sim

Simulate how quickly a surveillance system detects a newly emerging respiratory
pathogen. You give it an outbreak (new infections per day per location) and it
tells you, for a given surveillance design, **on which day the first case is
detected** and **how far the outbreak has spread by then**.

Two surveillance systems are modelled:

- **Hospital surveillance**: a fraction of hospitals tests hospitalised patients.
- **GP (sentinel practice) surveillance**: GP practices swab a fixed number of
  patients with influenza-like illness (ILI) each week. Patients with the new
  pathogen are mixed in with background seasonal ILI patients.

Outbreaks come from a spatial SIR model in which people commute between towns.
The commuting model is fitted to observed commuting data with MCMC.

The package ships ready-to-use data for the **Netherlands** (towns, hospitals,
commuting and ILI seasonality). All model inputs are plain arrays, so you can also
run it for any other country or region.

## Installation

```bash
git clone https://github.com/AMC-LAEB/NL_PP.git
cd NL_PP
pip install -e ".[fit]"
```

The core needs only NumPy and pandas. The `fit` extra adds `emcee` and `matplotlib`
for fitting the mobility model and plotting.

## Quick start

```bash
# Synthetic country, to see the workflow end to end on made-up data
python examples/run_example.py

# Netherlands: first fit the mobility model to Dutch commuting data (~20 min).
# Writes surveillance_sim/data/netherlands/mobility_posterior.csv and plots in figures/
python examples/fit_mobility_netherlands.py --walkers 32 --steps 600 --burnin 300

# ...then simulate outbreaks and surveillance with the fitted parameters
python examples/surveillance_netherlands.py --sims 20 --r0 1.5
```

## Workflow

```
 towns, distances,          observed regional         fit_mobility()      mobility parameters
 population           ───►  commuting shares     ───►  (MCMC, emcee)  ───►  (posterior)
                                                                                 │
                                                                                 ▼
 hospitals, GP data,  ◄───  outbreaks: new infections  ◄───  simulate_epidemic(mobility_matrix(...))
 ILI seasonality             per day per town
        │
        ▼
 simulate_hospital_surveillance() / simulate_gp_surveillance()  ───►  detection day, outbreak size
```

## Netherlands data

```python
from surveillance_sim.netherlands import load_netherlands

nl = load_netherlands()
nl.towns                # 1,109 towns: name, municipality, province, x, y, population, stay_home_frac
nl.hospitals            # 98 general hospitals: name, city, x, y
nl.distance_km()        # town-to-town distances
nl.close_hospitals()    # hospitals within 10 km of each town
nl.background_ili()     # weekly ILI consultation rate per person, index 0 = calendar week 1
nl.mobility_params()    # posterior median, after running examples/fit_mobility_netherlands.py
```

Sources and processing are described in
[`surveillance_sim/data/netherlands/README.md`](surveillance_sim/data/netherlands/README.md).

## Using your own data

### 1. Fit the mobility model

You need town locations and populations, which region each town belongs to, and
observed commuting shares between regions in two tables:

- `work_location`: rows = region of work, columns = region of residence. Each
  column gives where that region's residents work.
- `home_location`: rows = region of residence, columns = region of work. Each
  column gives where that region's workers live.

The columns may cover just a subset of regions (e.g. the larger cities). Values can be
percentages or fractions.

```python
from surveillance_sim import distance_matrix
from surveillance_sim.fitting import fit_mobility, plot_traces

fit = fit_mobility(
    distance_km=distance_matrix(town_coords) / 1000,  # projected coordinates in metres
    population=town_population,
    town_region=town_region_labels,                   # must match the table labels
    work_location=work_table,
    home_location=home_table,
    n_walkers=32, n_steps=600, burnin=300,
)
fit.median()            # {'dist_exp': ..., 'pop_exp': ..., 'offset': ..., 'epsilon': ..., 'log_sigma': ...}
fit.to_dataframe()      # posterior samples
plot_traces(fit)        # check convergence; also plot_posteriors, plot_distance_kernel, plot_fit
```

Parameters: `dist_exp` and `offset` shape how commuting decays with distance,
`pop_exp` scales the effect of origin population, and `epsilon` controls competition
between destinations. `log_sigma` is the observation noise. Priors are uniform.
Change them with `prior_bounds={"dist_exp": (0, 6), ...}`.

### 2. Simulate outbreaks

```python
from surveillance_sim import mobility_matrix, simulate_epidemic

p = fit.median()
flux = mobility_matrix(distance_km, town_population, stay_home_frac,
                       p["dist_exp"], p["pop_exp"], p["offset"], p["epsilon"])

out = simulate_epidemic(
    seed_index=0,               # town where the outbreak starts
    alpha=0.25 * 1.5,           # transmission rate (R0 = alpha / beta)
    beta=0.25,                  # recovery rate
    mobility=flux.T * 0.5,      # half of each day spent at the commute destination
    n_days=200,
    population=town_population,
    extinction_threshold=1e5,   # see note below
    poisson=True,               # stochastic
)
```

`stay_home_frac` is the share of residents working in their own town or
municipality, given as a scalar or one value per town.

Outbreaks that die out before reaching `extinction_threshold` infections are
rerun. The simulation also **stops** once it reaches that size, and every later day is
zero. So the threshold is the largest outbreak size you analyse. A system that
hasn't detected the outbreak by then counts as not detected.

You can skip this step and use trajectories from any other model.
`out` just needs to be an array of shape `(n_days, n_locations)` with **new
infections** per day.

### 3. Hospital surveillance

```python
from surveillance_sim import hospital_catchments, simulate_hospital_surveillance

catchments = hospital_catchments(town_coords, hospital_coords, radius=10_000)

result = simulate_hospital_surveillance(
    out,
    close_hospitals=catchments,  # hospitals each town's residents may attend
    n_hospitals=len(hospital_coords),
    prop_hospitals=0.25,         # 25% of hospitals participate
    hosp_p=0.0125,               # P(hospitalised | symptomatic)
    p_symp=0.6,                  # P(symptomatic | infected)
)
```

Each infection is detected with probability `hosp_p * p_symp`, multiplied by the
fraction of its town's nearby hospitals that participate. If you already know which
hospitals serve which area, pass your own `close_hospitals` lists.

### 4. GP surveillance

```python
from surveillance_sim import simulate_gp_surveillance

result = simulate_gp_surveillance(
    out,
    population=town_population,
    background_weekly_rate=ili_per_person,  # weekly ILI consultation rate per person, length 52
    n_gps=100,                              # number of sentinel practices
    p_symp=0.6,
    healthcare_seeking_p=0.2,               # P(visits GP | symptomatic)
    init_week=0,                            # index into background_weekly_rate at day 0
    n_weekly_samples=2,                     # swabs per GP per week
    gp_practice_size=3626,                  # patients per practice
)
```

The background ILI rate sets the season. In winter, sentinel GPs swab many
seasonal-flu patients, so a new pathogen is harder to find. If your data is "ILI per
100,000 per week", divide by 100,000.

### Output

Both surveillance functions return `None` if nothing is detected. Otherwise they return
a `Detection`:

| field | meaning |
|---|---|
| `cumulative_infections` | total infections up to and including the detection day |
| `detection_day` | first day with a detected case (0-based) |
| `n_locations_infected` | towns with at least one infection by then |

`Detection` is a named tuple, so `result[1]` also gives the detection day.

Each call is one random draw (which hospitals take part, where the GPs are, who gets
tested). Run each design many times, over many outbreaks, and compare the
distributions. Pass `rng=<seed>` or a `numpy.random.Generator` for reproducibility.

## Project layout

```
surveillance_sim/
    surveillance.py      simulate_hospital_surveillance, simulate_gp_surveillance
    epidemic.py          distance_kernel, mobility_matrix, simulate_epidemic
    fitting.py           fit_mobility (MCMC) and diagnostic plots
    geography.py         distance_matrix, hospital_catchments
    netherlands.py       load_netherlands
    data/netherlands/    shipped Dutch input data
examples/
    surveillance_netherlands.py
    fit_mobility_netherlands.py
    run_example.py       synthetic data
scripts/
    prepare_netherlands_data.py   rebuilds data/netherlands from the raw sources
tests/
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
