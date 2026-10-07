# Netherlands input data

Compact, pre-processed inputs for the Dutch model, built by
[`scripts/prepare_netherlands_data.py`](../../../scripts/prepare_netherlands_data.py)
from the raw sources below. Load them with
`surveillance_sim.netherlands.load_netherlands()`.

| file | contents |
|---|---|
| `towns.csv` | 1,109 population centres with more than 1,000 inhabitants: name, municipality, province, centroid `x`/`y` (EPSG:28992, metres), population, share of residents working in their own municipality (`stay_home_frac`; 0.37 where unknown) |
| `hospitals.csv` | 98 general hospitals (*algemeen ziekenhuis*), located at their 4-digit postcode centroid (EPSG:28992, metres) |
| `commuting_work_location.csv` | For residents of 43 cities (columns): % working in each municipality (rows) |
| `commuting_home_location.csv` | For people working in 43 cities (columns): % living in each municipality (rows) |
| `ili_per_100k.csv` | GP consultations for influenza-like illness per 100,000 people per calendar week (rows), by season (columns) |
| `mobility_posterior.csv` | MCMC posterior samples of the mobility parameters. Created by running `examples/fit_mobility_netherlands.py` |

## Sources

- **Population centres**: CBS, *Bevolkingskernen in Nederland 2021*
  (`2024-CBS_bevolkingskernen_2021_v2.gpkg`).
- **Municipality of each town**: CBS, *Wijk- en buurtkaart 2023 v2*
  (`wijkenbuurten_2023_v2.gpkg`, ~400 MB, not included). It's only used to look up
  which municipality each town centroid falls in.
- **Commuting**: CBS, *Woon- en werklocaties van werknemers, 2016*,
  <https://www.cbs.nl/nl-nl/achtergrond/2018/11/woon-en-werklocaties-van-werknemers-2016>.
  Municipalities are as of 2016, so a few later mergers don't match the 2023 names.
- **Hospital locations**: RIVM / VZinfo, *Ziekenhuislocaties 2025*,
  <https://www.vzinfo.nl/documenten/ziekenhuislocaties-2025>, geocoded with a
  4-digit postcode coordinate table (`4pp.csv`, not included).
- **ILI consultations**: Nivel Zorgregistraties Eerste Lijn, sentinel GP
  surveillance of influenza-like illness.

CBS data is published under CC BY 4.0.
