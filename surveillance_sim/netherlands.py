"""Ready-to-use input data for the Netherlands.

    from surveillance_sim.netherlands import load_netherlands
    nl = load_netherlands()

See ``data/netherlands/README.md`` for sources and how the files were built.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .geography import distance_matrix, hospital_catchments

DATA_DIR = Path(__file__).parent / "data" / "netherlands"

# ILI seasons used by default: pre-COVID and post-COVID, skipping 2019/20 - 2021/22
DEFAULT_ILI_SEASONS = ["2015/2016", "2016/2017", "2017/2018", "2018/2019", "2022/2023", "2023/2024"]


@dataclass
class NetherlandsData:
    towns: pd.DataFrame
    """One row per population centre (>1000 inhabitants): name, municipality, province,
    x, y (EPSG:28992 metres), population, stay_home_frac."""
    hospitals: pd.DataFrame
    """General hospitals: name, city, x, y (EPSG:28992 metres)."""
    commuting_work_location: pd.DataFrame
    """CBS: for residents of each city (columns), % working in each municipality (rows)."""
    commuting_home_location: pd.DataFrame
    """CBS: for workers in each city (columns), % living in each municipality (rows)."""
    ili_per_100k: pd.DataFrame
    """Nivel GP consultations for influenza-like illness per 100,000 per calendar week, by season."""
    mobility_posterior: Optional[pd.DataFrame] = None
    """Posterior samples of the fitted mobility parameters, if available."""

    @property
    def population(self) -> np.ndarray:
        return self.towns["population"].to_numpy()

    @property
    def coords(self) -> np.ndarray:
        return self.towns[["x", "y"]].to_numpy()

    @property
    def hospital_coords(self) -> np.ndarray:
        return self.hospitals[["x", "y"]].to_numpy()

    def distance_km(self) -> np.ndarray:
        """Town-to-town distance matrix in km."""
        return distance_matrix(self.coords) / 1000

    def close_hospitals(self, radius_m: float = 10_000):
        """Hospitals within ``radius_m`` of each town (nearest one if none)."""
        return hospital_catchments(self.coords, self.hospital_coords, radius_m)

    def background_ili(self, seasons: Sequence[str] = DEFAULT_ILI_SEASONS) -> np.ndarray:
        """Per-person weekly ILI consultation rate, averaged over ``seasons``.

        Index 0 is calendar week 1, so pass ``init_week = calendar_week - 1`` to
        :func:`surveillance_sim.simulate_gp_surveillance`.
        """
        return self.ili_per_100k.loc[:, list(seasons)].mean(axis=1).sort_index().to_numpy() / 1e5

    def mobility_params(self) -> dict:
        """Posterior median of the fitted mobility parameters."""
        if self.mobility_posterior is None:
            raise FileNotFoundError("No mobility_posterior.csv found; run examples/fit_mobility_netherlands.py")
        return self.mobility_posterior.median().to_dict()


def load_netherlands(data_dir: Optional[Path] = None) -> NetherlandsData:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    posterior_path = data_dir / "mobility_posterior.csv"
    return NetherlandsData(
        towns=pd.read_csv(data_dir / "towns.csv"),
        hospitals=pd.read_csv(data_dir / "hospitals.csv"),
        commuting_work_location=pd.read_csv(data_dir / "commuting_work_location.csv", index_col=0),
        commuting_home_location=pd.read_csv(data_dir / "commuting_home_location.csv", index_col=0),
        ili_per_100k=pd.read_csv(data_dir / "ili_per_100k.csv", index_col=0),
        mobility_posterior=pd.read_csv(posterior_path) if posterior_path.exists() else None,
    )
