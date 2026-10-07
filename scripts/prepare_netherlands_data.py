"""Build the compact Netherlands input files in surveillance_sim/data/netherlands/ from the raw sources.

You only need this to regenerate the shipped data. It needs geopandas and openpyxl:

    pip install geopandas openpyxl
    python scripts/prepare_netherlands_data.py --raw-dir /path/to/raw/data

Expected files in --raw-dir (see surveillance_sim/data/netherlands/README.md for where to get them):

    2024-CBS_bevolkingskernen_2021_v2.gpkg
    WijkBuurtkaart_2023_v2/wijkenbuurten_2023_v2.gpkg
    aandeel_woonplaats_werkplaats.csv
    waar_werken_werknemers.csv
    waar_komen_werknemers_vandaan.csv
    Ziekenhuis_locaties_rivm_2025.xlsx
    4pp.csv
    ILI_per_100k.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TOWN_POPULATION = 1000
DEFAULT_STAY_HOME_FRAC = 0.37


def build_towns(raw: Path) -> pd.DataFrame:
    kernen = gpd.read_file(raw / "2024-CBS_bevolkingskernen_2021_v2.gpkg").to_crs(epsg=28992)
    kernen = kernen[kernen["aantal_inwoners"] > MIN_TOWN_POPULATION].copy()
    kernen["centroid"] = kernen.geometry.centroid

    # Municipality of each town via its centroid
    gemeentes = gpd.read_file(raw / "WijkBuurtkaart_2023_v2" / "wijkenbuurten_2023_v2.gpkg", layer="buurten")
    gemeentes = gemeentes[gemeentes["water"] == "NEE"]
    gemeentes = gemeentes.dissolve(by="gemeentenaam", aggfunc={"aantal_inwoners": "sum"})
    gemeentes["GM_NAAM"] = gemeentes.index
    gemeentes = gemeentes.to_crs(epsg=28992)

    centroids = gpd.GeoDataFrame(kernen.drop(columns="geometry"), geometry=kernen["centroid"], crs=28992)
    joined = gpd.sjoin(centroids, gemeentes[["geometry", "GM_NAAM"]], how="left", predicate="within")
    kernen["gemeente"] = joined["GM_NAAM"]

    # Share of residents working in their own municipality
    stay_home = pd.read_csv(raw / "aandeel_woonplaats_werkplaats.csv", decimal=",", index_col=0) / 100
    kernen["stay_home_frac"] = kernen["gemeente"].map(stay_home.iloc[:, 0]).fillna(DEFAULT_STAY_HOME_FRAC)

    return pd.DataFrame({
        "name": kernen["naam_2021"].to_numpy(),
        "municipality": kernen["gemeente"].to_numpy(),
        "province": kernen["provincie"].to_numpy(),
        "x": kernen["centroid"].x.round(1).to_numpy(),
        "y": kernen["centroid"].y.round(1).to_numpy(),
        "population": kernen["aantal_inwoners"].astype(int).to_numpy(),
        "stay_home_frac": kernen["stay_home_frac"].round(4).to_numpy(),
    })


def build_hospitals(raw: Path) -> pd.DataFrame:
    hospitals = pd.read_excel(raw / "Ziekenhuis_locaties_rivm_2025.xlsx", sheet_name=1, skiprows=1)
    hospitals = hospitals[hospitals["Type"] == "Algemeen ziekenhuis"].copy()
    hospitals["pc4"] = hospitals["Postcode"].astype(str).str[:4].astype(int)

    postcode_to_coord = pd.read_csv(raw / "4pp.csv").set_index("postcode")
    lat = hospitals["pc4"].map(postcode_to_coord["latitude"])
    lon = hospitals["pc4"].map(postcode_to_coord["longitude"])
    points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(epsg=28992)

    return pd.DataFrame({
        "name": hospitals["Naam"].to_numpy(),
        "city": hospitals["Plaats"].to_numpy(),
        "x": np.round(points.x.to_numpy(), 1),
        "y": np.round(points.y.to_numpy(), 1),
    })


def clean_commuting_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, decimal=",", index_col=0)
    table.columns = table.columns.str.replace(" (%)", "", regex=False)
    table.index.name = "municipality"
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "surveillance_sim" / "data" / "netherlands")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    towns = build_towns(args.raw_dir)
    towns.to_csv(args.out_dir / "towns.csv", index=False)
    print(f"towns.csv: {len(towns)} towns, population {towns['population'].sum():,}")

    hospitals = build_hospitals(args.raw_dir)
    hospitals.to_csv(args.out_dir / "hospitals.csv", index=False)
    print(f"hospitals.csv: {len(hospitals)} general hospitals")

    for src, dst in [("waar_werken_werknemers.csv", "commuting_work_location.csv"),
                     ("waar_komen_werknemers_vandaan.csv", "commuting_home_location.csv")]:
        table = clean_commuting_table(args.raw_dir / src)
        table.to_csv(args.out_dir / dst)
        print(f"{dst}: {table.shape[0]} municipalities x {table.shape[1]} cities")

    ili = pd.read_csv(args.raw_dir / "ILI_per_100k.csv", sep=";", index_col=0)
    ili.index.name = "week"
    ili.sort_index().to_csv(args.out_dir / "ili_per_100k.csv")
    print(f"ili_per_100k.csv: {ili.shape[0]} weeks x {ili.shape[1]} seasons")


if __name__ == "__main__":
    main()
