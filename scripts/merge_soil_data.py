"""Build a unique_id-keyed soil feature file, reusing an already-fetched iSDAsoil pull.

Produces: data/soil/kenya_maize_soil_features.csv (unique_id + 7 soil columns)
Consumed by: scripts/build_features.py's merge_soil_features()

PROVENANCE -- why this doesn't call Google Earth Engine
---------------------------------------------------------
Checked directly before writing this script: every row in this project's
kenya_maize_cleaned.csv that has a non-null field GPS fix (23,159 of 23,674
rows, 97.8%) has an EXACT match (6-decimal lat/lon) against the sibling
project's site_lookup.csv. 100% match rate on the rows that have GPS at all.
So this reuses that pull via a coordinate join instead of re-running Earth
Engine (which would need fresh credentials, an async export, and a manual
Google Drive download for identical output).

Source files (copied into data/soil/source/ for self-containment):
  - site_lookup.csv               : (field_latitude, field_longitude) -> old site_id
                                     built by ../yield-prediction/scripts/merge_chirps_rainfall.py
  - kenya_maize_soil_properties_isda.csv : old site_id -> soil properties
                                     built by ../yield-prediction/scripts/merge_soil_data.py
                                     (see that script's docstring for the GEE asset IDs,
                                     back-transform formulas, and the one caveat on soil_ph:
                                     the /10 scaling is inferred from the catalog's value
                                     range, not an explicitly documented transform)

Run:
    python scripts/merge_soil_data.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SITE_LOOKUP_PATH = ROOT / "data" / "soil" / "source" / "site_lookup.csv"
SOIL_PROPERTIES_PATH = ROOT / "data" / "soil" / "source" / "kenya_maize_soil_properties_isda.csv"
CLEANED_CSV_PATH = ROOT / "data" / "cleaned" / "kenya_maize_cleaned.csv"
OUT_PATH = ROOT / "data" / "soil" / "kenya_maize_soil_features.csv"

SOIL_COLS = [
    "soil_ph", "soil_organic_carbon_gkg", "soil_nitrogen_gkg", "soil_clay_pct",
    "soil_sand_pct", "soil_cec_cmolkg", "soil_bulk_density_gcm3",
]


def main():
    for p in (SITE_LOOKUP_PATH, SOIL_PROPERTIES_PATH, CLEANED_CSV_PATH):
        if not p.exists():
            sys.exit(f"Missing required input: {p}")

    site_lookup = pd.read_csv(SITE_LOOKUP_PATH)
    soil_props = pd.read_csv(SOIL_PROPERTIES_PATH, dtype={"site_id": str})
    site_lookup["site_id"] = site_lookup["site_id"].astype(str)

    # (lat, lon) -> soil properties, via the old site_id join
    coord_soil = site_lookup.merge(soil_props, on="site_id", how="inner")
    missing_cols = [c for c in SOIL_COLS if c not in coord_soil.columns]
    if missing_cols:
        sys.exit(f"Expected soil columns missing from {SOIL_PROPERTIES_PATH.name}: {missing_cols}")

    coord_soil["_lat6"] = coord_soil["field_latitude"].round(6)
    coord_soil["_lon6"] = coord_soil["field_longitude"].round(6)
    coord_soil = coord_soil.drop_duplicates(subset=["_lat6", "_lon6"])

    cleaned = pd.read_csv(CLEANED_CSV_PATH, usecols=["unique_id", "field_latitude", "field_longitude"],
                          low_memory=False)
    n_total = len(cleaned)
    n_gps = cleaned[["field_latitude", "field_longitude"]].notna().all(axis=1).sum()

    cleaned["_lat6"] = cleaned["field_latitude"].round(6)
    cleaned["_lon6"] = cleaned["field_longitude"].round(6)

    merged = cleaned.merge(
        coord_soil[["_lat6", "_lon6"] + SOIL_COLS],
        on=["_lat6", "_lon6"], how="left",
    )
    assert len(merged) == n_total, (
        f"coordinate join changed row count ({n_total} -> {len(merged)}); "
        "site_lookup.csv must have duplicate (lat, lon) pairs mapping to different soil values"
    )

    n_matched = merged[SOIL_COLS[0]].notna().sum()
    print(f"Rows in kenya_maize_cleaned.csv: {n_total}")
    print(f"Rows with a field GPS fix:       {n_gps} ({n_gps / n_total:.1%})")
    print(f"Rows matched to soil data:       {n_matched} ({n_matched / n_total:.1%})")
    if n_gps and n_matched < n_gps:
        print(f"  ({n_gps - n_matched} GPS rows did not match -- likely coordinates present "
              f"in this project's survey extract but not in the sibling project's)")

    out = merged[["unique_id"] + SOIL_COLS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.relative_to(ROOT)} with {len(out)} rows.")
    print(out[SOIL_COLS].describe())


if __name__ == "__main__":
    main()
