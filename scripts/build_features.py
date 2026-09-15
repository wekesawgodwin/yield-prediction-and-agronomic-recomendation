"""
Kenya Maize Feature Engineering Pipeline
=========================================

Production feature-construction pipeline for the One Acre Fund MEL Agronomic
Survey (2021), scoped to Kenya maize. Consumes the output of
scripts/clean_kenya_maize.py and produces a modeling-ready feature matrix plus
an explicit feature *contract* (the manifest) for the downstream modeling
package.

This script is the executable source of truth behind:
  - notebooks/02_kenya_maize_eda_and_feature_engineering.ipynb
  - documentation/kenya_maize_feature_engineering_documentation.md

Run:
    python scripts/build_features.py

Reads:  data/cleaned/kenya_maize_cleaned.csv
Writes: data/features/kenya_maize_features_full.csv     (2016-2020)
        data/features/kenya_maize_features_core.csv     (2017-2020, see Gate C)
        data/features/kenya_maize_feature_manifest.csv  (the contract)
        data/features/kenya_maize_feature_stats.json    (audit numbers used in docs)

Implements CRISP-DM report (documentation/project1_crispdm_report.docx) Section
3.3 "Feature Construction" and Section 3.5 "Train / Validation / Test Split
Strategy".

DESIGN PRINCIPLE
----------------
The cleaning pipeline's rule was "never silently discard or overwrite a source
value". This pipeline's rule is narrower and different: **never emit a feature
that cannot mean the same thing in the training years and in the holdout
year.** The Kenya survey instrument rotates modules year to year, so a column
that is 100% present in 2018 and 0% present in 2016 is not a feature with
missing values - it is a year label wearing a feature's name. Every stage below
exists to catch one specific way that failure can happen.

Deliberately NOT done here (all recorded in the documentation):
  - No imputation. Missing stays missing; GBMs handle it natively and imputing
    is a modeling-time decision.
  - No compost -> nitrogen conversion (the required assumptions span an order
    of magnitude; see build_agronomic_features).
  - No site-level target encoding (see assign_cv_folds).
  - No unification of plants_sqm_spacing with design spacing (see
    ROLE_OVERRIDES / the documentation).

DEPENDENCIES: numpy + pandas only, deliberately. The `python3` interpreter on
the development machine has no matplotlib/sklearn, so keeping this script to
the same imports as clean_kenya_maize.py means it runs under either
interpreter. Plotting lives in the notebook, which runs on the Jupyter kernel.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = ROOT / "data" / "cleaned" / "kenya_maize_cleaned.csv"
WEATHER_FEATURES_PATH = ROOT / "data" / "weather" / "kenya_maize_weather_features.csv"
SOIL_FEATURES_PATH = ROOT / "data" / "soil" / "kenya_maize_soil_features.csv"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_FULL_PATH = FEATURES_DIR / "kenya_maize_features_full.csv"
FEATURES_CORE_PATH = FEATURES_DIR / "kenya_maize_features_core.csv"
MANIFEST_PATH = FEATURES_DIR / "kenya_maize_feature_manifest.csv"
STATS_PATH = FEATURES_DIR / "kenya_maize_feature_stats.json"

TARGET = "yield_kg_ph"
HOLDOUT_YEAR = 2020            # per CRISP-DM report Section 3.5: most recent year
CORE_DROPS_YEAR = 2016         # the reduced-instrument year traded away by Gate C
ALL_YEARS = (2016, 2017, 2018, 2019, 2020)
TRAIN_YEARS_FULL = (2016, 2017, 2018, 2019)
TRAIN_YEARS_CORE = (2017, 2018, 2019)

stats = {}  # collects every number used in the documentation narrative


def log(msg):
    print(f"[build_features] {msg}")


# ---------------------------------------------------------------------------
# 1. Load cleaned data & restore dtypes
# ---------------------------------------------------------------------------
# The cleaning pipeline cast 22 binary columns to pandas' nullable `boolean`
# dtype, but CSV has no dtype system: booleans round-trip as the *strings*
# "True"/"False", and pandas' CSV reader then guesses inconsistently. A column
# with no missing values (e.g. `hybrid`) is guessed as bool; the same kind of
# column *with* missing values (e.g. `drought`) is left as object holding a mix
# of Python bools and NaN. Left alone, that inconsistency silently breaks any
# `.sum()` / `.mean()` / `>` operation applied uniformly across these columns.
BOOLEAN_COLUMNS = [
    # survey binaries cast by clean_kenya_maize.finalize_dtypes
    "hybrid", "pest_disease", "drought", "flood", "intercrop", "compost",
    "FAW", "stemborer", "msv", "mlnd", "cutworms", "aphids", "blight", "striga",
    "pesticide", "electricity", "radio_binary", "bikes_binary", "cows_binary",
    "goats_binary", "chicken_binary", "comp_source_manure",
    # flags added by the cleaning pipeline
    "field_gps_flagged_invalid", "plant_date_flagged_implausible",
    "growing_season_days_flagged", "hybridseed_2017_methodology_flag",
    "seed_type_is_mixed", "weed_flagged_invalid",
    "yield_kg_ph_flagged_outlier", "yield_kg_ph_crop_failure",
    "dap_kg_ph_flagged_outlier", "urea_kg_ph_flagged_outlier",
    "npk_kg_ph_flagged_outlier", "can_kg_ph_flagged_outlier",
    "lime_kg_ph_flagged_outlier", "comp_wb_pa_flagged_outlier",
    "plot_acres_flagged_outlier",
    "fertility_is_missing", "slope_angle_num_is_missing", "slope_is_missing",
    "distance_meter_is_missing", "pest_disease_is_missing",
    "comp_method_is_missing", "comp_quality_is_missing",
]

_TRUE_TOKENS = {True, "True", "TRUE", "true", 1, 1.0, "1"}
_FALSE_TOKENS = {False, "False", "FALSE", "false", 0, 0.0, "0"}


def _to_boolean(series):
    def conv(v):
        if v in _TRUE_TOKENS:
            return True
        if v in _FALSE_TOKENS:
            return False
        return pd.NA
    return series.map(conv).astype("boolean")


def load_cleaned():
    log(f"Loading cleaned dataset: {CLEANED_CSV_PATH.name}")
    df = pd.read_csv(CLEANED_CSV_PATH, low_memory=False)
    stats["input_shape"] = list(df.shape)

    restored = []
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = _to_boolean(df[col])
            restored.append(col)
    stats["boolean_columns_restored"] = restored

    for col in ("plant_date", "harvest_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    log(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols; "
        f"restored {len(restored)} boolean columns lost to CSV round-tripping.")
    return df


# ---------------------------------------------------------------------------
# 1b. Merge external weather features (CHIRPS rainfall + ERA5-Land temperature)
# ---------------------------------------------------------------------------
# Built by the three-stage pipeline in scripts/weather_locations.py ->
# fetch_gee_weather.py -> build_weather_features.py, which resolves each row to
# a CHIRPS grid cell and pulls monthly rainfall and temperature for that cell's
# season from Google Earth Engine.
#
# Merged HERE, before drop_unusable_columns, for a mundane but load-bearing
# reason: the join key is `unique_id`, which the very next stage drops as a
# memorization key. Moving this call after that drop leaves nothing to join on.
#
# OPTIONAL BY DESIGN. The weather stages need Google credentials and a network
# round trip; this one does not, and a contributor without an Earth Engine
# account must still be able to reproduce the rest of the project. If the file
# is absent the pipeline logs it and continues, and every output is exactly the
# pre-weather output.
#
# THE SCREEN CANNOT SEE THE RISK IN THESE COLUMNS. Gates A-D all test coverage
# and holdout variance, and every weather column is 100% present in all five
# years with real variance in 2020 -- so all of them sail through. The failure
# mode weather brings is different in kind: regional rainfall moves together
# across a survey packed into a few hundred kilometres, so a raw monthly total
# is substantially a season label. That is measured rather than assumed, by the
# `year_variance_share` column added to the manifest, and it is why the
# anomaly-against-normal variants are emitted alongside the levels.
WEATHER_METADATA_COLUMNS = [
    "season_year", "location_key", "weather_lat", "weather_lon",
    "weather_location_source",
]
# weather_lat/weather_lon are deliberately NOT merged. They are the same 5-decimal
# household fingerprint that ROLE_OVERRIDES already quarantines field_latitude and
# field_longitude for, and re-admitting them under a new name would walk straight
# back into the out-of-time collapse documented there.

# Months known before the modal March planting. Everything else describes
# weather the farmer had not yet experienced when the decisions were made.
WEATHER_EX_ANTE_MONTH_SUFFIXES = ("_jan", "_feb")

WEATHER_FEATURE_COLUMNS = []  # populated by merge_weather_features


def _weather_tier(col):
    # A cell's 1991-2020 normal is fixed geography, knowable years ahead; the
    # pre-season block is last year's short rains, knowable before planting.
    if col.endswith("_normal") or "preseason" in col:
        return "ex_ante"
    if col.endswith(WEATHER_EX_ANTE_MONTH_SUFFIXES):
        return "ex_ante"
    # Conservative on the rest. March rain is already in the ground for a
    # farmer planting in April, but a tier is a property of the column, not of
    # the row, and the column has to be safe for the earliest planter in it.
    return "mid_season"


def _weather_role(col):
    # Anomalies are realized deviations from what the place normally gets --
    # the same kind of thing as `drought` and `flood`, which is what `shock`
    # already means in this taxonomy. Levels and normals describe the
    # environment a farm sits in, which is `condition`.
    return "shock" if "_anom_" in col else "condition"


def merge_weather_features(df):
    if not WEATHER_FEATURES_PATH.exists():
        stats["weather"] = {
            "merged": False,
            "reason": f"{WEATHER_FEATURES_PATH.relative_to(ROOT)} not found",
            "how_to_build": [
                "python scripts/weather_locations.py",
                "python scripts/fetch_gee_weather.py --project YOUR_GCP_PROJECT_ID",
                "python scripts/build_weather_features.py",
            ],
        }
        log("No weather feature file found -- continuing without rainfall and "
            "temperature. To build them: python scripts/weather_locations.py, "
            "then fetch_gee_weather.py, then build_weather_features.py.")
        return df

    weather = pd.read_csv(WEATHER_FEATURES_PATH, low_memory=False)
    feature_cols = [c for c in weather.columns
                    if c not in WEATHER_METADATA_COLUMNS + ["unique_id"]]

    # Keep the provenance of the coordinate as a quality flag, but as a boolean
    # rather than the three-level string: 97.8% of rows are a real field GPS
    # fix, and the distinction that matters downstream is "this row's weather
    # came from the farmer's own plot" versus "it came from a fallback that may
    # be 20 km away".
    weather["weather_location_is_field"] = (
        weather["weather_location_source"] == "field"
    ).astype("boolean")
    feature_cols.append("weather_location_is_field")

    before = len(df)
    df = df.merge(
        weather[["unique_id"] + feature_cols],
        on="unique_id", how="left", validate="one_to_one",
    )
    assert len(df) == before, (
        f"the weather merge changed the row count ({before} -> {len(df)}); "
        "kenya_maize_weather_features.csv is not one row per survey row"
    )

    # Guard: a partial weather join is worse than none. It would look like an
    # ordinary sparse feature to the gates while actually encoding which rows
    # happened to resolve to a grid cell.
    coverage = float(df["rain_mm_season"].notna().mean() * 100) \
        if "rain_mm_season" in df.columns else 0.0
    assert coverage > 99.0, (
        f"season rainfall covers only {coverage:.1f}% of rows after the merge; "
        "re-run scripts/build_weather_features.py before trusting these columns."
    )

    WEATHER_FEATURE_COLUMNS.clear()
    WEATHER_FEATURE_COLUMNS.extend(feature_cols)

    tiers = {c: _weather_tier(c) for c in feature_cols}
    stats["weather"] = {
        "merged": True,
        "source_file": str(WEATHER_FEATURES_PATH.relative_to(ROOT)),
        "n_features": len(feature_cols),
        "rainfall_dataset": "UCSB-CHG/CHIRPS/DAILY (0.05 deg)",
        "temperature_dataset": "ECMWF/ERA5_LAND/DAILY_AGGR (0.1 deg)",
        "season_rainfall_coverage_pct": round(coverage, 1),
        "tier_counts": pd.Series(tiers).value_counts().to_dict(),
        "field_gps_rows_pct": round(
            float(df["weather_location_is_field"].mean() * 100), 1
        ),
    }
    log(f"Merged {len(feature_cols)} weather features "
        f"(CHIRPS rainfall + ERA5-Land temperature via Earth Engine); "
        f"season rainfall covers {coverage:.1f}% of rows, "
        f"{stats['weather']['field_gps_rows_pct']}% from the farmer's own GPS fix.")
    return df


# ---------------------------------------------------------------------------
# 1c. Merge external soil features (iSDAsoil, via scripts/merge_soil_data.py)
# ---------------------------------------------------------------------------
# Built by scripts/merge_soil_data.py, which reuses an iSDAsoil pull already
# fetched by the sibling yield-prediction project via a coordinate join (see
# that script's docstring for provenance and the soil_ph scaling caveat).
#
# Merged HERE, before drop_unusable_columns, for the same reason as weather:
# the join key is `unique_id`, which the very next stage drops as a
# memorization key.
#
# OPTIONAL BY DESIGN, same contract as weather: a contributor without the
# sibling project's soil pull must still be able to reproduce the rest of the
# pipeline, so a missing file is logged and skipped rather than failing.
SOIL_FEATURE_COLUMNS = []  # populated by merge_soil_features


def merge_soil_features(df):
    if not SOIL_FEATURES_PATH.exists():
        stats["soil"] = {
            "merged": False,
            "reason": f"{SOIL_FEATURES_PATH.relative_to(ROOT)} not found",
            "how_to_build": ["python scripts/merge_soil_data.py"],
        }
        log("No soil feature file found -- continuing without soil properties. "
            "To build it: python scripts/merge_soil_data.py.")
        return df

    soil = pd.read_csv(SOIL_FEATURES_PATH, low_memory=False)
    feature_cols = [c for c in soil.columns if c != "unique_id"]

    before = len(df)
    df = df.merge(
        soil[["unique_id"] + feature_cols],
        on="unique_id", how="left", validate="one_to_one",
    )
    assert len(df) == before, (
        f"the soil merge changed the row count ({before} -> {len(df)}); "
        "kenya_maize_soil_features.csv is not one row per survey row"
    )

    SOIL_FEATURE_COLUMNS.clear()
    SOIL_FEATURE_COLUMNS.extend(feature_cols)

    coverage = {c: round(float(df[c].notna().mean() * 100), 1) for c in feature_cols}
    stats["soil"] = {
        "merged": True,
        "source_file": str(SOIL_FEATURES_PATH.relative_to(ROOT)),
        "n_features": len(feature_cols),
        "dataset": "iSDAsoil (via sibling project's coordinate join; see merge_soil_data.py)",
        "coverage_pct": coverage,
    }
    log(f"Merged {len(feature_cols)} soil features (iSDAsoil); "
        f"coverage {min(coverage.values()):.1f}-{max(coverage.values()):.1f}% across columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Drop columns that cannot be features under any screen
# ---------------------------------------------------------------------------
# Four distinct reasons, kept as separate lists so the manifest can report
# *which* reason applied rather than a generic "dropped".

# (a) Direct target leakage. yield_kg_pa is literally the target divided by the
#     acre->hectare constant: corr(yield_kg_pa, yield_kg_ph) = 0.9999999999999934.
#     The other three are computed *from* the target by the cleaning pipeline.
LEAKAGE_COLUMNS = [
    "yield_kg_pa",
    "yield_kg_ph_winsorized",
    "yield_kg_ph_flagged_outlier",
    "yield_kg_ph_crop_failure",
]

# (b) Row identifiers and cleaning-audit columns. `unique_id` is unique per row
#     (a perfect memorization key); `district_raw` is the pre-harmonization
#     spelling kept for audit, and using it would re-introduce the 4-way
#     Kakamega split the cleaning pipeline just fixed.
IDENTIFIER_COLUMNS = ["unique_id", "district_raw"]

# (c) Zero variance within the Kenya-maize scope. country/season/crop are
#     constant by construction (the whole file is Kenya/long_rains/maize);
#     npk_kg_ph_flagged_outlier is constant because zero rows exceeded the NPK
#     ceiling.
ZERO_VARIANCE_COLUMNS = ["country", "season", "crop", "npk_kg_ph_flagged_outlier"]

# (d) Unit-incoherent / never-cleaned columns. Both fail a units sanity check
#     that the cleaning pass did not cover:
#       prev_season_yield : median 180 against a target median of 2,930 kg/ha.
#                           Whatever unit this is, it is not kg/ha, and the
#                           workbook's Variables sheet does not say what it is.
#       comp_kg_pa        : max 40,000 kg/acre of compost. The cleaning pipeline
#                           winsorized comp_wb_pa (wheelbarrows) but never this
#                           parallel kg column, which is also 0% present in 2020.
UNIT_INCOHERENT_COLUMNS = ["prev_season_yield", "comp_kg_pa"]

DROP_REASONS = {
    "leakage": LEAKAGE_COLUMNS,
    "identifier": IDENTIFIER_COLUMNS,
    "zero_variance": ZERO_VARIANCE_COLUMNS,
    "unit_incoherent": UNIT_INCOHERENT_COLUMNS,
}


def drop_unusable_columns(df):
    # Guard: the leakage relationship is the single most important invariant in
    # this pipeline. If a future data pull breaks it, fail loudly rather than
    # quietly training on a column that is no longer a perfect target proxy
    # (or, worse, still is but under a different name).
    if "yield_kg_pa" in df.columns:
        ratio = (df[TARGET] / df["yield_kg_pa"].replace(0, np.nan)).dropna()
        stats["leakage_ratio_mean"] = float(ratio.mean())
        stats["leakage_ratio_std"] = float(ratio.std())
        assert ratio.std() < 1e-5, (
            "yield_kg_pa is no longer an exact rescaling of yield_kg_ph "
            f"(std={ratio.std():.3e}); re-check what changed before trusting "
            "the leakage screen."
        )

    dropped = {}
    for reason, cols in DROP_REASONS.items():
        present = [c for c in cols if c in df.columns]
        dropped[reason] = present
        df = df.drop(columns=present)
    stats["columns_dropped_unusable"] = dropped
    total = sum(len(v) for v in dropped.values())
    log(f"Dropped {total} unusable columns "
        + ", ".join(f"{k}={len(v)}" for k, v in dropped.items()) + ".")
    return df


# ---------------------------------------------------------------------------
# 3. Ordinal encoding of columns stored as text bins
# ---------------------------------------------------------------------------
# Two columns are named like numbers (`_meter`, `_num`) but hold *string range
# labels*. Any generic "loop over the numeric columns" in a downstream script
# either crashes on them (df['distance_meter'].mean() raises TypeError) or
# silently skips them. Encoding to bin midpoints here makes them usable and,
# more importantly, makes the string-ness visible in one place.
DISTANCE_METER_MIDPOINTS = {
    "1-50 m": 25, "50-100 m": 75, "100-200 m": 150,
    "200-500 m": 350, "500-1000 m": 750, "1000+ m": 1500,  # open bin: nominal
}
SEED_DEPTH_MIDPOINTS = {
    "10 cm": 10, "20 cm": 20, "30 cm": 30, "40 cm": 40, "40+ cm": 45,
}
# Only better/same/worse are ordered. "dont_know" and
# "this_is_the_only_field_i_farm" are not points on the scale - they are
# non-responses - so they map to NaN and are captured by fertility_is_missing.
FERTILITY_ORDINAL = {"worse": -1, "same": 0, "better": 1}
SLOPE_ANGLE_DES_ORDINAL = {"gentle": 1, "medium": 2, "steep": 3, "very_steep": 4}

ORDINAL_ENCODINGS = {
    "distance_meter": ("distance_meter_num", DISTANCE_METER_MIDPOINTS),
    "seed_depth_num": ("seed_depth_cm", SEED_DEPTH_MIDPOINTS),
    "fertility": ("fertility_ordinal", FERTILITY_ORDINAL),
    "slope_angle_des": ("slope_angle_des_ordinal", SLOPE_ANGLE_DES_ORDINAL),
}


def encode_ordinals(df):
    encoded = {}
    for src, (dest, mapping) in ORDINAL_ENCODINGS.items():
        if src not in df.columns:
            continue
        df[dest] = df[src].map(mapping).astype("Float64")
        encoded[dest] = {
            "source": src,
            "levels": len(mapping),
            "coverage_pct": round(float(df[dest].notna().mean() * 100), 1),
        }
    stats["ordinal_encodings"] = encoded
    log(f"Ordinal-encoded {len(encoded)} text-bin columns to numeric: "
        f"{sorted(encoded)}.")
    return df


# ---------------------------------------------------------------------------
# 4. Nutrient features
# ---------------------------------------------------------------------------
# CRISP-DM report Section 3.3 asks for a single "total_npk_equivalent_kg_ph -
# weighted sum of DAP/urea/NPK/CAN nitrogen-equivalent content". We build that
# for continuity, but we do NOT stop there, for two measured reasons:
#
#   1. N and P2O5 are not interchangeable. Summing them into one
#      "nitrogen-equivalent" scalar is agronomically meaningless, and it erases
#      the basal-vs-topdress structure that IS the recommendation lever.
#   2. "Five separate sparse columns" mischaracterizes this data. DAP is
#      nonzero in 85.4% of rows and CAN in 62.8% - these are the two real
#      columns (basal P at planting, topdress N at knee height). urea is
#      nonzero in 2.1%, NPK in 0.24%, lime in 4.0%.
#
# Built from the *_winsorized companions, not the raw columns: raw dap_kg_ph
# retains an unclipped tail that produces physically impossible nutrient loads
# (n_kg_ph up to 8,266 kg/ha). The cleaning pipeline already produced the
# capped versions for exactly this purpose.
#
# Mass fractions (N, P2O5, K2O) for Kenyan smallholder maize retail grades:
FERTILIZER_NUTRIENT_FRACTIONS = {
    "dap_kg_ph_winsorized":  (0.18, 0.46, 0.00),  # DAP 18-46-0, a defined compound
    "urea_kg_ph_winsorized": (0.46, 0.00, 0.00),  # urea 46-0-0
    "can_kg_ph_winsorized":  (0.26, 0.00, 0.00),  # CAN 26-0-0, the Kenyan retail
                                                  # grade -- deliberately NOT the
                                                  # European 27% N specification
    "npk_kg_ph_winsorized":  (0.17, 0.17, 0.17),  # ASSUMED 17-17-17, see below
}
# The Kenyan NPK blend is genuinely ambiguous (17-17-17, 20-20-0, 23-23-0,
# 25-5-5 are all retailed for maize) and the survey never records which. Rather
# than defend the choice agronomically, we retire it by measurement: NPK is
# nonzero in 58 of 23,674 rows, so mean n_kg_ph moves by 0.02 kg/ha across the
# entire plausible blend space. The sensitivity table is reproduced in the
# notebook. Alternatives kept here so that check is re-runnable.
NPK_BLEND_ASSUMED = "17-17-17"
NPK_BLEND_ALTERNATIVES = {
    "17-17-17": (0.17, 0.17, 0.17),
    "20-20-0": (0.20, 0.20, 0.00),
    "23-23-0": (0.23, 0.23, 0.00),
    "25-5-5": (0.25, 0.05, 0.05),
}
# lime_kg_ph is deliberately absent: CaCO3 is a pH amendment, not a nutrient.
# Folding it into a nutrient sum is a common and consequential error.


def build_nutrient_features(df):
    n = pd.Series(0.0, index=df.index)
    p = pd.Series(0.0, index=df.index)
    k = pd.Series(0.0, index=df.index)
    any_present = pd.Series(False, index=df.index)

    for col, (fn, fp, fk) in FERTILIZER_NUTRIENT_FRACTIONS.items():
        if col not in df.columns:
            continue
        vals = df[col].astype("float64")
        present = vals.notna()
        any_present |= present
        filled = vals.fillna(0.0)
        n += filled * fn
        p += filled * fp
        k += filled * fk

    # A row with no fertilizer column reported at all is unknown, not zero.
    n[~any_present] = np.nan
    p[~any_present] = np.nan
    k[~any_present] = np.nan

    df["n_kg_ph"] = n
    df["p2o5_kg_ph"] = p
    df["k2o_kg_ph"] = k
    df["total_nutrient_kg_ph"] = n + p + k

    # basal_n_share: what fraction of a plot's nitrogen arrived as DAP at
    # planting rather than as a CAN topdress later. Median 0.41, and the 75th
    # percentile is 1.0 - a quarter of these farmers apply no topdress N at
    # all. That split is a controllable lever the single-scalar version cannot
    # express, and it is the highest-value derived feature in this package.
    dap_n = df["dap_kg_ph_winsorized"].astype("float64").fillna(0.0) * \
        FERTILIZER_NUTRIENT_FRACTIONS["dap_kg_ph_winsorized"][0]
    df["basal_n_share"] = np.where(n > 0, dap_n / n.replace(0, np.nan), np.nan)
    df["topdress_applied"] = (
        df["can_kg_ph_winsorized"].astype("float64") > 0
    ).astype("boolean").where(df["can_kg_ph_winsorized"].notna())

    stats["nutrient_features"] = {
        "npk_blend_assumed": NPK_BLEND_ASSUMED,
        "n_kg_ph_mean": round(float(df["n_kg_ph"].mean()), 2),
        "n_kg_ph_max": round(float(df["n_kg_ph"].max()), 1),
        "p2o5_kg_ph_mean": round(float(df["p2o5_kg_ph"].mean()), 2),
        "k2o_kg_ph_nonzero_rows": int((df["k2o_kg_ph"] > 0).sum()),
        "basal_n_share_median": round(float(df["basal_n_share"].median()), 3),
        "fertilizer_nonzero_rows": {
            c.replace("_winsorized", ""): int((df[c] > 0).sum())
            for c in FERTILIZER_NUTRIENT_FRACTIONS if c in df.columns
        },
    }
    log(f"Built nutrient features (NPK blend assumed {NPK_BLEND_ASSUMED}): "
        f"n_kg_ph mean {df['n_kg_ph'].mean():.1f} kg/ha, "
        f"basal_n_share median {df['basal_n_share'].median():.2f}.")
    return df


def npk_blend_sensitivity(df):
    """Report mean n_kg_ph under each plausible NPK blend. Used by the notebook
    to show the blend assumption cannot matter at this NPK prevalence."""
    rows = []
    base = None
    for name, (fn, fp, fk) in NPK_BLEND_ALTERNATIVES.items():
        n = pd.Series(0.0, index=df.index)
        for col, (dn, dp, dk) in FERTILIZER_NUTRIENT_FRACTIONS.items():
            if col not in df.columns:
                continue
            frac = fn if col.startswith("npk") else dn
            n += df[col].astype("float64").fillna(0.0) * frac
        if base is None:
            base = n
        rows.append({
            "blend": name,
            "mean_n_kg_ph": round(float(n.mean()), 2),
            "max_abs_row_delta_vs_base": round(float((n - base).abs().max()), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Seed features -- replacing the broken `hybrid` column
# ---------------------------------------------------------------------------
# `hybrid` is unusable as a feature, and this is not a coverage problem:
#   - It is True for 100% of 2020's 6,190 rows, against 83% in the train years.
#   - Those same 2020 rows carry 884 `local` and 705 `mixed` seed_category
#     values, and 1,162 rows with localseed_kg_ph > 0.
# So `hybrid` is internally contradicted by the seed columns in its own row: it
# is a defaulted field, not a real 100% adoption event. It passes every
# coverage screen (100%/100%) and would enter the model as a pure year proxy.
# It also makes the report's Section 5.1 "hybrid seed advantage" domain check
# unrunnable on the holdout, since it has no variance there.
#
# The seed *quantity* columns are real and variable in all five years, so the
# replacement is built from those.
def build_seed_features(df):
    hyb = df["hybridseed_kg_ph"].astype("float64")
    loc = df["localseed_kg_ph"].astype("float64")

    df["uses_hybrid_seed"] = (hyb > 0).astype("boolean").where(hyb.notna())
    total_seed = hyb.fillna(0.0) + loc.fillna(0.0)
    both_missing = hyb.isna() & loc.isna()
    df["hybrid_seed_share"] = np.where(
        both_missing | (total_seed <= 0), np.nan, hyb.fillna(0.0) / total_seed
    )

    # Retained under an explicit name so the contradiction stays inspectable,
    # but tagged `excluded` in the manifest so it can never be picked up as a
    # live feature.
    if "hybrid" in df.columns:
        df = df.rename(columns={"hybrid": "hybrid_reported_raw"})

    by_year = df.groupby("year")["uses_hybrid_seed"].mean() * 100
    stats["seed_features"] = {
        "uses_hybrid_seed_pct_by_year": {
            int(y): round(float(v), 1) for y, v in by_year.items()
        },
        "hybrid_reported_raw_pct_by_year": {
            int(y): round(float(v), 1) for y, v in
            (df.groupby("year")["hybrid_reported_raw"].mean() * 100).items()
        } if "hybrid_reported_raw" in df.columns else {},
        "holdout_localseed_positive_rows": int(
            (df.loc[df["year"] == HOLDOUT_YEAR, "localseed_kg_ph"] > 0).sum()
        ),
    }
    log("Built uses_hybrid_seed / hybrid_seed_share; renamed contradicted "
        "`hybrid` -> `hybrid_reported_raw` (excluded from the feature set).")
    return df


# ---------------------------------------------------------------------------
# 6. Wealth index
# ---------------------------------------------------------------------------
# CRISP-DM report Section 3.3 asks for a wealth_index over livestock counts and
# asset binaries, as "first principal component or additive score".
#
# We take the additive option, and the reason is structural rather than a
# matter of taste: there is no single year in which all asset inputs are
# present (owns_cows is the ONLY one available in all five years). PC1 fitted
# on 2017's available assets and PC1 fitted on 2018's would be different linear
# combinations of different variables - they are not the same quantity and
# cannot share a column.
#
# THE TRAP, and why this feature gets no coverage flag:
# The instinctive companion feature - a count of available inputs, or an
# _is_missing / coverage flag - is a year label in disguise. The number of
# available asset inputs per row takes values {2, 3, 5, 7}, which uniquely
# identify 2017, 2019 and 2020; the value 2 covers 2016 and 2018 but over
# *different* asset sets (cows+electricity vs electricity+radio). So it is a
# near-perfect year detector. This inverts the general rule used everywhere
# else in this project ("pair every aggregate with a coverage flag") and is the
# one place that rule must not be applied.
#
# The subtler leak is through scale, not location: a naive global z-score mean
# has a year-flat *mean* but a monotonically shrinking *standard deviation*
# (0.737 -> 0.633 -> 0.503 -> 0.484 -> 0.446) as more inputs become available,
# because averaging more z-scores shrinks variance. A tree reads the year off
# that spread. Re-standardizing within year (step 4) removes it by construction.
WEALTH_COUNT_TO_BINARY = {
    "cows": "cows_binary",
    "chickens": "chicken_binary",
    "goats": "goats_binary",
    "bikes": "bikes_binary",
}
WEALTH_COUNT_ONLY = ["oxen"]
WEALTH_BINARY_ONLY = ["electricity", "radio_binary"]


def build_wealth_index(df):
    owns = {}
    agreement = {}

    for count_col, binary_col in WEALTH_COUNT_TO_BINARY.items():
        name = f"owns_{count_col}"
        count = df[count_col].astype("float64") if count_col in df.columns else None
        binary = df[binary_col].astype("boolean") if binary_col in df.columns else None

        if count is not None and binary is not None:
            # Guard: the harmonization below assumes count>0 and the binary say
            # the same thing. Verified at 100% agreement on the 6,024 holdout
            # rows carrying both. Fail loudly if a new pull breaks that.
            both = count.notna() & binary.notna()
            if both.sum() > 0:
                agree = ((count[both] > 0) == binary[both].astype("boolean")).mean()
                agreement[name] = round(float(agree) * 100, 1)
                assert agree > 0.99, (
                    f"{count_col}>0 and {binary_col} disagree on "
                    f"{(1-agree)*100:.1f}% of {int(both.sum())} rows carrying both; "
                    "the owns_* harmonization is no longer safe."
                )
            series = (count > 0).astype("boolean")
            series = series.where(count.notna(), binary)
        elif count is not None:
            series = (count > 0).astype("boolean")
        else:
            series = binary
        owns[name] = series

    for col in WEALTH_COUNT_ONLY:
        if col in df.columns:
            c = df[col].astype("float64")
            owns[f"owns_{col}"] = (c > 0).astype("boolean")
    for col in WEALTH_BINARY_ONLY:
        if col in df.columns:
            owns[f"owns_{col.replace('_binary', '')}"] = df[col].astype("boolean")

    for name, series in owns.items():
        df[name] = series

    asset_cols = list(owns)
    numeric = df[asset_cols].astype("Float64").astype("float64")

    # Step 2: z-score each input WITHIN year (removes cross-year level shifts).
    z = pd.DataFrame(index=df.index, columns=asset_cols, dtype="float64")
    for year, idx in df.groupby("year").groups.items():
        block = numeric.loc[idx]
        z.loc[idx] = ((block - block.mean()) / block.std(ddof=0)).values

    # Step 3: row-mean over whatever inputs that row actually has.
    raw_index = z.mean(axis=1, skipna=True)
    naive_std_by_year = {
        int(y): round(float(raw_index[df["year"] == y].std()), 3) for y in ALL_YEARS
    }

    # Step 4: re-standardize WITHIN year. This is what closes the scale leak -
    # every year now has mean 0 and sd 1 by construction, so the index's spread
    # can no longer encode which year a row came from.
    final = pd.Series(np.nan, index=df.index, dtype="float64")
    for year, idx in df.groupby("year").groups.items():
        block = raw_index.loc[idx]
        sd = block.std(ddof=0)
        final.loc[idx] = (block - block.mean()) / sd if sd and sd > 0 else 0.0
    df["wealth_index"] = final

    n_inputs = numeric.notna().sum(axis=1)
    stats["wealth_index"] = {
        "asset_columns": asset_cols,
        "count_vs_binary_agreement_pct": agreement,
        "n_inputs_by_year": {
            int(y): sorted(set(n_inputs[df["year"] == y].tolist())) for y in ALL_YEARS
        },
        "naive_index_std_by_year": naive_std_by_year,
        "final_std_by_year": {
            int(y): round(float(final[df["year"] == y].std()), 3) for y in ALL_YEARS
        },
        "corr_with_target": round(
            float(df[["wealth_index", TARGET]].corr().iloc[0, 1]), 4
        ),
    }
    log(f"Built wealth_index from {len(asset_cols)} harmonized asset columns "
        f"(within-year standardized; corr with target "
        f"{stats['wealth_index']['corr_with_target']:.3f}). "
        "No coverage flag emitted -- it would be a year label.")
    return df


# ---------------------------------------------------------------------------
# 7. Pest & disease
# ---------------------------------------------------------------------------
# Section 3.3 asks for "pest_disease_any retained as coarse aggregate". The
# obvious implementation - OR together the 8 specific pest binaries - is wrong
# here, and measurably so: the train years carry 1-4 of those columns while
# 2020 carries all 8 at ~100%. An OR over a different set of inputs each year
# is a different variable each year, which is precisely the failure this
# pipeline exists to prevent.
#
# The survey's own `pest_disease` column already asks the coarse question
# directly and is stable (100/100/100/99.2 across 2017-2020), so we use it. The
# two do not agree - 973 rows report pest_disease=0 while a specific pest
# binary is 1 - which is documented as a finding rather than silently
# reconciled.
PEST_BINARY_COLUMNS = [
    "FAW", "stemborer", "msv", "mlnd", "cutworms", "aphids", "blight", "striga",
]


def build_pest_disease_features(df):
    present = [c for c in PEST_BINARY_COLUMNS if c in df.columns]
    any_specific = df[present].astype("Float64").fillna(0).sum(axis=1) > 0
    coarse = df["pest_disease"].astype("boolean")

    disagree = int(((coarse == False) & any_specific).sum())

    coverage = {
        c: {int(y): round(float(v), 1)
            for y, v in (df.groupby("year")[c].apply(lambda s: s.notna().mean() * 100)).items()}
        for c in present
    }
    stats["pest_disease"] = {
        "coarse_column_used": "pest_disease",
        "specific_binaries_available": present,
        "specific_binary_coverage_by_year": coverage,
        "rows_coarse_false_but_specific_true": disagree,
    }
    log(f"Kept survey `pest_disease` as the coarse aggregate rather than OR-ing "
        f"{len(present)} year-fragmented pest binaries "
        f"({disagree} rows disagree between the two -- documented, not reconciled).")
    return df


# ---------------------------------------------------------------------------
# 8. Remaining agronomic features
# ---------------------------------------------------------------------------
LEGUME_TOKENS = ("bean", "groundnut", "peanut", "cowpea", "soy", "green_gram", "pea")


def build_agronomic_features(df):
    # plant_date_doy: the only genuinely ex-ante *timing* lever a farmer
    # controls. Signal is weak and sign-unstable year to year - built because
    # it is the one timing choice available, not because it is strong.
    df["plant_date_doy"] = df["plant_date"].dt.dayofyear.astype("Float64")

    # Design spacing, from the intended row/plant spacing recorded at planting.
    # Deliberately NOT unified with plants_sqm_spacing: see ROLE_OVERRIDES.
    if {"row_spacing", "plant_spacing"} <= set(df.columns):
        rs = df["row_spacing"].astype("float64")
        ps = df["plant_spacing"].astype("float64")
        denom = (rs * ps).replace(0, np.nan)
        df["design_plant_density"] = 10000.0 / denom

    # Compost stays in wheelbarrows, converted only by the same verified
    # acre->hectare constant the cleaning pipeline used. We explicitly REFUSE
    # to convert wheelbarrows to kg of nitrogen: that needs a wheelbarrow mass
    # (50-70 kg wet) times a manure N fraction (0.5-1.5% of dry matter), a
    # product spanning an order of magnitude, and comp_quality is only 8%
    # populated so the guess cannot even be conditioned on decomposition state.
    if "comp_wb_pa_winsorized" in df.columns:
        wb = df["comp_wb_pa_winsorized"].astype("float64")
        df["compost_wb_ph"] = wb * 2.471050
        df["compost_applied"] = (wb > 0).astype("boolean").where(wb.notna())

    # Intercropping with a legume is agronomically distinct from intercropping
    # generally (biological N fixation), and 99.8% of intercrop_type values are
    # a single token, so a coarse boolean loses almost nothing.
    if "intercrop_type" in df.columns:
        it = df["intercrop_type"].astype("string").str.lower()
        is_leg = it.apply(
            lambda v: pd.NA if pd.isna(v) else any(t in v for t in LEGUME_TOKENS)
        ).astype("boolean")
        df["intercrop_is_legume"] = is_leg

    stats["agronomic_features"] = {
        "plant_date_doy_coverage_pct": round(float(df["plant_date_doy"].notna().mean() * 100), 1),
        "design_plant_density_coverage_pct": round(
            float(df["design_plant_density"].notna().mean() * 100), 1
        ) if "design_plant_density" in df.columns else None,
        "intercrop_is_legume_true_rows": int(
            df["intercrop_is_legume"].fillna(False).sum()
        ) if "intercrop_is_legume" in df.columns else None,
        "compost_to_nitrogen_conversion": "refused -- see docstring",
    }
    log("Built plant_date_doy, design_plant_density, compost_wb_ph/applied, "
        "intercrop_is_legume. Compost->N conversion deliberately refused.")
    return df


# ---------------------------------------------------------------------------
# 9. Missingness flags (Section 3.3 mandate)
# ---------------------------------------------------------------------------
# The cleaning pipeline already added _is_missing flags for 7 sparse columns.
# Section 3.3 additionally names fertility, slope_angle_num and hybrid. The
# first two already have flags; the hybrid replacement needs one.
EXTRA_MISSINGNESS_FLAGS = ["uses_hybrid_seed", "n_kg_ph", "plant_date_doy"]


def add_missingness_flags(df):
    added = []
    for col in EXTRA_MISSINGNESS_FLAGS:
        if col in df.columns:
            flag = f"{col}_is_missing"
            df[flag] = df[col].isna()
            added.append(flag)
    stats["missingness_flags_added"] = added
    log(f"Added {len(added)} missingness indicators: {added}.")
    return df


# ---------------------------------------------------------------------------
# 10. Feature stability screen -- three gates
# ---------------------------------------------------------------------------
# Threshold choice barely matters here (a 50% cut keeps 74 columns, a 90% cut
# keeps 60); the *shape* of the test is what matters. A single pooled
# "coverage over 2016-2019" screen is actively misleading, because pooling
# hides that 2016 is a reduced survey instrument: fertility, pest_disease,
# weed, chickens, goats, oxen and growing_season_days are all 0% that year, and
# pooled coverage still reads 70-75%.
COVERAGE_THRESHOLD_PCT = 60.0
GATE_B_MIN_TRAIN_YEARS = 2


# The share of a feature's total variance that lies BETWEEN survey years rather
# than within them -- a one-way ANOVA eta-squared with year as the factor. 0.0
# means the feature says nothing about which year a row came from; 1.0 means it
# says nothing else.
#
# Gates A-D catch year proxies that betray themselves through COVERAGE (a column
# absent in 2016, constant in the holdout). This catches the other kind: a
# column present and varying in every year whose values still separate the years
# cleanly. Nothing is dropped on it -- a genuinely year-varying quantity like
# season rainfall IS mostly a year effect, and that is the truth about rainfall
# rather than a defect in the column -- but it is reported per feature so the
# modelling package can see which inputs are carrying a season label and weigh
# them against the out-of-time holdout deliberately.
YEAR_VARIANCE_WARN = 0.5


def _year_variance_shares(df, columns):
    shares = {}
    year = df["year"]
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.notna()
        if valid.sum() < 2:
            continue
        s, y = s[valid], year[valid]
        total = float(((s - s.mean()) ** 2).sum())
        if total <= 0:
            continue  # constant: no variance to apportion
        grand = s.mean()
        grouped = y.to_frame("y").assign(v=s).groupby("y")["v"].agg(["count", "mean"])
        between = float((grouped["count"] * (grouped["mean"] - grand) ** 2).sum())
        shares[col] = round(between / total, 4)
    return shares


def _coverage_by_year(df, columns):
    cov = {}
    for year in ALL_YEARS:
        block = df.loc[df["year"] == year, columns]
        cov[year] = block.notna().mean() * 100
    return pd.DataFrame(cov)


def screen_feature_stability(df):
    candidates = [c for c in df.columns if c not in (TARGET, "year")]
    cov = _coverage_by_year(df, candidates).round(1)

    # Gate A -- holdout viability. Hard and deliberately ASYMMETRIC: a column
    # that is absent in 2020 cannot be scored on the holdout at any train
    # coverage, so this is not the same question as train learnability.
    gate_a = cov[HOLDOUT_YEAR] >= COVERAGE_THRESHOLD_PCT

    # Gate B -- train learnability. Requires presence in >=2 TRAIN YEARS, not
    # >=60% pooled. Pooled coverage can be satisfied by a single year; two
    # years is the minimum at which a model can distinguish "this variable"
    # from "that year".
    gate_b_full = (cov[list(TRAIN_YEARS_FULL)] >= COVERAGE_THRESHOLD_PCT).sum(axis=1) \
        >= GATE_B_MIN_TRAIN_YEARS
    gate_b_core = (cov[list(TRAIN_YEARS_CORE)] >= COVERAGE_THRESHOLD_PCT).sum(axis=1) \
        >= GATE_B_MIN_TRAIN_YEARS

    # Gate C -- the rows-vs-columns trade. Columns that clear A and B but are
    # absent in 2016 specifically are not dropped; instead the 2016 ROWS are
    # dropped in the `core` output. 3,876 rows (16.4%) buys back roughly ten
    # substantive columns. Both outputs are emitted so the modeling package can
    # benchmark the trade rather than inherit it.
    in_core = gate_a & gate_b_core
    in_full = gate_a & gate_b_full & (cov[CORE_DROPS_YEAR] >= COVERAGE_THRESHOLD_PCT)
    gate_c = in_core & ~in_full

    screen = pd.DataFrame({
        "gate_a_holdout": gate_a,
        "gate_b_train_full": gate_b_full,
        "gate_b_train_core": gate_b_core,
        "gate_c_needs_2016_drop": gate_c,
        "in_full": in_full,
        "in_core": in_core,
    })
    for year in ALL_YEARS:
        screen[f"cov_{year}"] = cov[year]
    screen["cov_min_year"] = cov.min(axis=1).round(1)
    screen["n_years_ge_threshold"] = (cov >= COVERAGE_THRESHOLD_PCT).sum(axis=1)

    # Variance checks, run PER PERIOD rather than globally -- this is what
    # catches hybrid_reported_raw, which has variance overall but none in the
    # holdout.
    nzv, const_holdout = {}, {}
    holdout = df[df["year"] == HOLDOUT_YEAR]
    for col in candidates:
        s = df[col].dropna()
        nzv[col] = bool(len(s) > 0 and (s.value_counts(normalize=True).iloc[0] >= 0.95))
        hs = holdout[col].dropna()
        const_holdout[col] = bool(len(hs) > 0 and hs.nunique() <= 1)
    screen["near_zero_variance"] = pd.Series(nzv)
    screen["constant_in_holdout"] = pd.Series(const_holdout)
    screen["year_variance_share"] = pd.Series(_year_variance_shares(df, candidates))

    stats["screen"] = {
        "coverage_threshold_pct": COVERAGE_THRESHOLD_PCT,
        "gate_b_min_train_years": GATE_B_MIN_TRAIN_YEARS,
        "candidates": len(candidates),
        "failed_gate_a": sorted(screen.index[~screen["gate_a_holdout"]].tolist()),
        "gate_c_columns": sorted(screen.index[screen["gate_c_needs_2016_drop"]].tolist()),
        "n_in_full": int(screen["in_full"].sum()),
        "n_in_core": int(screen["in_core"].sum()),
        "constant_in_holdout": sorted(screen.index[screen["constant_in_holdout"]].tolist()),
    }
    log(f"Stability screen over {len(candidates)} candidates: "
        f"{int((~gate_a).sum())} fail Gate A (holdout), "
        f"{int(gate_c.sum())} are Gate C (bought by dropping 2016). "
        f"in_full={int(in_full.sum())}, in_core={int(in_core.sum())}.")
    return df, screen


# ---------------------------------------------------------------------------
# 11. Roles and availability tiers
# ---------------------------------------------------------------------------
# These two axes are orthogonal and both are needed, because they are how the
# four downstream consumer surfaces select their inputs:
#
#   recommendation layer (report 4.1)  -> role == "lever" AND tier == "ex_ante"
#   underwriting / loan sizing (1.1a#1)-> tier == "ex_ante"
#   mid-season re-forecast (1.1a#2)    -> tier in {ex_ante, mid_season}
#   post-season yield estimation       -> every tier except "leaky"
#
# IMPORTANT FRAMING: the entire survey is administered post-harvest, so every
# variable in this dataset was physically *recorded* after the outcome. The
# tier is about what the variable REFERS TO, not when it was written down.
# Without that distinction the taxonomy reads as arbitrary.
ROLE_LEVER = [
    "dap_kg_ph", "dap_kg_ph_winsorized", "can_kg_ph", "can_kg_ph_winsorized",
    "urea_kg_ph", "urea_kg_ph_winsorized", "npk_kg_ph", "npk_kg_ph_winsorized",
    "lime_kg_ph", "lime_kg_ph_winsorized",
    "n_kg_ph", "p2o5_kg_ph", "k2o_kg_ph", "total_nutrient_kg_ph",
    "basal_n_share", "topdress_applied",
    "localseed_kg_ph", "hybridseed_kg_ph", "uses_hybrid_seed",
    "hybrid_seed_share", "seed_category", "seed_type_primary", "seed_type_clean",
    "seed_type", "seed_type_is_mixed", "seed_depth_cm", "seed_depth_est",
    "plant_date_doy", "design_plant_density", "row_spacing", "plant_spacing",
    "compost_wb_ph", "compost_applied", "comp_wb_pa", "comp_wb_pa_winsorized",
    "comp_method", "comp_source", "comp_source_manure", "comp_quality",
    "compost", "lime_method", "basal_method",
    "intercrop", "intercrop_type", "intercrop_is_legume",
    "weed", "pesticide",
]
ROLE_CONDITION = [
    "district", "plot_acres", "plot_hectares", "hh_num", "hh_num_under18",
    "hh_num_under5", "wealth_index", "fertility", "fertility_ordinal",
    "slope", "slope_angle_num", "slope_angle_des", "slope_angle_des_ordinal",
    "distance_meter", "distance_meter_num",
    "field_latitude", "field_longitude",
    "cows", "chickens", "goats", "bikes", "oxen", "electricity", "radio_binary",
    "cows_binary", "chicken_binary", "goats_binary", "bikes_binary",
    "owns_cows", "owns_chickens", "owns_goats", "owns_bikes", "owns_oxen",
    "owns_electricity", "owns_radio",
]
ROLE_SHOCK = ["drought", "flood", "pest_disease"] + PEST_BINARY_COLUMNS
ROLE_GROUP_KEY = ["site", "cv_group", "cv_fold"]
ROLE_METADATA = ["year", "is_holdout"]
# growing_season_days is a real feature, not bookkeeping -- it is only ex_post.
ROLE_CONDITION_EXTRA = ["growing_season_days"]

# Raw datetime columns. Emitted nowhere as features: a GBM cannot consume a
# datetime, and both are already represented by columns that can be consumed
# (plant_date -> plant_date_doy, harvest_date -> growing_season_days).
RAW_DATE_COLUMNS = ["plant_date", "harvest_date"]

TIER_EX_POST = [
    "harvest_date", "growing_season_days", "growing_season_days_flagged",
    "plants_sqm_spacing",
]
TIER_MID_SEASON = ["drought", "flood", "pest_disease", "weed", "pesticide"] + \
    PEST_BINARY_COLUMNS

# Columns kept in the file but never usable as live features, with the reason.
ROLE_OVERRIDES = {
    "hybrid_reported_raw": (
        "excluded", "ex_ante",
        "Constant True across all 6,190 holdout rows while the same rows carry "
        "884 local / 705 mixed seed_category and 1,162 with localseed_kg_ph>0 -- "
        "a defaulted field and a pure year proxy. Replaced by uses_hybrid_seed."
    ),
    "plants_sqm_spacing": (
        "excluded", "ex_post",
        "A realized stand count from the harvest quadrat, not a spacing choice: "
        "corr with yield 0.534 in 2020 vs 0.097 for design spacing in train, and "
        "all 26 rows where it is 0 have exactly 0 yield. Quarantined rather than "
        "unified with design_plant_density."
    ),
    "field_latitude": (
        "excluded", "ex_ante",
        "5-decimal coordinates are a household fingerprint, and the geographic "
        "gradient is itself unstable: corr(field_longitude, yield) runs "
        "0.297/0.299/0.164/0.457 across train years then collapses to 0.045 in "
        "the holdout, which sampled 50 districts against train's 40."
    ),
    "field_longitude": (
        "excluded", "ex_ante",
        "See field_latitude -- same fingerprinting and same out-of-time collapse."
    ),
    "site": (
        "group_key", "ex_ante",
        "CV grouping key only, never a feature and never target-encoded: 2,133 "
        "levels over 23,674 rows, median 9 train rows per site, and train-site "
        "means correlate only 0.286 with holdout-site means. Use district "
        "(0.717) if a geographic encoding is wanted."
    ),
}


FLAG_SUFFIXES = (
    "_is_missing", "_flagged_outlier", "_flagged_invalid",
    "_flagged_implausible", "_flagged", "_flag",
)


def assign_roles_and_tiers(df, screen):
    roles, tiers, notes = {}, {}, {}
    for col in screen.index:
        if col in ROLE_OVERRIDES:
            roles[col], tiers[col], notes[col] = ROLE_OVERRIDES[col]
            continue
        notes[col] = ""
        if col in WEATHER_FEATURE_COLUMNS:
            # Handled first so a weather column is never mistaken for something
            # else by a name-suffix rule below -- `weather_location_is_field`
            # would otherwise be swept up by the FLAG_SUFFIXES branch, and the
            # tier rules here are specific to what the column measures.
            roles[col] = ("flag" if col == "weather_location_is_field"
                          else _weather_role(col))
            tiers[col] = ("ex_ante" if col == "weather_location_is_field"
                          else _weather_tier(col))
            continue
        if col in SOIL_FEATURE_COLUMNS:
            # Fixed geography, like a weather normal: knowable years before
            # planting and not a lever the farmer controls.
            roles[col] = "condition"
            tiers[col] = "ex_ante"
            continue
        if col in ROLE_GROUP_KEY:
            roles[col] = "group_key"
        elif col in ROLE_METADATA or col in RAW_DATE_COLUMNS:
            roles[col] = "metadata"
        elif col in ROLE_SHOCK:
            roles[col] = "shock"
        elif col in ROLE_CONDITION_EXTRA:
            roles[col] = "condition"
        elif col.endswith(FLAG_SUFFIXES):
            # Data-quality and missingness indicators. Section 3.3 wants these
            # kept ("distinguish 'not applicable here' from 'applicable but
            # unreported'"), but they are a different kind of thing from an
            # agronomic measurement and a consumer should be able to take or
            # leave the whole class deliberately.
            roles[col] = "flag"
        elif col in ROLE_LEVER:
            roles[col] = "lever"
        elif col in ROLE_CONDITION:
            roles[col] = "condition"
        else:
            roles[col] = "metadata"

        if col in LEAKAGE_COLUMNS:
            tiers[col] = "leaky"
        elif col in TIER_EX_POST:
            tiers[col] = "ex_post"
        elif col in TIER_MID_SEASON:
            tiers[col] = "mid_season"
        else:
            tiers[col] = "ex_ante"

    screen["role"] = pd.Series(roles)
    screen["availability_tier"] = pd.Series(tiers)
    screen["notes"] = pd.Series(notes)

    # Gate D -- no variance in the holdout. A column with a single value across
    # every holdout row cannot contribute anything to an out-of-time
    # prediction; all it can do is encode training-year structure. In practice
    # this catches exactly the year proxies: hybridseed_2017_methodology_flag
    # (True only in 2017), distance_meter_is_missing (distance_meter is 0% in
    # 2020, so the flag is uniformly True there), comp_method_is_missing, and
    # lime_kg_ph_flagged_outlier. Split/grouping columns are exempt: they are
    # emitted as infrastructure, never consumed as features.
    infrastructure = screen["role"].isin(["group_key", "metadata"])
    gate_d_fail = screen["constant_in_holdout"] & ~infrastructure
    screen["gate_d_holdout_variance"] = ~gate_d_fail
    screen.loc[gate_d_fail, ["in_full", "in_core"]] = False
    screen.loc[gate_d_fail, "notes"] = (
        "Single value across every holdout row, so it can only encode "
        "training-year structure -- dropped by Gate D."
    )
    stats["gate_d_dropped"] = sorted(screen.index[gate_d_fail].tolist())

    # Roles that are emitted as infrastructure rather than consumed as
    # features never count toward either feature set.
    screen.loc[infrastructure | (screen["role"] == "excluded"),
               ["in_full", "in_core"]] = False

    stats["roles"] = screen["role"].value_counts().to_dict()
    stats["availability_tiers"] = screen["availability_tier"].value_counts().to_dict()
    log(f"Gate D dropped {len(stats['gate_d_dropped'])} holdout-constant columns: "
        f"{stats['gate_d_dropped']}.")
    log("Assigned roles "
        + ", ".join(f"{k}={v}" for k, v in stats["roles"].items())
        + " | tiers "
        + ", ".join(f"{k}={v}" for k, v in stats["availability_tiers"].items())
        + ".")
    return screen


# ---------------------------------------------------------------------------
# 12. Cross-validation folds
# ---------------------------------------------------------------------------
# Report Section 3.5 mandates grouped k-fold by site for in-sample tuning, and
# an out-of-time holdout for the reported number. Emitting the fold assignment
# HERE, rather than leaving it to the modeling package, is what prevents a
# well-meaning random row split later: multiple plots from the same site across
# seasons are correlated, and a random split leaks them across the fold
# boundary.
#
# Implemented as a balanced greedy GroupKFold (numpy only, seeded): sites are
# shuffled, sorted by descending row count, then each is assigned to whichever
# fold currently holds the fewest rows.
N_FOLDS = 5
FOLD_SEED = 20260903


def assign_cv_folds(df):
    df["cv_group"] = df["site"].astype("string")
    df["is_holdout"] = df["year"] == HOLDOUT_YEAR

    train_mask = ~df["is_holdout"]
    counts = df.loc[train_mask, "cv_group"].value_counts()

    rng = np.random.default_rng(FOLD_SEED)
    sites = counts.index.to_numpy()
    shuffled = rng.permutation(len(sites))
    sites = sites[shuffled]
    sizes = counts.to_numpy()[shuffled]
    order = np.argsort(-sizes, kind="stable")

    fold_loads = np.zeros(N_FOLDS, dtype=np.int64)
    site_to_fold = {}
    for i in order:
        f = int(np.argmin(fold_loads))
        site_to_fold[sites[i]] = f
        fold_loads[f] += sizes[i]

    df["cv_fold"] = df["cv_group"].map(site_to_fold).astype("Int64")
    df.loc[df["is_holdout"], "cv_fold"] = -1  # holdout is never in a train fold

    stats["cv_folds"] = {
        "n_folds": N_FOLDS,
        "seed": FOLD_SEED,
        "group_key": "site",
        "train_sites": int(len(site_to_fold)),
        "rows_per_fold": {
            int(f): int(n) for f, n in
            df.loc[train_mask, "cv_fold"].value_counts().sort_index().items()
        },
        "holdout_rows": int(df["is_holdout"].sum()),
    }
    log(f"Assigned {N_FOLDS} site-grouped CV folds over "
        f"{len(site_to_fold)} training sites (seed {FOLD_SEED}); "
        f"{int(df['is_holdout'].sum())} holdout rows marked cv_fold=-1.")
    return df


# ---------------------------------------------------------------------------
# 13. Manifest & save
# ---------------------------------------------------------------------------
DERIVED_SOURCES = {
    "n_kg_ph": "dap_kg_ph_winsorized|urea_kg_ph_winsorized|can_kg_ph_winsorized|npk_kg_ph_winsorized",
    "p2o5_kg_ph": "dap_kg_ph_winsorized|npk_kg_ph_winsorized",
    "k2o_kg_ph": "npk_kg_ph_winsorized",
    "total_nutrient_kg_ph": "n_kg_ph|p2o5_kg_ph|k2o_kg_ph",
    "basal_n_share": "dap_kg_ph_winsorized|n_kg_ph",
    "topdress_applied": "can_kg_ph_winsorized",
    "uses_hybrid_seed": "hybridseed_kg_ph",
    "hybrid_seed_share": "hybridseed_kg_ph|localseed_kg_ph",
    "wealth_index": "owns_cows|owns_chickens|owns_goats|owns_bikes|owns_oxen|owns_electricity|owns_radio",
    "plant_date_doy": "plant_date",
    "design_plant_density": "row_spacing|plant_spacing",
    "compost_wb_ph": "comp_wb_pa_winsorized",
    "compost_applied": "comp_wb_pa_winsorized",
    "intercrop_is_legume": "intercrop_type",
    "distance_meter_num": "distance_meter",
    "seed_depth_cm": "seed_depth_num",
    "fertility_ordinal": "fertility",
    "slope_angle_des_ordinal": "slope_angle_des",
    "cv_fold": "site",
    "cv_group": "site",
    "is_holdout": "year",
}
AGGREGATE_FEATURES = {"wealth_index", "total_nutrient_kg_ph", "n_kg_ph", "p2o5_kg_ph", "k2o_kg_ph"}


def _weather_built_from(name):
    """Which Earth Engine collection a weather column ultimately came from.
    Rain-day counts and rainfall are CHIRPS; everything temperature-derived,
    degree days included, is ERA5-Land."""
    if name == "weather_location_is_field":
        return "weather_locations.py fallback ladder"
    if name.startswith("rain"):
        return "UCSB-CHG/CHIRPS/DAILY"
    return "ECMWF/ERA5_LAND/DAILY_AGGR"


def _decision_reason(row, name):
    if row["notes"]:
        return row["notes"]
    if name in WEATHER_FEATURE_COLUMNS and (row["in_full"] or row["in_core"]):
        share = row["year_variance_share"]
        # Weather clears the coverage gates trivially, so the useful thing to
        # record against it is the risk the gates cannot see.
        if pd.notna(share):
            return (f"External {row['availability_tier']} weather covariate; "
                    f"{share:.0%} of its variance is between seasons rather "
                    f"than between places"
                    + (" -- treat as substantially a year label."
                       if share >= YEAR_VARIANCE_WARN else "."))
        return f"External {row['availability_tier']} weather covariate."
    if row["role"] == "group_key":
        return "Cross-validation grouping key, emitted as infrastructure."
    if row["role"] == "metadata":
        if name in RAW_DATE_COLUMNS:
            return ("Raw datetime, unusable by a tree model directly and already "
                    "represented by plant_date_doy / growing_season_days.")
        return "Split or bookkeeping column, emitted but not a feature."
    if row["in_full"]:
        return "Stable across all five survey years and the holdout."
    if row["in_core"]:
        return (f"0% coverage in {CORE_DROPS_YEAR} (reduced survey instrument); "
                "recovered in the core file by dropping that year's rows.")
    if not row["gate_a_holdout"]:
        return (f"Gate A: only {row[f'cov_{HOLDOUT_YEAR}']:.1f}% coverage in the "
                f"{HOLDOUT_YEAR} holdout, so it cannot be scored out-of-time.")
    return (f"Gate B: present in fewer than {GATE_B_MIN_TRAIN_YEARS} training "
            "years, so the model cannot separate it from a year effect.")


def build_manifest(df, screen):
    rows = []
    for name, row in screen.iterrows():
        if name in WEATHER_FEATURE_COLUMNS or name in SOIL_FEATURE_COLUMNS:
            source = "external"
        elif name in DERIVED_SOURCES:
            source = "aggregate" if name in AGGREGATE_FEATURES else "derived"
        elif name.endswith(FLAG_SUFFIXES):
            source = "flag"
        else:
            source = "raw"

        if row["role"] == "excluded":
            decision = "quarantine"
        elif row["role"] in ("group_key", "metadata") and name in ALWAYS_EMIT + RAW_DATE_COLUMNS:
            decision = "infrastructure"
        elif row["in_full"]:
            decision = "keep"
        elif row["in_core"]:
            decision = "keep_core_only"
        else:
            decision = "drop"

        rows.append({
            "feature": name,
            "source": source,
            "built_from": (_weather_built_from(name) if name in WEATHER_FEATURE_COLUMNS
                           else "iSDAsoil via coordinate join (see merge_soil_data.py)"
                           if name in SOIL_FEATURE_COLUMNS
                           else DERIVED_SOURCES.get(name, "")),
            "dtype": str(df[name].dtype) if name in df.columns else "",
            "role": row["role"],
            "availability_tier": row["availability_tier"],
            **{f"cov_{y}": row[f"cov_{y}"] for y in ALL_YEARS},
            "cov_min_year": row["cov_min_year"],
            "n_years_ge_60": int(row["n_years_ge_threshold"]),
            "gate_a_holdout": bool(row["gate_a_holdout"]),
            "gate_b_train": bool(row["gate_b_train_full"]),
            "gate_c_needs_2016_drop": bool(row["gate_c_needs_2016_drop"]),
            "gate_d_holdout_variance": bool(row["gate_d_holdout_variance"]),
            "in_full": bool(row["in_full"]),
            "in_core": bool(row["in_core"]),
            "near_zero_variance": bool(row["near_zero_variance"]),
            "constant_in_holdout": bool(row["constant_in_holdout"]),
            "year_variance_share": row["year_variance_share"],
            "decision": decision,
            "decision_reason": _decision_reason(row, name),
        })

    manifest = pd.DataFrame(rows).sort_values(
        ["decision", "role", "feature"]
    ).reset_index(drop=True)
    stats["manifest_decisions"] = manifest["decision"].value_counts().to_dict()

    kept = manifest[manifest["decision"].isin(["keep", "keep_core_only"])]
    year_labels = kept[kept["year_variance_share"] >= YEAR_VARIANCE_WARN]
    stats["year_variance"] = {
        "warn_threshold": YEAR_VARIANCE_WARN,
        "kept_features_above_threshold": int(len(year_labels)),
        "worst": year_labels.nlargest(10, "year_variance_share")[
            ["feature", "source", "year_variance_share"]
        ].to_dict("records"),
    }
    if len(year_labels):
        log(f"{len(year_labels)} kept feature(s) carry >={YEAR_VARIANCE_WARN:.0%} "
            f"of their variance between years rather than within them "
            f"(worst: "
            + ", ".join(
                f"{r.feature} {r.year_variance_share:.2f}"
                for r in year_labels.nlargest(3, "year_variance_share").itertuples()
            )
            + "). Not dropped -- see year_variance_share in the manifest.")
    log(f"Built manifest over {len(manifest)} candidate columns: "
        + ", ".join(f"{k}={v}" for k, v in stats["manifest_decisions"].items()) + ".")
    return manifest


# Emitted in both outputs regardless of the screen: the split/grouping
# infrastructure, plus the raw dates so a consumer can re-derive timing
# features for the mid-season surface without going back to the cleaned file.
ALWAYS_EMIT = ["year", "is_holdout", "cv_group", "cv_fold", "site", "district",
               "plant_date", "harvest_date"]


def save(df, manifest):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    full_cols = manifest.loc[manifest["in_full"], "feature"].tolist()
    core_cols = manifest.loc[manifest["in_core"], "feature"].tolist()

    def assemble(cols):
        keep, seen = [], set()
        for c in [TARGET] + ALWAYS_EMIT + cols:
            if c in df.columns and c not in seen:
                keep.append(c)
                seen.add(c)
        return df[keep]

    full = assemble(full_cols)
    core = assemble(core_cols)
    core = core[core["year"] != CORE_DROPS_YEAR].reset_index(drop=True)

    # Guards. These are the invariants the whole pipeline exists to establish;
    # a future change that breaks one should fail here, not in a model that
    # scores suspiciously well.
    leaky = set(manifest.loc[manifest["availability_tier"] == "leaky", "feature"])
    for name, frame in (("full", full), ("core", core)):
        assert not (leaky & set(frame.columns)), \
            f"{name}: leaky columns reached the output: {leaky & set(frame.columns)}"
        assert "yield_kg_pa" not in frame.columns, f"{name}: yield_kg_pa leaked"
        assert "hybrid_reported_raw" not in frame.columns, \
            f"{name}: the contradicted `hybrid` column leaked into features"
        assert frame[TARGET].notna().all(), f"{name}: target has nulls"

    assert len(full) == len(df), "full output should keep every row"
    assert (core["year"] != CORE_DROPS_YEAR).all(), "core output still holds 2016 rows"

    full.to_csv(FEATURES_FULL_PATH, index=False)
    core.to_csv(FEATURES_CORE_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH, index=False)

    stats["output_full_shape"] = list(full.shape)
    stats["output_core_shape"] = list(core.shape)
    stats["output_full_columns"] = full.columns.tolist()
    stats["output_core_columns"] = core.columns.tolist()
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    log(f"Saved full features:  {FEATURES_FULL_PATH} ({full.shape[0]} x {full.shape[1]})")
    log(f"Saved core features:  {FEATURES_CORE_PATH} ({core.shape[0]} x {core.shape[1]})")
    log(f"Saved manifest:       {MANIFEST_PATH} ({len(manifest)} rows)")
    log(f"Saved audit stats:    {STATS_PATH}")
    return full, core, manifest


def run():
    df = load_cleaned()
    df = merge_weather_features(df)
    df = merge_soil_features(df)
    df = drop_unusable_columns(df)
    df = encode_ordinals(df)
    df = build_nutrient_features(df)
    df = build_seed_features(df)
    df = build_wealth_index(df)
    df = build_pest_disease_features(df)
    df = build_agronomic_features(df)
    df = add_missingness_flags(df)
    df = assign_cv_folds(df)
    df, screen = screen_feature_stability(df)
    screen = assign_roles_and_tiers(df, screen)
    manifest = build_manifest(df, screen)
    return save(df, manifest)


if __name__ == "__main__":
    run()
