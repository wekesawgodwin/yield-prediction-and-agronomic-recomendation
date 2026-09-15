"""Does adding soil properties to the existing recipe improve the district model?

This is deliberately the CHEAP experiment, not the full one: it refits the
notebook 06 recipe (the already-tuned top-5-configs-per-family, blended with
equal weights) exactly as train_district_model.py does, once with the current
`kenya_maize_recommended_features.csv` and once with the 7 soil columns added
on top -- same protocol (train 2016-2019, val 2018/2019, test 2020), same
hyperparameters, only the feature set differs. It answers "does this new
information help the existing model" without redoing notebook 03's
permutation-importance feature selection or notebook 05's hyperparameter
search for either arm.

If this shows a real gain, the honest next step is re-running 03 (to see
where soil ranks and whether it belongs in the curated recommended-features
list on its own merits) and 05 (to see whether hyperparameters tuned WITH
soil available do even better) -- not baking this shortcut in permanently.

Run:
    python scripts/experiment_soil_ablation.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from district_model import (DEFAULT_RECIPE, DistrictPipeline, PlotRecipe, Schema,
                            district_metrics, load_frame, load_schema, project_root)

SOIL_COLS = [
    "soil_ph", "soil_organic_carbon_gkg", "soil_nitrogen_gkg", "soil_clay_pct",
    "soil_sand_pct", "soil_cec_cmolkg", "soil_bulk_density_gcm3",
]

TRAIN_YEARS = [2016, 2017, 2018, 2019]
VAL_YEARS = [2018, 2019]
TEST_YEAR = 2020


def run_arm(name, schema, frame, recipe, min_plots=20):
    t0 = time.time()
    print(f"\n=== {name} ({len(schema.feats)} features) ===")
    pipe = DistrictPipeline(schema, recipe, use_nn=True, min_plots=min_plots)
    pipe.fit(frame, TRAIN_YEARS, VAL_YEARS)

    test = frame[frame.year == TEST_YEAR]
    table = pipe.predict_districts(test)
    m = district_metrics(table.actual_mean.to_numpy(), table.pred_mean.to_numpy(),
                         table.n_plots.to_numpy())
    cover = float(table.inside_band.mean())
    print(f"  {name}: test {TEST_YEAR} districts={m['n_districts']} R2={m['r2']:+.3f} "
          f"MAE={m['mae']:.0f} corr={m['corr']:+.3f} bias={m['bias']:+.0f}  [{time.time()-t0:.0f}s]")
    return {"name": name, "metrics": m, "coverage": cover, "pipe": pipe}


def main():
    root = project_root()
    schema = load_schema(root)
    frame = load_frame(root, schema)

    missing = [c for c in SOIL_COLS if c not in frame.columns]
    if missing:
        sys.exit(f"Soil columns missing from kenya_maize_features_full.csv: {missing}. "
                 "Run scripts/build_features.py after scripts/merge_soil_data.py first.")

    schema_soil = Schema(
        feats=schema.feats + SOIL_COLS,
        cats=schema.cats,
        weather=schema.weather,
        non_weather=schema.non_weather + SOIL_COLS,
    )

    recipe = PlotRecipe.from_json(root / DEFAULT_RECIPE, top_configs=5)

    baseline = run_arm("baseline (no soil)", schema, frame, recipe)
    with_soil = run_arm("with soil", schema_soil, frame, recipe)

    print("\n=== comparison, 2020 holdout, district-level ===")
    print(f"{'metric':<10}{'baseline':>12}{'with soil':>12}{'delta':>12}")
    for k in ("r2", "mae", "corr", "bias"):
        b, s = baseline["metrics"][k], with_soil["metrics"][k]
        print(f"{k:<10}{b:>12.4f}{s:>12.4f}{s - b:>+12.4f}")
    print(f"{'coverage':<10}{baseline['coverage']:>12.2%}{with_soil['coverage']:>12.2%}")


if __name__ == "__main__":
    main()
