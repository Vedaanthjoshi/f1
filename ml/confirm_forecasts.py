"""Evaluate saved circuit-held-out forecasts on a later season without refitting."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from ml.forecast_evaluation import KEY, examples, forecast, origins, write_reports


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--models", type=Path, default=Path("forecast_results"))
    parser.add_argument("--output", type=Path, default=Path("confirmation_results"))
    args = parser.parse_args()
    raw = pd.read_csv(args.data, low_memory=False)
    allocations = pd.read_csv(Path(__file__).resolve().parents[1]/"pipeline/compound_allocations.csv")
    x = examples(origins(raw, allocations))
    predictions = []
    folds = pd.read_csv(args.models/"folds.csv").set_index("Track").fold
    if not set(raw.Track).issubset(folds.index):
        raise ValueError("New circuits need an explicitly assigned frozen model.")
    for fold, rows in x.groupby(x.Track.map(folds)):
        bundle = joblib.load(args.models/f"fold_{fold}.joblib")
        if raw.Season.min() <= bundle["last_season"] or not set(rows.Track).issubset(bundle["held_out"]):
            raise ValueError("Confirmation requires later seasons and circuit-held-out models.")
        for name, values in [(bundle["name"], forecast(bundle, rows)), ("carry_forward", rows.pace)]:
            pred = rows[KEY + ["origin_lap","h","Compound","future_age","observations","first","actual"]].copy()
            predictions.append(pred.assign(model=name, fold=fold, prediction=values, selected=name==bundle["name"]))
    out = pd.concat(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output/"predictions.csv.gz", index=False)
    write_reports(out, args.output)


if __name__ == "__main__":
    main()
