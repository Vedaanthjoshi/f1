"""Circuit-held-out forecasts issued at the end of a clean lap."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from pipeline.features import CIRCUIT_PHYSICS

EVENT = ["Season", "Track"]
STINT = EVENT + ["Driver", "Stint"]
KEY = STINT + ["LapNumber"]
BASE = ["future_age", "compound_encoded", "TrackTemp", "field", "length", "corners", "abrasion", "h"]
WEAR = ["distance", "FreshTyre", "spec"]
TREND = ["relative", "slope", "observations"]
# Each entry specifies an origin-time offset and its predictors.
MODELS = {
    "absolute": ("zero", BASE),
    "first_lap": ("first", BASE),
    "three_laps": ("reference", BASE),
    "tyre_spec": ("reference", BASE + WEAR),
    "stint_trend": ("reference", BASE + WEAR + TREND),
    "field_relative": ("field", BASE + WEAR + TREND),
}


def origins(raw, allocations):
    clean = (raw.TrackStatus.astype(str).eq("1") & raw.LapTime_s.gt(0)
             & raw.Rainfall.eq(False)
             & raw.PitInTime.isna() & raw.PitOutTime.isna()
             & raw.Compound.isin(["HARD", "MEDIUM", "SOFT"]) & raw.LapNumber.gt(1))
    d = raw.loc[clean].copy()
    if d.duplicated(KEY).any():
        raise ValueError("Duplicate driver laps")
    d["Time"] = pd.to_timedelta(d.Time)
    d = d.sort_values(EVENT + ["Time", "Driver"])
    d["pace"] = d.LapTime_s - .03 * (110 - 1.7 * d.LapNumber).clip(lower=2)
    d["field"] = d.groupby(EVENT).pace.transform(lambda x: x.rolling(20, min_periods=1).median())
    d["relative"] = d.pace - d.field
    g = d.groupby(STINT, sort=False)
    d["observations"] = g.cumcount() + 1
    d["first"] = g.pace.transform("first")
    # At the third observation freeze the three-lap median for this stint.
    rolling = g.pace.transform(lambda x: x.rolling(3).median())
    d["reference"] = rolling.where(d.observations.eq(3))
    d["reference"] = g.reference.ffill()
    # Before three observations use field pace + training-only compound offset.
    lag_lap = g.LapNumber.shift(3)
    d["slope"] = (d.relative - g.relative.shift(3)) / (d.LapNumber - lag_lap)
    for col, key in [("length","circuit_length_km"), ("corners","num_corners"), ("abrasion","abrasion_index")]:
        d[col] = d.Track.map({track: values[key] for track, values in CIRCUIT_PHYSICS.items()})
    d = d.merge(allocations, on=EVENT, how="left", validate="many_to_one")
    d["spec"] = np.select([d.Compound.eq(c) for c in ["HARD","MEDIUM","SOFT"]],
                          [d[c] for c in ["HARD","MEDIUM","SOFT"]], default=np.nan)
    d["zero"] = 0.
    return d


def examples(d):
    frames = []
    outcome = d[KEY + ["pace"]].rename(columns={"pace":"actual"})
    for horizon in (1, 2, 3, 5):
        x = d.copy()
        x["origin_lap"] = x.LapNumber
        x["LapNumber"] += horizon
        x["h"] = horizon
        x["future_age"] = x.TyreLife + horizon
        x["distance"] = x.future_age * x.length
        frames.append(x.merge(outcome, on=KEY, validate="one_to_one"))
    return pd.concat(frames, ignore_index=True)


def offset(rows, name, priors):
    if name != "reference":
        return rows[name]
    return rows.reference.fillna(rows.field + rows.Compound.map(priors).fillna(0))


def fit_forecast(train, name):
    origin_rows = train.drop_duplicates(STINT + ["origin_lap"])
    priors = origin_rows.groupby("Compound").relative.median()
    ref, features = MODELS[name]
    model = HistGradientBoostingRegressor(max_iter=65, max_leaf_nodes=15, early_stopping=False, random_state=42)
    model.fit(train[features], train.actual - offset(train, ref, priors))
    return {"name": name, "model": model, "priors": priors}


def forecast(bundle, test):
    ref, features = MODELS[bundle["name"]]
    return bundle["model"].predict(test[features]) + offset(test, ref, bundle["priors"])


def predict(train, test, name):
    return forecast(fit_forecast(train, name), test)


def next_laps(origin):
    rows = pd.concat([origin.assign(h=h) for h in (1, 2, 3, 5)], ignore_index=True)
    rows["future_age"] = rows.TyreLife + rows.h
    rows["distance"] = rows.future_age * rows.length
    return rows


def benchmark_14(train, test):
    """Evaluate the documented XGBoost point model at a common one-lap horizon."""
    from xgboost import XGBRegressor

    def features(rows):
        frame = rows[["future_age", "compound_encoded", "field", "TrackTemp", "Track", "Driver",
                      "abrasion", "length", "corners"]].copy()
        frame["hardness"] = rows.Track.map({k: v["compound_hardness"] for k, v in CIRCUIT_PHYSICS.items()})
        frame["thermal_load"] = rows.future_age * rows.TrackTemp
        frame["abrasion_load"] = rows.future_age * rows.abrasion
        frame["corner_density"] = rows.corners / rows.length
        frame["wear_cliff"] = (rows.future_age / 15) ** 2
        return frame

    x_train, x_test = features(train), features(test)
    for col in ["Driver", "Track"]:
        dtype = pd.CategoricalDtype(categories=sorted(x_train[col].unique()))
        x_train[col] = x_train[col].astype(dtype)
        x_test[col] = x_test[col].astype(dtype)
    for name, reference in [("legacy_pace", "zero"), ("legacy_delta", "first")]:
        model = XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=8,
                             min_child_weight=5, subsample=.8, colsample_bytree=.8,
                             tree_method="hist", enable_categorical=True, random_state=42, n_jobs=3)
        model.fit(x_train, train.actual - train[reference])
        yield name, model.predict(x_test) + test[reference]


def metrics(rows):
    e = rows.prediction - rows.actual
    delta = rows.actual - rows["first"]
    return pd.Series(dict(n=len(e), mae=e.abs().mean(), bias=e.mean(),
                          pace_r2=r2_score(rows.actual, rows.prediction) if len(e)>1 else np.nan,
                          delta_r2=r2_score(delta, rows.prediction-rows["first"]) if len(e)>1 else np.nan))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", default="data/historical_laps.csv")
    parser.add_argument("--output", type=Path, default=Path("forecast_results"))
    args = parser.parse_args()
    raw = pd.read_csv(args.data, low_memory=False)
    allocations = pd.read_csv(Path(__file__).resolve().parents[1]/"pipeline/compound_allocations.csv")
    d = origins(raw, allocations)
    x = examples(d)
    args.output.mkdir(parents=True, exist_ok=True)
    # Assign circuits before exclusions and horizon expansion.
    manifest = args.output / "folds.csv"
    if manifest.exists():
        folds = pd.read_csv(manifest).set_index("Track").fold.to_dict()
        if set(folds) != set(raw.Track) or set(folds.values()) != set(range(1, 6)):
            raise ValueError("Circuit roster changed. Use a new output directory for a new benchmark.")
    else:
        folds = {}
        for fold, (_, valid) in enumerate(GroupKFold(5).split(raw, groups=raw.Track), 1):
            folds.update(dict.fromkeys(raw.Track.iloc[valid].unique(), fold))
        pd.Series(folds, name="fold").rename_axis("Track").to_csv(manifest)
    results = []
    for fold in range(1, 6):
        held = x.Track.map(folds).eq(fold)
        train, test = x.loc[~held], x.loc[held]
        assert set(train.Track).isdisjoint(test.Track)
        # Select the forecast using only inner held-out circuits and three-lap totals.
        scores = {name: [] for name in MODELS}
        for a, b in GroupKFold(3).split(train, groups=train.Track):
            val = train.iloc[b]
            val = val[val.h.le(3)]
            for name in MODELS:
                errors = val.assign(error=predict(train.iloc[a], val, name)-val.actual)
                grouped = errors.groupby(STINT + ["origin_lap"]).error
                scores[name].extend(grouped.sum()[grouped.size().eq(3)].abs())
        winner = min(scores, key=lambda name: np.mean(scores[name]))
        for name in [*MODELS, "carry_forward"]:
            if name == "carry_forward":
                pred = test.pace
            else:
                bundle = fit_forecast(train, name)
                pred = forecast(bundle, test)
                if name == winner:
                    bundle["last_season"] = int(raw.Season.max())
                    bundle["held_out"] = sorted(test.Track.unique())
                    joblib.dump(bundle, args.output/f"fold_{fold}.joblib")
            rows = test[KEY + ["origin_lap","h","Compound","future_age","observations","first","actual"]].copy()
            rows = rows.assign(model=name, fold=fold, prediction=pred, selected=name==winner)
            results.append(rows)
        first_horizon = test[test.h.eq(1)]
        for name, values in benchmark_14(train[train.h.eq(1)], first_horizon):
            rows = first_horizon[KEY + ["origin_lap","h","Compound","future_age","observations","first","actual"]]
            results.append(rows.assign(model=name, fold=fold, prediction=values, selected=False))
        print(f"Fold {fold}: {winner}", flush=True)
    out = pd.concat(results, ignore_index=True)
    out.to_csv(args.output/"predictions.csv.gz", index=False)
    write_reports(out, args.output)


def write_reports(out, directory):
    out = pd.concat([out, out[out.selected].assign(model="inner_selected")])
    out["age_band"] = pd.cut(out.future_age,[0,5,15,30,np.inf])
    out["early"] = out.observations.lt(3)
    for label, keys in {"fold":["fold"], "circuit":["fold","Track"], "tyres":["Compound","age_band"], "early":["early"]}.items():
        out.groupby(["model","h",*keys], observed=True).apply(metrics, include_groups=False).to_csv(directory/f"{label}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fold_four = out[(out.fold == 4) & (out.h == 1) & (out.model == "inner_selected")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for track, group in fold_four.groupby("Track"):
        by_lap = (group.prediction-group.actual).groupby(group.LapNumber).mean()
        ax.plot(by_lap.rolling(5, min_periods=1).mean(), label=track)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Forecast lap", ylabel="Predicted minus observed pace (s)",
           title="Fold 4: five-lap rolling mean of next-lap forecast errors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/"fold_4_residuals.png", dpi=140)
    plt.close(fig)
    totals = out[out.h.le(3)].groupby(["model","fold",*STINT,"origin_lap"])
    totals = totals.agg(actual=("actual","sum"), prediction=("prediction","sum"), n=("h","count"))
    totals = totals[totals.n.eq(3)].copy()
    totals["absolute_error"] = (totals.actual-totals.prediction).abs()
    totals.groupby(["model","fold"]).absolute_error.agg(["mean","count"]).to_csv(directory/"three_lap.csv")


if __name__ == "__main__":
    main()
