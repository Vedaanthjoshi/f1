"""Frozen later-season check on a named race, with no training or calibration."""
import argparse
import fastf1
import joblib
import pandas as pd
from pipeline.ingestion import clean_lap_data, merge_weather_data
from pipeline.features import apply_feature_engineering
from ml.forecast_evaluation import origins
from ml.strategy_quantiles import OUT, calibrated_prediction, prepare, score


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("season",type=int)
    parser.add_argument("track")
    parser.add_argument("--allocation",type=int,nargs=3,metavar=("HARD","MEDIUM","SOFT"),required=True)
    args=parser.parse_args()
    folds=pd.read_csv("forecast_results/folds.csv").set_index("Track").fold
    bundle=joblib.load(OUT/f"fold_{folds[args.track]}.joblib")
    training=pd.read_csv("data/historical_laps.csv",usecols=["Season"])
    if args.season<=training.Season.max() or args.track not in bundle["held_out"]:
        raise ValueError("Expected a later season and a circuit excluded during fitting.")
    session=fastf1.get_session(args.season,args.track,"R")
    session.load(telemetry=False,messages=False)
    raw=apply_feature_engineering(merge_weather_data(clean_lap_data(session.laps),session.weather_data),args.track)
    raw=raw.assign(Season=args.season,Track=args.track)
    allocations=pd.DataFrame([[args.season,args.track,*args.allocation]],columns=["Season","Track","HARD","MEDIUM","SOFT"])
    rows=prepare(origins(raw,allocations))
    rows[["lo","mid","hi"]]=calibrated_prediction(bundle,rows)
    summary=rows.groupby(["pit","h"]).apply(score,include_groups=False)
    summary.to_csv(OUT/f"confirmation_{args.season}_{args.track}.csv")
    print(score(rows))


if __name__=="__main__":
    main()
