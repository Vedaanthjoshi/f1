"""Replay the dashboard simulator at recorded stops on circuit-held-out models."""
import joblib
import numpy as np
import pandas as pd
from engine.monte_carlo import run_mc_simulation
from ml.forecast_evaluation import origins
from ml.strategy_quantiles import OUT, cached_timing, pit_records, score

NAMES = dict(tyre_age="future_age", compound_encoded="compound_encoded", TrackTemp="TrackTemp",
             Track="Track", Driver="Driver", abrasion_index="abrasion", circuit_length_km="length",
             num_corners="corners", track_evolution="evolution", compound_hardness="hardness",
             thermal_load="thermal", abrasion_load="surface", corner_density="density", wear_cliff="cliff")


class ReplayBaseline:
    def __init__(self,bundle):
        self.bundle=bundle

    def predict(self,frame):
        x=frame.rename(columns=NAMES).copy()
        for name,levels in self.bundle["categories"].items():
            x[name]=pd.Categorical(x[name].astype(str),categories=levels)
        model=self.bundle["model"]
        return model.predict(x[model.get_booster().feature_names])


def run():
    d=origins(pd.read_csv("data/historical_laps.csv",low_memory=False),pd.read_csv("pipeline/compound_allocations.csv"))
    stops=pit_records(d)
    folds=pd.read_csv("forecast_results/folds.csv").set_index("Track").fold
    records=[]
    transitions=[]
    for (season,track),race in d.groupby(["Season","Track"]):
        timing=cached_timing(season,track)
        if timing.empty: continue
        events=stops[(stops.Track==track)&(stops.Season==season)]
        fold=folds[track]
        challenger=joblib.load(OUT/f"fold_{fold}.joblib")
        assert track in challenger["held_out"]
        legacy=ReplayBaseline(joblib.load(OUT/f"baseline_{fold}.joblib"))
        # Score pit transit itself for every origin with an observed entry/exit pair.
        paired=race.merge(events[["Driver","LapNumber","duration"]],on=["Driver","LapNumber"])
        for r in paired.itertuples():
            prior=events.loc[events.PitOutTime<=r.Time,"duration"]
            estimate=prior.median() if len(prior) else 24.5
            transitions.append(dict(Season=season,Track=track,Driver=r.Driver,lap=r.LapNumber,
                                    estimate=estimate,actual=r.duration))
        # Fixed sample per race for the expensive two-engine simulation replay.
        for pit in [False,True]:
            pool=paired if pit else race
            if pool.empty: continue
            for idx in pool.sample(min(4,len(pool)),random_state=42).index:
                row=pool.loc[[idx]]
                r=row.iloc[0]
                lap=int(r.LapNumber)
                future=race[(race.Driver==r.Driver)&race.LapNumber.gt(lap)].set_index("LapNumber")
                target=r.Stint+int(pit)
                clean=future[future.Stint==target]
                if clean.empty or (pit and not clean.iloc[0].FreshTyre): continue
                comp=int(clean.iloc[0].compound_encoded)
                driver_timing=timing[timing.Driver.astype(str)==str(int(r.DriverNumber))].set_index("NumberOfLaps")
                horizons=[]
                for h in [3,5,15]:
                    required=range(lap+(3 if pit else 1),lap+h+1)
                    if all(i in clean.index for i in required):
                        times=driver_timing.LapTime.reindex(range(lap+1,lap+h+1))
                        if times.notna().all(): horizons.append((h,times.dt.total_seconds().sum()))
                if not horizons: continue
                state=dict(Track=track,Driver=r.Driver,tyre_age=r.TyreLife,compound_encoded=r.compound_encoded,
                           baseline_pace=r["first"])
                hist=race[race.Time<=r.Time]
                env=row.assign(track_evolution=hist.LapTime_s.median())
                prior=events.loc[events.PitOutTime<=r.Time,"duration"]
                for name,bundle in [("legacy",None),("calibrated",challenger)]:
                    sim=run_mc_simulation(legacy,lap,state,env,prior.median() if len(prior) else 24.5,
                        race_name=track,feature_cols=list(NAMES),quantile_bundle=bundle,forecast_origin=row,
                        record_totals=True,n_rollouts=500)
                    action=next(a for a in sim if (a["pit_offset"]==0 and a["compound_encoded"]==comp)) if pit else sim[0]
                    for h,actual in horizons:
                        lo,mid,hi=action["total_intervals"][h]
                        records.append(dict(Season=season,Track=track,Driver=r.Driver,origin=lap,h=h,pit=pit,
                                            model=name,lo=lo,mid=mid,hi=hi,actual=actual))
    rows=pd.DataFrame(records)
    rows.to_csv(OUT/"strategy_replay.csv",index=False)
    rows.groupby(["model","pit","h"]).apply(score,include_groups=False).to_csv(OUT/"strategy_replay_summary.csv")
    pd.DataFrame(transitions).to_csv(OUT/"pit_transit.csv",index=False)


if __name__=="__main__":
    run()
