"""Evaluate XGBoost strategy quantiles using frozen outer circuit folds."""
from pathlib import Path
from itertools import product
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from ml.forecast_evaluation import KEY, STINT, BASE, WEAR, TREND, origins
from pipeline.features import CIRCUIT_PHYSICS

Q = np.array([.05, .5, .95])
FEATURES = BASE + WEAR + TREND + ["pit"]
OUT = Path("quantile_results_v1")


def scenario(row, horizons, compound=None, pit_at=1):
    x = (row.copy() if np.isscalar(horizons) else pd.concat([row] * len(horizons), ignore_index=True))
    x["h"] = horizons
    x["pit"] = False if compound is None else x.h.ge(pit_at)
    x["future_age"] = np.where(x.pit, np.maximum(1,x.h-pit_at), x.TyreLife+x.h)
    x["distance"] = x.future_age*x.length
    if compound is not None:
        mask = x.pit
        x.loc[mask, ["relative","slope"]] = np.nan
        x.loc[mask, "observations"] = 0
        x.loc[mask, "FreshTyre"] = True
        x.loc[mask, "spec"] = x.loc[mask,compound]
        x.loc[mask, "compound_encoded"] = {"HARD":3,"MEDIUM":2,"SOFT":1}[compound]
    return x


def cached_timing(season, track):
    folder = Path("f1_cache") / str(int(season))
    name = track.replace(" ", "_")
    paths = list(folder.glob(f"*_{name}/*_Race/_extended_timing_data.ff1pkl"))
    if len(paths) != 1:
        return pd.DataFrame()
    return pd.read_pickle(paths[0])["data"][0]


def pit_records(d):
    events = []
    for (season,track),group in d.groupby(["Season","Track"]):
        timing = cached_timing(season,track)
        if timing.empty:
            continue
        entry = timing[timing.PitInTime.notna()].copy()
        entry["out_lap"] = entry.NumberOfLaps+1
        exit_laps = timing.loc[timing.PitOutTime.notna(),["Driver","NumberOfLaps","PitOutTime"]]
        paired = entry.drop(columns="PitOutTime").merge(exit_laps,left_on=["Driver","out_lap"],
                                                        right_on=["Driver","NumberOfLaps"],suffixes=("","_out"))
        drivers = group.drop_duplicates("DriverNumber").set_index("DriverNumber").Driver
        paired["Driver"] = pd.to_numeric(paired.Driver).map(drivers)
        paired["LapNumber"] = paired.NumberOfLaps-1
        paired["duration"] = (paired.PitOutTime-paired.PitInTime).dt.total_seconds()
        events.append(paired.assign(Season=season,Track=track))
    if not events:
        raise ValueError("Pit-entry and pit-exit timing records are missing for this dataset.")
    return pd.concat(events,ignore_index=True)


def prepare(d):
    labels = d[KEY+["pace","Compound","FreshTyre"]].rename(columns={
        "pace":"actual","Stint":"target_stint","Compound":"target_compound","FreshTyre":"target_fresh"})
    events = pit_records(d)
    join = ["Season","Track","Driver","LapNumber"]
    d = d.merge(events[join].drop_duplicates().assign(stop_next=True),on=join,how="left",validate="one_to_one")
    frames = []
    for h in range(1,16):
        x = d.assign(origin=d.LapNumber, LapNumber=d.LapNumber+h)
        x = x.merge(labels,on=[k for k in KEY if k!="Stint"],validate="one_to_one")
        stay = x.target_stint.eq(x.Stint)
        # Replay a stop after the final clean origin with the observed compound
        # as the proposed action. Exclude used replacements of unknown starting age.
        pit = x.target_stint.eq(x.Stint+1) & x.stop_next.eq(True) & x.target_fresh.eq(True)
        for compound, part in [(None,x[stay]), *list(x[pit].groupby("target_compound"))]:
            if len(part):
                frames.append(scenario(part,h,compound))
    return pd.concat(frames,ignore_index=True)


def calibrated_prediction(bundle, rows):
    raw = bundle["model"].predict(rows[FEATURES])+rows.field.to_numpy()[:,None]
    return adjust(ordered(raw), rows.h, bundle["corrections"])


def band(h):
    return np.select([h==1,h<=5],[0,1],default=2)


def ordered(pred):
    return np.column_stack((np.minimum(pred[:,0],pred[:,1]),pred[:,1],np.maximum(pred[:,2],pred[:,1])))


def adjust(pred,h,corrections):
    result = pred.copy()
    for group,(lower,upper) in enumerate(corrections):
        mask = band(h)==group
        result[mask,0] += lower
        result[mask,2] += upper
    return ordered(result)


def loss(y,p):
    e = np.asarray(y)[:,None]-p
    return np.maximum(Q*e,(Q-1)*e)


def reg(depth,weight,rate,trees,early=False):
    return XGBRegressor(objective="reg:quantileerror",quantile_alpha=Q,
        tree_method="hist",max_depth=depth,min_child_weight=weight,learning_rate=rate,
        n_estimators=trees,max_bin=64,n_jobs=3,random_state=42,
        early_stopping_rounds=8 if early else None)


def legacy_features(x, categories):
    f=x[["future_age","compound_encoded","TrackTemp","Track","Driver","abrasion","length","corners"]].copy()
    f["evolution"]=x.track_evolution
    f["hardness"]=x.Track.map({k:v["compound_hardness"] for k,v in CIRCUIT_PHYSICS.items()})
    f["thermal"]=x.future_age*x.TrackTemp
    f["surface"]=x.future_age*x.abrasion
    f["density"]=x.corners/x.length
    f["cliff"]=(x.future_age/15)**2
    for col,levels in categories.items():
        f[col]=pd.Categorical(f[col],categories=levels)
    return f


def score(frame):
    y=frame.actual.to_numpy()
    p=frame[["lo","mid","hi"]].to_numpy()
    lo,mid,hi=p.T
    return pd.Series(dict(n=len(y),coverage=np.mean((lo<=y)&(y<=hi)),
        width=np.mean(hi-lo),below=np.mean(y<lo),above=np.mean(y>hi),
        mae=np.mean(abs(y-mid)),interval_score=np.mean(hi-lo+20*(np.maximum(lo-y,0)+np.maximum(y-hi,0))),
        **dict(zip(["loss05","loss50","loss95"],loss(y,p).mean(axis=0)))))


def sample_paths(p,bank,n,rng):
    z=bank[rng.integers(len(bank),size=n),:len(p)]
    return p[:,1]+np.where(z<0,z*(p[:,1]-p[:,0]),z*(p[:,2]-p[:,1]))


def run():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_csv("data/historical_laps.csv",low_memory=False)
    alloc=pd.read_csv("pipeline/compound_allocations.csv")
    d=origins(raw,alloc)
    # Only past completed lap times contribute to the historical model's track input.
    d["track_evolution"]=d.groupby(["Season","Track"]).LapTime_s.transform(lambda s:s.expanding().median())
    x=prepare(d)
    folds=pd.read_csv("forecast_results/folds.csv").set_index("Track").fold
    x["fold"]=x.Track.map(folds)
    if x.fold.isna().any(): raise ValueError("Missing circuit fold")
    records=[]
    cumulative=[]
    for fold in range(1,6):
        train=x[x.fold!=fold]
        test=x[x.fold==fold]
        splits=list(GroupKFold(3).split(train,groups=train.Track))
        # Keep tuning, early stopping and calibration circuits disjoint.
        blocks=[train.iloc[b] for _,b in splits]
        fitting,stopping,cal=blocks
        circuit_sets = [set(rows.Track) for rows in [fitting, stopping, cal, test]]
        if any(left & right for i, left in enumerate(circuit_sets) for right in circuit_sets[i+1:]):
            raise ValueError("Circuit overlap between training, tuning, calibration or validation")
        def sample(rows,n):
            return rows.sample(min(len(rows),n),random_state=42)
        fit=sample(fitting,12000)
        stop=sample(stopping,6000)
        candidates=[]
        for dep,w,lr in product([3,5],[5,20],[.03,.07]):
            m=reg(dep,w,lr,100,True)
            m.fit(fit[FEATURES],fit.actual-fit.field,
                eval_set=[(stop[FEATURES],stop.actual-stop.field)],verbose=False)
            candidates.append((loss(stop.actual,m.predict(stop[FEATURES])+stop.field.to_numpy()[:,None]).mean(),
                               dep,w,lr,m.best_iteration+1))
        _,dep,w,lr,trees=min(candidates)
        fit=pd.concat([fitting,stopping])
        m=reg(dep,w,lr,trees)
        m.fit(fit[FEATURES],fit.actual-fit.field)
        def pred(rows):
            return m.predict(rows[FEATURES])+rows.field.to_numpy()[:,None]
        c=ordered(pred(cal))
        err=cal.actual.to_numpy()[:,None]-c
        corrections=[]
        for b in range(3):
            subset=err[band(cal.h)==b]
            if len(subset)<100: subset=err
            corrections.append((np.quantile(subset[:,0],.05),np.quantile(subset[:,2],.95)))
        c=adjust(c,cal.h,corrections)
        residual=cal.actual.to_numpy()-c[:,1]
        z=residual/np.maximum(.001,np.where(residual<0,c[:,1]-c[:,0],c[:,2]-c[:,1]))
        bank=cal.assign(z=z).query("not pit").pivot(index=STINT+["origin"],columns="h",values="z").dropna().to_numpy()
        if bank.shape[1]!=15 or not len(bank): raise ValueError("No complete calibration trajectories")
        bundle=dict(model=m,corrections=corrections,bank=bank.astype("float32"),held_out=sorted(test.Track.unique()),version=1)
        joblib.dump(bundle,OUT/f"fold_{fold}.joblib",compress=3)
        base=d[~d.Track.isin(test.Track.unique())].copy()
        base["future_age"]=base.TyreLife
        cats={col:sorted(base[col].unique()) for col in ["Track","Driver"]}
        legacy=reg(8,5,.05,400).set_params(enable_categorical=True,max_bin=256,subsample=.8,colsample_bytree=.8)
        legacy.fit(legacy_features(base,cats),base.pace-base["first"])
        joblib.dump(dict(model=legacy,categories=cats),OUT/f"baseline_{fold}.joblib",compress=3)
        old=legacy.predict(legacy_features(test,cats))+test["first"].to_numpy()[:,None]
        raw_pred=pred(test)
        for name,p in [("legacy",old),("challenger",raw_pred),("calibrated",adjust(ordered(raw_pred),test.h,corrections))]:
            r=test[STINT+["origin","h","future_age","target_compound","pit","actual","fold"]].copy()
            r["model"]=name
            r["crossed"]=(p[:,0]>p[:,1])|(p[:,1]>p[:,2])
            r[["lo","mid","hi"]]=ordered(p)
            records.append(r)
            if name=="challenger": continue
            rng=np.random.default_rng(42)
            for horizon in [3,5,15]:
                # Require all observed clean laps for cumulative ground truth.
                wide=r[r.h<=horizon].pivot(index=STINT+["origin","pit"],columns="h",values=["actual","lo","mid","hi"]).dropna()
                if wide.empty: continue
                total=wide.actual.sum(axis=1).to_numpy()
                for start in range(0,len(wide),128):
                    section=wide.iloc[start:start+128]
                    mid=section.mid.to_numpy()[:,None,:]
                    lo=section.lo.to_numpy()[:,None,:]
                    hi=section.hi.to_numpy()[:,None,:]
                    if name=="legacy":
                        samples=rng.normal(mid,(hi-lo)/3.29,size=(len(section),300,horizon))
                    else:
                        z=bank[rng.integers(len(bank),size=(len(section),300)),:horizon]
                        samples=mid+np.where(z<0,z*(mid-lo),z*(hi-mid))
                    bounds=np.quantile(samples.sum(axis=2),Q,axis=1).T
                    row=pd.DataFrame(bounds,columns=["lo","mid","hi"])
                    cumulative.append(row.assign(actual=total[start:start+len(section)],model=name,fold=fold,h=horizon))
        print(f"Finished fold {fold}",flush=True)
    results=pd.concat(records,ignore_index=True)
    results["age"]=pd.cut(results.future_age,[0,5,15,30,np.inf])
    results.to_csv(OUT/"predictions.csv.gz",index=False)
    for label,keys in {"summary":[],"fold":["fold"],"regime":["pit","h"],"circuit":["Track"],"tyre":["age","target_compound"]}.items():
        results.groupby(["model",*keys],observed=True).apply(score,include_groups=False).to_csv(OUT/f"{label}.csv")
    results.groupby("model").crossed.mean().to_csv(OUT/"crossings.csv")
    pd.concat(cumulative).groupby(["model","fold","h"]).apply(score,include_groups=False).to_csv(OUT/"cumulative.csv")


if __name__=="__main__":
    run()

