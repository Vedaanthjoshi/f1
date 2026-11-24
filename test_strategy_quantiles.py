import unittest
import joblib
import numpy as np
import pandas as pd
from engine.monte_carlo import run_mc_simulation
from ml.forecast_evaluation import origins
from ml.strategy_quantiles import OUT, calibrated_prediction, scenario, sample_paths


class StrategyQuantileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=pd.read_csv("data/historical_laps.csv",nrows=900)
        allocations=pd.read_csv("pipeline/compound_allocations.csv")
        x=origins(raw,allocations)
        cls.row=x.iloc[[len(x)//2]]
        cls.r=cls.row.iloc[0]
        cls.prefix=origins(raw[pd.to_timedelta(raw.Time)<=cls.r.Time],allocations)
        fold=pd.read_csv("forecast_results/folds.csv").set_index("Track").fold[cls.r.Track]
        cls.bundle=joblib.load(OUT/f"fold_{fold}.joblib")

    def test_holdout_cutoff_and_order(self):
        self.assertIn(self.r.Track,self.bundle["held_out"])
        prefix=self.prefix[(self.prefix.Driver==self.r.Driver)&(self.prefix.LapNumber==self.r.LapNumber)]
        for compound in [None,"HARD"]:
            a=calibrated_prediction(self.bundle,scenario(self.row,np.arange(1,16),compound))
            b=calibrated_prediction(self.bundle,scenario(prefix,np.arange(1,16),compound))
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.all(np.diff(a,axis=1)>=0))

    def test_fresh_set_and_seed(self):
        rows=scenario(self.row,np.arange(1,16),"HARD",pit_at=2)
        self.assertEqual(rows.observations.iloc[0],self.r.observations)
        self.assertTrue(rows.relative.iloc[1:].isna().all())
        q=calibrated_prediction(self.bundle,rows)
        def sample():
            return sample_paths(q,self.bundle["bank"],12,np.random.default_rng(1))
        np.testing.assert_array_equal(sample(),sample())

    def test_shared_pre_stop_samples(self):
        from unittest.mock import patch
        observed=[]
        def inspect(gains,offset,horizon,cover,n,rng):
            if offset: observed.append(gains[:,:offset].copy())
            return gains.sum(axis=1)
        state=dict(Track=self.r.Track,Driver=self.r.Driver,baseline_pace=self.r["first"],
                   tyre_age=self.r.TyreLife,compound_encoded=self.r.compound_encoded)
        with patch("engine.monte_carlo.apply_rival_reaction",side_effect=inspect):
            def simulate():
                return run_mc_simulation(None,int(self.r.LapNumber),state,self.row,23,
                    quantile_bundle=self.bundle,forecast_origin=self.row,record_totals=True,n_rollouts=20)
            self.assertEqual(simulate(),simulate())
        self.assertTrue(observed)
        for gains in observed: np.testing.assert_array_equal(gains,0.)
