"""Regression checks for the timestamp boundary used by both forecast paths."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ml.forecast_evaluation import forecast, next_laps, origins
from engine.monte_carlo import run_mc_simulation


class ForecastCutoffTest(unittest.TestCase):
    def test_remove_future_laps(self):
        raw = pd.read_csv("data/historical_laps.csv", nrows=1200)
        race = raw[raw.Track.eq(raw.Track.iloc[0])]
        cutoff = pd.to_timedelta(race.Time).median()
        allocations = pd.read_csv("pipeline/compound_allocations.csv")
        full = origins(race, allocations)
        prefix = origins(race[pd.to_timedelta(race.Time) <= cutoff], allocations)
        assert_frame_equal(full[full.Time <= cutoff].reset_index(drop=True), prefix)

    def test_frozen_model_prediction_is_prefix_invariant(self):
        from pathlib import Path
        import joblib

        root = Path("forecast_results")
        if not (root / "fold_5.joblib").exists():
            self.skipTest("Run circuit evaluation to generate the held-out models")
        raw = pd.read_csv("data/historical_laps.csv", nrows=1200)
        race = raw[raw.Track.eq(raw.Track.iloc[0])]
        folds = pd.read_csv(root / "folds.csv").set_index("Track").fold
        bundle = joblib.load(root / f"fold_{folds[race.Track.iloc[0]]}.joblib")
        self.assertIn(race.Track.iloc[0], bundle["held_out"])
        allocations = pd.read_csv("pipeline/compound_allocations.csv")
        full = origins(race, allocations)
        origin = full.iloc[[len(full)//2]]
        truncated = origins(race[pd.to_timedelta(race.Time) <= origin.Time.iloc[0]], allocations)
        same = truncated[(truncated.Driver == origin.Driver.iloc[0]) &
                         (truncated.LapNumber == origin.LapNumber.iloc[0])]
        np.testing.assert_array_equal(forecast(bundle, next_laps(origin)), forecast(bundle, next_laps(same)))

    def test_both_strategies_ignore_future_environment(self):
        class Predictor:
            def predict(self, frame):
                return np.asarray(frame.TrackTemp + frame.track_evolution + frame.tyre_age)

        env = pd.DataFrame(dict(LapNumber=[8,9], TrackTemp=[25,70], track_evolution=[90,230]))
        state = dict(tyre_age=5, compound_encoded=1, baseline_pace=90)
        def simulate(data):
            return run_mc_simulation(Predictor(),8,state,data,23,n_rollouts=20)
        self.assertEqual(simulate(env.iloc[:1]), simulate(env))
