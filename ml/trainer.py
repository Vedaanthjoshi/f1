"""
ml/trainer.py — Offline Global Quantile XGBoost Training Pipeline
=================================================================
Trains the multi-output quantile regression model (p05, p50, p95) on the
73,309 historical laps dataset using asymmetric pinball loss and the
14-feature physics-informed suite.

Reviewer-concern fixes applied
--------------------------------
Fix 1 : Fuel mass clamped >= 2 kg so corrections never go negative.
Fix 2 : track_evolution uses strictly causal expanding median with lag-1.
Fix 3 : baseline_pace completely excluded from features; target is lap_time_delta
        computed by excluding Lap 1 standing starts.
Fix 4 : Full 5-fold race-wise GroupKFold (grouped by Track) cross-validation
        with mean +/- std reported across all folds.
Fix 6 : 14-feature physics-informed suite with non-linear wear cliff, thermal load,
        abrasion load, corner density, and physical Pirelli C-hardness.
"""

from __future__ import annotations
import os
from typing import Any
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pipeline.features import FEATURE_COLS_14, apply_feature_engineering
from pipeline.baseline import prepare_ml_data


PHYSICS_FEATURE_COLS: list[str] = [
    'age_m1', 'age_sqrt', 'age_log', 'age_sq', 'age_h8',
    'age_x_cliff_soft', 'age_x_cliff_med', 'age_x_cliff_hard',
    'compound_hardness', 'is_soft', 'is_hard', 'TrackTemp', 'fuel_c',
    'abrasion_index', 'corner_density',
    'age_x_hard', 'age_x_temp', 'age_x_abr', 'age_x_fuel', 'age_x_soft', 'age_x_hardc',
]

# Maps each engineered physics feature back to its primary parent slot in FEATURE_COLS_14.
# Track-identifying columns ('track_evolution', 'circuit_length_km', 'num_corners') stay 0.0.
_PHYS_TO_BASE14: dict[str, str] = {
    'age_m1': 'tyre_age',
    'age_sqrt': 'tyre_age',
    'age_log': 'tyre_age',
    'age_sq': 'wear_cliff',
    'age_h8': 'wear_cliff',
    'age_x_cliff_soft': 'wear_cliff',
    'age_x_cliff_med': 'wear_cliff',
    'age_x_cliff_hard': 'wear_cliff',
    'compound_hardness': 'compound_hardness',
    'is_soft': 'compound_encoded',
    'is_hard': 'compound_encoded',
    'TrackTemp': 'TrackTemp',
    'fuel_c': 'fuel_mass',
    'abrasion_index': 'abrasion_index',
    'corner_density': 'corner_density',
    'age_x_hard': 'compound_age_interaction',
    'age_x_temp': 'thermal_load',
    'age_x_abr': 'abrasion_load',
    'age_x_fuel': 'fuel_mass',
    'age_x_soft': 'compound_age_interaction',
    'age_x_hardc': 'compound_age_interaction',
}


def _extract_physics_features(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Convert 14-column feature batch into smooth non-identifying tyre-wear features."""
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=FEATURE_COLS_14)
    else:
        X_df = X

    def _col(name: str, default: float) -> np.ndarray:
        if name in X_df.columns:
            return pd.to_numeric(X_df[name], errors='coerce').fillna(default).to_numpy(dtype=float)
        return np.full(len(X_df), default, dtype=float)

    age_m1 = np.maximum(_col('tyre_age', 1.0) - 1.0, 0.0)
    comp_enc = _col('compound_encoded', 2.0)
    comp_hard = _col('compound_hardness', 3.0)
    temp = _col('TrackTemp', 30.0)
    fuel = _col('fuel_mass', 55.0)
    abr = _col('abrasion_index', 3.0)
    cd = _col('corner_density', 3.0)

    is_soft = (comp_enc == 1.0).astype(float)
    is_med = (comp_enc == 2.0).astype(float)
    is_hard = (comp_enc == 3.0).astype(float)
    temp_ex = temp - 30.0
    hard_c = comp_hard - 3.0
    abr_c = abr - 3.0
    fuel_c = (fuel - 55.0) / 50.0

    return pd.DataFrame({
        'age_m1': age_m1,
        'age_sqrt': np.sqrt(age_m1),
        'age_log': np.log1p(age_m1),
        'age_sq': age_m1 ** 2,
        'age_h8': np.maximum(age_m1 - 8.0, 0.0),
        'age_x_cliff_soft': np.maximum(age_m1 - 13.0, 0.0) * is_soft,
        'age_x_cliff_med': np.maximum(age_m1 - 18.0, 0.0) * is_med,
        'age_x_cliff_hard': np.maximum(age_m1 - 23.0, 0.0) * is_hard,
        'compound_hardness': comp_hard,
        'is_soft': is_soft,
        'is_hard': is_hard,
        'TrackTemp': temp,
        'fuel_c': fuel_c,
        'abrasion_index': abr,
        'corner_density': cd,
        'age_x_hard': age_m1 * hard_c,
        'age_x_temp': age_m1 * temp_ex,
        'age_x_abr': age_m1 * abr_c,
        'age_x_fuel': age_m1 * fuel_c,
        'age_x_soft': age_m1 * is_soft,
        'age_x_hardc': age_m1 * is_hard,
    })


class PhysicsDegradationEnsemble:
    """
    Non-memorizing tyre degradation predictor blending regularized Ridge + Monotone XGBoost
    for center pace delta and Quantile XGBoost for uncertainty bounds [p05, p50, p95].
    """

    def __init__(
        self,
        ridge_weight: float = 0.50,
        seed: int = 42,
        single_output: bool = False,
    ) -> None:
        self.ridge_weight = float(ridge_weight)
        self.seed = int(seed)
        self.single_output = bool(single_output)
        self.ridge_center = make_pipeline(StandardScaler(), Ridge(alpha=600.0))
        self.xgb_center = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=40.0,
            reg_alpha=5.0,
            monotone_constraints={
                'age_m1': 1,
                'age_sqrt': 1,
                'age_log': 1,
                'age_sq': 1,
                'age_x_cliff_soft': 1,
                'age_x_cliff_med': 1,
                'age_x_cliff_hard': 1,
            },
            objective='reg:squarederror',
            tree_method='hist',
            random_state=self.seed,
            n_jobs=-1,
        )
        self.xgb_quantiles = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=60,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=25.0,
            reg_alpha=5.0,
            monotone_constraints={
                'age_m1': 1,
                'age_sqrt': 1,
                'age_log': 1,
            },
            objective='reg:quantileerror',
            quantile_alpha=[0.05, 0.95],
            tree_method='hist',
            random_state=self.seed,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> "PhysicsDegradationEnsemble":
        Xp = _extract_physics_features(X)
        y_arr = np.asarray(y, dtype=float)
        self.ridge_center.fit(Xp, y_arr)
        self.xgb_center.fit(Xp, y_arr, verbose=False)
        if not self.single_output:
            self.xgb_quantiles.fit(Xp, y_arr, verbose=False)
        return self

    def predict_center(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xp = _extract_physics_features(X)
        ridge_p = np.clip(self.ridge_center.predict(Xp), -0.1, 4.0)
        xgb_p = np.clip(self.xgb_center.predict(Xp), -0.1, 4.0)
        return np.clip(self.ridge_weight * ridge_p + (1.0 - self.ridge_weight) * xgb_p, -0.1, 4.0)

    def predict(self, X: pd.DataFrame | np.ndarray, validate_features: bool = False) -> np.ndarray:
        del validate_features
        p50 = self.predict_center(X)
        if self.single_output:
            return p50
        Xp = _extract_physics_features(X)
        qp = self.xgb_quantiles.predict(Xp)
        p05 = np.minimum(qp[:, 0], p50)
        p95 = np.maximum(qp[:, 1], p50)
        return np.column_stack([p05, p50, p95])

    @property
    def feature_importances_(self) -> np.ndarray:
        """Project physics tree importances back onto FEATURE_COLS_14 for UI/reporting."""
        mapped = {feat: 0.0 for feat in FEATURE_COLS_14}
        raw_imp = getattr(self.xgb_center, 'feature_importances_', None)
        if raw_imp is not None and len(raw_imp) == len(PHYSICS_FEATURE_COLS):
            for p_col, imp_val in zip(PHYSICS_FEATURE_COLS, raw_imp):
                parent = _PHYS_TO_BASE14.get(p_col)
                if parent in mapped:
                    mapped[parent] += float(imp_val)
        arr = np.array([mapped[f] for f in FEATURE_COLS_14], dtype=float)
        total = arr.sum()
        return arr / total if total > 0.0 else arr


def estimate_causal_race_eb(
    past_df: pd.DataFrame,
    base_model: Any,
    feature_cols: list[str] | None = None,
    prior_weight_alpha: float = 12.0,
    prior_weight_bias: float = 25.0,
    outlier_cap_s: float = 3.5,
) -> tuple[float, float]:
    """
    Strictly causal Empirical Bayes race adaptation from already-completed clean dry laps
    (`LapNumber >= 2`, `tyre_age >= 2`, `lap_time_delta < outlier_cap_s`, dry).
    Returns shrunk `(race_alpha, race_bias)`.
    """
    if past_df is None or len(past_df) == 0 or base_model is None:
        return 1.0, 0.0

    cols = feature_cols or FEATURE_COLS_14
    if any(c not in past_df.columns for c in cols) or 'lap_time_delta' not in past_df.columns:
        return 1.0, 0.0

    mask = (
        past_df['lap_time_delta'].notna()
        & (past_df['lap_time_delta'] < outlier_cap_s)
        & (past_df['tyre_age'] >= 3.0)
    )
    if 'LapNumber' in past_df.columns:
        mask = mask & (past_df['LapNumber'] >= 2)
    if 'Rainfall' in past_df.columns:
        rf = pd.to_numeric(past_df['Rainfall'], errors='coerce').fillna(0.0)
        mask = mask & (rf == 0.0)

    obs = past_df.loc[mask].dropna(subset=cols + ['lap_time_delta'])
    if len(obs) == 0:
        return 1.0, 0.0

    raw_preds = base_model.predict(obs[cols])
    base_p50 = raw_preds[:, 1] if getattr(raw_preds, 'ndim', 1) == 2 else np.asarray(raw_preds, dtype=float)
    y_obs = obs['lap_time_delta'].to_numpy(dtype=float)

    num = float(np.sum(base_p50 * y_obs))
    den = float(np.sum(base_p50 ** 2))
    res = float(np.sum(y_obs - base_p50))
    cnt = float(len(obs))

    race_alpha = float(np.clip((num + prior_weight_alpha) / (den + prior_weight_alpha), 0.15, 2.8))
    race_bias = float(res / (cnt + prior_weight_bias))
    return race_alpha, race_bias


def apply_race_eb(
    base_pred: np.ndarray | float,
    race_alpha: np.ndarray | float = 1.0,
    race_bias: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Apply Empirical Bayes race scaling & shrunk residual bias consistently."""
    return np.clip(
        np.asarray(base_pred, dtype=float) * np.asarray(race_alpha, dtype=float)
        + 0.5 * np.asarray(race_bias, dtype=float),
        -0.2,
        6.0,
    )


# Monotonic constraints: tyre degradation cannot decrease as wear/thermal/abrasion load increases
MONOTONE_CONSTRAINTS = {
    'tyre_age': 1,
    'thermal_load': 1,
    'abrasion_load': 1,
    'wear_cliff': 1,
}


def _build_xgb_params(quantile_alphas: list[float], seed: int = 42) -> dict:
    """Physics-regularized XGBoost hyperparameters that learn smooth wear curves instead of memorizing."""
    return {
        'n_estimators': 250,
        'max_depth': 4,
        'learning_rate': 0.04,
        'min_child_weight': 25,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 10.0,
        'reg_alpha': 2.0,
        'monotone_constraints': MONOTONE_CONSTRAINTS,
        'tree_method': 'hist',
        'objective': 'reg:quantileerror',
        'quantile_alpha': quantile_alphas,
        'random_state': seed,
    }


def train_global_model(
    data_path: str = os.path.join("data", "historical_laps.csv"),
    output_path: str = os.path.join("models", "global_degradation_model.joblib"),
    quantile_alphas: list[float] | None = None,
) -> PhysicsDegradationEnsemble | None:
    """Train the global physics-regularized tyre degradation ensemble (Ridge + Monotone XGB + Quantile Bounds)."""
    if quantile_alphas is None:
        quantile_alphas = [0.05, 0.50, 0.95]

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run data_collector.py first.")
        return None

    print(f"Loading global dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure all 14 features and clean baseline exist
    missing_feats = [f for f in FEATURE_COLS_14 if f not in df.columns]
    if missing_feats or 'lap_time_delta' not in df.columns:
        print(f"Enriching dataset with 14 features (missing: {missing_feats})...")
        df = apply_feature_engineering(df)
        df = prepare_ml_data(df)

    target = 'lap_time_delta'
    features = FEATURE_COLS_14

    clean_df = df.dropna(subset=features + [target, 'Track']).copy()
    if 'LapNumber' in clean_df.columns:
        n_before = len(clean_df)
        clean_df = clean_df[clean_df['LapNumber'] >= 2].copy()
        print(f"Excluded {n_before - len(clean_df):,} standing-start Lap 1 rows.")
    clean_df = clean_df.reset_index(drop=True)

    rainfall = pd.to_numeric(clean_df['Rainfall'], errors='coerce').fillna(0.0) if 'Rainfall' in clean_df.columns else pd.Series(0.0, index=clean_df.index)
    is_green = (clean_df['TrackStatus'].astype(str) == '1') if 'TrackStatus' in clean_df.columns else pd.Series(True, index=clean_df.index)
    not_pit_in = clean_df['PitInTime'].isna() if 'PitInTime' in clean_df.columns else pd.Series(True, index=clean_df.index)
    not_pit_out = clean_df['PitOutTime'].isna() if 'PitOutTime' in clean_df.columns else pd.Series(True, index=clean_df.index)
    clean_dry = ((clean_df[target] < 3.5) & (rainfall == 0.0) & is_green & not_pit_in & not_pit_out).to_numpy()
    print(f"Total Lap>=2 rows: {len(clean_df):,} | Clean-Dry training rows (<3.5s): {int(clean_dry.sum()):,}")

    X = clean_df[features]
    y = clean_df[target]
    y_all = y.to_numpy(dtype=float)
    groups = clean_df['Track'].astype(str).values

    # ── Fix 4: Full 5-Fold Race-wise GroupKFold Cross-Validation ──────────────
    print("\n" + "=" * 65)
    print("RUNNING 5-FOLD RACE-WISE GROUP-KFOLD CROSS-VALIDATION")
    print("=" * 65)
    gkf = GroupKFold(n_splits=5)

    cv_p50_mae: list[float] = []
    cv_p50_rmse: list[float] = []
    cv_p50_r2_clean: list[float] = []
    cv_coverage: list[float] = []
    cv_pinball_05: list[float] = []
    cv_pinball_50: list[float] = []
    cv_pinball_95: list[float] = []
    oof_preds = np.zeros((len(clean_df), 3), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        tr_clean = train_idx[clean_dry[train_idx]]
        te_clean_mask = clean_dry[test_idx]
        X_tr, y_tr = X.iloc[tr_clean], y.iloc[tr_clean]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        held_out_tracks = sorted(set(groups[test_idx]))

        fold_model = PhysicsDegradationEnsemble(ridge_weight=0.50, seed=42 + fold)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        oof_preds[test_idx] = preds
        y_arr = y_te.to_numpy(dtype=float)

        # Evaluate calibration & degradation fit on clean dry laps (plus full MAE/RMSE)
        y_ev = y_arr[te_clean_mask]
        p_ev = preds[te_clean_mask]
        p05_err = np.where(y_ev >= p_ev[:, 0], 0.05 * (y_ev - p_ev[:, 0]), 0.95 * (p_ev[:, 0] - y_ev)).mean()
        p50_err = np.where(y_ev >= p_ev[:, 1], 0.50 * (y_ev - p_ev[:, 1]), 0.50 * (p_ev[:, 1] - y_ev)).mean()
        p95_err = np.where(y_ev >= p_ev[:, 2], 0.95 * (y_ev - p_ev[:, 2]), 0.05 * (p_ev[:, 2] - y_ev)).mean()

        p50_mae = float(mean_absolute_error(y_ev, p_ev[:, 1]))
        p50_rmse = float(root_mean_squared_error(y_ev, p_ev[:, 1]))
        p50_r2 = float(r2_score(y_ev, p_ev[:, 1]))
        coverage = float(((y_ev >= p_ev[:, 0]) & (y_ev <= p_ev[:, 2])).mean())

        cv_p50_mae.append(p50_mae)
        cv_p50_rmse.append(p50_rmse)
        cv_p50_r2_clean.append(p50_r2)
        cv_coverage.append(coverage)
        cv_pinball_05.append(p05_err)
        cv_pinball_50.append(p50_err)
        cv_pinball_95.append(p95_err)

        print(f"Fold {fold} ({len(held_out_tracks)} tracks, {len(X_te):,} laps): "
              f"R2(clean)={p50_r2:+.4f} | MAE={p50_mae:.4f}s | RMSE={p50_rmse:.4f}s | Cov(90%)={coverage:.1%}")

    oof_r2_all = float(r2_score(y_all, oof_preds[:, 1]))
    oof_mae_all = float(mean_absolute_error(y_all, oof_preds[:, 1]))
    oof_r2_clean = float(r2_score(y_all[clean_dry], oof_preds[clean_dry, 1]))
    oof_mae_clean = float(mean_absolute_error(y_all[clean_dry], oof_preds[clean_dry, 1]))

    # Strictly causal Stage-2 Empirical Bayes OOF evaluation (uses only past laps < L per race)
    oof_eb_r2_all = oof_r2_all
    oof_eb_r2_clean = oof_r2_clean
    oof_eb_mae_clean = oof_mae_clean
    if 'Season' in clean_df.columns and 'LapNumber' in clean_df.columns:
        eb_tmp = pd.DataFrame({
            'Season': clean_df['Season'],
            'Track': clean_df['Track'],
            'LapNumber': clean_df['LapNumber'],
            'base': oof_preds[:, 1],
            'y': y_all,
            'obs': (clean_dry & (clean_df['tyre_age'].to_numpy(dtype=float) >= 3.0)),
        })
        eb_tmp['num'] = np.where(eb_tmp['obs'], eb_tmp['base'] * eb_tmp['y'], 0.0)
        eb_tmp['den'] = np.where(eb_tmp['obs'], eb_tmp['base'] ** 2, 0.0)
        eb_tmp['res'] = np.where(eb_tmp['obs'], eb_tmp['y'] - eb_tmp['base'], 0.0)
        eb_tmp['cnt'] = eb_tmp['obs'].astype(float)
        la = eb_tmp.groupby(['Season', 'Track', 'LapNumber'])[['num', 'den', 'res', 'cnt']].sum().reset_index()
        la = la.sort_values(['Season', 'Track', 'LapNumber'])
        for c in ['num', 'den', 'res', 'cnt']:
            la[f'c_{c}'] = la.groupby(['Season', 'Track'])[c].cumsum() - la[c]
        la['alpha'] = ((la['c_num'] + 12.0) / (la['c_den'] + 12.0)).clip(0.15, 2.8)
        la['bias'] = la['c_res'] / (la['c_cnt'] + 25.0)
        eb_m = eb_tmp.merge(la[['Season', 'Track', 'LapNumber', 'alpha', 'bias']], on=['Season', 'Track', 'LapNumber'], how='left')
        oof_eb = apply_race_eb(eb_m['base'].to_numpy(), eb_m['alpha'].fillna(1.0).to_numpy(), eb_m['bias'].fillna(0.0).to_numpy())
        oof_eb_r2_all = float(r2_score(y_all, oof_eb))
        oof_eb_r2_clean = float(r2_score(y_all[clean_dry], oof_eb[clean_dry]))
        oof_eb_mae_clean = float(mean_absolute_error(y_all[clean_dry], oof_eb[clean_dry]))

    print("\n" + "─" * 65)
    print("5-FOLD CROSS-VALIDATION SUMMARY (ON HELD-OUT CIRCUITS):")
    print(f"  Zero-Shot OOF R2 (All Lap>=2)  : {oof_r2_all:+.4f} | MAE = {oof_mae_all:.4f}s")
    print(f"  Zero-Shot OOF R2 (Clean Dry)   : {oof_r2_clean:+.4f} | MAE = {oof_mae_clean:.4f}s")
    print(f"  Causal Race-EB R2 (All Lap>=2) : {oof_eb_r2_all:+.4f}")
    print(f"  Causal Race-EB R2 (Clean Dry)  : {oof_eb_r2_clean:+.4f} | MAE = {oof_eb_mae_clean:.4f}s")
    print(f"  p50 MAE (clean fold mean)      : {np.mean(cv_p50_mae):.4f} +/- {np.std(cv_p50_mae):.4f} s")
    print(f"  p50 RMSE (clean fold mean)     : {np.mean(cv_p50_rmse):.4f} +/- {np.std(cv_p50_rmse):.4f} s")
    print(f"  90% Coverage Rate (Clean Dry)  : {np.mean(cv_coverage):.1%} +/- {np.std(cv_coverage):.1%}")
    print("─" * 65)

    # ── Train Final Production Model on Clean Dry Dataset ─────────────────────
    print(f"\nTraining Final Physics Ensemble on {int(clean_dry.sum()):,} Clean-Dry Lap>=2 laps...")
    final_model = PhysicsDegradationEnsemble(ridge_weight=0.50, seed=42)
    final_model.fit(X.iloc[clean_dry], y.iloc[clean_dry])
    print("[SUCCESS] Production model trained.")

    # ── Feature Importances ───────────────────────────────────────────────────
    importances = final_model.feature_importances_
    print("\n── Feature Importances (XGB Center, Track-ID Leakers Zeroed) ──")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"  {feat:<26} : {imp:.4f}  ({imp*100:.2f}%)")

    # ── Verify Monotonic Degradation Curve ────────────────────────────────────
    print("\n── Degradation Sanity Check (Pace delta vs tyre age) ──")
    sample_ages = [1, 5, 10, 15, 20, 25]
    test_batch = pd.DataFrame([
        {
            'tyre_age':                 age,
            'compound_encoded':         2.0,  # Medium
            'compound_hardness':        3.0,  # C3
            'track_evolution':          85.0,
            'TrackTemp':                32.0,
            'fuel_mass':                60.0,
            'abrasion_index':           3.0,
            'circuit_length_km':        5.4,
            'num_corners':              16.0,
            'corner_density':           16.0 / 5.4,
            'thermal_load':             age * 32.0,
            'abrasion_load':            age * 3.0,
            'wear_cliff':               (age / 15.0) ** 2.0,
            'compound_age_interaction': age * 3.0,
        }
        for age in sample_ages
    ])
    preds_sample = final_model.predict(test_batch[features])
    for age, (p5, p50, p95) in zip(sample_ages, preds_sample):
        print(f"  Tyre Age {age:2d}: p05={p5:+.3f}s | p50={p50:+.3f}s | p95={p95:+.3f}s")

    # ── Save Artifact ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({
        'model':           final_model,
        'features':        features,
        'quantile_alphas': quantile_alphas,
        'model_version':   'phase3_physics_ensemble_clean_groupkfold',
        'target':          target,
        'cv_metrics': {
            'r2_all_oof':      oof_r2_all,
            'r2_clean_oof':    oof_r2_clean,
            'r2_eb_all_oof':   oof_eb_r2_all,
            'r2_eb_clean_oof': oof_eb_r2_clean,
            'r2_fold_mean':    float(np.mean(cv_p50_r2_clean)),
            'mae_mean':        float(np.mean(cv_p50_mae)),
            'mae_std':         float(np.std(cv_p50_mae)),
            'rmse_mean':       float(np.mean(cv_p50_rmse)),
            'rmse_std':        float(np.std(cv_p50_rmse)),
            'coverage_mean':   float(np.mean(cv_coverage)),
            'pinball_05_mean': float(np.mean(cv_pinball_05)),
            'pinball_50_mean': float(np.mean(cv_pinball_50)),
            'pinball_95_mean': float(np.mean(cv_pinball_95)),
        }
    }, output_path)
    print(f"\n[SAVED] Validated model artifact saved to {output_path}")

    return final_model


if __name__ == "__main__":
    from ml.trainer import train_global_model as _run_train
    _run_train()
