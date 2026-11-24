# F1 Strategy Engineer: Probabilistic Race Strategy & Undercut Optimization

An end-to-end Formula 1 race analytics and pit strategy optimization platform. The system combines real-world telemetry from the **FastF1 API**, physics-informed feature engineering, multi-output quantile regression (**XGBoost**), and vectorized **Monte Carlo forward rollouts** to identify high-probability undercut windows and tactical pit stops.

---

## Current evaluation

The dataset contains **33,569 recorded lap rows**. The latest calibrated XGBoost evaluation achieved **89.7% coverage across 311,485 forecast cases per model**. Multiple forecast horizons can target the same recorded lap. The stay-out point model scored **0.9940 pace R²** and **0.8276 degradation R²** across five held-out circuit folds.

[Download the one-page results](docs/f1-current-evaluation.pdf) · [Read the evaluation and strategy report](docs/03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md)

## Circuit-held-out pace and tyre evaluation

Run `python -m ml.forecast_evaluation` from the repository root. The first run saves five circuit folds in `forecast_results/folds.csv`. Subsequent runs reuse them. Three inner circuit folds select a model by cumulative error over the next three consecutive clean laps in the same stint. The dashboard uses the saved model that held out the selected circuit for its stay-out pace forecast.

The forecast starts when a driver finishes a lap. The inputs use observed tyre life, fresh-set status, compound allocation for the event, current temperature, completed field laps and the driver's recent stint history. Scores apply to future dry, green, same-stint laps. Each model forecasts the same eligible observations. The degradation score measures fuel-corrected pace change relative to the first observed clean stint lap. Traffic, driving style and tyre wear can all contribute to this quantity.

The output includes every out-of-fold prediction, fold and circuit metrics, age and compound breakdowns, cumulative three-lap MAE, and a fold-4 residual plot. The 14-feature XGBoost comparison uses forecast-time inputs and the same one-lap horizon. Historical published scores used a different evaluation. The existing quantile model continues to drive Monte Carlo pit scenarios.

Run `python -m unittest test_forecast_cutoff` to verify prediction invariance after truncating race telemetry. For independent later-season confirmation, run `python -m ml.confirm_forecasts path/to/later_laps.csv` with a compatible CSV. This uses frozen models and rejects overlapping seasons. The historical collector excludes races with red flags, so the benchmark covers a selected race population.

## Experimental XGBoost strategy intervals

Run `python -m ml.strategy_quantiles` followed by `python -m ml.replay_quantiles` from the repository root. These commands use the historical lap CSV, existing circuit-fold manifest and local FastF1 race timing cache. Outputs are versioned under `quantile_results_v1`. Each outer fold reserves separate circuits for hyperparameter selection and asymmetric interval calibration. Only the former group rejoins the model-fitting circuits after tuning. Overlapping forecast horizons measure forecast instances and should not be interpreted as independent laps.

The dashboard checkbox **Compare experimental XGBoost pit probabilities** runs the held-out challenger beside the current engine. The strategy replay samples up to four stay-out and four fresh-set pit decisions per event with a fixed seed. It compares elapsed race times for the driven strategy, including pit-transition laps. The lap-level summary uses eligible clean target laps. These are different validation outcomes.

Run `python -m unittest test_forecast_cutoff test_strategy_quantiles` for timestamp cutoff, quantile ordering, shared pre-stop trajectories and seed checks. A later-season confirmation can be run with `python -m ml.confirm_quantiles YEAR "Grand Prix name" --allocation HARD MEDIUM SOFT`, supplying numeric Pirelli compound specifications and access to FastF1 timing and weather.

## Documentation Roadmap

This repository includes detailed academic and technical documentation structured across the three development stages of the project:

| Document | Focus & Paradigm | Key Highlights |
| :--- | :--- | :--- |
| **[`01_ORIGINAL_PROJECT.md`](docs/01_ORIGINAL_PROJECT.md)** | **Phase 1: Naive Baseline** | Analysis of the initial model, documentation of the 5 fatal leakage flaws (baseline pace leakage, lookahead bias, negative fuel mass, random 80/20 shuffle), and why $R^2=0.97$ was invalid. |
| **[`02_ORIGINAL_PROJECT_14_FEATURES.md`](docs/02_ORIGINAL_PROJECT_14_FEATURES.md)** | **Phase 1.5: 14-Feature Point Model** | Deterministic regression without Monte Carlo. Exact formulas for the 14 physics features, GroupKFold (k=5) evaluation on unseen circuits ($R^2=0.9078$ mean of listed pace folds, $R^2=0.5805$ on pure degradation, $\text{MAE}=0.4037\,\text{s}$), and gain importances. |
| **[`03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md`](docs/03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md)** | **Phase 2: Final Probabilistic Engine** | Full multi-output quantile regression with Pinball Loss, 500-rollout Monte Carlo optimizer, out-lap cold tyre warmup penalty, 4-tier competitor rival reaction model, 24-circuit Safety Car model, and 15-race backtest validation ($60\%$ accuracy, $100\%$ precision). |

---

## Evolution at a Glance

| Architectural Dimension | Phase 1 (Baseline) | Phase 1.5 (14-Feature Point) | Phase 2 (Final Quantile + MC) |
| :--- | :---: | :---: | :---: |
| **Model Type** | Point-Estimate XGBoost | Point-Estimate XGBoost | Multi-Output Quantile XGBoost |
| **Loss Objective** | Squared Error (MSE) | Squared Error (MSE) | Asymmetric Pinball Loss ($\tau = 0.05, 0.50, 0.95$) |
| **Features Count** | 10 (Leaky formulation) | **14 Physics-Informed** | **14 Physics-Informed** |
| **Cross-Validation** | Random 80/20 split | **Race-Wise GroupKFold ($k=5$)** | **Race-Wise GroupKFold ($k=5$)** |
| **Target Leakage** | Present (`baseline_pace`) | **Zero Leakage** | **Zero Leakage** |
| **Track Evolution** | Lookahead session mean | **Causal Lag-1 Expanding Median** | **Causal Lag-1 Expanding Median** |
| **Tyre Warmup Penalty** | $0.0\,\text{s}$ (Instant grip) | $+1.5\,\text{s}$ on Out-Lap | $+1.5\,\text{s}$ on Out-Lap |
| **Uncertainty Bounds** | None (Single float) | None (Point estimate) | **$90\%$ Confidence Intervals ($p05 - p95$)** |
| **Rival Tactical Reaction** | Ignored | Ignored | **Calibrated Cover Probability ($P_{\text{cover}}$)** |
| **Safety Car Dynamics** | Ignored | Ignored | **24-Circuit Poisson Rates + 65% Loss Drop** |
| **Backtest Precision** | $0.0\%$ | N/A (Deterministic) | **$100.0\%$ (Zero false pit alarms)** |
| **Backtest Accuracy** | $26.7\%$ | $46.7\%$ | **$60.0\%$ (9 / 15 correct decisions)** |

---

## Repository Structure

```text
.
├── docs/                               # Project reports and current evaluation PDF
├── README.md                           # Master technical overview
│
├── pipeline/                           # Data Pipeline & Feature Engineering
│   ├── ingestion.py                    # FastF1 telemetry ingestion & weather merge
│   ├── features.py                     # 14 physics features & circuit descriptors
│   └── baseline.py                     # Non-leaky stint baseline formulation
│
├── engine/                             # Strategy & Simulation Engine
│   ├── monte_carlo.py                  # Vectorized 500-rollout forward simulator
│   ├── strategies.py                   # 6 tactical candidate strategy definitions
│   ├── rival_reaction.py               # Empirical competitor cover reaction model
│   ├── safety_car.py                   # 24-circuit empirical SC rates & loss collapse
│   └── pit_loss.py                     # Dynamic session pit transit loss calculator
│
├── ml/                                 # Model Architecture & Training
│   ├── trainer.py                      # Multi-quantile XGBoost trainer & GroupKFold
│   ├── model_loader.py                 # Thread-safe model cache loader
│   └── explainability.py               # SHAP feature importance & attribution
│
├── data/                               # Historical datasets & metadata
│   ├── historical_laps.csv             # 33,569 clean racing laps (2024–2025)
│   ├── tracks.json                     # Circuit physics parameters
│   └── drivers.json                    # Grid driver profiles & styling
│
├── app.py                              # Streamlit Interactive Analytics Dashboard
├── backtester.py                       # 15-race historical validation suite
├── train_global_model.py               # CLI global model training script
└── simulation.py                       # Monte Carlo facade script
```

---

## Quick Start Guide

### 1. Environment Setup
Requires Python 3.10+ (tested on Python 3.13):
```bash
pip install -r requirements.txt
```

### 2. Train the Global Degradation Model
Trains the multi-quantile XGBoost model using the 14-feature physics suite with race-wise GroupKFold:
```bash
python train_global_model.py
```

### 3. Run Historical Undercut Backtesting
Executes the strategy engine across 15 real-world Grand Prix undercut battles:
```bash
python backtester.py
```

### 4. Launch the Interactive Dashboard
Starts the Streamlit strategy dashboard:
```bash
streamlit run app.py
```
