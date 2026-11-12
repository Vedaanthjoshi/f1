# F1 Strategy Engineer: Probabilistic Race Strategy & Undercut Optimization

An end-to-end Formula 1 race analytics and pit strategy optimization platform. The system combines real-world telemetry from the **FastF1 API**, physics-informed feature engineering, multi-output quantile regression (**XGBoost**), and vectorized **Monte Carlo forward rollouts** to identify high-probability undercut windows and tactical pit stops.

---

## Documentation Roadmap

This repository includes detailed academic and technical documentation structured across the three development stages of the project:

| Document | Focus & Paradigm | Key Highlights |
| :--- | :--- | :--- |
| **[`01_ORIGINAL_PROJECT.md`](file:///c:/f1/01_ORIGINAL_PROJECT.md)** | **Phase 1: Naive Baseline** | Analysis of the initial model, documentation of the 5 fatal leakage flaws (baseline pace leakage, lookahead bias, negative fuel mass, random 80/20 shuffle), and why $R^2=0.97$ was invalid. |
| **[`02_ORIGINAL_PROJECT_14_FEATURES.md`](file:///c:/f1/02_ORIGINAL_PROJECT_14_FEATURES.md)** | **Phase 1.5: 14-Feature Point Model** | Deterministic regression without Monte Carlo. Exact formulas for the 14 physics features, GroupKFold (k=5) evaluation on unseen circuits, and gain importances. |
| **[`03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md`](file:///c:/f1/03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md)** | **Phase 2: Final Probabilistic Engine** | Full multi-output quantile regression with Pinball Loss, 500-rollout Monte Carlo optimizer, out-lap cold tyre warmup penalty, two-car game-theoretic competitor cover model, 24-circuit Safety Car model, and 15-race backtest validation ($80.0\%$ operational precision). |

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
| **Rival Tactical Reaction** | Ignored | Ignored | **Two-Car Relative Undercut Differential** |
| **Safety Car Dynamics** | Ignored | Ignored | **24-Circuit Poisson Rates + 65% Loss Drop** |
| **Backtest Precision** | $0.0\%$ | N/A (Deterministic) | **$80.0\%$ (4 TP, 1 FP on real telemetry; in-sample sanity check (13 of 15 races were in the training data))** |
| **Backtest Accuracy** | $26.7\%$ | $46.7\%$ | **$53.3\%$ (8 / 15 correct decisions; in-sample sanity check (13 of 15 races were in the training data))** |

---

## Repository Structure

```text
.
├── 01_ORIGINAL_PROJECT.md              # Phase 1: Baseline flaws & leakage analysis
├── 02_ORIGINAL_PROJECT_14_FEATURES.md   # Phase 1.5: 14-Feature point-estimate model & GroupKFold
├── 03_FINAL_PROJECT_QUANTILE_MONTE_CARLO.md # Phase 2: Quantile regression & Monte Carlo engine
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
│   ├── historical_laps.csv             # 73,309 clean racing laps (2022–2025)
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
