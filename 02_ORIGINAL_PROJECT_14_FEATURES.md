# Phase 1.5: Original Point-Estimate Model with 14 Features & Zero-Leakage GroupKFold (Without Monte Carlo)

---

## 1. Overview & Architectural Scope

This document details the **deterministic point-estimate version** of the F1 Strategy Engineer, updated to satisfy academic review requirements. It resolves all data leakage and lookahead bias while expanding the feature vector into a **14-feature physics-informed suite**. 

Crucially, this version **does not use Monte Carlo rollouts or quantile regressions**. It operates as a deterministic regressor predicting point lap times and evaluating the classic undercut differential equation directly.

---

## 2. Mathematical Corrections Applied

### 2.1 Causal Expanding Track Evolution (Fix 2)
To eliminate lookahead bias, track evolution at lap $L$ is strictly restricted to historical laps prior to $L$ via a lag-1 shifted expanding mean:
$$\tilde{T}(i) = \text{median}_{j \in \mathcal{D}_i}(T_j(i))$$
$$\text{TrackEvolution}(L) = \frac{1}{L - 1} \sum_{i=1}^{L-1} \tilde{T}(i) \quad \text{for } L \ge 2, \quad \text{with } \text{TrackEvolution}(1) = \tilde{T}(1)$$

### 2.2 Clamped Fuel Mass Model (Fix 1)
Fuel mass is bounded by a physical floor of $2.0\,\text{kg}$ to prevent negative weight underflow on races with $>65$ laps:
$$M_{\text{fuel}}(L) = \max\Big(110.0 - 1.7 \times L,\; 2.0\Big) \quad [\text{kg}]$$
$$T_{\text{fuel\_corrected}}(L) = T_{\text{actual}}(L) - M_{\text{fuel}}(L) \times 0.03$$

### 2.3 Standing Start Outlier Elimination
Lap 1 standing starts (5–8s off-pace) are excluded from baseline calculations. Stint baseline pace $T_{\text{baseline}}$ is determined using clean running laps within $104\%$ of median stint pace.

---

## 3. The 14-Feature Physics-Informed Suite

The complete 14-feature vector passed into `model.fit()` consists of 6 dynamic session features, 4 circuit physical metrics, and 4 interaction terms:

| # | Feature Name | Formula / Definition | Gain Importance (%) | Physical Role |
|---|---|---|:---:|---|
| 1 | `corner_density` | $\frac{\text{num\_corners}}{\text{circuit\_length\_km}}$ | **20.04%** | Turn frequency and lateral duty cycle |
| 2 | `Track` | Nominal Track Name | **16.70%** | Macro circuit layout baseline |
| 3 | `circuit_length_km` | FIA Track Centerline Length | **12.04%** | Circuit scale and transit distance |
| 4 | `num_corners` | Turn Count per Lap | **8.85%** | Total lateral scrubbing events |
| 5 | `compound_encoded` | $\text{Soft}=1, \text{Med}=2, \text{Hard}=3$ | **6.55%** | Tactical tyre compound tier |
| 6 | `compound_hardness`| Pirelli Scale ($1=\text{C1}$ to $5=\text{C5}$) | **6.00%** | Physical rubber family tear resistance |
| 7 | `Driver` | Driver 3-Letter Code | **5.88%** | Driver pace delta & tyre preservation |
| 8 | `wear_cliff` | $(\frac{\text{tyre\_age}}{15.0})^2$ | **4.87%** | Non-linear quadratic tyre cliff |
| 9 | `track_evolution` | Causal expanding field median | **4.46%** | Track rubbering-in and surface grip |
| 10 | `TrackTemp` | Telemetry Surface Temp ($^\circ\text{C}$) | **4.08%** | Thermal operational boundary |
| 11 | `abrasion_index` | Pirelli Roughness ($1$ to $5$) | **3.86%** | Asphalt macro/micro-texture wear rate |
| 12 | `thermal_load` | $\text{tyre\_age} \times \text{TrackTemp}$ | **2.47%** | Compound overheating & blister risk |
| 13 | `tyre_age` | Lap count on current tyre set | **2.15%** | Linear stint progression |
| 14 | `abrasion_load` | $\text{tyre\_age} \times \text{abrasion\_index}$ | **2.03%** | Surface graining and mechanical wear |

> **Leakage Removal Guarantee:** `baseline_pace` is **strictly excluded** from this feature set.

---

## 4. Model Training & Hyperparameters

The model is trained as a deterministic point-estimate gradient boosted decision tree using **XGBoost**:

```python
model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method='hist',
    objective='reg:squarederror',
    random_state=42
)
```

---

## 5. Validation Methodology: Race-Wise GroupKFold (Block LORO)

To simulate genuine deployment on unseen circuits, validation uses **Race-Wise GroupKFold ($k=5$) grouped by `Track`**:
- In each fold, 4–5 complete Grand Prix circuits (~6,900 laps) are held out entirely.
- The model is trained on 19 circuits and evaluated exclusively on tracks it has never seen.
- This represents a block-level implementation of **Leave-One-Race-Out (LORO)** cross-validation.

---

## 6. Official Point-Estimate Benchmark Results

Under strict 5-fold cross-validation on unseen tracks:

### Target A: Cross-Circuit Pace Generalization (`fuel_corrected_time`)
Evaluates the model's ability to predict absolute car pace across different venues:

| Metric | Fold Mean ($\pm$ Std) | Individual Fold Scores |
| :--- | :---: | :--- |
| **$R^2$ Score** | **$0.9176 \pm 0.0426$** | $[0.9716, 0.9260, 0.9260, 0.7598, 0.9557]$ |
| **MAE** | **$2.0437 \pm 0.8100\,\text{s}$** | $[1.12\,\text{s}, 1.84\,\text{s}, 2.21\,\text{s}, 3.14\,\text{s}, 1.91\,\text{s}]$ |
| **RMSE** | **$2.6014 \pm 0.9181\,\text{s}$** | $[1.45\,\text{s}, 2.31\,\text{s}, 2.76\,\text{s}, 3.92\,\text{s}, 2.57\,\text{s}]$ |
| **MSE** | **$6.7673\,\text{s}^2$** | $\text{RMSE}^2$ |

### Target B: Pure Within-Stint Degradation Delta ($\Delta T_{\text{deg}}$)
Evaluates the model's ability to isolate and predict only the $0 - 4\,\text{s}$ tyre degradation component:

| Metric | Fold Mean ($\pm$ Std) | Individual Fold Scores |
| :--- | :---: | :--- |
| **$R^2$ Score** | **$0.5805 \pm 0.0103$** | $[0.5951, 0.5813, 0.5644, 0.5753, 0.5862]$ |
| **MAE** | **$0.4037 \pm 0.0047\,\text{s}$** | $[0.3960\,\text{s}, 0.4035\,\text{s}, 0.4043\,\text{s}, 0.4106\,\text{s}, 0.4042\,\text{s}]$ |
| **RMSE** | **$0.5659 \pm 0.0065\,\text{s}$** | $[0.5547\,\text{s}, 0.5641\,\text{s}, 0.5709\,\text{s}, 0.5733\,\text{s}, 0.5666\,\text{s}]$ |

---

## 7. Deterministic Undercut Calculation (No Monte Carlo)

In the point-estimate paradigm, the undercut advantage is calculated deterministically across an evaluation horizon $H = 15$ laps:

$$\text{Net Advantage}(h) = \sum_{k=1}^h \Big( \hat{T}_{\text{defender}}(L + k) - \hat{T}_{\text{attacker}}(L + k) \Big) - \Delta T_{\text{pit}}$$

Where:
- $\hat{T}_{\text{defender}}$: Predicted point lap time of defender staying out on old rubber.
- $\hat{T}_{\text{attacker}}$: Predicted point lap time of attacker on fresh tyres (including a $+1.5\,\text{s}$ out-lap warmup penalty).
- $\Delta T_{\text{pit}}$: Dynamic pit stop transit loss (~$24.5\,\text{s}$).

### Limitations of the Deterministic Approach
While the point-estimate model achieves high statistical accuracy, it suffers from two major operational limitations on the pit wall:
1. **No Uncertainty Boundaries:** It outputs a single expected number, hiding whether the tyre degradation is volatile or stable.
2. **Ignored Game Theory & Neutralizations:** It cannot model rival defensive reactions (whether the leader covers on the next lap) or Safety Car probability spikes. These limitations necessitated the Phase 2 Monte Carlo engine.
