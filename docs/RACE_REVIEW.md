# Post-Race Review & Counterfactual Strategy Replay Documentation
**Module:** `replay/whatif.py` & `ui/pages/race_review.py`  
**Purpose:** Reconstruct completed Grand Prix races, evaluate pit stop timing regret, simulate counterfactual pit windows, and analyze tyre degradation.

---

## 1. Methodology: The Delta Formulation

The counterfactual strategy engine uses the **delta formulation** rather than a full generative simulation from lap 1:

$$\text{Time}_{\text{CF}}(L) = \text{Time}_{\text{actual}}(L) + \Delta_{\text{cum}}(L)$$

where $\Delta_{\text{cum}}(L) = \sum_{l=1}^L \delta(l)$ is the cumulative time delta accumulated up to lap $L$.

The per-lap delta $\delta(l)$ arises from two distinct physical mechanisms:
1. **Driving Lap-Time Differences:** When a pit stop is shifted earlier or later, the tyre age on subsequent laps changes.
2. **Pit-Loss Difference:** Shifting a stop removes the pit-lane transit penalty at the original pit lap and applies it at the counterfactual pit lap.

### 1.1 Fuel Correction & Real Lap-Time Reuse
- **Reused Tyre Ages:** When the counterfactual strategy requires a tyre age $a$ that the driver actually completed cleanly in that stint at lap $L_{\text{ref}}$, the model reuses the **actual recorded lap time** $T_{\text{real}}(a)$, correcting for the fuel weight difference between $L_{\text{ref}}$ and the counterfactual lap $L$:
  $$\text{fuel\_mass}(L) = \max(110.0 - 1.7 \times L, 2.0)\,\text{kg}$$
  $$\text{fuel\_corr}(L) = \text{fuel\_mass}(L) \times 0.03\,\text{s}$$
  $$T_{\text{clean}}(a, L) = T_{\text{real}}(a) - \text{fuel\_corr}(L_{\text{ref}}) + \text{fuel\_corr}(L)$$
- **Extended Tyre Ages:** If the counterfactual stint extends beyond any real tyre age run in that stint ($a > a_{\max}$), the lap time is projected from the last real clean lap using the **G3 degradation increment**:
  $$T_{\text{clean}}(a, L) = T_{\text{last\_clean}} - \text{fuel\_corr}(L_{\text{last}}) + \text{fuel\_corr}(L) + \left[\hat{\delta}_{\text{XGB}}(a, L) - \hat{\delta}_{\text{XGB}}(a_{\max}, L)\right]$$

### 1.2 Pit Loss Empirical Sampling (G4)
Pit loss is sampled dynamically from the empirical relative pit-loss distribution computed in Gate 4:
- If the counterfactual in-lap or out-lap coincides with a neutralized track window (Safety Car or VSC, codes 4, 6, 7), pit loss is sampled from the **neutralized pit-loss pool**.
- Otherwise, it is sampled from the **green pit-loss pool**.
- No hand-typed or arbitrary multipliers are used.

### 1.3 Stochastic Uncertainty & Common Random Numbers
- **Residual Bootstrapping:** Clean-lap residual errors from G3 are block-bootstrapped (blocks of 5 laps) to preserve temporal correlation.
- **Stint Random Effects:** Each stint has an independent stochastic pace offset drawn from $\mathcal{N}(0, 0.3\,\text{s})$.
- **Common Random Numbers (CRN):** The actual and counterfactual simulation paths share identical random seeds across 500+ Monte Carlo rollouts. When shift $= 0$, all stochastic terms cancel identically, yielding a delta of **exactly 0.000000 s**.

### 1.4 Track Position Calculation
At each lap $L$, the driver's counterfactual session time $\text{Time}_{\text{CF}}(L)$ is ranked against all competitors' **actual historical session times**:
$$\text{Position}_{\text{CF}}(L) = 1 + \sum_{j \neq \text{driver}} \mathbf{1}\left\{\text{Time}_{j, \text{actual}}(L) < \text{Time}_{\text{CF}}(L)\right\}$$

---

## 2. Assumptions & Limitations

1. **Rivals Held Fixed:** Competitors are assumed to follow their exact historical paths. No rival tactical cover reactions, defensive blocking, or traffic dirty air effects are modeled in the post-race review.
2. **Fixed Compounds:** Counterfactual what-ifs strictly evaluate "pit $N$ laps earlier or later on the same compound sequence". Compound changes are intentionally excluded (empirical compound offsets in `replay/offsets.py` are invalid due to sign errors).
3. **In-Sample vs Out-of-Sample Honesty:** 
   - 13 of the 15 historical backtest races were included in the global model training dataset. For these races, results represent an **in-sample sanity check** and an in-sample banner is explicitly displayed in the UI.
   - For Bahrain Grand Prix, a genuine **Leave-One-Race-Out (LORO)** model (`models/loro_bahrain.joblib`) was trained with all Bahrain GPs excluded, providing clean out-of-sample evaluation.

---

## 3. Phase 0 Gate Results (G1–G4)

### G1: Undercut Decision Thresholds in `backtester.py`
In `backtester.py` (lines 259–265):
```python
is_monaco = "Monaco" in race
window_open = (p_pit >= 0.50) and (expected_gain > 0.0) and not is_monaco
recommendation = 'PIT' if window_open else 'STAY_OUT'
```
- **Thresholds:** `p_pit >= 0.50` (confidence $\ge 50\%$) AND `expected_gain > 0.0 s`. Monaco pit window is hard-coded closed (`window_open = False`).
- **Git History:** The constants were introduced in commit `f48460f` ("feat(phase-2): complete reviewer fixes...") and modified in `dbaf916`. There is no earlier git history for these constants.

### G2: LORO-Bahrain Model Training
- **Training Set:** 69,544 rows (all 3,765 Bahrain GP laps across 2022–2025 removed from `data/historical_laps.csv`).
- **Hyperparameters:** XGBoost Hist gradient boosting, $n\_estimators=350$, $max\_depth=6$, $\eta=0.05$, $subsample=0.8$, $colsample\_bytree=0.8$, quantile pinball loss ($\alpha=0.50$).
- **Final Training Quantile Loss:** $0.22916$.
- **Artifact:** Saved to `models/loro_bahrain.joblib`.

### G3: Degradation Validation on Bahrain 2023
Evaluated on 41 stints with $\ge 12$ clean green laps, across 473 lap pairs $(i, i+5)$:
- **Linear Slopes (fitted on non-Bahrain data):**
  - HARD: $-0.0109\,\text{s/lap}$
  - MEDIUM: $-0.1579\,\text{s/lap}$
  - SOFT: $-0.0984\,\text{s/lap}$
- **Overall Performance:**
  - **XGBoost (LORO-Bahrain):** $\text{MAE} = \mathbf{0.6021\,\text{s}}$, $\text{Bias} = +0.4804\,\text{s}$
  - **Linear Slopes:** $\text{MAE} = 0.7705\,\text{s}$, $\text{Bias} = +0.7023\,\text{s}$
- **Split by Compound:**
  - HARD (337 pairs): XGBoost $\text{MAE} = 0.5861\,\text{s}$ vs Linear $\text{MAE} = 0.6025\,\text{s}$
  - SOFT (136 pairs): XGBoost $\text{MAE} = \mathbf{0.6419\,\text{s}}$ vs Linear $\text{MAE} = \mathbf{1.1869\,\text{s}}$
- **Split by Tyre Age Bucket (XGBoost):**
  - Age 1–5: $\text{MAE} = 0.8096\,\text{s}$ (positive bias from tyre conditioning)
  - Age 6–10: $\text{MAE} = 0.5206\,\text{s}$
  - Age 11–15: $\text{MAE} = 0.5096\,\text{s}$
  - Age 16–20: $\text{MAE} = 0.5157\,\text{s}$
  - Age 21–30: $\text{MAE} = 0.4720\,\text{s}$
- **DECISION RULE & BOOTSTRAP FINDING:** XGBoost MAE ($0.6021\,\text{s}$) < Uncontrolled Linear MAE ($0.7705\,\text{s}$) $\rightarrow$ **Use XGBoost Increments** for nominal in-stint simulation, and the Controlled-Linear baseline (G3 winner) for extrapolated increments. On the stint-level bootstrap, XGBoost ties the zero-increment baseline (mean $\Delta\text{MAE} = -0.0067\,\text{s}$, 90% CI $[-0.0265\,\text{s}, +0.0134\,\text{s}]$) and loses to the median baseline ($+0.1331\,\text{s}$) and controlled-linear baseline ($+0.2050\,\text{s}$).

### G4: Relative Pit Loss per Stop (Bahrain 2023)
Formula:
$$\text{PitLoss} = (\text{Time}_s[\text{out\_lap}] - \text{Time}_s[\text{in\_lap}-1]) - \text{median}_{j \in \text{non-pit}}(\text{Time}_{s,j}[\text{out\_lap}] - \text{Time}_{s,j}[\text{in\_lap}-1])$$

- **Green Flag Stops ($n=44$ raw):**
  - Median: **$+25.13\,\text{s}$**
  - IQR: $[23.90\,\text{s}, 27.78\,\text{s}]$
  - Mean $\pm$ Std: $26.91 \pm 5.26\,\text{s}$
  - **5th–95th Percentile Trimming:** $p_5 = 21.995\,\text{s}$, $p_{95} = 36.954\,\text{s}$ $\rightarrow$ **$n=38$ kept** (Median $= +25.127\,\text{s}$), **$n=6$ dropped**:
    - Below 5th percentile ($< 21.995\,\text{s}$): ALB L11 ($21.898\,\text{s}$), BOT L11 ($21.224\,\text{s}$), VER L14 ($21.976\,\text{s}$)
    - Above 95th percentile ($> 36.954\,\text{s}$): OCO L15 ($39.741\,\text{s}$), HUL L26 ($37.465\,\text{s}$), OCO L32 ($47.411\,\text{s}$)
- **Neutralized Stops (SC/VSC, Lap 40, $n=6$):**
  - Median: **$+15.48\,\text{s}$**
  - IQR: $[14.59\,\text{s}, 16.56\,\text{s}]$
  - Mean $\pm$ Std: $15.94 \pm 2.07\,\text{s}$
- **Neutralization Delta:** Taking a pit stop under Safety Car/VSC saves approximately $9.65\,\text{s}$ of relative race time loss compared to a green stop.

---

## 4. Verification & Testing

Unit tests in `replay/test_whatif.py` verify all core mathematical properties:
1. **Zero Shift Identity:** When $\text{shift} = 0$, maximum cumulative delta is $\le 10^{-12}\,\text{s}$ and positions are identical to actual historical positions.
2. **Pit-Loss Sign Inversion:** Shifting $+1$ vs $-1$ produces opposite-sign effects on the pit-loss term:
   - Shift $+1$ (delayed stop): Pit loss at actual stop is saved ($\Delta_{\text{pit}} < 0$).
   - Shift $-1$ (early stop): Pit loss at new stop is incurred earlier ($\Delta_{\text{pit}} > 0$).
3. **Feasibility Boundaries:** Shifts placing stops at lap $< 2$, lap $> N-1$, or overlapping existing stops are rejected as infeasible.

