# Phase 1: Original Project Baseline (Naive Point-Estimate Model)

---

## 1. Executive Summary & Research Motivation

The original Formula 1 Strategy Engineer project aimed to predict the strategic **undercut window** between two competing drivers during a Grand Prix. The core concept of an undercut in Formula 1 is tactical: a trailing driver (Attacker) pits for fresh tyres one or two laps earlier than the leading driver (Defender). By exploiting the pace advantage of brand-new rubber, the attacker sets faster lap times while the defender remains on worn tyres, effectively leapfrogging the defender when they eventually pit.

To model this, the initial system ingested session lap telemetry via the **FastF1** Python API, trained a point-estimate machine learning regression model (**XGBoost**) on lap times, and computed a deterministic gap crossover curve to flag when an undercut was advantageous.

---

## 2. Dataset Dimensions & Scope (Table I)

The baseline dataset collected for the initial model captured the modern Formula 1 ground-effect aerodynamic era:

| Dataset Dimension | Baseline Value | Description |
| :--- | :---: | :--- |
| **Seasons Covered** | 2023 – 2024 | Modern ground-effect technical regulations |
| **Unique Grand Prix Venues** | **23 – 24** | 24 circuits modeled (23 held in 2023 due to Imola cancellation; 24 in 2024) |
| **Clean Racing Laps** | **33,569** | Laps under green-flag conditions (`TrackStatus == '1'`) |
| **Drivers Represented** | **27** | 20 full-time race drivers + reserve/stand-in drivers |
| **Telemetry Source** | FastF1 API | Official timing and telemetry feeds |

---

## 3. Original Feature Representation & Formulation

The original model passed a 10-feature vector into `model.fit()`:
1. `tyre_age`: Stint lap count (`TyreLife`)
2. `compound_encoded`: Nominal compound (`SOFT=1`, `MEDIUM=2`, `HARD=3`)
3. `track_evolution`: Track rubbering-in proxy
4. `TrackTemp`: Surface track temperature ($^\circ\text{C}$)
5. `Track`: Circuit identifier
6. `Driver`: Driver three-letter code (e.g. `VER`, `HAM`, `NOR`)
7. `abrasion_index`: Pirelli macro-roughness rating ($1$ to $5$)
8. `circuit_length_km`: FIA circuit centerline distance
9. `num_corners`: Turn count per lap
10. `compound_hardness`: Pirelli physical compound grade ($1=\text{C1}$ to $5=\text{C5}$)

### Original Target Variable
The target was formulated as:
$$\Delta T_{\text{target}} = T_{\text{fuel\_corrected}} - T_{\text{baseline}}$$

Where:
- $T_{\text{baseline}}$ was set as the lap time of the car at the start of the stint ($\text{tyre\_age} = 1$).
- $T_{\text{fuel\_corrected}} = T_{\text{actual}} - M_{\text{fuel}}(L) \times 0.03$.

---

## 4. Critical Methodological Flaws (Why Reviewers Rejected Phase 1)

Although initial evaluations claimed an astounding $R^2 = 0.9700$ with an $\text{MAE} = 0.42\,\text{s}$, peer reviewers correctly identified critical methodological failures and data leakage:

### Flaw 1: Target Leakage via Baseline Pace
* **The Error:** In the earliest iteration, `baseline_pace` was included directly in the feature vector while the target was $T - T_{\text{baseline}}$. The model was given the answer to memorize, artificially inflating $R^2$ to $0.97$.

### Flaw 2: Lookahead Bias in Track Evolution
* **The Error:** The track evolution proxy was computed as a session-wide average across all laps (including future laps $L+1 \dots L_{\text{end}}$). When predicting Lap 10, the model had access to track grip data from Lap 50.

### Flaw 3: Fuel Mass Depletion Underflow
* **The Error:** The fuel equation was $M_{\text{fuel}} = 110.0 - 1.7 \times L$. On races exceeding 65 laps (e.g., Zandvoort with 72 laps, Monaco with 78 laps), $M_{\text{fuel}}$ became negative, causing the formula to **add artificial lap time** at the end of the race.

### Flaw 4: Random Lap Shuffling vs. Cross-Circuit Generalization
* **The Error:** An ordinary 80/20 train/test random split was used. Laps from the exact same car, tyre set, and race were in both the training set and the test set, allowing XGBoost to memorize circuit-specific lap times rather than learning true tyre physics.

### Flaw 5: The Lap 1 Standing Start Outlier
* **The Error:** Lap 1 in every Grand Prix is 5 to 8 seconds slower due to standing starts and Turn 1 traffic. Because the code set Lap 1 as $T_{\text{baseline}}$, every subsequent clean lap had a **negative degradation delta** ($\Delta T \approx -5.2\,\text{s}$). The model concluded tyres were getting *faster* forever, advising cars to `STAY_OUT` indefinitely.

### Impact on Historical Backtesting
When tested against 15 real Grand Prix undercut battles:
* **Operational Accuracy:** Only **$26.7\%$**
* **Precision:** **$0.0\%$** (Failed to recommend a single correct undercut pit call).
