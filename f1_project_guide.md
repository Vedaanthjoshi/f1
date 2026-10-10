# 🏎️ F1 Strategy Engineer — Interview Learning Guide

> A complete walkthrough of everything you built, so you can explain it confidently.

---

## What Is This Project?

You built an **F1 Race Strategy Dashboard** — a Streamlit web app that:
1. Pulls **real F1 telemetry** from the FastF1 API
2. Cleans and engineers features from raw lap data
3. Uses a **trained ML model** (XGBoost) to predict tyre degradation
4. Runs a **pit stop undercut simulation** to tell a race engineer: *"Should I pit now?"*

Think of it as a simplified version of the tools real F1 teams use on the pit wall.

---

## 🗂️ Project Architecture — The Big Picture

```
User Selects Race
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                      app.py (Streamlit UI)              │
│  - Sidebar race selector                                │
│  - Renders charts + simulation                          │
└──────────┬──────────────────────────────────────────────┘
           │ calls
    ┌──────┴──────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
data_pipeline.py                          simulation.py
(Phase 1: Raw Data)                       (Phase 4: Sim)
    │                                             ▲
    ▼                                             │
feature_engineering.py                     ml_model.py
(Phase 2: Features)                        (Phase 3: ML)
    │                                             │
    └──────────────────────────────────────────→──┘
                    data flows upward

       ┌──────────────────────────┐
       │  train_global_model.py   │  ← Run ONCE offline
       │  data_collector.py       │  ← Collects historical CSVs
       └──────────────────────────┘
             saves → models/global_degradation_model.joblib
```

---

## 📁 File-by-File Breakdown

### 🔵 `app.py` — The Brain / Entry Point
**What it does:** The main Streamlit app. It wires everything together.

**Key concepts to understand:**

| Function | What It Does |
|---|---|
| `get_schedule(year)` | Fetches the race calendar for a given F1 season using FastF1 |
| `load_race_data(year, race)` | Loads a full race session (laps, weather, timing) from FastF1 |
| `load_global_model()` | Loads the pre-trained XGBoost model from disk |
| `process_backend_pipeline(session)` | Runs Phases 1-3: clean → engineer → prep ML data |
| `render_strategy_dashboard()` | Draws all 3 charts: Tyre degradation, Gap evolution, Undercut sim |

**`@st.cache_data` / `@st.cache_resource`** — Caching decorators so the app doesn't re-download F1 data on every user interaction. `cache_resource` is for non-serializable objects (like the ML model); `cache_data` is for DataFrames and plain data.

**`st.session_state`** — Streamlit's way of persisting state across reruns. Used to remember which race the user selected.

---

### 🔵 `data_pipeline.py` — Phase 1: Raw Data

**The job:** Take raw F1 session data → clean it → attach weather.

**Three key functions:**

#### `is_red_flag_race(session)`
```python
# TrackStatus '5' = Red Flag in FastF1
statuses = session.track_status['Status'].unique()
return '5' in statuses
```
Red flag races are **excluded** because the race restart creates a discontinuity in tyre data that breaks the degradation model.

#### `clean_lap_data(laps)`
Removes noise from the raw laps:
- **In-laps / Out-laps** → Filtered out via `PitInTime` / `PitOutTime` being not-null. These laps are anomalously slow (driver pitting) and would corrupt the model.
- **Non-green flag laps** → `TrackStatus == '1'` means fully green. Safety cars slow everyone down artificially.
- Converts `LapTime` (a Python `timedelta`) → `LapTime_s` (float seconds) for math.

#### `merge_weather_data(laps, weather)`
Uses `pd.merge_asof()` — a **time-series nearest-key join**. Attaches the most recent weather reading (TrackTemp, etc.) to each lap. This matters because higher track temperatures accelerate tyre degradation.

---

### 🔵 `feature_engineering.py` — Phase 2: Feature Engineering

**The job:** Turn clean laps into ML-ready features.

#### Features created:

| Feature | How | Why |
|---|---|---|
| `tyre_age` | Mapped from FastF1's `TyreLife` column | How worn the tyre is |
| `compound_encoded` | SOFT=1, MEDIUM=2, HARD=3 | Makes compound a numeric input for the model |
| `fuel_corrected_time` | `LapTime_s - (110 - 1.7×lap) × 0.03` | Removes fuel weight effect from lap time |
| `track_evolution` | Median lap time per lap number across all drivers | Captures how the track "rubbers in" and gets faster over the race |

**Why fuel correction?** A car burns ~1.7 kg of fuel per lap from a starting load of 110 kg. Heavier car = slower lap time. Without correcting for this, the model would incorrectly attribute slowdowns to tyre wear instead of fuel weight.

**Why track evolution?** The track gets grippier as rubber is laid down over a race. The median lap time captures this — if everyone is going faster on lap 40, it's the track, not just the tyres.

---

### 🔵 `ml_model.py` — Phase 3: ML Data Prep

**The job:** Define the ML target variable and prepare data for training.

#### `prepare_ml_data(df)`

The target variable is `lap_time_delta`:
```
lap_time_delta = fuel_corrected_time − baseline_pace
```

- **`baseline_pace`** = the fuel-corrected lap time on the **freshest tyre** (minimum tyre_age) of that driver's stint
- **`lap_time_delta`** = how much *slower* (in seconds) the driver is compared to when they had fresh rubber

This isolates tyre degradation from driver ability, car pace, or anything else. A `lap_time_delta` of `+2.5` means the tyres have degraded by 2.5 seconds since they were fresh.

#### `train_degradation_model(ml_df)` (used in older version)
Trains a `RandomForestRegressor`. The global model in `train_global_model.py` uses **XGBoost** instead (more powerful, GPU-capable, handles categorical features natively).

---

### 🔵 `train_global_model.py` — Offline Training

**The job:** Train one model on ALL historical races, not just one race.

**Why global?** A model trained on 3+ seasons across 20+ tracks generalises much better. It's learned how Softs degrade at Bahrain vs Silverstone, what VER's typical baseline looks like, etc.

**Model:** `XGBRegressor` with:
- `n_estimators=300` — 300 trees in the ensemble
- `max_depth=8` — how deep each tree can go
- `learning_rate=0.05` — smaller = slower but more precise learning
- `enable_categorical=True` — XGBoost natively handles `Track` and `Driver` as categories
- GPU training via `device='cuda'` with CPU fallback

**Saved as:** `models/global_degradation_model.joblib` (a dictionary with `model` + `features` list)

---

### 🔵 `simulation.py` — Phase 4: Undercut Simulation

**The most complex and interview-worthy file.**

#### `calculate_pit_loss(session)`
Calculates how many seconds are lost by driving through the pit lane. It matches **in-laps** (Stint N) with **out-laps** (Stint N+1) per driver and takes the median pit lane time across all stops. Typically 24-26 seconds.

#### `simulate_undercut_window(...)` — The Core Algorithm

For 15 laps ahead, it runs two parallel simulations:

**Scenario A — Stay Out:**
```
tyre_age = current_age + n  (tyres keep aging)
compound = same compound
→ model predicts degradation delta
→ add baseline_pace + fuel correction = predicted lap time
```

**Scenario B — Pit Now (Undercut):**
```
tyre_age = 1 + n  (fresh tyres!)
compound = next_compound (e.g., HARD)
→ model predicts (much lower) degradation delta
→ add baseline_pace + fuel correction = predicted lap time
→ BUT lap 0 costs pit_loss_delta extra seconds (pit lane time)
```

**Net Undercut Advantage:**
```python
lap_advantage = stay_out_time - pit_now_time  # per lap
net_undercut_advantage = cumulative_sum(lap_advantage)  # compounding gain
```

**Decision logic:**
- `window_open = True` if cumulative gain > 1.5s after just 3 laps (immediate and compounding)
- `crossover_lap` = first lap where the fresh tyre is faster than the old one

---

### 🔵 `data_collector.py` — Historical Data Collection

Loops through seasons (2018-2024) and multiple races, runs the full pipeline on each, and saves a combined CSV at `data/historical_laps.csv`. This CSV is used by `train_global_model.py`.

---

## 📊 The Three Charts Explained

### Chart 1 — Tyre Degradation (Raw Pace)
`px.scatter` — Raw lap times for Driver A vs Driver B, coloured by driver, shaped by tyre compound. The Y-axis is **reversed** (lower lap time = faster = higher on chart). Lets you visually see when tyres start degrading.

### Chart 2 — Gap Evolution
Merges both drivers' lap data by `LapNumber`, computes `Time_A - Time_B` per lap. Shows how the gap between the two drivers changes throughout the race. A closing gap = the driver behind is going faster / has fresher tyres.

### Chart 3 — Net Undercut Advantage
Plots `net_undercut_advantage` from the simulation (the cumulative seconds gained if you pit now). The dashed vertical line at `crossover_lap` shows exactly when the fresh tyre becomes faster. If the line crosses 0 and stays positive, pitting is worth it.

---

## 🧠 Key Libraries — What Each One Does

| Library | Role |
|---|---|
| `streamlit` | Web app framework — widgets, layout, caching, state |
| `fastf1` | Official F1 telemetry API — sessions, laps, weather, timing |
| `pandas` | DataFrames for all data manipulation |
| `xgboost` | Gradient boosted tree model for tyre degradation |
| `scikit-learn` | `train_test_split`, `RandomForestRegressor`, R² scoring |
| `plotly` | Interactive charts (`px.scatter`, `go.Figure`) |
| `joblib` | Serialising/deserialising the trained ML model to disk |

---

## ❓ Interview Q&A Prep

### "Walk me through what happens when I click 'Load Race'."
1. `load_race_data()` calls FastF1 to download the session (cached locally in `f1_cache/`)
2. `process_backend_pipeline()` runs: red flag check → clean laps → merge weather → feature engineering → ML prep
3. The UI renders the matchup preview, then the three charts
4. For the undercut sim, the pre-trained global model is called to predict lap times for 15 laps ahead

### "Why XGBoost instead of Random Forest?"
XGBoost is faster, more accurate, supports GPU training, and handles categorical features natively (`Track`, `Driver`) without one-hot encoding. Random Forest was a prototype; XGBoost is production.

### "What's `lap_time_delta` and why is it the target?"
It's the tyre degradation signal — how many seconds slower is this lap compared to fresh rubber? Predicting this instead of raw lap time makes the model generalise across drivers and tracks (different baseline speeds).

### "Why do you filter red flag races?"
Red flags cause a race reset — fresh tyres are mandated, all track strategy resets. The tyre life data becomes discontinuous and would inject noise/incorrect patterns into the degradation model.

### "What does the undercut simulation actually do?"
It's a Monte Carlo-style forward projection. For 15 laps, it asks the model: *"If this driver stays on old tyres vs pits for fresh ones, what's the predicted lap time each lap?"* The difference, summed up, tells you whether pitting now is a net gain or loss.

### "What is `@st.cache_data` doing?"
Streamlit reruns the entire script on every interaction. Without caching, it would re-download F1 data on every button click. `@st.cache_data` memoizes the return value — if called with the same arguments again, it returns the cached result instantly.

### "How do you handle weather in the model?"
`TrackTemp` (track temperature in °C) is a direct feature in the model. Hotter track = more tyre stress = faster degradation. It's merged into the lap data using `pd.merge_asof()` — a time-based nearest-key join.

### "What's baseline_pace?"
The fuel-corrected lap time on the freshest tyre of a driver's stint. It normalises for driver pace and car speed, so the model learns degradation patterns rather than pace differences between drivers/cars.

---

## 🗺️ Recommended Study Order

1. **[`data_pipeline.py`](file:///c:/f1/data_pipeline.py)** — Start here. Understand the raw data shape.
2. **[`feature_engineering.py`](file:///c:/f1/feature_engineering.py)** — Understand each feature and *why* it exists.
3. **[`ml_model.py`](file:///c:/f1/ml_model.py)** — Understand what `lap_time_delta` is and how the target is constructed.
4. **[`train_global_model.py`](file:///c:/f1/train_global_model.py)** — Understand XGBoost config choices.
5. **[`simulation.py`](file:///c:/f1/simulation.py)** — Understand the undercut algorithm step-by-step.
6. **[`app.py`](file:///c:/f1/app.py)** — Understand how everything plugs together in Streamlit.
7. **[`ui/`](file:///c:/f1/ui/)** — Last, understand the theming and chart styling.

---

> [!TIP]
> The single most impressive thing you can say in an interview: **"The target variable is `lap_time_delta` — the degradation delta relative to a fresh-tyre baseline — which lets the model generalise across drivers and circuits without learning car speed."** That shows you understand *why* the ML is designed the way it is.

> [!NOTE]
> You can run `python simulation.py` directly (it has a `__main__` block) to see a sample simulation output for Verstappen at Bahrain 2023 without opening the whole app.
