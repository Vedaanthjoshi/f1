# F1 Strategy Engineer: In-Depth Project Analysis

## 1. Executive Overview

The F1 Strategy Engineer is a comprehensive Python-based web dashboard designed to predict the optimal moment a Formula 1 driver should pit to successfully execute an "undercut" against a rival. By ingesting real F1 race telemetry via the FastF1 API, modeling non-linear tyre degradation using machine learning (XGBoost), and simulating a pit window, the tool provides pit-wall level strategic insights.

The product calculates two theoretical scenarios for every lap—"stay out" on current tyres vs. "pit now" for fresh ones—and pinpoints the crossover lap where the strategic advantage tips in favor of pitting. 

---

## 2. Platform & Architecture

The application is built on **Streamlit**, serving as the front-end interface, due to its rapid deployment capabilities from data pipeline to interactive UI, and its native support for visualization libraries like Matplotlib and Plotly.

### The Big Picture (Architecture Flow)
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
    │                                             │
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

## 3. Detailed Component Breakdown

### 3.1 Data Pipeline (`data_pipeline.py`)

The data ingestion relies on the FastF1 library. FastF1's built-in cache is utilized to prevent redundant API calls. 
Data points collected include lap times, sector times, tyre compound, tyre life, track status, pit stop timing, and weather data (specifically track temperature).

The data is rigorously cleaned before feature engineering:
- **Red Flag Races:** Entirely excluded because a race restart mandates fresh tyres, creating a discontinuity in tyre data that corrupts the degradation model. (`is_red_flag_race(session)`)
- **In-Laps and Out-Laps:** Filtered out as they are artificially slow due to pit lane travel.
- **Safety Car (SC) & Virtual Safety Car (VSC):** Removed because they enforce controlled, non-racing pace. Only green flag laps (`TrackStatus == '1'`) are preserved.
- **Time Conversion:** LapTime (timedelta) is converted to `LapTime_s` (float seconds) for mathematical modeling.
- **Weather Integration:** `pd.merge_asof()` (a time-series nearest-key join) attaches the most recent weather reading (like TrackTemp) to each lap, as higher temperatures accelerate degradation. (`merge_weather_data`)

### 3.2 Feature Engineering (`feature_engineering.py`)

Clean laps are transformed into Machine Learning-ready features to accurately represent race physics:

1. **Tyre Age (`tyre_age`):** Mapped directly from FastF1's `TyreLife` column.
2. **Compound Encoded (`compound_encoded`):** Categorical mapping (SOFT=1, MEDIUM=2, HARD=3) for numeric model input.
3. **Fuel Corrected Time (`fuel_corrected_time`):** 
   - A crucial adjustment. An F1 car burns approximately 1.7 kg of fuel per lap (starting from ~110 kg). Without correcting for the lightening car, the model would misattribute the resulting speed increase, hiding true tyre degradation.
   - *Formula:* `LapTime_s - (110 - 1.7 × lap) × 0.03` (approx 0.03s gained per kg).
4. **Track Evolution (`track_evolution`):** 
   - Tracks get faster ("rubber in") as the race progresses. This is modeled by taking the median lap time per lap number across all drivers.

### 3.3 Machine Learning Preparation (`ml_model.py`)

The ML target variable is meticulously constructed to isolate tyre degradation:
- **Baseline Pace:** The fuel-corrected lap time on the freshest tyre (minimum `tyre_age`) of a driver's stint.
- **Target Variable (`lap_time_delta`):** `fuel_corrected_time − baseline_pace`
  
By predicting `lap_time_delta` (how many seconds slower a lap is compared to fresh rubber), the model generalizes across different drivers and car paces, focusing solely on tyre wear.

### 3.4 Offline Training (`train_global_model.py` & `data_collector.py`)

Instead of per-driver models (which suffer from insufficient data points per race), a single **Global Model** is trained on historical data (2018-2024) across multiple tracks. 
- **Model Choice:** `XGBRegressor` (XGBoost) replaced earlier RandomForest prototypes due to its speed, accuracy, GPU support, and native handling of categorical features (`Track`, `Driver`).
- **Hyperparameters:** `n_estimators=300`, `max_depth=8`, `learning_rate=0.05`.
- **Artifact:** Saved to `models/global_degradation_model.joblib`.

### 3.5 Undercut Simulation Logic (`simulation.py`)

This is the core predictive engine. For each lap (L), it simulates 15 laps into the future under two scenarios:

**Scenario A: Stay Out**
The driver continues on current tyres. 
- `tyre_age = current_age + n`
- Model predicts degradation delta, added to `baseline_pace` and fuel correction.

**Scenario B: Pit Now (The Undercut)**
The driver pits for fresh tyres.
- `tyre_age = 1 + n`
- Model predicts the new (lower) degradation delta, added to baseline and fuel correction.
- *Crucially*, the first lap (Lap 0) incurs a `pit_loss_delta` (calculated dynamically as the median pit lane time for that specific session, usually 24-26 seconds).

**Decision Metric:** 
The simulation calculates the `net_undercut_advantage` (cumulative sum of the lap advantage). The **crossover lap** is identified as the first lap where the fresh tyre becomes inherently faster. If the cumulative gain exceeds 1.5s after 3 laps, the undercut window is flagged as OPEN.

---

## 4. The Dashboard & UI (`app.py` & `ui/`)

The Streamlit interface acts as the brain wiring these components together. It features:
- **Race/Driver Selectors:** Sidebar dropdowns dynamically populated (excluding red-flag races).
- **State Management & Caching:** Extensive use of `@st.cache_data` (for DataFrames) and `@st.cache_resource` (for the ML model) ensures the app doesn't re-download or re-train on every UI interaction. `st.session_state` persists user selections.

### Visualizations
1. **Tyre Degradation Chart:** A scatter plot (`px.scatter`) of raw lap times (Y-axis reversed) for both drivers, color-coded by compound, showing raw pace and degradation onset.
2. **Gap Evolution Chart:** Plots `Time_A - Time_B` over the race to visualize the actual on-track interval closing or opening.
3. **Net Undercut Advantage Chart:** Plots the simulation's `net_undercut_advantage`. A dashed vertical line denotes the `crossover_lap`. 

## 5. Summary Tech Stack

- **Data Source:** FastF1
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** XGBoost (`XGBRegressor`), scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Web Framework:** Streamlit
- **Persistence:** joblib (model saving)

This project fundamentally bridges complex physics-based feature engineering, robust time-series data cleaning, and machine learning to deliver a highly interactive, real-world analytical tool.
