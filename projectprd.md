# F1 Strategy Engineer — Product Requirements Document

*Version:* 1.0  
*Status:* Draft  
*Platform:* Streamlit (Web App)

---

## 1. Executive Summary

F1 Strategy Engineer is a Python-based dashboard that predicts the exact moment a driver becomes vulnerable to an undercut from a rival. It ingests real F1 race data, models tyre degradation using machine learning, and calculates a pit window — the critical laps where a driver must pit to maintain track position.

The product simulates two scenarios every lap — "stay out" vs "pit now" — and identifies the crossover point where pitting becomes strategically necessary.

---

## 2. The Problem

Tyre degradation in F1 is non-linear. As tyres age, lap times drop — but the rate varies by compound, fuel load, track temperature, and track evolution. Human intuition fails to account for all these variables simultaneously.

The undercut is one of the most consequential strategic decisions in a race. Get it wrong by even one lap and track position is lost. This tool calculates the math that pit walls do manually, and makes it accessible and visual.

---

## 3. Platform

*Streamlit* — chosen because:

- Fastest path from data pipeline to interactive UI
- Native support for Matplotlib and Plotly
- One-command deployment on Streamlit Cloud for public demo URL
- Allows full focus on data engineering, ML, and simulation logic

---

## 4. Core Features

### 4.1 Race Selector
- Dropdowns for season and race (dynamically populated from FastF1)
- Races with red flag events are automatically excluded from the dropdown to avoid edge cases in stint data
- Data fetched via FastF1 on selection, cached locally after first fetch
- Loading spinner displayed during fetch

### 4.2 Driver Comparison Selector
- Two dropdowns — Driver A and Driver B
- Populated from the selected race's finisher list
- All simulation and visualisation is scoped to this pairing

### 4.3 Tyre Degradation Chart
- Actual lap times per stint for both selected drivers
- In-laps, out-laps, safety car laps, and VSC laps filtered out before display
- Compound colour-coded (Soft = red, Medium = yellow, Hard = white/grey)
- Shows the raw data the ML model is trained on — gives the user context before the prediction

### 4.4 Gap Evolution Chart
- Actual on-track gap between Driver A and Driver B across the race
- Pit stop laps marked on the chart
- Contextualises the undercut simulation — shows whether the gap is closing or stable

### 4.5 Undercut Window Simulation (Core Feature)
The main output of the product.

*Simulation logic:*
1. Every lap, the model runs two predictions forward 15 laps:
   - *Stay Out:* predicted lap times if Driver A continues on current tyres
   - *Pit Now:* predicted lap times if Driver A pits for fresh tyres, with actual pit loss added to lap 1 of the new stint
2. The crossover lap — where the "Pit Now" curve drops below the "Stay Out" curve — is the undercut window
3. The dashboard highlights this lap with a vertical marker and a plain-English label: "Undercut window opens lap X — Driver B can undercut from this point"

*Pit loss delta:* calculated automatically from FastF1's actual pit stop timing data for the selected circuit, not hardcoded. Uses the median pit loss across all pit stops in the session.

### 4.6 AI Insight Panel (Optional Enhancement)
A brief plain-English summary generated via Claude API after simulation completes. Covers: which driver holds the strategic advantage, when the pit window opens, and what the model predicts will happen if neither driver changes strategy. This is a stretch goal — core product works without it.

---

## 5. Data Pipeline

### 5.1 Data Ingestion
- Source: FastF1 library
- Data points: lap times, sector times, tyre compound, tyre life, track status, pit stop timing, weather data (track temperature)
- Cache: FastF1 built-in cache enabled to avoid repeat API calls

### 5.2 Data Cleaning
Filters applied before any modelling:

| Filter | Reason |
|---|---|
| In-laps and out-laps | Artificially slow — distort degradation curve |
| Safety car laps | Controlled pace — not representative of racing pace |
| VSC laps | Same as safety car |
| Laps with track status flags | Any non-green flag laps removed |
| Red flagged races | Excluded from race selector entirely |

Result: a clean dataset of pure racing pace laps only.

---

## 6. Feature Engineering

Five features fed into the ML model:

| Feature | Description | Source |
|---|---|---|
| tyre_age | Number of laps on current set | FastF1 lap data |
| compound_encoded | Soft=1, Medium=2, Hard=3 | FastF1 compound data |
| fuel_corrected_time | Lap time adjusted for fuel mass (burning ~1.7kg/lap) | Calculated |
| track_evolution | Median lap time of all drivers per lap — proxy for rubber buildup | Calculated from session data |
| track_temp | Track surface temperature per lap | FastF1 weather data |

*Fuel correction formula:*

fuel_mass = 110 - (1.7 × lap_number)
fuel_correction = fuel_mass × 0.03  # ~0.03s per kg
corrected_lap_time = lap_time - fuel_correction


*Track evolution:*
python
track_evo = df.groupby('LapNumber')['LapTime'].median()
df['track_evolution'] = df['LapNumber'].map(track_evo)


---

## 7. Machine Learning

### Model
*Random Forest Regressor* (scikit-learn)

### Why Random Forest
- Tyre degradation is non-linear — compounds have "cliff" points where performance drops sharply
- Handles mixed feature types without scaling
- Works well on moderate dataset sizes (400–600 clean laps per race)
- Feature importance output useful for validation

### Training approach
- One global model trained on all drivers in the selected race
- Per-driver models rejected due to insufficient data per driver (~20–30 points)
- Driver-specific predictions achieved by feeding each driver's current feature values into the shared model
- Train/test split: 80/20
- Target variable: lap_time_delta — difference from the driver's fresh-tyre baseline for that stint

### Hyperparameters (defaults, tunable later)
- n_estimators: 100
- max_depth: 10
- random_state: 42

---

## 8. Undercut Simulation Logic


For each lap L in the race:

  Stay Out curve:
    For lap L to L+15:
      predicted_time = model.predict(tyre_age + n, current_compound, fuel_load - 1.7n, ...)

  Pit Now curve:
    lap_1 = model.predict(tyre_age=1, fresh_compound, ...) + pit_loss_delta
    For lap L+1 to L+15:
      predicted_time = model.predict(tyre_age + n, fresh_compound, ...)

  Crossover lap = first lap where Pit Now curve < Stay Out curve

  If crossover lap <= L+3:
    Flag as UNDERCUT WINDOW OPEN


*Pit loss delta:* median of all actual pit stop time losses in the session, calculated from FastF1 pit data.

---

## 9. Dashboard Layout


┌─────────────────────────────────────────┐
│  Sidebar                                │
│  - Season selector                      │
│  - Race selector                        │
│  - Driver A selector                    │
│  - Driver B selector                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Main Panel                             │
│                                         │
│  [Tyre Degradation Chart]               │
│  Actual lap times per stint, both       │
│  drivers, compound colour coded         │
│                                         │
│  [Gap Evolution Chart]                  │
│  On-track gap across the race           │
│                                         │
│  [Undercut Window Simulation]           │
│  Stay Out vs Pit Now curves             │
│  Crossover lap highlighted              │
│  Plain-English verdict below chart      │
└─────────────────────────────────────────┘


---

## 10. Tech Stack

| Layer | Technology |
|---|---|
| Data source | FastF1 |
| Data processing | Pandas, NumPy |
| ML model | scikit-learn (RandomForestRegressor) |
| Visualisation | Matplotlib, Plotly |
| UI | Streamlit |
| Caching | FastF1 cache + st.cache_data |
| Deployment | Streamlit Cloud |
| AI insights (stretch) | Claude API |

---

## 11. Scope Boundaries

### In scope
- Race sessions only (not qualifying or practice)
- Two-driver comparison
- 2018–2024 seasons
- Dry races only (wet weather tyre behaviour is significantly different)
- Races without red flags

### Out of scope (v1)
- Live race data
- More than two drivers simultaneously
- Wet weather compounds (Intermediate, Wet)
- Championship-level multi-race analysis
- Mobile optimisation

---

## 12. Development Phases

### Phase 1 — Data Pipeline
- FastF1 integration and caching
- Race selector with red flag filtering
- Data cleaning pipeline (remove SC, VSC, in/out laps)

### Phase 2 — Feature Engineering
- Fuel correction calculation
- Track evolution proxy
- Compound encoding
- Tyre life tracking

### Phase 3 — ML Model
- Random Forest training on clean race data
- Train/test split
- Prediction function for future lap times

### Phase 4 — Simulation Logic
- Stay Out vs Pit Now curve generation
- Pit loss delta calculation from actual session data
- Crossover lap detection
- Undercut window flagging

### Phase 5 — Dashboard
- Streamlit UI layout
- Tyre degradation chart
- Gap evolution chart
- Undercut window simulation chart
- Plain-English verdict output

### Phase 6 — Deploy & Polish
- Streamlit Cloud deployment
- Error handling for edge cases
- Performance optimisation for slow FastF1 fetches

### Phase 7 — Stretch Goals
- Model validation display (MAE, predicted vs actual chart)
- Claude API insight panel

---

## 13. Interview Talking Points

This project is designed to generate strong interview conversation:

- *Why Random Forest?* Non-linear degradation, handles mixed features, interpretable via feature importance
- *Why one global model?* Per-driver models have insufficient training data — 20 points is not enough for a reliable Random Forest
- *How did you calculate pit loss?* From actual FastF1 pit timing data, not hardcoded — circuit-specific and race-specific
- *What is track evolution?* Rubber buildup makes the track faster over a stint — proxied using median lap time of all drivers per lap
- *What would you improve?* Per-compound models, live data integration, wet weather handling, model validation display

---

F1 Strategy Engineer — a portfolio project demonstrating real-world data engineering, physics-based feature engineering, ML regression, and interactive visualisation on live F1 data.