# F1 Strategy Engineer — Current Product Requirements Document (PRD)

*Version:* 2.0 (Current State)
*Status:* Active Development
*Platform:* Streamlit Web Application

---

## 1. Executive Summary

F1 Strategy Engineer is an interactive, Python-based analytics dashboard designed to predict the exact moment a Formula 1 driver becomes vulnerable to an "undercut" from a rival. The application ingests real F1 race telemetry using the FastF1 library, models tyre degradation using machine learning (XGBoost), and calculates a pit window — the critical laps where a driver must pit to maintain or gain track position. 

The product evaluates a 15-lap simulation horizon comparing two scenarios — "stay out" vs "pit now" — and identifies the exact crossover point where pitting yields a strategic pace advantage.

---

## 2. Core Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Data Source** | FastF1 Library |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | XGBoost (GPU-accelerated XGBRegressor) |
| **Visualisation** | Plotly (Express & Graph Objects) |
| **Frontend/UI** | Streamlit |
| **Caching** | FastF1 local cache + `st.cache_data` & `st.cache_resource` |

---

## 3. Product Features & UI Components

### 3.1 Race Setup & Selection
- **Season & Race Dropdowns**: Users can select seasons from 2018 to 2025 and choose a Grand Prix. The schedule is dynamically populated via FastF1.
- **Red Flag Handling**: Races that experienced red flags are explicitly detected and filtered out to prevent distorted degradation curves.
- **State Synchronization**: Race selections are synchronized between the main layout and the sidebar for a seamless user experience.

### 3.2 Driver Comparison (Matchup)
- **Driver A (Target) vs Driver B (Attacker)**: Selectable from the pool of finishers for the chosen race. 
- All telemetry, gap calculations, and undercut simulations are directly scoped to this selected matchup.
- The UI displays custom styling and team colors mapped to each driver for visual clarity.

### 3.3 Visual Analytics
1. **Tyre Degradation Chart (Scatter Plot)**: 
   - Displays actual lap times for both selected drivers across all stints.
   - Points are color-coded by tyre compound (Soft, Medium, Hard) and styled by driver/team colors.
   - Outliers such as in-laps, out-laps, Safety Car, and Virtual Safety Car (VSC) laps are filtered out to show raw racing pace.
   
2. **Gap Evolution Chart (Line Chart)**:
   - Tracks the dynamic time gap (in seconds) between Driver A and Driver B throughout the race.
   - Helps contextualize whether an attacker is closing into the "undercut danger zone."

3. **Undercut Window Simulation (Line Chart)**:
   - **Core Logic**: Evaluates a user-selected lap and target tyre compound.
   - Displays a line chart of **Net Undercut Advantage** (seconds gained or lost per lap).
   - A zero-line indicates the crossover threshold.
   - A vertical marker indicates the **Crossover Lap** (the exact lap where old tyres become slower than fresh tyres).
   - Generates plain-English verdicts (e.g., "UNDERCUT WINDOW OPEN" or "Pace crossover happens on Lap X").

---

## 4. Backend & Data Pipeline

### 4.1 Data Ingestion (`data_pipeline.py` & `data_collector.py`)
- **FastF1 Integration**: Fetches session data, lap timings, sector times, tyre compound history, and weather telemetry.
- **Cleaning Filters**: Removes laps unrepresentative of true pace:
  - In-laps & out-laps
  - Safety Car & VSC laps
  - Non-green flag laps
- Merges lap timing data with track temperature and weather variables.

### 4.2 Feature Engineering (`feature_engineering.py`)
Generates critical derived features required by the ML model:
- `tyre_age`: The number of consecutive laps completed on the current set of tyres.
- `compound_encoded`: Numeric mapping (Soft=1, Medium=2, Hard=3).
- `fuel_corrected_time`: Lap time adjusted for fuel burn (~1.7kg burned per lap, equating to ~0.03s of pace improvement).
- `track_evolution`: The median lap time of all drivers on a given lap, serving as a proxy for rubber-in and track conditions.
- `baseline_pace`: The initial fuel-corrected pace of a driver on fresh tyres at the start of a stint.

### 4.3 Machine Learning (`train_global_model.py`)
- **Model Choice**: A global **XGBoost Regressor** trained on historical race laps. Replaced the earlier Random Forest approach (`ml_model.py`) to support GPU acceleration, better performance, and native categorical feature handling.
- **Features Used**: `tyre_age`, `compound_encoded`, `baseline_pace`, `track_evolution`, `TrackTemp`, `Track` (Categorical), `Driver` (Categorical).
- **Target Variable**: `lap_time_delta` (Current fuel-corrected pace minus the baseline pace of the stint).
- **Output**: Predicts how much slower a driver will be on any given lap compared to their fresh-tyre baseline. The model is saved as `global_degradation_model.joblib`.

### 4.4 Simulation Engine (`simulation.py`)
- **Dynamic Pit Loss**: Calculates the precise pit lane time loss by taking the median of actual pit stops recorded during the selected session (usually ~24-26 seconds depending on the circuit).
- **Stay Out vs Pit Now Forecasting**: 
  - Predicts 15 laps into the future.
  - Combines model-predicted degradation with ongoing fuel mass corrections and track evolution.
  - Computes the lap-by-lap pace difference.
- **Advantage Calculation**: Returns the cumulative net undercut advantage and determines if the 1.5-second strategic threshold is met within 3 laps to declare the window "open".

---

## 5. Scope & Limitations

### Included
- Comprehensive analysis of dry, full-distance Grand Prix sessions (2018–2025).
- Precise 1v1 driver comparisons.
- Advanced fuel correction and track surface evolution proxy metrics.

### Excluded / Limitations
- Wet weather races (Intermediate/Wet tyre behaviors are highly volatile and excluded).
- Races with Red Flags (Filtered out to maintain continuous degradation modeling).
- Live race data streaming (Currently processes post-session or cached historical telemetry).

---

## 6. Development Status 

The project has successfully transitioned from an initial Random Forest baseline to a high-performance **Global XGBoost Architecture**. The data pipeline accurately filters edge cases, the Streamlit UI provides a polished, interactive experience, and the underlying simulation engine successfully normalizes pace for fuel weight and track evolution to output highly accurate strategic forecasts.
