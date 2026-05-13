# F1 Strategy Engineer — How It Works (Deep Dive)

This document is a highly detailed, module-by-module breakdown of how your application actually processes data under the hood. You can use this to explain the deep technical mechanics of your project.

---

## 1. App Entry Point (`app.py`)
The application is a Streamlit script. When a user visits the page, Streamlit executes the file top-to-bottom.
It uses `@st.cache_data` and `@st.cache_resource` extensively to store API responses and trained machine learning models in the server's RAM.
When a user clicks "Load Race Data", it triggers the `process_backend_pipeline` function, which acts as the orchestrator for the following four modules.

---

## 2. Data Pipeline (`data_pipeline.py`)
This file is responsible for grabbing raw data and turning it into clean, pure racing data.

1. **Red Flag Check:** The code first checks track status. If a red flag (`TrackStatus == '5'`) is detected anywhere in the session, the app immediately throws an error and stops. Red flags reset tyre temperatures and allow free tyre changes, which ruins continuous degradation curves.
2. **Cleaning Laps:** It takes the raw lap telemetry and strips away anything that isn't a car driving at maximum pace.
   - It removes `In-laps` (coming into pits) and `Out-laps` (leaving pits).
   - It filters out any lap where the `TrackStatus` isn't `1` (Green Flag). This removes Safety Cars, VSCs, and local yellows, ensuring the lap times are representative of the car's actual pace, not a delta time enforced by race control.
3. **Weather Merge:** It takes ambient `TrackTemp` data from FastF1's weather feed and maps it to the closest timestamp for every single lap.

---

## 3. Feature Engineering (`feature_engineering.py`)
Machine Learning models need clean, mathematically comparable numbers. This module transforms the raw FastF1 metrics into ML-ready features.

1. **Tyre Age:** Copies FastF1's `TyreLife` tracker.
2. **Compound Encoding:** The model can't understand the strings "SOFT" or "HARD". This module maps them to integers: `SOFT=1`, `MEDIUM=2`, `HARD=3`.
3. **Fuel Correction:** 
   - A car burns ~1.7kg of fuel per lap.
   - Lighter cars are faster (~0.03 seconds per kg).
   - The script calculates the exact fuel weight for every lap and subtracts that time advantage from the actual lap time (`fuel_corrected_time = lap_time - fuel_correction`). This leaves a "pure pace" number where the only variable causing the car to slow down is the wearing tyres.
4. **Track Evolution:** 
   - A track gets faster as rubber is laid down. The script groups the entire dataset by `LapNumber` and calculates the **median** lap time of the entire grid for that lap. This creates a proxy metric that tracks how the surface grip is changing.

---

## 4. Machine Learning Module (`ml_model.py`)
This module creates and trains the Random Forest model dynamically based on the specific race selected.

1. **Baseline Calculation:** The script calculates the fastest "fresh tyre" lap (usually lap 2 or 3 of a stint) for every driver on every compound. 
2. **Target Variable Setup:** It calculates a `lap_time_delta` — the difference between the car's *current* lap time and its fresh-tyre baseline. **This delta is what the model actually learns to predict.**
3. **Training:** It sets up an 80/20 train/test split. It trains a `RandomForestRegressor` (100 estimators, max depth 10) to learn how `tyre_age`, `compound_encoded`, `track_temp`, and `track_evolution` affect the `lap_time_delta`. 
4. *Result:* The model is now capable of answering the question: *"If a driver is on 15-lap old mediums and the track temp is 35°C, how much slower will they be compared to their baseline?"*

---

## 5. Simulation Engine (`simulation.py`)
This is the core predictive feature of the app. Once the user clicks "Simulate", this module takes over.

1. **Pit Loss Calculation:** It scans the session data to find every time a car entered the pits (`PitInTime`) and exited (`PitOutTime`). It subtracts the two to find the pit lane transit time, and takes the median of all stops. This is the `pit_loss_delta` (usually around 24-26 seconds).
2. **The 15-Lap Forecast Loop:** The engine loops 15 times (simulating 15 laps into the future).
   - **Stay Out Path:** It queries the model: "What is the pace if tyre age increases by 1 each lap?" It takes the model's predicted degradation delta, adds it to the driver's baseline pace, and *un-corrects* for fuel to give a true predicted lap time.
   - **Pit Now Path:** It queries the model assuming the driver puts on fresh tyres (`tyre_age = 1`) of the user's chosen compound. For the very first lap, it mathematically adds the 25-second `pit_loss_delta` penalty. For the next 14 laps, it predicts the pace of the fresh tyres.
3. **Crossover Detection:** The engine compares the "Stay Out" lap times to the "Pit Now" lap times. The exact lap where the "Pit Now" time drops below the "Stay Out" time is flagged as the **Crossover Lap**. If this crossover happens within 3 laps, the app declares the "Undercut Window" open.
