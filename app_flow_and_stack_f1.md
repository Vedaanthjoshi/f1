# F1 Strategy Engineer — Architecture, Tech Stack, & App Flow

This document prepares you for interview questions focused on *how* the application is built from a high-level architectural standpoint, why you chose your specific technologies, and how the user request flows through the system.

---

## 1. Application Flow (How data moves through the app)

When a user interacts with the dashboard, the data flows linearly through several stages:

1. **User Input:** The user selects a Season and a Grand Prix from the Streamlit sidebar.
2. **Data Ingestion & Caching:** The app queries the `FastF1` API. If this is the first time the race is selected, it downloads Megabytes of raw telemetry and weather data. It then caches this locally using FastF1's built-in SQLite cache and Streamlit's `@st.cache_resource` so subsequent loads are instant.
3. **The Data Pipeline (`data_pipeline.py`):** 
   - Checks for red flags (and aborts if true).
   - Strips out invalid laps (safety cars, VSC, in-laps, out-laps).
   - Merges the lap data with ambient weather data based on timestamps.
4. **Feature Engineering (`feature_engineering.py`):** 
   - Calculates proxy variables (track evolution, fuel mass correction) and encodes categorical data (tyre compounds).
5. **Machine Learning (`ml_model.py`):** 
   - Prepares the training dataframe.
   - Instantiates and trains a Random Forest Regressor on the fly. The target variable is `lap_time_delta` (degradation pace).
6. **Simulation Prep (`simulation.py`):** 
   - Automatically calculates the actual pit loss time for that specific circuit.
7. **UI Rendering (`app.py`):** 
   - The user selects two drivers to compare. The app renders interactive Plotly charts showing actual historical pace and gap evolution.
   - When the user sets a "Simulate from Lap X" parameter, the backend isolates that driver's state on that exact lap and queries the trained Random Forest model to predict the next 15 laps, returning the Stay Out vs. Pit Now visual prediction.

---

## 2. Tech Stack & "Why?"

Be prepared to defend your technology choices. Here is your rationale:

### **Why Python?**
Python is the undisputed industry standard for Data Science and Machine Learning. The ecosystem for data manipulation (Pandas), mathematical modeling (Scikit-Learn), and API wrappers (FastF1) is unmatched. Doing this in JavaScript/Node would have required writing custom statistical models from scratch.

### **Why Streamlit? (Instead of React / Next.js / Vue)**
The goal of this project was data engineering and predictive modeling, not frontend web development. Streamlit is the fastest path from a Python script to an interactive web UI. Building a separate React frontend would have required standing up a dedicated FastAPI/Flask backend, managing REST endpoints, and dealing with CORS, which adds unnecessary overhead for a purely analytical dashboard.

### **Why Pandas & NumPy?**
F1 telemetry is essentially highly structured, dense time-series data. Pandas allows for vectorized operations (like applying fuel correction formulas to 1,000 rows instantly) which is significantly faster and more memory-efficient than using standard Python `for` loops.

### **Why Plotly? (Instead of Matplotlib/Seaborn)**
Matplotlib is great for static reports, but a strategy dashboard requires interactivity. Plotly allows users to hover over specific data points on the scatter plot to see exact tyre ages, lap times, and compounds. It also natively handles resizing and dark mode scaling within Streamlit much better than Matplotlib.

### **Why FastF1?**
It is the most reliable, community-tested Python library for interfacing with the F1 Ergast API and F1 live timing systems. It handles the extremely complex logic of syncing car telemetry packets with official track timing loops.

---

## 3. Potential Interview Questions

**Q: Your app downloads a massive amount of telemetry data. How do you prevent the UI from freezing or crashing?**
**Answer:** I heavily utilized caching. I used Streamlit's `@st.cache_resource` decorator on the data loading functions. When a user selects "Bahrain 2023", the backend downloads the data once and stores it in memory. If they switch to "Monaco", then back to "Bahrain", it loads instantly from memory rather than hitting the network again. I also enabled FastF1's on-disk SQLite cache to save network requests between server reboots.

**Q: What was the biggest architectural challenge you faced?**
**Answer:** State management and pipeline order of operations. Because I am training a machine learning model *on the fly* inside a web app based on user dropdown selections, I had to be very careful to structure my backend modules linearly. I separated the pipeline into distinct Python files (`data_pipeline.py`, `feature_engineering.py`, `ml_model.py`) so I could test the logic via CLI independently of the Streamlit UI, ensuring the web app was just a thin presentation layer over robust backend logic.

**Q: If you had to scale this app for 10,000 concurrent users, what would you change?**
**Answer:** Streamlit is great for prototypes but struggles with heavy concurrent state management. To scale, I would decouple the frontend and backend. I would build the frontend in Next.js/React. I would move the Python backend to a FastAPI service, and instead of training the model on the fly for every user session, I would pre-train the Random Forest models for all historical races asynchronously using Airflow or Celery, save the models to an S3 bucket (or as `.pkl` files), and just load the pre-trained weights when a user requests an API inference.
