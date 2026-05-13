# F1 Strategy Engineer — Interview Q&A Prep

This document contains anticipated, in-depth technical questions an interviewer might ask about your **F1 Strategy Engineer** project, along with strong, detailed answers that align with your codebase and PRD.

---

### 1. What exactly is an "undercut" in F1, and how does your dashboard predict it?
**Answer:**
An undercut is a strategy where a trailing driver pits earlier than the leading driver for fresh tyres. Because fresh tyres offer significantly more grip and faster lap times, the trailing driver uses that out-lap pace advantage to jump ahead of the leader when the leader eventually pits. 

My dashboard predicts this by simulating 15 laps into the future for two scenarios: 
1. **Stay Out:** The driver continues on their current, degrading tyres.
2. **Pit Now:** The driver pits immediately (incurring a ~25s pit loss penalty on lap 1) and emerges on fresh tyres.

The model plots predicted lap times for both scenarios. The exact moment the "Pit Now" predicted lap times drop below the "Stay Out" lap times is the **crossover point**. When this crossover occurs within a 3-lap horizon, the system flags the **Undercut Window as Open**.

---

### 2. Why did you use a Random Forest Regressor? Why not Deep Learning or a simple Linear Regression?
**Answer:**
I chose Random Forest because tyre degradation in Formula 1 is highly non-linear. 
- **Against Linear Regression:** Tyres don't degrade in a straight line; they have "cliff" points where performance drops off drastically. Linear regression cannot accurately capture this sharp drop-off.
- **Against Deep Learning:** Neural networks are data-hungry. Even combining all drivers in a single race yields only about 400-600 clean racing laps. Deep learning models would likely overfit on such a small dataset. 
- **Why Random Forest:** It handles non-linear relationships natively, doesn't require strict feature scaling for mixed data types (like encoded compounds vs. track temperature), and it provides feature importance metrics, which allowed me to validate that tyre age and fuel load were actually driving the predictions as expected.

---

### 3. I see you engineered some physics-based features. Can you explain the fuel mass correction and track evolution calculations?
**Answer:**
Yes, to get the model to focus purely on tyre degradation, I had to isolate the "noise" created by changing fuel loads and track conditions.

**Fuel Mass Correction:**
F1 cars start with 110kg of fuel and burn roughly 1.7kg per lap. Because a lighter car is faster (about 0.03 seconds faster per kg of fuel lost), lap times naturally improve as the race goes on, which mathematically masks tyre degradation. I calculated the fuel mass for any given lap (`110 - (1.7 * lap_number)`) and subtracted the fuel time advantage (`mass * 0.03s`) from the raw lap time.

**Track Evolution:**
As F1 cars drive, they lay down rubber, making the track surface inherently faster. I proxied this "track evolution" by calculating the median lap time of *all* drivers on the track for every single lap. By feeding this median pace into the model, it learned how the track grip was changing independently of a specific driver's tyres.

---

### 4. You filtered out safety cars and out-laps. What was the hardest part about cleaning the telemetry data?
**Answer:**
The hardest part was ensuring the model only trained on **pure racing pace**. If the model sees a lap that is 15 seconds slower than normal, it might assume the tyres have hit a cliff, when in reality, there was a yellow flag.

Using Pandas and the FastF1 library, I aggressively filtered the dataset. I removed:
- In-laps and out-laps (which are artificially slow due to pit lane entry/exit).
- Safety Car (SC) and Virtual Safety Car (VSC) laps.
- Any lap with a non-green track status flag.
- I entirely excluded races that were red-flagged, because the mandatory stoppage completely resets tyre temperatures and strategies, which breaks the continuity of degradation curves.

---

### 5. In your simulation, how did you calculate the time lost during a pit stop?
**Answer:**
Instead of hardcoding a generic 25-second penalty, I calculated a **dynamic, circuit-specific pit loss delta** directly from the race telemetry. 

I isolated all pit stops in the session by matching a driver's "In Lap" (when they entered the pit lane) to their subsequent "Out Lap" (when they exited). By subtracting the `PitInTime` from the `PitOutTime`, I calculated the exact duration each car spent in the pit lane. I then took the **median** of all these times across the session. This is far more robust because it automatically accounts for different pit lane lengths (like Silverstone vs. Monaco) and varying pit lane speed limits.

---

### 6. Why did you train one global model per race instead of per-driver models?
**Answer:**
It came down to sample size. To train a reliable machine learning model, you need sufficient data points. A single driver in a typical race might only have 20 to 30 clean, representative racing laps across different compounds. Trying to train a Random Forest on 20 rows of data would result in severe overfitting.

By training a **single global model** on the clean laps of *all* drivers (creating a dataset of ~500 rows), the model learned the universal physics of how the Soft, Medium, and Hard compounds degrade on that specific day and track temperature. I then achieved driver-specific predictions by feeding that driver's current state (tyre age, compound, baseline pace) into the global model during the simulation phase.

---

### 7. What are the current limitations of your simulation? If you had another month to work on it, what would you add?
**Answer:**
The current simulation calculates the mathematical undercut window in "free air." It does not account for track position traffic. If the simulation tells a driver to pit, but they emerge from the pit lane stuck behind a slower car, the undercut will fail because they can't utilize the pace of their fresh tyres.

If I had another month, I would add a **Traffic Forecasting Module**. I would calculate exactly where the driver would re-enter the track (current gap minus pit loss delta) and analyze the pace of the cars in that "re-entry window" to determine if they would be released into clean air or dirty air. I would also expand the model to handle wet-weather racing.
