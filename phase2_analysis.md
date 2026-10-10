# Phase 2 Analysis: new_changes.md vs. Current Project

> Grounded in the actual code — `simulation.py`, `train_global_model.py`, `ml_model.py`, `app.py`. No guessing.

---

## What the Current Project Actually Is (Baseline)

Before judging the plan, here is what Phase 1 *actually* does right now in code:

- **ML Model:** A single `XGBRegressor` trained with a point-estimate objective (MSE loss by default in XGBoost). It outputs **one float** — `lap_time_delta` — per lap prediction. No uncertainty. No distribution.
- **Simulation:** `simulate_undercut_window()` in `simulation.py` runs two deterministic for-loops across 15 laps: *Stay Out* (tyre_age keeps incrementing) vs *Pit Now* (tyre_age resets to 1). The "winner" is determined by a single hardcoded threshold: `net_undercut_advantage[2] >= 1.5` — if cumulative advantage after 3 laps is ≥ 1.5s, the window is flagged OPEN.
- **Strategies compared:** Exactly **2** — stay out vs. pit now for one user-chosen compound.
- **Rival modelling:** Zero. Driver B is never fed into `simulation.py`. The simulation is entirely about Driver A's tyres in isolation.
- **Safety Car:** Completely absent from the simulation. SC/VSC laps are filtered *out* of training data (correctly), but not modelled as future events.
- **Evaluation:** R² on an 80/20 split of `historical_laps.csv`. No backtesting against real race outcomes.

That is the honest baseline. Now let's go through each proposed feature.

---

## Feature 1: Probabilistic Tyre Degradation (Quantile Regression)

### What the plan says
Instead of one float output, train the model to output three quantiles: **p5, p50, p95** — representing a pessimistic, median, and optimistic degradation prediction.

### What this actually means for your code

**`train_global_model.py`** needs to change significantly. Right now:
```python
model = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, ...)
model.fit(X_train, y_train)
```
To do quantile regression properly you have three options:
1. **Train 3 separate `XGBRegressor` models**, each with `objective='reg:quantileerror'` and `quantile_alpha=0.05 / 0.50 / 0.95`. You'd save a dict of 3 models instead of 1. This is the safest approach.
2. **NGBoost** — a separate library that natively outputs a parametric distribution. Would require `pip install ngboost` and rewriting the training script significantly.
3. **XGBoost multi-output quantile** — newer XGBoost versions support `quantile_alpha=[0.05, 0.5, 0.95]` with a single model. Clean but requires XGBoost ≥ 2.0.

**`simulation.py`** needs to change significantly. Right now `model.predict(df)[0]` returns one number. With 3 quantile models it returns 3 numbers. Every line that reads `stay_out_degradation = model.predict(...)[0]` becomes 3 parallel predictions.

**`app.py`** `load_global_model()` currently loads one `.joblib` dict with keys `model` and `features`. This becomes a dict with `model_p5`, `model_p50`, `model_p95`, and `features`.

### Is it worth it?
**Yes — this is the correct foundational change.** Without it, Features 2, 3, 4, and 5 are all deterministic and lose most of their claimed value. Every downstream Monte Carlo rollout is meaningless without sampling from a real distribution. This is genuinely the linchpin and the plan correctly identifies it as "must-have foundational."

**Difficulty:** Medium. The hardest part is not the code — it's validating that the quantile model's p5–p95 interval is actually *calibrated* (i.e., 90% of real observations fall between p5 and p95). If it isn't calibrated, the downstream uncertainty claims are false.

---

## Feature 2: Multi-Strategy Monte Carlo Optimizer

### What the plan says
Evaluate a full strategy space — Stay Out, Pit→Soft, Pit→Medium, Pit→Hard, Pit now, Pit lap+1, Pit lap+2, two-stop strategies — using N ≥ 1,000 Monte Carlo rollouts that sample from Feature 1's quantile distributions.

### What this actually means for your code

Right now `simulate_undercut_window()` in `simulation.py` takes `next_compound_encoded` as a single parameter. It runs one comparison. You'd need to:

1. **Replace the function** (or write a new one) that loops over all candidate strategies. Each strategy is a tuple: `(pit_lap_offset, compound)`.
2. **For each strategy × rollout:** sample a degradation trajectory by randomly drawing from the (p5, p50, p95) distribution — likely by fitting a Gaussian between quantiles and sampling from it, or by simple linear interpolation + noise.
3. **Accumulate results** across 1,000 rollouts per strategy into an empirical distribution of total race time.
4. **Select** the strategy with the minimum expected race time (or maximum P(finish ahead of rival)).

The output replaces the current `window_open: True/False` with something like:
```
Pit→Medium (lap 32): P(net gain)=68%, expected gain +2.3s ± 1.1s
Stay Out: P(net gain)=22%, expected gain -1.1s ± 0.8s
```

**Performance concern:** 1,000 rollouts × 15 laps × say 6 strategies = 90,000 model predictions per button click. XGBoost inference is fast, but in Streamlit each prediction involves `pd.DataFrame` construction overhead. At the current code structure (one DataFrame per lap per rollout), this will be **very slow**. The fix is to vectorize — build one large batch DataFrame of all rollouts simultaneously and call `model.predict()` once. This is a non-trivial refactor.

### Is it worth it?
**Yes, but it's the hardest implementation item.** The shift from "binary OPEN/CLOSED" to "probabilistic strategy ranking" is the genuine research contribution that distinguishes this from Phase 1. The plan is right that this "redefines the research question entirely." The difficulty is in making it *fast enough* to be usable in the Streamlit UI.

---

## Feature 3: Adaptive Decision Rule (Remove the 1.5s Threshold)

### What the plan says
Replace the hardcoded `net_undercut_advantage[2] >= 1.5` with a principled rule using: `Expected Net Gain = Fresh-Tyre Gain − Pit-Lane Loss − Traffic/Position Cost`.

### What this actually means for your code

Currently in `simulation.py` line 101:
```python
window_open = net_undercut_advantage[2] >= 1.5 if len(net_undercut_advantage) >= 3 else False
```
This is literally one line. If Feature 2 is implemented correctly, this line disappears entirely — the Monte Carlo output *is* the decision. `window_open` becomes the strategy with the highest `P(net gain)`.

The plan correctly says: *"This falls out almost for free once Features 1 and 2 exist."* That is accurate. This is not a separate implementation task — it's a consequence of Feature 2. Do not scope this as additional work.

### Is it worth it?
**Yes, and it's essentially free** once Feature 2 exists. The narrative value is high — "we identified a flaw in our own Phase 1 design and fixed it" is strong in a dissertation. But do not spend extra time on this separately.

---

## Feature 4: Rival Reaction Model (Two-Agent)

### What the plan says
When Driver A pits, there is a calibrated probability `P(rival pits within 1 lap)` conditioned on the current gap. Mine FastF1 historical data to compute this "cover rate" and inject it into Monte Carlo rollouts as a branching event.

### What this actually means for your code

This requires a new data analysis step that doesn't exist anywhere in the current codebase:
1. **Mine FastF1 data** across all races in the dataset. For every pit stop by Driver A, check if Driver B pitted within the next 2 laps, and record the on-track gap at the time.
2. **Build a lookup table:** `cover_rate[gap_bin]` → probability. E.g., gap < 2s: 80% cover, gap 2–5s: 45% cover, gap > 5s: 15% cover.
3. **In the Monte Carlo rollout:** when a "pit Driver A" event is sampled, draw a Bernoulli with `p = cover_rate[current_gap]`. If rival pits too, their next-lap time also gets a fresh-tyre boost in the simulation.

**The current code has no concept of Driver B's lap times inside `simulation.py`.** Right now, Driver B is only used in `app.py` for the Gap Evolution chart. To model rival reaction meaningfully, Driver B's state (tyre_age, compound, baseline_pace) needs to be tracked inside the simulation loop too.

### Is it worth it?
**Yes, but scope it carefully.** The data mining for "cover rate" is genuinely novel work and defensible as a research contribution. The risk is that your historical dataset may not have enough pit stop pairs to estimate `cover_rate` by gap bin with any statistical confidence — check your data size before committing to this. If you only have ~200 pit stops, conditioning on gap bins will produce noisy estimates.

---

## Feature 5: Safety Car / VSC as Stochastic Events

### What the plan says
Precompute per-circuit SC probability from historical data. In each Monte Carlo rollout, sample a Bernoulli event per lap; if SC triggers, reduce `pit_loss_delta` by ~60–70%.

### What this actually means for your code

1. **New lookup table:** `sc_rate[circuit]` → probability per lap. E.g. Baku ≈ 0.12 per lap, Monza ≈ 0.04 per lap. This can be computed from your existing FastF1 data by checking `TrackStatus == '4' or '6'` across all laps in the dataset.
2. **In `simulate_undercut_window()`:** for each rollout lap, draw `Bernoulli(sc_rate[circuit])`. If True, override `pit_loss_delta` for that lap: `effective_pit_loss = pit_loss_delta * 0.35` (a ~65% reduction, since the pit lane delta collapses under SC).
3. **The plan says this is "cheap to add given Feature 2's architecture."** That is correct. If Feature 2 is implemented with vectorized rollouts, adding a Bernoulli sample per lap per rollout is literally one line of numpy: `sc_events = np.random.binomial(1, sc_rate, size=(n_rollouts, horizon))`.

### Is it worth it?
**Yes — this is the highest bang-for-buck feature on the list.** It is genuinely cheap to implement once Feature 2 exists, it models a real and important race phenomenon (pitting under SC is one of the most consequential strategy moments in F1), and it is very easy to demo visually. The SC probability table itself is a small but real data contribution.

---

## Feature 6: SHAP Explainability + Historical Backtesting

### What the plan says
- **SHAP:** Run `shap.TreeExplainer` on the degradation model to show which features (tyre age, rival gap, TrackTemp) drove each recommendation.
- **Backtesting:** Replay 30–50 historical races lap-by-lap, feed real telemetry state into the Phase 2 optimizer at each decision point, compare its recommended strategy to what the team actually did.

### What this actually means for your code

**SHAP** is straightforward:
```python
pip install shap
explainer = shap.TreeExplainer(model_p50)
shap_values = explainer.shap_values(X_test_sample)
```
Plotly/Streamlit can render the SHAP waterfall or beeswarm plot. This is essentially copy-paste from the SHAP documentation. Low effort, high visual impact.

**Backtesting** is genuinely hard and the most time-consuming item. It means:
1. For each of 30–50 races, run the full pipeline (data_pipeline → feature_engineering → ml_model) to get `ml_data`.
2. For each lap in each race, call the Phase 2 Monte Carlo optimizer with real telemetry state.
3. Record the optimizer's recommended strategy at each lap.
4. Compare to what the team *actually* did (pit lap, compound choice) from the FastF1 data.
5. Compute: simulated time delta, regret (did the model recommend better than reality?), success rate.

Step 4 is the tricky one — you need to define "what the team actually did" cleanly from FastF1 pit data, which is already partially built in `calculate_pit_loss()` but not fully structured for backtesting.

### Is it worth it?
**Backtesting is not optional — it is essential.** The plan is exactly right that R² alone is insufficient evidence. Without backtesting, you have a model that fits training data well but no proof it would have made a real race faster. This is your evaluation/results chapter and without it, the research contribution is incomplete. SHAP is a nice bonus on top — easy and visual.

---

## Overall Verdict: Is This Plan Better Than the Current Path?

### Yes — with two critical caveats.

**Why it's genuinely better:**

1. **The research question becomes real.** Phase 1 answers: "Does the undercut window look open?" Phase 2 answers: "What is the optimal strategy with quantified probability, and would it have worked on real historical races?" That is a fundamentally more defensible and interesting question.

2. **The dependency chain is well-designed.** Features 1→2→3 flow naturally. Feature 2 is the real engine and Features 3, 4, 5 are clean extensions of it. The plan correctly excluded over-scoped ideas (full multi-agent RL, LSTM degradation models) that would be scope-killers.

3. **Backtesting is the right evaluation methodology.** The plan correctly identifies this as "critical" and frames it as the results section. Without it, Phase 2 is just Phase 1 with more maths.

**Caveat 1 — Feature 1 calibration is non-trivial.**
Quantile regression is not "just changing the objective function and retraining." You *must* validate that the quantile estimates are calibrated — otherwise every downstream uncertainty claim is scientifically invalid. Budget time to compute calibration plots (predicted interval coverage vs. actual coverage) and potentially apply post-hoc calibration (isotonic regression or conformal prediction).

**Caveat 2 — Vectorize Feature 2 from day one.**
If you implement the Monte Carlo simulator naively (one `pd.DataFrame` per lap per rollout), 1,000 rollouts will take 10–30 seconds per button click in Streamlit. That makes it unusable as a live demo. Vectorize the batch prediction using numpy arrays from the start, not as an afterthought.

---

## Recommended Build Order

| Order | Feature | Why |
|---|---|---|
| **1st** | Quantile XGBoost (Feature 1) | Nothing else works without this |
| **2nd** | SC rate table + Backtesting data pipeline setup | Front-load the data work while the model bakes |
| **3rd** | Vectorized Monte Carlo optimizer (Feature 2) | The core engine |
| **4th** | Adaptive decision rule (Feature 3) | Falls out of Feature 2 for free |
| **5th** | SC stochastic events (Feature 5) | Cheap add-on to the MC loop |
| **6th** | Rival reaction model (Feature 4) | Needs data validation first |
| **7th** | SHAP (Feature 6a) | Easy, high visual payoff |
| **Last** | Backtesting evaluation (Feature 6b) | Requires everything else working |
