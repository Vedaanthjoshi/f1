F1 Strategy Engineer — Final Year Feature Plan
Curated Phase 2 feature set: uncertainty-aware, multi-strategy race optimization (final-year
extension of the Phase 1 undercut/degradation engine)
This plan curates the strongest ideas from the initial Phase 2 brainstorm into six features that are each
technically distinct from the Phase 1 system, realistically buildable on the existing FastF1 + XGBoost +
Streamlit pipeline within a final-year timeline, and defensible as genuine research contributions rather than
incremental tweaks. High-effort, high-uncertainty items from the original brainstorm (full multi-agent
reinforcement learning, wet-weather modeling, LSTM/state-space degradation models) were deliberately
excluded or downgraded to keep scope realistic.
Proposed Paper Title
"Uncertainty-Aware Monte Carlo Optimization of Formula 1 Pit Strategy with Empirically-Calibrated
Competitor Reactions and Stochastic Race Events"
1. Probabilistic Tyre Degradation Model (Quantile Regression)
What it is: Instead of XGBoost predicting a single lap_time_delta, train it (or a wrapper such as NGBoost, or
XGBoost with quantile/pinball loss at τ = 0.05, 0.5, 0.95) to output a distribution rather than a point estimate.
Why it matters: This is the single highest-leverage change. Every downstream feature (Monte Carlo
simulation, risk-aware decisions, confidence scores) depends on having uncertainty bounds instead of one
number. Reviewers will immediately recognize “point prediction → probabilistic prediction” as a real
methodological shift, not a tweak.
Implementation approach: XGBoost supports reg:quantileerror (recent versions) or you can train 3 separate
models at different quantiles. Output becomes (p5, p50, p95) per lap instead of one float.
Paper angle: “Does modeling degradation uncertainty change strategic recommendations versus
point-estimate models?” — a directly comparable ablation study.
2. Multi-Strategy Monte Carlo Optimizer
What it is: Expand simulation.py to evaluate a full strategy space: Stay Out, Pit→Soft, Pit→Medium,
Pit→Hard, Pit lap+1, Pit lap+2, and two-stop strategies — each run through N ≥ 1000 Monte Carlo rollouts
sampling from Feature 1's degradation distributions.
Why it matters: This is the “undercut detector → strategy optimizer” reframing, and it's the must-have change
that redefines your research question entirely, replacing the binary Stay Out vs. Pit Now comparison.
Implementation approach: For each candidate strategy, sample degradation trajectories from the quantile
model, sum simulated race time across rollouts, and obtain an empirical outcome distribution per strategy.
Select the strategy that minimizes expected race time (or maximizes P(finish ahead of rival)).
Paper angle: Example output: “Pit→Medium (lap 32): 68% P(net gain), expected gain +2.3s ± 1.1s” — far
richer than a binary OPEN/CLOSED signal.
3. Adaptive Decision Rule (Removing the Fixed 1.5s Threshold)
What it is: Replace the hardcoded rule with: Expected Net Gain = Fresh-Tyre Gain − Pit-Lane Loss −
Traffic/Position Cost, evaluated continuously against the actual rival gap, remaining laps, and circuit-specific
pit loss.
Why it matters: You already identified the fixed 1.5-second / 3-lap rule as an arbitrary heuristic in Phase 1.
Fixing your own documented weakness gives a strong, easy-to-defend narrative for a committee: “we found
this limitation, and here is the principled fix.”
Implementation approach: This falls out almost for free once Features 1 and 2 exist — it becomes a query
against the Monte Carlo output rather than a comparison to a fixed constant.
4. Rival Reaction Model (Lightweight Two-Agent, Not Full Multi-Agent
RL)
What it is: Rather than modeling a full multi-agent race, implement a tractable reactive model: when Driver A
pits, Driver B has a calibrated probability of reacting by pitting on the following lap.
Why it matters: Delivers real strategic-interaction value without the scope and risk of full multi-agent
reinforcement learning, which even the original brainstorm correctly flags as a stretch goal you should not
start with.
Implementation approach: Mine historical FastF1 data for a “cover rate” — P(rival pits within 1 lap of an
undercut attempt), conditioned on gap size. Feed this probability into the Monte Carlo simulator as a
branching event.
Paper angle: “Incorporating empirically-calibrated competitor reaction probabilities into undercut strategy
optimization.”
5. Safety Car / VSC as Stochastic Simulation Events
What it is: Inject Safety Car / VSC probability — estimated per circuit from historical race data (e.g. Baku or
Singapore have much higher SC likelihood than Monza) — into the Monte Carlo rollouts as random events
that collapse effective pit-loss cost.
Why it matters: Cheap to add given Feature 2's architecture, very intuitive to demo (“pitting under a Safety
Car is nearly free”), and models genuinely realistic race strategy behavior that Phase 1 ignores entirely.
Implementation approach: Precompute a per-circuit SC rate table from the 2022–2025 data you already
hold. Sample a Bernoulli event per rollout lap; if SC triggers, reduce pit_loss_delta by roughly 60–70%.
6. Explainability Layer + Historical Backtesting
What it is: Two components that work together as your evaluation methodology:
• SHAP on the degradation model, so any recommendation (e.g. “Pit Now”) shows top contributing factors
(tyre age, rival gap, degradation slope, pit loss) — turning a black-box number into decision support an
engineer could trust.
• Backtesting against real historical races: replay 30–50 historical races lap-by-lap, feed real telemetry state
into the Phase 2 optimizer at each decision point, and compare its recommended strategy against what the
team actually did.
Why it matters: The original brainstorm correctly calls backtesting “critical”: without it, the paper's only
evidence is R², which reviewers will rightly say proves the regression fits well but not that the strategy is
better. This feature is your actual results/evaluation section.
Implementation approach: Report metrics such as simulated race-time delta, undercut success rate, and
strategy regret across the backtested races.
Feature Summary
# Feature Type Depends On
1 Probabilistic degradation model (quantile regression) Foundational / Must-have —
2 Multi-strategy Monte Carlo optimizer Must-have Feature 1
3 Adaptive decision rule (remove 1.5s threshold) Must-have Features 1, 2
4 Rival reaction model (2-agent, calibrated) Strong addition Feature 2
5 Safety Car / VSC stochastic events Strong addition Feature 2
6 Explainability (SHAP) + historical backtesting Critical evaluation Features 1–5
Note: Features excluded from the original brainstorm (full multi-agent reinforcement learning, wet-weather
compound-switching strategy, and temporal/state-space degradation models) remain viable stretch goals if time
permits, but are not required for a strong, defensible final-year paper.