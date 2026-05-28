# Phase 1 Benchmark Wiki

## Objective

Phase 1 exists to identify the best verified momentum method we can build, run, and reproduce honestly under a `2xT4` budget on the current `110`-stock daily market-neutral setup. This is a strict benchmark, not a broad finance-model survey. LoRA and broader model comparisons stay out of scope until Phase 1 produces a local winner or a clear no-winner diagnosis.

## Current pivot

- The active Kaggle notebook is now `llm_research` first: LLM generation is enabled by default when `AUTORESEARCH_RUN_PROFILE=llm_research` and benchmark mode is off.
- The LLM contract remains `Strict Signal Math`: generate compact Pandas/NumPy alpha kernels, not full scripts, boilerplate execution code, or portfolio construction.
- Candidate outputs must keep the existing `=== CANDIDATE ===` parser format while focusing mutations on residual momentum, volume conditioning, volatility scaling, and optional VIX/TNX regime inputs.
- The execution sandbox injects `np` and `pd`, passes optional `vix`/`tnx` into compatible `signal(close, volume, vix=None, tnx=None)` kernels, and validates strict matrix dimensionality.
- Guarded heldout now admits scored `llm_autoresearch` rows from the research log alongside deterministic anchors, so viable strict LLM candidates are not stranded before final reporting.
- LLM-primary runs with zero recorded LLM generation/reflection calls are marked incomplete and cannot pass the deployment gate.
- Parameter-search evolution is no longer the default spend path for LLM-primary runs; it must be explicitly enabled with `AUTORESEARCH_RUN_PARAM_SEARCH=1`.

## Active implementation order

Current stage:

1. Deterministic momentum baselines
2. Risk-managed classical momentum baselines
3. Current AutoResearch / evolutionary search

Next stage, only after pure AutoResearch is honestly resolved:

4. Sharpe-optimized LSTM

Later, only if the evidence justifies the compute:

5. AutoResearch + LSTM combinations
6. Lightweight tabular ML
7. Gradient-boosted trees

## Shared input and evaluation contract

- Universe stays at `110` US equities for Phase 1.
- Every family uses the same enriched inputs: price, volume, VIX, TNX, shared volatility-state features, and shared regime indicators.
- Every family must share the same date ranges, train/validation/rolling-holdout protocol, market-neutral construction, cost model, turnover accounting, beta accounting, drawdown accounting, and report format.
- Multiple chronological holdouts are required. A single flattering holdout is not enough.

## Winner rule

- Score methods with one common composite that includes held-out Sharpe, drawdown, turnover and cost stress, recent-subperiod stability, rolling-window consistency, beta drift penalty, and fragility penalties where relevant.
- A method wins only if all three are true:
  - it has the highest composite score
  - it beats the best deterministic baseline by at least `+0.10` composite
  - it does not lose the majority of supporting diagnostics
- Supporting diagnostics must include at least cost stress, drawdown, recent-period stability, rolling-holdout stability, and beta behavior.
- If no family clears that bar, the correct outcome is `no winner yet`.

## Sequencing rule

- Do not start LSTM benchmarking until the pure AutoResearch stage has been re-run under the tightened benchmark contract and reviewed.
- Do not start AutoResearch + LSTM combinations unless LSTM shows standalone value and the evidence suggests complementarity.
- Do not describe deferred families as failed if they were not executed.

## Stagnation-diagnosis rule

- `3` generations without improvement in best-so-far composite score triggers diagnosis.
- Diagnosis is explicit, not silent. Emit a diagnosis artifact/report and continue the broader benchmark unless a separate hard stop fires.
- Diagnosis should summarize the stagnating family or branch, best-so-far composite trend, robustness trend, diversity trend, common failure reasons, and whether the issue looks like search collapse, score misalignment, validator pressure, or weak mutations.

## Source-of-truth file to edit

- Primary and only submission artifact:
  - `kaggle_submission/autoresearch_v2_final.ipynb`
- Local mirror:
  - `artifacts/notebooks/autoresearch_v2_final.ipynb`
- There is no active builder, patcher, or test pipeline in the notebook submission workflow.

## Artifact interpretation

- Treat `archive/outputs/*` old 20-stock results as historical only. They are not comparable evidence for the current guarded benchmark.
- Treat `archive/kaggle_run_v3/*` as the current guarded reference when interpreting present behavior and artifacts.

## Previous benchmark snapshot

- `archive/results13.zip` is the last complete deterministic/evolution reference run with full summaries.
- Best held-out leader was still deterministic: `regime_gate 54/90 @ VIX<20`, with composite score `+0.45` and test Sharpe `+0.42`.
- Best evolved held-out candidate reached composite score `+0.39` and test Sharpe `+0.45`, but did not beat the deterministic baseline by the required edge.
- Evolution stopped at `14` generations with stop reason `stagnation_reached`; valid-rate was high, but robust survivors were not converting into held-out edge.
- Main recurring blockers in that run:
  - zero robust winners across late generations despite many valid candidates
  - heavy concentration in `volume_confirm` / `regime_momentum` families
  - frequent near-constant signals caught by validator pressure
  - no evolved candidate clearing the `+0.10` edge-over-deterministic winner rule

