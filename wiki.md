# Phase 1 Benchmark Wiki

## Objective

Phase 1 exists to identify the best verified momentum method we can build, run, and reproduce honestly under a `2xT4` budget on the current `110`-stock daily market-neutral setup. This is a strict benchmark, not a broad finance-model survey. LoRA and broader model comparisons stay out of scope until Phase 1 produces a local winner or a clear no-winner diagnosis.

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

## Source-of-truth files to edit

- Primary benchmark/reporting sources:
  - `build_notebook_v2.py`
  - `patch_autoresearch_guardrails.py`
  - `metric_cell_18.py`
  - `metric_cell_24.py`
  - `metric_cell_28.py`
- Tests for evolution/diagnostic behavior:
  - `tests/test_evolution_guardrails.py`
- Generated notebooks are build artifacts, not the primary edit target:
  - `autoresearch_v2_final.ipynb`
  - `kaggle_submission/autoresearch_v2_final.ipynb`

## Artifact interpretation

- Treat `outputs/*` old 20-stock results as historical only. They are not comparable evidence for the current guarded benchmark.
- Treat `kaggle_run_v3/*` as the current guarded reference when interpreting present behavior and artifacts.
