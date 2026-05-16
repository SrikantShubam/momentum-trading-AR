# Phase 1 Momentum Benchmark Design

Date: 2026-05-12
Project: `C:\experiments\andrej karpathy auto research for momentum trading`
Status: approved design draft for user review

## Objective

Phase 1 is not "find the coolest model." Phase 1 is to determine the best verified momentum trading method we can build, run, and reproduce honestly under a `2xT4` compute budget on the current `110`-stock daily market-neutral setup.

The benchmark must be strict enough that weak methods cannot win by exploiting a soft baseline, a favorable single holdout, or unequal inputs.

## What Phase 1 is and is not

Phase 1 is:

- a winner-take-all benchmark across a small set of serious method families
- a shared evaluation contract
- a compute-bounded research program
- a test of whether the current AutoResearch system can compete with stronger non-LLM baselines

Phase 1 is not:

- a broad survey of every finance model class
- a fair test of public or self-trained financial LoRAs
- a claim about the best momentum method in the world
- a universe-expansion study

LoRAs and broader model families are deferred until after a local winner is established.

## Benchmark scope

### Universe

Keep the current `110` US equity universe for Phase 1.

Reason:

- preserves continuity with current guarded results
- avoids changing both method and dataset at the same time
- is large enough for a meaningful cross-sectional momentum benchmark
- fits the `2xT4` budget better than a larger universe when using rolling holdouts and multiple families

### Inputs

All method families must use the same enriched dataset.

Shared inputs:

- price
- volume
- VIX
- TNX
- shared volatility state features
- shared regime indicators derived from the common input set

No family gets privileged data. Either all methods get enriched inputs, or none do. Phase 1 uses enriched inputs for all.

### Portfolio construction and evaluation parity

All method families must share:

- same universe membership
- same date ranges
- same train, validation, and rolling holdout protocol
- same market-neutral portfolio construction
- same cost model
- same turnover accounting
- same beta accounting
- same drawdown accounting
- same evaluation report format

This is mandatory. Without it, the comparison is biased.

## Method families in active scope

The benchmark contract is broader than the immediate implementation sequence.
The active execution scope is intentionally staged.

Immediate active scope:

1. Deterministic momentum baselines
2. Risk-managed classical momentum baselines
3. Current AutoResearch / evolutionary search

Deferred until AutoResearch is hardened and honestly benchmarked:

4. Sharpe-optimized LSTM

Deferred beyond that unless the evidence justifies spending more compute:

5. AutoResearch + LSTM combination studies
6. Lightweight tabular ML
7. Gradient-boosted trees

### Rationale

This keeps the research disciplined under `2xT4`.

The immediate question is not "which of many model classes wins?" The immediate question is whether pure AutoResearch can beat stronger deterministic and classical momentum baselines under a strict held-out winner rule.

LSTM comes next because it is the strongest published adjacent method family in the research path already discussed. Combination studies only make sense after both standalone legs are credible.

## Winner definition

Phase 1 does not use test Sharpe alone.

The winner is determined by a common composite score that must combine:

- held-out Sharpe
- drawdown
- turnover and transaction-cost stress
- recent-subperiod stability
- rolling-window consistency
- beta drift penalty
- fragility penalties where relevant

### Hard winner rule

A method wins only if:

1. it achieves the highest common composite score
2. it beats the best deterministic baseline by at least `+0.10` composite score
3. it does not lose the majority of supporting diagnostics

Supporting diagnostics include at minimum:

- cost-stress behavior
- drawdown
- recent-period stability
- rolling holdout stability
- beta behavior

If no family clears this bar, Phase 1 ends with "no winner yet."

## Holdout policy

Phase 1 must use multiple chronological holdouts, not a single held-out period.

Reason:

- a single favorable regime can flatter weak momentum methods
- recent-period stability matters
- a method that wins only once is not a trustworthy benchmark champion

The exact rolling holdout scheme should be finalized in the implementation plan, but the principle is fixed now:

- same rolling holdouts for all families
- same winner rule across all of them

## Compute policy

Each method family gets a serious but bounded budget:

- target `4-8` hours per family

Fairness rule:

- equal compute budget across families
- equal high-level search budget across families
- family-specific hyperparameters are allowed inside that budget

Fairness is defined as equal opportunity under compute, not identical parameter shapes.

## AutoResearch-specific rule

Phase 1 is explicitly a test of whether AutoResearch can survive contact with stronger baselines.

If AutoResearch cannot reliably improve under the benchmark harness, the conclusion is not automatically "momentum is impossible." The likely conclusion is that the search process, scoring alignment, diagnostics, or guardrails are weaker than the baseline engineering.

This means the benchmark must test the research loop honestly, not just the best accidental output.

## Stagnation and diagnosis rule

The user requested a practical AutoResearch-style standard: if results do not improve for three successive generations, the path is suspect.

Phase 1 adopts that idea as follows:

- `3` generations without improvement in best-so-far composite score triggers diagnosis
- this is not an invisible event
- the system must emit a diagnosis report
- the run continues with the rest of the benchmark rather than silently pretending the frontier is healthy

The diagnosis report should include:

- stagnating family or branch
- best-so-far composite trend
- robustness trend
- diversity trend
- common failure reasons
- whether the issue appears to be search collapse, score misalignment, validator pressure, or weak mutations

This creates research evidence instead of silent compute burn.

## Phase ordering

Implementation and evaluation should proceed in this order:

1. Build the common benchmark contract
2. Strengthen deterministic and risk-managed classical baselines
3. Fix pure AutoResearch so the LLM stage is active, diagnosable, and comparable
4. Re-run pure AutoResearch against the deterministic/classical benchmark until it either wins honestly or yields a clear no-winner diagnosis
5. Only then add Sharpe-optimized LSTM as the next standalone benchmark family
6. Compare standalone LSTM against standalone AutoResearch and the deterministic/classical benchmark
7. Only if LSTM shows standalone merit and nontrivial complementarity, evaluate AutoResearch + LSTM combinations
8. Defer tabular ML and gradient-boosted trees unless the later benchmark evidence says the extra breadth is worth the resource cost
9. Declare a winner only if the full winner rule is satisfied

## Evidence gate for combination work

Combination studies are not allowed just because they are available.

AutoResearch + LSTM is only worth running if both of these are true:

1. LSTM shows standalone viability under the same winner contract, or at minimum demonstrates a credible held-out edge pattern that deterministic and AutoResearch do not already capture
2. Error patterns or subperiod behavior suggest complementarity rather than redundant exposure

If those conditions are not met, combination work is research theater and should be skipped.

## Deliverables for the current implementation stage

The current stage should produce:

- a common benchmark harness
- run-scoped artifact directories
- per-family benchmark summaries for deterministic, classical risk-managed, and pure AutoResearch
- rolling-holdout comparison tables
- stagnation diagnosis logs where triggered
- a final winner report or a no-winner report
- a `wiki.md` source-of-truth document summarizing the active benchmark contract for future agents

The next stage adds:

- standalone LSTM benchmark summaries
- a direct AutoResearch-vs-LSTM-vs-deterministic comparison

Only after that comes any combination study.

## Debate position

This design intentionally rejects two temptations:

1. comparing too many method families too early
2. moving into combinations or LoRAs before proving the standalone legs are worth the compute

That restraint is deliberate. Under `2xT4`, the best research move is to prove a stable winner on the current `110`-stock system first. Only then should the project broaden into LoRAs, external finance models, or larger universes.

## Risks

Main risks in Phase 1:

- AutoResearch still not truly active despite model configuration appearing in reports
- soft scoring allows weak methods to win
- rolling holdouts increase engineering complexity and runtime
- family implementations become inconsistent if the benchmark contract is not centralized
- stale artifacts from older runs contaminate interpretation

These risks should be handled directly in the implementation plan.

## Acceptance criteria for moving beyond Phase 1

We only broaden to LoRAs or external finance model comparisons after Phase 1 produces one of these:

- a clear benchmark winner satisfying the full winner rule
- or a clear diagnosis showing why no family can yet satisfy the rule under the current system

If neither condition is met, the benchmark design has not yet done its job.
