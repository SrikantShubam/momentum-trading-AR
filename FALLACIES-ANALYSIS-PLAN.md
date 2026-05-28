# Senior Researcher Post-Mortem: Fallacies in the Momentum AutoResearch Experiment

**Analysis Date**: May 2026 (updated 28 May)  
**Focus Artifacts**:
- `results (2).zip` (LLM-research run 2026-05-24)
- `results lastest 28 May.zip` (LLM-research run 2026-05-27, the newest as of 28 May)  
**Scope**: Phase 1 Benchmark (110-stock daily market-neutral momentum, strict winner rule per `wiki.md`)

---

## One-Page Executive Summary (for quick review / Kaggle / GitHub)

**The core problem**: After the May 2026 LLM-primary pivot, the autonomous research loop is still not producing evidence that can be judged under the Phase 1 benchmark contract defined in `wiki.md`.

- `results (2).zip` (24 May run): 51 LLM-generated candidates, some positive research scores in the memo, but **no held-out adjudication**.
- `results lastest 28 May.zip` (27 May run — newest): **21 actual LLM calls**, **33 log entries, 0 viable backtests (0% viable rate)**, complete stagnation abort after 25 batches. The LLM produced only non-viable signals (train Sharpe and score both negative on every candidate). The exported `best_signal.py` is the standard "NO APPROVED WINNER" guard. Held-out evaluation was never reached.

Historical deterministic + evolution runs show the same pattern: ~4% valid rate, zero robust survivors, early termination on zero-robust streaks, and evolved edges that never clear the required +0.10 composite + majority diagnostics bar.

**Top fallacies (ranked by threat to the "no winner yet" claim)**:

1. **Survivorship bias (highest impact)** — Fixed 110 current large-cap tickers (Cell 6). No delisteds, no point-in-time membership. Both baselines and LLM kernels are tested on winners that "made it" to 2026. Fix is hard (external historical constituents) but a diagnostic + warning is easy.

2. **LLM track bypasses the benchmark contract** — The new primary method (LLM kernels) is not being run through the same shortlist (`HELDOUT_SHORTLIST_K=20`, source quotas, cost stress, walk-forward, subperiods) and winner rule (`+0.10` edge + majority diagnostics) that the wiki and code (`_build_heldout_report`, `evaluate_on_test`) mandate. Even the May 27 run that finally produced an `experiment_summary.md` had 0 viable candidates and never reached held-out evaluation.

3. **LLM-generated code is currently incapable of clearing the basic viability gate** (new, from 28 May data) — 21 real LLM calls (Qwen 14B) → 33 candidates → **0 viable** (0% rate). Every signal failed the hard filter "train Sharpe > 0 and score > 0". Frequent shape errors (`(2014, 2124) != (2014, 110)`) and numpy boolean subtraction bugs. The loop aborts on stagnation with literally nothing positive to show. The research memo now says "Nothing is working yet."

4. **Premature death of the search + overly harsh early viability filter** — The combination of a strict positive-Sharpe requirement very early in the LLM loop + shape/runtime fragility + early stagnation abort means the autonomous research never gets past the first filter. Even the previous (24 May) run only had a handful of marginal positives in the memo.

5. **Train objective does not predict the final winner rule** — Search optimizes train "robust" score; the real bar (cost stress at 10-15 bps, wf_min, subperiod stability, beta) is applied only at the end. Large train-heldout gaps (+0.79) are predictable. Fix: move cost + stability metrics into early rejection and shortlist ranking (medium, reuses already-computed fields).

**Current state of the experiment** (as of 28 May 2026):
- The technical scaffolding (sandbox + validation, neutralization, cost stress, walk-forward + multi-window diagnostics, explicit "no winner" guards, `detect_lookahead` permutation test) remains unusually strong.
- The LLM track is in **deep trouble**: 21 real calls on the newest run produced 0 viable signals. The research memo now openly states "Nothing is working yet."
- The loop is being starved by the combination of (a) very strict early viability gate (must be positive Sharpe on train), (b) LLM code quality issues (shape errors, numpy boolean bugs), and (c) aggressive early stagnation stopping.

**Recommended immediate action**:
- The May 27 run must be treated as a **critical diagnostic**: the current viability filter + LLM prompt + sandbox is producing a 0% viable rate. Do not claim "LLM research is running" until this is fixed.
- Strongly consider temporarily relaxing the early `Sharpe > 0` gate for the LLM exploration phase (keep it for final shortlist) so the model can at least see some positive signals and learn from reflection.
- Next `llm_research` run should target **at least 80-100+ viable research-phase backtests** before any stagnation abort is allowed.
- Apply the fixes around early viability, diversity injection, and better error feedback to the LLM before the next long run.

If after these fixes the LLM track still cannot clear the +0.10 edge under honest conditions, the correct scientific conclusion per the wiki is "no winner yet — and the research method itself may need re-architecture."

---

## Context

The project is executing a **Karpathy-style autonomous research loop** ("AutoResearch v2") for daily market-neutral momentum alpha on a fixed 110 large-cap US equity universe (2015-01 to 2024-12). The system has two main tracks that must be compared under an identical, rigorous evaluation contract defined in `wiki.md`:

- Deterministic + classical risk-managed momentum baselines (parameter sweeps).
- LLM-driven / evolutionary search (originally parameter search, now pivoted to LLM-primary generation of strict `signal(close, volume, vix=None, tnx=None)` kernels).

**The strict winner rule** (wiki.md:44-50) requires any new family to:
1. Achieve the highest composite score on held-out data.
2. Beat the best deterministic baseline by **at least +0.10** composite.
3. Not lose the majority of supporting diagnostics (cost stress at 5/10/15bps, walk-forward min/median, subperiod stability, beta drift, benchmark-spread Sharpe, robustness score, etc.).
4. If no method clears the bar → "no winner yet".

Historical bundles (results2_latest/, results_latest/, archive/results13.zip and earlier) consistently show:
- Evolution/LLM candidates have ~4% valid rate after sandbox + neutralization + robustness filters.
- Zero or near-zero "robust_ok" survivors in many generations.
- Early stopping on `zero_robust_streak_reached`.
- Best evolved held-out edges over deterministic are +0.05 or negative; they lose on cost stress and recent subperiods.
- Extreme mode collapse around "regime_momentum" (EWMA fast/slow + simple VIX gate) variants.
- Large train → held-out degradation.

The May 2026 pivot (`memory.md`, wiki update) enabled **LLM-primary** (`AUTORESEARCH_RUN_PROFILE=llm_research`), with the active notebook `kaggle_submission/autoresearch_v2_final.ipynb` (synced from `artifacts/notebooks/autoresearch_v2_final.ipynb`). The latest `results (2).zip` is the first packaged output of this profile: 51 LLM-generated candidates, a research memo praising shorter-filter momentum variants (train/research Sharpe +0.87), but **no full held-out adjudication or deterministic comparison artifacts**.

This analysis treats the entire history + code + latest zip as the evidence base. The question is not "does momentum work?" but "**what fallacies are we systematically committing in how we discover, validate, and claim progress on momentum alphas?**"

---

## Recommended Approach

**Style**: Senior quant researcher post-mortem.

**Grounding**:
- Read the actual implementation in the source-of-truth notebook (all 13 code cells, especially load_prices, backtest, validate_*, detect_lookahead, _robustness_score_safe, walk_forward, held-out shortlist, winner rule reporting).
- Analyze every major result bundle (focus on results (2).zip research_log.jsonl + memo, plus extracted summaries from results*_latest/ and archive/).
- Cross-reference against the explicit contract in `wiki.md` and the May 12 improvement plan.
- Distinguish between:
  - Fallacies that invalidate claims (bias, leakage, selection bias).
  - Execution/operational gaps (LLM not driving, early stopping, insufficient budget).
  - Design choices that are reasonable but weak (cost model, universe).
- For each fallacy: (a) evidence from code + artifacts, (b) why it matters for the Phase 1 claim, (c) concrete fix, (d) doability (code-only, data, compute, Kaggle constraints).

**Scope boundaries** (per wiki):
- Stay within the 110-stock daily market-neutral contract.
- Do not propose broad "add LSTM now" unless it directly illuminates a fallacy in the current loop.
- Prioritize fixes that make the "no winner yet" or "winner" verdicts more trustworthy.

**Output of this work**:
- A clear, evidence-backed list of 6-10 fallacies ranked by severity/impact on conclusions.
- For each: minimal patch or experiment that would materially reduce the fallacy.
- Feasibility matrix (easy / medium / hard under current Kaggle + notebook discipline).
- Updated checklist items for the next iteration.
- (Optional) concrete diffs or cell patches if the fix is surgical.

---

## Critical Files & Artifacts to Analyze

### Source of Truth
- `kaggle_submission/autoresearch_v2_final.ipynb` (and its mirror `artifacts/notebooks/autoresearch_v2_final.ipynb`)
- `patch_autoresearch_guardrails.py` (if still active)
- `build_notebook_v2.py` (historical builder)

### Core Logic Cells
- Cell 6: `TICKERS` list + `load_prices()` (survivorship, coverage filter).
- Cell 8: `backtest(...)` — neutralization, turnover costing, beta, etc.
- Cell 10: `validate_code`, FORBIDDEN + shape-risk lints.
- Regime loading + `detect_lookahead` permutation test (already sophisticated).
- Later cells (~18/20/22/24/26/28): `run_signal_code`, `_robustness_score_safe`, `walk_forward`, held-out shortlist construction (`HELDOUT_SHORTLIST_K=20`, per-cluster caps, min deterministic/evolution/LLM counts), winner rule adjudication (`APPROVED_WINNER_EDGE_OVER_DETERMINISTIC=0.10`), reporting.

### Data & Regime
- `prices.parquet`, `regime.parquet`.
- VIX/TNX loading and alignment.

### Result Bundles (primary evidence)
- `results (2).zip` (latest LLM-research: 51 log entries, research_memo.txt, no held-out verdict bundle).
- `archive/results2_latest/`, `archive/results_latest/`, `archive/results13.zip` etc. (full `experiment_summary.md`, `deterministic_results.json`, `evolution_*.json`, `best_signal*.py`).
- `archive/outputs/2026-05-12-momentum-improvement-plan.md`.
- `run_manifest.json`, `research_log.jsonl`, `runtime_metadata.json` patterns.

### Governance & Memory
- `wiki.md` (Phase 1 contract, winner rule, sequencing, stagnation diagnosis).
- `memory.md` + `errors.md` + `checklist.md`.
- `AGENTS.md`.

---

## Key Fallacies Identified

### High Severity (invalidate or seriously weaken claims)

**New data point — 28 May run (`results lastest 28 May.zip`)**

This is the most recent execution (LLM-research profile, started 2026-05-27). Key facts:
- 21 actual LLM calls to Qwen/Qwen2.5-Coder-14B-Instruct (4-bit on 2xT4).
- 25 batches → 33 log entries → **0 viable backtests** (0% viable rate).
- Every candidate rejected with either shape errors (`(2014, 2124) != expected (2014, 110)`) or the hard viability gate: `NON_VIABLE: train Sharpe -0.XX < +0.00 or score -Y.YY < +0.00`.
- Loop aborted on `search_stagnation` after only modest improvement attempts.
- The exported `best_signal.py` and `best_signal_ensemble.py` contain only the standard "NO APPROVED WINNER — DEPLOYMENT BLOCKED" guard.
- LLM self-reflection memo: "**Nothing is working yet.**"
- No held-out evaluation was ever attempted.

This is materially worse than the 24 May run (which at least produced some candidates with positive research-phase scores in the memo). The autonomous LLM research loop is currently generating code that cannot even clear the most basic profitability filter on the training period.

---

**1. Survivorship Bias in Universe Construction (severe, unacknowledged)**  
Evidence: Hard-coded `TICKERS` list of 110 current (2026) large/mega-caps in Cell 6; `load_prices()` does yfinance download 2015-2024 + >70% coverage filter + ffill/bfill. No point-in-time index membership, no delisted tickers.  
Impact: Momentum signals that would have rotated away from names that later shrank or were dropped look artificially strong. Both baselines and LLM kernels are contaminated the same way. This is the single largest threat to any deployment claim.  
Fix options: (a) minimum — add explicit survivorship-bias diagnostic in every report; (b) rebuild historical cohorts from past index snapshots; (c) add a small "failed names" sleeve.  
Doability: **Hard** for full reconstruction under Kaggle. **Easy** for diagnostic + warning language.

**2. LLM-Primary Execution Path Does Not Yet Participate in the Benchmark Contract**  
Evidence: `results (2).zip` (llm_research_20260524) contains 51-line `research_log.jsonl` + memo praising momentum family (+1.26 research score / +0.87 Sharpe), but **zero held-out shortlist, zero deterministic comparison, zero winner-rule verdict**. Earlier runs showed `llm_calls=0`. The code and wiki require the identical contract for all families.  
Impact: The "ideas" track is not being judged by the rules that define Phase 1 success.  
Fix: Wire `llm_autoresearch` rows into the existing shortlist logic (`HELDOUT_MIN_LLM=10`) or explicitly mark LLM-only runs as exploratory-only with a hard guard.  
Doability: **Easy-Medium**. The shortlist code and guard logic already exist.

**3. Premature Termination + Starvation of the Autonomous Loop**  
Evidence: Repeated `zero_robust_streak_reached` after 6 generations with 0 robust survivors. Latest LLM run: only 51 candidates. Valid rate ~4.1-4.2%. The May 12 improvement plan already flagged this.  
Impact: The search never escapes the regime_momentum basin. Rare high-value ideas are never sampled enough.  
Fix: Raise patience + add hard `MIN_GENERATIONS` floor; explicit reseed with deterministic champions + diverse branches.  
Doability: **Easy**. Pure policy + loop control.

**4. Objective Misalignment — Train "Robust" Score Does Not Predict Held-Out Winner Rule**  
Evidence: Train-to-held-out gaps of +0.79. Shortlist still leans on train-centric metrics. The real winner rule (cost stress, wf_min, subperiods) is applied only at the end in `_build_heldout_report`. LLM memo judges purely on research-phase metrics.  
Impact: Evolution/LLM is rewarded for fitting noise that the final adjudication correctly rejects.  
Fix: Promote cost_stress (≥10bps), wf_min, and subperiod stability into the primary robustness gate and shortlist. Add flattery penalty.  
Doability: **Medium**. Reuses already-computed fields.

### Medium Severity

**5. Mode Collapse / Lack of Diversity Pressure**  
Evidence: Top clusters almost exclusively `regime_momentum:NN:MM` variants. LLM log also concentrates here.  
Fix: Family quotas, diversity penalty on return correlation, forced non-momentum seeds, structural dedup.  
Doability: **Medium**.

**6. Optimistic Cost & Market Impact Model**  
Evidence: Linear `turnover * bps/10000` only. No borrow fees, no impact, no name-specific costs. Stress only goes to 15bps after the fact.  
Fix: Add borrow proxy, make cost stress part of primary selection, optional square-root impact.  
Doability: **Easy-Medium**.

**7. Weak Statistical Power + Multiple Testing**  
Evidence: 1584+ deterministic + thousands of evolved candidates on 110 names × ~2500 days. Only a few independent regimes. +0.10 edge rule is manual, not calibrated.  
Fix: Pre-specify trial count, report effective trials, consider purged walk-forward + embargo, bootstrap the edge.  
Doability: **Medium-Hard**.

**8. VIX/TNX Data & Alignment Subtleties**  
Evidence: yfinance ^VIX/^TNX on equity calendar. Good `detect_lookahead` guard exists, but timing, settlement vs trading hours, and vintage issues unexamined.  
Fix: Document alignment + sensitivity test; consider FRED point-in-time if needed.  
Doability: **Easy** for docs/sensitivity.

### Lower Severity
9. Report & Artifact Stale-Mixing Risk (partially mitigated by recent cleanup).  
10. HF Token / Model Load Fragility ("LLM enabled but calls=0" in multiple summaries).

---

## Verification Section

**During analysis**:
- Full extraction of cells 18/22/24/26 + `_robustness_score_safe`, `walk_forward`, shortlist logic.
- In-memory analysis of all 51 entries in `results (2).zip` research_log.jsonl.
- Cross-check historical summaries for consistent patterns.
- Map every fallacy to exact cell/line or artifact quote.

**After fixes (future runs)**:
- Next `llm_research` run must produce a complete bundle with research log + full held-out shortlist + deterministic comparison + explicit winner-rule verdict using all 8+ diagnostics.
- Verify guardrail test suite still passes.
- If universe changes, provide bias diagnostic diff.
- Update `wiki.md` with any accepted irreducible limitations.

**Success criteria**:
- Every major fallacy backed by concrete code/artifact evidence.
- Top fixes actionable in 1-2 notebook patch cycles.
- Clear distinction between "makes the benchmark more honest" vs "will likely raise Sharpe".

---

## Open Questions

- Should `llm_research` profile be required to produce a full benchmark-admissible verdict, or is it intentionally an "idea generation" stage?
- Is expanding the 110-stock universe (historical survivors or small-cap sleeve) in scope for Phase 1?
- Realistic LLM call budget per Kaggle run before declaring the autonomous loop not viable at current scale?

---

## Next Steps (Recommended Execution Order)

1. Complete deep extraction of robustness scoring, walk-forward, shortlist, and full research_log analysis from `results (2).zip`.
2. Write the final ranked fallacy list with evidence excerpts.
3. Produce doability matrix + minimal patch sketches for the top 4-5 items.
4. Update `memory.md` / `checklist.md` (or create `fallacies.md`).
5. (Optional) Produce a short executive briefing suitable for Kaggle notebook or GitHub.

This plan is deliberately scoped to be executable in a single focused session using only existing artifacts + notebook source.

---

**Generated / Updated**: May 2026 (final update 28 May after full in-memory analysis of `results lastest 28 May.zip` — the newest run with 21 LLM calls and 0 viable signals).  
**Sources analyzed**:
- `results (2).zip` (24 May)
- `results lastest 28 May.zip` (27 May run)
- All prior extracted bundles in `archive/`
- Full notebook source (`autoresearch_v2_final.ipynb`)
- `wiki.md`, `memory.md`, previous improvement plan

**Final delivered location**: `FALLACIES-ANALYSIS-PLAN.md` (project root)