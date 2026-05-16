# Phase 1 Momentum Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current guarded notebook into a Phase 1 benchmark harness that honestly activates pure AutoResearch, strengthens benchmark discipline, adds stagnation diagnostics, and packages the run for Kaggle under the approved `2xT4` benchmark design. LSTM and any combination work come only after the standalone AutoResearch stage is honestly resolved.

**Architecture:** Keep the notebook generator and patch layer as the source of truth. Implement benchmark policy and evolution logic in `metric_cell_18.py`, held-out/composite evaluation in `metric_cell_24.py`, final reporting in `metric_cell_28.py`, and notebook/runtime config in `build_notebook_v2.py` plus `patch_autoresearch_guardrails.py`. Preserve the generated `autoresearch_v2_final.ipynb` as a build artifact, not the primary hand-edited source.

**Tech Stack:** Python, Jupyter notebook generation, pandas/numpy, PyTorch/transformers, Kaggle kernels, yfinance-based equity data, local tests via `pytest`

---

### Task 1: Write source-of-truth docs for agents and benchmark scope

**Files:**
- Create: `wiki.md`
- Reference: `docs/superpowers/specs/2026-05-12-phase1-momentum-benchmark-design.md`
- Reference: `outputs/2026-05-12-momentum-autoresearch-analysis.md`
- Reference: `outputs/2026-05-12-momentum-improvement-plan.md`

- [ ] **Step 1: Draft `wiki.md` with the benchmark contract**

Include:
- the Phase 1 objective
- approved method families
- shared data and evaluation rules
- winner rule
- stagnation-diagnosis rule
- source files that future agents must modify
- explicit note that `outputs/*` old 20-stock run is historical only, while `kaggle_run_v3/*` is the current guarded reference

- [ ] **Step 2: Review `wiki.md` for contradictions with the approved spec**

Check that:
- winner margin is `+0.10`
- inputs are enriched and shared
- universe remains `110`
- broader LoRA comparisons are deferred until after Phase 1

### Task 2: Make notebook runtime honest about LLM activation and benchmark mode

**Files:**
- Modify: `build_notebook_v2.py`
- Modify: `patch_autoresearch_guardrails.py`
- Modify: `metric_cell_28.py`
- Regenerate: `autoresearch_v2_final.ipynb`

- [ ] **Step 1: Update notebook config defaults for Phase 1**

Make the generator/patch layer expose:
- benchmark mode flags
- explicit LLM stage enablement
- explicit benchmark-family toggles
- run-scoped artifact naming to avoid stale report confusion

- [ ] **Step 2: Ensure final report states actual runtime mode**

Report must clearly distinguish:
- configured model id
- whether LLM stages were enabled
- whether model load actually occurred
- which benchmark families ran

- [ ] **Step 3: Preserve secret loading without leaking tokens**

Keep:
- Hugging Face token support
- W&B token support

Do not print secret values. Only print presence/absence and active logging/auth status.

### Task 3: Strengthen evolution policy and stagnation diagnosis

**Files:**
- Modify: `metric_cell_18.py`
- Test: `tests/test_evolution_guardrails.py`

- [ ] **Step 1: Add Phase 1 benchmark policy constants**

Add or update constants for:
- benchmark winner margin
- rolling holdout mode flag
- stagnation diagnosis trigger
- diagnosis artifact path
- richer evolution policy metadata

- [ ] **Step 2: Replace silent stagnation with explicit diagnosis logging**

When best-so-far does not improve for 3 generations:
- emit structured diagnosis artifact
- summarize score trend, robustness trend, diversity trend, and top failure reasons
- continue the broader benchmark unless a hard stop condition is reached

- [ ] **Step 3: Improve zero-robust handling**

Keep the existing zero-robust logic, but make it more diagnostic:
- log why candidates failed `robust_ok`
- make reseed behavior explicit in artifacts
- prevent ambiguous summaries that look like healthy frontier progress

- [ ] **Step 4: Add tests for stagnation-diagnosis behavior**

Extend `tests/test_evolution_guardrails.py` to cover:
- diagnosis trigger after 3 stagnant generations
- continuation behavior after diagnosis
- artifact payload shape for the diagnosis record

### Task 4: Implement stricter held-out benchmark and composite winner rule

**Files:**
- Modify: `metric_cell_24.py`
- Modify: `metric_cell_28.py`

- [ ] **Step 1: Upgrade held-out evaluation from single winner scoring to benchmark comparison mode**

Add support for:
- multiple chronological holdout windows
- per-family summaries
- baseline-versus-challenger comparison tables

- [ ] **Step 2: Encode the approved winner rule**

A method only wins if:
- it has highest composite score
- it beats the best deterministic baseline by at least `+0.10`
- it does not lose the majority of supporting diagnostics

Supporting diagnostics must include:
- cost stress
- drawdown
- recent-period stability
- beta behavior
- rolling-window consistency

- [ ] **Step 3: Make report language reflect “winner” vs “no winner yet”**

The report must not imply success when:
- no method clears the minimum margin
- a method wins Sharpe but fails the diagnostic majority rule

### Task 5: Strengthen deterministic and classical baseline coverage inside current search space

**Files:**
- Modify: `metric_cell_18.py`
- Modify: `patch_autoresearch_guardrails.py`
- Modify: `metric_cell_24.py`

- [ ] **Step 1: Expand risk-managed classical baseline emphasis**

Increase benchmark visibility and selection support for:
- regime-aware momentum
- volume-conditioned momentum
- other already-supported risk-managed classical variants that fit the current search space

- [ ] **Step 2: Keep benchmark families apples-to-apples**

Ensure stronger classical baselines use:
- the same enriched features
- the same neutrality pipeline
- the same held-out scoring logic

- [ ] **Step 3: Surface the deterministic champion clearly**

The final report must show:
- best deterministic benchmark
- its composite score
- whether any challenger actually cleared the winner rule against it

### Task 6: Make pure AutoResearch the only active advanced family in this stage

**Files:**
- Modify: `build_notebook_v2.py`
- Modify: `patch_autoresearch_guardrails.py`
- Modify: `metric_cell_28.py`

- [ ] **Step 1: Mark benchmark family configuration honestly**

Keep the broad benchmark roadmap in config/docs if useful, but runtime reporting for this stage must state that the active advanced family is pure AutoResearch only.

- [ ] **Step 2: Add execution-status reporting for deferred families**

The report should distinguish:
- configured in roadmap
- implemented
- executed in this run
- deferred by design

- [ ] **Step 3: Block accidental claims about LSTM, tabular ML, GBT, or combinations**

If they did not execute, the report must say they were not run rather than implying poor performance.

### Task 7: Regenerate notebook artifacts and package for Kaggle

**Files:**
- Modify if needed: `kaggle_submission/kernel-metadata.json`
- Regenerate: `autoresearch_v2_final.ipynb`
- Copy/update: `kaggle_submission/autoresearch_v2_final.ipynb`

- [ ] **Step 1: Regenerate the final notebook from source**

Use the existing generator/patch flow so notebook JSON reflects source changes.

- [ ] **Step 2: Verify the generated notebook contains the new benchmark/reporting logic**

Check for:
- benchmark mode flags
- explicit LLM activation reporting
- stagnation diagnosis references
- winner-rule reporting

- [ ] **Step 3: Sync the notebook into `kaggle_submission`**

Ensure the Kaggle package references the updated notebook and correct kernel metadata.

### Task 8: Local verification and Kaggle launch preparation

**Files:**
- Modify if needed: `tests/test_evolution_guardrails.py`
- Use: `kaggle_submission/*`

- [ ] **Step 1: Run focused local verification**

At minimum run:
- `pytest tests/test_evolution_guardrails.py -v`

- [ ] **Step 2: Do a notebook-generation sanity check**

Verify that source regeneration completes and output notebook exists.

- [ ] **Step 3: Prepare Kaggle push/run command**

Use the existing Kaggle kernel package in `kaggle_submission`.
Expected push path:
- `kaggle kernels push -p kaggle_submission`

- [ ] **Step 4: If local push succeeds, start the Kaggle run**

Capture:
- kernel/version id
- submission timestamp
- any immediate CLI response needed to monitor the run later

## Deferred work after this implementation stage

Do not start these until the pure AutoResearch stage is complete and reviewed:

- standalone Sharpe-optimized LSTM benchmark
- AutoResearch + LSTM combination studies
- tabular ML
- gradient-boosted trees

Combination studies require an evidence gate:

- LSTM must show standalone merit under the shared benchmark contract
- results must suggest complementarity rather than redundant exposure

## Notes for execution

- Do not hand-edit `autoresearch_v2_final.ipynb` unless source regeneration is insufficient for a narrow patch.
- Keep `wiki.md` concise and operational; it should be usable by future agents without rereading the full spec.
- Preserve old artifacts, but make new reporting clearly run-scoped to avoid cross-run confusion.
- The current turn focuses on benchmark-contract and pure AutoResearch honesty upgrades, not implementation of LSTM or any broader model family.
