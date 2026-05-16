log = load_log()
det = [r for r in load_deterministic() if r.get("score") is not None]
good = [e for e in log if e.get("sharpe") is not None]
fail = [e for e in log if e.get("error")]
best_train_det = max(det, key=lambda e: e["score"]) if det else None
best_train_llm = max(good, key=lambda e: e["score"]) if good else None
best_test = max(test_results, key=lambda r: r["test_score"]) if test_results else None
clusters = cluster_summary(det if det else log)
heldout_source_counts = {}
for r in test_results:
    src = r.get("source", "unknown")
    heldout_source_counts[src] = heldout_source_counts.get(src, 0) + 1

md_text = f"""# AutoResearch v2 - Momentum Alpha Discovery

**Model:** {MODEL_ID} (4-bit, Kaggle T4x2)
**Universe:** {len(close_all.columns)} US equities | {close_all.index.min().date()} -> {close_all.index.max().date()}
**Train:** through {TRAIN_END} | **Test (held-out):** after

## Execution state
- baseline sweep: {RUN_BASELINE_SWEEP}
- deterministic search: {RUN_DETERMINISTIC_SEARCH}
- llm stage: {RUN_LLM_STAGE}
- held-out eval: {RUN_HELDOUT_EVAL}
- held-out shortlist evaluated: {len(test_results)}
- held-out sources: {heldout_source_counts if heldout_source_counts else "none"}
- reports: {RUN_REPORTS}

## Deterministic search
- candidates evaluated: {len(det)}
"""
if best_train_det:
    md_text += f"- best deterministic train score: **{best_train_det['score']:+.2f}** ({best_train_det['cluster_id']})\n"
    md_text += f"- best deterministic train Sharpe: **{best_train_det['train_sharpe']:+.2f}**\n"
else:
    md_text += "- deterministic search not run or no valid rows\n"

md_text += "\n## Optional LLM stage\n"
md_text += f"- log entries: {len(log)}\n- successful backtests: {len(good)}\n- failures: {len(fail)}\n"
if best_train_llm:
    md_text += f"- best LLM train score: **{best_train_llm['score']:+.2f}** (iter {best_train_llm['iter']})\n"
else:
    md_text += "- LLM stage skipped or no valid survivors\n"

md_text += "\n## Top clusters\n"
if clusters:
    for r in clusters[:5]:
        md_text += f"- {r['cluster_id']}: bestScore={r['best_score']:+.2f} | trainSh={r['best_sharpe']:+.2f} | count={r['count']} | mut={r['mutation_type']}\n"
else:
    md_text += "- none\n"

md_text += f"""

## Reflection memo
{load_memo()}

## Best on held-out test
"""
if best_test:
    md_text += f"""- iter {best_test['iter']} ({best_test['cluster_id']}): train score={best_test['train_score']:+.2f} -> test score=**{best_test['test_score']:+.2f}**
- train Sh={best_test['train_sharpe']:+.2f} | test Sh={best_test['test_sharpe']:+.2f}
- test AnnRet: {best_test['test_ret']:+.1%} | test DD: {best_test['test_dd']:+.1%} | beta: {best_test['test_beta']:+.2f} | turnover: {best_test['test_turnover']:.2f}
- test excess Sharpe: {best_test['test_excess_sharpe']:+.2f}
- walk-forward median Sh: {best_test['wf_median']:+.2f}

### Hypothesis
{best_test['hypothesis']}

### Code
```python
{best_test['code']}
```
"""
else:
    md_text += "- held-out evaluation paused or no surviving candidate passed filters.\n"

if RUN_REPORTS or RUN_HELDOUT_EVAL:
    SUMMARY_MD.write_text(md_text)
print(md_text)
