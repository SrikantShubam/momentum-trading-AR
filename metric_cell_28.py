def _float_or(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_json_value(path, default):
    if path is None:
        return default
    try:
        if (not path.exists()) or path.stat().st_size == 0:
            return default
        text = path.read_text()
        if not text.strip():
            return default
        value = json.loads(text)
        return default if value is None else value
    except Exception:
        return default


def _as_row_list(value):
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    if isinstance(value, dict):
        for key in ("rows", "results", "candidates", "lineage", "programs"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        return [value] if value.get("score") is not None else []
    return []


def _safe_json_rows(path):
    return _as_row_list(_safe_json_value(path, []))


def _safe_json_dict(path):
    value = _safe_json_value(path, {})
    return value if isinstance(value, dict) else {}


def _path_from_global(name, fallback_name=None):
    path = globals().get(name)
    if path is None and fallback_name and "OUT" in globals():
        path = OUT / fallback_name
    return path


def _safe_rows_from_loader(loader_name, file_name=None):
    loader = globals().get(loader_name)
    if callable(loader):
        try:
            rows = _as_row_list(loader())
            if rows:
                return rows
        except Exception:
            pass
    if file_name:
        return _safe_json_rows(_path_from_global(file_name))
    return []


def _safe_dict_from_global(name, fallback_name=None):
    return _safe_json_dict(_path_from_global(name, fallback_name))


def _safe_text(path):
    try:
        if path is None or (not path.exists()) or path.stat().st_size == 0:
            return ""
        return path.read_text()
    except Exception:
        return ""


def _runtime_scope_metadata():
    return _safe_dict_from_global("RUNTIME_METADATA_FILE")


def _runtime_family_scope_lists():
    runtime_metadata = _runtime_scope_metadata()
    configured = runtime_metadata.get("configured_method_families")
    if not isinstance(configured, list) or not configured:
        configured = list(globals().get("ROADMAP_METHOD_FAMILIES", globals().get("BENCHMARK_METHOD_FAMILIES", [])))
    executed = runtime_metadata.get("executed_method_families")
    if not isinstance(executed, list) or not executed:
        executed = list(globals().get("EXECUTED_METHOD_FAMILIES", []))
    configured = [str(x) for x in configured]
    executed = [str(x) for x in executed]
    deferred = [family for family in configured if family not in executed]
    active_sources = runtime_metadata.get("active_result_sources")
    if not isinstance(active_sources, list) or not active_sources:
        active_sources = list(globals().get("ACTIVE_RESULT_SOURCES", ["deterministic", "parameter_search_evolution"]))
    active_scope = runtime_metadata.get("active_execution_scope") or globals().get(
        "ACTIVE_EXECUTION_SCOPE",
        "deterministic/classical/autoresearch only",
    )
    return {
        "configured": configured,
        "executed": executed,
        "deferred": deferred,
        "active_sources": [str(x) for x in active_sources],
        "active_scope": str(active_scope),
    }


def _row_family_tokens(row):
    if not isinstance(row, dict):
        return set()
    tokens = set()
    for key in (
        "source",
        "family",
        "base_family",
        "benchmark_family",
        "model_family",
        "study_family",
        "study_type",
        "label",
        "name",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip().lower()
        if text:
            tokens.add(text)
    return tokens


def _row_matches_deferred_stage_family(row):
    deferred_aliases = {
        "lstm",
        "lstm_sharpe",
        "tabular",
        "tabular_ml",
        "gbt",
        "gradient_boosted_trees",
        "combination",
        "combination_studies",
        "ensemble_combination",
    }
    tokens = _row_family_tokens(row)
    return any(alias in token for token in tokens for alias in deferred_aliases)


def _robustness_summary(rows):
    values = [_float_or(r.get("robustness_score"), None) for r in rows if r.get("robustness_score") is not None]
    values = [v for v in values if v is not None]
    if not values:
        return None
    robust_ok = len([r for r in rows if r.get("robust_ok")])
    zero_score = len([v for v in values if v <= 0.0])
    return f"avg={sum(values) / len(values):.1f} best={max(values):.1f} robust_ok={robust_ok}/{len(rows)} zero_score={zero_score}"


def _evolution_health_lines(evolution_summary, partial_report=None):
    partial_report = partial_report if isinstance(partial_report, dict) else {}
    generations = evolution_summary.get("generations", []) if isinstance(evolution_summary, dict) else []
    generations = [g for g in generations if isinstance(g, dict)]
    eval_total = sum(int(_float_or(g.get("eval_count"), 0.0)) for g in generations)
    valid_total = sum(int(_float_or(g.get("valid_count"), 0.0)) for g in generations)
    robust_total = sum(int(_float_or(g.get("robust_count"), 0.0)) for g in generations)

    valid_rate = partial_report.get("valid_rate", partial_report.get("overall_valid_rate"))
    if valid_rate is None and eval_total:
        valid_rate = valid_total / max(eval_total, 1)

    zero_robust = partial_report.get("zero_robust_generations")
    if zero_robust is None and generations:
        zero_robust = [g.get("gen_id") for g in generations if int(_float_or(g.get("robust_count"), 0.0)) == 0]

    zero_robust_streak = partial_report.get("zero_robust_streak")
    if zero_robust_streak is None and generations:
        zero_robust_streak = 0
        for g in reversed(generations):
            if int(_float_or(g.get("robust_count"), 0.0)) == 0:
                zero_robust_streak += 1
            else:
                break

    lines = []
    if valid_rate is not None:
        try:
            rate_text = f"{float(valid_rate):.1%}"
        except Exception:
            rate_text = str(valid_rate)
        suffix = f" ({valid_total}/{eval_total} valid/evaluated)" if eval_total else ""
        lines.append(f"valid_rate={rate_text}{suffix}")
    if zero_robust is not None:
        if isinstance(zero_robust, list):
            lines.append(f"zero_robust_generations={len(zero_robust)} latest={zero_robust[-5:] if zero_robust else []}")
        else:
            lines.append(f"zero_robust_generations={zero_robust}")
    if zero_robust_streak:
        lines.append(f"zero_robust_streak={zero_robust_streak}")
    if eval_total:
        lines.append(f"robust_total={robust_total}/{valid_total} valid rows")
    return lines


def _partial_report_line(partial_report):
    if not isinstance(partial_report, dict) or not partial_report:
        return None
    fields = []
    for key in (
        "status",
        "phase",
        "gen_id",
        "generation",
        "generations_executed",
        "valid_rate",
        "robust_count",
        "zero_robust_streak",
        "stop_reason",
        "updated_at",
        "timestamp",
    ):
        if key in partial_report:
            fields.append(f"{key}={partial_report.get(key)}")
    if not fields:
        fields.append("keys=" + ",".join(list(partial_report.keys())[:8]))
    return " ".join(fields)


test_results = [r for r in globals().get("test_results", []) if isinstance(r, dict)]
log = _safe_rows_from_loader("load_log", "LOG_FILE")
scope_meta = _runtime_family_scope_lists()
active_sources = set(scope_meta.get("active_sources", []))
det_all = [r for r in _safe_rows_from_loader("load_deterministic", "DETERMINISTIC_FILE") if r.get("score") is not None]
param_all = [r for r in _safe_rows_from_loader("load_parameter_search", "PARAM_SEARCH_FILE") if r.get("score") is not None]
if "load_search_results" in globals():
    search_rows_all = _safe_rows_from_loader("load_search_results")
else:
    search_rows_all = []
    if "PARAM_SEARCH_FILE" in globals() and PARAM_SEARCH_FILE.exists():
        search_rows_all.extend(_safe_json_rows(PARAM_SEARCH_FILE))
    if "DETERMINISTIC_FILE" in globals() and DETERMINISTIC_FILE.exists():
        search_rows_all.extend(_safe_json_rows(DETERMINISTIC_FILE))
deferred_train_rows = [r for r in search_rows_all if _row_matches_deferred_stage_family(r)]
search_rows = [
    r for r in search_rows_all
    if (not _row_matches_deferred_stage_family(r))
    and (r.get("source") in active_sources or r.get("source") in (None, ""))
]
det = [r for r in det_all if not _row_matches_deferred_stage_family(r)]
param = [r for r in param_all if not _row_matches_deferred_stage_family(r)]
good = [e for e in search_rows if e.get("train_sharpe") is not None]
fail = [e for e in log if e.get("error")]
best_train_det = max(det, key=lambda e: _float_or(e.get("score"), -999.0)) if det else None
best_train_search = max(param, key=lambda e: _float_or(e.get("score"), -999.0)) if param else None
deferred_stage_test_rows = [r for r in test_results if _row_matches_deferred_stage_family(r)]
test_results = [r for r in test_results if not _row_matches_deferred_stage_family(r)]
heldout_report_builder = globals().get("_build_heldout_report")
heldout_report = heldout_report_builder(test_results) if callable(heldout_report_builder) else {}
best_test = heldout_report.get("leader") if isinstance(heldout_report, dict) else None
if best_test is None:
    best_test = max(test_results, key=lambda r: _float_or(r.get("test_score"), -999.0)) if test_results else None
heldout_winner = heldout_report.get("winner") if isinstance(heldout_report, dict) else None
heldout_best_det = heldout_report.get("best_deterministic") if isinstance(heldout_report, dict) else None
heldout_best_evo = heldout_report.get("best_evolution") if isinstance(heldout_report, dict) else None
heldout_comparison = heldout_report.get("comparison", {}) if isinstance(heldout_report, dict) else {}
heldout_rule_reasons = heldout_report.get("reasons", []) if isinstance(heldout_report, dict) else []
heldout_rule_status = heldout_report.get("status", "no_winner_yet") if isinstance(heldout_report, dict) else "no_winner_yet"
try:
    clusters = cluster_summary(search_rows if search_rows else log)
except Exception:
    clusters = []
heldout_source_counts = {}
for r in test_results:
    src = r.get("source", "unknown")
    heldout_source_counts[src] = heldout_source_counts.get(src, 0) + 1
configured_families = list(dict.fromkeys(scope_meta.get("configured", [])))
metadata_executed_families = list(dict.fromkeys(scope_meta.get("executed", [])))
active_execution_scope = scope_meta.get("active_scope", "deterministic/classical/autoresearch only")
if not configured_families:
    configured_families = list(globals().get("ROADMAP_METHOD_FAMILIES", globals().get("BENCHMARK_METHOD_FAMILIES", [])))
inferred_executed_families = []
if det:
    inferred_executed_families.append("deterministic")
if any(r.get("mutation_type") in {"regime_gate", "regime_momentum", "volume_gate", "volume_confirm", "vol_scale", "vol_adjusted"} for r in det):
    inferred_executed_families.append("classical_risk_managed")
if param or heldout_source_counts.get("parameter_search_evolution", 0) > 0:
    inferred_executed_families.append("autoresearch_evolution")
executed_families = list(dict.fromkeys(metadata_executed_families + inferred_executed_families))
deferred_families = [family for family in configured_families if family not in executed_families]
evolution_summary = _safe_dict_from_global("EVOLUTION_SUMMARY_FILE")
evolution_memory = _safe_dict_from_global("EVOLUTION_MEMORY_FILE")
partial_run_report = _safe_dict_from_global("PARTIAL_REPORT_FILE", "partial_run_report.json")
evolution_health_lines = _evolution_health_lines(evolution_summary, partial_run_report)
partial_run_report_line = _partial_report_line(partial_run_report)
search_robustness_line = _robustness_summary(search_rows)

unique_param_sigs = len({r.get("signature") for r in param if r.get("signature")})
param_unique_ratio = unique_param_sigs / max(len(param), 1)
heldout_has_det = heldout_source_counts.get("deterministic", 0) > 0
heldout_has_evo = heldout_source_counts.get("parameter_search_evolution", 0) > 0
heldout_has_approved_winner = bool(heldout_winner)
heldout_winner_is_evo = bool(heldout_winner and heldout_winner.get("source") == "parameter_search_evolution")
best_train_score = _float_or(best_train_search.get("score"), None) if best_train_search else None
best_test_score = _float_or(best_test.get("test_score"), None) if best_test else None
generalization_gap = None
if best_train_score is not None and best_test_score is not None:
    generalization_gap = float(best_train_score - best_test_score)
best_det_test = heldout_best_det or max([r for r in test_results if r.get("source") == "deterministic"], key=lambda r: _float_or(r.get("test_score"), -999.0), default=None)
best_evo_test = heldout_best_evo or max([r for r in test_results if r.get("source") == "parameter_search_evolution"], key=lambda r: _float_or(r.get("test_score"), -999.0), default=None)
evolution_test_edge = None
if best_det_test and best_evo_test:
    evolution_test_edge = float(_float_or(best_evo_test.get("test_score")) - _float_or(best_det_test.get("test_score")))
leader_vs_det_edge = heldout_report.get("score_edge_vs_deterministic") if isinstance(heldout_report, dict) else None
economic_sharpe_floor = globals().get("ECONOMIC_SHARPE_FLOOR", 0.50)
economic_edge_floor = globals().get("ECONOMIC_EDGE_OVER_DETERMINISTIC", 0.05)
approved_winner_edge_floor = globals().get("APPROVED_WINNER_EDGE_OVER_DETERMINISTIC", 0.10)
economic_success = bool(
    best_evo_test
    and best_evo_test.get("test_sharpe", 0.0) >= economic_sharpe_floor
    and (best_det_test is None or evolution_test_edge >= economic_edge_floor)
)

adherence_score = 0
adherence_notes = []
evolution_policy = evolution_summary.get("policy", {}) if isinstance(evolution_summary.get("policy"), dict) else {}
if evolution_summary and evolution_summary.get("generations_executed", 0) >= 20:
    adherence_score += 20
else:
    adherence_notes.append("not enough autonomous generations")
if evolution_policy.get("train_val_firewall"):
    adherence_score += 20
else:
    adherence_notes.append("missing train/validation firewall")
if evolution_memory and evolution_memory.get("generation_reflections"):
    adherence_score += 15
else:
    adherence_notes.append("reflection memory is absent or unused")
if param_unique_ratio >= 0.25:
    adherence_score += 15
else:
    adherence_notes.append("candidate lineage is too duplicate-heavy")
if heldout_has_det and heldout_has_evo:
    adherence_score += 15
else:
    adherence_notes.append("held-out shortlist lacks both baseline anchors and evolved candidates")
if generalization_gap is not None and generalization_gap <= 0.20:
    adherence_score += 15
else:
    adherence_notes.append("train-to-test gap is still large")
if heldout_has_approved_winner and heldout_winner_is_evo:
    adherence_score += 10
else:
    adherence_notes.append("held-out winner rule not met yet")
if not economic_success:
    adherence_notes.append("economic success gate not met")
adherence_score = min(100, adherence_score)
if not heldout_has_approved_winner:
    adherence_score = min(90, adherence_score)
if not economic_success:
    adherence_score = min(85, adherence_score)
research_result_status = "successful" if heldout_has_approved_winner else "not-yet-successful"

runtime_metadata = _runtime_scope_metadata()
configured_model_id = runtime_metadata.get("configured_model_id", globals().get("CONFIGURED_MODEL_ID", globals().get("MODEL_ID", "unknown")))
actual_model_id = runtime_metadata.get("actual_model_id") or globals().get("ACTIVE_MODEL_ID") or "(not loaded)"
llm_enabled = bool(runtime_metadata.get("llm_stage_enabled", globals().get("RUN_LLM_STAGE", False)))
llm_loaded = bool(runtime_metadata.get("llm_stage_loaded", False) or globals().get("ACTIVE_MODEL_ID") or globals().get("model") is not None)
llm_executed = bool(runtime_metadata.get("llm_stage_executed", False))
llm_calls = int(_float_or(runtime_metadata.get("llm_calls"), 0.0))
if llm_loaded and (not actual_model_id or actual_model_id == "(not loaded)"):
    actual_model_id = globals().get("ACTIVE_MODEL_ID") or globals().get("MODEL_ID") or configured_model_id
llm_call_accounting_note = "none"
if "autoresearch_evolution" in executed_families and llm_calls == 0:
    llm_call_accounting_note = "program evolution executed with zero recorded LLM generation/reflection calls"
hf_meta = runtime_metadata.get("token_status", {}).get("hf", {})
hf_state = "present" if hf_meta.get("present") else "missing"
hf_source = hf_meta.get("source", "none")
run_id = runtime_metadata.get("run_id", globals().get("RUN_ID", "unknown"))
runtime_profile = runtime_metadata.get("run_profile", globals().get("RUN_PROFILE", "unknown"))
artifact_scope = runtime_metadata.get("report_scope", globals().get("REPORT_SCOPE", "unscoped"))
benchmark_mode = bool(runtime_metadata.get("benchmark_mode", globals().get("BENCHMARK_MODE", False)))
if "sync_runtime_metadata" in globals():
    try:
        runtime_metadata = sync_runtime_metadata(
            configured_method_families=list(configured_families),
            executed_method_families=list(executed_families),
            deferred_method_families=list(deferred_families),
            active_execution_scope=active_execution_scope,
            actual_model_id=None if actual_model_id == "(not loaded)" else actual_model_id,
            llm_stage_enabled=llm_enabled,
            llm_stage_loaded=llm_loaded,
            llm_stage_executed=llm_executed,
            llm_calls=llm_calls,
        )
        configured_model_id = runtime_metadata.get("configured_model_id", configured_model_id)
        actual_model_id = runtime_metadata.get("actual_model_id") or actual_model_id
    except Exception:
        pass

md_text = f"""# AutoResearch v2 - Momentum Alpha Discovery

**Run:** {run_id} | profile={runtime_profile} | scope={artifact_scope}
**Benchmark mode:** {benchmark_mode}
**Configured model:** {configured_model_id}
**Actual loaded model:** {actual_model_id}
**LLM stage:** enabled={llm_enabled} | loaded={llm_loaded} | executed={llm_executed} | calls={llm_calls}
**LLM call accounting:** {llm_call_accounting_note}
**HF token:** {hf_state} (source={hf_source})
**Universe:** {len(close_all.columns)} US equities | {close_all.index.min().date()} -> {close_all.index.max().date()}
**Train:** through {TRAIN_END} | **Test (held-out):** after
**Selection objective:** market-neutral net Sharpe (`SELECTION_OBJECTIVE={SELECTION_OBJECTIVE}`)

## Method-family scope
- configured roadmap families: {", ".join(configured_families) if configured_families else "none recorded"}
- executed in this stage: {", ".join(executed_families) if executed_families else "none recorded"}
- deferred or not executed in this stage: {", ".join(deferred_families) if deferred_families else "none recorded"}
- active execution scope: {active_execution_scope}
- note: LSTM, tabular ML, GBT, and combination studies were not executed in this stage

## Execution state
- baseline sweep: {RUN_BASELINE_SWEEP}
- deterministic search: {RUN_DETERMINISTIC_SEARCH}
- parameter search: {RUN_PARAM_SEARCH}
- held-out eval: {RUN_HELDOUT_EVAL}
- held-out shortlist evaluated: {len(test_results)}
- held-out sources: {heldout_source_counts if heldout_source_counts else "none"}
- held-out deferred rows ignored: {len(deferred_stage_test_rows)}
- reports: {RUN_REPORTS}
- partial run report: {partial_run_report_line if partial_run_report_line else "not present"}

## Deterministic search
- candidates evaluated: {len(det)}
"""
if best_train_det:
    md_text += f"- best deterministic train score: **{_float_or(best_train_det.get('score')):+.2f}** ({best_train_det.get('cluster_id', 'unknown')})\n"
    md_text += f"- best deterministic train Sharpe: **{_float_or(best_train_det.get('train_sharpe', best_train_det.get('sharpe'))):+.2f}**\n"
else:
    md_text += "- deterministic search not run or no valid rows\n"

md_text += "\n## Parameter search stage\n"
md_text += f"- parameter-search rows in active scope: {len(param)}\n"
md_text += f"- deferred train rows ignored in this stage: {len(deferred_train_rows)}\n"
md_text += f"- combined robust survivors: {len([r for r in search_rows if r.get('robust_ok')])}\n"
if search_robustness_line:
    md_text += f"- robustness score: {search_robustness_line}\n"
if best_train_search:
    md_text += f"- best parameter-search train score: **{_float_or(best_train_search.get('score')):+.2f}** ({best_train_search.get('cluster_id', 'unknown')})\n"
else:
    md_text += "- parameter search skipped or no valid survivors\n"
if (not RUN_PARAM_SEARCH) and len(param) == 0:
    md_text += "- note: held-out shortlist can still be drawn from deterministic survivors when parameter search is paused.\n"
if evolution_summary:
    md_text += (
        f"- evolution generations executed: {evolution_summary.get('generations_executed', 0)} "
        f"(stop: {evolution_summary.get('stop_reason', 'n/a')})\n"
    )
if evolution_health_lines:
    md_text += "- evolution validity/robustness: " + "; ".join(evolution_health_lines) + "\n"
if "RUN_MANIFEST_FILE" in globals() and RUN_MANIFEST_FILE.exists():
    md_text += f"- run manifest: `{RUN_MANIFEST_FILE.name}`\n"
if "EVOLUTION_PROGRAM_FILE" in globals() and EVOLUTION_PROGRAM_FILE.exists():
    md_text += f"- evolution program log: `{EVOLUTION_PROGRAM_FILE.name}`\n"

md_text += "\n## Top clusters\n"
if clusters:
    for r in clusters[:5]:
        md_text += f"- {r.get('cluster_id', 'unknown')}: bestScore={_float_or(r.get('best_score')):+.2f} | trainSh={_float_or(r.get('best_sharpe')):+.2f} | count={r.get('count', 0)} | mut={r.get('mutation_type', 'unknown')}\n"
else:
    md_text += "- none\n"

md_text += "\n## AutoResearch Adherence\n"
md_text += f"- score: **{adherence_score}/100**\n"
md_text += f"- result status: **{research_result_status}**\n"
md_text += f"- held-out verdict: **{heldout_rule_status.replace('_', ' ')}**\n"
md_text += f"- unique parameter signatures: {unique_param_sigs}/{len(param)} ({param_unique_ratio:.1%})\n"
if generalization_gap is not None:
    md_text += f"- best train-to-held-out score gap: {generalization_gap:+.2f}\n"
if best_evo_test:
    md_text += f"- best evolved held-out: score={_float_or(best_evo_test.get('test_score')):+.2f} | Sh={_float_or(best_evo_test.get('test_sharpe')):+.2f}\n"
if best_det_test:
    md_text += f"- best deterministic held-out: score={_float_or(best_det_test.get('test_score')):+.2f} | Sh={_float_or(best_det_test.get('test_sharpe')):+.2f}\n"
if evolution_test_edge is not None:
    md_text += f"- evolved edge over deterministic: {evolution_test_edge:+.2f}\n"
if leader_vs_det_edge is not None:
    md_text += f"- current leader edge over deterministic baseline: {leader_vs_det_edge:+.2f}\n"
if heldout_comparison:
    md_text += (
        "- supporting diagnostics vs deterministic baseline: "
        f"{heldout_comparison.get('wins', 0)} wins / {heldout_comparison.get('losses', 0)} losses / "
        f"{heldout_comparison.get('ties', 0)} ties\n"
    )
if search_robustness_line:
    md_text += f"- train robustness score summary: {search_robustness_line}\n"
if evolution_health_lines:
    md_text += "- evolution zero-robust/valid-rate: " + "; ".join(evolution_health_lines) + "\n"
md_text += f"- economic success gate: evolved Sh >= {economic_sharpe_floor:.2f} and edge >= {economic_edge_floor:+.2f}\n"
md_text += (
    f"- approved held-out winner gate: highest composite score, edge >= {approved_winner_edge_floor:+.2f} "
    "vs best deterministic baseline, and must not lose a majority of supporting diagnostics\n"
)
if adherence_notes:
    md_text += "- open issues: " + "; ".join(adherence_notes) + "\n"
else:
    md_text += "- open issues: none from automated audit\n"
if heldout_rule_reasons and not heldout_has_approved_winner:
    md_text += "- winner rule gaps: " + "; ".join(heldout_rule_reasons) + "\n"

md_text += f"""

## Reflection memo
{load_memo()}

## Held-out verdict
"""
if best_test:
    cs = best_test.get("cost_stress", {})
    cs_line = "n/a"
    if cs:
        keys = [k for k in ("5bps", "10bps", "15bps") if k in cs]
        if keys:
            cs_line = " | ".join([f"{k}:{_float_or(cs.get(k, {}).get('sharpe')):+.2f}" for k in keys])
    subs = best_test.get("subperiods", [])
    sub_line = "n/a"
    if subs:
        sub_line = " | ".join([f"{s.get('label', 'period')}:{_float_or(s.get('sharpe')):+.2f}" for s in subs if isinstance(s, dict)])
    train_rob = best_test.get("train_robustness_score")
    test_rob = best_test.get("test_robustness_score")
    robust_line = "n/a"
    if train_rob is not None or test_rob is not None:
        robust_line = f"train={_float_or(train_rob):.1f}" if train_rob is not None else "train=n/a"
        robust_line += " | " + (f"test={_float_or(test_rob):.1f}" if test_rob is not None else "test=n/a")
    md_text += f"""- leader by composite score: iter {best_test.get('iter')} ({best_test.get('cluster_id', 'unknown')}): train score={_float_or(best_test.get('train_score')):+.2f} -> test score=**{_float_or(best_test.get('test_score')):+.2f}**
- train Sh={_float_or(best_test.get('train_sharpe')):+.2f} | test Sh={_float_or(best_test.get('test_sharpe')):+.2f}
- robustness score: {robust_line}
- test AnnRet: {_float_or(best_test.get('test_ret')):+.1%} | test DD: {_float_or(best_test.get('test_dd')):+.1%} | beta: {_float_or(best_test.get('test_beta')):+.2f} | turnover: {_float_or(best_test.get('test_turnover')):.2f}
- benchmark-spread Sharpe diagnostic: {_float_or(best_test.get('test_benchmark_spread_sharpe')):+.2f}
- walk-forward median/min Sh: {_float_or(best_test.get('wf_median')):+.2f}/{_float_or(best_test.get('wf_min')):+.2f}
- cost-stress test Sh (5/10/15 bps): {cs_line}
- subperiod Sharpe (2015-2018/2019-2021/2022-2024): {sub_line}
"""
    if best_det_test:
        md_text += f"- best deterministic baseline: iter {best_det_test.get('iter')} ({best_det_test.get('cluster_id', 'unknown')}) | test score={_float_or(best_det_test.get('test_score')):+.2f} | test Sh={_float_or(best_det_test.get('test_sharpe')):+.2f}\n"
    if heldout_winner:
        md_text += (
            f"- approved winner: **{heldout_winner.get('cluster_id', 'unknown')}** "
            f"(edge vs deterministic {_float_or(leader_vs_det_edge, 0.0):+.2f}; diagnostics "
            f"{heldout_comparison.get('wins', 0)}W/{heldout_comparison.get('losses', 0)}L/{heldout_comparison.get('ties', 0)}T)\n"
        )
    else:
        md_text += "- approved winner: **no winner yet**\n"
        if heldout_rule_reasons:
            md_text += "- rule gaps: " + "; ".join(heldout_rule_reasons) + "\n"
    if heldout_comparison.get("diagnostics"):
        md_text += "- leader vs deterministic diagnostics:\n"
        for diag in heldout_comparison.get("diagnostics", []):
            md_text += (
                f"  - {diag.get('label')}: leader={_float_or(diag.get('candidate')):+.2f} | "
                f"baseline={_float_or(diag.get('baseline')):+.2f} | delta={_float_or(diag.get('delta')):+.2f} "
                f"| {diag.get('verdict')}\n"
            )

    md_text += f"""
### Hypothesis
{best_test.get('hypothesis', '')}

### Code
```python
{best_test.get('code', '')}
```
"""
else:
    md_text += "- no winner yet: held-out evaluation paused or no surviving candidate passed filters.\n"

if "EVOLUTION_TLDR_FILE" in globals() and EVOLUTION_TLDR_FILE.exists():
    evolution_tldr_text = _safe_text(EVOLUTION_TLDR_FILE)
    if evolution_tldr_text.strip():
        md_text += "\n## Evolution TLDR\n"
        md_text += evolution_tldr_text + "\n"
if evolution_memory and evolution_memory.get("generation_reflections"):
    last_ref = evolution_memory["generation_reflections"][-1]
    md_text += "\n## Latest Generation Reflection\n"
    md_text += "```\n" + str(last_ref.get("reflection", "")) + "\n```\n"

if RUN_REPORTS or RUN_HELDOUT_EVAL:
    SUMMARY_MD.write_text(md_text)
print(md_text)
