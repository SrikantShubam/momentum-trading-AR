import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
METRIC_CELL = ROOT / "metric_cell_18.py"
HELDOUT_CELL = ROOT / "metric_cell_24.py"
REPORT_CELL = ROOT / "metric_cell_28.py"
NOTEBOOK = ROOT / "autoresearch_v2_final.ipynb"


def _module_ast():
    return ast.parse(METRIC_CELL.read_text())


def _constant_assignments(names):
    wanted = set(names)
    values = {}
    for node in _module_ast().body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                values[target.id] = ast.literal_eval(node.value)
    return values


def _load_parent_selection_namespace():
    keep = {"_merge_parent_candidates", "_select_next_generation_parents"}
    nodes = []
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id.startswith("EVOLVE_") for t in node.targets):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {}

    def top_parents(rows, limit):
        return sorted(rows, key=lambda row: -row.get("score", -999.0))[:limit]

    def tournament(rows, limit, rng):
        return list(rows[:limit])

    ns["_top_parents_from_rows"] = top_parents
    ns["_diverse_top_parents_from_rows"] = top_parents
    ns["_family_capped_rows"] = lambda rows, limit: list(rows[:limit])
    ns["_tournament_select"] = tournament
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC_CELL), "exec"), ns)
    return ns


def _load_diagnosis_namespace():
    keep = {
        "_benchmark_policy_payload",
        "_evolution_policy_payload",
        "_robustness_failure_counts",
        "_generation_diagnosis_events",
    }
    nodes = []
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name)
                and (
                    t.id.startswith("EVOLVE_")
                    or t.id in {
                        "ROBUSTNESS_SCORE_FLOOR",
                        "ECONOMIC_SHARPE_FLOOR",
                        "ECONOMIC_EDGE_OVER_DETERMINISTIC",
                        "SELECTION_OBJECTIVE",
                        "BETA_LIMIT",
                        "TURNOVER_LIMIT",
                        "MIN_ACTIVE_TURNOVER",
                        "MIN_SIGNAL_ACTIVITY",
                        "MIN_RAW_CS_STD",
                        "MIN_LONG_SHORT_FRAC",
                    }
                )
                for t in node.targets
            ):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC_CELL), "exec"), ns)
    return ns


class EvolutionGuardrailTests(unittest.TestCase):
    def test_reseeded_robust_parents_are_not_overwritten_by_nonrobust_survivors(self):
        ns = _load_parent_selection_namespace()
        ns["EVOLVE_ROBUST_RESET_THRESHOLD"] = 20
        ns["EVOLVE_BEAM_WIDTH"] = 4
        ns["EVOLVE_SURVIVORS"] = 2

        current = [{"parent_id": "current", "score": 0.1, "robust_ok": False}]
        valid = [
            {"parent_id": "bad-a", "score": 10.0, "robust_ok": False},
            {"parent_id": "bad-b", "score": 9.0, "robust_ok": False},
        ]
        robust = []
        global_rows = [
            {"parent_id": "robust-seed-a", "score": 1.0, "robust_ok": True},
            {"parent_id": "robust-seed-b", "score": 0.9, "robust_ok": True},
        ]

        parents, survivors, did_reseed = ns["_select_next_generation_parents"](
            current,
            valid,
            robust,
            global_rows,
            rng=None,
        )

        self.assertTrue(did_reseed)
        self.assertEqual(["bad-a", "bad-b"], [row["parent_id"] for row in survivors])
        self.assertEqual(
            ["robust-seed-a", "robust-seed-b"],
            [row["parent_id"] for row in parents[:2]],
        )
        self.assertNotIn("bad-a", [row["parent_id"] for row in parents[:2]])

    def test_zero_robust_guardrails_fail_fast(self):
        constants = _constant_assignments(
            [
                "EVOLVE_MAX_GENERATIONS",
                "EVOLVE_MAX_TOTAL_EVALS",
                "EVOLVE_MAX_WALLCLOCK_HOURS",
                "EVOLVE_STAGNATION_PATIENCE",
                "EVOLVE_HARD_STAGNATION_PATIENCE",
                "EVOLVE_ZERO_ROBUST_PATIENCE",
                "EVOLVE_ZERO_ROBUST_MIN_GENERATIONS",
                "EVOLVE_ZERO_ROBUST_MIN_VALID_RATE",
                "EVOLVE_BENCHMARK_SOURCE",
                "EVOLVE_BENCHMARK_MUTATION",
            ]
        )
        self.assertLessEqual(constants["EVOLVE_MAX_GENERATIONS"], 40)
        self.assertLessEqual(constants["EVOLVE_MAX_TOTAL_EVALS"], 10000)
        self.assertLessEqual(constants["EVOLVE_MAX_WALLCLOCK_HOURS"], 2.0)
        self.assertLessEqual(constants["EVOLVE_STAGNATION_PATIENCE"], 3)
        self.assertLessEqual(constants["EVOLVE_HARD_STAGNATION_PATIENCE"], 6)
        self.assertLessEqual(constants["EVOLVE_ZERO_ROBUST_PATIENCE"], 3)
        self.assertLessEqual(constants["EVOLVE_ZERO_ROBUST_MIN_GENERATIONS"], 3)
        self.assertLessEqual(constants["EVOLVE_ZERO_ROBUST_MIN_VALID_RATE"], 0.05)
        self.assertEqual(constants["EVOLVE_BENCHMARK_SOURCE"], "deterministic")
        self.assertEqual(constants["EVOLVE_BENCHMARK_MUTATION"], "regime_momentum")

    def test_evolution_policy_payload_exposes_benchmark_and_stagnation_metadata(self):
        ns = _load_diagnosis_namespace()

        policy = ns["_evolution_policy_payload"]()

        self.assertEqual(policy["stagnation_patience"], ns["EVOLVE_STAGNATION_PATIENCE"])
        self.assertEqual(policy["benchmark_policy"]["source"], "deterministic")
        self.assertIn("selection_objective", policy["benchmark_policy"])
        self.assertIn("economic_edge_over_deterministic", policy["benchmark_policy"])

    def test_generation_diagnosis_emits_stagnation_event_after_three_generations(self):
        ns = _load_diagnosis_namespace()

        valid_rows = [
            {
                "parent_id": "cand-a",
                "score": 1.11,
                "robust_ok": True,
                "cluster_id": "cluster-a",
                "wf_median": 0.24,
                "wf_min": 0.03,
            }
        ]
        generation = {
            "gen_id": 5,
            "valid_count": 1,
            "valid_rate": 1.0,
            "robust_count": 1,
            "best_score": 1.11,
            "best_so_far": 1.37,
            "best_so_far_streak": 3,
            "zero_robust_streak": 0,
            "new_cluster_count": 0,
            "diversity": 0.08,
        }

        diagnoses = ns["_generation_diagnosis_events"](
            generation,
            valid_rows,
            [],
            benchmark_row={"parent_id": "det-anchor", "score": 1.33, "mutation_type": "regime_gate"},
        )

        stagnation = next(d for d in diagnoses if d["kind"] == "stagnation")
        self.assertEqual(stagnation["trigger"], "best_so_far_streak")
        self.assertEqual(stagnation["streak"], 3)
        self.assertEqual(stagnation["benchmark"]["parent_id"], "det-anchor")
        self.assertAlmostEqual(stagnation["benchmark_gap"], -0.22, places=6)
        self.assertEqual(stagnation["top_valid_rows"][0]["parent_id"], "cand-a")

    def test_generation_diagnosis_captures_zero_robust_failure_breakdown(self):
        ns = _load_diagnosis_namespace()

        valid_rows = [
            {
                "parent_id": "wf-fail",
                "score": 0.91,
                "robust_ok": False,
                "train_sharpe": 0.3,
                "wf_median": 0.04,
                "wf_min": -0.31,
                "beta": 0.04,
                "turnover": 0.03,
                "signal_activity": 0.7,
                "raw_cs_std": 0.05,
                "raw_long_frac": 0.2,
                "raw_short_frac": 0.2,
                "robustness_score": 39.0,
            },
            {
                "parent_id": "activity-fail",
                "score": 0.88,
                "robust_ok": False,
                "train_sharpe": 0.5,
                "wf_median": 0.2,
                "wf_min": 0.1,
                "beta": 0.03,
                "turnover": 0.03,
                "signal_activity": 0.3,
                "raw_cs_std": 0.01,
                "raw_long_frac": 0.04,
                "raw_short_frac": 0.03,
                "robustness_score": 42.0,
            },
        ]
        generation = {
            "gen_id": 3,
            "valid_count": 2,
            "valid_rate": 1.0,
            "robust_count": 0,
            "best_score": 0.91,
            "best_so_far": 1.2,
            "best_so_far_streak": 2,
            "zero_robust_streak": 3,
            "zero_robust_counted": True,
        }

        diagnoses = ns["_generation_diagnosis_events"](generation, valid_rows, [])

        zero_robust = next(d for d in diagnoses if d["kind"] == "zero_robust")
        self.assertEqual(zero_robust["trigger"], "zero_robust_streak")
        self.assertEqual(zero_robust["streak"], 3)
        self.assertEqual(zero_robust["failure_counts"]["wf_min"], 1)
        self.assertEqual(zero_robust["failure_counts"]["signal_activity"], 1)
        self.assertEqual(zero_robust["failure_counts"]["raw_cs_std"], 1)
        self.assertEqual(zero_robust["failure_counts"]["long_short_balance"], 1)
        self.assertEqual(
            ["wf-fail", "activity-fail"],
            [row["parent_id"] for row in zero_robust["top_valid_rows"]],
        )

    def test_notebook_cell_uses_same_evolution_guardrails(self):
        nb = json.loads(NOTEBOOK.read_text())
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        metric_source = next(src for src in cell_sources if src.startswith("BASELINE_RESULTS = []"))

        self.assertIn("EVOLVE_MAX_GENERATIONS = 40", metric_source)
        self.assertIn("EVOLVE_ZERO_ROBUST_PATIENCE = 3", metric_source)
        self.assertIn('EVOLVE_BENCHMARK_MUTATION = "regime_momentum"', metric_source)
        self.assertIn("EVOLVE_CHAMPION_VARIANTS", metric_source)
        self.assertIn("EVOLVE_ADAPTIVE_DIAGNOSTIC_RUN = True", metric_source)
        self.assertIn("EVOLVE_FAMILY_QUOTAS", metric_source)
        self.assertIn("EVOLVE_PARENT_FAMILY_CAP = 0.40", metric_source)
        self.assertIn("EVOLVE_VAL_WF_FLOOR_PENALTY", metric_source)
        self.assertIn("_select_next_generation_parents", metric_source)
        self.assertIn("zero_robust_streak_reached", metric_source)
        self.assertNotIn("EVOLVE_MAX_GENERATIONS = 200", metric_source)

    def test_evolution_repairs_search_collapse_with_family_quotas(self):
        metric_source = METRIC_CELL.read_text()

        self.assertIn("def _entry_variant_tags(entry):", metric_source)
        self.assertIn('if "dead after market-neutral normalization" in err:', metric_source)
        self.assertIn('if min(raw_long_frac, raw_short_frac) < MIN_LONG_SHORT_FRAC:', metric_source)
        self.assertIn('issues.append("bad_composite_anchor")', metric_source)
        self.assertIn('if dominant_cluster_frac >= 0.50:', metric_source)
        self.assertIn('stop_reason = "stagnation_reached"', metric_source)
        self.assertIn('deduped = {}', metric_source)
        self.assertIn('fallback_focus = ["regime_momentum", "volume_confirm", "volume_gate", "rank_norm", "plain", "regime_gate", "ts_momentum"]', metric_source)
        self.assertIn('"volume_gate": 0.24', metric_source)
        self.assertIn('"volume_confirm": 0.20', metric_source)
        self.assertIn('"composite": 0.16', metric_source)
        self.assertIn('for family, quota in quota_counts.items():', metric_source)
        self.assertIn('f"family_quota:{family}"', metric_source)
        self.assertIn("def _repair_program_after_failure(program, error_text, gen_program=None, rng=None):", metric_source)
        self.assertIn('if "beta drift" in err:', metric_source)
        self.assertIn('if "dead after market-neutral" in err or "not genuinely long-short" in err or "low signal activity" in err:', metric_source)
        self.assertIn('op_choices = ["tweak", "tweak", "tweak", "swap_variant", "swap_variant", "reweight", "add"]', metric_source)
        self.assertIn("def _diverse_top_parents_from_rows(rows, k):", metric_source)
        self.assertIn("for variant in EVOLVE_CHAMPION_VARIANTS:", metric_source)
        self.assertIn("def _family_capped_rows(rows, limit, cap_frac=EVOLVE_PARENT_FAMILY_CAP):", metric_source)
        self.assertIn("family_telemetry = _generation_family_telemetry(candidates, gen_rows, survivors)", metric_source)
        self.assertIn("- EVOLVE_VAL_WF_FLOOR_PENALTY * wf_floor_shortfall", metric_source)
        self.assertIn("val_wf_missing_penalty", metric_source)
        self.assertIn("sig_w = sig_core.iloc[lo:hi]", metric_source)
        self.assertIn("sig_w = sig_val.iloc[lo:hi]", metric_source)

    def test_heldout_export_preserves_program_evolution_code(self):
        heldout_source = HELDOUT_CELL.read_text()
        self.assertIn('fn_code = row.get("code") or deterministic_code(row)', heldout_source)
        self.assertIn("def _oriented_signal_code(code_str, flipped=False):", heldout_source)
        self.assertIn('"code": _oriented_signal_code(raw_code, flipped)', heldout_source)
        self.assertIn('"code_is_oriented": True', heldout_source)
        self.assertIn("def _slice_eval(panel_sig, panel_close, cost_bps=0.0):", heldout_source)
        self.assertIn("sub_m, sub_oriented = _slice_eval(sig.iloc[lo:hi], close_test.iloc[lo:hi])", heldout_source)

    def test_notebook_heldout_cell_matches_safe_export_logic(self):
        nb = json.loads(NOTEBOOK.read_text())
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        heldout_source = next(src for src in cell_sources if src.startswith("TOP_K = 5\nHELDOUT_SHORTLIST_K = 20"))
        self.assertIn('fn_code = row.get("code") or deterministic_code(row)', heldout_source)
        self.assertIn("def _oriented_signal_code(code_str, flipped=False):", heldout_source)
        self.assertIn('"code": _oriented_signal_code(raw_code, flipped)', heldout_source)
        self.assertIn("def _slice_eval(panel_sig, panel_close, cost_bps=0.0):", heldout_source)
        self.assertIn('evaluated = []', heldout_source)
        self.assertIn('held-out eval skipped', heldout_source)
        self.assertNotIn('return sorted([evaluate_row_on_test(r) for r in det_ranked]', heldout_source)

    def test_heldout_logic_caps_mutation_crowding_and_labels_evolution_rows(self):
        heldout_source = HELDOUT_CELL.read_text()
        self.assertIn('HELDOUT_MAX_PER_MUTATION = 4', heldout_source)
        self.assertIn('if per_mutation.get(mutation_type, 0) >= HELDOUT_MAX_PER_MUTATION:', heldout_source)
        self.assertIn('return f"evo:{cluster_id}:{signature}"', heldout_source)

    def test_regime_momentum_uses_both_spans_everywhere(self):
        metric_source = METRIC_CELL.read_text()
        heldout_source = HELDOUT_CELL.read_text()

        self.assertIn('trend_fast = close.pct_change(short_span)', metric_source)
        self.assertIn('trend_slow = close.pct_change(long_span)', metric_source)
        self.assertIn('trend = trend_fast - trend_slow', metric_source)
        self.assertIn('lines.append(f"    ret_fast_{i} = close.pct_change({ss})")', metric_source)
        self.assertIn('lines.append(f"    ret_slow_{i} = close.pct_change({ls})")', metric_source)
        self.assertIn('lines.append(f"    sig_{i} = (((ret_fast_{i} - ret_slow_{i}) * reg_{i}).rank(axis=1,pct=True)-0.5)*2")', metric_source)

        self.assertIn('trend_fast = close.pct_change({short_span})', heldout_source)
        self.assertIn('trend_slow = close.pct_change({long_span})', heldout_source)
        self.assertIn('trend = trend_fast - trend_slow', heldout_source)

    def test_notebook_report_cell_matches_safe_summary_logic(self):
        report_source = REPORT_CELL.read_text()
        self.assertIn('def _runtime_family_scope_lists():', report_source)
        self.assertIn('active_scope = runtime_metadata.get("active_execution_scope") or globals().get(', report_source)
        self.assertIn('configured_families = list(dict.fromkeys(scope_meta.get("configured", [])))', report_source)
        self.assertIn('metadata_executed_families = list(dict.fromkeys(scope_meta.get("executed", [])))', report_source)
        self.assertIn('deferred_families = [family for family in configured_families if family not in executed_families]', report_source)
        self.assertIn('if llm_loaded and (not actual_model_id or actual_model_id == "(not loaded)"):', report_source)
        self.assertIn('program evolution executed with zero recorded LLM generation/reflection calls', report_source)
        self.assertIn('## Method-family scope', report_source)
        self.assertIn('- configured roadmap families: {", ".join(configured_families) if configured_families else "none recorded"}', report_source)
        self.assertIn('- executed in this stage: {", ".join(executed_families) if executed_families else "none recorded"}', report_source)
        self.assertIn('- note: LSTM, tabular ML, GBT, and combination studies were not executed in this stage', report_source)
        self.assertIn('runtime_metadata = sync_runtime_metadata(', report_source)
        self.assertIn('heldout_report_builder = globals().get("_build_heldout_report")', report_source)
        self.assertIn('heldout_rule_status = heldout_report.get("status", "no_winner_yet")', report_source)
        self.assertIn('heldout_has_approved_winner = bool(heldout_winner)', report_source)
        nb = json.loads(NOTEBOOK.read_text())
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        notebook_report = next(src for src in cell_sources if "## AutoResearch Adherence" in src and "heldout_rule_status" in src)
        self.assertIn('def _runtime_family_scope_lists():', notebook_report)
        self.assertIn('## Method-family scope', notebook_report)
        self.assertIn('- executed in this stage: {", ".join(executed_families) if executed_families else "none recorded"}', notebook_report)
        self.assertIn('deferred_families = [family for family in configured_families if family not in executed_families]', notebook_report)
        self.assertIn('program evolution executed with zero recorded LLM generation/reflection calls', notebook_report)
        self.assertIn('heldout_report_builder = globals().get("_build_heldout_report")', notebook_report)
        self.assertIn('heldout_rule_status = heldout_report.get("status", "no_winner_yet")', notebook_report)
        self.assertIn('heldout_has_approved_winner = bool(heldout_winner)', notebook_report)
        self.assertIn('**Adaptive diagnostic run:** {adaptive_diagnostic_run}', notebook_report)
        self.assertIn('latest generation family telemetry', notebook_report)
        self.assertIn('adaptive diagnostic caveat', notebook_report)
        self.assertIn('evolved failure mode diagnosis', notebook_report)
        notebook_heldout = next(src for src in cell_sources if src.startswith("TOP_K = 5\nHELDOUT_SHORTLIST_K = 20"))
        self.assertIn('trend_fast = close.pct_change({short_span})', notebook_heldout)
        self.assertIn('trend_slow = close.pct_change({long_span})', notebook_heldout)

    def test_notebook_disables_unsafe_bnb_model_load_by_default(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        config_source = next(src for src in cell_sources if "RUN_BASELINE_SWEEP" in src and "RUN_LLM_STAGE" in src)
        model_source = next(src for src in cell_sources if "REQUESTED_LLM_MODEL" in src and "bnb_model_load_disabled" in src)

        self.assertIn('RUN_LLM_STAGE            = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=False)', config_source)
        self.assertIn('RUN_BNB_MODEL_LOAD       = _env_truthy(os.getenv("AUTORESEARCH_ENABLE_BNB_LOAD"), default=False)', config_source)
        self.assertIn('RUN_LLM_SMOKE            = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)', config_source)
        self.assertIn("REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)", model_source)
        self.assertIn("SHOULD_LOAD_LLM_MODEL = bool(REQUESTED_LLM_MODEL and RUN_BNB_MODEL_LOAD)", model_source)
        self.assertIn("if RUN_LLM_SMOKE:", model_source)
        self.assertIn('LLM_LOAD_ERROR = "bnb_model_load_disabled"', model_source)
        self.assertIn("def _bnb_config():", model_source)
        self.assertNotIn("SHOULD_LOAD_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE)", model_source)

    def test_builder_scope_does_not_label_executed_families_as_deferred(self):
        build_source = (ROOT / "build_notebook_v2.py").read_text(encoding="utf-8")

        self.assertIn('EXECUTED_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution"]', build_source)
        self.assertIn('ACTIVE_EXECUTION_SCOPE = "deterministic/classical/autoresearch only"', build_source)
        self.assertNotIn('ACTIVE_EXECUTION_SCOPE = "pure autoresearch only"', build_source)
        self.assertIn('RUN_LLM_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=False)', build_source)
        self.assertIn('RUN_BNB_MODEL_LOAD = _env_truthy(os.getenv("AUTORESEARCH_ENABLE_BNB_LOAD"), default=False)', build_source)
        self.assertIn('RUN_LLM_SMOKE = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)', build_source)
        self.assertIn("REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)", build_source)
        self.assertIn("SHOULD_LOAD_LLM_MODEL = bool(REQUESTED_LLM_MODEL and RUN_BNB_MODEL_LOAD)", build_source)


if __name__ == "__main__":
    unittest.main()
