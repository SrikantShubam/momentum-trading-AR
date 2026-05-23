import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CELL = ROOT / "metric_cell_8.py"
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
        "_evolution_robust_fail_reasons",
        "_robustness_failure_counts",
        "_recent_robust_failure_profile",
        "_zero_robust_adaptive_overrides",
        "_generation_diagnosis_events",
        "robustness_score",
        "shortlist_ok",
        "robust_ok",
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


def _load_stop_policy_namespace():
    keep = {"_should_stop_for_zero_robust"}
    nodes = []
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name)
                and t.id in {
                    "EVOLVE_ZERO_ROBUST_PATIENCE",
                    "EVOLVE_ZERO_ROBUST_MIN_GENERATIONS",
                    "EVOLVE_ZERO_ROBUST_MIN_VALID_RATE",
                }
                for t in node.targets
            ):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC_CELL), "exec"), ns)
    return ns


def _load_tournament_namespace():
    keep = {"_tournament_select"}
    nodes = []
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name)
                and t.id in {"EVOLVE_TOURNAMENT_K", "EVOLVE_DIVERSITY_PENALTY"}
                for t in node.targets
            ):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {"np": np, "pd": pd}
    ns["_family_capped_rows"] = lambda rows, limit: list(rows[:limit])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC_CELL), "exec"), ns)
    return ns


def _load_report_helper_namespace():
    keep = {"_artifact_status_is_partial", "_llm_primary_execution_gap"}
    nodes = []
    for node in ast.parse(REPORT_CELL.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPORT_CELL), "exec"), ns)
    return ns


def _load_heldout_guard_namespace():
    keep = {"_write_no_winner_export_guards"}
    nodes = []
    for node in ast.parse(HELDOUT_CELL.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELDOUT_CELL), "exec"), ns)
    return ns


def _load_evolution_artifact_namespace():
    keep = {
        "_atomic_write_text",
        "_atomic_write_json",
        "row_identity",
        "_clean_rows_for_json",
        "_score_sort_value",
        "_sorted_artifact_rows",
        "_best_score_from_rows",
        "_benchmark_policy_payload",
        "_evolution_policy_payload",
        "_run_flags_payload",
        "_portfolio_construction_payload",
        "_evolution_summary_payload",
        "_run_manifest_payload",
        "_partial_report_payload",
        "_artifact_stop_reason",
        "_write_evolution_artifacts",
    }
    nodes = []
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name)
                and (
                    t.id.startswith("EVOLVE_")
                    or t.id
                    in {
                        "SELECTION_OBJECTIVE",
                        "BETA_LIMIT",
                        "TURNOVER_LIMIT",
                        "MIN_ACTIVE_TURNOVER",
                        "MIN_SIGNAL_ACTIVITY",
                        "MIN_RAW_CS_STD",
                        "MIN_LONG_SHORT_FRAC",
                        "WF_WINDOWS",
                        "ROBUSTNESS_SCORE_FLOOR",
                        "ECONOMIC_SHARPE_FLOOR",
                        "ECONOMIC_EDGE_OVER_DETERMINISTIC",
                        "SCHEMA_VERSION",
                        "_WARNINGS",
                    }
                )
                for t in node.targets
            ):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)

    ns = {"np": np, "json": json, "time": __import__("time"), "sys": __import__("sys")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC_CELL), "exec"), ns)
    ns["_safe_load_memo_text"] = lambda: ""
    return ns


def _load_backtest_namespace():
    ns = {"np": np, "pd": pd}
    exec(compile(BACKTEST_CELL.read_text(), str(BACKTEST_CELL), "exec"), ns)
    return ns


class EvolutionGuardrailTests(unittest.TestCase):
    def test_partial_evolution_writer_reports_checkpoint_in_progress_before_completion(self):
        ns = _load_evolution_artifact_namespace()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ns["PARAM_SEARCH_FILE"] = tmp / "param_search_results.json"
            ns["EVOLUTION_LINEAGE_FILE"] = tmp / "evolution_lineage.json"
            ns["EVOLUTION_SUMMARY_FILE"] = tmp / "evolution_summary.json"
            ns["PARTIAL_REPORT_FILE"] = tmp / "partial_run_report.json"
            ns["EVOLUTION_MEMORY_FILE"] = tmp / "evolution_memory.json"
            ns["EVOLUTION_PROGRAM_FILE"] = tmp / "evolution_program.json"
            ns["RUN_MANIFEST_FILE"] = tmp / "run_manifest.json"

            ns["_write_evolution_artifacts"](
                [{"parent_id": "candidate-a", "score": 1.2, "robust_ok": True}],
                [{"gen_id": 2, "best_score": 1.2}],
                [],
                [],
                [],
                2,
                "max_generations_reached",
                partial=True,
            )

            summary = json.loads(ns["EVOLUTION_SUMMARY_FILE"].read_text())
            report = json.loads(ns["PARTIAL_REPORT_FILE"].read_text())
            manifest = json.loads(ns["RUN_MANIFEST_FILE"].read_text())

        self.assertEqual(summary["artifact_status"], "partial")
        self.assertEqual(summary["generations_executed"], 1)
        self.assertEqual(summary["stop_reason"], "checkpoint_in_progress")
        self.assertEqual(report["stop_reason"], "checkpoint_in_progress")
        self.assertEqual(manifest["result"]["stop_reason"], "checkpoint_in_progress")

    def test_final_evolution_writer_can_report_max_generations_reached(self):
        ns = _load_evolution_artifact_namespace()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ns["PARAM_SEARCH_FILE"] = tmp / "param_search_results.json"
            ns["EVOLUTION_LINEAGE_FILE"] = tmp / "evolution_lineage.json"
            ns["EVOLUTION_SUMMARY_FILE"] = tmp / "evolution_summary.json"
            ns["PARTIAL_REPORT_FILE"] = tmp / "partial_run_report.json"
            ns["EVOLUTION_MEMORY_FILE"] = tmp / "evolution_memory.json"
            ns["EVOLUTION_PROGRAM_FILE"] = tmp / "evolution_program.json"
            ns["RUN_MANIFEST_FILE"] = tmp / "run_manifest.json"

            ns["_write_evolution_artifacts"](
                [{"parent_id": "candidate-a", "score": 1.2, "robust_ok": True}],
                [{"gen_id": ns["EVOLVE_MAX_GENERATIONS"], "best_score": 1.2}],
                [],
                [],
                [],
                ns["EVOLVE_MAX_GENERATIONS"],
                "max_generations_reached",
                partial=False,
            )

            summary = json.loads(ns["EVOLUTION_SUMMARY_FILE"].read_text())
            report = json.loads(ns["PARTIAL_REPORT_FILE"].read_text())
            manifest = json.loads(ns["RUN_MANIFEST_FILE"].read_text())

        self.assertEqual(summary["artifact_status"], "final")
        self.assertEqual(summary["stop_reason"], "max_generations_reached")
        self.assertEqual(report["stop_reason"], "max_generations_reached")
        self.assertEqual(manifest["result"]["stop_reason"], "max_generations_reached")

    def test_checkpoint_stop_reason_preserves_explicit_stop_conditions(self):
        ns = _load_evolution_artifact_namespace()
        explicit_reasons = [
            "max_total_evals_reached",
            "max_wallclock_reached",
            "search_space_exhausted",
            "zero_robust_streak_reached",
            "stagnation_reached",
            "convergence_reached",
        ]

        for reason in explicit_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(ns["_artifact_stop_reason"](reason, partial=True), reason)

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

    def test_tournament_selection_handles_rows_with_series_payloads(self):
        ns = _load_tournament_namespace()

        class FirstKSampler:
            def sample(self, population, k):
                return list(population[:k])

        dates = pd.bdate_range("2024-01-01", periods=40)
        rows = [
            {
                "parent_id": "a",
                "signature": "sig-a",
                "cluster_id": "cluster-a",
                "score": 1.0,
                "score_train": 1.0,
                "robust_ok": True,
                "_val_ret": pd.Series(np.linspace(0.0, 0.01, len(dates)), index=dates),
            },
            {
                "parent_id": "b",
                "signature": "sig-b",
                "cluster_id": "cluster-b",
                "score": 0.9,
                "score_train": 0.9,
                "robust_ok": True,
                "_val_ret": pd.Series(np.linspace(0.01, 0.0, len(dates)), index=dates),
            },
        ]

        selected = ns["_tournament_select"](rows, 2, FirstKSampler())

        self.assertEqual(["a", "b"], [row["parent_id"] for row in selected])

    def test_zero_robust_guardrails_do_not_stop_when_validity_is_healthy(self):
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
        self.assertGreaterEqual(constants["EVOLVE_ZERO_ROBUST_PATIENCE"], 3)
        self.assertLessEqual(constants["EVOLVE_ZERO_ROBUST_MIN_GENERATIONS"], 3)
        self.assertLessEqual(constants["EVOLVE_ZERO_ROBUST_MIN_VALID_RATE"], 0.05)
        self.assertEqual(constants["EVOLVE_BENCHMARK_SOURCE"], "deterministic")
        self.assertEqual(constants["EVOLVE_BENCHMARK_MUTATION"], "regime_momentum")

        ns = _load_stop_policy_namespace()
        should_stop = ns["_should_stop_for_zero_robust"]
        self.assertFalse(should_stop(gen_id=3, zero_robust_streak=3, valid_rate=0.98))
        self.assertTrue(should_stop(gen_id=3, zero_robust_streak=3, valid_rate=0.01))

    def test_evolution_policy_payload_exposes_benchmark_and_stagnation_metadata(self):
        ns = _load_diagnosis_namespace()

        policy = ns["_evolution_policy_payload"]()

        self.assertEqual(policy["stagnation_patience"], ns["EVOLVE_STAGNATION_PATIENCE"])
        self.assertEqual(policy["benchmark_policy"]["source"], "deterministic")
        self.assertEqual(policy["benchmark_policy"]["anchor"], "best_deterministic_by_train_score")
        self.assertEqual(policy["benchmark_policy"]["fallback_seed_mutation_type"], "regime_momentum")
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

    def test_program_robust_fail_reasons_identify_val_wf_min_floor(self):
        ns = _load_diagnosis_namespace()

        row = {
            "source": "parameter_search_evolution",
            "parent_id": "top-nonrobust",
            "score": 0.921,
            "robust_ok": False,
            "train_sharpe": 0.353,
            "val_sharpe": 0.831,
            "wf_median": 0.12,
            "wf_min": 0.04,
            "val_wf_median": 0.18,
            "val_wf_min": -0.231,
            "beta": 0.02,
            "turnover": 0.03,
            "signal_activity": 0.8,
            "raw_cs_std": 0.08,
            "raw_long_frac": 0.2,
            "raw_short_frac": 0.2,
            "robustness_score": 70.0,
        }

        reasons = ns["_evolution_robust_fail_reasons"](row)

        self.assertEqual(["val_wf_min"], reasons)

    def test_zero_robust_failure_counts_use_explicit_program_reasons(self):
        ns = _load_diagnosis_namespace()

        valid_rows = [
            {
                "source": "parameter_search_evolution",
                "parent_id": "top-nonrobust",
                "score": 0.921,
                "robust_ok": False,
                "train_sharpe": 0.353,
                "val_sharpe": 0.831,
                "wf_median": 0.12,
                "wf_min": 0.04,
                "val_wf_median": 0.18,
                "val_wf_min": -0.231,
                "beta": 0.02,
                "turnover": 0.03,
                "signal_activity": 0.8,
                "raw_cs_std": 0.08,
                "raw_long_frac": 0.2,
                "raw_short_frac": 0.2,
                "robustness_score": 70.0,
            }
        ]
        generation = {
            "gen_id": 2,
            "valid_count": 1,
            "valid_rate": 1.0,
            "robust_count": 0,
            "best_score": 0.921,
            "best_so_far": 0.921,
            "zero_robust_streak": 2,
            "zero_robust_counted": True,
        }

        diagnoses = ns["_generation_diagnosis_events"](generation, valid_rows, [])

        zero_robust = next(d for d in diagnoses if d["kind"] == "zero_robust")
        self.assertEqual(1, zero_robust["failure_counts"]["val_wf_min"])
        self.assertEqual(["val_wf_min"], zero_robust["top_valid_rows"][0]["robust_fail_reasons"])

    def test_zero_robust_adaptive_policy_reacts_to_beta_and_walk_forward_failures(self):
        ns = _load_diagnosis_namespace()
        generation = {
            "gen_id": 4,
            "valid_rate": 0.99,
            "robust_count": 0,
            "zero_robust_streak": 4,
            "zero_robust_counted": True,
        }
        diagnoses = [
            {
                "kind": "zero_robust",
                "failure_counts": {
                    "beta": 2,
                    "val_wf_min": 3,
                    "program_robustness_score": 2,
                },
            }
        ]

        overrides = ns["_zero_robust_adaptive_overrides"](
            generation,
            diagnoses,
            benchmark_row={"mutation_type": "regime_momentum", "short_span": 57, "long_span": 90},
        )

        self.assertTrue(overrides["force_explore"])
        self.assertTrue(overrides["force_simple_components"])
        self.assertEqual("regime_momentum", overrides["focus_variants"][0])
        self.assertIn("vol_scale", overrides["avoid_variants"])
        self.assertIn("family_quotas", overrides)
        self.assertEqual(57, overrides["short_center"])
        self.assertEqual(90, overrides["long_center"])

    def test_robust_failure_profile_counts_beta_and_walk_forward_pressure(self):
        ns = _load_diagnosis_namespace()
        rows = [
            {"score": 1.0, "robust_ok": False, "robust_fail_reasons": ["beta"]},
            {"score": 0.9, "robust_ok": False, "robust_fail_reasons": ["val_wf_min"]},
            {"score": 0.8, "robust_ok": False, "robust_fail_reasons": ["program_robustness_score"]},
        ]

        profile = ns["_recent_robust_failure_profile"](rows)

        self.assertEqual(1, profile["counts"]["beta"])
        self.assertEqual(1, profile["counts"]["val_wf_min"])
        self.assertGreater(profile["beta_pressure"], 0.0)
        self.assertGreater(profile["wf_min_pressure"], 0.0)
        self.assertGreater(profile["program_robustness_pressure"], 0.0)

    def test_program_robustness_gates_are_not_relaxed_by_adaptive_policy(self):
        metric_source = METRIC_CELL.read_text()

        self.assertIn("val_wf_median >= EVOLVE_PROGRAM_MIN_VAL_WF_MEDIAN", metric_source)
        self.assertIn("val_wf_min >= EVOLVE_PROGRAM_MIN_VAL_WF_MIN", metric_source)
        self.assertIn("wf_min > -0.20", metric_source)
        self.assertIn("program_robustness_score >= EVOLVE_PROGRAM_ROBUSTNESS_FLOOR", metric_source)
        self.assertIn("robust_ok(robustness_payload, wf_floor=-0.25)", metric_source)
        self.assertNotIn("wf_floor=-0.30", metric_source)
        self.assertNotIn("EVOLVE_PROGRAM_MIN_VAL_WF_MIN - 0.", metric_source)
        self.assertNotIn("EVOLVE_PROGRAM_ROBUSTNESS_FLOOR -", metric_source)

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
        self.assertIn('"volume_gate": 0.18', metric_source)
        self.assertIn('"volume_confirm": 0.12', metric_source)
        self.assertIn('"regime_momentum": 0.28', metric_source)
        self.assertIn('"composite": 0.18', metric_source)
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

    def test_shared_backtest_uses_lagged_beta_neutral_position_construction(self):
        backtest_source = BACKTEST_CELL.read_text()

        self.assertIn("BETA_NEUTRALIZE_POSITIONS = True", backtest_source)
        self.assertIn("BETA_NEUTRAL_LOOKBACK = 126", backtest_source)
        self.assertIn("def _rolling_asset_betas(close_df, lookback=BETA_NEUTRAL_LOOKBACK):", backtest_source)
        self.assertIn(".shift(1)", backtest_source)
        self.assertIn("def _beta_neutralize_positions(pos, close_df):", backtest_source)
        self.assertIn("beta_exposure = (centered * betas).sum(axis=1)", backtest_source)
        self.assertIn('"beta_neutralized"', backtest_source)
        self.assertIn('"beta_raw"', backtest_source)

    def test_beta_neutralization_reduces_synthetic_market_beta_without_mutating_signal(self):
        ns = _load_backtest_namespace()
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=420)
        columns = ["low_beta_a", "low_beta_b", "high_beta_a", "high_beta_b"]
        beta_vec = pd.Series([-1.0, -0.4, 0.8, 1.6], index=columns)
        market = pd.Series(rng.normal(0.0005, 0.012, len(dates)), index=dates)
        noise = pd.DataFrame(rng.normal(0.0, 0.002, (len(dates), len(columns))), index=dates, columns=columns)
        returns = noise.add(market.values[:, None] * beta_vec.values, axis=1)
        close = 100.0 * (1.0 + returns).cumprod()
        signal = pd.DataFrame(np.tile(beta_vec.values, (len(dates), 1)), index=dates, columns=columns)
        before = signal.copy(deep=True)

        metrics = ns["backtest"](signal, close, cost_bps=0.0)

        pd.testing.assert_frame_equal(signal, before)
        self.assertTrue(metrics["beta_neutralized"])
        self.assertLess(abs(metrics["beta_neutralized_value"]), abs(metrics["beta_raw"]))
        self.assertGreater(metrics["beta_reduction"], 0.0)

    def test_evolution_scores_beta_drift_instead_of_censoring_volume_families(self):
        metric_source = METRIC_CELL.read_text()

        self.assertNotIn("SIGNAL_VALIDATION: beta drift before walk-forward", metric_source)
        self.assertIn('beta_gate_status = "scored_beta_high"', metric_source)
        self.assertIn('"beta_gate_status": beta_gate_status', metric_source)
        self.assertIn('"beta_raw": beta_raw', metric_source)
        self.assertIn('"beta_neutralized_value": beta_val', metric_source)
        self.assertIn('"beta_reduction": beta_reduction', metric_source)

    def test_evolution_score_penalizes_residual_beta_and_walk_forward_tail_risk(self):
        metric_source = METRIC_CELL.read_text()

        self.assertIn("EVOLVE_BETA_EXCESS_PENALTY", metric_source)
        self.assertIn("EVOLVE_TRAIN_WF_MIN_PENALTY", metric_source)
        self.assertIn("beta_excess_penalty = EVOLVE_BETA_EXCESS_PENALTY", metric_source)
        self.assertIn("train_wf_min_penalty = EVOLVE_TRAIN_WF_MIN_PENALTY", metric_source)
        self.assertIn("val_score -= beta_excess_penalty", metric_source)
        self.assertIn("val_score -= train_wf_min_penalty", metric_source)
        self.assertIn('"beta_excess_penalty": beta_excess_penalty', metric_source)
        self.assertIn('"train_wf_min_penalty": train_wf_min_penalty', metric_source)

    def test_beta_drift_repair_does_not_convert_everything_to_volume_gate(self):
        metric_source = METRIC_CELL.read_text()

        self.assertIn('if "beta drift" in err:', metric_source)
        self.assertIn("return None", metric_source[metric_source.index('if "beta drift" in err:'):metric_source.index('if "dead after market-neutral" in err')])
        self.assertNotIn('repaired = _component_with_variant(base, "volume_gate")\n        repaired["params"]', metric_source)

    def test_notebook_backtest_cell_matches_beta_neutral_construction(self):
        nb = json.loads(NOTEBOOK.read_text())
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        backtest_source = next(src for src in cell_sources if "def backtest(signal_df, close_df" in src and "signal_quality" in src)

        self.assertIn("BETA_NEUTRALIZE_POSITIONS = True", backtest_source)
        self.assertIn("def _rolling_asset_betas(close_df, lookback=BETA_NEUTRAL_LOOKBACK):", backtest_source)
        self.assertIn("def _beta_neutralize_positions(pos, close_df):", backtest_source)

    def test_heldout_export_preserves_program_evolution_code(self):
        heldout_source = HELDOUT_CELL.read_text()
        self.assertIn('fn_code = row.get("code") or deterministic_code(row)', heldout_source)
        self.assertIn("def _oriented_signal_code(code_str, flipped=False):", heldout_source)
        self.assertIn('"code": _oriented_signal_code(raw_code, flipped)', heldout_source)
        self.assertIn('"code_is_oriented": True', heldout_source)
        self.assertIn("def _slice_eval(panel_sig, panel_close, cost_bps=0.0):", heldout_source)
        self.assertIn("sub_m, sub_oriented = _slice_eval(sig.iloc[lo:hi], close_test.iloc[lo:hi])", heldout_source)
        self.assertIn("def _write_no_winner_export_guards(heldout_status, reasons=None):", heldout_source)
        self.assertIn("if approved_winner:", heldout_source)
        self.assertIn("NO APPROVED WINNER - DEPLOYMENT BLOCKED", heldout_source)
        self.assertIn("not exported as deployable code", heldout_source)
        self.assertIn("raise RuntimeError('No approved held-out winner; best_signal.py is non-deployable for this run.')", heldout_source)
        self.assertIn('_write_no_winner_export_guards(\n        "heldout_not_evaluated"', heldout_source)

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
        self.assertIn("if approved_winner:", heldout_source)
        self.assertIn("NO APPROVED WINNER - DEPLOYMENT BLOCKED", heldout_source)
        self.assertIn("not exported as deployable code", heldout_source)
        self.assertIn('_write_no_winner_export_guards(\n        "heldout_not_evaluated"', heldout_source)
        self.assertNotIn('return sorted([evaluate_row_on_test(r) for r in det_ranked]', heldout_source)

    def test_no_winner_export_guard_overwrites_stale_deployable_files(self):
        ns = _load_heldout_guard_namespace()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            best_code = tmp / "best_signal.py"
            ensemble = tmp / "best_signal_ensemble.py"
            best_code.write_text("def signal(close, volume):\n    return close * 0\n")
            ensemble.write_text("def signal(close, volume):\n    return close * 1\n")

            ns["BEST_CODE"] = best_code
            ns["ensemble_file"] = ensemble
            ns["_write_no_winner_export_guards"](
                "heldout_not_evaluated",
                ["held-out evaluation skipped or no surviving candidates"],
            )

            best_text = best_code.read_text()
            ensemble_text = ensemble.read_text()

        self.assertIn("NO APPROVED WINNER - DEPLOYMENT BLOCKED", best_text)
        self.assertIn("heldout_not_evaluated", best_text)
        self.assertIn("No approved held-out winner", best_text)
        self.assertIn("NO APPROVED WINNER - DEPLOYMENT BLOCKED", ensemble_text)
        self.assertIn("best_signal_ensemble.py is non-deployable", ensemble_text)
        self.assertNotIn("return close * 0", best_text)
        self.assertNotIn("return close * 1", ensemble_text)

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
        self.assertIn('def _artifact_status_is_partial(*artifacts):', report_source)
        self.assertIn('legacy_status = str(artifact.get("status", "")).strip().lower()', report_source)
        self.assertIn('if stop_reason == "checkpoint_in_progress":', report_source)
        self.assertIn('deterministic program evolution only; LLM-backed AutoResearch generation/reflection did not execute', report_source)
        self.assertIn('cannot produce an approved winner or deployment gate', report_source)
        self.assertIn('research_result_status = "incomplete" if partial_artifact_status else', report_source)
        self.assertIn('partial_artifact_status = _artifact_status_is_partial(evolution_summary, partial_run_report)', report_source)
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
        self.assertIn('deterministic program evolution only; LLM-backed AutoResearch generation/reflection did not execute', notebook_report)
        self.assertIn('cannot produce an approved winner or deployment gate', notebook_report)
        self.assertIn('partial_artifact_status = _artifact_status_is_partial(evolution_summary, partial_run_report)', notebook_report)
        self.assertIn('legacy_status = str(artifact.get("status", "")).strip().lower()', notebook_report)
        self.assertIn('if stop_reason == "checkpoint_in_progress":', notebook_report)
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

    def test_report_partial_gate_detects_legacy_checkpoint_shapes(self):
        ns = _load_report_helper_namespace()
        is_partial = ns["_artifact_status_is_partial"]

        self.assertTrue(is_partial({"artifact_status": "partial"}))
        self.assertTrue(is_partial({"status": "partial"}))
        self.assertTrue(is_partial({"phase": "checkpoint"}))
        self.assertTrue(is_partial({"stop_reason": "checkpoint_in_progress"}))
        self.assertFalse(is_partial({"artifact_status": "final", "stop_reason": "max_generations_reached"}))

    def test_report_marks_llm_primary_runs_without_calls_incomplete(self):
        ns = _load_report_helper_namespace()
        has_gap = ns["_llm_primary_execution_gap"]

        self.assertTrue(
            has_gap(
                {
                    "run_profile": "llm_research",
                    "llm_stage_enabled": True,
                    "llm_stage_executed": False,
                    "llm_calls": 0,
                }
            )
        )
        self.assertFalse(
            has_gap(
                {
                    "run_profile": "llm_research",
                    "llm_stage_enabled": True,
                    "llm_stage_executed": True,
                    "llm_calls": 2,
                }
            )
        )
        self.assertFalse(
            has_gap(
                {
                    "run_profile": "benchmark",
                    "llm_stage_enabled": False,
                    "llm_stage_executed": False,
                    "llm_calls": 0,
                }
            )
        )

    def test_llm_strict_signal_path_accepts_optional_regime_inputs(self):
        build_source = (ROOT / "build_notebook_v2.py").read_text(encoding="utf-8")

        self.assertIn("def run_signal_code(code_str, close_df, volume_df, vix_s=None, tnx_s=None", build_source)
        self.assertIn('result[0] = ns["signal"](', build_source)
        self.assertIn("vix_s.copy() if vix_s is not None else None", build_source)
        self.assertIn("tnx_s.copy() if tnx_s is not None else None", build_source)
        self.assertIn("def detect_lookahead(code_str, close_df, volume_df, vix_s=None, tnx_s=None", build_source)
        self.assertIn("Write: def signal(close, volume, vix=None, tnx=None) -> pd.DataFrame", build_source)
        self.assertIn("def signal(close, volume, vix=None, tnx=None):", build_source)

    def test_zero_robust_adaptive_quota_limits_failed_volume_branches(self):
        constants = _constant_assignments(["EVOLVE_ADAPTIVE_SIMPLE_FAMILY_QUOTAS"])
        quotas = constants["EVOLVE_ADAPTIVE_SIMPLE_FAMILY_QUOTAS"]

        self.assertLessEqual(quotas["volume_gate"] + quotas["volume_confirm"], 0.28)
        self.assertGreaterEqual(quotas["regime_momentum"], 0.32)
        self.assertGreaterEqual(quotas["rank_norm"] + quotas["regime_gate"], 0.24)

    def test_heldout_shortlist_admits_scored_llm_research_rows(self):
        heldout_source = HELDOUT_CELL.read_text()

        self.assertIn("HELDOUT_MIN_LLM = 10", heldout_source)
        self.assertIn("def _safe_load_llm_rows_local():", heldout_source)
        self.assertIn('candidate["source"] = "llm_autoresearch"', heldout_source)
        self.assertIn('llm_rows = [r for r in active_scope_rows if r.get("source") == "llm_autoresearch"]', heldout_source)
        self.assertIn("picked.extend(_pick_diverse(llm_rows, HELDOUT_MIN_LLM", heldout_source)

    def test_notebook_enables_llm_research_mode_by_default(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell_sources = [
            "".join(cell.get("source", ""))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        ]
        config_source = next(src for src in cell_sources if "DEFAULT_LLM_RESEARCH_MODE" in src and "RUN_LLM_STAGE" in src)
        model_source = next(src for src in cell_sources if "REQUESTED_LLM_MODEL" in src and "bnb_model_load_disabled" in src)

        self.assertIn('DEFAULT_LLM_RESEARCH_MODE = (RUN_PROFILE == "llm_research" and not BENCHMARK_MODE)', config_source)
        self.assertIn('RUN_LLM_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=DEFAULT_LLM_RESEARCH_MODE)', config_source)
        self.assertIn('RUN_BNB_MODEL_LOAD = _env_truthy(', config_source)
        self.assertIn('default=(DEFAULT_LLM_RESEARCH_MODE or RUN_LLM_SMOKE)', config_source)
        self.assertIn('RUN_LLM_SMOKE = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)', config_source)
        self.assertIn("REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)", model_source)
        self.assertIn("SHOULD_LOAD_LLM_MODEL = bool(REQUESTED_LLM_MODEL and RUN_BNB_MODEL_LOAD)", model_source)
        self.assertIn("if RUN_LLM_SMOKE:", model_source)
        self.assertIn('llm_stage_error="bnb_model_load_disabled"', model_source)
        self.assertIn("def _bnb_config():", model_source)
        self.assertNotIn("SHOULD_LOAD_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE)", model_source)

    def test_builder_scope_does_not_label_executed_families_as_deferred(self):
        build_source = (ROOT / "build_notebook_v2.py").read_text(encoding="utf-8")

        self.assertIn('EXECUTED_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution"]', build_source)
        self.assertIn('ACTIVE_RESULT_SOURCES = ["deterministic", "llm_autoresearch", "parameter_search_evolution"]', build_source)
        self.assertIn('ACTIVE_EXECUTION_SCOPE = "deterministic/classical/strict-llm autoresearch only"', build_source)
        self.assertNotIn('ACTIVE_EXECUTION_SCOPE = "pure autoresearch only"', build_source)
        self.assertIn('DEFAULT_LLM_RESEARCH_MODE = (RUN_PROFILE == "llm_research" and not BENCHMARK_MODE)', build_source)
        self.assertIn('RUN_LLM_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=DEFAULT_LLM_RESEARCH_MODE)', build_source)
        self.assertIn('RUN_BNB_MODEL_LOAD = _env_truthy(', build_source)
        self.assertIn('default=(DEFAULT_LLM_RESEARCH_MODE or RUN_LLM_SMOKE)', build_source)
        self.assertIn('RUN_LLM_SMOKE = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)', build_source)
        self.assertIn("REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)", build_source)
        self.assertIn("SHOULD_LOAD_LLM_MODEL = bool(REQUESTED_LLM_MODEL and RUN_BNB_MODEL_LOAD)", build_source)


if __name__ == "__main__":
    unittest.main()




