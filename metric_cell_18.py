BASELINE_RESULTS = []
DETERMINISTIC_RESULTS = []
BASELINE_FILE = OUT / "baseline_results.json"
SELECTION_OBJECTIVE = "market_neutral_net_sharpe"
BETA_LIMIT = 0.10
TURNOVER_LIMIT = 0.50
MIN_ACTIVE_TURNOVER = 0.002
MIN_SIGNAL_ACTIVITY = 0.50
MIN_RAW_CS_STD = 0.02
MIN_LONG_SHORT_FRAC = 0.08
WF_WINDOWS = 4
NEAR_DUP_SHORT = 2
NEAR_DUP_LONG = 6
EVOLVE_ENABLED = True
EVOLVE_MAX_GENERATIONS = 40
EVOLVE_TRIALS_PER_GENERATION = 160
EVOLVE_BEAM_WIDTH = 16
EVOLVE_SURVIVORS = 4
EVOLVE_MIN_GEN_GROWTH = 0.01
EVOLVE_MAX_STALLED_GENS = 20
EVOLVE_MIN_GENERATIONS = 8
EVOLVE_PARENT_MIN_TRIALS = 12
EVOLVE_PARENT_MAX_TRIALS = 60
EVOLVE_MIN_TRIALS_PER_GENERATION = 100
EVOLVE_MAX_TOTAL_EVALS = 9600
EVOLVE_MAX_WALLCLOCK_HOURS = 2.0
EVOLVE_RANDOM_EXPLORATION_FRAC = 0.20
EVOLVE_REQUIRE_NOVELTY_FOR_STALL = True
EVOLVE_MIN_NEW_CLUSTERS_PER_GEN = 1
EVOLVE_ROBUST_RESET_THRESHOLD = 20
EVOLVE_ZERO_ROBUST_PATIENCE = 3
EVOLVE_ZERO_ROBUST_MIN_GENERATIONS = 3
EVOLVE_ZERO_ROBUST_MIN_VALID_RATE = 0.03
EVOLVE_PARTIAL_WRITE_EVERY = 1
ROBUSTNESS_SCORE_FLOOR = 45.0
EVOLVE_PROGRAM_ROBUSTNESS_FLOOR = 60.0
EVOLVE_PROGRAM_MIN_VAL_WF_MEDIAN = 0.10
EVOLVE_PROGRAM_MIN_VAL_WF_MIN = -0.10
EVOLVE_VAL_WF_FLOOR_PENALTY = 2.00
EVOLVE_VAL_WF_MISSING_PENALTY = 0.35
EVOLVE_MIN_FOCUS_VARIANTS = 5
EVOLVE_PROXY_WF_SPREAD_PENALTY = 0.25
EVOLVE_PROXY_WFMIN_PENALTY = 0.20
EVOLVE_PROXY_CONSISTENCY_PENALTY = 0.10
EVOLVE_TOURNAMENT_K = 5
EVOLVE_DIVERSITY_PENALTY = 0.15
EVOLVE_CONVERGENCE_PATIENCE = 25
EVOLVE_STAGNATION_PATIENCE = 3
EVOLVE_HARD_STAGNATION_PATIENCE = 6
EVOLVE_MIN_BEST_GAIN = 0.003
EVOLVE_MIN_MEDIAN_GAIN = 0.002
EVOLVE_MIN_NOVELTY_GAIN = 1
EVOLVE_MIN_DIVERSITY_GAIN = 0.01
EVOLVE_BENCHMARK_SOURCE = "deterministic"
EVOLVE_BENCHMARK_MUTATION = "regime_momentum"
EVOLVE_CHAMPION_VARIANTS = ["regime_momentum", "volume_gate", "volume_confirm", "rank_norm", "plain", "regime_gate", "ts_momentum"]
EVOLVE_ADAPTIVE_DIAGNOSTIC_RUN = True
EVOLVE_PARENT_FAMILY_CAP = 0.40
EVOLVE_FAMILY_QUOTAS = {
    "volume_gate": 0.24,
    "volume_confirm": 0.20,
    "regime_momentum": 0.20,
    "rank_norm": 0.12,
    "plain": 0.08,
    "composite": 0.16,
}
EVOLVE_SAFE_COMPOSITE_PAIRS = [
    ("volume_gate", "rank_norm"),
    ("volume_confirm", "plain"),
    ("volume_gate", "regime_gate"),
    ("volume_confirm", "ts_momentum"),
]
EVOLVE_DIAGNOSIS_TOP_ROWS = 5
VAL_FRACTION = 0.20
EVAL_TIMEOUT_SEC = 25
EVOLVE_RESTART_ATTEMPTS = 6
EVOLVE_MIN_CANDIDATE_FLOOR = 32
EVOLVE_DETERMINISTIC_CHAMPIONS = 12
EVOLVE_COMPLEXITY_PENALTY = 0.025
EVOLVE_TRAIN_VAL_GAP_PENALTY = 1.25
EVOLVE_VOL_SCALE_PENALTY = 0.015
ECONOMIC_SHARPE_FLOOR = 0.50
ECONOMIC_EDGE_OVER_DETERMINISTIC = 0.05
EVOLUTION_SUMMARY_FILE = OUT / "evolution_summary.json"
EVOLUTION_LINEAGE_FILE = OUT / "evolution_lineage.json"
EVOLUTION_MEMORY_FILE = OUT / "evolution_memory.json"
EVOLUTION_PROGRAM_FILE = OUT / "evolution_program.json"
RUN_MANIFEST_FILE = OUT / "run_manifest.json"
PARTIAL_REPORT_FILE = OUT / "partial_run_report.json"
SCHEMA_VERSION = "evolution-v3"
_WARNINGS = []


def _warn(msg):
    _WARNINGS.append(str(msg))
    print(f"[warn] {msg}")


def _atomic_write_text(path_obj, text):
    tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path_obj)


def _atomic_write_json(path_obj, obj):
    _atomic_write_text(path_obj, json.dumps(obj, indent=2))
    try:
        if "wandb_log" in globals():
            stem = getattr(path_obj, "stem", "json")
            payload = {}
            if isinstance(obj, list):
                rows = [r for r in obj if isinstance(r, dict)]
                payload[f"{stem}/rows"] = len(rows)
                payload[f"{stem}/errors"] = sum(1 for r in rows if r.get("error"))
                payload[f"{stem}/robust"] = sum(1 for r in rows if r.get("robust_ok"))
                scored = [r for r in rows if isinstance(r.get("score"), (int, float))]
                if scored:
                    best = max(scored, key=lambda r: r["score"])
                    payload[f"{stem}/best_score"] = float(best["score"])
                    for key in ("train_sharpe", "wf_median", "wf_min", "beta", "turnover"):
                        if isinstance(best.get(key), (int, float)):
                            payload[f"{stem}/best_{key}"] = float(best[key])
            elif isinstance(obj, dict):
                for key in ("generations_executed", "total_evals", "best_score"):
                    if isinstance(obj.get(key), (int, float)):
                        payload[f"{stem}/{key}"] = float(obj[key])
                latest = obj.get("latest") if isinstance(obj.get("latest"), dict) else {}
                for key in ("gen_id", "valid_count", "robust_count", "best_score", "best_gain", "diversity"):
                    if isinstance(latest.get(key), (int, float)):
                        payload[f"{stem}/latest_{key}"] = float(latest[key])
            if payload:
                wandb_log(payload)
    except Exception:
        pass


def row_identity(row, default_prefix="row"):
    if not isinstance(row, dict):
        return f"{default_prefix}:unknown"
    for key in ("parent_id", "iter", "program_hash", "signature", "cluster_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{default_prefix}:unknown"


def _clean_rows_for_json(rows):
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out = {k: v for k, v in row.items() if k != "_val_ret"}
        cleaned.append(out)
    return cleaned


def _score_sort_value(row):
    try:
        score = float(row.get("score", -999.0))
        if np.isfinite(score):
            return score
    except Exception:
        pass
    return -999.0


def _sorted_artifact_rows(rows):
    cleaned = _clean_rows_for_json(rows)
    deduped = {}
    ordered = []
    for row in cleaned:
        key = (
            row.get("signature")
            or row.get("program_hash")
            or row.get("program_sig")
            or row.get("cluster_id")
            or row_identity(row)
        )
        prev = deduped.get(key)
        if prev is None:
            deduped[key] = row
            ordered.append(key)
            continue
        prev_tuple = (
            bool(prev.get("robust_ok", False)),
            _score_sort_value(prev),
            float(prev.get("robustness_score", 0.0) or 0.0),
        )
        row_tuple = (
            bool(row.get("robust_ok", False)),
            _score_sort_value(row),
            float(row.get("robustness_score", 0.0) or 0.0),
        )
        if row_tuple > prev_tuple:
            deduped[key] = row
    return sorted(
        [deduped[key] for key in ordered],
        key=lambda r: (
            not bool(r.get("robust_ok", False)),
            -_score_sort_value(r),
            row_identity(r),
        ),
    )


def _best_score_from_rows(rows):
    scores = [_score_sort_value(r) for r in rows if isinstance(r, dict) and r.get("score") is not None]
    return max(scores) if scores else None


def _benchmark_policy_payload():
    return {
        "source": EVOLVE_BENCHMARK_SOURCE,
        "anchor_mutation_type": EVOLVE_BENCHMARK_MUTATION,
        "selection_objective": SELECTION_OBJECTIVE,
        "deterministic_champions": EVOLVE_DETERMINISTIC_CHAMPIONS,
        "economic_sharpe_floor": ECONOMIC_SHARPE_FLOOR,
        "economic_edge_over_deterministic": ECONOMIC_EDGE_OVER_DETERMINISTIC,
    }


def _evolution_policy_payload():
    return {
        "max_generations": EVOLVE_MAX_GENERATIONS,
        "trials_per_generation": EVOLVE_TRIALS_PER_GENERATION,
        "beam_width": EVOLVE_BEAM_WIDTH,
        "survivors": EVOLVE_SURVIVORS,
        "min_generations": EVOLVE_MIN_GENERATIONS,
        "min_trials_per_generation": EVOLVE_MIN_TRIALS_PER_GENERATION,
        "robust_reset_threshold": EVOLVE_ROBUST_RESET_THRESHOLD,
        "zero_robust_patience": EVOLVE_ZERO_ROBUST_PATIENCE,
        "zero_robust_min_generations": EVOLVE_ZERO_ROBUST_MIN_GENERATIONS,
        "zero_robust_min_valid_rate": EVOLVE_ZERO_ROBUST_MIN_VALID_RATE,
        "stagnation_patience": EVOLVE_STAGNATION_PATIENCE,
        "partial_write_every": EVOLVE_PARTIAL_WRITE_EVERY,
        "random_exploration_frac": EVOLVE_RANDOM_EXPLORATION_FRAC,
        "convergence_patience": EVOLVE_CONVERGENCE_PATIENCE,
        "min_best_gain": EVOLVE_MIN_BEST_GAIN,
        "min_median_gain": EVOLVE_MIN_MEDIAN_GAIN,
        "min_novelty_gain": EVOLVE_MIN_NOVELTY_GAIN,
        "min_diversity_gain": EVOLVE_MIN_DIVERSITY_GAIN,
        "diversity_penalty": EVOLVE_DIVERSITY_PENALTY,
        "complexity_penalty": EVOLVE_COMPLEXITY_PENALTY,
        "train_val_gap_penalty": EVOLVE_TRAIN_VAL_GAP_PENALTY,
        "vol_scale_penalty": EVOLVE_VOL_SCALE_PENALTY,
        "robustness_score_floor": ROBUSTNESS_SCORE_FLOOR,
        "deterministic_champions": EVOLVE_DETERMINISTIC_CHAMPIONS,
        "economic_sharpe_floor": ECONOMIC_SHARPE_FLOOR,
        "economic_edge_over_deterministic": ECONOMIC_EDGE_OVER_DETERMINISTIC,
        "adaptive_diagnostic_run": EVOLVE_ADAPTIVE_DIAGNOSTIC_RUN,
        "family_quotas": dict(EVOLVE_FAMILY_QUOTAS),
        "parent_family_cap": EVOLVE_PARENT_FAMILY_CAP,
        "safe_composite_pairs": list(EVOLVE_SAFE_COMPOSITE_PAIRS),
        "max_total_evals": EVOLVE_MAX_TOTAL_EVALS,
        "max_wallclock_hours": EVOLVE_MAX_WALLCLOCK_HOURS,
        "train_val_firewall": True,
        "diagnosis_top_rows": EVOLVE_DIAGNOSIS_TOP_ROWS,
        "benchmark_policy": _benchmark_policy_payload(),
    }


def _robustness_failure_counts(rows):
    counts = {
        "train_sharpe": 0,
        "wf_median": 0,
        "wf_min": 0,
        "beta": 0,
        "turnover": 0,
        "signal_activity": 0,
        "raw_cs_std": 0,
        "long_short_balance": 0,
        "robustness_score": 0,
    }
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("train_sharpe", -99) <= 0:
            counts["train_sharpe"] += 1
        if row.get("wf_median", -99) <= 0:
            counts["wf_median"] += 1
        if row.get("wf_min", -99) <= -0.20:
            counts["wf_min"] += 1
        if abs(float(row.get("beta", 0.0) or 0.0)) > BETA_LIMIT:
            counts["beta"] += 1
        turnover = float(row.get("turnover", row.get("avg_turnover", 0.0)) or 0.0)
        if not (MIN_ACTIVE_TURNOVER <= turnover <= TURNOVER_LIMIT):
            counts["turnover"] += 1
        if float(row.get("signal_activity", 1.0) or 0.0) < MIN_SIGNAL_ACTIVITY:
            counts["signal_activity"] += 1
        if float(row.get("raw_cs_std", 1.0) or 0.0) < MIN_RAW_CS_STD:
            counts["raw_cs_std"] += 1
        if min(
            float(row.get("raw_long_frac", 1.0) or 0.0),
            float(row.get("raw_short_frac", 1.0) or 0.0),
        ) < MIN_LONG_SHORT_FRAC:
            counts["long_short_balance"] += 1
        if float(row.get("robustness_score", 0.0) or 0.0) < ROBUSTNESS_SCORE_FLOOR:
            counts["robustness_score"] += 1
    return counts


def _generation_diagnosis_events(generation, valid_rows, robust_rows, benchmark_row=None):
    generation = generation or {}
    valid_rows = [r for r in (valid_rows or []) if isinstance(r, dict)]
    robust_rows = [r for r in (robust_rows or []) if isinstance(r, dict)]

    def _snapshot_rows(rows):
        ranked = sorted(
            rows,
            key=lambda row: (-float(row.get("score", -999.0) or -999.0), str(row.get("parent_id") or row.get("signature") or "")),
        )
        snapped = []
        for row in ranked[: max(1, int(EVOLVE_DIAGNOSIS_TOP_ROWS))]:
            snapped.append(
                {
                    "parent_id": row.get("parent_id"),
                    "signature": row.get("signature"),
                    "cluster_id": row.get("cluster_id"),
                    "family": row.get("family"),
                    "mutation_type": row.get("mutation_type"),
                    "score": row.get("score"),
                    "robust_ok": bool(row.get("robust_ok")),
                    "train_sharpe": row.get("train_sharpe"),
                    "wf_median": row.get("wf_median"),
                    "wf_min": row.get("wf_min"),
                    "beta": row.get("beta"),
                    "turnover": row.get("turnover", row.get("avg_turnover")),
                    "signal_activity": row.get("signal_activity"),
                    "raw_cs_std": row.get("raw_cs_std"),
                    "raw_long_frac": row.get("raw_long_frac"),
                    "raw_short_frac": row.get("raw_short_frac"),
                    "robustness_score": row.get("robustness_score"),
                    "error": row.get("error"),
                }
            )
        return snapped

    benchmark_snapshot = None
    benchmark_score = None
    if isinstance(benchmark_row, dict):
        benchmark_snapshot = {
            "parent_id": benchmark_row.get("parent_id"),
            "signature": benchmark_row.get("signature"),
            "cluster_id": benchmark_row.get("cluster_id"),
            "family": benchmark_row.get("family"),
            "mutation_type": benchmark_row.get("mutation_type"),
            "score": benchmark_row.get("score"),
            "source": benchmark_row.get("source", EVOLVE_BENCHMARK_SOURCE),
        }
        try:
            benchmark_score = float(benchmark_row.get("score")) if benchmark_row.get("score") is not None else None
        except Exception:
            benchmark_score = None

    best_score = generation.get("best_score")
    best_so_far = generation.get("best_so_far")
    try:
        best_score_value = float(best_score) if best_score is not None else None
    except Exception:
        best_score_value = None
    try:
        best_so_far_value = float(best_so_far) if best_so_far is not None else None
    except Exception:
        best_so_far_value = None

    base_payload = {
        "gen_id": generation.get("gen_id"),
        "valid_count": int(generation.get("valid_count", len(valid_rows)) or 0),
        "valid_rate": float(generation.get("valid_rate", 0.0) or 0.0),
        "robust_count": int(generation.get("robust_count", len(robust_rows)) or 0),
        "best_score": best_score,
        "best_so_far": best_so_far,
        "benchmark_gap": None if best_score_value is None or benchmark_score is None else float(best_score_value - benchmark_score),
        "benchmark_policy": _benchmark_policy_payload(),
        "benchmark": benchmark_snapshot,
        "top_valid_rows": _snapshot_rows(valid_rows),
        "top_robust_rows": _snapshot_rows(robust_rows),
    }
    events = []

    best_so_far_streak = int(generation.get("best_so_far_streak", 0) or 0)
    if best_so_far_streak >= EVOLVE_STAGNATION_PATIENCE:
        events.append(
            {
                **base_payload,
                "kind": "stagnation",
                "trigger": "best_so_far_streak",
                "streak": best_so_far_streak,
                "new_cluster_count": int(generation.get("new_cluster_count", 0) or 0),
                "diversity": float(generation.get("diversity", 0.0) or 0.0),
            }
        )

    zero_robust_streak = int(generation.get("zero_robust_streak", 0) or 0)
    if generation.get("zero_robust_counted") and zero_robust_streak >= 1:
        failure_rows = [r for r in valid_rows if not r.get("robust_ok")]
        events.append(
            {
                **base_payload,
                "kind": "zero_robust",
                "trigger": "zero_robust_streak",
                "streak": zero_robust_streak,
                "failure_counts": _robustness_failure_counts(failure_rows),
            }
        )
    return events


def _run_flags_payload():
    return {
        "RUN_BASELINE_SWEEP": bool(globals().get("RUN_BASELINE_SWEEP", False)),
        "RUN_DETERMINISTIC_SEARCH": bool(globals().get("RUN_DETERMINISTIC_SEARCH", False)),
        "RUN_PARAM_SEARCH": bool(globals().get("RUN_PARAM_SEARCH", False)),
        "RUN_HELDOUT_EVAL": bool(globals().get("RUN_HELDOUT_EVAL", False)),
        "RUN_REPORTS": bool(globals().get("RUN_REPORTS", False)),
    }


def _evolution_summary_payload(all_rows, generations, total_evals, stop_reason, diagnosis_events=None, partial=False, started=None):
    latest = generations[-1] if generations else {}
    diagnosis_events = list(diagnosis_events or [])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evolution_enabled": True,
        "artifact_status": "partial" if partial else "final",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": _evolution_policy_payload(),
        "stop_reason": stop_reason,
        "generations_executed": len(generations),
        "total_evals": int(total_evals),
        "best_score": _best_score_from_rows(all_rows),
        "warnings": list(_WARNINGS),
        "latest": latest,
        "generations": generations,
        "diagnosis_count": len(diagnosis_events),
        "latest_diagnosis": diagnosis_events[-1] if diagnosis_events else None,
        "diagnosis_events": diagnosis_events,
    }
    if started is not None:
        payload["elapsed_hours"] = float((time.time() - started) / 3600.0)
    return payload


def _run_manifest_payload(all_rows, generations, total_evals, stop_reason, diagnosis_events=None, partial=False, started=None):
    diagnosis_events = list(diagnosis_events or [])
    result = {
        "stop_reason": stop_reason,
        "artifact_status": "partial" if partial else "final",
        "generations_executed": len(generations),
        "total_evals": int(total_evals),
        "best_score": _best_score_from_rows(all_rows),
        "diagnosis_count": len(diagnosis_events),
        "latest_diagnosis": diagnosis_events[-1] if diagnosis_events else None,
    }
    if started is not None:
        result["elapsed_hours"] = float((time.time() - started) / 3600.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "selection_objective": SELECTION_OBJECTIVE,
        "benchmark_policy": _benchmark_policy_payload(),
        "flags": _run_flags_payload(),
        "result": result,
    }


def _partial_report_payload(all_rows, generations, total_evals, stop_reason, diagnosis_events=None, partial=False, started=None):
    artifact_rows = _sorted_artifact_rows(all_rows)
    robust_rows = [r for r in artifact_rows if r.get("robust_ok")]
    latest = generations[-1] if generations else {}
    diagnosis_events = list(diagnosis_events or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "partial" if partial else "final",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stop_reason": stop_reason,
        "generations_executed": len(generations),
        "total_evals": int(total_evals),
        "latest": latest,
        "best_score": _best_score_from_rows(artifact_rows),
        "robust_count": len(robust_rows),
        "top_rows": artifact_rows[:25],
        "top_robust_rows": robust_rows[:25],
        "diagnosis_count": len(diagnosis_events),
        "latest_diagnosis": diagnosis_events[-1] if diagnosis_events else None,
        "diagnosis_events": diagnosis_events,
        "warnings": list(_WARNINGS),
        "elapsed_hours": None if started is None else float((time.time() - started) / 3600.0),
    }


def _write_evolution_artifacts(all_rows, generations, memory_notes, programs, diagnosis_events, total_evals, stop_reason, partial=False, started=None, suppress_errors=False):
    def _write_all():
        artifact_rows = _sorted_artifact_rows(all_rows)
        _atomic_write_json(PARAM_SEARCH_FILE, artifact_rows)
        _atomic_write_json(EVOLUTION_LINEAGE_FILE, artifact_rows)
        _atomic_write_json(
            EVOLUTION_SUMMARY_FILE,
            _evolution_summary_payload(
                artifact_rows,
                generations,
                total_evals,
                stop_reason,
                diagnosis_events=diagnosis_events,
                partial=partial,
                started=started,
            ),
        )
        _atomic_write_json(
            PARTIAL_REPORT_FILE,
            _partial_report_payload(
                artifact_rows,
                generations,
                total_evals,
                stop_reason,
                diagnosis_events=diagnosis_events,
                partial=partial,
                started=started,
            ),
        )
        _atomic_write_json(
            EVOLUTION_MEMORY_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_status": "partial" if partial else "final",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "memo_seed": _safe_load_memo_text(),
                "generation_reflections": memory_notes,
                "generation_programs": programs,
                "diagnosis_events": diagnosis_events,
            },
        )
        _atomic_write_json(
            EVOLUTION_PROGRAM_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_status": "partial" if partial else "final",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "programs": programs,
            },
        )
        _atomic_write_json(
            RUN_MANIFEST_FILE,
            _run_manifest_payload(
                artifact_rows,
                generations,
                total_evals,
                stop_reason,
                diagnosis_events=diagnosis_events,
                partial=partial,
                started=started,
            ),
        )
        return artifact_rows

    if suppress_errors:
        try:
            return _write_all()
        except Exception as e:
            _warn(f"partial evolution artifact write failed: {e}")
            return _clean_rows_for_json(all_rows)
    return _write_all()


def robustness_score(metric_row):
    """0-100 diagnostic score for real, stable market-neutral behavior."""
    row = metric_row or {}
    train_sh = float(row.get("train_sharpe", row.get("sharpe", 0.0)) or 0.0)
    wf_median = float(row.get("wf_median", 0.0) or 0.0)
    wf_min = float(row.get("wf_min", 0.0) or 0.0)
    beta = abs(float(row.get("beta", 0.0) or 0.0))
    turnover = float(row.get("turnover", row.get("avg_turnover", 0.0)) or 0.0)
    consistency = float(row.get("consistency", 0.0) or 0.0)
    activity = float(row.get("signal_activity", 1.0) or 0.0)
    cs_std = float(row.get("raw_cs_std", 1.0) or 0.0)
    long_frac = float(row.get("raw_long_frac", 1.0) or 0.0)
    short_frac = float(row.get("raw_short_frac", 1.0) or 0.0)

    score = 0.0
    score += 22.0 * min(max(train_sh, 0.0), 1.0)
    score += 20.0 * min(max(wf_median, 0.0), 0.8) / 0.8
    score += 14.0 * min(max(wf_min + 0.35, 0.0), 0.55) / 0.55
    score += 12.0 * max(0.0, 1.0 - beta / max(BETA_LIMIT, 1e-6))
    score += 10.0 if MIN_ACTIVE_TURNOVER <= turnover <= TURNOVER_LIMIT else 0.0
    score += 8.0 * min(max(consistency, 0.0), 1.0)
    score += 6.0 * min(max(activity, 0.0), 1.0)
    score += 4.0 * min(max(cs_std, 0.0), 0.12) / 0.12
    score += 2.0 * min(max(min(long_frac, short_frac), 0.0), 0.25) / 0.25
    return float(round(score, 4))


def selection_score(sharpe, wf_median=0.0, consistency=0.0, beta=0.0, avg_turnover=0.0, wf_min=0.0, max_dd=0.0):
    """Score market-neutral net Sharpe, not benchmark-relative long-only spread."""
    turnover_floor_penalty = 5.0 * max(0.0, MIN_ACTIVE_TURNOVER - float(avg_turnover))
    wf_downside_penalty = 0.30 * max(0.0, -float(wf_min))
    dd_penalty = 0.08 * max(0.0, abs(float(max_dd)) - 0.25)
    return float(
        sharpe
        + 0.35 * wf_median
        + 0.15 * consistency
        - 2.5 * abs(beta)
        - 0.35 * avg_turnover
        - turnover_floor_penalty
        - wf_downside_penalty
        - dd_penalty
    )


def evolution_score(val_sharpe, val_wf_median, val_wf_min, consistency, beta, avg_turnover):
    """Evolution ranking score: optimize held-out-like Sharpe, not headline score inflation."""
    downside_wf = max(0.0, -float(val_wf_min))
    wf_floor_shortfall = max(0.0, EVOLVE_PROGRAM_MIN_VAL_WF_MEDIAN - float(val_wf_median))
    wf_min_shortfall = max(0.0, EVOLVE_PROGRAM_MIN_VAL_WF_MIN - float(val_wf_min))
    turnover_floor_penalty = 5.0 * max(0.0, MIN_ACTIVE_TURNOVER - float(avg_turnover))
    return float(
        1.25 * float(val_sharpe)
        + 0.55 * float(val_wf_median)
        + 0.10 * float(consistency)
        - 0.45 * downside_wf
        - EVOLVE_VAL_WF_FLOOR_PENALTY * wf_floor_shortfall
        - EVOLVE_VAL_WF_FLOOR_PENALTY * wf_min_shortfall
        - 2.5 * abs(float(beta))
        - 0.35 * float(avg_turnover)
        - turnover_floor_penalty
    )


def shortlist_ok(beta, avg_turnover):
    return bool(abs(beta) <= BETA_LIMIT and MIN_ACTIVE_TURNOVER <= avg_turnover <= TURNOVER_LIMIT)


def robust_ok(metric_row, wf_floor=-0.20):
    metric_row = metric_row or {}
    return bool(
        metric_row.get("train_sharpe", -99) > 0
        and metric_row.get("wf_median", -99) > 0
        and metric_row.get("wf_min", -99) > wf_floor
        and shortlist_ok(metric_row.get("beta", 0.0), metric_row.get("turnover", 0.0))
        and metric_row.get("signal_activity", 1.0) >= MIN_SIGNAL_ACTIVITY
        and metric_row.get("raw_cs_std", 1.0) >= MIN_RAW_CS_STD
        and metric_row.get("raw_long_frac", 1.0) >= MIN_LONG_SHORT_FRAC
        and metric_row.get("raw_short_frac", 1.0) >= MIN_LONG_SHORT_FRAC
        and robustness_score(metric_row) >= ROBUSTNESS_SCORE_FLOOR
    )


def wf_summary(windows):
    vals = [w["sharpe"] for w in windows if "sharpe" in w]
    if not vals:
        return 0.0, 0.0
    return float(np.median(vals)), float(min(vals))


def _finish_signal(out):
    return out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)


def baseline_signal_factory(kind, n1, n2=None):
    if kind == "momentum":
        def signal(close, volume, vix=None, tnx=None):
            r = close.pct_change(n1).rank(axis=1, pct=True)
            return _finish_signal((r - 0.5) * 2)
        return signal

    if kind == "mean_reversion":
        def signal(close, volume, vix=None, tnx=None):
            z = -(close / close.rolling(n1).mean() - 1.0)
            r = z.rank(axis=1, pct=True)
            return _finish_signal((r - 0.5) * 2)
        return signal

    if kind == "ewm":
        def signal(close, volume, vix=None, tnx=None):
            fast = close.ewm(span=n1).mean()
            slow = close.ewm(span=n2).mean()
            x = (fast / slow - 1.0).rank(axis=1, pct=True)
            return _finish_signal((x - 0.5) * 2)
        return signal

    if kind == "volume_momentum":
        def signal(close, volume, vix=None, tnx=None):
            mom = close.pct_change(n1)
            vr = volume / volume.rolling(n2).mean()
            x = (mom * vr).rank(axis=1, pct=True)
            return _finish_signal((x - 0.5) * 2)
        return signal

    if kind == "ts_momentum":
        def signal(close, volume, vix=None, tnx=None):
            core = close.pct_change(n1)
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
            return _finish_signal(out)
        return signal

    if kind == "short_reversal":
        def signal(close, volume, vix=None, tnx=None):
            core = -close.pct_change(n1)
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
            return _finish_signal(out)
        return signal

    if kind == "vol_adjusted_momentum":
        def signal(close, volume, vix=None, tnx=None):
            ret = close.pct_change(n1)
            vol = close.pct_change().rolling(n2).std().replace(0, np.nan)
            core = ret.div(vol.clip(lower=1e-6), fill_value=0.0).clip(-3, 3)
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
            return _finish_signal(out)
        return signal

    if kind == "volume_confirmed_momentum":
        def signal(close, volume, vix=None, tnx=None):
            ret = close.pct_change(n1)
            vr = volume / volume.rolling(n2).mean()
            core = ret * vr.clip(lower=0.5, upper=1.8)
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
            return _finish_signal(out)
        return signal

    if kind == "multi_factor":
        def signal(close, volume, vix=None, tnx=None):
            mom = close.pct_change(n1).rank(axis=1, pct=True) - 0.5
            rev = (-close.pct_change(max(5, n1 // 8))).rank(axis=1, pct=True) - 0.5
            vol = close.pct_change().rolling(n2).std().replace(0, np.nan)
            low_vol = (-vol).rank(axis=1, pct=True) - 0.5
            core = 0.55 * mom + 0.25 * rev + 0.20 * low_vol
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
            return _finish_signal(out)
        return signal

    raise ValueError(kind)


def deterministic_signal_factory(short_span, long_span, variant="plain", aux=None):
    aux = aux or {}
    vol_win = int(aux.get("vol_window", 20))
    vol_cap = aux.get("vol_cap", 3.0)
    vol_gate = int(aux.get("vol_gate_window", 20))
    regime_thr = aux.get("vix_threshold", 24.0)

    def signal(close, volume, vix=None, tnx=None):
        fast = close.ewm(span=short_span, min_periods=short_span).mean().shift(1)
        slow = close.ewm(span=long_span, min_periods=long_span).mean().shift(1)
        core = fast / slow - 1.0
        if variant == "plain":
            x = core.rank(axis=1, pct=True)
            out = (x - 0.5) * 2
        elif variant == "rank_norm":
            out = (core.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "vol_scale":
            vol = close.pct_change().rolling(vol_win).std().replace(0, np.nan)
            scaled = core.div(vol.clip(lower=1e-6), fill_value=0.0).clip(-vol_cap, vol_cap)
            out = (scaled.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "volume_gate":
            vratio = volume / volume.rolling(vol_gate).mean()
            gated = core * vratio.clip(lower=0.5, upper=1.5)
            out = (gated.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "regime_gate":
            if vix is None:
                gated = core
            else:
                regime = (vix.rolling(5).mean() < regime_thr).astype(float).values[:, None]
                gated = core * regime
            out = (gated.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "ts_momentum":
            trend = close.pct_change(long_span)
            out = (trend.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "short_reversal":
            rev = -close.pct_change(short_span)
            out = (rev.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "vol_adjusted":
            trend = close.pct_change(long_span)
            vol = close.pct_change().rolling(vol_win).std().replace(0, np.nan)
            scaled = trend.div(vol.clip(lower=1e-6), fill_value=0.0).clip(-vol_cap, vol_cap)
            out = (scaled.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "volume_confirm":
            trend = close.pct_change(long_span)
            vratio = volume / volume.rolling(vol_gate).mean()
            confirmed = trend * vratio.clip(lower=0.5, upper=1.8)
            out = (confirmed.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "regime_momentum":
            trend_fast = close.pct_change(short_span)
            trend_slow = close.pct_change(long_span)
            trend = trend_fast - trend_slow
            if vix is not None:
                low_vix = (vix.rolling(5).mean() < regime_thr).astype(float).values[:, None]
                trend = trend * low_vix
            out = (trend.rank(axis=1, pct=True) - 0.5) * 2
        elif variant == "multi_factor":
            trend = close.pct_change(long_span).rank(axis=1, pct=True) - 0.5
            reversal = (-close.pct_change(short_span)).rank(axis=1, pct=True) - 0.5
            vol = close.pct_change().rolling(vol_win).std().replace(0, np.nan)
            low_vol = (-vol).rank(axis=1, pct=True) - 0.5
            score = 0.55 * trend + 0.25 * reversal + 0.20 * low_vol
            out = (score.rank(axis=1, pct=True) - 0.5) * 2
        else:
            raise ValueError(variant)
        return _finish_signal(out)

    return signal


def _normalized_mutation_key(mutation_type):
    key = str(mutation_type or "plain")
    aliases = {
        "volatility_scaling": "vol_scale",
        "vol_scaling": "vol_scale",
        "volume_confirmation": "volume_confirm",
        "regime_filter": "regime_gate",
        "regime_filter_momentum": "regime_momentum",
        "trend_momentum": "ts_momentum",
        "time_series_momentum": "ts_momentum",
        "factor_blend": "multi_factor",
        "multi_factor_blend": "multi_factor",
    }
    return aliases.get(key, key)


def deterministic_family_for_mutation(mutation_type):
    key = _normalized_mutation_key(mutation_type)
    mapping = {
        "plain": ("ewm", "ewm"),
        "span_tweak": ("ewm", "ewm"),
        "rank_norm": ("ewm", "ewm"),
        "rank_normalization": ("ewm", "ewm"),
        "vol_scale": ("ewm_volscale", "ewm"),
        "volume_gate": ("ewm_volume", "ewm"),
        "regime_gate": ("ewm_regime", "ewm"),
        "ts_momentum": ("momentum", "momentum"),
        "short_reversal": ("mean_reversion", "mean_reversion"),
        "vol_adjusted": ("volatility_momentum", "momentum"),
        "volume_confirm": ("volume_momentum", "momentum"),
        "regime_momentum": ("regime_momentum", "momentum"),
        "multi_factor": ("multi_factor", "multi_factor"),
    }
    return mapping.get(key, ("ewm", "ewm"))


def cluster_id_for_signal(base_family, mutation_type, short_span, long_span, params=None):
    params = normalize_aux_params(params)
    key = _normalized_mutation_key(mutation_type)
    family, mapped_base = deterministic_family_for_mutation(key)
    prefix = base_family if key == "span_tweak" and base_family else (family or base_family or mapped_base or "signal")
    if short_span is None or long_span is None:
        return f"{prefix}:generic"
    if key in ("regime_gate", "regime_momentum"):
        thr = params.get("vix_threshold")
        return f"{prefix}:{short_span}:{long_span}:{thr}" if thr is not None else f"{prefix}:{short_span}:{long_span}"
    if key in ("vol_scale", "vol_adjusted", "multi_factor"):
        win = params.get("vol_window")
        return f"{prefix}:{short_span}:{long_span}:{win}" if win is not None else f"{prefix}:{short_span}:{long_span}"
    if key in ("volume_gate", "volume_confirm"):
        win = params.get("vol_gate_window")
        return f"{prefix}:{short_span}:{long_span}:{win}" if win is not None else f"{prefix}:{short_span}:{long_span}"
    return f"{prefix}:{short_span}:{long_span}"


def baseline_walk_forward(fn, close_df, volume_df, vix_s=None, tnx_s=None, flipped=False, n_windows=WF_WINDOWS):
    n = len(close_df)
    sz = max(n // n_windows, 1)
    out = []
    for w in range(n_windows):
        lo = w * sz
        hi = (w + 1) * sz if w < n_windows - 1 else n
        sub_c = close_df.iloc[lo:hi]
        sub_v = volume_df.iloc[lo:hi]
        sub_vix = vix_s.iloc[lo:hi] if vix_s is not None else None
        sub_tnx = tnx_s.iloc[lo:hi] if tnx_s is not None else None
        if len(sub_c) < 40:
            continue
        try:
            sig = fn(sub_c, sub_v, sub_vix, sub_tnx)
            if flipped:
                sig = -sig
            m = backtest(sig, sub_c)
            out.append({"window": w, "sharpe": m["sharpe"], "beta": m["beta"], "ann_return": m["ann_return"], "max_dd": m["max_dd"]})
        except Exception as e:
            out.append({"window": w, "error": str(e)})
    return out


def run_baseline_sweep():
    grid = [
        ("ewm", 30, 100), ("ewm", 32, 110), ("ewm", 34, 115), ("ewm", 36, 120),
        ("ewm", 38, 122), ("ewm", 40, 120), ("ewm", 42, 125), ("ewm", 44, 128),
        ("ewm", 46, 130), ("ewm", 48, 135), ("ewm", 50, 140), ("ewm", 55, 150),
        ("ewm", 40, 160), ("ewm", 20, 80), ("ewm", 20, 100), ("ewm", 30, 120),
        ("volume_momentum", 20, 20), ("volume_momentum", 40, 20), ("volume_momentum", 60, 20),
        ("momentum", 40, None), ("momentum", 80, None), ("mean_reversion", 10, None),
        ("ts_momentum", 63, None), ("ts_momentum", 126, None), ("short_reversal", 5, None),
        ("short_reversal", 10, None), ("vol_adjusted_momentum", 63, 20),
        ("vol_adjusted_momentum", 126, 30), ("volume_confirmed_momentum", 63, 20),
        ("multi_factor", 126, 20),
    ]
    rows = []
    for kind, n1, n2 in grid:
        fn = baseline_signal_factory(kind, n1, n2)
        sig = fn(close_train, volume_train, vix_train, tnx_train)
        m = backtest(sig, close_train)
        flipped = bool(m["prefer_inverted"])
        beta = m["beta_inverted"] if flipped else m["beta"]
        train_sharpe = m["sharpe_inverted"] if flipped else m["sharpe"]
        ann_return = m["ann_return_inverted"] if flipped else m["ann_return"]
        dd = m["max_dd_inverted"] if flipped else m["max_dd"]
        consistency = m["consistency_inverted"] if flipped else m["consistency"]
        wf = baseline_walk_forward(fn, close_train, volume_train, vix_train, tnx_train, flipped=flipped)
        wf_median, wf_min = wf_summary(wf)
        turnover = m["avg_turnover"]
        score = selection_score(train_sharpe, wf_median, consistency, beta, turnover, wf_min=wf_min, max_dd=dd)
        family = "ewm" if kind == "ewm" else kind
        if kind == "ewm":
            meta = signature_for_signal(
                family,
                "span_tweak",
                f"ewm(span={n1}) ewm(span={n2})",
                short_span=n1,
                long_span=n2,
                base_family="ewm",
                params={},
            )
        else:
            meta = signature_for_signal(
                family,
                "span_tweak",
                f"{kind}:{n1}:{n2}",
                base_family=family,
                params={"n1": n1, "n2": n2},
            )
        priority_weight = 3.0 if kind == "ewm" and n1 >= 34 and n1 <= 48 and (n2 or 0) >= 115 and (n2 or 0) <= 135 else (2.0 if kind == "ewm" else 1.0)
        metric_payload = {
            "train_sharpe": train_sharpe,
            "wf_median": wf_median,
            "wf_min": wf_min,
            "beta": beta,
            "turnover": turnover,
            "consistency": consistency,
            "raw_cs_std": m.get("raw_cs_std", 1.0),
            "raw_long_frac": m.get("raw_long_frac", 1.0),
            "raw_short_frac": m.get("raw_short_frac", 1.0),
            "signal_activity": m.get("signal_activity", 1.0),
        }
        rows.append({
            "parent_id": f"base:{kind}:{n1}:{n2 if n2 is not None else '-'}",
            "family": kind,
            "n1": n1,
            "n2": n2,
            "train_sharpe": train_sharpe,
            "raw_sharpe": m["sharpe"],
            "inv_sharpe": m["sharpe_inverted"],
            "flipped": flipped,
            "beta": beta,
            "turnover": turnover,
            "consistency": consistency,
            "ann_return": ann_return,
            "dd": dd,
            "wf_median": wf_median,
            "wf_min": wf_min,
            "score": score,
            "robustness_score": robustness_score(metric_payload),
            "raw_cs_std": m.get("raw_cs_std", 1.0),
            "raw_long_frac": m.get("raw_long_frac", 1.0),
            "raw_short_frac": m.get("raw_short_frac", 1.0),
            "signal_activity": m.get("signal_activity", 1.0),
            "shortlist_ok": shortlist_ok(beta, turnover),
            "priority_weight": priority_weight,
            "robust_ok": robust_ok(metric_payload),
            "component_count": 1,
            "model_size": 1,
            "model_size_key": "components=1",
            **meta,
        })
    rows = sorted(rows, key=lambda r: -(r["score"] + 0.03 * r["priority_weight"]))
    _atomic_write_json(BASELINE_FILE, rows)
    return rows


def deterministic_variant_grid():
    variants = [("plain", {}), ("rank_norm", {})]
    variants += [("vol_scale", {"vol_window": w}) for w in (20, 30)]
    variants += [("volume_gate", {"vol_gate_window": w}) for w in (20, 40)]
    variants += [("regime_gate", {"vix_threshold": thr}) for thr in (20.0, 24.0, 28.0)]
    variants += [("ts_momentum", {}), ("short_reversal", {})]
    variants += [("vol_adjusted", {"vol_window": w}) for w in (20, 30, 40)]
    variants += [("volume_confirm", {"vol_gate_window": w}) for w in (20, 40, 60)]
    variants += [("regime_momentum", {"vix_threshold": thr}) for thr in (18.0, 22.0, 26.0)]
    variants += [("multi_factor", {"vol_window": 20}), ("multi_factor", {"vol_window": 40})]
    return variants


def parameter_search_variant_space():
    variants = [("plain", {}), ("rank_norm", {})]
    variants += [("vol_scale", {"vol_window": w}) for w in PARAM_SEARCH_VOL_WINDOWS]
    variants += [("volume_gate", {"vol_gate_window": w}) for w in PARAM_SEARCH_VOL_GATE_WINDOWS]
    variants += [("regime_gate", {"vix_threshold": thr}) for thr in PARAM_SEARCH_REGIME_THRESHOLDS]
    variants += [("ts_momentum", {}), ("short_reversal", {})]
    variants += [("vol_adjusted", {"vol_window": w}) for w in PARAM_SEARCH_VOL_WINDOWS]
    variants += [("volume_confirm", {"vol_gate_window": w}) for w in PARAM_SEARCH_VOL_GATE_WINDOWS]
    variants += [("regime_momentum", {"vix_threshold": thr}) for thr in PARAM_SEARCH_REGIME_THRESHOLDS]
    variants += [("multi_factor", {"vol_window": w}) for w in PARAM_SEARCH_VOL_WINDOWS]
    return variants


def sample_parameter_trials(n_trials=PARAM_SEARCH_TRIALS, seed=PARAM_SEARCH_SEED):
    rng = random.Random(seed)
    variants = parameter_search_variant_space()
    trials = []
    seen = set()
    max_attempts = max(n_trials * 20, 2000)
    attempts = 0
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        short_span = rng.choice(PARAM_SEARCH_SHORT_SPANS)
        long_span = rng.choice(PARAM_SEARCH_LONG_SPANS)
        if long_span < short_span + 30:
            continue
        variant, aux = rng.choice(variants)
        aux_norm = normalize_aux_params(aux)
        trial_key = (variant, short_span, long_span, tuple(sorted(aux_norm.items())))
        if trial_key in seen:
            continue
        seen.add(trial_key)
        trials.append(
            {
                "mutation_type": variant,
                "short_span": short_span,
                "long_span": long_span,
                "params": aux_norm,
            }
        )
    return trials


def evaluate_structured_trial(trial, source="parameter_search"):
    short_span = trial["short_span"]
    long_span = trial["long_span"]
    variant = trial["mutation_type"]
    aux_norm = normalize_aux_params(trial.get("params", {}))
    fn = deterministic_signal_factory(short_span, long_span, variant=variant, aux=aux_norm)
    m = None
    sig = fn(close_train, volume_train, vix_train, tnx_train)
    m = backtest(sig, close_train)
    flipped = bool(m["prefer_inverted"])
    beta = m["beta_inverted"] if flipped else m["beta"]
    train_sharpe = m["sharpe_inverted"] if flipped else m["sharpe"]
    ann_return = m["ann_return_inverted"] if flipped else m["ann_return"]
    dd = m["max_dd_inverted"] if flipped else m["max_dd"]
    consistency = m["consistency_inverted"] if flipped else m["consistency"]
    wf = baseline_walk_forward(fn, close_train, volume_train, vix_train, tnx_train, flipped=flipped)
    wf_median, wf_min = wf_summary(wf)
    turnover = m["avg_turnover"]
    score = selection_score(train_sharpe, wf_median, consistency, beta, turnover, wf_min=wf_min, max_dd=dd)
    family, base_family = deterministic_family_for_mutation(variant)
    code_label = f"ewm(span={short_span}) ewm(span={long_span}) {variant} {aux_norm}"
    meta = signature_for_signal(
        family,
        variant,
        code_label,
        short_span=short_span,
        long_span=long_span,
        base_family=base_family,
        params=aux_norm,
    )
    aux_tag = ",".join(f"{k}={v}" for k, v in aux_norm.items()) if aux_norm else "base"
    metric_payload = {
        "train_sharpe": train_sharpe,
        "wf_median": wf_median,
        "wf_min": wf_min,
        "beta": beta,
        "turnover": turnover,
        "consistency": consistency,
        "raw_cs_std": m.get("raw_cs_std", 1.0),
        "raw_long_frac": m.get("raw_long_frac", 1.0),
        "raw_short_frac": m.get("raw_short_frac", 1.0),
        "signal_activity": m.get("signal_activity", 1.0),
    }
    return {
        "source": source,
        "parent_id": f"search:{variant}:{short_span}:{long_span}:{aux_tag}",
        "family": family,
        "base_family": base_family,
        "mutation_type": variant,
        "short_span": short_span,
        "long_span": long_span,
        "params": aux_norm,
        "train_sharpe": train_sharpe,
        "ann_return": ann_return,
        "beta": beta,
        "turnover": turnover,
        "max_dd": dd,
        "consistency": consistency,
        "wf_median": wf_median,
        "wf_min": wf_min,
        "score": score,
        "robustness_score": robustness_score(metric_payload),
        "raw_cs_std": m.get("raw_cs_std", 1.0),
        "raw_long_frac": m.get("raw_long_frac", 1.0),
        "raw_short_frac": m.get("raw_short_frac", 1.0),
        "signal_activity": m.get("signal_activity", 1.0),
        "flipped": flipped,
        "shortlist_ok": shortlist_ok(beta, turnover),
        "robust_ok": robust_ok(metric_payload),
        "component_count": 1,
        "model_size": 1,
        "model_size_key": "components=1",
        **meta,
    }


def _trial_key(trial):
    aux = normalize_aux_params(trial.get("params", {}))
    return (
        trial.get("mutation_type"),
        int(trial.get("short_span")),
        int(trial.get("long_span")),
        tuple(sorted(aux.items())),
    )


def _row_to_trial(row):
    return {
        "mutation_type": row["mutation_type"],
        "short_span": int(row["short_span"]),
        "long_span": int(row["long_span"]),
        "params": normalize_aux_params(row.get("params", {})),
    }


def _safe_load_rows_from_file(path_obj):
    try:
        if path_obj is not None and path_obj.exists():
            payload = json.loads(path_obj.read_text())
            if isinstance(payload, list):
                return payload
    except Exception:
        return []
    return []


def _safe_load_search_results():
    if "load_search_results" in globals():
        try:
            rows = load_search_results()
            if isinstance(rows, list):
                return rows
        except Exception:
            _warn("load_search_results() failed; using file-based fallback")
    rows = []
    if "load_parameter_search" in globals():
        try:
            rows.extend([r for r in load_parameter_search() if isinstance(r, dict)])
        except Exception:
            pass
    elif "PARAM_SEARCH_FILE" in globals():
        rows.extend(_safe_load_rows_from_file(PARAM_SEARCH_FILE))
    if "load_deterministic" in globals():
        try:
            rows.extend([r for r in load_deterministic() if isinstance(r, dict)])
        except Exception:
            pass
    elif "DETERMINISTIC_FILE" in globals():
        rows.extend(_safe_load_rows_from_file(DETERMINISTIC_FILE))
    return rows


def _safe_load_log_rows():
    if "load_log" in globals():
        try:
            rows = load_log()
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            pass
    if "RESEARCH_LOG" in globals() and RESEARCH_LOG.exists():
        out = []
        for line in RESEARCH_LOG.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                pass
        return out
    return []


def _safe_load_memo_text():
    if "load_memo" in globals():
        try:
            txt = load_memo()
            if isinstance(txt, str):
                return txt
        except Exception:
            pass
    if "MEMO_FILE" in globals() and MEMO_FILE.exists():
        try:
            return MEMO_FILE.read_text()
        except Exception:
            pass
    return "(No memo yet - deterministic-first mode.)"


def _entry_variant_tags(entry):
    tags = []
    allowed = {v for v, _ in parameter_search_variant_space()}

    def add(tag):
        if tag in allowed and tag not in tags:
            tags.append(tag)

    if not isinstance(entry, dict):
        return tags

    add(entry.get("mutation_type"))

    program = entry.get("program")
    if isinstance(program, dict):
        for comp in program.get("components", []):
            if isinstance(comp, dict):
                add(comp.get("mutation_type"))

    for comp in entry.get("program_components", []):
        if isinstance(comp, dict):
            add(comp.get("mutation_type"))

    text = " ".join(
        str(entry.get(key, ""))
        for key in ("cluster_id", "signature", "parent_id", "code", "hypothesis")
    ).lower()
    for variant in allowed:
        if variant in text:
            add(variant)
    return tags


def _variant_failure_bias():
    logs = _safe_load_log_rows()
    if not logs:
        return {}
    last = logs[-300:]
    counts = {v: 0.0 for v, _ in parameter_search_variant_space()}
    for e in last:
        err = str(e.get("error", "")).lower()
        if not err:
            continue
        weight = 1.0
        if "dead after market-neutral normalization" in err:
            weight = 3.0
        elif "not genuinely long-short" in err:
            weight = 3.0
        elif "beta" in err or "wf_min" in err or "robustness" in err:
            weight = 1.5

        tags = _entry_variant_tags(e)
        if not tags:
            code = str(e.get("code", "")).lower()
            hyp = str(e.get("hypothesis", "")).lower()
            text = code + " " + hyp
            if "vix" in text or "regime" in text:
                tags.extend(["regime_gate", "regime_momentum"])
            if "volume" in text:
                tags.extend(["volume_gate", "volume_confirm"])
            if "std" in text or "vol" in text:
                tags.extend(["vol_scale", "vol_adjusted"])
            if "rank" in text:
                tags.append("rank_norm")
            if "reversal" in text or "contrarian" in text:
                tags.append("short_reversal")
            if "multi" in text or "blend" in text or "composite" in text:
                tags.append("multi_factor")
            if "momentum" in text or "trend" in text:
                tags.append("ts_momentum")
            if not any(k in text for k in ("vix", "regime", "volume", "std", "vol", "rank")):
                tags.append("plain")

        for tag in set(tags):
            counts[tag] = counts.get(tag, 0.0) + weight
    total = sum(counts.values()) or 1
    return {k: counts[k] / total for k in counts}


def _summarize_generation_reflection(gen_id, gen_rows, survivors):
    valid = [r for r in gen_rows if r.get("score") is not None]
    robust = [r for r in valid if r.get("robust_ok")]
    fails = [r for r in gen_rows if r.get("error")]
    top = sorted(robust if robust else valid, key=lambda r: -r.get("score", -999))[:5]
    lines = [f"Generation {gen_id} reflection"]
    lines.append(f"- valid={len(valid)} robust={len(robust)} fails={len(fails)}")
    if top:
        lines.append("- winners:")
        for r in top:
            lines.append(
                f"  - {r.get('cluster_id')} score={r.get('score', 0.0):+.2f} "
                f"Sh={r.get('train_sharpe', 0.0):+.2f} wf={r.get('wf_median', 0.0):+.2f}/{r.get('wf_min', 0.0):+.2f}"
            )
    if fails:
        recent_fail = fails[-5:]
        lines.append("- recent failures:")
        for f in recent_fail:
            lines.append(f"  - {str(f.get('error'))[:140]}")
    lines.append("- survivor signatures:")
    for s in survivors:
        lines.append(f"  - {s.get('signature')}")
    return "\n".join(lines)


def _bootstrap_ci(values, n_boot=300, alpha=0.10, seed=42):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
        means.append(float(np.mean(sample)))
    means.sort()
    lo_idx = int((alpha / 2) * len(means))
    hi_idx = int((1 - alpha / 2) * len(means)) - 1
    lo_idx = max(0, min(lo_idx, len(means) - 1))
    hi_idx = max(0, min(hi_idx, len(means) - 1))
    return (means[lo_idx], means[hi_idx])


def _build_generation_program(gen_id, parents, recent_rows):
    fail_bias = _variant_failure_bias()
    memo = _safe_load_memo_text().lower()
    valid = [r for r in recent_rows if r.get("score") is not None]
    robust = [r for r in valid if r.get("robust_ok")]
    ranked = sorted(robust if robust else valid, key=lambda r: -_heldout_proxy_score(r))
    top = ranked[:8]
    top_clusters = [r.get("cluster_id") for r in top if r.get("cluster_id")]
    dominant_cluster_frac = 0.0
    if top_clusters:
        dominant_cluster_frac = max(top_clusters.count(cid) for cid in set(top_clusters)) / max(len(top_clusters), 1)

    if top:
        short_center = int(round(float(np.median([r.get("short_span", 54) for r in top]))))
        long_center = int(round(float(np.median([r.get("long_span", 90) for r in top]))))
    elif parents:
        short_center = int(round(float(np.median([p.get("short_span", 54) for p in parents]))))
        long_center = int(round(float(np.median([p.get("long_span", 90) for p in parents]))))
    else:
        short_center, long_center = 54, 90

    short_deltas = [-9, -6, -3, 0, 3, 6, 9]
    long_deltas = [-18, -12, -6, 0, 6, 12, 18]
    if "faster" in memo or "short horizon" in memo:
        short_deltas = [-6, -3, 0, 3, 6]
    if "slower" in memo or "long horizon" in memo:
        long_deltas = [-24, -18, -12, -6, 0, 6, 12]

    variant_space = [v for v, _ in parameter_search_variant_space()]
    if fail_bias:
        variant_priority = sorted(variant_space, key=lambda v: fail_bias.get(v, 0.0))
    else:
        variant_priority = list(variant_space)

    avoid_variants = [v for v, b in fail_bias.items() if b > 0.22]
    focus_variants = [v for v in variant_priority if v not in avoid_variants][:3]
    fallback_focus = ["regime_momentum", "volume_confirm", "volume_gate", "rank_norm", "plain", "regime_gate", "ts_momentum"]
    if dominant_cluster_frac >= 0.50:
        fallback_focus = ["volume_confirm", "volume_gate", "rank_norm", "plain", "regime_momentum", "regime_gate", "ts_momentum"]
    for variant in fallback_focus:
        if variant in variant_space and variant not in focus_variants and variant not in avoid_variants:
            focus_variants.append(variant)
        if len(focus_variants) >= EVOLVE_MIN_FOCUS_VARIANTS:
            break
    if not focus_variants:
        focus_variants = ["regime_momentum", "volume_confirm", "volume_gate", "rank_norm", "regime_gate"]

    program = {
        "gen_id": gen_id,
        "short_center": short_center,
        "long_center": long_center,
        "short_deltas": short_deltas,
        "long_deltas": long_deltas,
        "focus_variants": focus_variants,
        "avoid_variants": avoid_variants,
        "fail_bias": fail_bias,
        "dominant_cluster_frac": float(dominant_cluster_frac),
        "notes": "reflection-driven mutation program",
    }
    return program


def _program_variant_mix(base_variant, program):
    focus = list(program.get("focus_variants", []))
    avoid = set(program.get("avoid_variants", []))
    all_variants = [v for v, _ in parameter_search_variant_space()]
    ordered = [base_variant] + focus + [v for v in all_variants if v != base_variant and v not in focus]
    out = []
    for v in ordered:
        if v in avoid and v != base_variant:
            continue
        if v not in out:
            out.append(v)
    return out[:4]


def _heldout_proxy_score(row):
    base = float(row.get("score", -999.0))
    wf_median = float(row.get("wf_median", 0.0))
    wf_min = float(row.get("wf_min", 0.0))
    consistency = float(row.get("consistency", 0.0))
    spread = max(0.0, wf_median - wf_min)
    neg_wfmin = max(0.0, -wf_min)
    low_consistency = max(0.0, 0.5 - consistency)
    proxy = (
        base
        - EVOLVE_PROXY_WF_SPREAD_PENALTY * spread
        - EVOLVE_PROXY_WFMIN_PENALTY * neg_wfmin
        - EVOLVE_PROXY_CONSISTENCY_PENALTY * low_consistency
    )
    return float(proxy)


def _top_parents_from_rows(rows, k):
    valid = [r for r in rows if r.get("score") is not None]
    if not valid:
        return []
    ranked = sorted(valid, key=lambda r: -_heldout_proxy_score(r))
    out = []
    seen_sig = set()
    for r in ranked:
        sig = r.get("signature")
        if sig and sig in seen_sig:
            continue
        if sig:
            seen_sig.add(sig)
        out.append(r)
        if len(out) >= k:
            break
    return out


def _diverse_top_parents_from_rows(rows, k):
    valid = [r for r in rows if r.get("score") is not None]
    if not valid:
        return []
    ranked = sorted(valid, key=lambda r: -_heldout_proxy_score(r))
    out = []
    seen_sig = set()

    def add(row):
        sig = row.get("signature") or row_identity(row, "parent")
        if sig in seen_sig:
            return False
        seen_sig.add(sig)
        out.append(row)
        return True

    for variant in EVOLVE_CHAMPION_VARIANTS:
        family_rows = [r for r in ranked if r.get("mutation_type") == variant]
        if family_rows:
            add(family_rows[0])
        if len(out) >= k:
            return out[:k]
    for row in ranked:
        if len(out) >= k:
            break
        add(row)
    return out[:k]


def _program_variants(program):
    comps = program.get("components", []) if isinstance(program, dict) else []
    return [str(_sanitize_component(c).get("mutation_type", EVOLVE_BENCHMARK_MUTATION)) for c in comps]


def _program_primary_family(program):
    variants = _program_variants(program)
    if len(variants) >= 2:
        return "composite"
    return variants[0] if variants else EVOLVE_BENCHMARK_MUTATION


def _row_family(row):
    if not isinstance(row, dict):
        return "unknown"
    if isinstance(row.get("program"), dict):
        return _program_primary_family(row["program"])
    return str(row.get("candidate_family") or row.get("mutation_type") or row.get("family") or "unknown")


def _family_capped_rows(rows, limit, cap_frac=EVOLVE_PARENT_FAMILY_CAP):
    rows = [r for r in rows or [] if isinstance(r, dict)]
    if not rows or limit <= 0:
        return []
    cap = max(1, int(np.ceil(float(limit) * float(cap_frac))))
    families = { _row_family(r) for r in rows }
    if len(families) <= 1:
        return rows[:limit]
    selected = []
    family_counts = {}
    deferred = []
    for row in rows:
        fam = _row_family(row)
        if family_counts.get(fam, 0) >= cap:
            deferred.append(row)
            continue
        selected.append(row)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if len(selected) >= limit:
            return selected
    for row in deferred:
        if len(selected) >= limit:
            break
        selected.append(row)
    return selected[:limit]


def _aux_candidates_for_variant(variant, base_params):
    base = normalize_aux_params(base_params)
    if variant == "regime_gate":
        vals = sorted(set(PARAM_SEARCH_REGIME_THRESHOLDS + [float(base.get("vix_threshold", 20.0))]))
        return [{"vix_threshold": float(v)} for v in vals]
    if variant == "vol_scale":
        vals = sorted(set(PARAM_SEARCH_VOL_WINDOWS + [int(base.get("vol_window", 20))]))
        return [{"vol_window": int(v)} for v in vals]
    if variant == "vol_adjusted":
        vals = sorted(set(PARAM_SEARCH_VOL_WINDOWS + [int(base.get("vol_window", 20))]))
        return [{"vol_window": int(v)} for v in vals]
    if variant == "multi_factor":
        vals = sorted(set(PARAM_SEARCH_VOL_WINDOWS + [int(base.get("vol_window", 20))]))
        return [{"vol_window": int(v)} for v in vals]
    if variant == "volume_gate":
        vals = sorted(set(PARAM_SEARCH_VOL_GATE_WINDOWS + [int(base.get("vol_gate_window", 20))]))
        return [{"vol_gate_window": int(v)} for v in vals]
    if variant == "volume_confirm":
        vals = sorted(set(PARAM_SEARCH_VOL_GATE_WINDOWS + [int(base.get("vol_gate_window", 20))]))
        return [{"vol_gate_window": int(v)} for v in vals]
    if variant == "regime_momentum":
        vals = sorted(set(PARAM_SEARCH_REGIME_THRESHOLDS + [float(base.get("vix_threshold", 20.0))]))
        return [{"vix_threshold": float(v)} for v in vals]
    return [{}]


def _allocate_parent_quotas(parents, total_trials):
    n = len(parents)
    if n == 0:
        return []
    if n == 1:
        return [max(total_trials, 0)]
    min_q = EVOLVE_PARENT_MIN_TRIALS
    max_q = EVOLVE_PARENT_MAX_TRIALS
    if total_trials <= n * min_q:
        base = total_trials // n
        rem = total_trials - base * n
        return [base + (1 if i < rem else 0) for i in range(n)]
    quotas = [min_q] * n
    extra = total_trials - n * min_q
    ranks = list(range(n, 0, -1))
    sw = sum(ranks)
    fractional = []
    for i, w in enumerate(ranks):
        add_f = extra * (w / sw)
        add_i = int(add_f)
        quotas[i] += add_i
        fractional.append((add_f - add_i, i))
    rem = total_trials - sum(quotas)
    for _, i in sorted(fractional, reverse=True):
        if rem <= 0:
            break
        quotas[i] += 1
        rem -= 1
    overflow = 0
    for i in range(n):
        if quotas[i] > max_q:
            overflow += quotas[i] - max_q
            quotas[i] = max_q
    if overflow > 0:
        for i in range(n):
            room = max_q - quotas[i]
            if room <= 0:
                continue
            take = min(room, overflow)
            quotas[i] += take
            overflow -= take
            if overflow <= 0:
                break
    return quotas


def _build_branch_trials(parent, quota, gen_id, branch_id, rng, program):
    base_trial = _row_to_trial(parent)
    base_short = int(base_trial["short_span"])
    base_long = int(base_trial["long_span"])
    variant = base_trial["mutation_type"]
    variant_mix = _program_variant_mix(variant, program)
    short_deltas = list(program.get("short_deltas", [-9, -6, -3, 0, 3, 6, 9]))
    long_deltas = list(program.get("long_deltas", [-18, -12, -6, 0, 6, 12, 18]))
    short_center = int(program.get("short_center", base_short))
    long_center = int(program.get("long_center", base_long))
    trials = []
    seen = set()
    for d_short in short_deltas:
        for d_long in long_deltas:
            blended_short = int(round((base_short + short_center) / 2.0))
            blended_long = int(round((base_long + long_center) / 2.0))
            short_span = min(max(blended_short + d_short, min(PARAM_SEARCH_SHORT_SPANS)), max(PARAM_SEARCH_SHORT_SPANS))
            long_span = min(max(blended_long + d_long, min(PARAM_SEARCH_LONG_SPANS)), max(PARAM_SEARCH_LONG_SPANS))
            if long_span < short_span + 30:
                continue
            for v in variant_mix[:3]:
                aux_variants = _aux_candidates_for_variant(v, base_trial["params"])
                rng.shuffle(aux_variants)
                for aux in aux_variants[:2]:
                    trial = {
                        "mutation_type": v,
                        "short_span": short_span,
                        "long_span": long_span,
                        "params": normalize_aux_params(aux),
                        "origin": "branch_local",
                        "gen_id": gen_id,
                        "branch_id": branch_id,
                        "parent_id": row_identity(parent, "parent"),
                        "parent_signature": parent.get("signature"),
                        "parent_score": parent.get("score"),
                    }
                    k = _trial_key(trial)
                    if k in seen:
                        continue
                    seen.add(k)
                    trials.append(trial)
                    if len(trials) >= quota:
                        return trials
    return trials[:quota]


def _seed_parents_for_generation():
    search_rows = [r for r in _safe_load_search_results() if r.get("score") is not None]
    robust = [r for r in search_rows if r.get("robust_ok")]
    return _diverse_top_parents_from_rows(robust if robust else search_rows, EVOLVE_BEAM_WIDTH)


def _run_generation_trials(gen_id, parents, remaining_cap, seen_trial_keys, program):
    rng = random.Random(PARAM_SEARCH_SEED + 191 * gen_id)
    target_trials = min(EVOLVE_TRIALS_PER_GENERATION, remaining_cap)
    random_budget = int(target_trials * EVOLVE_RANDOM_EXPLORATION_FRAC)
    branch_budget = max(target_trials - random_budget, 0)
    parents_ranked = sorted(parents, key=lambda r: -r.get("score", -999))
    quotas = _allocate_parent_quotas(parents_ranked, branch_budget)
    planned = []
    for i, (parent, quota) in enumerate(zip(parents_ranked, quotas)):
        branch_trials = _build_branch_trials(parent, quota, gen_id, i, rng, program)
        for t in branch_trials:
            k = _trial_key(t)
            if k in seen_trial_keys:
                continue
            seen_trial_keys.add(k)
            t["allocation_quota"] = quota
            planned.append(t)
    if len(planned) < target_trials:
        filler_n = target_trials - len(planned)
        fillers = sample_parameter_trials(n_trials=filler_n * 3, seed=PARAM_SEARCH_SEED + 9973 * gen_id)
        for t in fillers:
            ft = {
                **t,
                "origin": "random_fill",
                "gen_id": gen_id,
                "branch_id": -1,
                "parent_id": f"random_fill:g{gen_id}",
                "parent_signature": None,
                "parent_score": None,
                "allocation_quota": 0,
            }
            k = _trial_key(ft)
            if k in seen_trial_keys:
                continue
            seen_trial_keys.add(k)
            planned.append(ft)
            if len(planned) >= target_trials:
                break

    rows = []
    for trial in planned:
        try:
            row = evaluate_structured_trial(trial, source="parameter_search_evolution")
            row["gen_id"] = gen_id
            row["branch_id"] = trial.get("branch_id")
            row["origin"] = trial.get("origin", "branch_local")
            row["parent_id"] = trial.get("parent_id")
            row["parent_signature"] = trial.get("parent_signature")
            row["allocation_quota"] = trial.get("allocation_quota", 0)
            parent_score = trial.get("parent_score")
            row["score_gain_vs_parent"] = float(row["score"] - parent_score) if parent_score is not None else None
        except Exception as e:
            row = {
                "source": "parameter_search_evolution",
                "gen_id": gen_id,
                "branch_id": trial.get("branch_id"),
                "origin": trial.get("origin", "branch_local"),
                "parent_id": trial.get("parent_id"),
                "parent_signature": trial.get("parent_signature"),
                "allocation_quota": trial.get("allocation_quota", 0),
                "mutation_type": trial["mutation_type"],
                "short_span": trial["short_span"],
                "long_span": trial["long_span"],
                "params": normalize_aux_params(trial.get("params", {})),
                "error": str(e),
            }
        rows.append(row)
    return rows, planned


def _sanitize_component(comp):
    c = dict(comp)
    c["short_span"] = int(c.get("short_span", 54))
    c["long_span"] = int(c.get("long_span", 90))
    if c["long_span"] < c["short_span"] + 30:
        c["long_span"] = c["short_span"] + 30
    c["mutation_type"] = c.get("mutation_type", EVOLVE_BENCHMARK_MUTATION)
    params = normalize_aux_params(c.get("params", {}))
    if c["mutation_type"] in ("vol_scale", "vol_adjusted", "multi_factor"):
        params["vol_window"] = int(params.get("vol_window", 20))
    if c["mutation_type"] in ("volume_gate", "volume_confirm"):
        params["vol_gate_window"] = int(params.get("vol_gate_window", 20))
    if c["mutation_type"] in ("regime_gate", "regime_momentum"):
        params["vix_threshold"] = float(params.get("vix_threshold", 20.0))
    c["params"] = params
    c["weight"] = float(c.get("weight", 1.0))
    return c


def _program_identity(program):
    comps = [_sanitize_component(c) for c in program.get("components", [])]
    if not comps:
        comps = [_sanitize_component({"mutation_type": EVOLVE_BENCHMARK_MUTATION, "short_span": 54, "long_span": 90, "params": {"vix_threshold": 26.0}, "weight": 1.0})]
    payload = [
        {
            "mutation_type": c["mutation_type"],
            "short_span": int(c["short_span"]),
            "long_span": int(c["long_span"]),
            "params": normalize_aux_params(c.get("params", {})),
            "weight": round(float(c.get("weight", 1.0)), 6),
        }
        for c in comps
    ]
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    variants = "+".join(sorted({c["mutation_type"] for c in comps}))
    short_bucket = int(round(float(np.median([c["short_span"] for c in comps])) / 3.0) * 3)
    long_bucket = int(round(float(np.median([c["long_span"] for c in comps])) / 3.0) * 3)
    cluster_id = f"program:{variants}:{short_bucket}:{long_bucket}"
    return payload, fingerprint, cluster_id


def _program_code(program):
    comps = [_sanitize_component(c) for c in program.get("components", [])]
    if not comps:
        comps = [_sanitize_component({"mutation_type": EVOLVE_BENCHMARK_MUTATION, "short_span": 54, "long_span": 90, "params": {"vix_threshold": 26.0}, "weight": 1.0})]
    lines = ["import numpy as np", "def signal(close, volume, vix=None, tnx=None):", "    parts = []", "    weights = []"]
    for i, c in enumerate(comps):
        ss = int(c["short_span"])
        ls = int(c["long_span"])
        var = c["mutation_type"]
        w = float(c.get("weight", 1.0))
        p = c.get("params", {})
        lines.append(f"    # comp_{i}: {var} {ss}/{ls}")
        lines.append(f"    fast_{i} = close.ewm(span={ss}, min_periods={ss}).mean().shift(1)")
        lines.append(f"    slow_{i} = close.ewm(span={ls}, min_periods={ls}).mean().shift(1)")
        lines.append(f"    core_{i} = fast_{i} / slow_{i} - 1.0")
        if var == "vol_scale":
            vw = int(p.get("vol_window", 20))
            lines.append(f"    vol_{i} = close.pct_change().rolling({vw}).std().replace(0, np.nan)")
            lines.append(f"    sig_{i} = (core_{i}.div(vol_{i}.clip(lower=1e-6), fill_value=0.0).clip(-3,3).rank(axis=1,pct=True)-0.5)*2")
        elif var == "vol_adjusted":
            vw = int(p.get("vol_window", 20))
            lines.append(f"    ret_{i} = close.pct_change({ls})")
            lines.append(f"    vol_{i} = close.pct_change().rolling({vw}).std().replace(0, np.nan)")
            lines.append(f"    sig_{i} = (ret_{i}.div(vol_{i}.clip(lower=1e-6), fill_value=0.0).clip(-3,3).rank(axis=1,pct=True)-0.5)*2")
        elif var == "volume_gate":
            gw = int(p.get("vol_gate_window", 20))
            lines.append(f"    vr_{i} = volume / volume.rolling({gw}).mean()")
            lines.append(f"    sig_{i} = ((core_{i} * vr_{i}.clip(lower=0.5, upper=1.5)).rank(axis=1,pct=True)-0.5)*2")
        elif var == "volume_confirm":
            gw = int(p.get("vol_gate_window", 20))
            lines.append(f"    ret_{i} = close.pct_change({ls})")
            lines.append(f"    vr_{i} = volume / volume.rolling({gw}).mean()")
            lines.append(f"    sig_{i} = ((ret_{i} * vr_{i}.clip(lower=0.5, upper=1.8)).rank(axis=1,pct=True)-0.5)*2")
        elif var == "regime_gate":
            thr = float(p.get("vix_threshold", 20.0))
            lines.append(f"    reg_{i} = 1.0 if vix is None else (vix.rolling(5).mean() < {thr}).astype(float).values[:, None]")
            lines.append(f"    sig_{i} = ((core_{i} * reg_{i}).rank(axis=1,pct=True)-0.5)*2")
        elif var == "regime_momentum":
            thr = float(p.get("vix_threshold", 20.0))
            lines.append(f"    ret_fast_{i} = close.pct_change({ss})")
            lines.append(f"    ret_slow_{i} = close.pct_change({ls})")
            lines.append(f"    reg_{i} = 1.0 if vix is None else (vix.rolling(5).mean() < {thr}).astype(float).values[:, None]")
            lines.append(f"    sig_{i} = (((ret_fast_{i} - ret_slow_{i}) * reg_{i}).rank(axis=1,pct=True)-0.5)*2")
        elif var == "ts_momentum":
            lines.append(f"    ret_{i} = close.pct_change({ls})")
            lines.append(f"    sig_{i} = (ret_{i}.rank(axis=1,pct=True)-0.5)*2")
        elif var == "short_reversal":
            lines.append(f"    rev_{i} = -close.pct_change({ss})")
            lines.append(f"    sig_{i} = (rev_{i}.rank(axis=1,pct=True)-0.5)*2")
        elif var == "multi_factor":
            vw = int(p.get("vol_window", 20))
            lines.append(f"    mom_{i} = close.pct_change({ls}).rank(axis=1,pct=True)-0.5")
            lines.append(f"    rev_{i} = (-close.pct_change({ss})).rank(axis=1,pct=True)-0.5")
            lines.append(f"    vol_{i} = close.pct_change().rolling({vw}).std().replace(0, np.nan)")
            lines.append(f"    lowvol_{i} = (-vol_{i}).rank(axis=1,pct=True)-0.5")
            lines.append(f"    combo_{i} = 0.55*mom_{i} + 0.25*rev_{i} + 0.20*lowvol_{i}")
            lines.append(f"    sig_{i} = (combo_{i}.rank(axis=1,pct=True)-0.5)*2")
        else:
            lines.append(f"    sig_{i} = (core_{i}.rank(axis=1,pct=True)-0.5)*2")
        lines.append(f"    parts.append(sig_{i}); weights.append({w})")
    lines += [
        "    wsum = sum(weights) if len(weights) else 1.0",
        "    out = sum(p * w for p, w in zip(parts, weights)) / wsum",
        "    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)",
        "    return out",
    ]
    return "\n".join(lines)


def _row_to_program(row):
    comp = {
        "mutation_type": row.get("mutation_type", EVOLVE_BENCHMARK_MUTATION),
        "short_span": int(row.get("short_span", 54)),
        "long_span": int(row.get("long_span", 90)),
        "params": normalize_aux_params(row.get("params", {})),
        "weight": 1.0,
    }
    return {"components": [comp]}


def _mutate_program(parent_program, gen_program, rng):
    program = {"components": [dict(c) for c in parent_program.get("components", [])]}
    if not program["components"]:
        program["components"] = [{"mutation_type": "regime_momentum", "short_span": 54, "long_span": 90, "params": {"vix_threshold": 26.0}, "weight": 1.0}]
    op_choices = ["tweak", "tweak", "tweak", "swap_variant", "swap_variant", "reweight", "add"]
    if len(program["components"]) > 1:
        op_choices.append("drop")
    op = rng.choice(op_choices)
    focus = list(gen_program.get("focus_variants", ["regime_momentum", "volume_confirm", "rank_norm"]))
    avoid = set(gen_program.get("avoid_variants", []))
    idx = rng.randrange(0, len(program["components"]))
    if op == "tweak":
        c = dict(program["components"][idx])
        c["short_span"] = int(c.get("short_span", 54)) + rng.choice([-6, -3, 3, 6])
        c["long_span"] = int(c.get("long_span", 90)) + rng.choice([-12, -6, 6, 12])
        program["components"][idx] = _sanitize_component(c)
    elif op == "add" and len(program["components"]) < 4:
        base = dict(program["components"][idx])
        base["mutation_type"] = rng.choice([v for v in focus if v not in avoid] or ["regime_momentum"])
        base["short_span"] = int(base.get("short_span", 54)) + rng.choice([-9, -6, 6, 9])
        base["long_span"] = int(base.get("long_span", 90)) + rng.choice([-18, -12, 12, 18])
        base["weight"] = rng.uniform(0.6, 1.4)
        program["components"].append(_sanitize_component(base))
    elif op == "drop" and len(program["components"]) > 1:
        del program["components"][idx]
    elif op == "reweight":
        for i in range(len(program["components"])):
            c = dict(program["components"][i])
            c["weight"] = float(max(0.2, min(2.0, c.get("weight", 1.0) * rng.uniform(0.8, 1.25))))
            program["components"][i] = _sanitize_component(c)
    elif op == "swap_variant":
        c = dict(program["components"][idx])
        c["mutation_type"] = rng.choice([v for v in focus if v not in avoid] or ["regime_momentum"])
        program["components"][idx] = _sanitize_component(c)
    program["components"] = [_sanitize_component(c) for c in program["components"]]
    return program


def _critic_assess_program(program):
    issues = []
    score_adj = 0.0
    comps = program.get("components", [])
    allowed_variants = {v for v, _ in parameter_search_variant_space()}
    core_variants = {"regime_momentum", "volume_confirm", "volume_gate"}
    composite_support = core_variants | {"rank_norm", "plain", "regime_gate", "ts_momentum"}
    if len(comps) == 0:
        issues.append("empty_program")
        score_adj -= 1.0
    if len(comps) > 4:
        issues.append("too_many_components")
        score_adj -= 0.15
    if len(comps) > 2:
        issues.append("bad_component_count")
        score_adj -= 0.35
    comp_variants = []
    for c in comps:
        ss = int(c.get("short_span", 54))
        ls = int(c.get("long_span", 90))
        variant = c.get("mutation_type", EVOLVE_BENCHMARK_MUTATION)
        comp_variants.append(variant)
        if ls < ss + 30:
            issues.append("invalid_span_gap")
            score_adj -= 0.20
        if variant not in allowed_variants:
            issues.append("unknown_variant")
            score_adj -= 0.50
            continue
        p = normalize_aux_params(c.get("params", {}))
        if variant in ("vol_scale", "vol_adjusted", "multi_factor"):
            if "vol_window" in p and int(p["vol_window"]) <= 1:
                issues.append("bad_vol_window")
                score_adj -= 0.30
            if "vol_window" in p and int(p["vol_window"]) > 252:
                issues.append("oversized_vol_window")
                score_adj -= 0.05
        if variant in ("volume_gate", "volume_confirm"):
            if "vol_gate_window" in p and int(p["vol_gate_window"]) <= 1:
                issues.append("bad_vol_gate_window")
                score_adj -= 0.30
        if variant in ("regime_gate", "regime_momentum"):
            threshold = float(p.get("vix_threshold", 20.0))
            if threshold <= 5.0 or threshold >= 80.0:
                issues.append("bad_vix_threshold")
                score_adj -= 0.30
    if len(comp_variants) >= 2:
        if len(set(comp_variants)) != len(comp_variants):
            issues.append("bad_duplicate_variant_composite")
            score_adj -= 0.25
        if not any(v in core_variants for v in comp_variants):
            issues.append("bad_composite_anchor")
            score_adj -= 0.40
        if any(v not in composite_support for v in comp_variants):
            issues.append("bad_composite_variant_mix")
            score_adj -= 0.30
        if any(v in ("vol_scale", "vol_adjusted", "multi_factor") for v in comp_variants):
            issues.append("bad_fragile_vol_composite")
            score_adj -= 0.45
    reject_issues = [i for i in issues if i.startswith("invalid_") or i.startswith("bad_") or i.startswith("unknown_")]
    return {"ok": len(reject_issues) == 0, "issues": issues, "score_adjust": float(score_adj)}


def _component_with_variant(template, variant):
    c = dict(template or {})
    c["mutation_type"] = variant
    params = normalize_aux_params(c.get("params", {}))
    if variant in ("volume_gate", "volume_confirm"):
        params = {"vol_gate_window": int(params.get("vol_gate_window", 20))}
    elif variant in ("regime_gate", "regime_momentum"):
        params = {"vix_threshold": float(params.get("vix_threshold", 22.0))}
    elif variant in ("vol_scale", "vol_adjusted", "multi_factor"):
        params = {"vol_window": int(params.get("vol_window", 20))}
    else:
        params = {}
    c["params"] = params
    return _sanitize_component(c)


def _repair_program_pre_eval(program, gen_program=None, rng=None):
    rng = rng or random.Random(PARAM_SEARCH_SEED)
    comps = [_sanitize_component(c) for c in (program or {}).get("components", [])]
    if not comps:
        comps = [_sanitize_component({"mutation_type": "volume_gate", "short_span": 60, "long_span": 105, "params": {"vol_gate_window": 20}, "weight": 1.0})]
    if len(comps) > 2:
        comps = comps[:2]
    seen = set()
    repaired = []
    replacement_pool = ["volume_gate", "volume_confirm", "rank_norm", "plain", "regime_gate", "ts_momentum"]
    for c in comps:
        variant = c.get("mutation_type", EVOLVE_BENCHMARK_MUTATION)
        if variant in seen:
            variant = next((v for v in replacement_pool if v not in seen), "volume_gate")
            c = _component_with_variant(c, variant)
        seen.add(c.get("mutation_type", variant))
        repaired.append(_sanitize_component(c))
    return {"components": repaired}


def _repair_program_after_failure(program, error_text, gen_program=None, rng=None):
    err = str(error_text or "").lower()
    comps = [_sanitize_component(c) for c in (program or {}).get("components", [])]
    base = comps[0] if comps else {"short_span": 60, "long_span": 105, "weight": 1.0}
    if "beta drift" in err:
        repaired = _component_with_variant(base, "volume_gate")
        repaired["params"] = {"vol_gate_window": int(repaired.get("params", {}).get("vol_gate_window", 20))}
        return {"components": [_sanitize_component(repaired)]}
    if "dead after market-neutral" in err or "not genuinely long-short" in err or "low signal activity" in err:
        repaired = _component_with_variant(base, "volume_confirm")
        repaired["params"] = {"vol_gate_window": int(repaired.get("params", {}).get("vol_gate_window", 20))}
        return {"components": [_sanitize_component(repaired)]}
    if "duplicate" in err or "bad_composite" in err:
        a = _component_with_variant(base, "volume_gate")
        b = _component_with_variant(base, "rank_norm")
        b["short_span"] = int(max(36, min(66, int(b.get("short_span", 54)) - 3)))
        b["long_span"] = int(max(b["short_span"] + 30, min(150, int(b.get("long_span", 90)) + 6)))
        return _repair_program_pre_eval({"components": [a, b]}, gen_program=gen_program, rng=rng)
    return None


def _split_train_validation():
    n = len(close_train)
    n_val = max(int(n * VAL_FRACTION), 252)
    n_val = min(max(60, n_val), max(60, n - 120))
    cut = n - n_val
    core_idx = close_train.index[:cut]
    val_idx = close_train.index[cut:]
    return (
        close_train.loc[core_idx],
        volume_train.loc[core_idx],
        (vix_train.loc[core_idx] if "vix_train" in globals() and vix_train is not None else None),
        (tnx_train.loc[core_idx] if "tnx_train" in globals() and tnx_train is not None else None),
        close_train.loc[val_idx],
        volume_train.loc[val_idx],
        (vix_train.loc[val_idx] if "vix_train" in globals() and vix_train is not None else None),
        (tnx_train.loc[val_idx] if "tnx_train" in globals() and tnx_train is not None else None),
    )


def _eval_program_trial(program, context):
    code = _program_code(program)
    c_core, v_core, x_core, t_core, c_val, v_val, x_val, t_val = context
    sig_core, err_core = run_signal_code(code, c_core, v_core, vix_s=x_core, tnx_s=t_core, timeout=EVAL_TIMEOUT_SEC)
    if err_core:
        return {"error": f"core_eval: {err_core}", "code": code}
    m_core = backtest(sig_core, c_core)
    flipped = bool(m_core.get("prefer_inverted", False))
    if flipped:
        sig_core = -sig_core
        m_core = backtest(sig_core, c_core)

    sig_val, err_val = run_signal_code(code, c_val, v_val, vix_s=x_val, tnx_s=t_val, timeout=EVAL_TIMEOUT_SEC)
    if err_val:
        return {"error": f"val_eval: {err_val}", "code": code}
    if flipped:
        sig_val = -sig_val
    m_val = backtest(sig_val, c_val)

    raw_cs_std = float(m_val.get("raw_cs_std", m_core.get("raw_cs_std", 1.0)) or 0.0)
    raw_long_frac = float(m_val.get("raw_long_frac", m_core.get("raw_long_frac", 1.0)) or 0.0)
    raw_short_frac = float(m_val.get("raw_short_frac", m_core.get("raw_short_frac", 1.0)) or 0.0)
    signal_activity = float(m_val.get("signal_activity", m_core.get("signal_activity", 1.0)) or 0.0)
    beta_val = float(m_val.get("beta", 0.0))
    if signal_activity < MIN_SIGNAL_ACTIVITY:
        return {"error": f"val_eval: SIGNAL_VALIDATION: low signal activity before walk-forward (active_frac={signal_activity:.2f})", "code": code}
    if raw_cs_std < MIN_RAW_CS_STD:
        return {"error": f"val_eval: SIGNAL_VALIDATION: weak cross-section before walk-forward (raw_cs_std={raw_cs_std:.3f})", "code": code}
    if min(raw_long_frac, raw_short_frac) < MIN_LONG_SHORT_FRAC:
        return {"error": f"val_eval: SIGNAL_VALIDATION: not genuinely long-short before walk-forward (long={raw_long_frac:.2f}, short={raw_short_frac:.2f})", "code": code}
    if abs(beta_val) > max(BETA_LIMIT * 1.5, 0.06):
        return {"error": f"val_eval: SIGNAL_VALIDATION: beta drift before walk-forward (beta={beta_val:+.3f})", "code": code}

    wf = []
    n = len(c_core)
    sz = max(n // WF_WINDOWS, 1)
    for w in range(WF_WINDOWS):
        lo = w * sz
        hi = (w + 1) * sz if w < WF_WINDOWS - 1 else n
        sc = c_core.iloc[lo:hi]
        sig_w = sig_core.iloc[lo:hi]
        mw = backtest(sig_w, sc)
        wf.append({"window": w, "sharpe": mw.get("sharpe", 0.0)})
    wf_median, wf_min = wf_summary(wf)

    val_wf = []
    n_val = len(c_val)
    val_windows = min(WF_WINDOWS, max(1, n_val // 126))
    val_sz = max(n_val // val_windows, 1)
    for w in range(val_windows):
        lo = w * val_sz
        hi = (w + 1) * val_sz if w < val_windows - 1 else n_val
        sc = c_val.iloc[lo:hi]
        if len(sc) < 60:
            continue
        sig_w = sig_val.iloc[lo:hi]
        mw = backtest(sig_w, sc)
        val_wf.append({"window": w, "sharpe": mw.get("sharpe", 0.0)})
    val_wf_median, val_wf_min = wf_summary(val_wf)
    val_wf_count = len(val_wf)

    train_sh = float(m_core.get("sharpe", 0.0))
    val_sh = float(m_val.get("sharpe", 0.0))
    train_cons = float(m_core.get("consistency", 0.0))
    val_cons = float(m_val.get("consistency", 0.0))
    to_val = float(m_val.get("avg_turnover", 0.0))
    max_dd_val = float(m_val.get("max_dd", 0.0))
    comps = [_sanitize_component(c) for c in program.get("components", [])]
    n_components = max(len(comps), 1)
    has_vol_scale = any(c.get("mutation_type") in ("vol_scale", "vol_adjusted") for c in comps)
    train_score = selection_score(
        train_sh,
        wf_median,
        train_cons,
        float(m_core.get("beta", 0.0)),
        float(m_core.get("avg_turnover", 0.0)),
        wf_min=wf_min,
        max_dd=float(m_core.get("max_dd", 0.0)),
    )
    raw_val_score = evolution_score(val_sh, val_wf_median, val_wf_min, val_cons, beta_val, to_val)
    val_score = raw_val_score
    val_wf_missing_penalty = EVOLVE_VAL_WF_MISSING_PENALTY if val_wf_count < 2 else 0.0
    val_score -= EVOLVE_COMPLEXITY_PENALTY * max(0, n_components - 1)
    val_score -= EVOLVE_VOL_SCALE_PENALTY if has_vol_scale else 0.0
    val_score -= val_wf_missing_penalty
    val_score -= EVOLVE_TRAIN_VAL_GAP_PENALTY * max(0.0, train_score - raw_val_score)
    robustness_payload = {
        "train_sharpe": val_sh,
        "wf_median": val_wf_median,
        "wf_min": val_wf_min,
        "beta": beta_val,
        "turnover": to_val,
        "consistency": val_cons,
        "raw_cs_std": raw_cs_std,
        "raw_long_frac": raw_long_frac,
        "raw_short_frac": raw_short_frac,
        "signal_activity": signal_activity,
    }
    program_robustness_score = robustness_score(robustness_payload)
    robust = (
        train_sh > 0
        and val_sh > 0
        and val_wf_median >= EVOLVE_PROGRAM_MIN_VAL_WF_MEDIAN
        and val_wf_min >= EVOLVE_PROGRAM_MIN_VAL_WF_MIN
        and wf_min > -0.20
        and robust_ok(robustness_payload, wf_floor=-0.25)
        and program_robustness_score >= EVOLVE_PROGRAM_ROBUSTNESS_FLOOR
    )

    comp0 = _sanitize_component(program.get("components", [{}])[0])
    program_payload, program_hash, program_cluster = _program_identity(program)
    meta = signature_for_signal(
        "program_evolution",
        "program",
        code,
        short_span=comp0.get("short_span", 54),
        long_span=comp0.get("long_span", 90),
        base_family="program_evolution",
        params={"n_components": len(program_payload), "program_hash": program_hash},
        cluster_id=program_cluster,
    )

    val_ret = (sig_val.shift(1).fillna(0) * c_val.pct_change().fillna(0)).mean(axis=1)
    return {
        "source": "parameter_search_evolution",
        "family": "program_evolution",
        "base_family": "program_evolution",
        "mutation_type": comp0.get("mutation_type", "program"),
        "short_span": int(comp0.get("short_span", 54)),
        "long_span": int(comp0.get("long_span", 90)),
        "params": {"n_components": len(program_payload), "program_hash": program_hash},
        "component_params": normalize_aux_params(comp0.get("params", {})),
        "program": program,
        "program_components": program_payload,
        "program_hash": program_hash,
        "code": code,
        "train_sharpe": train_sh,
        "val_sharpe": val_sh,
        "consistency": val_cons,
        "wf_median": wf_median,
        "wf_min": wf_min,
        "val_wf_median": val_wf_median,
        "val_wf_min": val_wf_min,
        "val_wf_count": val_wf_count,
        "beta": beta_val,
        "turnover": to_val,
        "max_dd": max_dd_val,
        "score_train": train_score,
        "score_val_raw": raw_val_score,
        "score_val": val_score,
        "complexity_penalty": EVOLVE_COMPLEXITY_PENALTY * max(0, n_components - 1),
        "val_wf_missing_penalty": val_wf_missing_penalty,
        "train_val_gap_penalty": EVOLVE_TRAIN_VAL_GAP_PENALTY * max(0.0, train_score - raw_val_score),
        "score": val_score,
        "robustness_score": program_robustness_score,
        "raw_cs_std": raw_cs_std,
        "raw_long_frac": raw_long_frac,
        "raw_short_frac": raw_short_frac,
        "signal_activity": signal_activity,
        "robust_ok": bool(robust),
        "component_count": n_components,
        "model_size": n_components,
        "model_size_key": f"components={n_components}",
        "flipped": flipped,
        "_val_ret": val_ret,
        **meta,
    }


def _candidate_programs(parents, gen_program, n_trials, rng):
    if not parents:
        parents = [{
            "parent_id": f"seed:{EVOLVE_BENCHMARK_MUTATION}:54:90",
            "program": {"components": [{"mutation_type": "regime_momentum", "short_span": 54, "long_span": 90, "params": {"vix_threshold": 26.0}, "weight": 1.0}]},
        }]
    candidates = []
    seen = set()
    parent_records = []
    for p in parents:
        parent_records.append(
            {
                "row": p,
                "program": p.get("program") if isinstance(p.get("program"), dict) else _row_to_program(p),
                "parent_id": row_identity(p, "parent"),
                "parent_signature": p.get("signature"),
                "parent_program_hash": p.get("program_hash"),
                "parent_score": p.get("score"),
            }
        )
    min_short = int(min(PARAM_SEARCH_SHORT_SPANS))
    max_short = int(max(PARAM_SEARCH_SHORT_SPANS))
    min_long = int(min(PARAM_SEARCH_LONG_SPANS))
    max_long = int(max(PARAM_SEARCH_LONG_SPANS))
    regime_thresholds = [float(x) for x in PARAM_SEARCH_REGIME_THRESHOLDS] if "PARAM_SEARCH_REGIME_THRESHOLDS" in globals() else [20.0, 24.0, 28.0]
    vol_windows = [int(x) for x in PARAM_SEARCH_VOL_WINDOWS] if "PARAM_SEARCH_VOL_WINDOWS" in globals() else [20, 30]
    vol_gate_windows = [int(x) for x in PARAM_SEARCH_VOL_GATE_WINDOWS] if "PARAM_SEARCH_VOL_GATE_WINDOWS" in globals() else [20, 40]
    force_explore = bool(gen_program.get("force_explore", False))
    allowed_variants = {v for v, _ in parameter_search_variant_space()}
    focus_variants = [v for v in gen_program.get("focus_variants", ["regime_momentum"]) if v in allowed_variants]
    fallback_focus = ["regime_momentum", "volume_confirm", "volume_gate", "rank_norm", "plain", "regime_gate", "ts_momentum"]
    if float(gen_program.get("dominant_cluster_frac", 0.0) or 0.0) >= 0.50:
        fallback_focus = ["volume_confirm", "volume_gate", "rank_norm", "plain", "regime_momentum", "regime_gate", "ts_momentum"]
    for variant in fallback_focus:
        if variant in allowed_variants and variant not in focus_variants:
            focus_variants.append(variant)
        if len(focus_variants) >= EVOLVE_MIN_FOCUS_VARIANTS:
            break
    if not focus_variants:
        focus_variants = ["regime_momentum", "volume_confirm", "volume_gate", "rank_norm", "regime_gate"]

    def _sample_component_for_variant(var):
        ss = int(rng.randint(min_short, max_short))
        ls = int(rng.randint(max(min_long, ss + 30), max_long))
        params = {}
        if var in ("regime_gate", "regime_momentum"):
            params["vix_threshold"] = float(rng.choice(regime_thresholds))
        elif var in ("vol_scale", "vol_adjusted", "multi_factor"):
            params["vol_window"] = int(rng.choice(vol_windows))
        elif var in ("volume_gate", "volume_confirm"):
            params["vol_gate_window"] = int(rng.choice(vol_gate_windows))
        return _sanitize_component(
            {
                "mutation_type": var,
                "short_span": ss,
                "long_span": ls,
                "params": params,
                "weight": float(rng.uniform(0.6, 1.4)),
            }
        )

    def _sample_component():
        return _sample_component_for_variant(rng.choice(focus_variants))

    def _quota_program(family):
        if family == "composite":
            a, b = rng.choice(EVOLVE_SAFE_COMPOSITE_PAIRS)
            return {"components": [_sample_component_for_variant(a), _sample_component_for_variant(b)]}
        return {"components": [_sample_component_for_variant(family)]}

    def _add_candidate(prog, parent_meta, origin):
        prog = _repair_program_pre_eval(prog, gen_program, rng)
        _, sig, _ = _program_identity(prog)
        if sig in seen:
            return False
        seen.add(sig)
        candidates.append(
            {
                "program": prog,
                "program_sig": sig,
                "candidate_family": _program_primary_family(prog),
                "parent_id": parent_meta.get("parent_id"),
                "parent_signature": parent_meta.get("parent_signature"),
                "parent_program_hash": parent_meta.get("parent_program_hash"),
                "parent_score": parent_meta.get("parent_score"),
                "origin": origin,
            }
        )
        return True

    quota_total = max(0, min(n_trials, int(round(n_trials * 0.65))))
    quota_items = list(EVOLVE_FAMILY_QUOTAS.items())
    quota_counts = {family: max(1, int(round(quota_total * frac))) for family, frac in quota_items}
    while sum(quota_counts.values()) > quota_total and quota_counts:
        family = max(quota_counts, key=lambda k: quota_counts[k])
        quota_counts[family] -= 1
    for family, quota in quota_counts.items():
        if family not in allowed_variants and family != "composite":
            continue
        attempts = 0
        while len(candidates) < n_trials and quota > 0 and attempts < max(quota * 20, 50):
            attempts += 1
            parent_meta = {
                "parent_id": f"quota:{family}:g{gen_program.get('gen_id', 'unknown')}",
                "parent_signature": None,
                "parent_program_hash": None,
                "parent_score": None,
            }
            if _add_candidate(_quota_program(family), parent_meta, f"family_quota:{family}"):
                quota -= 1

    while len(candidates) < n_trials:
        parent_meta = {
            "parent_id": f"explore:g{gen_program.get('gen_id', 'unknown')}",
            "parent_signature": None,
            "parent_program_hash": None,
            "parent_score": None,
        }
        if force_explore or rng.random() < EVOLVE_RANDOM_EXPLORATION_FRAC:
            n_comp = 1 if rng.random() < 0.88 else 2
            base = {"components": [_sample_component() for _ in range(n_comp)]}
            prog = _mutate_program(base, gen_program, rng)
        else:
            parent_meta = parent_records[rng.randrange(0, len(parent_records))]
            parent_prog = parent_meta["program"]
            prog = _mutate_program(parent_prog, gen_program, rng)
        _add_candidate(prog, parent_meta, "program_mutation")
        if len(seen) > max(n_trials * 60, 20000):
            break
    return candidates


def _deterministic_champion_candidates(global_rows, limit):
    anchors = [
        r for r in global_rows
        if r.get("source") == EVOLVE_BENCHMARK_SOURCE
        and r.get("robust_ok")
        and r.get("score") is not None
    ]
    anchors = sorted(anchors, key=lambda r: (-_heldout_proxy_score(r), abs(int(r.get("long_span", 90)) - 90)))
    candidates = []
    seen = set()
    
    def add_row(row):
        program = _row_to_program(row)
        _, program_hash, _ = _program_identity(program)
        if program_hash in seen:
            return False
        seen.add(program_hash)
        candidates.append(
            {
                "program": program,
                "program_sig": program_hash,
                "origin": "deterministic_champion",
                "parent_id": row_identity(row, "deterministic"),
                "parent_signature": row.get("signature"),
                "parent_program_hash": row.get("program_hash"),
                "parent_score": row.get("score"),
            }
        )
        return True

    for variant in EVOLVE_CHAMPION_VARIANTS:
        for row in [r for r in anchors if r.get("mutation_type") == variant]:
            add_row(row)
            break
        if len(candidates) >= limit:
            return candidates[:limit]
    for row in anchors:
        if len(candidates) >= limit:
            break
        add_row(row)
    return candidates


def _tournament_select(rows, k_survivors, rng):
    pool = [r for r in rows if r.get("robust_ok") and r.get("score") is not None]
    if not pool:
        pool = [r for r in rows if r.get("score") is not None]
    original_pool = list(pool)
    selected = []
    used = set()
    used_clusters = set()
    while len(selected) < k_survivors and pool:
        contenders = rng.sample(pool, k=min(EVOLVE_TOURNAMENT_K, len(pool)))
        best = None
        best_u = -1e9
        for c in contenders:
            u = float(c.get("score", -999.0)) + 0.15 * float(c.get("score_train", c.get("score", -999.0)))
            if selected and c.get("_val_ret") is not None:
                cors = []
                for s in selected:
                    if s.get("_val_ret") is None:
                        continue
                    ix = c["_val_ret"].index.intersection(s["_val_ret"].index)
                    if len(ix) < 30:
                        continue
                    corr = c["_val_ret"].loc[ix].corr(s["_val_ret"].loc[ix])
                    if pd.notna(corr):
                        cors.append(abs(float(corr)))
                if cors:
                    u -= EVOLVE_DIVERSITY_PENALTY * float(np.mean(cors))
            if u > best_u:
                best_u = u
                best = c
        if best is None:
            break
        sig = best.get("signature")
        if sig and sig in used:
            pool = [p for p in pool if p is not best]
            continue
        cid = best.get("cluster_id")
        if cid and cid in used_clusters and len(pool) > 1:
            pool = [p for p in pool if p is not best]
            continue
        if sig:
            used.add(sig)
        if cid:
            used_clusters.add(cid)
        selected.append(best)
        pool = [p for p in pool if p is not best]
    return _family_capped_rows(selected + [p for p in original_pool if p not in selected], k_survivors)


def _merge_parent_candidates(*groups, limit):
    merged = []
    seen = set()
    for group in groups:
        for row in group or []:
            ident = (
                row.get("program_hash")
                or row.get("signature")
                or row.get("parent_id")
                or row.get("cluster_id")
                or row_identity(row, "parent")
            )
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(row)
    return _family_capped_rows(merged, limit)


def _select_next_generation_parents(current_parents, valid, robust, global_rows, rng):
    reseed = []
    did_reseed = False
    if len(robust) < EVOLVE_ROBUST_RESET_THRESHOLD:
        robust_global = [r for r in global_rows if r.get("robust_ok")]
        reseed = _diverse_top_parents_from_rows(robust_global, EVOLVE_BEAM_WIDTH)
        did_reseed = bool(reseed)

    survivor_pool = robust if robust else valid
    survivors = _tournament_select(survivor_pool, EVOLVE_SURVIVORS, rng)

    if did_reseed:
        parents = _merge_parent_candidates(reseed, survivors, current_parents, limit=EVOLVE_BEAM_WIDTH)
    elif survivors:
        parents = survivors
    else:
        parents = list(current_parents or [])
    return parents, survivors, did_reseed


def _inc_count(mapping, key, n=1):
    key = str(key or "unknown")
    mapping[key] = int(mapping.get(key, 0)) + int(n)


def _generation_family_telemetry(candidates, rows, survivors):
    telemetry = {
        "generated": {},
        "critic_rejected": {},
        "validation_failed": {},
        "scored": {},
        "robust": {},
        "survivor": {},
        "failure_reasons": {},
    }
    for cand in candidates or []:
        _inc_count(telemetry["generated"], cand.get("candidate_family") or _program_primary_family(cand.get("program", {})))
    for row in rows or []:
        fam = _row_family(row)
        err = row.get("error")
        if row.get("score") is not None:
            _inc_count(telemetry["scored"], fam)
            if row.get("robust_ok"):
                _inc_count(telemetry["robust"], fam)
        elif err:
            bucket = "critic_rejected" if str(err).startswith("critic_reject:") else "validation_failed"
            _inc_count(telemetry[bucket], fam)
            reason = str(err).split(":", 1)[0] if ":" in str(err) else str(err)
            reason_map = telemetry["failure_reasons"].setdefault(fam, {})
            _inc_count(reason_map, reason)
    for row in survivors or []:
        _inc_count(telemetry["survivor"], _row_family(row))
    return telemetry


def run_evolution_search():
    all_rows = []
    generations = []
    memory_notes = []
    programs = []
    diagnosis_events = []
    parents = _seed_parents_for_generation()
    rng = random.Random(PARAM_SEARCH_SEED)
    c_core, v_core, x_core, t_core, c_val, v_val, x_val, t_val = _split_train_validation()
    eval_context = (c_core, v_core, x_core, t_core, c_val, v_val, x_val, t_val)
    total_evals = 0
    stop_reason = "max_generations_reached"
    seen_clusters = set([p.get("cluster_id") for p in parents if p.get("cluster_id")])
    started = time.time()
    prev_best = -1e9
    prev_median = -1e9
    prev_div = 1.0
    patience_bad = 0
    zero_robust_streak = 0
    best_so_far_streak = 0
    global_rows = [r for r in _safe_load_search_results() if r.get("score") is not None]

    for gen_id in range(1, EVOLVE_MAX_GENERATIONS + 1):
        if total_evals >= EVOLVE_MAX_TOTAL_EVALS:
            stop_reason = "max_total_evals_reached"
            break
        if EVOLVE_MAX_WALLCLOCK_HOURS and (time.time() - started) / 3600.0 >= EVOLVE_MAX_WALLCLOCK_HOURS:
            stop_reason = "max_wallclock_reached"
            break

        recent_rows = all_rows[-400:] if all_rows else global_rows[:400]
        gen_program = _build_generation_program(gen_id, parents, recent_rows)
        programs.append(gen_program)

        remaining = EVOLVE_MAX_TOTAL_EVALS - total_evals
        target = min(EVOLVE_TRIALS_PER_GENERATION, remaining)
        candidates = _candidate_programs(parents, gen_program, target, rng)
        champion_candidates = _deterministic_champion_candidates(global_rows, EVOLVE_DETERMINISTIC_CHAMPIONS)
        if champion_candidates:
            seen_programs = {c.get("program_sig") for c in candidates}
            injected = []
            for cand in champion_candidates:
                if cand.get("program_sig") in seen_programs:
                    continue
                injected.append(cand)
                seen_programs.add(cand.get("program_sig"))
            candidates = injected + candidates
            candidates = candidates[:target]
        if len(candidates) < EVOLVE_MIN_TRIALS_PER_GENERATION:
            min_needed = EVOLVE_MIN_TRIALS_PER_GENERATION
            attempts = 0
            while len(candidates) < min_needed and attempts < EVOLVE_RESTART_ATTEMPTS:
                attempts += 1
                refill_program = dict(gen_program)
                refill_program["force_explore"] = True
                refill_target = max(min_needed - len(candidates), EVOLVE_MIN_TRIALS_PER_GENERATION)
                extra = _candidate_programs(parents, refill_program, refill_target, rng)
                if not extra:
                    continue
                seen_prog = {c.get("program_sig") for c in candidates}
                for c in extra:
                    ps = c.get("program_sig")
                    if ps in seen_prog:
                        continue
                    seen_prog.add(ps)
                    candidates.append(c)
                if len(candidates) >= min_needed:
                    break
            if len(candidates) < min_needed:
                floor = min(EVOLVE_MIN_CANDIDATE_FLOOR, target)
                if len(candidates) < floor:
                    stop_reason = "search_space_exhausted"
                    break
                _warn(f"g{gen_id}: candidate floor relaxed to {len(candidates)} due sparse search space")

        gen_rows = []
        for cand in candidates:
            critique = _critic_assess_program(cand["program"])
            if not critique["ok"]:
                repaired_program = _repair_program_after_failure(
                    cand["program"],
                    "critic_reject:" + ",".join(critique["issues"]),
                    gen_program=gen_program,
                    rng=rng,
                )
                if repaired_program is not None:
                    repaired_payload, repaired_hash, _ = _program_identity(repaired_program)
                    if repaired_hash != cand.get("program_sig"):
                        cand["program"] = repaired_program
                        cand["program_sig"] = repaired_hash
                        cand["candidate_family"] = _program_primary_family(repaired_program)
                        cand["repair_applied"] = "critic_pre_eval"
                        critique = _critic_assess_program(cand["program"])
                if not critique["ok"]:
                    program_payload, program_hash, program_cluster = _program_identity(cand["program"])
                    primary_family = _program_primary_family(cand["program"])
                    gen_rows.append(
                        {
                            "source": "parameter_search_evolution",
                            "family": "program_evolution",
                            "base_family": "program_evolution",
                            "mutation_type": primary_family,
                            "candidate_family": cand.get("candidate_family", primary_family),
                            "error": "critic_reject:" + ",".join(critique["issues"]),
                            "gen_id": gen_id,
                            "parent_id": cand.get("parent_id") or f"unknown:g{gen_id}",
                            "parent_signature": cand.get("parent_signature"),
                            "parent_program_hash": cand.get("parent_program_hash"),
                            "program_sig": cand.get("program_sig"),
                            "program_hash": program_hash,
                            "cluster_id": program_cluster,
                            "program": cand["program"],
                            "program_components": program_payload,
                            "origin": cand.get("origin", "program_mutation"),
                            "repair_applied": cand.get("repair_applied"),
                        }
                    )
                    continue
            row = _eval_program_trial(cand["program"], eval_context)
            if row.get("score") is None and row.get("error"):
                repaired_program = _repair_program_after_failure(cand["program"], row.get("error"), gen_program=gen_program, rng=rng)
                if repaired_program is not None:
                    repaired_payload, repaired_hash, _ = _program_identity(repaired_program)
                    if repaired_hash != cand.get("program_sig"):
                        repaired_row = _eval_program_trial(repaired_program, eval_context)
                        repaired_row["repair_source_error"] = row.get("error")
                        repaired_row["repair_applied"] = "failure_repair"
                        cand["program"] = repaired_program
                        cand["program_sig"] = repaired_hash
                        cand["candidate_family"] = _program_primary_family(repaired_program)
                        row = repaired_row
            row.setdefault("source", "parameter_search_evolution")
            row.setdefault("family", "program_evolution")
            row.setdefault("base_family", "program_evolution")
            row["mutation_type"] = row.get("mutation_type") or _program_primary_family(cand["program"])
            row["candidate_family"] = cand.get("candidate_family", _program_primary_family(cand["program"]))
            row["gen_id"] = gen_id
            row["parent_id"] = cand.get("parent_id") or row.get("parent_id") or f"unknown:g{gen_id}"
            row["parent_signature"] = cand.get("parent_signature")
            row["parent_program_hash"] = cand.get("parent_program_hash")
            row["program_sig"] = cand["program_sig"]
            row["origin"] = cand.get("origin", "program_mutation")
            if row.get("score") is not None:
                row["score"] = float(row["score"] + critique.get("score_adjust", 0.0))
            parent_score = cand.get("parent_score")
            row["score_gain_vs_parent"] = float(row["score"] - parent_score) if row.get("score") is not None and parent_score is not None else None
            gen_rows.append(row)

        total_evals += len(gen_rows)
        valid = [r for r in gen_rows if r.get("score") is not None]
        robust = [r for r in valid if r.get("robust_ok")]
        valid_rate = float(len(valid) / len(gen_rows)) if gen_rows else 0.0
        zero_robust_counted = len(robust) == 0 and valid_rate >= EVOLVE_ZERO_ROBUST_MIN_VALID_RATE
        zero_robust_streak = zero_robust_streak + 1 if zero_robust_counted else 0
        parents, survivors, did_reseed = _select_next_generation_parents(parents, valid, robust, global_rows, rng)
        if did_reseed:
            _warn(f"g{gen_id}: robust_count={len(robust)} below threshold; reseeded parents")

        gen_scores = [r.get("score") for r in robust if r.get("score") is not None]
        ci_lo, ci_hi = _bootstrap_ci(gen_scores, n_boot=300, alpha=0.10, seed=PARAM_SEARCH_SEED + gen_id)
        best_score = max(gen_scores, default=max([r.get("score", -1e9) for r in valid], default=-1e9))
        med_score = float(np.median(gen_scores)) if gen_scores else -1e9
        new_clusters = set([r.get("cluster_id") for r in valid if r.get("cluster_id") and r.get("cluster_id") not in seen_clusters])
        new_cluster_count = len(new_clusters)
        seen_clusters.update(new_clusters)
        prior_best = None if prev_best <= -1e8 else float(prev_best)
        if prior_best is None or best_score > prior_best:
            best_so_far_streak = 0
        else:
            best_so_far_streak += 1
        best_so_far = None if max(prev_best, best_score) <= -1e8 else float(max(prev_best, best_score))
        diversity = 0.0
        if len(survivors) >= 2:
            cors = []
            for i in range(len(survivors)):
                for j in range(i + 1, len(survivors)):
                    si, sj = survivors[i].get("_val_ret"), survivors[j].get("_val_ret")
                    if si is None or sj is None:
                        continue
                    ix = si.index.intersection(sj.index)
                    if len(ix) < 30:
                        continue
                    c = si.loc[ix].corr(sj.loc[ix])
                    if pd.notna(c):
                        cors.append(abs(float(c)))
            diversity = float(1.0 - np.mean(cors)) if cors else 0.0

        family_telemetry = _generation_family_telemetry(candidates, gen_rows, survivors)
        best_gain = 0.0 if prev_best <= -1e8 else float(best_score - prev_best)
        median_gain = 0.0 if prev_median <= -1e8 else float(med_score - prev_median)
        diversity_gain = float(diversity - prev_div)
        prev_best = max(prev_best, best_score)
        prev_median = med_score
        prev_div = diversity

        no_growth = (
            best_gain < EVOLVE_MIN_BEST_GAIN
            and median_gain < EVOLVE_MIN_MEDIAN_GAIN
            and new_cluster_count < EVOLVE_MIN_NOVELTY_GAIN
            and diversity_gain < EVOLVE_MIN_DIVERSITY_GAIN
        )
        patience_bad = patience_bad + 1 if no_growth else 0

        for r in gen_rows:
            if "_val_ret" in r:
                del r["_val_ret"]
        all_rows.extend(gen_rows)
        global_rows.extend([r for r in valid if r.get("score") is not None])

        benchmark_candidates = [
            r for r in global_rows
            if r.get("source") == EVOLVE_BENCHMARK_SOURCE
            and r.get("score") is not None
        ]
        benchmark_candidates = sorted(
            benchmark_candidates,
            key=lambda row: (not bool(row.get("robust_ok")), -float(row.get("score", -999.0) or -999.0)),
        )
        benchmark_row = benchmark_candidates[0] if benchmark_candidates else None

        generation_payload = {
            "gen_id": gen_id,
            "planned_trials": len(candidates),
            "eval_count": len(gen_rows),
            "valid_count": len(valid),
            "valid_rate": valid_rate,
            "robust_count": len(robust),
            "zero_robust_streak": int(zero_robust_streak),
            "zero_robust_counted": bool(zero_robust_counted),
            "best_score": None if best_score <= -1e8 else float(best_score),
            "best_so_far": best_so_far,
            "best_so_far_streak": int(best_so_far_streak),
            "median_score": None if med_score <= -1e8 else float(med_score),
            "best_gain": float(best_gain),
            "median_gain": float(median_gain),
            "new_cluster_count": int(new_cluster_count),
            "diversity": float(diversity),
            "score_ci_90": [ci_lo, ci_hi],
            "survivor_count": len(survivors),
            "survivor_signatures": [s.get("signature") for s in survivors],
            "survivor_families": family_telemetry.get("survivor", {}),
            "family_telemetry": family_telemetry,
            "program_focus_variants": list(gen_program.get("focus_variants", [])),
            "adaptive_diagnostic_run": bool(EVOLVE_ADAPTIVE_DIAGNOSTIC_RUN),
        }
        gen_diagnoses = _generation_diagnosis_events(generation_payload, valid, robust, benchmark_row=benchmark_row)
        generation_payload["diagnoses"] = gen_diagnoses
        diagnosis_events.extend(gen_diagnoses)
        for event in gen_diagnoses:
            if event.get("kind") == "stagnation":
                _warn(
                    f"g{gen_id}: stagnation diagnosis triggered after "
                    f"{event.get('streak', 0)} generations without best-so-far improvement"
                )
            elif event.get("kind") == "zero_robust":
                _warn(
                    f"g{gen_id}: zero-robust diagnosis "
                    f"{event.get('failure_counts', {})}"
                )

        reflection_text = _summarize_generation_reflection(gen_id, gen_rows, survivors)
        memory_notes.append({
            "gen_id": gen_id,
            "reflection": reflection_text,
            "best_gain": best_gain,
            "median_gain": median_gain,
            "new_clusters": new_cluster_count,
            "diversity": diversity,
            "valid_rate": valid_rate,
            "zero_robust_streak": zero_robust_streak,
            "best_so_far_streak": best_so_far_streak,
            "diagnoses": gen_diagnoses,
        })

        generations.append(generation_payload)

        stop_after_generation = False
        if gen_id >= EVOLVE_ZERO_ROBUST_MIN_GENERATIONS and zero_robust_streak >= EVOLVE_ZERO_ROBUST_PATIENCE:
            stop_reason = "zero_robust_streak_reached"
            stop_after_generation = True
        elif gen_id >= EVOLVE_MIN_GENERATIONS and best_so_far_streak >= EVOLVE_HARD_STAGNATION_PATIENCE:
            stop_reason = "stagnation_reached"
            stop_after_generation = True
        elif gen_id >= EVOLVE_MIN_GENERATIONS and patience_bad >= EVOLVE_CONVERGENCE_PATIENCE:
            stop_reason = "convergence_reached"
            stop_after_generation = True

        if EVOLVE_PARTIAL_WRITE_EVERY and gen_id % max(1, int(EVOLVE_PARTIAL_WRITE_EVERY)) == 0:
            _write_evolution_artifacts(
                all_rows,
                generations,
                memory_notes,
                programs,
                diagnosis_events,
                total_evals,
                stop_reason,
                partial=True,
                started=started,
                suppress_errors=True,
            )

        if stop_after_generation:
            break

    all_rows = _write_evolution_artifacts(
        all_rows,
        generations,
        memory_notes,
        programs,
        diagnosis_events,
        total_evals,
        stop_reason,
        partial=False,
        started=started,
    )
    return all_rows


def run_parameter_search(n_trials=PARAM_SEARCH_TRIALS):
    if EVOLVE_ENABLED:
        return run_evolution_search()
    trials = sample_parameter_trials(n_trials=n_trials)
    rows = []
    seen_sig = set()
    for trial in trials:
        try:
            row = evaluate_structured_trial(trial, source="parameter_search")
        except Exception as e:
            row = {
                "source": "parameter_search",
                "parent_id": f"search:{trial['mutation_type']}:{trial['short_span']}:{trial['long_span']}",
                "family": deterministic_family_for_mutation(trial["mutation_type"])[0],
                "base_family": "ewm",
                "mutation_type": trial["mutation_type"],
                "short_span": trial["short_span"],
                "long_span": trial["long_span"],
                "params": normalize_aux_params(trial.get("params", {})),
                "error": str(e),
            }
        sig = row.get("signature")
        if sig and sig in seen_sig:
            continue
        if sig:
            seen_sig.add(sig)
        rows.append(row)
    rows = sorted(rows, key=lambda r: (not r.get("robust_ok", False), -r.get("score", -999)))
    _atomic_write_json(PARAM_SEARCH_FILE, rows)
    return rows


def run_deterministic_search():
    short_grid = [36, 39, 42, 45, 48, 51, 54, 57, 60]
    long_grid = [90, 95, 100, 105, 110, 120, 130, 140]
    variants = deterministic_variant_grid()
    rows = []
    for short_span in short_grid:
        for long_span in long_grid:
            if long_span < short_span + 30:
                continue
            for variant, aux in variants:
                fn = deterministic_signal_factory(short_span, long_span, variant=variant, aux=aux)
                try:
                    sig = fn(close_train, volume_train, vix_train, tnx_train)
                    m = backtest(sig, close_train)
                    flipped = bool(m["prefer_inverted"])
                    beta = m["beta_inverted"] if flipped else m["beta"]
                    train_sharpe = m["sharpe_inverted"] if flipped else m["sharpe"]
                    ann_return = m["ann_return_inverted"] if flipped else m["ann_return"]
                    dd = m["max_dd_inverted"] if flipped else m["max_dd"]
                    consistency = m["consistency_inverted"] if flipped else m["consistency"]
                    wf = baseline_walk_forward(fn, close_train, volume_train, vix_train, tnx_train, flipped=flipped)
                    wf_median, wf_min = wf_summary(wf)
                    turnover = m["avg_turnover"]
                    score = selection_score(train_sharpe, wf_median, consistency, beta, turnover, wf_min=wf_min, max_dd=dd)
                    family, base_family = deterministic_family_for_mutation(variant)
                    aux_norm = normalize_aux_params(aux)
                    code_label = f"ewm(span={short_span}) ewm(span={long_span}) {variant} {aux_norm}"
                    meta = signature_for_signal(
                        family,
                        variant,
                        code_label,
                        short_span=short_span,
                        long_span=long_span,
                        base_family=base_family,
                        params=aux_norm,
                    )
                    aux_tag = ",".join(f"{k}={v}" for k, v in aux_norm.items()) if aux_norm else "base"
                    metric_payload = {
                        "train_sharpe": train_sharpe,
                        "wf_median": wf_median,
                        "wf_min": wf_min,
                        "beta": beta,
                        "turnover": turnover,
                        "consistency": consistency,
                        "raw_cs_std": m.get("raw_cs_std", 1.0),
                        "raw_long_frac": m.get("raw_long_frac", 1.0),
                        "raw_short_frac": m.get("raw_short_frac", 1.0),
                        "signal_activity": m.get("signal_activity", 1.0),
                    }
                    row = {
                        "source": "deterministic",
                        "parent_id": f"det:{variant}:{short_span}:{long_span}:{aux_tag}",
                        "family": family,
                        "base_family": base_family,
                        "mutation_type": variant,
                        "short_span": short_span,
                        "long_span": long_span,
                        "params": aux_norm,
                        "train_sharpe": train_sharpe,
                        "ann_return": ann_return,
                        "beta": beta,
                        "turnover": turnover,
                        "max_dd": dd,
                        "consistency": consistency,
                        "wf_median": wf_median,
                        "wf_min": wf_min,
                        "score": score,
                        "robustness_score": robustness_score(metric_payload),
                        "raw_cs_std": m.get("raw_cs_std", 1.0),
                        "raw_long_frac": m.get("raw_long_frac", 1.0),
                        "raw_short_frac": m.get("raw_short_frac", 1.0),
                        "signal_activity": m.get("signal_activity", 1.0),
                        "flipped": flipped,
                        "shortlist_ok": shortlist_ok(beta, turnover),
                        "robust_ok": robust_ok(metric_payload),
                        "component_count": 1,
                        "model_size": 1,
                        "model_size_key": "components=1",
                        **meta,
                    }
                    rows.append(row)
                except Exception as e:
                    rows.append({
                        "source": "deterministic",
                        "parent_id": f"det:{variant}:{short_span}:{long_span}",
                        "family": deterministic_family_for_mutation(variant)[0],
                        "base_family": "ewm",
                        "mutation_type": variant,
                        "short_span": short_span,
                        "long_span": long_span,
                        "params": normalize_aux_params(aux),
                        "error": str(e),
                    })
    rows = sorted(rows, key=lambda r: (not r.get("robust_ok", False), -r.get("score", -999)))
    _atomic_write_json(DETERMINISTIC_FILE, rows)
    return rows


def _cols_present(df, cols):
    return [c for c in cols if c in df.columns]


if RUN_BASELINE_SWEEP:
    BASELINE_RESULTS = run_baseline_sweep()
    baseline_df = pd.DataFrame(BASELINE_RESULTS)
    display(baseline_df.head(12))
    print("baseline summary:\n" + baseline_df.head(12).to_string(index=False))
else:
    print("baseline sweep paused")

if RUN_DETERMINISTIC_SEARCH:
    DETERMINISTIC_RESULTS = run_deterministic_search()
    deterministic_df = pd.DataFrame([r for r in DETERMINISTIC_RESULTS if r.get("score") is not None])
    if not deterministic_df.empty:
        raw_cols = [
            "score", "robustness_score", "train_sharpe", "wf_median", "wf_min", "beta", "turnover",
            "family", "base_family", "mutation_type", "short_span", "long_span",
            "cluster_id", "signature", "params", "model_size_key", "robust_ok",
        ]
        display(deterministic_df.sort_values("score", ascending=False)[_cols_present(deterministic_df, raw_cols)].head(12))
        frontier_df = deterministic_df[deterministic_df["robust_ok"]].sort_values("score", ascending=False)
        if not frontier_df.empty:
            frontier_cols = [
                "score", "robustness_score", "train_sharpe", "wf_median", "wf_min", "beta", "turnover",
                "family", "mutation_type", "short_span", "long_span", "cluster_id", "signature", "params", "model_size_key",
            ]
            frontier_cols = _cols_present(frontier_df, frontier_cols)
            display(frontier_df[frontier_cols].head(12))
            print("frontier after filters:\n" + frontier_df[frontier_cols].head(12).to_string(index=False))
        else:
            print("frontier after filters: no deterministic rows passed robust gates")
    else:
        print("deterministic search produced no valid rows")
else:
    print("deterministic search paused")

if RUN_PARAM_SEARCH:
    PARAM_SEARCH_RESULTS = run_parameter_search()
    parameter_df = pd.DataFrame([r for r in PARAM_SEARCH_RESULTS if r.get("score") is not None])
    if EVOLVE_ENABLED and EVOLUTION_SUMMARY_FILE.exists():
        evo_summary = json.loads(EVOLUTION_SUMMARY_FILE.read_text())
        gens = evo_summary.get("generations", [])
        best_evo = evo_summary.get("best_score")
        best_evo_txt = f"{best_evo:+.2f}" if isinstance(best_evo, (int, float)) else "n/a"
        print(
            f"evolution: gens={evo_summary.get('generations_executed', 0)} "
            f"stop={evo_summary.get('stop_reason', 'n/a')} "
            f"best={best_evo_txt} evals={evo_summary.get('total_evals', 0)}"
        )
        for gg in gens:
            gg_best = gg.get("best_score")
            gg_best_txt = f"{gg_best:+.2f}" if isinstance(gg_best, (int, float)) else "n/a"
            print(
                f"  g{gg.get('gen_id')}: trials={gg.get('planned_trials')} valid={gg.get('valid_count')} "
                f"valid_rate={gg.get('valid_rate', 0.0):.2%} robust={gg.get('robust_count')} "
                f"zero_streak={gg.get('zero_robust_streak', 0)} best={gg_best_txt} "
                f"best_gain={gg.get('best_gain', 0.0):+.3f} median_gain={gg.get('median_gain', 0.0):+.3f}"
            )
    if not parameter_df.empty:
        param_cols = [
            "score", "robustness_score", "train_sharpe", "wf_median", "wf_min", "beta", "turnover",
            "family", "mutation_type", "short_span", "long_span", "cluster_id", "signature", "params", "model_size_key", "robust_ok",
        ]
        param_cols = _cols_present(parameter_df, param_cols)
        display(parameter_df.sort_values(["robust_ok", "score"], ascending=[False, False])[param_cols].head(12))
        print("parameter search frontier:\n" + parameter_df.sort_values(["robust_ok", "score"], ascending=[False, False])[param_cols].head(12).to_string(index=False))
    else:
        print("parameter search produced no valid rows")
else:
    print("parameter search paused")
