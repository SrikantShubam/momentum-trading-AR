TOP_K = 5
HELDOUT_SHORTLIST_K = 20
HELDOUT_MAX_PER_CLUSTER = 2
HELDOUT_MIN_DETERMINISTIC = 5
HELDOUT_MIN_EVOLUTION = 10
HELDOUT_MIN_EVOLUTION_CHAMPIONS = 5
HELDOUT_MAX_PER_MUTATION = 4
COST_STRESS_BPS = (5.0, 10.0, 15.0)
ENSEMBLE_TOP_N = 3
ECONOMIC_SHARPE_FLOOR = globals().get("ECONOMIC_SHARPE_FLOOR", 0.50)
ECONOMIC_EDGE_OVER_DETERMINISTIC = globals().get("ECONOMIC_EDGE_OVER_DETERMINISTIC", 0.05)
APPROVED_WINNER_EDGE_OVER_DETERMINISTIC = globals().get("APPROVED_WINNER_EDGE_OVER_DETERMINISTIC", 0.10)
CHRONOLOGICAL_HOLDOUT_SEGMENTS = max(0, int(globals().get("CHRONOLOGICAL_HOLDOUT_SEGMENTS", 3) or 0))
ROLLING_HOLDOUT_SEGMENTS = max(0, int(globals().get("ROLLING_HOLDOUT_SEGMENTS", 3) or 0))
ROLLING_HOLDOUT_MIN_POINTS = max(20, int(globals().get("ROLLING_HOLDOUT_MIN_POINTS", 126) or 126))
SUBPERIOD_WINDOWS = (
    ("2015-2018", "2015-01-01", "2018-12-31"),
    ("2019-2021", "2019-01-01", "2021-12-31"),
    ("2022-2024", "2022-01-01", "2024-12-31"),
)
EVOLUTION_DEEP_DIVE_FILE = OUT / "evolution_deep_dive.md"
EVOLUTION_TLDR_FILE = OUT / "evolution_tldr.md"


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


def _runtime_scope_metadata():
    path = globals().get("RUNTIME_METADATA_FILE")
    return _safe_json_dict(path) if path is not None else {}


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
    keys = (
        "source",
        "family",
        "base_family",
        "benchmark_family",
        "model_family",
        "study_family",
        "study_type",
        "label",
        "name",
    )
    tokens = set()
    for key in keys:
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


def _partial_report_path():
    path = globals().get("PARTIAL_REPORT_FILE")
    if path is None and "OUT" in globals():
        path = OUT / "partial_run_report.json"
    return path


def _load_partial_run_report():
    return _safe_json_dict(_partial_report_path())


def _row_identity_safe(row, default_prefix="row"):
    helper = globals().get("row_identity")
    if callable(helper):
        try:
            return str(helper(row, default_prefix=default_prefix))
        except TypeError:
            try:
                return str(helper(row, default_prefix))
            except Exception:
                pass
        except Exception:
            pass
    if not isinstance(row, dict):
        return f"{default_prefix}:unknown"
    for key in ("parent_id", "iter", "program_hash", "program_sig", "signature", "cluster_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{default_prefix}:unknown"


def _heldout_iter_label(row):
    if not isinstance(row, dict):
        return "row:unknown"
    source = row.get("source", "unknown")
    signature = str(row.get("signature") or row.get("program_hash") or row.get("program_sig") or "nosig")[:8]
    cluster_id = str(row.get("cluster_id", "unknown"))
    if source == "parameter_search_evolution":
        return f"evo:{cluster_id}:{signature}"
    if source == "deterministic":
        return str(row.get("parent_id") or _row_identity_safe(row, default_prefix="det"))
    return _row_identity_safe(row, default_prefix=str(source or default_prefix))


def _selection_score_safe(sharpe, wf_median, consistency, beta, avg_turnover, wf_min=0.0, max_dd=0.0):
    try:
        return selection_score(
            sharpe,
            wf_median,
            consistency,
            beta,
            avg_turnover,
            wf_min=wf_min,
            max_dd=max_dd,
        )
    except TypeError:
        return selection_score(sharpe, wf_median, consistency, beta, avg_turnover)


def _robustness_score_safe(metric_row):
    helper = globals().get("robustness_score")
    if callable(helper):
        try:
            return helper(metric_row)
        except Exception:
            return None
    return None


def _median_value(values):
    vals = sorted([float(v) for v in values if v is not None])
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _window_median_sharpe(windows):
    if not isinstance(windows, list):
        return None
    return _median_value([_float_or(w.get("sharpe"), None) for w in windows if isinstance(w, dict)])


def _cost_stress_avg_sharpe(row):
    stress = row.get("cost_stress", {}) if isinstance(row, dict) else {}
    vals = []
    if isinstance(stress, dict):
        for payload in stress.values():
            if isinstance(payload, dict):
                vals.append(_float_or(payload.get("sharpe"), None))
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _build_chronological_window_specs(index_like, segments, min_points=60):
    index = list(index_like) if index_like is not None else []
    n = len(index)
    if segments <= 0 or n < min_points:
        return []
    window_len = max(n // segments, min_points)
    if window_len > n:
        return []
    specs = []
    lo = 0
    slot = 1
    while lo < n and len(specs) < segments:
        hi = n if len(specs) == segments - 1 else min(n, lo + window_len)
        if hi - lo < min_points:
            break
        specs.append(
            {
                "label": f"chrono_{slot}",
                "lo": lo,
                "hi": hi,
                "start": str(index[lo])[:10],
                "end": str(index[hi - 1])[:10],
            }
        )
        lo = hi
        slot += 1
    return specs


def _build_rolling_window_specs(index_like, segments, min_points=126):
    index = list(index_like) if index_like is not None else []
    n = len(index)
    if segments <= 0 or n < min_points:
        return []
    window_len = max(min_points, n // max(segments, 1))
    if window_len > n:
        return []
    max_start = max(n - window_len, 0)
    if segments == 1:
        starts = [max_start]
    else:
        stride = max(max_start // max(segments - 1, 1), 1)
        starts = []
        for i in range(segments):
            starts.append(min(i * stride, max_start))
        starts = sorted(set(starts))
        if starts[-1] != max_start:
            starts.append(max_start)
    specs = []
    for idx, lo in enumerate(starts, start=1):
        hi = min(n, lo + window_len)
        if hi - lo < min_points:
            continue
        specs.append(
            {
                "label": f"roll_{idx}",
                "lo": lo,
                "hi": hi,
                "start": str(index[lo])[:10],
                "end": str(index[hi - 1])[:10],
            }
        )
        if len(specs) >= segments:
            break
    return specs


def _supporting_diagnostic_report(candidate, baseline):
    diagnostics = []
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        return {"diagnostics": diagnostics, "wins": 0, "losses": 0, "ties": 0, "comparable": 0, "not_losing_majority": False}

    metric_specs = (
        ("test_score", "composite score", _float_or(candidate.get("test_score"), None), _float_or(baseline.get("test_score"), None)),
        ("test_sharpe", "test Sharpe", _float_or(candidate.get("test_sharpe"), None), _float_or(baseline.get("test_sharpe"), None)),
        ("wf_median", "WF median", _float_or(candidate.get("wf_median"), None), _float_or(baseline.get("wf_median"), None)),
        ("wf_min", "WF min", _float_or(candidate.get("wf_min"), None), _float_or(baseline.get("wf_min"), None)),
        (
            "test_robustness_score",
            "robustness score",
            _float_or(candidate.get("test_robustness_score", candidate.get("train_robustness_score")), None),
            _float_or(baseline.get("test_robustness_score", baseline.get("train_robustness_score")), None),
        ),
        (
            "test_benchmark_spread_sharpe",
            "benchmark-spread Sharpe",
            _float_or(candidate.get("test_benchmark_spread_sharpe"), None),
            _float_or(baseline.get("test_benchmark_spread_sharpe"), None),
        ),
        ("cost_stress_avg_sharpe", "cost-stress avg Sharpe", _cost_stress_avg_sharpe(candidate), _cost_stress_avg_sharpe(baseline)),
        ("subperiod_median_sharpe", "subperiod median Sharpe", _window_median_sharpe(candidate.get("subperiods")), _window_median_sharpe(baseline.get("subperiods"))),
        (
            "chronological_median_sharpe",
            "chronological holdout median Sharpe",
            _window_median_sharpe(candidate.get("chronological_holdout")),
            _window_median_sharpe(baseline.get("chronological_holdout")),
        ),
        (
            "rolling_median_sharpe",
            "rolling holdout median Sharpe",
            _window_median_sharpe(candidate.get("rolling_holdout")),
            _window_median_sharpe(baseline.get("rolling_holdout")),
        ),
    )

    wins = 0
    losses = 0
    ties = 0
    for key, label, candidate_value, baseline_value in metric_specs:
        if candidate_value is None or baseline_value is None:
            continue
        delta = float(candidate_value - baseline_value)
        if abs(delta) <= 1e-9:
            verdict = "tie"
            ties += 1
        elif delta > 0:
            verdict = "win"
            wins += 1
        else:
            verdict = "loss"
            losses += 1
        diagnostics.append(
            {
                "key": key,
                "label": label,
                "candidate": float(candidate_value),
                "baseline": float(baseline_value),
                "delta": delta,
                "verdict": verdict,
            }
        )

    comparable = wins + losses + ties
    return {
        "diagnostics": diagnostics,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "comparable": comparable,
        "not_losing_majority": bool(comparable and losses * 2 <= comparable),
    }


def _heldout_rule_reason_lines(heldout_report):
    if not isinstance(heldout_report, dict):
        return ["held-out winner rule not evaluated"]
    reasons = []
    leader = heldout_report.get("leader")
    baseline = heldout_report.get("best_deterministic")
    comparison = heldout_report.get("comparison", {})
    if leader is None:
        return ["held-out evaluation produced no leader"]
    if baseline is None:
        reasons.append("no deterministic baseline available for held-out comparison")
    elif leader.get("source") == "deterministic":
        reasons.append("highest composite-score candidate is still the deterministic baseline")
    edge = heldout_report.get("score_edge_vs_deterministic")
    required_edge = heldout_report.get("required_edge", APPROVED_WINNER_EDGE_OVER_DETERMINISTIC)
    if edge is None:
        reasons.append("score edge vs deterministic baseline unavailable")
    elif edge < required_edge:
        reasons.append(f"composite-score edge {edge:+.2f} is below required {required_edge:+.2f}")
    if not comparison.get("not_losing_majority"):
        reasons.append(
            "supporting diagnostics lost majority "
            f"({comparison.get('losses', 0)} losses across {comparison.get('comparable', 0)} comparable diagnostics)"
        )
    return reasons or ["winner rule met"]


def _build_heldout_report(test_rows):
    rows = [r for r in test_rows if isinstance(r, dict)]
    ranked = sorted(rows, key=lambda r: -_float_or(r.get("test_score"), -999.0))
    leader = ranked[0] if ranked else None
    best_deterministic = max(
        [r for r in rows if r.get("source") == "deterministic"],
        key=lambda r: _float_or(r.get("test_score"), -999.0),
        default=None,
    )
    best_evolution = max(
        [r for r in rows if r.get("source") == "parameter_search_evolution"],
        key=lambda r: _float_or(r.get("test_score"), -999.0),
        default=None,
    )
    score_edge = None
    if leader and best_deterministic:
        score_edge = float(_float_or(leader.get("test_score")) - _float_or(best_deterministic.get("test_score")))
    comparison = _supporting_diagnostic_report(leader, best_deterministic)
    edge_ok = bool(best_deterministic and score_edge is not None and score_edge >= APPROVED_WINNER_EDGE_OVER_DETERMINISTIC)
    source_ok = bool(leader and leader.get("source") != "deterministic")
    diagnostics_ok = bool(comparison.get("not_losing_majority"))
    winner = leader if leader and source_ok and edge_ok and diagnostics_ok else None
    return {
        "ranked": ranked,
        "leader": leader,
        "winner": winner,
        "best_deterministic": best_deterministic,
        "best_evolution": best_evolution,
        "score_edge_vs_deterministic": score_edge,
        "required_edge": APPROVED_WINNER_EDGE_OVER_DETERMINISTIC,
        "comparison": comparison,
        "reasons": _heldout_rule_reason_lines(
            {
                "leader": leader,
                "best_deterministic": best_deterministic,
                "comparison": comparison,
                "score_edge_vs_deterministic": score_edge,
                "required_edge": APPROVED_WINNER_EDGE_OVER_DETERMINISTIC,
            }
        ),
        "status": "approved_winner" if winner else "no_winner_yet",
    }


def walk_forward(code_str, close_df, volume_df, flipped=False, n_windows=WF_WINDOWS):
    N = len(close_df)
    sz = max(N // n_windows, 1)
    per_window = []
    for w in range(n_windows):
        lo = w * sz
        hi = (w + 1) * sz if w < n_windows - 1 else N
        sub_c, sub_v = close_df.iloc[lo:hi], volume_df.iloc[lo:hi]
        sig, err = run_signal_code(code_str, sub_c, sub_v, vix_s=vix_test.iloc[lo:hi] if vix_test is not None else None, tnx_s=tnx_test.iloc[lo:hi] if tnx_test is not None else None, timeout=20)
        if err:
            per_window.append({"window": w, "error": err})
            continue
        if flipped:
            sig = -sig
        m = backtest(sig, sub_c)
        per_window.append({"window": w, "sharpe": m.get("sharpe", 0.0), "benchmark_spread_sharpe": m.get("benchmark_spread_sharpe", m.get("excess_sharpe", 0.0)), "ann_return": m.get("ann_return", 0.0), "beta": m.get("beta", 0.0), "max_dd": m.get("max_dd", 0.0), "turnover": m.get("avg_turnover", 0.0)})
    return per_window


def deterministic_code(row):
    if row.get("code"):
        return row["code"]
    short_span = row["short_span"]
    long_span = row["long_span"]
    variant = row["mutation_type"]
    params = normalize_aux_params(row.get("params", {}))
    vol_window = int(params.get("vol_window", 20))
    vol_gate_window = int(params.get("vol_gate_window", 20))
    vol_cap = float(params.get("vol_cap", 3.0))
    vix_threshold = float(params.get("vix_threshold", 24.0))
    if variant == "plain":
        return f"""def signal(close, volume, vix=None, tnx=None):
    fast = close.ewm(span={short_span}, min_periods={short_span}).mean().shift(1)
    slow = close.ewm(span={long_span}, min_periods={long_span}).mean().shift(1)
    core = fast / slow - 1.0
    out = (core.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "rank_norm":
        return f"""def signal(close, volume, vix=None, tnx=None):
    fast = close.ewm(span={short_span}, min_periods={short_span}).mean().shift(1)
    slow = close.ewm(span={long_span}, min_periods={long_span}).mean().shift(1)
    core = fast / slow - 1.0
    out = (core.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "vol_scale":
        return f"""def signal(close, volume, vix=None, tnx=None):
    fast = close.ewm(span={short_span}, min_periods={short_span}).mean().shift(1)
    slow = close.ewm(span={long_span}, min_periods={long_span}).mean().shift(1)
    core = fast / slow - 1.0
    vol = close.pct_change().rolling({vol_window}).std().replace(0, np.nan)
    scaled = core.div(vol.clip(lower=1e-6), fill_value=0.0).clip(-3.0, 3.0)
    out = (scaled.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "volume_gate":
        return f"""def signal(close, volume, vix=None, tnx=None):
    fast = close.ewm(span={short_span}, min_periods={short_span}).mean().shift(1)
    slow = close.ewm(span={long_span}, min_periods={long_span}).mean().shift(1)
    core = fast / slow - 1.0
    vratio = volume / volume.rolling({vol_gate_window}).mean()
    gated = core * vratio.clip(lower=0.5, upper=1.5)
    out = (gated.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "regime_gate":
        return f"""def signal(close, volume, vix=None, tnx=None):
    fast = close.ewm(span={short_span}, min_periods={short_span}).mean().shift(1)
    slow = close.ewm(span={long_span}, min_periods={long_span}).mean().shift(1)
    core = fast / slow - 1.0
    regime = 1.0 if vix is None else (vix.rolling(5).mean() < {vix_threshold}).astype(float).values[:, None]
    gated = core * regime
    out = (gated.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "ts_momentum":
        return f"""def signal(close, volume, vix=None, tnx=None):
    trend = close.pct_change({long_span})
    out = (trend.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "short_reversal":
        return f"""def signal(close, volume, vix=None, tnx=None):
    reversal = -close.pct_change({short_span})
    out = (reversal.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant in ("vol_adjusted", "vol_adjusted_momentum"):
        return f"""def signal(close, volume, vix=None, tnx=None):
    trend = close.pct_change({long_span})
    vol = close.pct_change().rolling({vol_window}).std().replace(0, np.nan)
    scaled = trend.div(vol.clip(lower=1e-6), fill_value=0.0).clip(-{vol_cap}, {vol_cap})
    out = (scaled.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant in ("volume_confirm", "volume_confirmed_momentum"):
        return f"""def signal(close, volume, vix=None, tnx=None):
    trend = close.pct_change({long_span})
    vratio = volume / volume.rolling({vol_gate_window}).mean()
    confirmed = trend * vratio.clip(lower=0.5, upper=1.8)
    out = (confirmed.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "regime_momentum":
        return f"""def signal(close, volume, vix=None, tnx=None):
    trend_fast = close.pct_change({short_span})
    trend_slow = close.pct_change({long_span})
    trend = trend_fast - trend_slow
    if vix is not None:
        regime = (vix.rolling(5).mean() < {vix_threshold}).astype(float).values[:, None]
        trend = trend * regime
    out = (trend.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    if variant == "multi_factor":
        return f"""def signal(close, volume, vix=None, tnx=None):
    trend = close.pct_change({long_span}).rank(axis=1, pct=True) - 0.5
    reversal = (-close.pct_change({short_span})).rank(axis=1, pct=True) - 0.5
    vol = close.pct_change().rolling({vol_window}).std().replace(0, np.nan)
    low_vol = (-vol).rank(axis=1, pct=True) - 0.5
    combined = 0.55 * trend + 0.25 * reversal + 0.20 * low_vol
    out = (combined.rank(axis=1, pct=True) - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out"""

    raise ValueError(variant)


def _oriented_metrics(backtest_row, flipped=False):
    if flipped:
        return {
            "sharpe": backtest_row["sharpe_inverted"],
            "ann_return": backtest_row["ann_return_inverted"],
            "max_dd": backtest_row["max_dd_inverted"],
            "beta": backtest_row["beta_inverted"],
            "consistency": backtest_row["consistency_inverted"],
            "equity": backtest_row["equity_inverted"],
        }
    return {
        "sharpe": backtest_row["sharpe"],
        "ann_return": backtest_row["ann_return"],
        "max_dd": backtest_row["max_dd"],
        "beta": backtest_row["beta"],
        "consistency": backtest_row["consistency"],
        "equity": backtest_row["equity"],
    }


def _concat_optional(train_series, test_series):
    if train_series is None and test_series is None:
        return None
    if train_series is None:
        return test_series
    if test_series is None:
        return train_series
    return pd.concat([train_series, test_series]).sort_index()


def _full_test_panels():
    close_full = globals().get("close_all")
    if close_full is None:
        close_full = pd.concat([close_train, close_test]).sort_index()
    volume_full = globals().get("volume_all")
    if volume_full is None:
        volume_full = pd.concat([volume_train, volume_test]).sort_index()
    vix_full = globals().get("vix_all")
    if vix_full is None:
        vix_full = _concat_optional(globals().get("vix_train"), globals().get("vix_test"))
    tnx_full = globals().get("tnx_all")
    if tnx_full is None:
        tnx_full = _concat_optional(globals().get("tnx_train"), globals().get("tnx_test"))
    return close_full, volume_full, vix_full, tnx_full


def _oriented_signal_code(code_str, flipped=False):
    """Return executable signal code in the same orientation used during evaluation."""
    code_str = str(code_str or "")
    if not flipped:
        return code_str
    if "def signal(" not in code_str:
        return code_str
    raw_code = code_str.replace("def signal(", "def _raw_signal(", 1)
    return (
        raw_code.rstrip()
        + "\n\n\n"
        + "def signal(close, volume, vix=None, tnx=None):\n"
        + "    out = -_raw_signal(close, volume, vix=vix, tnx=tnx)\n"
        + "    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)\n"
        + "    return out\n"
    )


def _ensemble_code(rows):
    blocks = []
    calls = []
    for i, row in enumerate(rows):
        fn_code = row.get("code") or deterministic_code(row)
        if not row.get("code_is_oriented"):
            fn_code = _oriented_signal_code(fn_code, bool(row.get("flipped")))
        fn_code = fn_code.replace("def signal(", f"def _signal_{i}(", 1)
        blocks.append(fn_code)
        calls.append(f"    s{i} = _signal_{i}(close, volume, vix=vix, tnx=tnx)")
    avg_expr = " + ".join([f"s{i}" for i in range(len(rows))])
    meta_lines = "\n".join(
        [
            f"# member_{i + 1}: iter={row.get('iter')} cluster={row.get('cluster_id')} test_sc={row.get('test_score', 0.0):+.2f}"
            for i, row in enumerate(rows)
        ]
    )
    return (
        f"# objective=market_neutral_net_sharpe ensemble_top_n={len(rows)}\n"
        f"{meta_lines}\n\n"
        + "import numpy as np\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + "def signal(close, volume, vix=None, tnx=None):\n"
        + "\n".join(calls)
        + "\n"
        + f"    out = ({avg_expr}) / {float(len(rows)):.1f}\n"
        + "    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)\n"
        + "    return out\n"
    )


def _safe_load_search_results_local():
    if "load_search_results" in globals():
        try:
            rows = _as_row_list(load_search_results())
            if rows:
                return rows
        except Exception:
            pass
    rows = []
    if "load_parameter_search" in globals():
        try:
            rows.extend(_as_row_list(load_parameter_search()))
        except Exception:
            pass
    elif "PARAM_SEARCH_FILE" in globals() and PARAM_SEARCH_FILE.exists():
        rows.extend(_safe_json_rows(PARAM_SEARCH_FILE))
    if "load_deterministic" in globals():
        try:
            rows.extend(_as_row_list(load_deterministic()))
        except Exception:
            pass
    elif "DETERMINISTIC_FILE" in globals() and DETERMINISTIC_FILE.exists():
        rows.extend(_safe_json_rows(DETERMINISTIC_FILE))
    return rows


def _rank_candidates(rows):
    candidates = [r for r in rows if r.get("score") is not None]
    robust = [r for r in candidates if r.get("robust_ok")]
    return sorted(robust if robust else candidates, key=lambda r: (not r.get("robust_ok", False), -_float_or(r.get("score"), -999.0)))


def _pick_diverse(rows, limit, seen_sig=None, per_cluster=None, per_mutation=None):
    picked = []
    seen_sig = seen_sig if seen_sig is not None else set()
    per_cluster = per_cluster if per_cluster is not None else {}
    per_mutation = per_mutation if per_mutation is not None else {}
    for row in _rank_candidates(rows):
        sig = row.get("signature")
        cid = row.get("cluster_id", "unknown")
        mutation_type = row.get("mutation_type", "unknown")
        if sig and sig in seen_sig:
            continue
        if per_cluster.get(cid, 0) >= HELDOUT_MAX_PER_CLUSTER:
            continue
        if per_mutation.get(mutation_type, 0) >= HELDOUT_MAX_PER_MUTATION:
            continue
        picked.append(row)
        if sig:
            seen_sig.add(sig)
        per_cluster[cid] = per_cluster.get(cid, 0) + 1
        per_mutation[mutation_type] = per_mutation.get(mutation_type, 0) + 1
        if len(picked) >= limit:
            break
    return picked


def _evolution_health_lines(evo_summary, partial_report=None):
    partial_report = partial_report if isinstance(partial_report, dict) else {}
    generations = evo_summary.get("generations", []) if isinstance(evo_summary, dict) else []
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
            tail = zero_robust[-5:] if zero_robust else []
            lines.append(f"zero_robust_generations={len(zero_robust)} latest={tail}")
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


def _write_evolution_analysis(test_rows):
    evo_summary = _safe_json_dict(globals().get("EVOLUTION_SUMMARY_FILE"))
    partial_report = _load_partial_run_report()
    if not evo_summary and not partial_report:
        return

    search_rows = [r for r in _safe_load_search_results_local() if r.get("score") is not None]
    robust_rows = [r for r in search_rows if r.get("robust_ok")]
    by_cluster = {}
    for r in robust_rows:
        cid = r.get("cluster_id", "unknown")
        prev = by_cluster.get(cid)
        if prev is None or _float_or(r.get("score"), -999.0) > _float_or(prev.get("score"), -999.0):
            by_cluster[cid] = r
    top_clusters = sorted(by_cluster.values(), key=lambda r: -_float_or(r.get("score"), -999.0))[:8]
    generations = [g for g in evo_summary.get("generations", []) if isinstance(g, dict)]
    health_lines = _evolution_health_lines(evo_summary, partial_report)
    partial_line = _partial_report_line(partial_report)

    deep = []
    deep.append("# Evolution Deep Dive\n")
    deep.append("## Run Policy")
    p = evo_summary.get("policy", {}) if isinstance(evo_summary.get("policy"), dict) else {}
    deep.append(
        f"- generations={evo_summary.get('generations_executed')} stop_reason={evo_summary.get('stop_reason')} "
        f"max_generations={p.get('max_generations')} trials_per_generation={p.get('trials_per_generation')} "
        f"beam={p.get('beam_width')} survivors={p.get('survivors')} min_gain={p.get('min_gen_growth')}"
    )
    if health_lines:
        deep.append("\n## Validity / Robustness Watch")
        deep.extend([f"- {line}" for line in health_lines])
    if partial_line:
        deep.append("\n## Partial Run Report")
        deep.append(f"- {partial_line}")
    deep.append("\n## Generation Trace")
    for rr in generations:
        deep.append(
            f"- g{rr.get('gen_id')}: valid={rr.get('valid_count')} robust={rr.get('robust_count')} "
            f"best={rr.get('best_score')} gain={rr.get('improvement_vs_prev_best')} stalled={rr.get('stalled')} "
            f"survivors={rr.get('survivor_count')}"
        )
    deep.append("\n## Best Robust Clusters (Train)")
    for r in top_clusters:
        line = (
            f"- {r.get('cluster_id')}: score={_float_or(r.get('score')):+.2f} "
            f"trainSh={_float_or(r.get('train_sharpe')):+.2f} wf={_float_or(r.get('wf_median')):+.2f}/{_float_or(r.get('wf_min')):+.2f} "
            f"beta={_float_or(r.get('beta')):+.2f} to={_float_or(r.get('turnover')):.2f}"
        )
        if r.get("robustness_score") is not None:
            line += f" robustScore={_float_or(r.get('robustness_score')):.1f}"
        deep.append(line)
    if test_rows:
        heldout_report = _build_heldout_report(test_rows)
        leader = heldout_report.get("leader")
        winner = heldout_report.get("winner")
        best_det = heldout_report.get("best_deterministic")
        comparison = heldout_report.get("comparison", {})
        score_edge = heldout_report.get("score_edge_vs_deterministic")
        best_evo = heldout_report.get("best_evolution")
        evo_edge = None if not (best_det and best_evo) else float(best_evo["test_score"] - best_det["test_score"])
        econ_success = bool(
            best_evo
            and best_evo["test_sharpe"] >= ECONOMIC_SHARPE_FLOOR
            and (best_det is None or evo_edge >= ECONOMIC_EDGE_OVER_DETERMINISTIC)
        )
        deep.append("\n## Held-out Verdict")
        if leader:
            deep.append(
                f"- Leader by composite score: {leader['cluster_id']} ({leader['iter']}) "
                f"score={leader['test_score']:+.2f} testSh={leader['test_sharpe']:+.2f} "
                f"dd={leader['test_dd']:+.1%} beta={leader['test_beta']:+.2f}"
            )
        if best_det:
            deep.append(
                f"- Best deterministic baseline: {best_det['cluster_id']} ({best_det['iter']}) "
                f"score={best_det['test_score']:+.2f} testSh={best_det['test_sharpe']:+.2f}"
            )
        if winner:
            deep.append(
                f"- Approved winner: {winner['cluster_id']} ({winner['iter']}) "
                f"edge_vs_det={_float_or(score_edge):+.2f} diagnostics="
                f"{comparison.get('wins', 0)}W/{comparison.get('losses', 0)}L/{comparison.get('ties', 0)}T"
            )
        else:
            deep.append("- Approved winner: no winner yet")
            deep.extend([f"- Rule gap: {reason}" for reason in heldout_report.get("reasons", [])])
        deep.append(
            f"- Winner rule: highest composite score, edge >= {APPROVED_WINNER_EDGE_OVER_DETERMINISTIC:+.2f} "
            "vs best deterministic baseline, and no majority loss across supporting diagnostics."
        )
        deep.append(
            f"- Economic success: {econ_success} "
            f"(requires evolved Sharpe >= {ECONOMIC_SHARPE_FLOOR:.2f} and edge >= {ECONOMIC_EDGE_OVER_DETERMINISTIC:+.2f})"
        )
        if best_det and leader:
            deep.append(
                f"- Leader vs deterministic baseline: leader={leader['test_score']:+.2f}/{leader['test_sharpe']:+.2f} "
                f"det={best_det['test_score']:+.2f}/{best_det['test_sharpe']:+.2f} "
                f"edge={_float_or(score_edge):+.2f} diagnostics="
                f"{comparison.get('wins', 0)}W/{comparison.get('losses', 0)}L/{comparison.get('ties', 0)}T"
            )
            for diag in comparison.get("diagnostics", []):
                deep.append(
                    f"  - {diag['label']}: candidate={diag['candidate']:+.2f} "
                    f"baseline={diag['baseline']:+.2f} delta={diag['delta']:+.2f} [{diag['verdict']}]"
                )
        cs = (leader or {}).get("cost_stress", {})
        if cs:
            deep.append(
                "- Cost stress Sharpe: "
                + " | ".join([f"{k}:{v['sharpe']:+.2f}" for k, v in cs.items()])
            )
        subs = (leader or {}).get("subperiods", [])
        if subs:
            deep.append(
                "- Subperiod Sharpe: "
                + " | ".join([f"{s['label']}:{s['sharpe']:+.2f}" for s in subs])
            )

    EVOLUTION_DEEP_DIVE_FILE.write_text("\n".join(deep).strip() + "\n")

    tldr = []
    tldr.append("# Evolution TLDR")
    tldr.append(
        f"- Stop reason: {evo_summary.get('stop_reason')} after {evo_summary.get('generations_executed')} generations."
    )
    if health_lines:
        tldr.append("- Robustness/validity: " + "; ".join(health_lines[:3]) + ".")
    if partial_line:
        tldr.append(f"- Partial run report: {partial_line}.")
    if top_clusters:
        best_train = top_clusters[0]
        tldr.append(
            f"- Best train robust cluster: {best_train.get('cluster_id')} "
            f"(score {_float_or(best_train.get('score')):+.2f}, Sh {_float_or(best_train.get('train_sharpe')):+.2f})."
        )
    if test_rows:
        heldout_report = _build_heldout_report(test_rows)
        leader = heldout_report.get("leader")
        winner = heldout_report.get("winner")
        best_det = heldout_report.get("best_deterministic")
        best_evo = heldout_report.get("best_evolution")
        evo_edge = None if not (best_det and best_evo) else float(best_evo["test_score"] - best_det["test_score"])
        econ_success = bool(
            best_evo
            and best_evo["test_sharpe"] >= ECONOMIC_SHARPE_FLOOR
            and (best_det is None or evo_edge >= ECONOMIC_EDGE_OVER_DETERMINISTIC)
        )
        if winner:
            tldr.append(
                f"- Held-out winner: {winner['cluster_id']} (test score {winner['test_score']:+.2f}, "
                f"test Sharpe {winner['test_sharpe']:+.2f}, edge vs det {heldout_report.get('score_edge_vs_deterministic', 0.0):+.2f})."
            )
        elif leader:
            tldr.append(
                f"- Held-out verdict: no winner yet. Leader is {leader['cluster_id']} "
                f"(score {leader['test_score']:+.2f}, test Sharpe {leader['test_sharpe']:+.2f})."
            )
            tldr.extend([f"- Rule gap: {reason}." for reason in heldout_report.get("reasons", [])])
        else:
            tldr.append("- Held-out verdict: no winner yet.")
        tldr.append(f"- Economic success: {econ_success}.")
        tldr.append(
            f"- Recommended deployment set: top {min(ENSEMBLE_TOP_N, len(test_rows))} held-out variants "
            f"(saved in best_signal_ensemble.py)."
        )
    EVOLUTION_TLDR_FILE.write_text("\n".join(tldr).strip() + "\n")


def evaluate_row_on_test(row):
    short_span, long_span = row.get("short_span", 54), row.get("long_span", 90)
    flipped = bool(row.get("flipped"))
    aux = normalize_aux_params(row.get("params", {}))
    mutation_type = row.get("mutation_type", "plain")
    code_override = row.get("code")
    raw_code = code_override or deterministic_code(row)
    fn = None if code_override else deterministic_signal_factory(short_span, long_span, variant=mutation_type, aux=aux)

    def _run_panel_eval(panel_close, panel_volume, panel_vix, panel_tnx, timeout=25):
        if code_override:
            panel_sig, panel_err = run_signal_code(
                code_override,
                panel_close,
                panel_volume,
                vix_s=panel_vix,
                tnx_s=panel_tnx,
                timeout=timeout,
            )
            if panel_err:
                raise RuntimeError(f"heldout eval code error: {panel_err}")
        else:
            panel_sig = fn(panel_close, panel_volume, panel_vix, panel_tnx)
        if flipped:
            panel_sig = -panel_sig
        panel_metrics = backtest(panel_sig, panel_close)
        return panel_sig, panel_metrics, _oriented_metrics(panel_metrics, flipped=False)

    sig, m, oriented = _run_panel_eval(close_test, volume_test, vix_test, tnx_test, timeout=25)
    beta = oriented["beta"]
    consistency = oriented["consistency"]
    test_sharpe = oriented["sharpe"]
    test_ret = oriented["ann_return"]
    test_dd = oriented["max_dd"]
    test_raw = m.get("sharpe_raw", m.get("sharpe", 0.0))
    bench_spread_sh = m.get("benchmark_spread_sharpe", m.get("excess_sharpe", 0.0))
    wf = []
    n = len(close_test)
    sz = max(n // WF_WINDOWS, 1)
    for w in range(WF_WINDOWS):
        lo = w * sz
        hi = (w + 1) * sz if w < WF_WINDOWS - 1 else n
        try:
            _, sub_m, sub_oriented = _run_panel_eval(
                close_test.iloc[lo:hi],
                volume_test.iloc[lo:hi],
                vix_test.iloc[lo:hi] if vix_test is not None else None,
                tnx_test.iloc[lo:hi] if tnx_test is not None else None,
                timeout=20,
            )
        except Exception:
            continue
        wf.append({"window": w, "sharpe": sub_oriented.get("sharpe", sub_m.get("sharpe", 0.0))})
    wf_median, wf_min = wf_summary(wf)
    test_score = _selection_score_safe(test_sharpe, wf_median, consistency, beta, m.get("avg_turnover", 0.0), wf_min=wf_min, max_dd=test_dd)
    test_metric_payload = {
        "train_sharpe": test_sharpe,
        "wf_median": wf_median,
        "wf_min": wf_min,
        "beta": beta,
        "turnover": m.get("avg_turnover", 0.0),
        "consistency": consistency,
        "signal_activity": m.get("signal_activity", 1.0),
        "raw_cs_std": m.get("raw_cs_std", 1.0),
        "raw_long_frac": m.get("raw_long_frac", 1.0),
        "raw_short_frac": m.get("raw_short_frac", 1.0),
    }
    test_robustness = _robustness_score_safe(test_metric_payload)
    cost_stress = {}
    for cost_bps in COST_STRESS_BPS:
        m_cost = backtest(sig, close_test, cost_bps=cost_bps)
        cost_oriented = _oriented_metrics(m_cost, flipped=False)
        cost_stress[f"{int(cost_bps)}bps"] = {
            "sharpe": cost_oriented["sharpe"],
            "ann_return": cost_oriented["ann_return"],
            "max_dd": cost_oriented["max_dd"],
        }

    close_full, volume_full, vix_full, tnx_full = _full_test_panels()
    subperiods = []
    for label, start, end in SUBPERIOD_WINDOWS:
        sub_close = close_full.loc[start:end]
        if len(sub_close) < 60:
            continue
        sub_volume = volume_full.reindex(index=sub_close.index, columns=sub_close.columns)
        sub_vix = vix_full.loc[start:end] if vix_full is not None else None
        sub_tnx = tnx_full.loc[start:end] if tnx_full is not None else None
        try:
            _, sub_m, sub_oriented = _run_panel_eval(sub_close, sub_volume, sub_vix, sub_tnx, timeout=25)
        except Exception:
            continue
        subperiods.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "sharpe": sub_oriented["sharpe"],
                "ann_return": sub_oriented["ann_return"],
                "max_dd": sub_oriented["max_dd"],
                "beta": sub_oriented["beta"],
                "turnover": sub_m["avg_turnover"],
            }
        )

    chronological_holdout = []
    for spec in _build_chronological_window_specs(close_test.index, CHRONOLOGICAL_HOLDOUT_SEGMENTS, min_points=60):
        try:
            _, sub_m, sub_oriented = _run_panel_eval(
                close_test.iloc[spec["lo"]:spec["hi"]],
                volume_test.iloc[spec["lo"]:spec["hi"]],
                vix_test.iloc[spec["lo"]:spec["hi"]] if vix_test is not None else None,
                tnx_test.iloc[spec["lo"]:spec["hi"]] if tnx_test is not None else None,
                timeout=20,
            )
        except Exception:
            continue
        chronological_holdout.append(
            {
                "label": spec["label"],
                "start": spec["start"],
                "end": spec["end"],
                "sharpe": sub_oriented["sharpe"],
                "ann_return": sub_oriented["ann_return"],
                "max_dd": sub_oriented["max_dd"],
                "beta": sub_oriented["beta"],
                "turnover": sub_m["avg_turnover"],
                "benchmark_spread_sharpe": sub_m.get("benchmark_spread_sharpe", sub_m.get("excess_sharpe", 0.0)),
            }
        )

    rolling_holdout = []
    for spec in _build_rolling_window_specs(close_test.index, ROLLING_HOLDOUT_SEGMENTS, min_points=ROLLING_HOLDOUT_MIN_POINTS):
        try:
            _, sub_m, sub_oriented = _run_panel_eval(
                close_test.iloc[spec["lo"]:spec["hi"]],
                volume_test.iloc[spec["lo"]:spec["hi"]],
                vix_test.iloc[spec["lo"]:spec["hi"]] if vix_test is not None else None,
                tnx_test.iloc[spec["lo"]:spec["hi"]] if tnx_test is not None else None,
                timeout=20,
            )
        except Exception:
            continue
        rolling_holdout.append(
            {
                "label": spec["label"],
                "start": spec["start"],
                "end": spec["end"],
                "sharpe": sub_oriented["sharpe"],
                "ann_return": sub_oriented["ann_return"],
                "max_dd": sub_oriented["max_dd"],
                "beta": sub_oriented["beta"],
                "turnover": sub_m["avg_turnover"],
                "benchmark_spread_sharpe": sub_m.get("benchmark_spread_sharpe", sub_m.get("excess_sharpe", 0.0)),
            }
        )

    row_source = row.get("source", "deterministic")
    row_iter = _heldout_iter_label(row)
    parent_id = str(row.get("parent_id") or _row_identity_safe(row, default_prefix="parent"))
    hypothesis = row.get("hypothesis")
    if not hypothesis:
        if code_override:
            hypothesis = f"Program evolution {row.get('cluster_id', 'unknown')}"
        else:
            hypothesis = f"Deterministic {mutation_type} EWM {short_span}/{long_span}"

    return {
        "iter": row_iter,
        "parent_id": parent_id,
        "source": row.get("source", "deterministic"),
        "mutation_type": mutation_type,
        "short_span": short_span,
        "long_span": long_span,
        "family": row.get("family", "ewm"),
        "base_family": row.get("base_family", "ewm"),
        "signature": row.get("signature"),
        "cluster_id": row.get("cluster_id", "unknown"),
        "params": aux,
        "hypothesis": hypothesis,
        "train_score": row.get("score", 0.0),
        "test_score": test_score,
        "train_sharpe": row.get("train_sharpe", row.get("sharpe", 0.0)),
        "test_sharpe": test_sharpe,
        "test_consistency": consistency,
        "train_robustness_score": row.get("robustness_score"),
        "test_robustness_score": test_robustness,
        "test_raw": test_raw,
        "test_benchmark_spread_sharpe": bench_spread_sh,
        "test_beta": beta,
        "test_turnover": m["avg_turnover"],
        "test_ret": test_ret,
        "test_dd": test_dd,
        "cost_stress": cost_stress,
        "subperiods": subperiods,
        "chronological_holdout": chronological_holdout,
        "rolling_holdout": rolling_holdout,
        "wf_median": wf_median,
        "wf_min": wf_min,
        "wf_windows": wf,
        "equity": oriented["equity"],
        "code": _oriented_signal_code(raw_code, flipped),
        "raw_code": raw_code,
        "code_is_oriented": True,
        "flipped": flipped,
    }


def evaluate_on_test(top_k=TOP_K):
    if not RUN_HELDOUT_EVAL:
        print("held-out evaluation paused")
        return []
    scope_meta = _runtime_family_scope_lists()
    active_sources = set(scope_meta.get("active_sources", []))
    det = [r for r in _safe_load_search_results_local() if r.get("score") is not None]
    deferred_rows = [r for r in det if _row_matches_deferred_stage_family(r)]
    active_scope_rows = [r for r in det if not _row_matches_deferred_stage_family(r)]
    deterministic = [r for r in active_scope_rows if r.get("source") == "deterministic"]
    evolution = [r for r in active_scope_rows if r.get("source") == "parameter_search_evolution"]
    evolution_champions = [r for r in evolution if r.get("origin") == "deterministic_champion"]
    evolution_rest = [r for r in evolution if r.get("origin") != "deterministic_champion"]
    ignored_other = [r for r in active_scope_rows if r.get("source") not in active_sources]
    picked = []
    seen_sig = set()
    per_cluster = {}
    per_mutation = {}
    picked.extend(_pick_diverse(evolution_champions, HELDOUT_MIN_EVOLUTION_CHAMPIONS, seen_sig, per_cluster, per_mutation))
    picked.extend(_pick_diverse(evolution_rest, max(0, HELDOUT_MIN_EVOLUTION - len(picked)), seen_sig, per_cluster, per_mutation))
    picked.extend(_pick_diverse(deterministic, HELDOUT_MIN_DETERMINISTIC, seen_sig, per_cluster, per_mutation))
    remaining = max(top_k, HELDOUT_SHORTLIST_K) - len(picked)
    if remaining > 0:
        already = {id(r) for r in picked}
        rest = [r for r in active_scope_rows if id(r) not in already and r.get("source") in active_sources]
        picked.extend(_pick_diverse(rest, remaining, seen_sig, per_cluster, per_mutation))
    det_ranked = picked[:max(top_k, HELDOUT_SHORTLIST_K)]
    print("held-out execution scope:", scope_meta.get("active_scope"))
    print("held-out configured roadmap families:", ", ".join(scope_meta.get("configured", [])))
    print("held-out executed families:", ", ".join(scope_meta.get("executed", [])))
    print("held-out deferred families:", ", ".join(scope_meta.get("deferred", [])))
    if deferred_rows:
        print(f"held-out deferred rows ignored: {len(deferred_rows)}")
    if ignored_other:
        print(f"held-out non-scope rows ignored: {len(ignored_other)}")
    if det_ranked:
        print(f"held-out shortlist: {len(det_ranked)} rows (max {HELDOUT_MAX_PER_CLUSTER} per cluster)")
        source_counts = {}
        for row in det_ranked:
            src = row.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        print("shortlist sources:", ", ".join([f"{k}={v}" for k, v in sorted(source_counts.items())]))
        evaluated = []
        for r in det_ranked:
            try:
                evaluated.append(evaluate_row_on_test(r))
            except Exception as e:
                print(f"held-out eval skipped {_row_identity_safe(r)}: {e}")
        return sorted(evaluated, key=lambda r: -_float_or(r.get("test_score"), -999.0))
    return []


test_results = evaluate_on_test()
heldout_report = _build_heldout_report(test_results)
if test_results:
    print(f"\n{'iter':>10} {'train_sc':>9} {'test_sc':>8} {'train_Sh':>9} {'test_Sh':>8} {'rob':>6} {'beta':>7} {'to':>6} {'wf_med':>7} {'test_dd':>8}")
    for r in test_results:
        rob = r.get("test_robustness_score", r.get("train_robustness_score"))
        rob_text = "n/a" if rob is None else f"{_float_or(rob):.1f}"
        print(f"{str(r['iter']):>10} {r['train_score']:>+9.2f} {r['test_score']:>+8.2f} {r['train_sharpe']:>+9.2f} {r['test_sharpe']:>+8.2f} {rob_text:>6} {r['test_beta']:>+7.2f} {r['test_turnover']:>6.2f} {r['wf_median']:>+7.2f} {r['test_dd']:>+8.1%}")
    best = heldout_report.get("leader")
    approved_winner = heldout_report.get("winner")
    comparison = heldout_report.get("comparison", {})
    BEST_CODE.write_text(
        f"# objective=market_neutral_net_sharpe  heldout_status={heldout_report.get('status')}  "
        f"iter {best['iter']}  cluster={best['cluster_id']}  train_score={best['train_score']:+.2f}  "
        f"test_score={best['test_score']:+.2f}  train_Sh={best['train_sharpe']:+.2f}  test_Sh={best['test_sharpe']:+.2f}\n"
        f"# parent={best['parent_id']}  mutation={best['mutation_type']}\n"
        f"# winner_rule_edge_vs_det={_float_or(heldout_report.get('score_edge_vs_deterministic'), 0.0):+.2f}  "
        f"diagnostics={comparison.get('wins', 0)}W/{comparison.get('losses', 0)}L/{comparison.get('ties', 0)}T\n"
        f"# HYPOTHESIS: {best['hypothesis']}\n\nimport numpy as np\n\n{best['code']}\n"
    )
    ensemble_rows = sorted(test_results, key=lambda r: -_float_or(r.get("test_score"), -999.0))[:min(ENSEMBLE_TOP_N, len(test_results))]
    ensemble_file = OUT / "best_signal_ensemble.py"
    ensemble_file.write_text(_ensemble_code(ensemble_rows))
    print(f"\nheld-out leader (by composite score): iter {best['iter']}  saved to {BEST_CODE.name}")
    if approved_winner:
        print(
            "approved winner: "
            f"iter {approved_winner['iter']} edge_vs_det={_float_or(heldout_report.get('score_edge_vs_deterministic'), 0.0):+.2f} "
            f"diagnostics={comparison.get('wins', 0)}W/{comparison.get('losses', 0)}L/{comparison.get('ties', 0)}T"
        )
    else:
        print("approved winner: no winner yet")
        for reason in heldout_report.get("reasons", []):
            print("  rule gap:", reason)
    print(f"ensemble ({len(ensemble_rows)} members) saved to {ensemble_file.name}")
else:
    print("held-out evaluation skipped or no surviving candidates")

_write_evolution_analysis(test_results)
