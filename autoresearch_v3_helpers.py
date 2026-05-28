"""
AutoResearch V3 - Extracted Helpers (Production + Testable)

These functions contain the real logic for:
- Precheck / executable panel validation
- Exploration scoring (v1)
- Canonical run state classification (exactly per the V3 plan)
- Structural signature (lightweight v1)

The notebook should delegate to these where possible.
Tests should import from here to validate behavior.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# Configuration defaults (can be overridden by notebook globals)
# =============================================================================
DEFAULTS = {
    "STAGE_B_MIN_ACTIVITY_STD": 0.01,
    "STAGE_B_MIN_CROSS_SECTIONAL_DISPERSION": 0.02,
    "STAGE_B_MAX_DIRECTIONAL_BIAS": 0.85,
    "AUTORESEARCH_LLM_GATE_MODE": "staged",
}


def get_config(name: str, default: Any = None) -> Any:
    """Get config preferring notebook globals, then module defaults."""
    try:
        import __main__
        if hasattr(__main__, name):
            return getattr(__main__, name)
    except Exception:
        pass
    return DEFAULTS.get(name, default)


# =============================================================================
# 1. Precheck Candidate (real implementation)
# =============================================================================
def precheck_candidate(
    code_str: str,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    vix_s: Optional[pd.Series] = None,
    tnx_s: Optional[pd.Series] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Lightweight executable panel validation before full backtest.
    Returns stable dict with failure_class for repair + logging.
    """
    result: Dict[str, Any] = {
        "is_executable": False,
        "failure_class": None,
        "failure_detail": None,
        "output_type": None,
        "output_shape": None,
        "index_match": False,
        "columns_match": False,
        "finite_ok": False,
        "activity_stats": {
            "panel_std": None,
            "cross_sectional_dispersion": None,
            "mean_abs_exposure": None,
        },
    }

    # We assume run_signal_code exists in the caller's scope when used from notebook.
    # For tests we will mock or provide a minimal version.
    try:
        # This will only work when called from inside the notebook context
        from __main__ import run_signal_code  # type: ignore
    except Exception:
        result["failure_class"] = "runtime_error"
        result["failure_detail"] = "run_signal_code not available in this context"
        return result

    try:
        sig, err = run_signal_code(
            code_str, close_df, volume_df,
            vix_s=vix_s, tnx_s=tnx_s, timeout=timeout
        )
    except Exception as e:
        result["failure_class"] = "runtime_error"
        result["failure_detail"] = str(e)[:400]
        return result

    if err:
        err_lower = str(err).lower()
        if "shape" in err_lower or "broadcast" in err_lower or "dimension" in err_lower:
            result["failure_class"] = "shape_mismatch"
        elif "boolean" in err_lower and ("subtract" in err_lower or "arithmetic" in err_lower or "type" in err_lower):
            result["failure_class"] = "boolean_arithmetic"
        else:
            result["failure_class"] = "runtime_error"
        result["failure_detail"] = str(err)[:400]
        return result

    if not isinstance(sig, pd.DataFrame):
        result["failure_class"] = "type_mismatch"
        result["failure_detail"] = f"Expected DataFrame, got {type(sig).__name__}"
        result["output_type"] = str(type(sig))
        return result

    result["output_type"] = "DataFrame"
    result["output_shape"] = sig.shape
    result["index_match"] = list(sig.index) == list(close_df.index)
    result["columns_match"] = list(sig.columns) == list(close_df.columns)

    if not (result["index_match"] and result["columns_match"]):
        result["failure_class"] = "shape_mismatch"
        result["failure_detail"] = f"Index/columns mismatch. Got {sig.shape}, expected {close_df.shape}"
        return result

    vals = sig.values
    if not np.isfinite(vals).all():
        result["failure_class"] = "non_finite"
        result["failure_detail"] = "Non-finite values detected"
        return result
    result["finite_ok"] = True

    # Activity / degeneracy
    panel_std = float(np.std(vals))
    cross_disp = float(np.mean(np.std(vals, axis=1)))
    mean_abs = float(np.mean(np.abs(vals)))

    result["activity_stats"] = {
        "panel_std": panel_std,
        "cross_sectional_dispersion": cross_disp,
        "mean_abs_exposure": mean_abs,
    }

    min_std = get_config("STAGE_B_MIN_ACTIVITY_STD", 0.01)
    min_disp = get_config("STAGE_B_MIN_CROSS_SECTIONAL_DISPERSION", 0.02)
    max_bias = get_config("STAGE_B_MAX_DIRECTIONAL_BIAS", 0.85)

    if panel_std < min_std or cross_disp < min_disp:
        result["failure_class"] = "degenerate"
        result["failure_detail"] = f"Low activity (std={panel_std:.4f}, disp={cross_disp:.4f})"
        return result

    mean_sign = float(np.sign(vals).mean())
    if abs(mean_sign) > max_bias:
        result["failure_class"] = "directional_bias"
        result["failure_detail"] = f"Strong directional bias: {mean_sign:.2f}"
        return result

    result["is_executable"] = True
    return result


# =============================================================================
# 2. Exploration Scoring (v1 - cheap, tunable)
# =============================================================================
def compute_exploration_score(
    train_sharpe: float,
    consistency: float = 0.5,
    min_annual: float = 0.0,
    turnover: float = 0.15,
    beta: float = 0.0,
    activity_penalty: float = 0.0,
) -> float:
    """V1 cheap exploration ranking score. Coefficients are intentionally tunable."""
    score = (
        np.clip(train_sharpe, -1.5, 1.5)
        + 0.35 * np.clip(consistency - 0.5, -0.5, 0.5)
        + 0.20 * np.clip(min_annual, -1.0, 1.0)
        - 1.25 * max(turnover - 0.20, 0.0)
        - 0.75 * max(abs(beta) - 0.15, 0.0)
        - 0.50 * activity_penalty
    )
    return float(score)


# =============================================================================
# 3. Canonical V3 Run State Classification (exact contract)
# =============================================================================
def classify_v3_run_state(
    exploration_admissible_count: int,
    heldout_result_count: int,
    winner_rule_passed: bool = False,
    explicit_heldout_skip: bool = False,
) -> Dict[str, Any]:
    """
    Exact state machine per AutoResearch Remediation V3 plan.

    exploration_succeeded_but_heldout_skipped = ONLY for explicit config/debug skips.
    Normal failure to produce any heldout_eligible candidates MUST be
    exploration_failed_before_heldout.
    """
    if exploration_admissible_count <= 0:
        return {
            "exploration_status": "exploration_failed_before_heldout",
            "heldout_status": "not_reached",
            "benchmark_admissible": False,
        }

    if explicit_heldout_skip:
        return {
            "exploration_status": "exploration_succeeded_but_heldout_skipped",
            "heldout_status": "skipped",
            "benchmark_admissible": False,
        }

    if heldout_result_count <= 0:
        # Normal case: we had some exploration success but nothing made it to held-out
        return {
            "exploration_status": "exploration_succeeded",
            "heldout_status": "not_reached",
            "benchmark_admissible": False,
        }

    if winner_rule_passed:
        return {
            "exploration_status": "exploration_succeeded",
            "heldout_status": "approved_winner",
            "benchmark_admissible": True,
        }

    return {
        "exploration_status": "exploration_succeeded",
        "heldout_status": "heldout_evaluated_no_winner",
        "benchmark_admissible": False,
    }


# =============================================================================
# 4. Lightweight Structural Signature (V3 narrow version)
# =============================================================================
def derive_structural_signature(code: str, metrics: Optional[Dict] = None) -> str:
    """
    Very lightweight structural signature for V1 diversity tracking.
    """
    code_lower = (code or "").lower()
    sig_parts = []

    if "vix" in code_lower:
        sig_parts.append("vix")
    if "tnx" in code_lower:
        sig_parts.append("tnx")

    if "pct_change" in code_lower or "momentum" in code_lower:
        sig_parts.append("momentum")
    elif "rolling" in code_lower and ("mean" in code_lower or "std" in code_lower):
        sig_parts.append("reversal_or_vol")

    metrics = metrics or {}
    turnover = metrics.get("turnover", metrics.get("avg_turnover", 0.15))
    if turnover > 0.25:
        sig_parts.append("high_turn")
    elif turnover < 0.08:
        sig_parts.append("low_turn")
    else:
        sig_parts.append("mid_turn")

    return "|".join(sorted(set(sig_parts))) or "generic"


# =============================================================================
# 5. Convenience: Decide if candidate should get expensive evaluation
# =============================================================================
def should_run_full_backtest(
    precheck_result: Dict[str, Any],
    exploration_tier: str = "weak",
    gate_mode: Optional[str] = None,
) -> bool:
    """
    Core behavioral gate: should this candidate get full backtest + reflection?
    This is the most important control for keeping the loop from dying on junk.
    """
    if gate_mode is None:
        gate_mode = get_config("AUTORESEARCH_LLM_GATE_MODE", "staged")

    if not precheck_result.get("is_executable", False):
        return False

    if gate_mode == "strict":
        # Old behavior - let the existing viability code decide
        return True

    # Staged mode: only promising tier gets full expensive evaluation
    return exploration_tier == "promising"
