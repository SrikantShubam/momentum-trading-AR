"""Build autoresearch_v2.ipynb - AutoResearch v2 for Kaggle T4x2.

Current guarded stage:
  - deterministic and classical risk-managed momentum baselines
  - program-evolution candidates evaluated against the same data
  - explicit runtime reporting for model loading, LLM calls, and deferred methods
  - annual consistency scoring to reduce single-regime overfit
  - thread-based sandbox timeout
"""
import json
from pathlib import Path
import nbformat
from nbformat.validator import normalize

CELLS = []

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": text})


# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
md("""# AutoResearch v2 - Momentum Alpha Discovery (T4x2)

This guarded stage evaluates deterministic/classical momentum baselines plus
program-evolution candidates on 100 US equities. If an LLM stage is enabled,
the report records loaded model id, execution state, and call count explicitly.

**Key differences from v1:**
- Model: Qwen2.5-Coder-32B-Instruct can run in 4-bit, with 14B fallback for T4x2 stability
- Universe: ~100 liquid US equities (vs 20)
- Reflection: reported only when recorded LLM/reflection calls actually execute
- Scoring: annual consistency (not just aggregate Sharpe)

**Kaggle setup:** Accelerator = **GPU T4 x2**, Internet = **On**
""")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install
# ═════════════════════════════════════════════════════════════════════════════
md("## Cell 1 — Install dependencies")
code('!pip install -q yfinance bitsandbytes "transformers>=4.44" "accelerate>=0.33"')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports + Configuration
# ═════════════════════════════════════════════════════════════════════════════
md("## Cell 2 — Imports, configuration, paths")
code('''import os, sys, json, re, time, random, gc, ast, traceback, hashlib
import threading as _th
import inspect as _inspect
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Reproducibility ──
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def _env_truthy(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _read_dotenv_value(path, *names):
    try:
        p = Path(path)
        if not p.exists():
            return None, "none"
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in names:
                continue
            value = value.strip().strip('"').strip("'")
            if value:
                return value, f"dotenv:{key}"
    except Exception:
        pass
    return None, "none"

def _resolve_kaggle_secret(*names):
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in names:
            try:
                value = client.get_secret(name)
            except Exception:
                value = None
            if value:
                return value, f"kaggle:{name}"
    except Exception:
        pass
    return None, "none"

def _resolve_token(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value, f"env:{name}"
    value, source = _resolve_kaggle_secret(*names)
    if value:
        return value, source
    value, source = _read_dotenv_value(".env", *names)
    if value:
        return value, source
    return None, "none"

def token_presence_text(name, value, source):
    state = "present" if value else "missing"
    return f"{name}: {state} (source={source})"

HF_TOKEN, HF_TOKEN_SOURCE = _resolve_token(
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
)

# ── Paths ──
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_PROFILE = (os.getenv("AUTORESEARCH_RUN_PROFILE", "llm_research").strip()
               or "llm_research")
BENCHMARK_MODE = _env_truthy(os.getenv("AUTORESEARCH_BENCHMARK_MODE"), default=False)
ROADMAP_METHOD_FAMILIES = [
    "deterministic",
    "classical_risk_managed",
    "autoresearch_evolution",
    "lstm_sharpe",
    "tabular_ml",
    "gbt",
    "combination_studies",
]
EXECUTED_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution"]
DEFERRED_METHOD_FAMILIES = [
    family for family in ROADMAP_METHOD_FAMILIES
    if family not in EXECUTED_METHOD_FAMILIES
]
ACTIVE_RESULT_SOURCES = ["deterministic", "llm_autoresearch", "parameter_search_evolution"]
ACTIVE_EXECUTION_SCOPE = "deterministic/classical/strict-llm autoresearch only"
FAMILY_EXECUTION_STATUS = {
    family: ("executed" if family in EXECUTED_METHOD_FAMILIES else "deferred")
    for family in ROADMAP_METHOD_FAMILIES
}
REPORT_SCOPE = f"{RUN_PROFILE}_{RUN_ID}"

def scoped_output(name, include_scope=True):
    stem = f"{REPORT_SCOPE}__{name}" if include_scope else name
    return OUT / stem

RESEARCH_LOG = scoped_output("research_log.jsonl")
PRICES_CACHE = OUT / "prices.parquet"
BEST_CODE    = scoped_output("best_signal.py")
SHARPE_PLOT  = scoped_output("sharpe_progress.png")
EQUITY_PLOT  = scoped_output("equity_curves.png")
SUMMARY_MD   = scoped_output("experiment_summary.md")
MEMO_FILE    = scoped_output("research_memo.txt")
RUNTIME_METADATA_FILE = scoped_output("runtime_metadata.json")
RUN_MANIFEST_FILE = scoped_output("run_manifest.json")
LATEST_RUNTIME_METADATA_FILE = OUT / "latest_runtime_metadata.json"
LATEST_RUN_MANIFEST_FILE = OUT / "latest_run_manifest.json"

# ── Model ──
# Qwen2.5-Coder-32B at Q4 ≈ 18 GB → fits T4×2 (32 GB combined).
# For faster runs, uncomment the 14B line below.
MODEL_VARIANT = (os.getenv("AUTORESEARCH_MODEL_VARIANT", "14b").strip().lower() or "14b")
CONFIGURED_MODEL_ID = (
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    if MODEL_VARIANT in {"32b", "32", "large"}
    else "Qwen/Qwen2.5-Coder-14B-Instruct"
)
MODEL_ID = CONFIGURED_MODEL_ID
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
CONFIGURED_FALLBACK_MODEL_ID = FALLBACK_MODEL_ID
ACTIVE_MODEL_ID = None

# ── Loop config ──
N_BATCHES       = 25          # each batch → 1 LLM call → up to 3 candidates
REFLECT_EVERY   = 2           # reflection memo every N batches after first trigger
FIRST_REFLECTION_BATCH = int(os.getenv("AUTORESEARCH_FIRST_REFLECTION_BATCH", "2"))
SANDBOX_TIMEOUT = 30          # seconds per signal execution
TRAIN_END       = "2022-12-31"
LLM_PROMPT_MAX_TOKENS = int(os.getenv("AUTORESEARCH_PROMPT_MAX_TOKENS", "12000"))
LLM_BATCH_MAX_NEW_TOKENS = int(os.getenv("AUTORESEARCH_BATCH_MAX_NEW_TOKENS", "1100"))
LLM_REFLECTION_MAX_NEW_TOKENS = int(os.getenv("AUTORESEARCH_REFLECTION_MAX_NEW_TOKENS", "700"))
LLM_REPAIR_MAX_NEW_TOKENS = int(os.getenv("AUTORESEARCH_REPAIR_MAX_NEW_TOKENS", "900"))
EARLY_ABORT_NEGATIVE_SUCCESSFUL = int(os.getenv("AUTORESEARCH_EARLY_ABORT_NEGATIVE_SUCCESSFUL", "9"))
EARLY_ABORT_NEGATIVE_SHARPE = float(os.getenv("AUTORESEARCH_EARLY_ABORT_NEGATIVE_SHARPE", "-0.50"))
EARLY_ABORT_NEGATIVE_SCORE = float(os.getenv("AUTORESEARCH_EARLY_ABORT_NEGATIVE_SCORE", "-1.00"))
EARLY_ABORT_MIN_BATCH = int(os.getenv("AUTORESEARCH_EARLY_ABORT_MIN_BATCH", "2"))
EARLY_ABORT_MIN_REFLECTIONS = int(os.getenv("AUTORESEARCH_EARLY_ABORT_MIN_REFLECTIONS", "2"))
SEARCH_COLLAPSE_MIN_BATCHES = int(os.getenv("AUTORESEARCH_SEARCH_COLLAPSE_MIN_BATCHES", "3"))
SEARCH_COLLAPSE_FAILURE_RATE = float(os.getenv("AUTORESEARCH_SEARCH_COLLAPSE_FAILURE_RATE", "0.67"))
SEARCH_COLLAPSE_MIN_ERRORS = int(os.getenv("AUTORESEARCH_SEARCH_COLLAPSE_MIN_ERRORS", "6"))
MIN_VIABLE_TRAIN_SHARPE = float(os.getenv("AUTORESEARCH_MIN_VIABLE_TRAIN_SHARPE", "0.00"))
MIN_VIABLE_RESEARCH_SCORE = float(os.getenv("AUTORESEARCH_MIN_VIABLE_RESEARCH_SCORE", "0.00"))
OBJECTIVE_MODE = (
    os.getenv(
        "AUTORESEARCH_OBJECTIVE_MODE",
        "market_neutral" if RUN_PROFILE == "llm_research" and not BENCHMARK_MODE else "benchmark_relative",
    ).strip().lower() or "market_neutral"
)
if OBJECTIVE_MODE not in {"market_neutral", "benchmark_relative"}:
    OBJECTIVE_MODE = "market_neutral"
PRIMARY_OBJECTIVE_LABEL = (
    "market-neutral portfolio Sharpe"
    if OBJECTIVE_MODE == "market_neutral"
    else "benchmark-relative active Sharpe"
)
DEFAULT_LLM_RESEARCH_MODE = (RUN_PROFILE == "llm_research" and not BENCHMARK_MODE)
RUN_LLM_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=DEFAULT_LLM_RESEARCH_MODE)
RUN_MOE_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_MOE_STAGE"), default=False)
RUN_LLM_SMOKE = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)
RUN_BNB_MODEL_LOAD = _env_truthy(
    os.getenv("AUTORESEARCH_ENABLE_BNB_LOAD"),
    default=(DEFAULT_LLM_RESEARCH_MODE or RUN_LLM_SMOKE),
)
RUN_PARAM_SEARCH = _env_truthy(os.getenv("AUTORESEARCH_RUN_PARAM_SEARCH"), default=False)
RUN_HELDOUT_EVAL = True
RUN_REPORTS = True
REFLECTION_ENABLED = bool(RUN_LLM_STAGE and REFLECT_EVERY > 0)

# Structured parameter search constants must exist even when the stage is off:
# metric_cell_18 defines function defaults from these names at cell execution.
PARAM_SEARCH_TRIALS = 200
PARAM_SEARCH_SEED = SEED
PARAM_SEARCH_MODE = "random"
PARAM_SEARCH_SHORT_SPANS = [36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66]
PARAM_SEARCH_LONG_SPANS = [90, 95, 100, 105, 110, 120, 130, 140, 150]
PARAM_SEARCH_REGIME_THRESHOLDS = [18.0, 20.0, 22.0, 24.0, 28.0]
PARAM_SEARCH_VOL_WINDOWS = [10, 20, 30]
PARAM_SEARCH_VOL_GATE_WINDOWS = [10, 20, 40]

if DEFAULT_LLM_RESEARCH_MODE and not HF_TOKEN:
    print(
        "HF_TOKEN not found; attempting anonymous Hugging Face download. "
        "Add HF_TOKEN/HUGGINGFACEHUB_API_TOKEN only if the selected model requires auth."
    )

RUNTIME_STATE = {
    "run_id": RUN_ID,
    "run_profile": RUN_PROFILE,
    "report_scope": REPORT_SCOPE,
    "benchmark_mode": BENCHMARK_MODE,
    "configured_method_families": list(ROADMAP_METHOD_FAMILIES),
    "executed_method_families": list(EXECUTED_METHOD_FAMILIES),
    "deferred_method_families": list(DEFERRED_METHOD_FAMILIES),
    "family_execution_status": dict(FAMILY_EXECUTION_STATUS),
    "active_result_sources": list(ACTIVE_RESULT_SOURCES),
    "active_execution_scope": ACTIVE_EXECUTION_SCOPE,
    "configured_model_id": CONFIGURED_MODEL_ID,
    "configured_fallback_model_id": CONFIGURED_FALLBACK_MODEL_ID,
    "objective_mode": OBJECTIVE_MODE,
    "primary_objective_label": PRIMARY_OBJECTIVE_LABEL,
    "actual_model_id": None,
    "llm_stage_enabled": bool(RUN_LLM_STAGE),
    "moe_stage_enabled": bool(RUN_MOE_STAGE),
    "llm_smoke_enabled": bool(RUN_LLM_SMOKE),
    "llm_stage_loaded": False,
    "bnb_model_load_enabled": bool(RUN_BNB_MODEL_LOAD),
    "llm_stage_executed": False,
    "llm_calls": 0,
    "reflection_enabled": REFLECTION_ENABLED,
    "reflection_calls": 0,
    "structured_candidate_count": 0,
    "repair_batches": 0,
    "duplicate_rejections": 0,
    "semantic_family_rejections": 0,
    "format_failures": 0,
    "token_status": {
        "hf": {"present": bool(HF_TOKEN), "source": HF_TOKEN_SOURCE},
    },
}

ARTIFACT_PATHS = {
    "research_log": str(RESEARCH_LOG),
    "prices_cache": str(PRICES_CACHE),
    "best_code": str(BEST_CODE),
    "sharpe_plot": str(SHARPE_PLOT),
    "equity_plot": str(EQUITY_PLOT),
    "summary": str(SUMMARY_MD),
    "memo": str(MEMO_FILE),
    "runtime_metadata": str(RUNTIME_METADATA_FILE),
    "run_manifest": str(RUN_MANIFEST_FILE),
}

def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value

def sync_runtime_metadata(**updates):
    if updates:
        RUNTIME_STATE.update(updates)
    payload = dict(RUNTIME_STATE)
    payload["artifact_paths"] = dict(ARTIFACT_PATHS)
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    text = json.dumps(_json_safe(payload), indent=2)
    RUNTIME_METADATA_FILE.write_text(text)
    LATEST_RUNTIME_METADATA_FILE.write_text(text)
    manifest = {
        "run_id": RUN_ID,
        "report_scope": REPORT_SCOPE,
        "summary_file": str(SUMMARY_MD),
        "runtime_metadata_file": str(RUNTIME_METADATA_FILE),
        "research_log_file": str(RESEARCH_LOG),
    }
    manifest_text = json.dumps(manifest, indent=2)
    RUN_MANIFEST_FILE.write_text(manifest_text)
    LATEST_RUN_MANIFEST_FILE.write_text(manifest_text)
    return payload

def update_runtime_state(**updates):
    return sync_runtime_metadata(**updates)

def runtime_stage_summary():
    return (
        f"llm_enabled={RUN_LLM_STAGE} | moe_enabled={RUN_MOE_STAGE} | "
        f"benchmark_mode={BENCHMARK_MODE} | reflection_enabled={REFLECTION_ENABLED}"
    )

def runtime_family_scope_summary():
    return (
        f"scope={ACTIVE_EXECUTION_SCOPE} | "
        f"executed={','.join(EXECUTED_METHOD_FAMILIES)} | "
        f"deferred={','.join(DEFERRED_METHOD_FAMILIES)}"
    )

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  {mem:.1f} GB")
print(token_presence_text("HF_TOKEN", HF_TOKEN, HF_TOKEN_SOURCE))
print("runtime profile:", RUN_PROFILE, "| report scope:", REPORT_SCOPE)
print("configured model:", CONFIGURED_MODEL_ID,
      "| fallback:", CONFIGURED_FALLBACK_MODEL_ID)
print("stage flags:", runtime_stage_summary())
print("roadmap families:", ", ".join(ROADMAP_METHOD_FAMILIES))
print("family scope:", runtime_family_scope_summary())
sync_runtime_metadata()
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 3 — Data
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 3 — Download and cache price data

~100 liquid US equities, 2015–2024 via yfinance (free).
Tickers with <70 % data coverage are dropped automatically.
Cached to parquet after first download.
Split: 2015–2022 = train, 2023–2024 = held-out test.
""")
code('''TICKERS = [
    # ── Tech / Semis ──
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO",
    "ORCL","CRM","ADBE","CSCO","ACN","INTC","AMD","QCOM",
    "TXN","AMAT","MU","LRCX","ADI","IBM","INTU",
    # ── Financials ──
    "JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP","C",
    "SPGI","CME","SCHW",
    # ── Healthcare ──
    "UNH","JNJ","LLY","PFE","ABT","TMO","MRK","ABBV",
    "DHR","AMGN","MDT","BMY","GILD","ISRG","VRTX","REGN",
    # ── Consumer ──
    "WMT","PG","KO","PEP","COST","MCD","NKE","SBUX",
    "TGT","CL","MDLZ","EL","PM","MO",
    # ── Industrials ──
    "CAT","HON","UPS","BA","GE","RTX","LMT","DE",
    "MMM","UNP","FDX","EMR","ETN",
    # ── Energy ──
    "XOM","CVX","COP","SLB","EOG","PSX",
    # ── Communication ──
    "DIS","NFLX","CMCSA","T","VZ",
    # ── Materials ──
    "LIN","APD","ECL","SHW","FCX","NEM",
    # ── Utilities ──
    "NEE","DUK","SO","D",
    # ── Real Estate ──
    "PLD","AMT","CCI","SPG","EQIX",
    # ── Retail / Other ──
    "HD","LOW","TJX","ROST","PYPL",
]

START = "2015-01-01"
END   = "2024-12-31"

def load_prices():
    if PRICES_CACHE.exists():
        df = pd.read_parquet(PRICES_CACHE)
        cols = df.columns.get_level_values(1).unique()
        print(f"cached: {len(cols)} tickers, "
              f"{df.index.min().date()} \u2192 {df.index.max().date()}")
        return df
    print(f"downloading {len(TICKERS)} tickers from yfinance ...")
    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(TICKERS, start=START, end=END,
                              auto_adjust=True, progress=True, threads=True)
            if raw.empty:
                raise ValueError("empty dataframe")
            break
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError(f"yfinance download failed after 3 attempts: {last_err}")

    close = raw["Close"].dropna(how="all")

    # keep tickers with >70 % data coverage
    coverage = close.notna().mean()
    good = sorted(coverage[coverage > 0.7].index.tolist())
    close = close[good].ffill().bfill()
    # drop any remaining all-NaN columns
    close = close.dropna(axis=1, how="any")
    volume = raw["Volume"].reindex(columns=close.columns,
                                    index=close.index).fillna(0)
    df = pd.concat({"close": close, "volume": volume}, axis=1)
    df.to_parquet(PRICES_CACHE)
    print(f"saved: {close.shape[1]} tickers, {close.shape[0]} days")
    return df

prices   = load_prices()
close_all  = prices["close"].astype(float)
volume_all = prices["volume"].astype(float)

mask = close_all.index <= pd.Timestamp(TRAIN_END)
close_train,  close_test  = close_all[mask], close_all[~mask]
volume_train, volume_test = volume_all[mask], volume_all[~mask]

print(f"train : {close_train.shape}  ({close_train.index.min().date()} \u2192 "
      f"{close_train.index.max().date()})")
print(f"test  : {close_test.shape}   ({close_test.index.min().date()} \u2192 "
      f"{close_test.index.max().date()})")
print(f"universe: {len(close_all.columns)} tickers")

# Regime data: VIX + 10Y yield aligned to the equity trading calendar.
REGIME_CACHE = OUT / "regime.parquet"

def load_regime_data(index):
    if REGIME_CACHE.exists():
        df = pd.read_parquet(REGIME_CACHE)
        print(f"regime cached: {df.index.min().date()} -> {df.index.max().date()}")
        return df.reindex(index).ffill().bfill()
    print("downloading ^VIX and ^TNX ...")
    frames = {}
    for sym, col in [("^VIX", "VIX"), ("^TNX", "TNX")]:
        raw = yf.download(sym, start=START, end=END,
                          auto_adjust=True, progress=False)
        close_col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        frames[col] = close_col.squeeze()
    df = pd.DataFrame(frames).reindex(index).ffill().bfill()
    df.to_parquet(REGIME_CACHE)
    print(f"saved regime: {df.shape}")
    return df

regime_all = load_regime_data(close_all.index)
vix_all = regime_all["VIX"]
tnx_all = regime_all["TNX"]
vix_train, vix_test = vix_all[mask], vix_all[~mask]
tnx_train, tnx_test = tnx_all[mask], tnx_all[~mask]
print(f"VIX  train mean={vix_train.mean():.1f}  test mean={vix_test.mean():.1f}")
print(f"TNX  train mean={tnx_train.mean():.2f}%  test mean={tnx_test.mean():.2f}%")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4 — Backtester
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 4 — Backtester with annual consistency

Vectorised pandas. Positions = `signal.shift(1)` (no lookahead).
Equal-weight across assets. 5 bps per-turn transaction cost.
Reports the primary objective Sharpe plus annual consistency.
""")
code('''def backtest(signal_df, close_df, cost_bps=5.0):
    sig = (signal_df
           .reindex(close_df.index)
           .reindex(columns=close_df.columns)
           .astype(float).fillna(0).clip(-1, 1))
    pos = sig.shift(1).fillna(0)

    ret      = close_df.pct_change().fillna(0)
    gross    = (pos * ret).mean(axis=1)
    turnover = pos.diff().abs().mean(axis=1).fillna(0)
    cost     = turnover * (cost_bps / 10_000)
    net      = gross - cost

    bench   = ret.mean(axis=1)          # equal-weight long-only diagnostic
    active  = net - bench
    primary = net if OBJECTIVE_MODE == "market_neutral" else active

    eq         = (1 + net).cumprod()
    eq_active  = (1 + active).cumprod()
    eq_primary = (1 + primary).cumprod()

    ann_r   = net.mean()     * 252; ann_v   = net.std()     * np.sqrt(252)
    ann_ra  = active.mean()  * 252; ann_va  = active.std()  * np.sqrt(252)
    ann_rp  = primary.mean() * 252; ann_vp  = primary.std() * np.sqrt(252)
    sharpe      = float(ann_r  / ann_v)  if ann_v  > 0 else 0.0
    sharpe_act  = float(ann_ra / ann_va) if ann_va > 0 else 0.0
    sharpe_main = float(ann_rp / ann_vp) if ann_vp > 0 else 0.0

    cov  = np.cov(net.values, bench.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0

    # ── annual consistency ──
    annual_sh = []
    for yr in sorted(primary.index.year.unique()):
        a = primary[primary.index.year == yr]
        if len(a) < 50:
            continue
        v = a.std() * np.sqrt(252)
        annual_sh.append(float(a.mean() * 252 / v) if v > 0 else 0.0)
    consistency = (sum(1 for s in annual_sh if s > 0)
                   / max(len(annual_sh), 1))

    return {
        "sharpe_active": sharpe_main,
        "sharpe_raw":    sharpe,
        "benchmark_spread_sharpe": sharpe_act,
        "ann_active":    float(ann_rp),
        "ann_return":    float(ann_r),
        "ann_benchmark_spread": float(ann_ra),
        "beta":          beta,
        "max_dd":        float((eq / eq.cummax() - 1).min()),
        "max_dd_active": float((eq_primary / eq_primary.cummax() - 1).min()),
        "max_dd_benchmark_spread": float((eq_active / eq_active.cummax() - 1).min()),
        "hit_rate":      float((net > 0).mean()),
        "avg_turnover":  float(turnover.mean()),
        "equity":        eq,
        "equity_active": eq_active,
        "equity_primary": eq_primary,
        "annual_sharpes": annual_sh,
        "consistency":    consistency,
        "min_annual":     min(annual_sh) if annual_sh else 0.0,
        "objective_mode": OBJECTIVE_MODE,
        "primary_objective_label": PRIMARY_OBJECTIVE_LABEL,
    }

# ── sanity check: 20-day cross-sectional momentum ──
_cs = close_train.pct_change(20).rank(axis=1, pct=True) * 2 - 1
_cs = _cs.sub(_cs.mean(axis=1), axis=0)
_m  = backtest(_cs, close_train)
print(f"sanity: mode={OBJECTIVE_MODE} primarySh={_m['sharpe_active']:+.2f}  benchSpread={_m['benchmark_spread_sharpe']:+.2f}  "
      f"beta={_m['beta']:+.2f}  consistency={_m['consistency']:.0%}  "
      f"DD={_m['max_dd_active']:+.1%}")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 5 — Sandbox executor
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 5 — Sandbox executor

AST-validated, restricted builtins, thread-based timeout.
Works from any thread (main or ThreadPoolExecutor worker).
""")
code('''FORBIDDEN = [
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "import requests", "import urllib",
    "open(", "__import__", "eval(", "exec(",
    "compile(", "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "input(", "quit(", "exit(", "__builtins__", "__class__", "__bases__",
    "__subclasses__", "pathlib", "Path(",
]
ALLOWED_IMPORTS = {"numpy", "np", "pandas", "pd", "math"}

def validate_code(code_str):
    low = code_str.lower()
    for bad in FORBIDDEN:
        if bad.lower() in low:
            return False, f"forbidden: {bad!r}"
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"disallowed from-import: {node.module}"
    # Cheap static lint for row-Series/DataFrame broadcast bugs that repeatedly
    # create shape explosions in generated code.
    if re.search(r"\.mean\s*\(\s*axis\s*=\s*1\s*\)\s*\.reindex_like\s*\(", code_str):
        return False, "shape-risk: row-wise Series reindex_like(DataFrame); use .sub(..., axis=0) or .values[:, None]"
    if "def signal" not in code_str:
        return False, "no `def signal` found"
    return True, "ok"

SAFE_BUILTINS = {
    k: (__builtins__[k] if isinstance(__builtins__, dict)
        else getattr(__builtins__, k))
    for k in ["len","range","min","max","abs","sum","int","float","bool",
              "list","dict","tuple","set","str","True","False","None",
              "enumerate","zip","sorted","reversed","map","filter","round",
              "isinstance","type","any","all","print","slice"]
    if (k in __builtins__ if isinstance(__builtins__, dict)
        else hasattr(__builtins__, k))
}

def run_signal_code(code_str, close_df, volume_df, vix_s=None, tnx_s=None, timeout=SANDBOX_TIMEOUT):
    """Execute signal code in a sandboxed thread with timeout."""
    ok, msg = validate_code(code_str)
    if not ok:
        return None, f"VALIDATION: {msg}"

    ns = {"np": np, "pd": pd, "__builtins__": SAFE_BUILTINS}
    result = [None, None]          # [output, error_string]

    def _run():
        try:
            exec(code_str, ns)
            if "signal" not in ns or not callable(ns["signal"]):
                result[1] = "no callable `signal` after exec"
                return
            try:
                params = set(_inspect.signature(ns["signal"]).parameters)
            except (ValueError, TypeError):
                params = set()
            kwargs = {}
            if "vix" in params:
                kwargs["vix"] = vix_s.copy() if vix_s is not None else None
            if "tnx" in params:
                kwargs["tnx"] = tnx_s.copy() if tnx_s is not None else None
            result[0] = ns["signal"](close_df.copy(), volume_df.copy(), **kwargs)
        except Exception as e:
            result[1] = f"{type(e).__name__}: {e}"

    t = _th.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, f"TIMEOUT (>{timeout}s)"
    if result[1] is not None:
        return None, result[1]
    out = result[0]

    if not isinstance(out, pd.DataFrame):
        return None, f"returned {type(out).__name__}, need DataFrame"
    if out.shape != close_df.shape:
        return None, f"shape {out.shape} != expected {close_df.shape}"
    return out, None

def validate_signal(sig_df):
    """Reject degenerate signals (near-constant, one-sided, etc.)."""
    s = sig_df.clip(-1, 1).fillna(0)
    if s.std().mean() < 0.03:
        return False, "near-constant signal (std<0.03)"
    long_frac  = float((s >  0.05).mean().mean())
    short_frac = float((s < -0.05).mean().mean())
    if min(long_frac, short_frac) < 0.05:
        return False, f"not long-short (L={long_frac:.2f} S={short_frac:.2f})"
    if s.mean().abs().mean() > 0.5:
        return False, f"directionally biased (|mean|={s.mean().abs().mean():.2f})"
    return True, "ok"

def detect_lookahead(code_str, close_df, volume_df, vix_s=None, tnx_s=None, split_frac=0.6, seed=0):
    """Shuffle future data and check if past-slice signal changes."""
    sig_real, err = run_signal_code(code_str, close_df, volume_df, vix_s=vix_s, tnx_s=tnx_s, timeout=20)
    if err is not None:
        return False, err
    T = int(len(close_df) * split_frac)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(close_df) - T)
    c2, v2 = close_df.copy(), volume_df.copy()
    vix2 = None
    tnx2 = None
    if vix_s is not None:
        vix2 = vix_s.copy(); vix2.iloc[T:] = vix_s.iloc[T:].values[perm]
    if tnx_s is not None:
        tnx2 = tnx_s.copy(); tnx2.iloc[T:] = tnx_s.iloc[T:].values[perm]
    c2.iloc[T:] = close_df.iloc[T:].values[perm]
    v2.iloc[T:] = volume_df.iloc[T:].values[perm]
    sig_perm, err = run_signal_code(code_str, c2, v2, vix_s=vix2, tnx_s=tnx2, timeout=20)
    if err is not None:
        return False, f"shuffle_run: {err}"
    a = sig_real.iloc[:T].fillna(0).values
    b = sig_perm.iloc[:T].fillna(0).values
    diff = float(np.abs(a - b).mean())
    if diff > 1e-6:
        return False, f"LOOKAHEAD detected (diff={diff:.6f})"
    return True, "ok"

# ── self-test ──
_test = """
def signal(close, volume):
    r = close.pct_change(20).rank(axis=1, pct=True)
    out = (r - 0.5) * 2
    return out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
"""
_s, _e = run_signal_code(_test, close_train, volume_train)
print("sandbox self-test:", "OK" if _e is None else f"FAIL: {_e}")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 6 — Load model
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 6 — Load model (4-bit quantised)

The model load is lazy. If no LLM stage is enabled, this cell skips the
bitsandbytes/CUDA path entirely so deterministic research can still run.
""")
code('''REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)
SHOULD_LOAD_LLM_MODEL = bool(REQUESTED_LLM_MODEL and RUN_BNB_MODEL_LOAD)
tok = None
model = None
update_runtime_state(
    llm_stage_enabled=bool(RUN_LLM_STAGE),
    moe_stage_enabled=bool(RUN_MOE_STAGE),
    llm_smoke_enabled=bool(RUN_LLM_SMOKE),
    bnb_model_load_enabled=bool(RUN_BNB_MODEL_LOAD),
    llm_stage_loaded=False,
    llm_stage_executed=False,
    actual_model_id=None,
)

def _require_kaggle_t4_x2():
    n_gpus = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    if n_gpus != 2 or any("T4" not in name.upper() for name in names):
        update_runtime_state(
            llm_stage_error="requires_gpu_t4_x2",
            cuda_device_count=n_gpus,
            cuda_device_names=names,
        )
        raise RuntimeError(
            "This notebook must run on Kaggle GPU T4 x2. "
            f"Detected {n_gpus} CUDA device(s): {names}"
        )
    return names

if SHOULD_LOAD_LLM_MODEL:
    _require_kaggle_t4_x2()

hub_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

def _load_tokenizer(model_id):
    t = AutoTokenizer.from_pretrained(model_id, **hub_kwargs)
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    return t

def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def _load_model(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config(),
        device_map="auto",
        dtype=torch.float16,
        **hub_kwargs,
    )

def ensure_model_loaded():
    global tok, model, MODEL_ID, ACTIVE_MODEL_ID
    if tok is not None and model is not None:
        return tok, model
    if not RUN_BNB_MODEL_LOAD:
        update_runtime_state(llm_stage_error="bnb_model_load_disabled")
        raise RuntimeError(
            "LLM model loading is disabled for this Kaggle run because the current "
            "CUDA/bitsandbytes stack can hard-crash the kernel. Set "
            "AUTORESEARCH_ENABLE_BNB_LOAD=1 only for a dedicated model-load test."
        )

    active_model_id = CONFIGURED_MODEL_ID
    load_strategy = "configured_primary"
    tok = _load_tokenizer(active_model_id)
    try:
        model = _load_model(active_model_id)
    except Exception as e:
        msg = str(e).lower()
        is_oom = ("outofmemory" in msg) or ("out of memory" in msg) or isinstance(e, torch.OutOfMemoryError)
        if (not is_oom) or active_model_id == FALLBACK_MODEL_ID:
            update_runtime_state(llm_stage_error=f"{type(e).__name__}: {e}")
            raise
        print(f"Primary model OOM: {CONFIGURED_MODEL_ID}")
        print("Falling back to configured backup model for stability on current T4 session...")
        gc.collect(); torch.cuda.empty_cache()
        active_model_id = CONFIGURED_FALLBACK_MODEL_ID
        load_strategy = "oom_fallback"
        tok = _load_tokenizer(active_model_id)
        model = _load_model(active_model_id)

    model.eval()
    ACTIVE_MODEL_ID = active_model_id
    MODEL_ID = active_model_id
    update_runtime_state(
        actual_model_id=ACTIVE_MODEL_ID,
        llm_stage_loaded=True,
        llm_stage_error=None,
        llm_load_strategy=load_strategy,
    )
    print(f"model loaded: configured={CONFIGURED_MODEL_ID} actual={MODEL_ID} ({load_strategy})")
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM allocated: {vram:.1f} GB")
    return tok, model

if SHOULD_LOAD_LLM_MODEL:
    ensure_model_loaded()
    if RUN_LLM_SMOKE:
        try:
            smoke_ids = tok("LLM smoke test", return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                _ = model.generate(smoke_ids, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
            update_runtime_state(llm_smoke_executed=True, llm_smoke_ok=True)
            print("LLM smoke test: OK")
        except Exception as e:
            update_runtime_state(llm_smoke_executed=True, llm_smoke_ok=False, llm_smoke_error=f"{type(e).__name__}: {e}")
            print(f"LLM smoke test failed: {type(e).__name__}: {e}")
else:
    if REQUESTED_LLM_MODEL:
        update_runtime_state(llm_load_strategy="skipped_bnb_disabled", llm_stage_error="bnb_model_load_disabled")
        print("LLM stage requested, but bitsandbytes model load is disabled for kernel stability.")
        print("Set AUTORESEARCH_ENABLE_BNB_LOAD=1 only for a dedicated model-load smoke test.")
    else:
        update_runtime_state(llm_load_strategy="skipped_disabled")
        print("LLM stages disabled; skipping model load.")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 7 — LLM generation helper
# ═════════════════════════════════════════════════════════════════════════════
md("## Cell 7 — LLM generation helper")
code('''@torch.inference_mode()
def llm(messages, max_new_tokens=LLM_BATCH_MAX_NEW_TOKENS, temperature=0.8, top_p=0.95):
    ensure_model_loaded()
    update_runtime_state(
        llm_stage_executed=True,
        llm_calls=int(RUNTIME_STATE.get("llm_calls", 0)) + 1,
        actual_model_id=ACTIVE_MODEL_ID or MODEL_ID,
    )
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=LLM_PROMPT_MAX_TOKENS).to(model.device)
    try:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    finally:
        pass
    gen = out[0, enc["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    del gen, out, enc
    gc.collect(); torch.cuda.empty_cache()
    return text
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 8 — Prompts + extraction
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 8 — Prompts + candidate extraction

The system prompt includes a **feature catalog** so the model knows what
building blocks are available. The user prompt includes the research log,
the latest reflection memo, and recent failure examples.
""")
code('''SYSTEM_PROMPT = """You are a quantitative researcher discovering alpha signals for a long-short equity strategy on ~100 US stocks.

Given:
- `close`: pd.DataFrame of daily adjusted close prices (rows=dates, cols=tickers)
- `volume`: pd.DataFrame of daily volumes (same shape)

Write: def signal(close, volume, vix=None, tnx=None) -> pd.DataFrame
Return values in [-1, +1]. +1 = strongest long, -1 = strongest short.
Positions are lagged by 1 day automatically — no lookahead needed in your code.

Hard rules:
- Use ONLY numpy (np) and pandas (pd). No other imports.
- Return shape MUST equal close.shape.
- Keep functions under 35 lines.
- End EVERY signal with:
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)

CRITICAL — lookahead bias auto-rejects your signal:
- NEVER use full-series aggregations: close.mean(), volume.std(), close.quantile()
  These use FUTURE data. They will be detected and rejected.
- ONLY use trailing windows: .pct_change(N), .rolling(N).mean(), .rolling(N).std(),
  .ewm(span=N).mean(), .shift(N) with N > 0.
- Cross-sectional (axis=1) ops on a single date are fine:
  .rank(axis=1), .mean(axis=1), .std(axis=1)

CRITICAL — signal must be long-short:
- Both positive AND negative values on most days.
- A near-constant signal will be rejected.
- Pattern: r = ...rank(axis=1, pct=True); out = (r - 0.5) * 2

Feature catalog (all require explicit lookback N):
  Momentum:       close.pct_change(N)
  Ranking:        .rank(axis=1, pct=True) → cross-sectional percentile [0,1]
  Volatility:     close.pct_change().rolling(N).std()
  MA ratio:       close / close.rolling(N).mean() - 1
  EWM crossover:  close.ewm(span=N).mean() / close.ewm(span=M).mean() - 1
  Volume ratio:   volume / volume.rolling(N).mean()
  Bollinger z:    (close - close.rolling(N).mean()) / close.rolling(N).std()
  12-1 momentum:  close.pct_change(252) - close.pct_change(21)
  Multi-horizon:  average of rank signals at different lookbacks
"""

USER_PROMPT_TMPL = """## Research log (sorted by active Sharpe)
{log_summary}

## Research memo
{memo}
{failures}

---
Batch {iter_n} of {n_batches}.

Propose 3 NEW signal functions, each testing a DIFFERENT hypothesis.
Vary across: lookback horizons, feature combinations, normalization, filters.
Do NOT repeat signals that already appear in the research log.

Output format — follow EXACTLY:

=== CANDIDATE 1 ===
HYPOTHESIS: <one sentence>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return result
```

=== CANDIDATE 2 ===
HYPOTHESIS: <one sentence>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return result
```

=== CANDIDATE 3 ===
HYPOTHESIS: <one sentence>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return result
```
"""

REFLECTION_PROMPT = """Review this experiment log and write a research memo.

## Successful signals (sorted by active Sharpe)
{successes}

## Failures
{failure_summary}

## Current best
{best_info}

Write a concise RESEARCH MEMO (5-7 bullet points):
1. Which strategy families show the most promise? Cite Sharpe values.
2. What lookback horizons work best?
3. What is the most common failure mode?
4. What areas of the strategy space are UNEXPLORED?
5. What specific experiments should be tried next?
6. Could any existing signals be combined?
Be specific and quantitative.
"""

# ── Extraction ──
_CAND_RE = re.compile(
    r"===\\s*CANDIDATE\\s*(\\d+)\\s*===(.*?)(?====\\s*CANDIDATE|\\Z)",
    re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"```(?:python)?\\s*\\n(.*?)```", re.DOTALL)
_HYP_RE  = re.compile(r"HYPOTHESIS:\\s*(.+?)(?:\\n|$)", re.IGNORECASE)

def extract_candidates(text):
    """Extract up to 3 candidates from LLM output."""
    results = []
    for m in _CAND_RE.finditer(text):
        body = m.group(2)
        cm = _CODE_RE.search(body)
        if not cm:
            continue
        hm = _HYP_RE.search(body)
        results.append({
            "idx": int(m.group(1)),
            "hypothesis": hm.group(1).strip() if hm else "",
            "code": cm.group(1).strip(),
        })
    # fallback: just extract all code blocks
    if not results:
        for i, cm in enumerate(_CODE_RE.finditer(text)):
            code_text = cm.group(1).strip()
            if "def signal" in code_text:
                results.append({"idx": i + 1, "hypothesis": "", "code": code_text})
    return results[:3]

# -- Adapted prompt pack for this notebook's simpler loop --
SYSTEM_PROMPT = f"""You are a quantitative researcher discovering alpha signals for a market-neutral long-short equity strategy on ~100 US stocks.

INPUTS
- `close`: pd.DataFrame of daily adjusted close prices (rows=dates, cols=tickers)
- `volume`: pd.DataFrame of daily volumes (same shape)

Write: def signal(close, volume, vix=None, tnx=None) -> pd.DataFrame
Return values in [-1, +1]. +1 = strongest long, -1 = strongest short.
Positions are lagged by 1 day automatically in the harness. Do not add your own forward-looking shift.

WHAT YOU ARE REALLY OPTIMIZING
The notebook uses a train-time research score based on metrics it actually measures:
- {PRIMARY_OBJECTIVE_LABEL} is the main objective
- `vix`, `tnx`: optional pd.Series regime inputs aligned to the equity dates
- consistency (fraction of positive calendar-year Sharpes) is a bonus
- min_annual matters because unstable signals often fail held-out review
- beta above 0.10 is penalized
- turnover above 0.50 is penalized
- very deep drawdowns are penalized
- benchmark-relative Sharpe is a diagnostic, not the main target in market-neutral mode

The fastest way to lose is to maximize train Sharpe while ignoring stability.
Prefer signals that should stay long-short, diversified, and time-stable.

HARD RULES
1. Use ONLY numpy (np) and pandas (pd). No imports.
2. Return shape MUST equal close.shape.
3. Keep functions under 40 lines.
4. End EVERY signal with:
       out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
5. Do NOT use one-sided threshold rules that collapse to all-long or all-short.
6. Prefer percentile ranks:
       r = feature.rank(axis=1, pct=True)
       out = (r - 0.5) * 2
7. If you compute a row-wise Series such as feature.mean(axis=1), broadcast it back with
       feature.sub(row_series, axis=0)
   or
       feature - row_series.values[:, None]
   Never use row_series.reindex_like(feature) inside arithmetic.

LOOKAHEAD BIAS = AUTO-REJECT
- NEVER use full-series aggregations: close.mean(), volume.std(), close.quantile()
- ONLY use trailing windows: .pct_change(N), .rolling(N).mean(), .rolling(N).std(),
  .ewm(span=N).mean(), .shift(N) with N > 0
- Cross-sectional same-date ops are fine: .rank(axis=1), .mean(axis=1), .std(axis=1)

SIGNAL VALIDITY
- Both positive AND negative values on most days
- Near-constant cross-sections are rejected
- Strong directional bias is rejected

FACTOR FAMILY MENU
You are NOT limited to EWM crossovers. Legitimate families include:
1. momentum      - medium-term return ranks, multi-horizon momentum, 12-1 style momentum
2. mean_reversion - short-term loser bounce, pullback after stretched moves
3. volume        - price move confirmed by relative volume, turnover-conditioned trend
4. volatility    - vol-scaled momentum, low-vol continuation, breakout normalized by vol
5. ewm           - EWM crossover, EWM spread normalized by vol, EWM with trailing filters
6. multi_factor  - blends of two or more weakly correlated ranked signals
7. regime        - switching weights based on trailing realized volatility or volume stress

Use under-explored families when the search is stagnating.

SEED TEMPLATES
- momentum seed: 20d or 63d return rank, optionally vol-scaled
- mean_reversion seed: negative 5d return rank, optionally gated by recent stretch
- ewm seed: fast/slow EWM spread normalized by trailing vol
- volume seed: 21d return multiplied by relative volume rank
- regime seed: momentum blended with reversal under trailing stress
"""

USER_PROMPT_TMPL = """## Research log (sorted by research score)
{log_summary}

## Current best candidate
{current_best}

## Deterministic seed parents under the live objective
{seed_summary}

## Family balance in the current log
{family_summary}

## Research memo
{memo}
{failures}

## Stagnation status
{stagnation_status}

---
Batch {iter_n} of {n_batches}.

INSTRUCTIONS:
{diversity_instructions}

STRUCTURE IS SCORED BEFORE ALPHA.
If a candidate is missing PARENT_ID, MUTATION_TYPE, FAMILY, HYPOTHESIS, or ROBUSTNESS_RATIONALE,
it will be discarded before backtesting.

For EACH candidate provide:
  - PARENT_ID: copy an existing iter id when refining prior work, or "seed" if starting fresh
  - MUTATION_TYPE: what structural change you are making
  - FAMILY: one of momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor
  - HYPOTHESIS: one sentence of economic rationale
  - ROBUSTNESS_RATIONALE: one sentence explaining why this should survive across years
  - code block

Hard constraints:
  - Do NOT repeat signals already in the research log.
  - Do NOT include import lines.
  - Do NOT produce all-long or all-short signals.
  - Candidate 1 should mutate the current best template only when a viable parent exists; otherwise start from a deterministic seed parent.
  - Candidate 2 must come from a different family than candidate 1.
  - Candidate 3 must be either a blend or a regime-conditioned variant.
  - Generic 5d/10d/20d rank-only momentum and naive mislabeled reversion have already failed; do not repeat them.

Output format - follow EXACTLY:

=== CANDIDATE 1 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```

=== CANDIDATE 2 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```

=== CANDIDATE 3 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```
"""

REFLECTION_PROMPT = """Review this experiment log and write a research memo.

## Top scored signals (sorted by research score)
{successes}

## Family / cluster summary
{cluster_summary}

## Failures
{failure_summary}

## Current best
{best_info}

## Stagnation context
Generations without improvement: {stagnation_gens}
Most common failure reason: {top_failure_reason}

Write a RESEARCH MEMO with exactly these sections:

### 1. WHAT IS WORKING
For each promising family, cite research score, primary Sharpe, consistency, and min_annual.
If all cited ideas have negative score or negative Sharpe, start this section with exactly: Nothing is working yet.

### 2. ROBUSTNESS DIAGNOSIS
For the top 3 ideas, explain why they may fail across years.
Focus on consistency, min_annual, beta drift, turnover, and drawdown behavior.

### 3. UNEXPLORED TERRITORY
List 3 structural ideas or families that have not been explored enough.
For each give:
- family name
- economic rationale
- a 3-5 line code sketch

### 4. ANTI-PATTERN LIST
List the top 3 mutation patterns wasting compute and why.

### 5. NEXT 3 HYPOTHESES
Give three specific next experiments.
Each must include family, mutation type, and robustness argument.
At least one must come from unexplored territory.

Be specific and quantitative. Do not recycle vague EWM advice.
"""

def _field(pattern, body, default=""):
    match = re.search(pattern, body, re.IGNORECASE)
    return match.group(1).strip() if match else default

VALID_FAMILIES = {
    "momentum", "mean_reversion", "volume",
    "volatility", "ewm", "regime", "multi_factor",
}

def _normalize_family(value):
    fam = (value or "").strip().lower()
    return fam if fam in VALID_FAMILIES else "unknown"

def _is_structured_candidate(cand):
    return (
        bool(cand.get("parent_id"))
        and bool(cand.get("mutation_type"))
        and _normalize_family(cand.get("family")) != "unknown"
        and bool(cand.get("hypothesis"))
        and bool(cand.get("robustness_rationale"))
        and "def signal" in (cand.get("code") or "")
    )

def code_fingerprint(code_text):
    normalized = re.sub(r"#.*", "", code_text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

def validate_candidate_semantics(cand):
    family = _normalize_family(cand.get("family"))
    code = (cand.get("code") or "").lower()
    if family == "mean_reversion":
        if not any(token in code for token in ["0.5 -", "-close.pct_change", "-ret", "-returns", "(-", "negative"]):
            return False, "SEMANTIC_FAMILY: mean_reversion must invert recent-return direction"
    elif family == "volume":
        if code.count("volume") <= 1 or not any(token in code for token in ["volume.rolling", "volume.pct_change", "volume /", "/ volume"]):
            return False, "SEMANTIC_FAMILY: volume family must materially depend on volume features"
    elif family == "multi_factor":
        if code.count(".rank(axis=1") < 2:
            return False, "SEMANTIC_FAMILY: multi_factor must combine at least two ranked components"
    return True, "ok"

REPAIR_PROMPT = """Rewrite the draft response into the EXACT 3-candidate format below.
Keep the same candidate ideas when possible, but repair missing metadata and invalid formatting.
Do not add imports. Do not add prose outside the candidate blocks.

=== CANDIDATE 1 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```

=== CANDIDATE 2 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```

=== CANDIDATE 3 ===
PARENT_ID: <iter id or "seed">
MUTATION_TYPE: <what changed>
FAMILY: <momentum|mean_reversion|volume|volatility|ewm|regime|multi_factor>
HYPOTHESIS: <economic rationale in one sentence>
ROBUSTNESS_RATIONALE: <why the effect should remain stable across years>
```python
def signal(close, volume, vix=None, tnx=None):
    ...
    return out
```
"""

def extract_candidates(text):
    """Extract up to 3 candidates from LLM output."""
    results = []
    for m in _CAND_RE.finditer(text):
        body = m.group(2)
        cm = _CODE_RE.search(body)
        if not cm:
            continue
        results.append({
            "idx": int(m.group(1)),
            "parent_id": _field(r"PARENT_ID:\\s*(.+?)(?:\\n|$)", body, "seed"),
            "mutation_type": _field(r"MUTATION_TYPE:\\s*(.+?)(?:\\n|$)", body, "unspecified"),
            "family": _normalize_family(_field(r"FAMILY:\\s*(.+?)(?:\\n|$)", body, "unknown")),
            "hypothesis": _field(r"HYPOTHESIS:\\s*(.+?)(?:\\n|$)", body),
            "robustness_rationale": _field(r"ROBUSTNESS_RATIONALE:\\s*(.+?)(?:\\n|$)", body),
            "code": cm.group(1).strip(),
        })
    return [cand for cand in results[:3] if _is_structured_candidate(cand)]

def repair_and_extract_candidates(text):
    cands = extract_candidates(text)
    if len(cands) >= 3:
        return cands, text, False
    repaired = llm(
        [
            {"role": "system", "content": "You are a strict output formatter."},
            {"role": "user", "content": REPAIR_PROMPT + "\\n\\nDRAFT RESPONSE:\\n" + text[:6000]},
        ],
        max_new_tokens=LLM_REPAIR_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
    )
    return extract_candidates(repaired), repaired, True
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 9 — Research log
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 9 — Research log + reflection memo

Append-only JSONL log. The log is the model's **memory**: it sees its own
past experiments (ranked by Sharpe) plus recent failures. The reflection
memo is a separate file updated every few batches.
""")
code('''def append_log(entry):
    enriched = dict(entry)
    enriched.setdefault("run_id", RUN_ID)
    enriched.setdefault("run_profile", RUN_PROFILE)
    with open(RESEARCH_LOG, "a") as f:
        f.write(json.dumps(enriched, default=str) + "\\n")

def load_log():
    if not RESEARCH_LOG.exists():
        return []
    out = []
    for line in open(RESEARCH_LOG):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def save_memo(text):
    MEMO_FILE.write_text(text)
    update_runtime_state(last_memo_file=str(MEMO_FILE))

def load_memo():
    if MEMO_FILE.exists():
        return MEMO_FILE.read_text()
    return "(No memo yet — early exploration phase.)"

def log_summary_for_prompt(log, top_k=5, code_lines=6):
    good = [e for e in log if e.get("sharpe") is not None]
    if not good:
        return "(empty — this is your first experiment)"
    top = sorted(good, key=lambda e: -e["sharpe"])[:top_k]
    lines = []
    for i, e in enumerate(top):
        lines.append(
            f"#{i+1}  iter {e['iter']}  activeSh={e['sharpe']:+.2f}  "
            f"rawSh={e.get('sharpe_raw', 0):+.2f}  beta={e.get('beta', 0):+.2f}  "
            f"consistency={e.get('consistency', 0):.0%}  "
            f"DD={e.get('max_dd', 0):+.1%}"
        )
        if e.get("hypothesis"):
            lines.append(f"    HYP: {e['hypothesis'][:150]}")
        code_snip = "\\n".join(
            "    " + ln for ln in e["code"].splitlines()[:code_lines])
        lines.append(code_snip)
        if len(e["code"].splitlines()) > code_lines:
            lines.append("    ...")
        lines.append("")
    return "\\n".join(lines)

def failures_for_prompt(log, n_reject=5, n_runtime=3):
    fails = [e for e in log if e.get("error")]
    rejects = [e for e in fails
               if any(k in str(e.get("error", ""))
                      for k in ["REJECT", "DEGENERATE", "LOOKAHEAD"])]
    runtime = [e for e in fails if e not in rejects]
    chosen = rejects[-n_reject:] + runtime[-n_runtime:]
    if not chosen:
        return ""
    lines = ["\\n## Recent failures (avoid these mistakes)"]
    for e in chosen:
        lines.append(f"- iter {e.get('iter', '?')}: {str(e['error'])[:180]}")
        if e.get("hypothesis"):
            lines.append(f"    was: {e['hypothesis'][:120]}")
    return "\\n".join(lines)

STAGNATION_EXPLORATION_FAMILIES = [
    "momentum",
    "mean_reversion",
    "volume",
    "volatility",
    "multi_factor",
    "regime",
]

def infer_family(entry):
    family = str(entry.get("family", "") or "").strip().lower()
    if family in {
        "momentum", "mean_reversion", "volume", "volatility",
        "ewm", "regime", "multi_factor",
    }:
        return family
    text = "\\n".join([
        str(entry.get("mutation_type", "") or ""),
        str(entry.get("hypothesis", "") or ""),
        str(entry.get("code", "") or ""),
    ]).lower()
    if "ewm(" in text:
        return "ewm"
    if "volume" in text:
        return "volume"
    if "rolling" in text and ".std(" in text:
        return "volatility"
    if "blend" in text or "mix" in text:
        return "multi_factor"
    if "reversion" in text or "-close.pct_change(5)" in text:
        return "mean_reversion"
    if "regime" in text or "stress" in text:
        return "regime"
    return "unknown"

def research_score(entry):
    sharpe = float(entry.get("sharpe", 0.0) or 0.0)
    consistency = float(entry.get("consistency", 0.0) or 0.0)
    min_annual = float(entry.get("min_annual", 0.0) or 0.0)
    beta_penalty = max(0.0, abs(float(entry.get("beta", 0.0) or 0.0)) - 0.10)
    turnover_penalty = max(0.0, float(entry.get("turnover", 0.0) or 0.0) - 0.50)
    dd_penalty = max(0.0, abs(float(entry.get("max_dd", 0.0) or 0.0)) - 0.25)
    return (
        sharpe
        + 0.35 * consistency
        + 0.20 * min_annual
        - 1.00 * beta_penalty
        - 0.50 * turnover_penalty
        - 0.30 * dd_penalty
    )

def is_viable_candidate(entry):
    if entry.get("sharpe") is None:
        return False
    return (
        float(entry.get("sharpe", 0.0) or 0.0) >= MIN_VIABLE_TRAIN_SHARPE
        and research_score(entry) >= MIN_VIABLE_RESEARCH_SCORE
    )

def _successful_log_rows(log):
    return [e for e in log if is_viable_candidate(e)]

def _scored_log_rows(log):
    return [e for e in log if e.get("sharpe") is not None]

def _prompt_parent_rows(log):
    return _successful_log_rows(log)

SEED_TEMPLATE_LIBRARY = [
    {
        "seed_id": "seed_momentum_63",
        "family": "momentum",
        "label": "63d momentum rank",
        "code": """def signal(close, volume):
    feature = close.pct_change(63)
    r = feature.rank(axis=1, pct=True)
    out = (r - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out""",
    },
    {
        "seed_id": "seed_reversion_5",
        "family": "mean_reversion",
        "label": "negative 5d reversal",
        "code": """def signal(close, volume):
    feature = -close.pct_change(5)
    r = feature.rank(axis=1, pct=True)
    out = (r - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out""",
    },
    {
        "seed_id": "seed_ewm_10_50",
        "family": "ewm",
        "label": "10/50 ewm spread",
        "code": """def signal(close, volume):
    fast = close.ewm(span=10, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    feature = (fast - slow) / slow.replace(0, np.nan)
    r = feature.rank(axis=1, pct=True)
    out = (r - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out""",
    },
    {
        "seed_id": "seed_regime_blend",
        "family": "regime",
        "label": "63d momentum + 5d reversal blend",
        "code": """def signal(close, volume):
    mom = close.pct_change(63).rank(axis=1, pct=True) - 0.5
    rev = (-close.pct_change(5)).rank(axis=1, pct=True) - 0.5
    feature = 0.6 * mom + 0.4 * rev
    r = feature.rank(axis=1, pct=True)
    out = (r - 0.5) * 2
    out = out.sub(out.mean(axis=1), axis=0).clip(-1, 1).fillna(0.0)
    return out""",
    },
]
_SEED_SUMMARY_CACHE = None

def seed_summary_for_prompt():
    global _SEED_SUMMARY_CACHE
    if _SEED_SUMMARY_CACHE is not None:
        return _SEED_SUMMARY_CACHE
    lines = []
    for seed in SEED_TEMPLATE_LIBRARY:
        sig, err = run_signal_code(seed["code"], close_train, volume_train, vix_s=vix_train, tnx_s=tnx_train, timeout=20)
        if err:
            lines.append(f"- {seed['seed_id']} | family={seed['family']} | ERROR: {err}")
            continue
        m = backtest(sig, close_train)
        lines.append(
            f"- {seed['seed_id']} | family={seed['family']} | label={seed['label']} | "
            f"score={research_score({'sharpe': m['sharpe_active'], 'consistency': m['consistency'], 'min_annual': m['min_annual'], 'beta': m['beta'], 'turnover': m['avg_turnover'], 'max_dd': m['max_dd_active']}):+.2f} | "
            f"primarySh={m['sharpe_active']:+.2f} | rawSh={m['sharpe_raw']:+.2f} | "
            f"benchSpread={m.get('benchmark_spread_sharpe', 0.0):+.2f} | beta={m['beta']:+.2f}"
        )
    _SEED_SUMMARY_CACHE = "\\n".join(lines)
    return _SEED_SUMMARY_CACHE

def _top_families_in_log(log_rows, n=3):
    counts = {}
    for row in _prompt_parent_rows(log_rows):
        fam = infer_family(row)
        if fam == "unknown":
            continue
        counts[fam] = counts.get(fam, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [fam for fam, _ in ordered[:n]]

def build_diversity_instructions(log_rows, stagnation_gens=0):
    top_families = _top_families_in_log(log_rows)
    unexplored = [f for f in STAGNATION_EXPLORATION_FAMILIES if f not in top_families]
    if stagnation_gens <= 0:
        return (
            "Propose 3 candidates. At least ONE must come from a family that is not "
            f"currently dominant in the log (dominant: {', '.join(top_families) or 'none yet'}). "
            "The other two may refine the current best area."
        )
    if stagnation_gens <= 3:
        fam_hint = ", ".join(unexplored[:2]) if unexplored else "multi_factor, regime"
        return (
            f"STAGNATION DETECTED: {stagnation_gens} batches without a new best score. "
            "At least TWO candidates must come from families outside the dominant cluster. "
            f"Suggested under-explored families: {fam_hint}. "
            "Do not spend the whole batch on small EWM span tweaks."
        )
    fam_hint = ", ".join(unexplored) if unexplored else "volatility, multi_factor, regime"
    return (
        f"HARD STAGNATION: {stagnation_gens} batches without improvement. "
        "ALL THREE candidates must open new territory. "
        f"Choose from: {fam_hint}. "
        "Do not anchor on the current winner. Structural change is required."
    )

def _prompt_progress_state(log_rows):
    good = sorted(
        _scored_log_rows(log_rows),
        key=lambda e: (int(e.get("batch", -1)), int(e.get("iter", -1))),
    )
    if not good:
        return {"stalled_gens": 0, "best_score": 0.0, "best_gain": 0.0, "generations_run": 0}
    batch_best = {}
    for entry in good:
        batch_id = int(entry.get("batch", -1))
        batch_best[batch_id] = max(batch_best.get(batch_id, -1e9), research_score(entry))
    best_so_far = -1e9
    stalled = 0
    last_gain = 0.0
    for batch_id in sorted(batch_best):
        score = batch_best[batch_id]
        if score > best_so_far + 1e-9:
            last_gain = 0.0 if best_so_far <= -1e8 else score - best_so_far
            best_so_far = score
            stalled = 0
        else:
            stalled += 1
            last_gain = score - best_so_far
    return {
        "stalled_gens": stalled,
        "best_score": 0.0 if best_so_far <= -1e8 else float(best_so_far),
        "best_gain": float(last_gain),
        "generations_run": len(batch_best),
    }

def build_stagnation_status(stagnation_gens, best_score, best_score_delta, generations_run):
    if stagnation_gens <= 0:
        return f"No stagnation. Best research score: {best_score:.3f} (improved in the latest scored batch)."
    return (
        f"Stagnation: {stagnation_gens} batches without improvement in best research score.\\n"
        f"Best research score so far: {best_score:.3f} (last gain: {best_score_delta:+.4f}).\\n"
        f"Total scored batches: {generations_run}.\\n"
        "The search is spending too much time in familiar territory. A family-level change is required."
    )

def current_best_for_prompt(log_rows):
    good = _prompt_parent_rows(log_rows)
    if not good:
        return "(no viable parents yet - mutate one of the deterministic seed parents instead of mutating losers)"
    best = max(good, key=research_score)
    return (
        f"iter {best.get('iter', '?')} | family={infer_family(best)} | "
        f"score={research_score(best):+.2f} | primarySh={float(best.get('sharpe', 0.0) or 0.0):+.2f} | "
        f"consistency={float(best.get('consistency', 0.0) or 0.0):.0%} | "
        f"min_annual={float(best.get('min_annual', 0.0) or 0.0):+.2f} | "
        f"beta={float(best.get('beta', 0.0) or 0.0):+.2f}"
    )

def family_summary_for_prompt(log_rows):
    good = _prompt_parent_rows(log_rows)
    if not good:
        return "(no viable families yet)"
    counts = {}
    for entry in good:
        fam = infer_family(entry)
        counts[fam] = counts.get(fam, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "\\n".join([f"- {fam}: {count}" for fam, count in ordered[:6]])

def failure_bucket_counts(log_rows):
    counts = {}
    for entry in log_rows[-40:]:
        err = str(entry.get("error", "") or "").lower()
        if err:
            if "lookahead" in err:
                key = "lookahead"
            elif "structured_parse_failed" in err:
                key = "format_failure"
            elif "duplicate" in err:
                key = "duplicate_candidate"
            elif "semantic_family" in err:
                key = "semantic_family"
            elif "non_viable" in err:
                key = "non_viable_alpha"
            elif "degenerate" in err or "near-constant" in err:
                key = "degenerate_signal"
            elif "validation" in err:
                key = "validation"
            elif "timeout" in err:
                key = "timeout"
            elif "backtest" in err:
                key = "backtest_error"
            else:
                key = "runtime_error"
        elif entry.get("sharpe") is not None:
            if abs(float(entry.get("beta", 0.0) or 0.0)) > 0.10:
                key = "beta_drift"
            elif float(entry.get("turnover", 0.0) or 0.0) > 0.50:
                key = "high_turnover"
            elif float(entry.get("min_annual", 0.0) or 0.0) < 0.0:
                key = "year_instability"
            elif float(entry.get("consistency", 0.0) or 0.0) < 0.50:
                key = "low_consistency"
            else:
                continue
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts

def cluster_summary_for_prompt(log_rows):
    good = sorted(_prompt_parent_rows(log_rows), key=research_score, reverse=True)
    if not good:
        return "(no viable clusters yet)"
    by_family = {}
    for entry in good:
        fam = infer_family(entry)
        by_family.setdefault(fam, []).append(entry)
    lines = []
    for fam, items in sorted(by_family.items(), key=lambda item: (-len(item[1]), item[0]))[:6]:
        top = max(items, key=research_score)
        lines.append(
            f"- {fam}: count={len(items)} topScore={research_score(top):+.2f} "
            f"topSh={float(top.get('sharpe', 0.0) or 0.0):+.2f} "
            f"cons={float(top.get('consistency', 0.0) or 0.0):.0%}"
        )
    return "\\n".join(lines)

def should_abort_early_negative(log_rows):
    evaluated = sorted(
        _scored_log_rows(log_rows),
        key=lambda e: int(e.get("iter", 10**9)),
    )
    n = max(0, int(EARLY_ABORT_NEGATIVE_SUCCESSFUL))
    if (
        n <= 0
        or len(evaluated) < n
        or len({int(e.get("batch", -1)) for e in evaluated}) < EARLY_ABORT_MIN_BATCH
        or int(RUNTIME_STATE.get("reflection_calls", 0) or 0) < EARLY_ABORT_MIN_REFLECTIONS
    ):
        return False, ""
    first_n = evaluated[:n]
    all_bad = all(
        float(row.get("sharpe", 0.0) or 0.0) <= EARLY_ABORT_NEGATIVE_SHARPE
        and research_score(row) <= EARLY_ABORT_NEGATIVE_SCORE
        for row in first_n
    )
    if not all_bad:
        return False, ""
    sample = ", ".join(
        f"iter {row.get('iter', '?')}: Sh={float(row.get('sharpe', 0.0) or 0.0):+.2f}, score={research_score(row):+.2f}"
        for row in first_n[:4]
    )
    reason = (
        f"first {n} scored backtests all <= thresholds "
        f"(Sh <= {EARLY_ABORT_NEGATIVE_SHARPE:+.2f}, score <= {EARLY_ABORT_NEGATIVE_SCORE:+.2f}); "
        f"sample: {sample}"
    )
    return True, reason

def should_abort_search_collapse(log_rows):
    if (
        len({int(e.get("batch", -1)) for e in log_rows}) < SEARCH_COLLAPSE_MIN_BATCHES
        or int(RUNTIME_STATE.get("reflection_calls", 0) or 0) < 1
    ):
        return False, ""
    scored = _scored_log_rows(log_rows)
    collapse_errors = []
    for row in log_rows:
        err = str(row.get("error", "") or "").lower()
        if any(tag in err for tag in ["structured_parse_failed", "duplicate", "semantic_family"]):
            collapse_errors.append(row)
    if len(collapse_errors) < SEARCH_COLLAPSE_MIN_ERRORS:
        return False, ""
    total_attempts = len(scored) + len(collapse_errors)
    if total_attempts <= 0:
        return False, ""
    failure_rate = len(collapse_errors) / total_attempts
    if failure_rate < SEARCH_COLLAPSE_FAILURE_RATE:
        return False, ""
    sample = ", ".join(
        str(row.get("error", "") or "").split("\\n", 1)[0][:80]
        for row in collapse_errors[:4]
    )
    reason = (
        f"search-quality collapse: {len(collapse_errors)}/{total_attempts} candidate attempts failed "
        f"due to format/duplicate/semantic-family issues (rate={failure_rate:.0%}); sample: {sample}"
    )
    return True, reason

def log_summary_for_prompt(log, top_k=5, code_lines=6):
    good = _prompt_parent_rows(log)
    if not good:
        return "(no viable candidates yet - do not mutate losing rows)"
    top = sorted(good, key=research_score, reverse=True)[:top_k]
    lines = []
    for i, e in enumerate(top):
        lines.append(
            f"#{i+1}  iter {e['iter']}  family={infer_family(e)}  score={research_score(e):+.2f}  "
            f"primarySh={float(e.get('sharpe', 0.0) or 0.0):+.2f}  "
            f"beta={float(e.get('beta', 0.0) or 0.0):+.2f}  "
            f"consistency={float(e.get('consistency', 0.0) or 0.0):.0%}  "
            f"minYr={float(e.get('min_annual', 0.0) or 0.0):+.2f}  "
            f"DD={float(e.get('max_dd', 0.0) or 0.0):+.1%}"
        )
        if e.get("hypothesis"):
            lines.append(f"    HYP: {e['hypothesis'][:150]}")
        if e.get("robustness_rationale"):
            lines.append(f"    ROBUST: {e['robustness_rationale'][:150]}")
        code_snip = "\\n".join(
            "    " + ln for ln in e["code"].splitlines()[:code_lines])
        lines.append(code_snip)
        if len(e["code"].splitlines()) > code_lines:
            lines.append("    ...")
        lines.append("")
    return "\\n".join(lines)

def failures_for_prompt(log, n_reject=5, n_runtime=3):
    fails = [e for e in log if e.get("error")]
    rejects = [e for e in fails
               if any(k in str(e.get("error", ""))
                      for k in ["REJECT", "DEGENERATE", "LOOKAHEAD"])]
    runtime = [e for e in fails if e not in rejects]
    chosen = rejects[-n_reject:] + runtime[-n_runtime:]
    if not chosen:
        return ""
    lines = ["\\n## Recent failures (avoid these mistakes)"]
    for e in chosen:
        lines.append(f"- iter {e.get('iter', '?')}: {str(e['error'])[:180]}")
        if e.get("hypothesis"):
            lines.append(f"    was: {e['hypothesis'][:120]}")
        if e.get("family"):
            lines.append(f"    family: {e.get('family')}")
    return "\\n".join(lines)
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 10 — The AutoResearch loop
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 10 — The AutoResearch loop

Each batch:
1. Build prompt from research log + reflection memo + failures
2. Generate 3 candidates via LLM
3. Sandbox → validate → lookahead check → backtest each
4. Append results to log

Every `REFLECT_EVERY` batches the model reviews its full log and writes
a **research memo** — what's working, what's not, what to try next.
This memo is included in all future prompts: the model learns from itself.
""")
code('''from concurrent.futures import ThreadPoolExecutor

_COUNTER = [0]

def process_batch(batch_id, response, close_df, volume_df, t_gen):
    """Parse candidates, sandbox + validate + backtest each."""
    cands, parsed_text, repaired = repair_and_extract_candidates(response)
    update_runtime_state(
        structured_candidate_count=len(cands),
        repair_batches=int(RUNTIME_STATE.get("repair_batches", 0)) + (1 if repaired else 0),
    )
    if len(cands) < 3:
        append_log({"batch": batch_id, "iter": _COUNTER[0],
                     "error": "structured_parse_failed",
                     "raw": response[:700],
                     "repaired": repaired,
                     "parsed_text": parsed_text[:700],
                     "structured_candidate_count": len(cands)})
        update_runtime_state(format_failures=int(RUNTIME_STATE.get("format_failures", 0) or 0) + 1)
        _COUNTER[0] += 1
        print(f"  [b{batch_id:02d}] structured parse failed ({len(cands)}/3 candidates)")
        return

    existing_fingerprints = {
        e.get("code_fingerprint")
        for e in load_log()
        if e.get("code_fingerprint")
    }
    batch_fingerprints = set()
    for cand in cands:
        it = _COUNTER[0]; _COUNTER[0] += 1
        t0 = time.time()
        fingerprint = code_fingerprint(cand["code"])

        if fingerprint in existing_fingerprints or fingerprint in batch_fingerprints:
            update_runtime_state(
                duplicate_rejections=int(RUNTIME_STATE.get("duplicate_rejections", 0)) + 1
            )
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"],
                         "code_fingerprint": fingerprint,
                         "error": "DUPLICATE: repeated candidate fingerprint",
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] DUPLICATE candidate")
            continue
        batch_fingerprints.add(fingerprint)

        ok, why = validate_candidate_semantics(cand)
        if not ok:
            update_runtime_state(
                semantic_family_rejections=int(RUNTIME_STATE.get("semantic_family_rejections", 0)) + 1
            )
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"],
                         "code_fingerprint": fingerprint,
                         "error": why,
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] {why}")
            continue

        # 1. run signal
        sig_df, err = run_signal_code(cand["code"], close_df, volume_df, vix_s=vix_train, tnx_s=tnx_train)
        if err is not None:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"], "error": err,
                         "code_fingerprint": fingerprint,
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] {err[:80]}")
            continue

        # 2. validate signal distribution
        ok, why = validate_signal(sig_df)
        if not ok:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"],
                         "code_fingerprint": fingerprint,
                         "error": f"DEGENERATE: {why}",
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] DEGENERATE: {why}")
            continue

        # 3. lookahead check
        ok, why = detect_lookahead(cand["code"], close_df, volume_df, vix_s=vix_train, tnx_s=tnx_train)
        if not ok:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"],
                         "code_fingerprint": fingerprint,
                         "error": f"REJECT: {why}",
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] REJECT: {why[:70]}")
            continue

        # 4. backtest
        try:
            m = backtest(sig_df, close_df)
        except Exception as e:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "parent_id": cand.get("parent_id", "seed"),
                         "mutation_type": cand.get("mutation_type", "unspecified"),
                         "family": cand.get("family", "unknown"),
                         "hypothesis": cand["hypothesis"],
                         "robustness_rationale": cand.get("robustness_rationale", ""),
                         "code": cand["code"],
                         "code_fingerprint": fingerprint,
                         "error": f"backtest: {e}",
                         "repaired": repaired})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] backtest error: {e}")
            continue

        entry = {
            "batch": batch_id, "iter": it, "cand_idx": cand["idx"],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parent_id": cand.get("parent_id", "seed"),
            "mutation_type": cand.get("mutation_type", "unspecified"),
            "family": cand.get("family", "unknown"),
            "hypothesis": cand["hypothesis"],
            "robustness_rationale": cand.get("robustness_rationale", ""),
            "repaired": repaired,
            "code_fingerprint": fingerprint,
            "code": cand["code"],
            "sharpe":      m["sharpe_active"],
            "sharpe_raw":  m["sharpe_raw"],
            "ann_return":  m["ann_active"],
            "beta":        m["beta"],
            "max_dd":      m["max_dd_active"],
            "hit_rate":    m["hit_rate"],
            "turnover":    m["avg_turnover"],
            "consistency": m["consistency"],
            "min_annual":  m["min_annual"],
            "annual_sharpes": m["annual_sharpes"],
            "gen_dt": t_gen,
            "bt_dt": round(time.time() - t0, 1),
        }
        entry["research_score"] = research_score(entry)
        if not is_viable_candidate(entry):
            entry["error"] = (
                "NON_VIABLE: "
                f"train Sharpe {entry['sharpe']:+.2f} < {MIN_VIABLE_TRAIN_SHARPE:+.2f} "
                f"or score {entry['research_score']:+.2f} < {MIN_VIABLE_RESEARCH_SCORE:+.2f}"
            )
            append_log(entry)
            print(f"  [b{batch_id:02d}.c{cand['idx']}] "
                  f"NON_VIABLE  score={entry['research_score']:+.2f}  Sh={m['sharpe_active']:+.2f}  "
                  f"beta={m['beta']:+.2f}  cons={m['consistency']:.0%}  DD={m['max_dd_active']:+.1%}")
            continue
        append_log(entry)
        print(f"  [b{batch_id:02d}.c{cand['idx']}] "
              f"score={entry['research_score']:+.2f}  Sh={m['sharpe_active']:+.2f}  beta={m['beta']:+.2f}  "
              f"cons={m['consistency']:.0%}  DD={m['max_dd_active']:+.1%}")


def do_reflection(batch_id):
    """Model reviews its full research log and writes a memo."""
    log = load_log()
    good = sorted(_scored_log_rows(log),
                  key=research_score, reverse=True)
    fails = [e for e in log if e.get("error")]

    if not good:
        return   # nothing to reflect on yet

    succ_lines = []
    for e in good[:12]:
        succ_lines.append(
            f"iter {e['iter']}  family={infer_family(e)}  score={research_score(e):+.2f}  primarySh={e['sharpe']:+.2f}  "
            f"beta={e.get('beta',0):+.2f}  "
            f"cons={e.get('consistency',0):.0%}  minYr={e.get('min_annual',0):+.2f}  "
            f"hyp: {e.get('hypothesis','')[:120]}")

    fail_lines = []
    for e in fails[-10:]:
        fail_lines.append(
            f"iter {e.get('iter','?')}: {str(e['error'])[:150]}")

    best = good[0]
    progress = _prompt_progress_state(log)
    fail_counts = failure_bucket_counts(log)
    top_failure = max(fail_counts, key=fail_counts.get) if fail_counts else "unknown"
    best_info = (f"iter {best['iter']}  family={infer_family(best)}  score={research_score(best):+.2f}  "
                 f"primarySh={best['sharpe']:+.2f}  beta={best.get('beta',0):+.2f}\\n"
                 f"    {best.get('hypothesis','')}")
    if all((float(e.get("sharpe", 0.0) or 0.0) < 0.0 or research_score(e) < 0.0) for e in good[:12]):
        best_info = "Nothing is working yet.\\n" + best_info

    msg = REFLECTION_PROMPT.format(
        successes="\\n".join(succ_lines),
        cluster_summary=cluster_summary_for_prompt(log),
        failure_summary="\\n".join(fail_lines) if fail_lines else "(none)",
        best_info=best_info,
        stagnation_gens=progress["stalled_gens"],
        top_failure_reason=f"{top_failure} ({fail_counts.get(top_failure, 0)} recent hits)",
    )
    try:
        memo = llm(
            [{"role": "system",
              "content": "You are a senior quantitative researcher."},
             {"role": "user", "content": msg}],
            max_new_tokens=LLM_REFLECTION_MAX_NEW_TOKENS, temperature=0.2)
        save_memo(memo)
        update_runtime_state(
            reflection_calls=int(RUNTIME_STATE.get("reflection_calls", 0)) + 1,
            llm_stage_executed=True,
        )
        print(f"  [reflection @ b{batch_id:02d}] memo updated "
              f"({len(memo)} chars)")
    except Exception as e:
        update_runtime_state(last_reflection_error=f"{type(e).__name__}: {e}")
        print(f"  [reflection @ b{batch_id:02d}] failed: {e}")


def run_research_loop(n_batches, close_df, volume_df):
    existing = load_log()
    update_runtime_state(loop_status="running", existing_log_entries=len(existing))
    _COUNTER[0] = max((e.get("iter", -1) for e in existing),
                      default=-1) + 1
    start_batch = max((e.get("batch", -1) for e in existing),
                      default=-1) + 1
    print(f"\\n{'='*60}")
    print(f"AutoResearch loop: batches {start_batch}\u2192{n_batches-1}  "
          f"(log={len(existing)} entries, counter={_COUNTER[0]})")
    print(f"{'='*60}\\n")

    executor = ThreadPoolExecutor(max_workers=1)
    pending  = None

    for b in range(start_batch, n_batches):
        log = load_log()
        memo = load_memo()
        progress = _prompt_progress_state(log)

        user_msg = USER_PROMPT_TMPL.format(
            iter_n=b + 1,
            n_batches=n_batches,
            log_summary=log_summary_for_prompt(log),
            current_best=current_best_for_prompt(log),
            seed_summary=seed_summary_for_prompt(),
            family_summary=family_summary_for_prompt(log),
            memo=memo,
            failures=failures_for_prompt(log),
            stagnation_status=build_stagnation_status(
                progress["stalled_gens"],
                progress["best_score"],
                progress["best_gain"],
                progress["generations_run"],
            ),
            diversity_instructions=build_diversity_instructions(
                log_rows=log,
                stagnation_gens=progress["stalled_gens"],
            ),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        t0 = time.time()
        try:
            response = llm(messages, max_new_tokens=LLM_BATCH_MAX_NEW_TOKENS,
                           temperature=0.65)
        except Exception as e:
            append_log({"batch": b, "error": f"llm: {e}"})
            print(f"[b{b:02d}] LLM generation failed: {e}")
            continue
        t_gen = round(time.time() - t0, 1)
        print(f"[b{b:02d}] generated in {t_gen}s  "
              f"({len(response)} chars)")

        # wait for prev batch, then decide whether this newly generated batch is worth scoring
        if pending is not None:
            pending.result()
            pending = None
        log = load_log()
        collapse_now, collapse_reason = should_abort_search_collapse(log)
        if collapse_now:
            update_runtime_state(
                loop_status="aborted_search_collapse",
                loop_stop_reason="search_quality_collapse",
                loop_stop_detail=collapse_reason,
                final_log_entries=len(log),
            )
            print(f"[early-stop/search] {collapse_reason}")
            break
        abort_now, abort_reason = should_abort_early_negative(log)
        if abort_now:
            update_runtime_state(
                loop_status="aborted_early_negative",
                loop_stop_reason="early_negative_candidates",
                loop_stop_detail=abort_reason,
                final_log_entries=len(log),
            )
            print(f"[early-stop] {abort_reason}")
            break
        pending = executor.submit(
            process_batch, b, response, close_df, volume_df, t_gen)

        gc.collect(); torch.cuda.empty_cache()

        # ── periodic reflection ──
        should_reflect = (
            REFLECTION_ENABLED
            and (b + 1) >= FIRST_REFLECTION_BATCH
            and ((b + 1) == FIRST_REFLECTION_BATCH or ((b + 1 - FIRST_REFLECTION_BATCH) % REFLECT_EVERY == 0))
        )
        if should_reflect:
            if pending is not None:
                pending.result()
                pending = None
            do_reflection(b)
            gc.collect(); torch.cuda.empty_cache()

    if pending is not None:
        pending.result()
    executor.shutdown(wait=True)
    if RUNTIME_STATE.get("loop_status") not in {"aborted_early_negative", "aborted_search_collapse"}:
        update_runtime_state(loop_status="completed", final_log_entries=len(load_log()))
    print(f"\\n{'='*60}")
    print("AutoResearch loop complete.")
    print(f"{'='*60}")

if RUN_LLM_STAGE:
    run_research_loop(N_BATCHES, close_train, volume_train)
else:
    update_runtime_state(loop_status="skipped_llm_disabled")
    print("RUN_LLM_STAGE is False; skipping AutoResearch loop.")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 11 — Held-out evaluation
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 11 — Held-out evaluation (2023–2024)

Take the top-K train signals and re-run on untouched test data.
Walk-forward: slice held-out into quarters and check consistency.
""")
code('''TOP_K = 5
WF_WINDOWS = 4

def walk_forward(code_str, close_df, volume_df, n_windows=WF_WINDOWS):
    N  = len(close_df)
    sz = N // n_windows
    per_window = []
    for w in range(n_windows):
        lo = w * sz
        hi = (w + 1) * sz if w < n_windows - 1 else N
        sub_c, sub_v = close_df.iloc[lo:hi], volume_df.iloc[lo:hi]
        sig, err = run_signal_code(code_str, sub_c, sub_v, vix_s=vix_test.iloc[lo:hi], tnx_s=tnx_test.iloc[lo:hi], timeout=20)
        if err:
            per_window.append({"window": w, "error": err})
            continue
        m = backtest(sig, sub_c)
        per_window.append({
            "window": w,
            "sharpe_active": m["sharpe_active"],
            "ann_active":    m["ann_active"],
            "beta":          m["beta"],
            "max_dd":        m["max_dd_active"],
        })
    return per_window

def evaluate_on_test(top_k=TOP_K):
    log  = load_log()
    good = _successful_log_rows(log)
    top  = sorted(good, key=research_score, reverse=True)[:top_k]
    results = []
    for e in top:
        ok, why = detect_lookahead(e["code"], close_test, volume_test, vix_s=vix_test, tnx_s=tnx_test)
        if not ok:
            print(f"  iter {e['iter']}: REJECTED on test data \u2014 {why[:80]}")
            continue
        sig, err = run_signal_code(e["code"], close_test, volume_test,
                                   vix_s=vix_test, tnx_s=tnx_test, timeout=25)
        if err:
            print(f"  iter {e['iter']}: test run failed \u2014 {err[:80]}")
            continue
        m  = backtest(sig, close_test)
        wf = walk_forward(e["code"], close_test, volume_test)
        wf_sh = [w["sharpe_active"] for w in wf if "sharpe_active" in w]
        results.append({
            "iter":         e["iter"],
            "family":       infer_family(e),
            "hypothesis":   e.get("hypothesis", ""),
            "train_score":  research_score(e),
            "train_sharpe": e["sharpe"],
            "test_sharpe":  m["sharpe_active"],
            "test_raw":     m["sharpe_raw"],
            "test_bench_spread": m.get("benchmark_spread_sharpe", 0.0),
            "test_beta":    m["beta"],
            "test_ret":     m["ann_active"],
            "test_dd":      m["max_dd_active"],
            "wf_median":    float(np.median(wf_sh)) if wf_sh else 0.0,
            "wf_min":       float(min(wf_sh))       if wf_sh else 0.0,
            "wf_windows":   wf,
            "equity":       m["equity_primary"],
            "code":         e["code"],
        })
    return results

test_results = evaluate_on_test()
print(f"\\n{'iter':>4} {'train_pSh':>10} {'test_pSh':>9} {'beta':>7} "
      f"{'wf_med':>7} {'test_ret':>8} {'test_dd':>8}")
for r in test_results:
    print(f"{r['iter']:>4} {r['train_sharpe']:>+9.2f} "
          f"{r['test_sharpe']:>+8.2f} {r['test_beta']:>+7.2f} "
          f"{r['wf_median']:>+7.2f} {r['test_ret']:>+8.1%} "
          f"{r['test_dd']:>+8.1%}")

if test_results:
    best = max(test_results, key=lambda r: r["test_sharpe"])
    BEST_CODE.write_text(
        f"# iter {best['iter']}  family={best.get('family', 'unknown')}  train_score={best.get('train_score', 0.0):+.2f}  train_Sh={best['train_sharpe']:+.2f}  "
        f"test_Sh={best['test_sharpe']:+.2f}\\n"
        f"# HYPOTHESIS: {best['hypothesis']}\\n\\n{best['code']}\\n")
    print(f"\\nbest (by test Sharpe): iter {best['iter']}  "
          f"saved to {BEST_CODE.name}")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 12 — Plots
# ═════════════════════════════════════════════════════════════════════════════
md("## Cell 12 \u2014 Plots")
code('''log     = load_log()
iters   = [e.get("iter", 0) for e in log]
sharpes = [e.get("sharpe") for e in log]

running_best, cur = [], -np.inf
for s in sharpes:
    if s is not None and s > cur:
        cur = s
    running_best.append(cur if cur > -np.inf else np.nan)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(iters,
           [s if s is not None else np.nan for s in sharpes],
           alpha=0.5, s=40, label="iter Sharpe")
ax.plot(iters, running_best, "-", lw=2, label="running best")
ax.axhline(0, ls="--", color="grey", alpha=0.4)
ax.set_xlabel("Iteration"); ax.set_ylabel("Active Sharpe (train)")
ax.set_title("AutoResearch v2 \u2014 alpha discovery progress")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(SHARPE_PLOT, dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
for r in test_results:
    r["equity"].plot(
        ax=ax,
        label=f"iter {r['iter']}  trSh={r['train_sharpe']:+.1f} "
              f"teSh={r['test_sharpe']:+.1f}")
ax.set_ylabel("Equity (active)")
ax.set_title(f"Top-{TOP_K} signals on held-out test (2023\u20132024)")
ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(EQUITY_PLOT, dpi=140)
plt.close(fig)
print("plots saved")
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 13 — Summary
# ═════════════════════════════════════════════════════════════════════════════
md("## Cell 13 \u2014 Experiment summary")
code('''log  = load_log()
good = [e for e in log if e.get("sharpe") is not None]
fail = [e for e in log if e.get("error")]
viable = _successful_log_rows(log)
best_train = (max(viable, key=lambda e: e["sharpe"])
              if viable else {"iter": -1, "sharpe": 0, "hypothesis": "(none)"})
best_test  = (max(test_results, key=lambda r: r["test_sharpe"])
              if test_results else None)
runtime_snapshot = sync_runtime_metadata(
    summary_pending=True,
    heldout_result_count=len(test_results),
    successful_backtests=len(viable),
)
actual_model = runtime_snapshot.get("actual_model_id") or "(not loaded)"
hf_meta = runtime_snapshot.get("token_status", {}).get("hf", {})
hf_state = "present" if hf_meta.get("present") else "missing"
hf_source = hf_meta.get("source", "none")
configured_families = runtime_snapshot.get("configured_method_families", ROADMAP_METHOD_FAMILIES)
executed_families = runtime_snapshot.get("executed_method_families", EXECUTED_METHOD_FAMILIES)
deferred_families = runtime_snapshot.get("deferred_method_families", DEFERRED_METHOD_FAMILIES)
active_execution_scope = runtime_snapshot.get("active_execution_scope", ACTIVE_EXECUTION_SCOPE)

md_text = f"""# AutoResearch v2 \u2014 Momentum Alpha Discovery

**Run:** {RUN_ID} | profile={RUN_PROFILE} | scope={REPORT_SCOPE}
**Benchmark mode:** {BENCHMARK_MODE}
**Configured model:** {CONFIGURED_MODEL_ID}
**Actual model loaded:** {actual_model}
**LLM stage:** enabled={runtime_snapshot.get("llm_stage_enabled")} | loaded={runtime_snapshot.get("llm_stage_loaded")} | executed={runtime_snapshot.get("llm_stage_executed")} | calls={runtime_snapshot.get("llm_calls")}
**HF token:** {hf_state} (source={hf_source})
**Universe:** {len(close_all.columns)} US equities | \
{close_all.index.min().date()} \u2192 {close_all.index.max().date()}
**Train:** through {TRAIN_END} | **Test (held-out):** after

## Method-family scope
- configured roadmap families: {", ".join(configured_families)}
- executed in this stage: {", ".join(executed_families)}
- deferred or not executed in this stage: {", ".join(deferred_families)}
- active execution scope: {active_execution_scope}
- note: LSTM, tabular ML, GBT, and combination studies are deferred in this pure-AutoResearch notebook

## Loop stats
- total batches: {N_BATCHES}
- log entries: {len(log)}
- viable backtests: {len(viable)}
- failures: {len(fail)}
- viable rate: {len(viable)/max(len(log),1):.0%}
- objective mode: `{RUNTIME_STATE.get("objective_mode", OBJECTIVE_MODE)}`
- primary objective: `{RUNTIME_STATE.get("primary_objective_label", PRIMARY_OBJECTIVE_LABEL)}`
- structured candidates recovered: {int(RUNTIME_STATE.get("structured_candidate_count", 0) or 0)}
- repair batches: {int(RUNTIME_STATE.get("repair_batches", 0) or 0)}
- format failures: {int(RUNTIME_STATE.get("format_failures", 0) or 0)}
- duplicate rejections: {int(RUNTIME_STATE.get("duplicate_rejections", 0) or 0)}
- semantic-family rejections: {int(RUNTIME_STATE.get("semantic_family_rejections", 0) or 0)}
- loop status: `{RUNTIME_STATE.get("loop_status", "unknown")}`
- stop reason: `{RUNTIME_STATE.get("loop_stop_reason", "n/a")}`
- stop detail: `{RUNTIME_STATE.get("loop_stop_detail", "n/a")}`
- best train primary Sharpe: **{best_train['sharpe']:+.2f}** \
(iter {best_train['iter']})

## Output files
- summary: `{SUMMARY_MD.name}`
- runtime metadata: `{RUNTIME_METADATA_FILE.name}`
- run manifest: `{RUN_MANIFEST_FILE.name}`
- research log: `{RESEARCH_LOG.name}`

## Reflection memo
{load_memo()}

## Best on held-out test
"""
if best_test:
    md_text += f"""- iter {best_test['iter']}: \
train primarySh={best_test['train_sharpe']:+.2f} \u2192 \
test primarySh=**{best_test['test_sharpe']:+.2f}**
- test primary AnnRet: {best_test['test_ret']:+.1%} | \
test primary DD: {best_test['test_dd']:+.1%} | \
beta: {best_test['test_beta']:+.2f}
- test benchmark-spread Sh: {best_test.get('test_bench_spread', 0.0):+.2f}
- walk-forward median primarySh: {best_test['wf_median']:+.2f}
- overfit gap (primarySh): {best_test['train_sharpe'] - best_test['test_sharpe']:+.2f}

### Hypothesis
{best_test['hypothesis']}

### Code
```python
{best_test['code']}
```
"""
else:
    md_text += "- no signal survived held-out evaluation.\\n"

sync_runtime_metadata(
    summary_pending=False,
    summary_written=True,
    summary_file=str(SUMMARY_MD),
    best_train_iter=best_train.get("iter"),
    best_test_iter=(best_test.get("iter") if best_test else None),
)
SUMMARY_MD.write_text(md_text)
print(md_text)
''')


# ═════════════════════════════════════════════════════════════════════════════
# Write notebook
# ═════════════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {
            "accelerator": "gpu",
            "language": "python",
            "isInternetEnabled": True,
            "isGpuEnabled": True,
            "isPrivate": True,
        },
    },
    "cells": CELLS,
}

cell_source_replacements = [
    ("metric_cell_8.py", lambda src: "def _series_metrics(series):" in src and "def backtest(signal_df" in src),
    ("metric_cell_18.py", lambda src: src.startswith("BASELINE_RESULTS = []") and "def selection_score" in src),
    ("metric_cell_28.py", lambda src: "## AutoResearch Adherence" in src and "heldout_rule_status" in src),
]
for c in nb["cells"]:
    src = c.get("source", "")
    src_text = src if isinstance(src, str) else "".join(src)
    for filename, predicate in cell_source_replacements:
        if predicate(src_text):
            c["source"] = Path(filename).read_text(encoding="utf-8").rstrip()
            break

# normalise source to list[str] (ipynb format)
for c in nb["cells"]:
    s = c["source"]
    if isinstance(s, str):
        lines = s.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1]
        c["source"] = lines
for i, c in enumerate(nb["cells"], start=1):
    c.setdefault("id", f"cell-{i:03d}")

OUT_PATH = "autoresearch_v2.ipynb"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    nb_node = nbformat.from_dict(nb)
    _, nb_node = normalize(nb_node)
    nbformat.write(nb_node, f)
print(f"wrote {OUT_PATH} — {len(CELLS)} cells")



