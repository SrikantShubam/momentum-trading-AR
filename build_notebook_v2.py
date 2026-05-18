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
from pathlib import Path
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

def _resolve_token(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value, f"env:{name}"
    return None, "none"

def token_presence_text(name, value, source):
    state = "present" if value else "missing"
    return f"{name}: {state} (source={source})"

HF_TOKEN, HF_TOKEN_SOURCE = _resolve_token("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

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
ACTIVE_EXECUTION_SCOPE = "deterministic/classical/autoresearch only"
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
CONFIGURED_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
# CONFIGURED_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"   # faster, still good
MODEL_ID = CONFIGURED_MODEL_ID
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
CONFIGURED_FALLBACK_MODEL_ID = FALLBACK_MODEL_ID
ACTIVE_MODEL_ID = None

# ── Loop config ──
N_BATCHES       = 25          # each batch → 1 LLM call → up to 3 candidates
REFLECT_EVERY   = 5           # reflection memo every N batches
SANDBOX_TIMEOUT = 30          # seconds per signal execution
TRAIN_END       = "2022-12-31"
RUN_LLM_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_STAGE"), default=False)
RUN_MOE_STAGE   = _env_truthy(os.getenv("AUTORESEARCH_RUN_MOE_STAGE"), default=False)
RUN_BNB_MODEL_LOAD = _env_truthy(os.getenv("AUTORESEARCH_ENABLE_BNB_LOAD"), default=False)
RUN_LLM_SMOKE = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)
RUN_HELDOUT_EVAL = True
RUN_REPORTS = True
REFLECTION_ENABLED = bool(RUN_LLM_STAGE and REFLECT_EVERY > 0)

RUNTIME_STATE = {
    "run_id": RUN_ID,
    "run_profile": RUN_PROFILE,
    "report_scope": REPORT_SCOPE,
    "benchmark_mode": BENCHMARK_MODE,
    "configured_method_families": list(ROADMAP_METHOD_FAMILIES),
    "executed_method_families": list(EXECUTED_METHOD_FAMILIES),
    "deferred_method_families": list(DEFERRED_METHOD_FAMILIES),
    "family_execution_status": dict(FAMILY_EXECUTION_STATUS),
    "active_execution_scope": ACTIVE_EXECUTION_SCOPE,
    "configured_model_id": CONFIGURED_MODEL_ID,
    "configured_fallback_model_id": CONFIGURED_FALLBACK_MODEL_ID,
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
''')


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4 — Backtester
# ═════════════════════════════════════════════════════════════════════════════
md("""## Cell 4 — Backtester with annual consistency

Vectorised pandas. Positions = `signal.shift(1)` (no lookahead).
Equal-weight across assets. 5 bps per-turn transaction cost.
Reports both aggregate Sharpe and **annual consistency** (fraction of
calendar years with positive active Sharpe).
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

    bench  = ret.mean(axis=1)          # equal-weight long-only
    active = net - bench

    eq        = (1 + net).cumprod()
    eq_active = (1 + active).cumprod()

    ann_r  = net.mean()    * 252;  ann_v  = net.std()    * np.sqrt(252)
    ann_ra = active.mean() * 252;  ann_va = active.std() * np.sqrt(252)
    sharpe     = float(ann_r  / ann_v)  if ann_v  > 0 else 0.0
    sharpe_act = float(ann_ra / ann_va) if ann_va > 0 else 0.0

    cov  = np.cov(net.values, bench.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0

    # ── annual consistency ──
    annual_sh = []
    for yr in sorted(active.index.year.unique()):
        a = active[active.index.year == yr]
        if len(a) < 50:
            continue
        v = a.std() * np.sqrt(252)
        annual_sh.append(float(a.mean() * 252 / v) if v > 0 else 0.0)
    consistency = (sum(1 for s in annual_sh if s > 0)
                   / max(len(annual_sh), 1))

    return {
        "sharpe_active": sharpe_act,
        "sharpe_raw":    sharpe,
        "ann_active":    float(ann_ra),
        "ann_return":    float(ann_r),
        "beta":          beta,
        "max_dd":        float((eq / eq.cummax() - 1).min()),
        "max_dd_active": float((eq_active / eq_active.cummax() - 1).min()),
        "hit_rate":      float((net > 0).mean()),
        "avg_turnover":  float(turnover.mean()),
        "equity":        eq,
        "equity_active": eq_active,
        "annual_sharpes": annual_sh,
        "consistency":    consistency,
        "min_annual":     min(annual_sh) if annual_sh else 0.0,
    }

# ── sanity check: 20-day cross-sectional momentum ──
_cs = close_train.pct_change(20).rank(axis=1, pct=True) * 2 - 1
_cs = _cs.sub(_cs.mean(axis=1), axis=0)
_m  = backtest(_cs, close_train)
print(f"sanity: Sh_active={_m['sharpe_active']:+.2f}  "
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

def run_signal_code(code_str, close_df, volume_df, timeout=SANDBOX_TIMEOUT):
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
            result[0] = ns["signal"](close_df.copy(), volume_df.copy())
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

def detect_lookahead(code_str, close_df, volume_df,
                     split_frac=0.6, seed=0):
    """Shuffle future data and check if past-slice signal changes."""
    sig_real, err = run_signal_code(code_str, close_df, volume_df, timeout=20)
    if err is not None:
        return False, err
    T = int(len(close_df) * split_frac)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(close_df) - T)
    c2, v2 = close_df.copy(), volume_df.copy()
    c2.iloc[T:] = close_df.iloc[T:].values[perm]
    v2.iloc[T:] = volume_df.iloc[T:].values[perm]
    sig_perm, err = run_signal_code(code_str, c2, v2, timeout=20)
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

if torch.cuda.device_count() == 0 and SHOULD_LOAD_LLM_MODEL:
    update_runtime_state(llm_stage_error="no_cuda_gpu")
    raise RuntimeError("No CUDA GPU detected. This notebook is configured for T4 sessions.")

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
def llm(messages, max_new_tokens=1200, temperature=0.8, top_p=0.95):
    ensure_model_loaded()
    update_runtime_state(
        llm_stage_executed=True,
        llm_calls=int(RUNTIME_STATE.get("llm_calls", 0)) + 1,
        actual_model_id=ACTIVE_MODEL_ID or MODEL_ID,
    )
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=16_000).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()
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

Write: def signal(close, volume) -> pd.DataFrame
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
def signal(close, volume):
    ...
    return result
```

=== CANDIDATE 2 ===
HYPOTHESIS: <one sentence>
```python
def signal(close, volume):
    ...
    return result
```

=== CANDIDATE 3 ===
HYPOTHESIS: <one sentence>
```python
def signal(close, volume):
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

def log_summary_for_prompt(log, top_k=8, code_lines=12):
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
    cands = extract_candidates(response)
    if not cands:
        append_log({"batch": batch_id, "iter": _COUNTER[0],
                     "error": "no_candidates_extracted",
                     "raw": response[:500]})
        _COUNTER[0] += 1
        print(f"  [b{batch_id:02d}] no candidates extracted")
        return

    for cand in cands:
        it = _COUNTER[0]; _COUNTER[0] += 1
        t0 = time.time()

        # 1. run signal
        sig_df, err = run_signal_code(cand["code"], close_df, volume_df)
        if err is not None:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "hypothesis": cand["hypothesis"],
                         "code": cand["code"], "error": err})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] {err[:80]}")
            continue

        # 2. validate signal distribution
        ok, why = validate_signal(sig_df)
        if not ok:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "hypothesis": cand["hypothesis"],
                         "code": cand["code"],
                         "error": f"DEGENERATE: {why}"})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] DEGENERATE: {why}")
            continue

        # 3. lookahead check
        ok, why = detect_lookahead(cand["code"], close_df, volume_df)
        if not ok:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "hypothesis": cand["hypothesis"],
                         "code": cand["code"],
                         "error": f"REJECT: {why}"})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] REJECT: {why[:70]}")
            continue

        # 4. backtest
        try:
            m = backtest(sig_df, close_df)
        except Exception as e:
            append_log({"batch": batch_id, "iter": it,
                         "cand_idx": cand["idx"],
                         "hypothesis": cand["hypothesis"],
                         "code": cand["code"],
                         "error": f"backtest: {e}"})
            print(f"  [b{batch_id:02d}.c{cand['idx']}] backtest error: {e}")
            continue

        entry = {
            "batch": batch_id, "iter": it, "cand_idx": cand["idx"],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hypothesis": cand["hypothesis"],
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
        append_log(entry)
        print(f"  [b{batch_id:02d}.c{cand['idx']}] "
              f"Sh={m['sharpe_active']:+.2f}  beta={m['beta']:+.2f}  "
              f"cons={m['consistency']:.0%}  DD={m['max_dd_active']:+.1%}")


def do_reflection(batch_id):
    """Model reviews its full research log and writes a memo."""
    log = load_log()
    good = sorted([e for e in log if e.get("sharpe") is not None],
                  key=lambda e: -e["sharpe"])
    fails = [e for e in log if e.get("error")]

    if not good:
        return   # nothing to reflect on yet

    succ_lines = []
    for e in good[:12]:
        succ_lines.append(
            f"iter {e['iter']}  Sh={e['sharpe']:+.2f}  "
            f"beta={e.get('beta',0):+.2f}  "
            f"cons={e.get('consistency',0):.0%}  "
            f"hyp: {e.get('hypothesis','')[:120]}")

    fail_lines = []
    for e in fails[-10:]:
        fail_lines.append(
            f"iter {e.get('iter','?')}: {str(e['error'])[:150]}")

    best = good[0]
    best_info = (f"iter {best['iter']}  Sh={best['sharpe']:+.2f}  "
                 f"beta={best.get('beta',0):+.2f}\\n"
                 f"    {best.get('hypothesis','')}")

    msg = REFLECTION_PROMPT.format(
        successes="\\n".join(succ_lines),
        failure_summary="\\n".join(fail_lines) if fail_lines else "(none)",
        best_info=best_info,
    )
    try:
        memo = llm(
            [{"role": "system",
              "content": "You are a senior quantitative researcher."},
             {"role": "user", "content": msg}],
            max_new_tokens=500, temperature=0.3)
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
        log  = load_log()
        memo = load_memo()

        user_msg = USER_PROMPT_TMPL.format(
            iter_n=b + 1,
            n_batches=n_batches,
            log_summary=log_summary_for_prompt(log),
            memo=memo,
            failures=failures_for_prompt(log),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        t0 = time.time()
        try:
            response = llm(messages, max_new_tokens=1200,
                           temperature=0.85)
        except Exception as e:
            append_log({"batch": b, "error": f"llm: {e}"})
            print(f"[b{b:02d}] LLM generation failed: {e}")
            continue
        t_gen = round(time.time() - t0, 1)
        print(f"[b{b:02d}] generated in {t_gen}s  "
              f"({len(response)} chars)")

        # wait for prev batch, then submit this one
        if pending is not None:
            pending.result()
        pending = executor.submit(
            process_batch, b, response, close_df, volume_df, t_gen)

        gc.collect(); torch.cuda.empty_cache()

        # ── periodic reflection ──
        if (b + 1) % REFLECT_EVERY == 0:
            if pending is not None:
                pending.result()
                pending = None
            do_reflection(b)
            gc.collect(); torch.cuda.empty_cache()

    if pending is not None:
        pending.result()
    executor.shutdown(wait=True)
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
        sig, err = run_signal_code(code_str, sub_c, sub_v, timeout=20)
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
    good = [e for e in log if e.get("sharpe") is not None]
    top  = sorted(good, key=lambda e: -e["sharpe"])[:top_k]
    results = []
    for e in top:
        ok, why = detect_lookahead(e["code"], close_test, volume_test)
        if not ok:
            print(f"  iter {e['iter']}: REJECTED on test data \u2014 {why[:80]}")
            continue
        sig, err = run_signal_code(e["code"], close_test, volume_test,
                                   timeout=25)
        if err:
            print(f"  iter {e['iter']}: test run failed \u2014 {err[:80]}")
            continue
        m  = backtest(sig, close_test)
        wf = walk_forward(e["code"], close_test, volume_test)
        wf_sh = [w["sharpe_active"] for w in wf if "sharpe_active" in w]
        results.append({
            "iter":         e["iter"],
            "hypothesis":   e.get("hypothesis", ""),
            "train_sharpe": e["sharpe"],
            "test_sharpe":  m["sharpe_active"],
            "test_raw":     m["sharpe_raw"],
            "test_beta":    m["beta"],
            "test_ret":     m["ann_active"],
            "test_dd":      m["max_dd_active"],
            "wf_median":    float(np.median(wf_sh)) if wf_sh else 0.0,
            "wf_min":       float(min(wf_sh))       if wf_sh else 0.0,
            "wf_windows":   wf,
            "equity":       m["equity_active"],
            "code":         e["code"],
        })
    return results

test_results = evaluate_on_test()
print(f"\\n{'iter':>4} {'train_Sh':>9} {'test_Sh':>8} {'beta':>7} "
      f"{'wf_med':>7} {'test_ret':>8} {'test_dd':>8}")
for r in test_results:
    print(f"{r['iter']:>4} {r['train_sharpe']:>+9.2f} "
          f"{r['test_sharpe']:>+8.2f} {r['test_beta']:>+7.2f} "
          f"{r['wf_median']:>+7.2f} {r['test_ret']:>+8.1%} "
          f"{r['test_dd']:>+8.1%}")

if test_results:
    best = max(test_results, key=lambda r: r["test_sharpe"])
    BEST_CODE.write_text(
        f"# iter {best['iter']}  train_Sh={best['train_sharpe']:+.2f}  "
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
best_train = (max(good, key=lambda e: e["sharpe"])
              if good else {"iter": -1, "sharpe": 0, "hypothesis": "(none)"})
best_test  = (max(test_results, key=lambda r: r["test_sharpe"])
              if test_results else None)
runtime_snapshot = sync_runtime_metadata(
    summary_pending=True,
    heldout_result_count=len(test_results),
    successful_backtests=len(good),
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
- successful backtests: {len(good)}
- failures: {len(fail)}
- success rate: {len(good)/max(len(log),1):.0%}
- best train active Sharpe: **{best_train['sharpe']:+.2f}** \
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
train Sh={best_test['train_sharpe']:+.2f} \u2192 \
test Sh=**{best_test['test_sharpe']:+.2f}**
- test AnnRet: {best_test['test_ret']:+.1%} | \
test DD: {best_test['test_dd']:+.1%} | \
beta: {best_test['test_beta']:+.2f}
- walk-forward median Sh: {best_test['wf_median']:+.2f}
- overfit gap: {best_test['train_sharpe'] - best_test['test_sharpe']:+.2f}

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
