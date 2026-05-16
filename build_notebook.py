"""Build autoresearch_kaggle.ipynb — proper AutoResearch loop for momentum trading."""
import json

CELLS = []

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": text})

# ─────────────────────────────────────────────────────────────────────────────
md("""# AutoResearch — Momentum Alpha Discovery at 7B (Q4)

A 7B-parameter code model acts as a **quantitative researcher**: it proposes
alpha signals as Python functions, we backtest them in a sandbox, and it reads
its own research log (sorted by Sharpe) to decide what to try next.

**This is NOT prompt optimisation.** The search space is *code*; the objective
is *Sharpe on held-out price data*; memory across iterations is a real research log.

**Kaggle setup:** Accelerator = **GPU T4 x2**, Internet = **On**, Persistence = Files only.
""")

# ─── CELL 1: install ─────────────────────────────────────────────────────────
md("## Cell 1 — Install deps")
code('''!pip -q install yfinance bitsandbytes "transformers>=4.44" "accelerate>=0.33"''')

# ─── CELL 2: imports + paths ─────────────────────────────────────────────────
md("## Cell 2 — Imports, seeds, paths")
code('''import os, sys, json, re, time, random, gc, io, ast
import traceback, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
RESEARCH_LOG = OUT / "research_log.jsonl"
PRICES_CACHE = OUT / "prices.parquet"
BEST_CODE    = OUT / "best_signal.py"
SHARPE_PLOT  = OUT / "sharpe_progress.png"
EQUITY_PLOT  = OUT / "equity_curves.png"
SUMMARY_MD   = OUT / "experiment_summary.md"

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
''')

# ─── CELL 3: data loader ─────────────────────────────────────────────────────
md("""## Cell 3 — Data: yfinance OHLCV, cached

20 liquid US equities, 2015-2024. Cached to `/kaggle/working/prices.parquet`
after first download so reruns don't hit yfinance again.
Split: 2015-2022 = train (loop sees), 2023-2024 = held-out.""")
code('''TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","WMT",
           "XOM","UNH","HD","PG","KO","PFE","CSCO","INTC","BA","DIS"]
START      = "2015-01-01"
END        = "2024-12-31"
TRAIN_END  = "2022-12-31"

def load_prices():
    if PRICES_CACHE.exists():
        df = pd.read_parquet(PRICES_CACHE)
        print(f"cached: {df.shape}  ({df.index.min().date()} → {df.index.max().date()})")
        return df
    print("downloading yfinance…")
    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(TICKERS, start=START, end=END,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                raise ValueError("empty frame")
            break
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError(f"yfinance failed: {last_err}")

    close  = raw["Close"].dropna(how="all").ffill().dropna()
    volume = raw["Volume"].reindex(close.index).fillna(0)
    common = close.columns.intersection(volume.columns)
    close, volume = close[common], volume[common]
    df = pd.concat({"close": close, "volume": volume}, axis=1)
    df.to_parquet(PRICES_CACHE)
    print(f"saved: {close.shape}")
    return df

prices_all = load_prices()
close_all  = prices_all["close"].astype(float)
volume_all = prices_all["volume"].astype(float)

mask = close_all.index <= pd.Timestamp(TRAIN_END)
close_train, close_test   = close_all[mask], close_all[~mask]
volume_train, volume_test = volume_all[mask], volume_all[~mask]

print(f"train: {close_train.shape}  test: {close_test.shape}")
print(f"tickers: {list(close_all.columns)}")
''')

# ─── CELL 4: backtester ──────────────────────────────────────────────────────
md("""## Cell 4 — Backtester

Vectorised pandas. Position = `signal.shift(1)` (no lookahead). Equal-weight
across assets. 5 bps per-turn transaction cost. Annualised Sharpe on daily returns.""")
code('''def backtest(signal_df, close_df, cost_bps=5.0):
    signal_df = signal_df.reindex(close_df.index).reindex(columns=close_df.columns)
    signal_df = signal_df.astype(float).fillna(0).clip(-1, 1)
    position  = signal_df.shift(1).fillna(0)

    ret       = close_df.pct_change().fillna(0)
    gross     = (position * ret).mean(axis=1)
    turnover  = position.diff().abs().mean(axis=1).fillna(0)
    cost      = turnover * (cost_bps / 10_000.0)
    net       = gross - cost

    # Benchmark: equal-weight long-only (market beta)
    bench  = ret.mean(axis=1)
    active = net - bench            # alpha component

    equity        = (1 + net).cumprod()
    equity_active = (1 + active).cumprod()

    ann_r  = net.mean()    * 252
    ann_v  = net.std()     * np.sqrt(252)
    sharpe = float(ann_r / ann_v) if ann_v > 0 else 0.0

    ann_ra = active.mean() * 252
    ann_va = active.std()  * np.sqrt(252)
    sharpe_active = float(ann_ra / ann_va) if ann_va > 0 else 0.0

    # Market beta via covariance
    cov   = np.cov(net.values, bench.values)
    beta  = float(cov[0,1] / cov[1,1]) if cov[1,1] > 0 else 0.0

    dd   = float((equity        / equity.cummax()        - 1).min())
    dd_a = float((equity_active / equity_active.cummax() - 1).min())
    hit  = float((net > 0).mean())

    return {
        "sharpe_active": sharpe_active,    # PRIMARY metric now
        "sharpe_raw":    sharpe,
        "ann_active":    float(ann_ra),
        "ann_return":    float(ann_r),
        "beta":          beta,
        "max_dd":        dd,
        "max_dd_active": dd_a,
        "hit_rate":      hit,
        "avg_turnover":  float(turnover.mean()),
        "total_return":  float(equity.iloc[-1] - 1),
        "equity":        equity,
        "equity_active": equity_active,
    }

# sanity: 20d cross-sectional momentum (zero-mean by construction → pure alpha test)
_cs = close_train.pct_change(20).rank(axis=1, pct=True) * 2 - 1
_cs = _cs.sub(_cs.mean(axis=1), axis=0)   # zero-sum cross-section
_m  = backtest(_cs, close_train)
print(f"sanity: active Sh={_m['sharpe_active']:+.2f}  raw Sh={_m['sharpe_raw']:+.2f}  "
      f"beta={_m['beta']:+.2f}  DD={_m['max_dd_active']:+.1%}")
''')

# ─── CELL 5: sandbox ─────────────────────────────────────────────────────────
md("""## Cell 5 — Sandbox executor

AST-parse + forbidden-string check + restricted builtins + thread-based timeout.
Works from any thread (main or worker).""")
code('''FORBIDDEN = [
    "import os", "import sys", "import subprocess", "import socket", "import shutil",
    "import requests", "import urllib", "open(", "__import__", "eval(", "exec(",
    "compile(", "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "input(", "quit(", "exit(", "__builtins__", "__class__", "__bases__",
    "__subclasses__", "pathlib", "Path(",
]

ALLOWED_IMPORTS = {"numpy", "np", "pandas", "pd", "math"}

def validate_code(code_str):
    low = code_str.lower()
    for bad in FORBIDDEN:
        if bad.lower() in low:
            return False, f"forbidden token: {bad!r}"
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"disallowed import: {n.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"disallowed import from: {node.module}"
    if "def signal" not in code_str:
        return False, "no `def signal` function defined"
    return True, "ok"

SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ["len","range","min","max","abs","sum","int","float","bool","list",
             "dict","tuple","set","str","True","False","None","enumerate","zip",
             "sorted","reversed","map","filter","round","isinstance","type",
             "any","all","print"]
    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}

import threading as _th

def run_signal_code(code_str, close_df, volume_df, timeout=30):
    """Run signal code with a thread-based timeout (works from any thread)."""
    ok, msg = validate_code(code_str)
    if not ok:
        return None, f"VALIDATION: {msg}"

    ns = {"np": np, "pd": pd, "__builtins__": SAFE_BUILTINS}
    result_holder = [None, None]  # [output, error_string]

    def _target():
        try:
            exec(code_str, ns)
            if "signal" not in ns or not callable(ns["signal"]):
                result_holder[1] = "no callable `signal` after exec"
                return
            out = ns["signal"](close_df.copy(), volume_df.copy())
            result_holder[0] = out
        except Exception as e:
            result_holder[1] = f"{type(e).__name__}: {e}"

    t = _th.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, f"TIMEOUT (>{timeout}s)"
    if result_holder[1] is not None:
        return None, result_holder[1]
    out = result_holder[0]

    if not isinstance(out, pd.DataFrame):
        return None, f"returned {type(out).__name__}, need DataFrame"
    if out.shape != close_df.shape:
        return None, f"shape {out.shape} != close {close_df.shape}"
    return out, None

def validate_signal(sig_df):
    """Reject degenerate signals. Returns (ok, reason)."""
    s = sig_df.clip(-1, 1).fillna(0)
    if s.std().mean() < 0.05:
        return False, "near-constant signal (std<0.05)"
    if s.mean().abs().mean() > 0.6:
        return False, f"directionally biased (|mean|={s.mean().abs().mean():.2f})"
    if s.std(axis=1).mean() < 0.1:
        return False, "no cross-sectional spread"
    long_frac  = (s >  0.1).mean().mean()
    short_frac = (s < -0.1).mean().mean()
    if min(long_frac, short_frac) < 0.08:
        return False, f"not long-short (long={long_frac:.2f} short={short_frac:.2f})"
    return True, "ok"

def detect_lookahead(code_str, close_df, volume_df, split_frac=0.6, seed=0):
    """
    Run signal on real data, then on data where rows AFTER split are shuffled.
    If past-slice output changes, the code uses future info.
    """
    sig_real, err = run_signal_code(code_str, close_df, volume_df, timeout=25)
    if err is not None:
        return False, err
    T = int(len(close_df) * split_frac)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(close_df) - T)
    c2 = close_df.copy();  v2 = volume_df.copy()
    c2.iloc[T:] = close_df.iloc[T:].values[perm]
    v2.iloc[T:] = volume_df.iloc[T:].values[perm]
    sig_perm, err = run_signal_code(code_str, c2, v2, timeout=25)
    if err is not None:
        return False, f"shuffle_run: {err}"
    a = sig_real.iloc[:T].fillna(0).values
    b = sig_perm.iloc[:T].fillna(0).values
    diff = float(np.abs(a - b).mean())
    if diff > 1e-6:
        return False, f"LOOKAHEAD detected (past diff={diff:.4f})"
    return True, "ok"

# quick self-test
_test_code = """
def signal(close, volume):
    return close.pct_change(20).rank(axis=1, pct=True) * 2 - 1
"""
_s, _e = run_signal_code(_test_code, close_train, volume_train)
print("sandbox self-test:", "OK" if _e is None else f"FAIL: {_e}")
''')

# ─── CELL 6: model ───────────────────────────────────────────────────────────
md("""## Cell 6 — Load Qwen2.5-Coder-7B-Instruct (4-bit)

Code-specialised 7B. Q4 via bitsandbytes → ~4.5 GB VRAM. Fits T4 easily.""")
code('''MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto"
)
model.eval()
print("model loaded. VRAM:",
      f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
''')

# ─── CELL 7: LLM gen ─────────────────────────────────────────────────────────
md("## Cell 7 — LLM generation helper")
code('''@torch.inference_mode()
def llm(messages, max_new_tokens=900, temperature=0.8, top_p=0.95):
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=10_000).to(model.device)
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

# ─── CELL 8: research prompt ────────────────────────────────────────────────
md("""## Cell 8 — Research prompt + code extraction

The researcher sees a *sorted log* of best attempts plus recent *failures*.
This is the AutoResearch part: memory is real, ranked, and narrative.""")
code('''SYSTEM_PROMPT = """You are a quantitative researcher. Your job is to discover alpha signals for a long-short momentum trading strategy on US equities.

You are given:
- `close`: pd.DataFrame of daily adjusted close prices (rows=dates, columns=tickers)
- `volume`: pd.DataFrame of daily volumes (same shape)

You write Python functions that return a signal in [-1, 1] per (date, ticker):
  +1 = strongest long, -1 = strongest short, 0 = flat.
Signals are traded next day (shift(1) applied for you — no lookahead needed).

Hard constraints:
- Use ONLY numpy (np) and pandas (pd). No other imports.
- Function signature: def signal(close, volume) -> pd.DataFrame
- Return shape MUST equal close.shape.
- Keep the function under 40 lines.

CRITICAL — lookahead bias will auto-reject your signal:
- NEVER use full-series statistics: volume.mean(), close.std(), close.quantile(),
  close.mean(), volume.median() — these aggregate over the ENTIRE series including
  future dates. They leak information.
- Use ONLY rolling/expanding windows with an explicit lookback:
  close.pct_change(N), close.rolling(N).mean(), close.rolling(N).std(),
  close.ewm(span=N).mean(), close.pct_change().rolling(N).std()
- Cross-sectional ops at a single date are fine: close.rank(axis=1),
  row.mean(axis=1), row.std(axis=1) — these only use same-date info.
- Do NOT use close.shift(-k) with negative k. That pulls future data.

CRITICAL — signal must be genuinely long-short:
- Return values must have BOTH positive and negative entries on most days.
- A near-constant signal (always +1, always 0) will be rejected.
- If you use rank(), subtract 0.5 BEFORE normalising: (rank - 0.5) not sign(rank - 0.5).
- Example correct pattern:
    r = close.pct_change(20).rank(axis=1, pct=True)   # [0,1]
    out = (r - 0.5) * 2                               # [-1,1], zero mean
  Then optionally multiply by a magnitude factor.

Good heuristics:
- Cross-sectional rank on past returns (winners minus losers), zero-sum per day
- Volatility scaling: signal / close.pct_change().rolling(60).std()
- Multi-horizon blend: mean of 1m, 3m, 6m momentum ranks
- 12-1 momentum: pct_change(252) - pct_change(21)  (skip last month)
- Rolling volume filter: mask where volume < volume.rolling(60).mean()
"""

USER_PROMPT_TMPL = """Your research log so far (best Sharpe first — these are YOUR prior experiments on training data):

{log_summary}
{failures}

Iteration batch: {iter_n}

Propose THREE NEW signals, each exploring a genuinely different hypothesis. Do not submit trivial variants of the same idea — make them substantively different (e.g. different time horizon, different ranking method, different filter, different regime).

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

_CAND_RE = re.compile(r"===\\s*CANDIDATE\\s*(\\d+)\\s*===(.*?)(?====\\s*CANDIDATE|\\Z)", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"```(?:python)?\\s*\\n(.*?)```", re.DOTALL)
_HYP_RE  = re.compile(r"HYPOTHESIS:\\s*(.+?)(?:\\n|$)", re.IGNORECASE)

def extract_candidates(text):
    """Return list of dicts: {idx, hypothesis, code}. Up to 3."""
    results = []
    for m in _CAND_RE.finditer(text):
        idx = int(m.group(1))
        body = m.group(2)
        cm = _CODE_RE.search(body)
        if not cm: continue
        hm = _HYP_RE.search(body)
        results.append({
            "idx": idx,
            "hypothesis": hm.group(1).strip() if hm else "",
            "code": cm.group(1).strip(),
        })
    # Fallback: if no CANDIDATE markers, extract all code blocks
    if not results:
        for i, cm in enumerate(_CODE_RE.finditer(text)):
            results.append({"idx": i+1, "hypothesis": "", "code": cm.group(1).strip()})
    return results[:3]
''')

# ─── CELL 9: research log ────────────────────────────────────────────────────
md("## Cell 9 — Research log (append-only JSONL, used as memory)")
code('''def append_log(entry):
    with open(RESEARCH_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\\n")

def load_log():
    if not RESEARCH_LOG.exists(): return []
    out = []
    for line in open(RESEARCH_LOG):
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except: pass
    return out

def log_summary_for_prompt(log, top_k=6, code_lines=18):
    good = [e for e in log if e.get("sharpe") is not None]
    if not good:
        return "(empty — this is your first experiment)"
    # sort by ACTIVE sharpe (alpha), not raw
    top = sorted(good, key=lambda e: -e["sharpe"])[:top_k]
    lines = ["(metrics are market-NEUTRAL: Sharpe is alpha vs equal-weight long-only benchmark)"]
    for i, e in enumerate(top):
        lines.append(
            f"#{i+1}  iter {e['iter']}  activeSh={e['sharpe']:+.2f}  "
            f"rawSh={e.get('sharpe_raw', 0):+.2f}  beta={e.get('beta', 0):+.2f}  "
            f"AnnAlpha={e['ann_return']:+.2%}  DD={e['max_dd']:+.1%}"
        )
        if e.get("hypothesis"):
            lines.append(f"    HYP: {e['hypothesis'][:160]}")
        code_snip = "\\n".join("    " + ln for ln in e["code"].splitlines()[:code_lines])
        lines.append(code_snip)
        if len(e["code"].splitlines()) > code_lines:
            lines.append("    ...")
        lines.append("")
    return "\\n".join(lines)

def failures_for_prompt(log, n=5):
    """Surface REJECT/DEGENERATE errors prominently — these are teaching signals."""
    fails = [e for e in log if e.get("error")]
    # prioritise lookahead / degenerate rejections over runtime errors
    rejects  = [e for e in fails if "REJECT" in str(e.get("error","")) or
                                      "DEGENERATE" in str(e.get("error",""))]
    runtime  = [e for e in fails if e not in rejects]
    chosen   = rejects[-n:] + runtime[-2:]
    if not chosen: return ""
    lines = ["\\nFAILED attempts (DO NOT repeat these mistakes):"]
    for e in chosen:
        err = str(e["error"])[:200]
        lines.append(f"- iter {e.get('iter','?')}: {err}")
        if e.get("hypothesis"):
            lines.append(f"    was: {e['hypothesis'][:120]}")
    return "\\n".join(lines)
''')

# ─── CELL 10: the loop ───────────────────────────────────────────────────────
md("## Cell 10 — The AutoResearch loop")
code('''from concurrent.futures import ThreadPoolExecutor

N_BATCHES = 30          # each batch = 1 LLM call → up to 3 signals
SANDBOX_TIMEOUT = 25
_COUNTER = [0]           # global iter counter (each candidate gets unique iter id)

def _process_batch(batch_id, response, close_df, volume_df, t_gen):
    """CPU work: parse candidates, sandbox + backtest each, append to log."""
    cands = extract_candidates(response)
    if not cands:
        append_log({"batch": batch_id, "iter": _COUNTER[0],
                    "error": "no_candidates", "raw": response[:400]})
        _COUNTER[0] += 1
        print(f"[b{batch_id:02d}] no candidates extracted")
        return

    for cand in cands:
        it = _COUNTER[0]; _COUNTER[0] += 1
        t0 = time.time()
        sig_df, err = run_signal_code(cand["code"], close_df, volume_df,
                                      timeout=SANDBOX_TIMEOUT)
        if err is not None:
            append_log({"batch": batch_id, "iter": it, "cand_idx": cand["idx"],
                        "hypothesis": cand["hypothesis"], "code": cand["code"],
                        "error": err})
            print(f"[b{batch_id:02d}.c{cand['idx']}] {err[:80]}")
            continue

        ok, why = validate_signal(sig_df)
        if not ok:
            append_log({"batch": batch_id, "iter": it, "cand_idx": cand["idx"],
                        "hypothesis": cand["hypothesis"], "code": cand["code"],
                        "error": f"DEGENERATE: {why}"})
            print(f"[b{batch_id:02d}.c{cand['idx']}] DEGENERATE: {why}")
            continue

        ok, why = detect_lookahead(cand["code"], close_df, volume_df)
        if not ok:
            append_log({"batch": batch_id, "iter": it, "cand_idx": cand["idx"],
                        "hypothesis": cand["hypothesis"], "code": cand["code"],
                        "error": f"REJECT: {why}"})
            print(f"[b{batch_id:02d}.c{cand['idx']}] REJECT: {why[:70]}")
            continue

        try:
            m = backtest(sig_df, close_df)
        except Exception as e:
            append_log({"batch": batch_id, "iter": it, "cand_idx": cand["idx"],
                        "hypothesis": cand["hypothesis"], "code": cand["code"],
                        "error": f"backtest: {e}"})
            print(f"[b{batch_id:02d}.c{cand['idx']}] backtest: {e}")
            continue

        entry = {
            "batch": batch_id, "iter": it, "cand_idx": cand["idx"],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hypothesis": cand["hypothesis"], "code": cand["code"],
            "sharpe":        m["sharpe_active"],      # headline = alpha Sharpe
            "sharpe_raw":    m["sharpe_raw"],
            "ann_return":    m["ann_active"],
            "ann_return_raw": m["ann_return"],
            "beta":          m["beta"],
            "max_dd":        m["max_dd_active"],
            "max_dd_raw":    m["max_dd"],
            "hit_rate":      m["hit_rate"],
            "turnover":      m["avg_turnover"],
            "bt_dt": round(time.time()-t0, 1), "gen_dt": t_gen,
        }
        append_log(entry)
        print(f"[b{batch_id:02d}.c{cand['idx']}] activeSh={m['sharpe_active']:+.2f}  "
              f"rawSh={m['sharpe_raw']:+.2f}  beta={m['beta']:+.2f}  "
              f"DD={m['max_dd_active']:+.1%}  bt={entry['bt_dt']}s")

def run_research_loop(n_batches, close_df, volume_df):
    # Recover counter from existing log
    existing = load_log()
    _COUNTER[0] = max((e.get("iter", -1) for e in existing), default=-1) + 1
    start_batch = max((e.get("batch", -1) for e in existing), default=-1) + 1
    print(f"starting batch {start_batch}  (log has {len(existing)} entries, "
          f"counter at {_COUNTER[0]})")

    executor = ThreadPoolExecutor(max_workers=1)
    pending  = None   # Future for previous batch's CPU work

    for b in range(start_batch, n_batches):
        log = load_log()
        user_msg = USER_PROMPT_TMPL.format(
            iter_n=b,
            log_summary=log_summary_for_prompt(log),
            failures=failures_for_prompt(log),
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}]

        t_gen0 = time.time()
        try:
            response = llm(messages, max_new_tokens=1400, temperature=0.85)
        except Exception as e:
            append_log({"batch": b, "error": f"llm_gen: {e}"})
            print(f"[b{b:02d}] llm failed: {e}")
            continue
        t_gen = round(time.time()-t_gen0, 1)
        print(f"[b{b:02d}] gen {t_gen}s  len={len(response)}")

        # Wait for previous batch's CPU work to finish, then submit this one
        if pending is not None:
            pending.result()
        pending = executor.submit(_process_batch, b, response,
                                  close_df, volume_df, t_gen)

        gc.collect(); torch.cuda.empty_cache()

    # Drain final batch
    if pending is not None:
        pending.result()
    executor.shutdown(wait=True)

run_research_loop(N_BATCHES, close_train, volume_train)
''')

# ─── CELL 11: evaluate on held-out ───────────────────────────────────────────
md("""## Cell 11 — Held-out evaluation

Take the top-K train-Sharpe signals and re-run on the untouched 2023-2024 period.
**Train-vs-test Sharpe gap = overfitting measure.**""")
code('''TOP_K = 5
WF_WINDOWS = 4   # quarterly slices in the held-out period

def walk_forward(code_str, close_df, volume_df, n_windows=WF_WINDOWS):
    """Slice held-out into N chunks, backtest each independently."""
    N  = len(close_df)
    sz = N // n_windows
    per_window = []
    for w in range(n_windows):
        lo, hi = w*sz, (w+1)*sz if w < n_windows-1 else N
        sub_c = close_df.iloc[lo:hi]
        sub_v = volume_df.iloc[lo:hi]
        sig, err = run_signal_code(code_str, sub_c, sub_v, timeout=25)
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
        # re-verify lookahead one more time on the test data
        ok, why = detect_lookahead(e["code"], close_test, volume_test)
        if not ok:
            print(f"  iter {e['iter']}: REJECTED at test — {why[:80]}")
            continue
        sig, err = run_signal_code(e["code"], close_test, volume_test, timeout=25)
        if err:
            print(f"  iter {e['iter']}: test run failed — {err[:80]}")
            continue
        m  = backtest(sig, close_test)
        wf = walk_forward(e["code"], close_test, volume_test)
        wf_sharpes = [w["sharpe_active"] for w in wf if "sharpe_active" in w]
        wf_median  = float(np.median(wf_sharpes)) if wf_sharpes else 0.0
        wf_min     = float(min(wf_sharpes))       if wf_sharpes else 0.0
        results.append({
            "iter":            e["iter"],
            "hypothesis":      e.get("hypothesis", ""),
            "train_sharpe":    e["sharpe"],
            "test_sharpe":     m["sharpe_active"],
            "test_sharpe_raw": m["sharpe_raw"],
            "test_beta":       m["beta"],
            "test_ret":        m["ann_active"],
            "test_dd":         m["max_dd_active"],
            "wf_median":       wf_median,
            "wf_min":          wf_min,
            "wf_windows":      wf,
            "equity":          m["equity_active"],
            "code":            e["code"],
        })
    return results

test_results = evaluate_on_test()
print("\\n%4s %9s %8s %7s %7s %8s %8s" %
      ("iter","train_Sh","test_Sh","beta","wf_med","test_ret","test_dd"))
for r in test_results:
    print(f"{r['iter']:>4} {r['train_sharpe']:>+9.2f} {r['test_sharpe']:>+8.2f} "
          f"{r['test_beta']:>+7.2f} {r['wf_median']:>+7.2f} "
          f"{r['test_ret']:>+8.1%} {r['test_dd']:>+8.1%}")

# save the single best (by test Sharpe) code
if test_results:
    best = max(test_results, key=lambda r: r["test_sharpe"])
    BEST_CODE.write_text(
        f"# iter {best['iter']}  train_Sh={best['train_sharpe']:+.2f}  "
        f"test_Sh={best['test_sharpe']:+.2f}\\n"
        f"# HYPOTHESIS: {best['hypothesis']}\\n\\n{best['code']}\\n"
    )
    print(f"\\nbest (by test Sharpe): iter {best['iter']}  saved to {BEST_CODE.name}")
''')

# ─── CELL 12: plots ──────────────────────────────────────────────────────────
md("## Cell 12 — Plots")
code('''log = load_log()
iters   = [e["iter"] for e in log]
sharpes = [e.get("sharpe") for e in log]

running_best, cur = [], -np.inf
for s in sharpes:
    if s is not None and s > cur: cur = s
    running_best.append(cur if cur > -np.inf else np.nan)

plt.figure(figsize=(10, 5))
plt.scatter(iters, [s if s is not None else np.nan for s in sharpes],
            alpha=0.5, s=40, label="iter Sharpe")
plt.plot(iters, running_best, "-", linewidth=2, label="running best")
plt.axhline(0, ls="--", color="grey", alpha=0.4)
plt.xlabel("Iteration"); plt.ylabel("Train Sharpe")
plt.title("AutoResearch — alpha discovery progress")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(SHARPE_PLOT, dpi=140); plt.close()

plt.figure(figsize=(10, 5))
for r in test_results:
    r["equity"].plot(
        label=f"iter {r['iter']}  trainSh={r['train_sharpe']:+.1f} "
              f"testSh={r['test_sharpe']:+.1f}"
    )
plt.ylabel("Equity"); plt.title("Top-5 signals on held-out 2023-2024")
plt.legend(fontsize=8, loc="best"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(EQUITY_PLOT, dpi=140); plt.close()
print("plots saved")
''')

# ─── CELL 13: summary ────────────────────────────────────────────────────────
md("## Cell 13 — Human-readable summary")
code('''log = load_log()
good = [e for e in log if e.get("sharpe") is not None]
fail = [e for e in log if e.get("error")]
if good:
    best_train = max(good, key=lambda e: e["sharpe"])
else:
    best_train = {"iter": -1, "sharpe": 0, "hypothesis": "(none)"}

best_test = (max(test_results, key=lambda r: r["test_sharpe"])
             if test_results else None)

md_text = f"""# AutoResearch — Momentum Alpha Discovery

**Model:** Qwen2.5-Coder-7B-Instruct (4-bit, Kaggle T4)
**Universe:** {len(close_all.columns)} US equities | {close_all.index.min().date()} → {close_all.index.max().date()}
**Train:** through {TRAIN_END} | **Test (held-out):** after

## Loop stats
- total iterations: {len(log)}
- successful backtests: {len(good)}
- failures: {len(fail)}
- best train Sharpe: **{best_train["sharpe"]:+.2f}** (iter {best_train["iter"]})

## Best on held-out test
"""
if best_test:
    md_text += f"""- iter {best_test['iter']}: train Sh={best_test['train_sharpe']:+.2f} → test Sh=**{best_test['test_sharpe']:+.2f}**
- test AnnRet: {best_test["test_ret"]:+.1%} | test DD: {best_test["test_dd"]:+.1%}
- overfit gap (train − test Sharpe): {best_test["train_sharpe"] - best_test["test_sharpe"]:+.2f}

### Hypothesis
{best_test["hypothesis"]}

### Code
```python
{best_test["code"]}
```
"""
else:
    md_text += "- no signal survived held-out evaluation.\\n"

SUMMARY_MD.write_text(md_text)
print(md_text)
''')

# ─── write notebook ──────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "cells": CELLS,
}

# normalise source fields to list[str] with newlines
for c in nb["cells"]:
    s = c["source"]
    if isinstance(s, str):
        lines = s.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1]
        c["source"] = lines

OUT_PATH = "autoresearch_kaggle.ipynb"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT_PATH} — {len(CELLS)} cells")
