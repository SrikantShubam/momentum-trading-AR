import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "autoresearch_v2_final.ipynb"

MODEL_AB_CONFIG_LINES = [
    "CODE_MODEL_CANDIDATES = [MODEL_ID, FALLBACK_MODEL_ID]",
    "RUN_MODEL_AB = False",
    'MODEL_AB_REPORT_FILE = OUT / "model_ab_report.json"',
    'BENCHMARK_MODE = True',
    'BENCHMARK_OBJECTIVE = "best_verified_momentum_2xt4"',
    'BENCHMARK_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution", "lstm_sharpe", "tabular_ml", "gbt"]',
    "APPROVED_WINNER_EDGE_OVER_DETERMINISTIC = 0.10",
    "CHRONOLOGICAL_HOLDOUT_SEGMENTS = 3",
    "ROLLING_HOLDOUT_SEGMENTS = 3",
    "ROLLING_HOLDOUT_MIN_POINTS = 126",
]

EXPANDED_MUTATIONS = (
    "span_tweak",
    "rank_normalization",
    "rank_norm",
    "vol_scaling",
    "vol_scale",
    "volume_gate",
    "regime_gate",
    "ts_momentum",
    "short_reversal",
    "vol_adjusted",
    "volume_confirm",
    "regime_momentum",
    "multi_factor",
)
PROMPT_MUTATION_TEXT = (
    "span_tweak, rank_normalization/rank_norm, vol_scaling/vol_scale, "
    "volume_gate, regime_gate, ts_momentum, short_reversal, vol_adjusted, "
    "volume_confirm, regime_momentum, multi_factor"
)
PROMPT_FAMILY_TEXT = "ewm|momentum|mean_reversion|volume|volatility|regime|multi_factor"


def set_source(cell, text):
    cell["source"] = [line + "\n" for line in text.splitlines()]


def read_metric_cell(name):
    return (ROOT / name).read_text(encoding="utf-8").rstrip()


def replace_function(src, name, replacement):
    pattern = rf"def {re.escape(name)}\([^\n]*\):\n(?:(?:    .*)?\n)*"
    src, count = re.subn(pattern, lambda _m: replacement.rstrip() + "\n\n", src, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace {name} block")
    return src


TOKEN_SETUP_BLOCK = '''HF_TOKEN = None
HF_TOKEN_SOURCE = "none"
WANDB_API_KEY = None
WANDB_API_KEY_SOURCE = "none"

def _env_truthy(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _resolve_secret(secret_name, *env_names):
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        value = user_secrets.get_secret(secret_name)
        if value:
            return value, f"kaggle:{secret_name}"
    except Exception:
        pass
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value, f"env:{env_name}"
    return None, "none"

def _token_presence_record(name, value, source):
    return {"name": name, "present": bool(value), "source": source}

def _token_presence_text(record):
    state = "present" if record.get("present") else "missing"
    return f"{record.get('name')}: {state} (source={record.get('source', 'none')})"

HF_TOKEN, HF_TOKEN_SOURCE = _resolve_secret("HF_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
WANDB_API_KEY, WANDB_API_KEY_SOURCE = _resolve_secret("WANDB_API_KEY", "WANDB_API_KEY")
HF_TOKEN_STATUS = _token_presence_record("HF_TOKEN", HF_TOKEN, HF_TOKEN_SOURCE)
WANDB_TOKEN_STATUS = _token_presence_record("WANDB_API_KEY", WANDB_API_KEY, WANDB_API_KEY_SOURCE)

print(_token_presence_text(HF_TOKEN_STATUS))
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ.setdefault("WANDB_SILENT", "true")
print(_token_presence_text(WANDB_TOKEN_STATUS))
'''

HONESTY_CONFIG_BLOCK = '''# Runtime profile / scoped artifacts
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
BENCHMARK_MODE = True
BENCHMARK_OBJECTIVE = "best_verified_momentum_2xt4"
BENCHMARK_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution", "lstm_sharpe", "tabular_ml", "gbt"]
ROADMAP_METHOD_FAMILIES = list(dict.fromkeys(BENCHMARK_METHOD_FAMILIES + ["combination_studies"]))
EXECUTED_METHOD_FAMILIES = ["deterministic", "classical_risk_managed", "autoresearch_evolution"]
DEFERRED_METHOD_FAMILIES = [
    family for family in ROADMAP_METHOD_FAMILIES
    if family not in EXECUTED_METHOD_FAMILIES
]
ACTIVE_RESULT_SOURCES = ["deterministic", "parameter_search_evolution"]
ACTIVE_EXECUTION_SCOPE = "deterministic/classical/autoresearch only"
FAMILY_EXECUTION_STATUS = {
    family: ("executed" if family in EXECUTED_METHOD_FAMILIES else "deferred")
    for family in ROADMAP_METHOD_FAMILIES
}
APPROVED_WINNER_EDGE_OVER_DETERMINISTIC = 0.10
CHRONOLOGICAL_HOLDOUT_SEGMENTS = 3
ROLLING_HOLDOUT_SEGMENTS = 3
ROLLING_HOLDOUT_MIN_POINTS = 126
RUN_PROFILE = (
    os.getenv("AUTORESEARCH_RUN_PROFILE",
              "benchmark" if BENCHMARK_MODE else "full_autoresearch").strip()
    or ("benchmark" if BENCHMARK_MODE else "full_autoresearch")
)
REPORT_SCOPE = f"{RUN_PROFILE}_{RUN_ID}"

def scoped_output_path(name, scoped=True):
    stem = f"{REPORT_SCOPE}__{name}" if scoped else name
    return OUT / stem

RESEARCH_LOG = scoped_output_path("research_log.jsonl")
LOG_FILE = RESEARCH_LOG
PRICES_CACHE = OUT / "prices.parquet"
BEST_CODE    = scoped_output_path("best_signal.py")
SHARPE_PLOT  = scoped_output_path("sharpe_progress.png")
EQUITY_PLOT  = scoped_output_path("equity_curves.png")
SUMMARY_MD   = scoped_output_path("experiment_summary.md")
MEMO_FILE    = scoped_output_path("research_memo.txt")
DETERMINISTIC_FILE = scoped_output_path("deterministic_results.json")
PARAM_SEARCH_FILE = scoped_output_path("parameter_search_results.json")
MODEL_AB_REPORT_FILE = scoped_output_path("model_ab_report.json")
RUNTIME_METADATA_FILE = scoped_output_path("runtime_metadata.json")
RUN_MANIFEST_FILE = scoped_output_path("run_manifest.json")
LATEST_RUNTIME_METADATA_FILE = OUT / "latest_runtime_metadata.json"
LATEST_RUN_MANIFEST_FILE = OUT / "latest_run_manifest.json"

# Model selection
CONFIGURED_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_ID = CONFIGURED_MODEL_ID
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
CONFIGURED_FALLBACK_MODEL_ID = FALLBACK_MODEL_ID
ACTIVE_MODEL_ID = None
CODE_MODEL_CANDIDATES = [CONFIGURED_MODEL_ID, CONFIGURED_FALLBACK_MODEL_ID]
RUN_MODEL_AB = False

# Loop config
N_BATCHES       = 6
REFLECT_EVERY   = 5
SANDBOX_TIMEOUT = 30
TRAIN_END       = "2022-12-31"

# Execution flags
RUN_BASELINE_SWEEP       = True
RUN_DETERMINISTIC_SEARCH = True
RUN_LLM_STAGE            = False
RUN_MOE_STAGE            = False
RUN_BNB_MODEL_LOAD       = _env_truthy(os.getenv("AUTORESEARCH_ENABLE_BNB_LOAD"), default=False)
RUN_LLM_SMOKE            = _env_truthy(os.getenv("AUTORESEARCH_RUN_LLM_SMOKE"), default=False)
RUN_PARAM_SEARCH         = True
RUN_HELDOUT_EVAL         = True
RUN_REPORTS              = True
if BENCHMARK_MODE:
    RUN_LLM_STAGE = False
    RUN_MOE_STAGE = False

# Structured parameter search
PARAM_SEARCH_TRIALS = 200
PARAM_SEARCH_SEED = SEED
PARAM_SEARCH_MODE = "random"
PARAM_SEARCH_SHORT_SPANS = [36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66]
PARAM_SEARCH_LONG_SPANS = [90, 95, 100, 105, 110, 120, 130, 140, 150]
PARAM_SEARCH_REGIME_THRESHOLDS = [18.0, 20.0, 22.0, 24.0, 28.0]
PARAM_SEARCH_VOL_WINDOWS = [10, 20, 30]
PARAM_SEARCH_VOL_GATE_WINDOWS = [10, 20, 40]

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
    "actual_model_id": None,
    "llm_stage_enabled": bool(RUN_LLM_STAGE),
    "moe_stage_enabled": bool(RUN_MOE_STAGE),
    "llm_smoke_enabled": bool(RUN_LLM_SMOKE),
    "llm_stage_loaded": False,
    "llm_stage_executed": False,
    "llm_calls": 0,
    "token_status": {
        "hf": HF_TOKEN_STATUS,
        "wandb": WANDB_TOKEN_STATUS,
    },
}

ARTIFACT_PATHS = {
    "research_log": str(RESEARCH_LOG),
    "best_code": str(BEST_CODE),
    "summary": str(SUMMARY_MD),
    "memo": str(MEMO_FILE),
    "deterministic": str(DETERMINISTIC_FILE),
    "parameter_search": str(PARAM_SEARCH_FILE),
    "model_ab_report": str(MODEL_AB_REPORT_FILE),
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
        f"baseline={RUN_BASELINE_SWEEP} deterministic={RUN_DETERMINISTIC_SEARCH} "
        f"param_search={RUN_PARAM_SEARCH} heldout={RUN_HELDOUT_EVAL} reports={RUN_REPORTS} "
        f"llm={RUN_LLM_STAGE} moe={RUN_MOE_STAGE} benchmark_mode={BENCHMARK_MODE}"
    )

def runtime_family_scope_summary():
    return (
        f"scope={ACTIVE_EXECUTION_SCOPE} | "
        f"executed={','.join(EXECUTED_METHOD_FAMILIES)} | "
        f"deferred={','.join(DEFERRED_METHOD_FAMILIES)} | "
        f"result_sources={','.join(ACTIVE_RESULT_SOURCES)}"
    )

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  {mem:.1f} GB")
print("runtime profile:", RUN_PROFILE, "| report scope:", REPORT_SCOPE)
print("configured model:", CONFIGURED_MODEL_ID,
      "| fallback:", CONFIGURED_FALLBACK_MODEL_ID)
print("stage flags:", runtime_stage_summary())
print("roadmap families:", ", ".join(ROADMAP_METHOD_FAMILIES))
print("family scope:", runtime_family_scope_summary())
sync_runtime_metadata()
'''

FINAL_MODEL_LOAD_CELL = '''# Clear GPU memory before optional model loading
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

REQUESTED_LLM_MODEL = bool(RUN_LLM_STAGE or RUN_MOE_STAGE or RUN_LLM_SMOKE)
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

n_gpus = torch.cuda.device_count()
per_gpu_cap = "11GiB" if n_gpus >= 2 else "13GiB"
max_memory = {i: per_gpu_cap for i in range(n_gpus)}
max_memory["cpu"] = "48GiB"
offload_dir = OUT / "hf_offload"
offload_dir.mkdir(exist_ok=True)

def _load_model(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config(),
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder=str(offload_dir),
        dtype=torch.float16,
        **hub_kwargs,
    )

def refresh_llm_runtime_status():
    status = {
        "configured_model_id": CONFIGURED_MODEL_ID,
        "fallback_model_id": CONFIGURED_FALLBACK_MODEL_ID,
        "llm_stage_enabled": bool(RUN_LLM_STAGE),
        "moe_stage_enabled": bool(RUN_MOE_STAGE),
        "llm_smoke_enabled": bool(RUN_LLM_SMOKE),
        "should_load": bool(SHOULD_LOAD_LLM_MODEL),
        "loaded": bool(tok is not None and model is not None),
        "active_model_id": ACTIVE_MODEL_ID,
        "benchmark_mode": BENCHMARK_MODE,
        "llm_calls": int(RUNTIME_STATE.get("llm_calls", 0)),
    }
    globals()["LLM_RUNTIME_STATUS"] = status
    update_runtime_state(llm_runtime_status=status)
    return status

def ensure_model_loaded():
    global tok, model, MODEL_ID, ACTIVE_MODEL_ID
    if tok is not None and model is not None:
        refresh_llm_runtime_status()
        return tok, model
    if not RUN_BNB_MODEL_LOAD:
        update_runtime_state(llm_stage_error="bnb_model_load_disabled")
        raise RuntimeError(
            "LLM model loading is disabled for this Kaggle run because the current "
            "CUDA/bitsandbytes stack can hard-crash the kernel. Set "
            "AUTORESEARCH_ENABLE_BNB_LOAD=1 only for a dedicated model-load test."
        )

    requested_model_id = CONFIGURED_MODEL_ID
    active_model_id = requested_model_id
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
    MODEL_ID = ACTIVE_MODEL_ID
    update_runtime_state(
        actual_model_id=ACTIVE_MODEL_ID,
        llm_stage_loaded=True,
        llm_stage_error=None,
        llm_load_strategy=load_strategy,
    )
    refresh_llm_runtime_status()
    print(f"model loaded: configured={CONFIGURED_MODEL_ID} actual={MODEL_ID} ({load_strategy})")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated / {total:.1f} GB total")
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
    refresh_llm_runtime_status()

def unload_llm_model():
    global tok, model, ACTIVE_MODEL_ID
    globals().pop("model", None)
    globals().pop("tok", None)
    model = None
    tok = None
    ACTIVE_MODEL_ID = None
    update_runtime_state(llm_stage_loaded=False, actual_model_id=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    refresh_llm_runtime_status()
    return True

def switch_llm_model(model_id):
    global MODEL_ID, ACTIVE_MODEL_ID, model, tok
    target = str(model_id)
    if not target:
        raise ValueError("model_id is required")

    current = globals().get("ACTIVE_MODEL_ID", globals().get("MODEL_ID"))
    if target == current and "model" in globals() and "tok" in globals():
        refresh_llm_runtime_status()
        return model

    previous = current
    unload_llm_model()
    try:
        ACTIVE_MODEL_ID = target
        tok = _load_tokenizer(ACTIVE_MODEL_ID)
        model = _load_model(ACTIVE_MODEL_ID)
        model.eval()
        MODEL_ID = ACTIVE_MODEL_ID
        update_runtime_state(
            actual_model_id=ACTIVE_MODEL_ID,
            llm_stage_loaded=True,
            llm_load_strategy="manual_switch",
        )
        refresh_llm_runtime_status()
        print(f"model switched: {MODEL_ID}")
        return model
    except Exception:
        unload_llm_model()
        if previous and previous != target:
            try:
                ACTIVE_MODEL_ID = previous
                tok = _load_tokenizer(ACTIVE_MODEL_ID)
                model = _load_model(ACTIVE_MODEL_ID)
                model.eval()
                MODEL_ID = ACTIVE_MODEL_ID
                update_runtime_state(
                    actual_model_id=ACTIVE_MODEL_ID,
                    llm_stage_loaded=True,
                    llm_load_strategy="restore_previous",
                )
                refresh_llm_runtime_status()
                print(f"model switch failed; restored: {MODEL_ID}")
            except Exception:
                unload_llm_model()
        raise
'''

FINAL_LLM_HELPER_CELL = '''# Lazy cache for system-prompt token IDs (order-safe across cells)
# This avoids NameError when this cell runs before SYSTEM_PROMPT is defined.
_SYS_IDS = None

def _ensure_sys_ids():
    global _SYS_IDS
    if _SYS_IDS is not None:
        return _SYS_IDS
    if "SYSTEM_PROMPT" not in globals():
        raise RuntimeError("SYSTEM_PROMPT is not defined yet. Run the prompt cell before generating.")
    _sys_text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        tokenize=False, add_generation_prompt=False,
    )
    _SYS_IDS = tok(_sys_text, return_tensors="pt",
                   add_special_tokens=False)["input_ids"].to("cuda:0")
    print(f"system prefix cached: {_SYS_IDS.shape[1]} tokens")
    return _SYS_IDS

@torch.inference_mode()
def llm(user_content, max_new_tokens=800, temperature=0.8, top_p=0.95):
    """user_content: plain string for the user turn.
    System prompt is prepended automatically from cached token IDs.
    """
    ensure_model_loaded()
    update_runtime_state(
        llm_stage_executed=True,
        llm_calls=int(RUNTIME_STATE.get("llm_calls", 0)) + 1,
        actual_model_id=ACTIVE_MODEL_ID or MODEL_ID,
    )
    refresh_llm_runtime_status()
    sys_ids = _ensure_sys_ids()
    user_text = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )
    user_ids = tok(user_text, return_tensors="pt",
                   add_special_tokens=False)["input_ids"].to("cuda:0")

    input_ids = torch.cat([sys_ids, user_ids], dim=1)
    if input_ids.shape[1] > 16_000:
        keep_user = 16_000 - sys_ids.shape[1]
        user_ids  = user_ids[:, -keep_user:]
        input_ids = torch.cat([sys_ids, user_ids], dim=1)

    attn = torch.ones_like(input_ids)
    out  = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0, input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

print("llm() ready (lazy system-prefix cache)")
'''

def patch_config_cell(src):
    changed = False
    token_pattern = (
        r'HF_TOKEN = None\nHF_TOKEN_SOURCE = "none"\nWANDB_API_KEY = None\nWANDB_API_KEY_SOURCE = "none"\n'
        r'(?:.*\n)*?print\(_token_presence_text\(WANDB_TOKEN_STATUS\)\)\n'
    )
    if "def _resolve_secret(" not in src:
        legacy_pattern = (
            r'HF_TOKEN = None\nWANDB_API_KEY = None\n'
            r'(?:.*\n)*?print\("WANDB_API_KEY not set: W&B logging disabled"\)\n'
        )
        src, count = re.subn(legacy_pattern, TOKEN_SETUP_BLOCK, src, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("Could not replace token setup block")
        changed = True

    config_pattern = (
        r'# \?\? Paths \?\?\nOUT = Path\("/kaggle/working"\); OUT\.mkdir\(exist_ok=True\)\n'
        r'(?:.*\n)*?print\(f"flags: baseline=\{RUN_BASELINE_SWEEP\} deterministic=\{RUN_DETERMINISTIC_SEARCH\} '
        r'param_search=\{RUN_PARAM_SEARCH\} heldout=\{RUN_HELDOUT_EVAL\} reports=\{RUN_REPORTS\}"\)\n'
    )
    if 'CONFIGURED_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"' not in src or "scoped_output_path" not in src:
        src, count = re.subn(config_pattern, HONESTY_CONFIG_BLOCK, src, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("Could not replace runtime/config block")
        changed = True

    if '"model_id": MODEL_ID,' in src:
        src = src.replace(
            '"model_id": MODEL_ID,\n',
            '"configured_model_id": CONFIGURED_MODEL_ID,\n'
            '        "configured_fallback_model_id": CONFIGURED_FALLBACK_MODEL_ID,\n'
            '        "benchmark_mode": BENCHMARK_MODE,\n'
            '        "run_profile": RUN_PROFILE,\n',
            1,
        )
        changed = True
    return src, changed


MODEL_SWITCH_HELPERS = '''

# Model A/B helpers. Safe to call even when no model is currently loaded.
def unload_llm_model():
    globals().pop("model", None)
    globals().pop("tok", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return True


def switch_llm_model(model_id):
    global MODEL_ID, ACTIVE_MODEL_ID, model, tok
    target = str(model_id)
    if not target:
        raise ValueError("model_id is required")

    current = globals().get("ACTIVE_MODEL_ID", globals().get("MODEL_ID"))
    if target == current and "model" in globals() and "tok" in globals():
        return model

    previous = current
    unload_llm_model()
    try:
        ACTIVE_MODEL_ID = target
        tok = _load_tokenizer(ACTIVE_MODEL_ID)
        model = _load_model(ACTIVE_MODEL_ID)
        model.eval()
        MODEL_ID = ACTIVE_MODEL_ID
        print(f"model switched: {MODEL_ID}")
        return model
    except Exception:
        unload_llm_model()
        if previous and previous != target:
            try:
                ACTIVE_MODEL_ID = previous
                tok = _load_tokenizer(ACTIVE_MODEL_ID)
                model = _load_model(ACTIVE_MODEL_ID)
                model.eval()
                MODEL_ID = ACTIVE_MODEL_ID
                print(f"model switch failed; restored: {MODEL_ID}")
            except Exception:
                unload_llm_model()
        raise


def refresh_llm_runtime_status():
    globals()["LLM_RUNTIME_STATUS"] = {
        "configured_model_id": globals().get("MODEL_ID"),
        "fallback_model_id": globals().get("FALLBACK_MODEL_ID"),
        "llm_stage_enabled": bool(globals().get("RUN_LLM_STAGE", False)),
        "moe_stage_enabled": bool(globals().get("RUN_MOE_STAGE", False)),
        "should_load": bool(globals().get("SHOULD_LOAD_LLM_MODEL", False)),
        "loaded": bool(globals().get("tok") is not None and globals().get("model") is not None),
        "active_model_id": globals().get("MODEL_ID"),
        "benchmark_mode": globals().get("BENCHMARK_MODE", False),
    }
    return globals()["LLM_RUNTIME_STATUS"]


refresh_llm_runtime_status()
'''


def patch_model_load_cell(src):
    normalized = FINAL_MODEL_LOAD_CELL.rstrip()
    return (src, False) if src.rstrip() == normalized else (normalized, True)


def patch_llm_helper_cell(src):
    normalized = FINAL_LLM_HELPER_CELL.rstrip()
    return (src, False) if src.rstrip() == normalized else (normalized, True)


def patch_artifact_path_cell(src):
    replacements = {
        'EVOLUTION_SUMMARY_FILE = OUT / "evolution_summary.json"': 'EVOLUTION_SUMMARY_FILE = scoped_output_path("evolution_summary.json")',
        'EVOLUTION_LINEAGE_FILE = OUT / "evolution_lineage.json"': 'EVOLUTION_LINEAGE_FILE = scoped_output_path("evolution_lineage.json")',
        'EVOLUTION_MEMORY_FILE = OUT / "evolution_memory.json"': 'EVOLUTION_MEMORY_FILE = scoped_output_path("evolution_memory.json")',
        'EVOLUTION_PROGRAM_FILE = OUT / "evolution_program.json"': 'EVOLUTION_PROGRAM_FILE = scoped_output_path("evolution_program.json")',
        'RUN_MANIFEST_FILE = OUT / "run_manifest.json"': 'RUN_MANIFEST_FILE = scoped_output_path("run_manifest.json")',
        'PARTIAL_REPORT_FILE = OUT / "partial_run_report.json"': 'PARTIAL_REPORT_FILE = scoped_output_path("partial_run_report.json")',
        'EVOLUTION_DEEP_DIVE_FILE = OUT / "evolution_deep_dive.md"': 'EVOLUTION_DEEP_DIVE_FILE = scoped_output_path("evolution_deep_dive.md")',
        'EVOLUTION_TLDR_FILE = OUT / "evolution_tldr.md"': 'EVOLUTION_TLDR_FILE = scoped_output_path("evolution_tldr.md")',
    }
    changed = False
    for old, new in replacements.items():
        if old in src:
            src = src.replace(old, new)
            changed = True
    return src, changed


REPORT_INTRO_OLD = '''md_text = f"""# AutoResearch v2 - Momentum Alpha Discovery

**Model:** {MODEL_ID} (4-bit, Kaggle T4x2)
**Universe:** {len(close_all.columns)} US equities | {close_all.index.min().date()} -> {close_all.index.max().date()}
**Train:** through {TRAIN_END} | **Test (held-out):** after
**Selection objective:** market-neutral net Sharpe (`SELECTION_OBJECTIVE={SELECTION_OBJECTIVE}`)
'''

REPORT_INTRO_NEW = '''runtime_metadata = _safe_json_dict(globals().get("RUNTIME_METADATA_FILE"))
configured_model_id = runtime_metadata.get("configured_model_id", globals().get("CONFIGURED_MODEL_ID", globals().get("MODEL_ID", "unknown")))
actual_model_id = runtime_metadata.get("actual_model_id") or "(not loaded)"
llm_enabled = bool(runtime_metadata.get("llm_stage_enabled", globals().get("RUN_LLM_STAGE", False)))
llm_loaded = bool(runtime_metadata.get("llm_stage_loaded", False))
llm_executed = bool(runtime_metadata.get("llm_stage_executed", False))
llm_calls = int(_float_or(runtime_metadata.get("llm_calls"), 0.0))
hf_meta = runtime_metadata.get("token_status", {}).get("hf", {})
hf_state = "present" if hf_meta.get("present") else "missing"
hf_source = hf_meta.get("source", "none")
run_id = runtime_metadata.get("run_id", globals().get("RUN_ID", "unknown"))
runtime_profile = runtime_metadata.get("run_profile", globals().get("RUN_PROFILE", "unknown"))
artifact_scope = runtime_metadata.get("report_scope", globals().get("REPORT_SCOPE", "unscoped"))
benchmark_mode = bool(runtime_metadata.get("benchmark_mode", globals().get("BENCHMARK_MODE", False)))
configured_families = runtime_metadata.get("configured_method_families", list(globals().get("ROADMAP_METHOD_FAMILIES", globals().get("BENCHMARK_METHOD_FAMILIES", []))))
executed_families = runtime_metadata.get("executed_method_families", list(globals().get("EXECUTED_METHOD_FAMILIES", [])))
deferred_families = runtime_metadata.get("deferred_method_families", list(globals().get("DEFERRED_METHOD_FAMILIES", [])))
active_execution_scope = runtime_metadata.get("active_execution_scope", globals().get("ACTIVE_EXECUTION_SCOPE", "deterministic/classical/autoresearch only"))

md_text = f"""# AutoResearch v2 - Momentum Alpha Discovery

**Run:** {run_id} | profile={runtime_profile} | scope={artifact_scope}
**Benchmark mode:** {benchmark_mode}
**Configured model:** {configured_model_id}
**Actual loaded model:** {actual_model_id}
**LLM stage:** enabled={llm_enabled} | loaded={llm_loaded} | executed={llm_executed} | calls={llm_calls}
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

## Runtime artifacts
- summary: `{SUMMARY_MD.name}`
- runtime metadata: `{RUNTIME_METADATA_FILE.name}`
- run manifest: `{RUN_MANIFEST_FILE.name}`
- research log: `{RESEARCH_LOG.name}`
'''


def patch_final_report_cell(src):
    changed = False
    if REPORT_INTRO_OLD in src:
        src = src.replace(REPORT_INTRO_OLD, REPORT_INTRO_NEW, 1)
        changed = True
    if "sync_runtime_metadata(" not in src and "SUMMARY_MD.write_text(md_text)" in src:
        src = src.replace(
            'if RUN_REPORTS or RUN_HELDOUT_EVAL:\n    SUMMARY_MD.write_text(md_text)\n',
            'if "sync_runtime_metadata" in globals():\n'
            '    sync_runtime_metadata(\n'
            '        summary_written=bool(RUN_REPORTS or RUN_HELDOUT_EVAL),\n'
            '        summary_file=str(SUMMARY_MD),\n'
            '        report_status=research_result_status,\n'
            '        best_test_iter=(best_test.get("iter") if best_test else None),\n'
            '        best_test_score=(_float_or(best_test.get("test_score")) if best_test else None),\n'
            '    )\n'
            'if RUN_REPORTS or RUN_HELDOUT_EVAL:\n    SUMMARY_MD.write_text(md_text)\n',
            1,
        )
        changed = True
    return src, changed


def dedupe_exact_lines(src, repeated_lines):
    seen = set()
    out = []
    changed = False
    for line in src.splitlines():
        if line in repeated_lines:
            if line in seen:
                changed = True
                continue
            seen.add(line)
        out.append(line)
    return "\n".join(out), changed


SIGNAL_FAMILIES_BLOCK = '''SIGNAL_FAMILIES = {
    "momentum": ["pct_change", "momentum", "12_1", "12-1", "trend", "ts_momentum", "time_series"],
    "mean_reversion": ["reversion", "reversal", "short_reversal", "bollinger", "zscore", "z_score", "mean_rev", "contrarian", "overbought"],
    "volume": ["volume", "vol_ratio", "turnover", "amihud", "volume_confirm", "liquidity"],
    "volatility": ["volatility", ".std()", "variance", "atr", "realised", "realized", "vol_adjusted", "vol_scale"],
    "ewm": ["ewm(", "ema", "exponential", "crossover", "fast / slow", "fast/slow"],
    "regime": ["vix", "tnx", "regime", "conditional", "high_vol", "rising", "state", "regime_momentum"],
    "multi_factor": ["combine", "average", "blend", "weight", "composite", "multi_factor", "multi factor", "ensemble"],
}'''

NORMALIZE_MUTATION_BLOCK = '''def normalize_mutation(raw):
    if not raw:
        return "span_tweak"
    key = raw.lower().strip()
    key = re.sub(r"[`*_#>:\\-.]+", "_", key)
    key = re.sub(r"\\s+", "_", key).strip("_")
    aliases = {
        "parameter_tweak": "span_tweak",
        "param_tweak": "span_tweak",
        "normalization": "rank_norm",
        "rank_normalization": "rank_norm",
        "vol_scaling": "vol_scale",
        "volatility_scaling": "vol_scale",
        "regime_filter": "regime_gate",
        "factor_blend": "multi_factor",
        "multi_factor_blend": "multi_factor",
        "trend_momentum": "ts_momentum",
        "time_series_momentum": "ts_momentum",
        "volume_confirmation": "volume_confirm",
        "regime_filter_momentum": "regime_momentum",
    }
    key = aliases.get(key, key)
    return key if key in VALID_MUTATIONS else "span_tweak"'''

DETERMINISTIC_FAMILY_BLOCK = '''def deterministic_family_for_mutation(mutation_type):
    key = normalize_mutation(mutation_type)
    mapping = {
        "plain": ("ewm", "ewm"),
        "span_tweak": ("ewm", "ewm"),
        "rank_norm": ("ewm", "ewm"),
        "rank_normalization": ("ewm", "ewm"),
        "vol_scale": ("ewm_volscale", "ewm"),
        "vol_scaling": ("ewm_volscale", "ewm"),
        "volume_gate": ("ewm_volume", "ewm"),
        "regime_gate": ("ewm_regime", "ewm"),
        "ts_momentum": ("momentum", "momentum"),
        "short_reversal": ("mean_reversion", "mean_reversion"),
        "vol_adjusted": ("volatility_momentum", "momentum"),
        "volume_confirm": ("volume_momentum", "momentum"),
        "regime_momentum": ("regime_momentum", "momentum"),
        "multi_factor": ("multi_factor", "multi_factor"),
    }
    return mapping.get(key, ("ewm", "ewm"))'''

CLUSTER_ID_BLOCK = '''def cluster_id_for_signal(base_family, mutation_type, short_span, long_span, params=None):
    params = normalize_aux_params(params)
    key = normalize_mutation(mutation_type)
    family, mapped_base = deterministic_family_for_mutation(key)
    prefix = family or base_family or mapped_base or "signal"
    if short_span is None or long_span is None:
        return f"{prefix}:generic"
    if key in ("regime_gate", "regime_momentum"):
        thr = params.get("vix_threshold")
        return f"{prefix}:{short_span}:{long_span}:{thr}" if thr is not None else f"{prefix}:{short_span}:{long_span}"
    if key in ("vol_scale", "vol_scaling", "vol_adjusted", "multi_factor"):
        win = params.get("vol_window")
        return f"{prefix}:{short_span}:{long_span}:{win}" if win is not None else f"{prefix}:{short_span}:{long_span}"
    if key in ("volume_gate", "volume_confirm"):
        win = params.get("vol_gate_window")
        return f"{prefix}:{short_span}:{long_span}:{win}" if win is not None else f"{prefix}:{short_span}:{long_span}"
    if key == "ts_momentum":
        return f"{prefix}:{long_span}"
    if key == "short_reversal":
        return f"{prefix}:{short_span}"
    return f"{prefix}:{short_span}:{long_span}"'''


def patch_prompt_helper_cell(src):
    changed = False
    src, did_dedupe = dedupe_exact_lines(
        src,
        {
            "- Do NOT use np.sign(rank - 0.5), np.sign(rank), or any one-sided threshold that can collapse to all-long/all-short.",
            "- Prefer percentile ranks: r = feature.rank(axis=1, pct=True); out = (r - 0.5) * 2; then row-demean as above.",
        },
    )
    changed = changed or did_dedupe

    new_family_section = (
        "CRITICAL - signal family discipline:\n"
        f"- Allowed mutation types in this pass: {PROMPT_MUTATION_TEXT}.\n"
        "- Allowed factor families: ewm, momentum, mean_reversion, volume, volatility, regime, multi_factor.\n"
        "- Preserve the parent signal's core idea when mutating a winner anchor.\n"
        "- Use cross-sectional percentile ranks and trailing windows; no unsupported data sources.\n"
    )
    src, count = re.subn(
        r"CRITICAL - signal (?:must remain an EWM-backbone long-short signal|family discipline):\n"
        r"(?:- .*\n){3,5}",
        new_family_section,
        src,
        count=1,
    )
    changed = changed or count == 1

    replacements = {
        "Each candidate MUST preserve an EWM backbone.":
            "Each candidate MUST preserve the selected parent family's core structure; EWM parents should keep their EWM backbone.",
        "Use only these mutation types: span_tweak, rank_normalization, vol_scaling, volume_gate, regime_gate.":
            f"Use only these mutation types: {PROMPT_MUTATION_TEXT}.",
        "At most ONE candidate may use volume or regime conditioning, and only as a light modifier on the EWM signal.":
            "Volume, volatility, regime, reversal, trend, and multi-factor variants are allowed when they use trailing data and remain market-neutral.",
        "Do NOT propose unrelated families from scratch.":
            "Do NOT abandon the parent cluster without declaring and justifying the mutation family.",
        "MUTATION_TYPE: <span_tweak|rank_normalization|vol_scaling|volume_gate|regime_gate>":
            f"MUTATION_TYPE: <{PROMPT_MUTATION_TEXT.replace(', ', '|').replace('/', '|')}>",
        "FAMILY: <ewm|volume|volatility|regime>":
            f"FAMILY: <{PROMPT_FAMILY_TEXT}>",
        "Which EWM clusters and mutation types show the most promise? Cite score and train Sharpe values.":
            "Which clusters, factor families, and mutation types show the most promise? Cite score and train Sharpe values.",
        "What specific next EWM mutations should be tried?":
            "What specific next mutations should be tried?",
    }
    for old, new in replacements.items():
        if old in src and new not in src:
            src = src.replace(old, new)
            changed = True

    src, count = re.subn(
        r"SIGNAL_FAMILIES = \{\n(?:    .*\n)*?\}",
        SIGNAL_FAMILIES_BLOCK,
        src,
        count=1,
    )
    changed = changed or count == 1

    valid_mutations_line = "VALID_MUTATIONS = {" + ", ".join(repr(m) for m in EXPANDED_MUTATIONS) + "}"
    src, count = re.subn(r"VALID_MUTATIONS = \{[^\n]*\}", valid_mutations_line, src, count=1)
    changed = changed or count == 1

    src = replace_function(src, "normalize_mutation", NORMALIZE_MUTATION_BLOCK)
    src = replace_function(src, "deterministic_family_for_mutation", DETERMINISTIC_FAMILY_BLOCK)
    src = replace_function(src, "cluster_id_for_signal", CLUSTER_ID_BLOCK)
    return src.rstrip(), True or changed


PARAMETER_SEARCH_SUMMARY_BLOCK = '''def parameter_search_summary(rows):
    good = [r for r in rows if isinstance(r, dict) and r.get("score") is not None]
    robust = [r for r in good if r.get("robust_ok")]
    if not good:
        return "(parameter search has not produced any valid rows)"
    best = max(good, key=lambda r: r.get("score", -1e99))
    if "row_identity" in globals():
        best_id = row_identity(best, "parameter_search")
    else:
        best_id = (
            best.get("parent_id")
            or best.get("signature")
            or best.get("cluster_id")
            or best.get("program_hash")
            or "parameter_search:unknown"
        )
    score = float(best.get("score", 0.0) or 0.0)
    train_sharpe = float(best.get("train_sharpe", best.get("sharpe", 0.0)) or 0.0)
    wf_median = float(best.get("wf_median", 0.0) or 0.0)
    wf_min = float(best.get("wf_min", 0.0) or 0.0)
    lines = [
        f"rows={len(rows)} valid={len(good)} robust={len(robust)}",
        f"best={best_id} score={score:+.2f} trainSh={train_sharpe:+.2f} wf={wf_median:+.2f}/{wf_min:+.2f}",
    ]
    return "\\n".join(lines)'''


def patch_parameter_summary_cell(src):
    if "best_id =" in src and "best.get(\"parent_id\")" in src:
        return src, False
    return replace_function(src, "parameter_search_summary", PARAMETER_SEARCH_SUMMARY_BLOCK).rstrip(), True


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    metric_sources = {
        "backtest cell": ("metric_cell_8.py", lambda src: "def _series_metrics(series):" in src and "def backtest(signal_df" in src),
        "baseline/evolution cell": ("metric_cell_18.py", lambda src: src.startswith("BASELINE_RESULTS = []") and "def selection_score" in src),
        "held-out cell": ("metric_cell_24.py", lambda src: src.startswith("TOP_K = 5") and "def walk_forward" in src),
        "final report cell": ("metric_cell_28.py", lambda src: "SUMMARY_MD.write_text(md_text)" in src and "adherence_score" in src),
    }
    replacement_status = {name: False for name in metric_sources}
    patched_config = False
    patched_model_load = False
    patched_llm_helper = False
    patched_report = False
    patched_artifact_paths = False
    patched_executor = False
    patched_prompt = False
    patched_parameter_summary = False

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        replaced = False
        for name, (filename, matcher) in metric_sources.items():
            if matcher(src):
                src = read_metric_cell(filename)
                if name == "final report cell":
                    src, _ = patch_final_report_cell(src)
                    patched_report = True
                set_source(cell, src.rstrip())
                replacement_status[name] = True
                replaced = True
                break
        if replaced:
            continue

        if "MODEL_ID =" in src and "FALLBACK_MODEL_ID =" in src and "RUN_BASELINE_SWEEP" in src:
            src, did_patch = patch_config_cell(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_config = True
            continue

        if "def _load_model(model_id):" in src and "model loaded:" in src:
            src, did_patch = patch_model_load_cell(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_model_load = True
            continue

        if "def llm(user_content" in src and "system prefix cached:" in src:
            src, did_patch = patch_llm_helper_cell(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_llm_helper = True
            continue

        if "EVOLUTION_SUMMARY_FILE = OUT /" in src or "EVOLUTION_DEEP_DIVE_FILE = OUT /" in src:
            src, did_patch = patch_artifact_path_cell(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_artifact_paths = True
            continue

        if "def run_signal_code" in src and "def validate_signal" in src:
            src = src.replace(
                '    if out.shape != close_df.shape:\n'
                '        return None, f"shape {out.shape} != expected {close_df.shape}"\n'
                '    return out, None\n',
                '    if out.shape != close_df.shape:\n'
                '        return None, f"shape {out.shape} != expected {close_df.shape}"\n'
                '    ok_signal, sig_msg = validate_signal(out, expected_shape=close_df.shape)\n'
                '    if not ok_signal:\n'
                '        return None, f"SIGNAL_VALIDATION: {sig_msg}"\n'
                '    out = out.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)\n'
                '    return out, None\n',
            )
            src = src.replace(
                '    ok_signal, sig_msg = validate_signal(out)\n',
                '    ok_signal, sig_msg = validate_signal(out, expected_shape=close_df.shape)\n',
            )

            new_validator = '''def validate_signal(sig_df, expected_shape=None):
    s = sig_df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)
    if expected_shape is not None and s.shape != expected_shape:
        return False, f"shape {s.shape} != expected {expected_shape}"

    row_std = float(s.std(axis=1).mean())
    long_frac = float((s > 0.05).mean().mean())
    short_frac = float((s < -0.05).mean().mean())
    abs_row_mean = float(s.mean(axis=1).abs().mean())
    demeaned = s.sub(s.mean(axis=1), axis=0)
    active_frac = float((demeaned.abs().sum(axis=1) > 1e-8).mean())

    if row_std < 0.02:
        return False, f"near-constant cross-section (row_std={row_std:.4f})"
    if min(long_frac, short_frac) < 0.08:
        return False, f"not genuinely long-short (L={long_frac:.2f} S={short_frac:.2f})"
    if abs_row_mean > 0.50:
        return False, f"directionally biased before neutralization (abs_row_mean={abs_row_mean:.2f})"
    if active_frac < 0.50:
        return False, f"dead after market-neutral normalization (active_frac={active_frac:.2f})"
    return True, "ok"'''

            src, count = re.subn(
                r"def validate_signal\(sig_df(?:, expected_shape=None)?\):\n(?:.*\n)*?    return True, \"ok\"\n",
                new_validator + "\n",
                src,
                count=1,
            )
            if count != 1:
                raise RuntimeError("Could not replace validate_signal block")
            set_source(cell, src.rstrip())
            patched_executor = True
            continue

        if "SYSTEM_PROMPT = " in src and "VALID_MUTATIONS" in src and "deterministic_family_for_mutation" in src:
            src, _ = patch_prompt_helper_cell(src)
            set_source(cell, src.rstrip())
            patched_prompt = True
            continue

        if "def parameter_search_summary(rows):" in src:
            src, did_patch = patch_parameter_summary_cell(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_parameter_summary = True
            continue

    missing = [
        name
        for name, ok in [
            *replacement_status.items(),
            ("config/model A-B cell", patched_config),
            ("model load switch helpers", patched_model_load),
            ("llm helper cell", patched_llm_helper),
            ("report honesty patch", patched_report),
            ("artifact path scoping cells", patched_artifact_paths),
            ("executor validator cell", patched_executor),
            ("prompt/helper cell", patched_prompt),
            ("parameter search summary cell", patched_parameter_summary),
        ]
        if not ok
    ]
    if missing:
        raise RuntimeError("Missing patch targets: " + ", ".join(missing))

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("patched", NB_PATH)


if __name__ == "__main__":
    main()
