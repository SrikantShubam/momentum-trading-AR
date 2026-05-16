import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "autoresearch_v2_final.ipynb"

EXTRA_ARTIFACT_LINES = [
    '        OUT / "partial_run_report.json",',
    '        OUT / "model_ab_report.json",',
    '        OUT / "evolution_memory.json",',
    '        OUT / "evolution_program.json",',
    '        OUT / "run_manifest.json",',
]


def set_source(cell, text):
    cell["source"] = [line + "\n" for line in text.splitlines()]


def ensure_report_artifacts(src):
    missing = [line for line in EXTRA_ARTIFACT_LINES if line not in src]
    if not missing:
        return src, False
    insert = "\n".join(missing) + "\n"
    preferred_anchor = '        OUT / "evolution_tldr.md",\n'
    if preferred_anchor in src:
        return src.replace(preferred_anchor, preferred_anchor + insert, 1), True
    close_anchor = "    ])\n    wandb_log({"
    if close_anchor not in src:
        raise RuntimeError("Could not find W&B artifact list close anchor")
    return src.replace(close_anchor, insert + close_anchor, 1), True


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched_install = patched_config = patched_log = patched_report = False

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if src.startswith("!pip install") and "wandb" not in src:
            src = src.replace("yfinance bitsandbytes", "yfinance wandb bitsandbytes")
            set_source(cell, src.rstrip())
            patched_install = True
            continue
        if src.startswith("!pip install") and "wandb" in src:
            patched_install = True

        if "HF_TOKEN = None" in src and "WANDB_API_KEY" not in src:
            src = src.replace(
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n"
                "try:\n"
                "    import wandb\n"
                "except Exception:\n"
                "    wandb = None\n",
            )
            old_secret = '''HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    print("HF_TOKEN detected: authenticated HF Hub downloads enabled")
else:
    print("HF_TOKEN not set: using anonymous HF Hub access (lower rate limits)")'''
            new_secret = '''HF_TOKEN = None
WANDB_API_KEY = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
except Exception:
    HF_TOKEN = None
    WANDB_API_KEY = None
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not WANDB_API_KEY:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if HF_TOKEN:
    print("HF_TOKEN detected: authenticated HF Hub downloads enabled")
else:
    print("HF_TOKEN not set: using anonymous HF Hub access (lower rate limits)")
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ.setdefault("WANDB_SILENT", "true")
    print("WANDB_API_KEY detected: W&B logging enabled")
else:
    print("WANDB_API_KEY not set: W&B logging disabled")'''
            src = src.replace(old_secret, new_secret)

            wandb_block = '''

# Optional W&B telemetry. This must never be required for the research loop to run.
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "momentum trading")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "srikantsubham10000-na")
WANDB_RUN = None
WANDB_DISABLED_REASON = None


def _wandb_scalar(v):
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return None


def wandb_init_once():
    global WANDB_RUN, WANDB_DISABLED_REASON
    if WANDB_RUN is not None:
        return WANDB_RUN
    if not WANDB_API_KEY:
        WANDB_DISABLED_REASON = "missing_api_key"
        return None
    if wandb is None:
        WANDB_DISABLED_REASON = "wandb_import_failed"
        return None
    config = {
        "model_id": MODEL_ID,
        "fallback_model_id": FALLBACK_MODEL_ID,
        "seed": SEED,
        "selection_objective": globals().get("SELECTION_OBJECTIVE", "market_neutral_net_sharpe"),
        "run_baseline_sweep": RUN_BASELINE_SWEEP,
        "run_deterministic_search": RUN_DETERMINISTIC_SEARCH,
        "run_param_search": RUN_PARAM_SEARCH,
        "run_heldout_eval": RUN_HELDOUT_EVAL,
        "param_search_trials": PARAM_SEARCH_TRIALS,
        "train_end": TRAIN_END,
    }
    try:
        WANDB_RUN = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"autoresearch-v2-{time.strftime('%Y%m%d-%H%M%S')}",
            config=config,
            reinit=True,
        )
    except Exception as e:
        if " " in WANDB_PROJECT:
            try:
                WANDB_RUN = wandb.init(
                    entity=WANDB_ENTITY,
                    project=WANDB_PROJECT.replace(" ", "-"),
                    name=f"autoresearch-v2-{time.strftime('%Y%m%d-%H%M%S')}",
                    config=config,
                    reinit=True,
                )
            except Exception as e2:
                WANDB_DISABLED_REASON = f"init_failed:{type(e2).__name__}"
                print(f"W&B disabled: {WANDB_DISABLED_REASON}")
                return None
        else:
            WANDB_DISABLED_REASON = f"init_failed:{type(e).__name__}"
            print(f"W&B disabled: {WANDB_DISABLED_REASON}")
            return None
    return WANDB_RUN


def wandb_log(metrics, step=None):
    run = wandb_init_once()
    if run is None:
        return
    clean = {}
    for k, v in dict(metrics).items():
        sv = _wandb_scalar(v)
        if sv is not None:
            clean[str(k)] = sv
    if clean:
        wandb.log(clean, step=step)


def wandb_log_candidate(entry):
    if not isinstance(entry, dict):
        return
    step = entry.get("iter") if isinstance(entry.get("iter"), int) else None
    metrics = {"candidate/error": 1 if entry.get("error") else 0}
    for key in (
        "score", "train_sharpe", "raw_sharpe", "inv_sharpe", "beta", "turnover",
        "consistency", "ann_return", "dd", "wf_median", "wf_min",
        "raw_cs_std", "raw_long_frac", "raw_short_frac", "signal_activity",
    ):
        if isinstance(entry.get(key), (int, float, np.integer, np.floating)):
            metrics[f"candidate/{key}"] = float(entry[key])
    wandb_log(metrics, step=step)


def wandb_log_artifacts(paths):
    run = wandb_init_once()
    if run is None:
        return
    for p in paths:
        try:
            p = Path(p)
            if p.exists():
                wandb.save(str(p))
        except Exception:
            pass


wandb_init_once()
'''
            src = src.replace(
                'print(f"flags: baseline={RUN_BASELINE_SWEEP} deterministic={RUN_DETERMINISTIC_SEARCH} param_search={RUN_PARAM_SEARCH} heldout={RUN_HELDOUT_EVAL} reports={RUN_REPORTS}")\n',
                'print(f"flags: baseline={RUN_BASELINE_SWEEP} deterministic={RUN_DETERMINISTIC_SEARCH} param_search={RUN_PARAM_SEARCH} heldout={RUN_HELDOUT_EVAL} reports={RUN_REPORTS}")\n'
                + wandb_block,
            )
            set_source(cell, src.rstrip())
            patched_config = True
            continue
        if "WANDB_API_KEY" in src and "def wandb_log_candidate" in src:
            patched_config = True

        if "def append_log(entry):" in src and "wandb_log_candidate(entry)" not in src:
            src = src.replace(
                '        with open(RESEARCH_LOG, "a") as f:\n'
                '            f.write(json.dumps(entry, default=str) + "\\n")\n',
                '        with open(RESEARCH_LOG, "a") as f:\n'
                '            f.write(json.dumps(entry, default=str) + "\\n")\n'
                '    try:\n'
                '        if "wandb_log_candidate" in globals():\n'
                '            wandb_log_candidate(entry)\n'
                '    except Exception:\n'
                '        pass\n',
            )
            set_source(cell, src.rstrip())
            patched_log = True
            continue
        if "def append_log(entry):" in src and "wandb_log_candidate(entry)" in src:
            patched_log = True

        if "SUMMARY_MD.write_text(md_text)" in src and "wandb_log_artifacts" not in src:
            src = src.rstrip() + '''

if "wandb_log_artifacts" in globals():
    wandb_log_artifacts([
        RESEARCH_LOG,
        DETERMINISTIC_FILE,
        PARAM_SEARCH_FILE,
        BEST_CODE,
        SHARPE_PLOT,
        EQUITY_PLOT,
        SUMMARY_MD,
        OUT / "best_signal_ensemble.py",
        OUT / "evolution_summary.json",
        OUT / "evolution_deep_dive.md",
        OUT / "evolution_tldr.md",
        OUT / "partial_run_report.json",
        OUT / "model_ab_report.json",
        OUT / "evolution_memory.json",
        OUT / "evolution_program.json",
        OUT / "run_manifest.json",
    ])
    wandb_log({
        "final/adherence_score": adherence_score,
        "final/economic_success": 1 if economic_success else 0,
        "final/heldout_candidates": len(test_results),
    })
    if wandb is not None and WANDB_RUN is not None:
        wandb.finish()
'''
            set_source(cell, src)
            patched_report = True
            continue
        if "SUMMARY_MD.write_text(md_text)" in src and "wandb_log_artifacts" in src:
            src, did_patch = ensure_report_artifacts(src)
            if did_patch:
                set_source(cell, src.rstrip())
            patched_report = True

    missing = [
        name
        for name, ok in [
            ("install cell", patched_install),
            ("config cell", patched_config),
            ("append_log cell", patched_log),
            ("report artifact cell", patched_report),
        ]
        if not ok
    ]
    if missing:
        raise RuntimeError("Missing patch targets: " + ", ".join(missing))
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("patched W&B instrumentation", NB_PATH)


if __name__ == "__main__":
    main()
