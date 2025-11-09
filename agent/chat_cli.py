#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CLI for DeepKEGG-Agent.
"""

import os, sys, json, shlex, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR    = PROJECT_ROOT / "agent"
CONFIGS_DIR  = PROJECT_ROOT / "configs"
LAST_CFG     = CONFIGS_DIR / "last_run_config.json"

# [EN] TODO: translate comment from Chinese.
STEP1_DESIGN = "agent.step1_design"
STEP2 = "agent.step2_nl"
STEP3 = "agent.step3_run"
STEP5 = "agent.step5_report"
# ------------------------------------

def _run(module_path: str, args: str = "", check: bool = True):
    """
    使用 'python -m' 在项目根目录执行一个模块。
    这能确保所有相对导入都正确工作。
    """
    cmd = f"python -m {module_path} {args}"
    print(f"\n$ (cd {PROJECT_ROOT}) {cmd}")
    # [EN] TODO: translate comment from Chinese.
    p = subprocess.run(cmd, shell=True, text=True, cwd=str(PROJECT_ROOT))
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}")
    return p.returncode

def _load_cfg():
    if not LAST_CFG.exists():
        return None
    try:
        return json.loads(LAST_CFG.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read config: {e}")
        return None

def _show_cfg(cfg: dict):
    if not cfg:
        print("No config to show."); return
    print("\n=== CURRENT CONFIG SUMMARY ===")
    print(f"task      : {cfg.get('task')}")
    print(f"disease   : {cfg.get('disease') or cfg.get('cancer')}")
    print(f"modalities: {cfg.get('modalities') or cfg.get('omics')}")
    print(f"model     : {cfg.get('model')}")
    cv = cfg.get("cv") or {}
    print(f"cv        : {cv.get('type')} k={cv.get('k')} stratified={cv.get('stratified')} seed={cv.get('seed')}")
    paths = cfg.get("paths") or {}
    print("paths     :")
    for k in ("clinical","mrna","mirna","snv","kegg_gmt","kegg_map_long"):
        if paths.get(k): print(f"  - {k:12s} {paths[k]}")
    print(f"output_root: {cfg.get('output_root')}")
    meta = cfg.get("_meta") or {}
    print("meta      :", {k: meta.get(k) for k in ("source","provider","created_utc")})
    print("==============================\n")

def _ensure_env():
    prov = os.environ.get("LLM_PROVIDER", "")
    strict = os.environ.get("REQUIRE_LLM", "")
    abs_on = os.environ.get("WANT_ABS_PATHS", "0")
    base   = os.environ.get("BASE_DIR", "/tmp/DeepKEGG-master")
    out    = os.environ.get("OUTPUT_ROOT", str((PROJECT_ROOT / "runs").as_posix()))
    codegen= os.environ.get("USE_LLM_CODEGEN","0")
    print(f"[env] REQUIRE_LLM={strict or '0'}  LLM_PROVIDER={prov or '-'}  WANT_ABS_PATHS={abs_on}  USE_LLM_CODEGEN={codegen}")
    if abs_on in {"1","true","TRUE"}:
        print(f"[env] BASE_DIR={base}")
        print(f"[env] OUTPUT_ROOT={out}")

def cmd_help():
    print("""
Commands:
  design <name> <desc> Design a new model with a given name and description.
  nl <free text>       Parse natural language with LLM (strict if REQUIRE_LLM=1).
  show                 Show current config summary.
  train                Run step3_run.py (training/evaluation).
  report               Run step5_report.py (HTML/PDF report).
  set abs on|off       Toggle WANT_ABS_PATHS (1/0). 'on' maps to /tmp/DeepKEGG-master by default.
  set provider <deepseek|openai>
  set key <sk-...>     Set DEEPSEEK_API_KEY or OPENAI_API_KEY based on provider.
  codegen on|off       Toggle LLM-driven estimator generation at training stage.
  env                  Print effective env toggles.
  exit | quit          Leave the chat.

Tips:
  • You can just type plain English or Chinese. I’ll interpret it with LLM.
  • Examples you can say:
      - design KEGG_GAT "a graph attention model on the global KEGG graph"
      - LIHC with SVM using mRNA+miRNA, 5-fold CV
      - 我想用 XGBoost 做 BRCA，模态 mRNA+SNV，英文报告
      - Predict PRAD with DeepKEGG (all omics)
""")

INTRO = (
    "Hi! I’m DeepKEGG-Agent — a conversational runner for multi-omics cancer prediction.\n"
    "I can parse your natural language to set up datasets, models (DeepKEGG/SVM/XGBoost/LR),\n"
    "cross-validation and reporting, then train and produce an evaluation report.\n"
)
FIRST_QUESTION = "Which cancer type would you like to forecast? (e.g., LIHC / BRCA / BLCA / PRAD / AML / WT)"

def _kickoff_if_needed():
    """On start: introduce the project, then ask for disease if no config exists yet."""
    print(INTRO)
    cfg = _load_cfg()
    disease = (cfg or {}).get("disease") or (cfg or {}).get("cancer")
    if not disease:
        print(FIRST_QUESTION)
        try:
            ans = input("> ").strip()
        except EOFError:
            return
        if ans:
            os.environ.setdefault("REQUIRE_LLM","1")
            print("[info] Got it. I’ll interpret that with LLM.")
            # [EN] TODO: translate comment from Chinese.
            _run(STEP2, args=shlex.quote(ans), check=True)
            _show_cfg(_load_cfg())

def main():
    print("DeepKEGG-Agent Chat CLI (English). Type 'help' for commands.")
    _ensure_env()
    _kickoff_if_needed()

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue

        args = shlex.split(line)
        cmd  = args[0].lower()

        if cmd in {"exit","quit"}:
            break
        if cmd == "help":
            cmd_help(); continue
        if cmd == "env":
            _ensure_env(); continue
        if cmd == "show":
            _show_cfg(_load_cfg()); continue

        if cmd == "design":
            if len(args) < 3:
                print("Usage: design <model_name> <description of the new model>")
                print("Example: design KEGG_GAT 'design a graph attention network based on KEGG'")
                continue
            model_name = args[1]
            description = " ".join(args[2:])
            # [EN] TODO: translate comment from Chinese.
            _run(STEP1_DESIGN, args=f"{shlex.quote(model_name)} {shlex.quote(description)}", check=True)
            print(f"\n[ok] New model '{model_name}' designed. You can now try to use it in a 'nl' or 'train' command.")
            continue
            
        if cmd == "codegen" and len(args) >= 2:
            val = args[1].lower()
            os.environ["USE_LLM_CODEGEN"] = "1" if val in {"on","1","true"} else "0"
            print(f"[ok] USE_LLM_CODEGEN={os.environ['USE_LLM_CODEGEN']}")
            continue

        if cmd == "set" and len(args) >= 3:
            sub = args[1].lower()
            if sub == "abs":
                val = args[2].lower()
                if val in {"on","1","true"}:
                    os.environ["WANT_ABS_PATHS"]="1"
                    os.environ.setdefault("BASE_DIR","/tmp/DeepKEGG-master")
                    os.environ.setdefault("OUTPUT_ROOT","/tmp/DeepKEGG-agent/runs")
                    print("[ok] WANT_ABS_PATHS=1")
                else:
                    os.environ["WANT_ABS_PATHS"]="0"
                    print("[ok] WANT_ABS_PATHS=0")
            elif sub == "provider":
                prov = args[2].lower()
                if prov not in {"deepseek","openai"}:
                    print("provider must be deepseek|openai"); continue
                os.environ["LLM_PROVIDER"] = prov
                print(f"[ok] LLM_PROVIDER={prov}")
            elif sub == "key":
                key = args[2]
                prov = os.environ.get("LLM_PROVIDER","").lower()
                if prov == "deepseek":
                    os.environ["DEEPSEEK_API_KEY"]=key
                    print("[ok] DEEPSEEK_API_KEY set")
                elif prov == "openai":
                    os.environ["OPENAI_API_KEY"]=key
                    print("[ok] OPENAI_API_KEY set")
                else:
                    print("Set provider first: set provider deepseek|openai")
            else:
                print("Unknown 'set' subcommand. Use: set abs on|off | set provider | set key")
            continue

        if cmd == "nl":
            if len(args) < 2:
                print("Usage: nl <free-text request>")
                continue
            nl = line[len("nl"):].strip()
            os.environ.setdefault("REQUIRE_LLM","1")
            # [EN] TODO: translate comment from Chinese.
            _run(STEP2, args=shlex.quote(nl), check=True)
            _show_cfg(_load_cfg())
            continue

        if cmd == "train":
            if not LAST_CFG.exists():
                print("[error] Config file not found. Please run a natural language command first (e.g., 'Run DeepKEGG on LIHC').")
                continue
            llm_provider = os.environ.get("LLM_PROVIDER", "N/A")
            # [EN] TODO: translate comment from Chinese.
            _run(STEP3, args=f"--cfg {LAST_CFG} --llm-provider {llm_provider}", check=True)
            print("[ok] training finished.")
            continue

        if cmd == "report":
            # [EN] TODO: translate comment from Chinese.
            _run(STEP5, check=True)
            print("[ok] report generated.")
            continue

        # [EN] TODO: translate comment from Chinese.
        print("[info] Treating input as natural-language request")
        os.environ.setdefault("REQUIRE_LLM","1")
        # [EN] TODO: translate comment from Chinese.
        _run(STEP2, args=shlex.quote(line), check=True)
        _show_cfg(_load_cfg())

if __name__ == "__main__":
    main()
