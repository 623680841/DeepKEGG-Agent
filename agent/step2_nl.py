#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-driven natural language -> structured config.
Includes a debug and auto-correction layer to ensure LLM's output
adheres to user's core intent.
"""

from __future__ import annotations
import os, sys, re, json
from pathlib import Path
from datetime import datetime
import typing as T

import requests
REQUIRE_LLM = os.environ.get("REQUIRE_LLM", "0").strip() in {"1","true","TRUE"}

# -------------------- project paths --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)
LAST_RUN_CONFIG = os.environ.get("LAST_RUN_CONFIG", str(CFG_DIR / "last_run_config.json"))

# -------------------- server absolute paths (opt-in) --------------------
WANT_ABS_PATHS = os.environ.get("WANT_ABS_PATHS", "0").strip() in {"1", "true", "TRUE"}
BASE_DIR = os.environ.get("BASE_DIR", "/tmp/DeepKEGG-master")
OUTPUT_ROOT_REL = str((PROJECT_ROOT / "runs").as_posix())
OUTPUT_ROOT_ABS = os.environ.get("OUTPUT_ROOT", "/tmp/DeepKEGG-agent/runs")

# -------------------- read NL --------------------
if len(sys.argv) > 1:
    NL = " ".join(sys.argv[1:]).strip()
else:
    try:
        NL = input("Enter request (e.g. 'Run BLCA with SVM using mRNA+miRNA+SNV, 10-fold CV. English report.'): ").strip()
    except EOFError:
        NL = ""

if not NL:
    print("No input. Example:\n  Run BLCA with SVM using mRNA+miRNA+SNV, 10-fold CV. English report.")
    sys.exit(1)

# -------------------- helpers --------------------
SYSTEM_PROMPT = """You are a data science planner. Always reply in English.
Return ONLY a compact JSON object with this schema:
{
  "task": "recurrence|survival|classification",
  "disease": "LIHC|BLCA|BRCA|PRAD|AML|WT|...",
  "modalities": ["mRNA","miRNA","SNV"],
  "model": "PathFinder|DeepKEGG|SVM|XGBoost|RandomForest|LR",
  "run_biomarker_discovery": false,  // <--- 新增这一行
  "cv": {"type":"kfold","k":10,"stratified":true,"seed":42},
  "paths": {
    "clinical": "data/<DISEASE>/response.csv",
    "mrna":     "data/<DISEASE>/mRNA_data.csv",
    "mirna":    "data/<DISEASE>/miRNA_data.csv",
    "snv":      "data/<DISEASE>/snv_data.csv",
    "id_column": "index",
    "label_column": "response"
  }
}
Rules:
- If the user explicitly names a model, keep it.
- Your primary job is to extract entities from the user's text into the JSON.
- Output pure JSON. No Markdown, no commentary.
"""

def _extract_json(text: str) -> dict:
    s = text.strip()
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if not m:
        m = re.search(r"```\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"[ERROR] LLM returned invalid JSON: {s}")
        return {}

# [EN] TODO: translate comment from Chinese.
def _chat_completion_llm(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    一个更通用的LLM调用函数，允许传入自定义的system_prompt。
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider not in {"deepseek","openai"}:
        raise RuntimeError("LLM_PROVIDER must be 'deepseek' or 'openai'")

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key: raise RuntimeError("DEEPSEEK_API_KEY is not set")
        url, model = "https://api.deepseek.com/chat/completions", os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else: # openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY is not set")
        url, model = "https://api.openai.com/v1/chat/completions", os.environ.get("OPENAI_MODEL", "gpt-4o")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},  # <-- 使用传入的system_prompt
            {"role": "user", "content": user_prompt}
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200: raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
    
    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content

# [EN] TODO: translate comment from Chinese.
def _debug_and_correct_config(config_draft: dict, user_prompt: str) -> dict:
    """
    一个由LLM驱动的Debug和修正模块，用来审查和修正配置草案。
    """
    print(">>> 启动LLM驱动的Debug与自动调整模块 (智能质检员)...")
    if not config_draft:
        print("[DEBUG] 配置草案为空，跳过质检。")
        return {}

    draft_str = json.dumps(config_draft, indent=2, ensure_ascii=False)
    
    correction_system_prompt = """You are a meticulous configuration verifier. Your task is to check if a generated JSON config strictly follows the user's original request.
- The user's request is the ultimate source of truth.
- Pay close attention to specific model names, column names, and file paths mentioned by the user.
- If the JSON config is already perfect, return it EXACTLY as is.
- If the JSON config has ANY deviation from the user's request, you MUST output a corrected version of the JSON.
- Your output MUST be ONLY the final, pure JSON object, without any commentary, explanations, or markdown fences."""

    correction_user_prompt = f"""Please verify the following JSON configuration draft against the user's original request.

[User's Original Request]:
{user_prompt}

[Generated JSON Draft to Verify]:
{draft_str}

Remember: If corrections are needed, output only the corrected JSON. If it's already correct, output the original JSON draft."""

    try:
        print("[DEBUG] 正在调用“智能质检员”LLM进行二次审查...")
        corrected_content = _chat_completion_llm(correction_user_prompt, system_prompt=correction_system_prompt)
        final_config = _extract_json(corrected_content)
        print(">>> “智能质检员”审查完成！")
        return final_config
    
    except Exception as e:
        print(f"[ERROR] “智能质检员”在审查过程中发生错误: {e}")
        print("      将返回未经审查的原始草案。")
        return config_draft

# [EN] TODO: translate comment from Chinese.
def _infer_hints_from_user_nl(nl: str) -> dict:
    t = nl.lower()
    
    # [EN] TODO: translate comment from Chinese.
    hints = {}

    # disease
    disease = None
    for c in ["lihc", "blca", "brca", "prad", "aml", "wt"]:
        if re.search(rf"\b{c}\b", t):
            disease = c.upper()
            break
    if disease:
        hints["disease"] = disease

    # model
    model = None
    model_map = {
        "pathfinder": "PathFinder", "randomforest": "RandomForest", 
        "xgboost": "XGBoost", "xgb": "XGBoost", "svm": "SVM", 
        "lr": "LR", "logistic": "LR", "deepkegg": "DeepKEGG", 
        "innovate": "INNOVATE_MODEL", # <-- 新增这一行
    }
    for k, v in model_map.items():
        if re.search(rf"\b{k}\b", t):
            model = v
            break
    if model:
        hints["model"] = model

    # modalities
    mods: T.List[str] = []
    if "mrna" in t: mods.append("mRNA")
    if "mirna" in t: mods.append("miRNA")
    if "snv" in t: mods.append("SNV")
    if mods:
        hints["modalities"] = mods
    
    # cv k
    cv_k = None
    m = re.search(r"(\d+)\s*-\s*fold|\b(\d+)\s*fold|\bcv\s*=\s*(\d+)", t)
    if m:
        cv_k = next(int(g) for g in m.groups() if g)
        hints["cv_k"] = cv_k

    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    biomarker_keywords = ["biomarker", "标志物", "重要特征", "importance score"]
    if any(keyword in t for keyword in biomarker_keywords):
       hints["run_biomarker_discovery"] = True
    # [EN] TODO: translate comment from Chinese.
    
    # [EN] TODO: translate comment from Chinese.
    return hints


def _apply_user_overrides(cfg: dict, hints: dict):
    if hints.get("disease"):    cfg["disease"]    = hints["disease"]
    if hints.get("model"):      cfg["model"]      = hints["model"]
    # [EN] TODO: translate comment from Chinese.
    if hints.get("run_biomarker_discovery"): cfg["run_biomarker_discovery"] = True
    # [EN] TODO: translate comment from Chinese.
    if hints.get("modalities"): cfg["modalities"] = hints["modalities"]
    if hints.get("modalities"): cfg["modalities"] = hints["modalities"]
    if hints.get("cv_k"):
        cv = cfg.get("cv") or {}
        cv["type"] = cv.get("type", "kfold")
        cv["k"] = hints["cv_k"]
        cv["stratified"] = cv.get("stratified", True)
        cv["seed"] = cv.get("seed", 42)
        cfg["cv"] = cv

def _absolutize_paths(cfg: dict, base_dir: str, output_root_abs: str):
    """Map relative 'data/<D>/*' and 'pathways/*' to server absolute paths under BASE_DIR."""
    disease = (cfg.get("disease") or cfg.get("cancer") or "LIHC").upper()
    p = cfg.setdefault("paths", {})
    def _abs(path: str) -> str:
        if not path: return path
        if path.startswith(base_dir): return path
        path = path.replace("<DISEASE>", disease)
        if path.startswith("data/"):
            return f"{base_dir}/{disease}_data/" + path.split("/", 2)[-1].replace("<DISEASE>/", "")
        if path.startswith("pathways/"):
            return f"{base_dir}/KEGG_pathways/" + path.split("/", 1)[-1]
        return path
    for k in ["clinical", "mrna", "mirna", "snv", "kegg_gmt", "kegg_map_long"]:
        if p.get(k): p[k] = _abs(p.get(k))
    p["id_column"] = p.get("id_column", "index") # 默认值
    p["label_column"] = p.get("label_column", "response") # 默认值
    cfg["output_root"] = output_root_abs
# ======================================================================================

def _rule_based(nl: str) -> dict: # <-- 这个函数现在只在LLM完全失败时作为备用
    t = nl.lower()
    hints = _infer_hints_from_user_nl(nl)
    disease = hints.get("disease", "LIHC")
    model = hints.get("model", "SVM")
    modalities = hints.get("modalities", ["mRNA","miRNA","SNV"])
    kfold = hints.get("cv_k", 5)
    
    cfg = {
        "task": "classification", "disease": disease, "modalities": modalities, "model": model,
        "cv": {"type":"kfold","k":kfold,"stratified":True,"seed":42},
        "metrics": ["AUC","AUPR","ACC","F1"], "seeds": [42],
        "paths": {
            "clinical": f"data/{disease}/response.csv", "mrna": f"data/{disease}/mRNA_data.csv",
            "mirna": f"data/{disease}/miRNA_data.csv", "snv": f"data/{disease}/snv_data.csv",
            "kegg_gmt": "pathways/kegg.gmt", "kegg_map_long": "pathways/kegg_map.txt",
            "id_column": "index", "label_column": "response",
        },
    }
    return cfg

# [EN] TODO: translate comment from Chinese.
def main():
    use_llm = os.environ.get("LLM_PROVIDER", "").strip().lower() in {"deepseek","openai"}
    cfg_draft = {}
    src = "step2_nl.py"

    if use_llm:
        try:
            print("[info] LLM provider detected. Attempting to parse with LLM (Generator)...")
            initial_content = _chat_completion_llm(NL)
            cfg_draft = _extract_json(initial_content)
            src = "step2_nl.py (LLM)"
        except Exception as e:
            print(f"[warn] Initial LLM call failed: {e}")
            cfg_draft = {} # 即使失败，也传一个空字典给Debug模块
            src = "step2_nl.py (LLM failed)"
    else:
        print("[error] No LLM provider detected. Using rule-based fallback.")
        cfg_draft = _rule_based(NL)
        src = "step2_nl.py (rule-based)"
        
    # [EN] TODO: translate comment from Chinese.
    cfg = _debug_and_correct_config(cfg_draft, NL)

    # [EN] TODO: translate comment from Chinese.
    if not cfg:
        print("[warn] Auto-correction failed or resulted in empty config. Applying rule-based hints as a final fallback.")
        cfg = cfg_draft # 使用质检前的草案
        hints = _infer_hints_from_user_nl(NL)
        _apply_user_overrides(cfg, hints)

    # [EN] TODO: translate comment from Chinese.
    if WANT_ABS_PATHS:
        _absolutize_paths(cfg, base_dir=BASE_DIR, output_root_abs=OUTPUT_ROOT_ABS)

    cfg["_meta"] = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": src,
        "provider": os.environ.get("LLM_PROVIDER", ""),
        "user_nl": NL,
    }

    Path(os.path.dirname(LAST_RUN_CONFIG)).mkdir(parents=True, exist_ok=True)
    with open(LAST_RUN_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n===== Parsed from Natural Language (after auto-correction) =====")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print("\nSaved to:", LAST_RUN_CONFIG)
    print("Next:\n  python agent/step3_run.py")

if __name__ == "__main__":
    main()

