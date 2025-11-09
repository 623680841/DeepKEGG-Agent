# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import requests
import subprocess
import importlib
import importlib.util
# [EN] TODO: translate comment from Chinese.
import importlib
# [EN] TODO: translate comment from Chinese.
try:
    # [EN] TODO: translate comment from Chinese.
    from .llm_codegen import suggest_estimator, build_estimator_from_spec, _strip_fences
    from .deepkegg_model import train_and_evaluate_deepkegg, prepare_pathway_data
    from .biomarker_discovery import find_biomarkers_for_deepkegg, find_biomarkers_for_tree_models
    from .deep_learning_trainer import train_and_evaluate_dl_model
except ImportError:
    # [EN] TODO: translate comment from Chinese.
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from llm_codegen import suggest_estimator, build_estimator_from_spec, _strip_fences
    from deepkegg_model import train_and_evaluate_deepkegg, prepare_pathway_data
    from biomarker_discovery import find_biomarkers_for_deepkegg, find_biomarkers_for_tree_models
    from deep_learning_trainer import train_and_evaluate_dl_model

# [EN] TODO: translate comment from Chinese.
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

# [EN] TODO: translate comment from Chinese.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"  # 用于定位 paper_methodology.txt、generated_models、step3b_innovate_and_run.py 等

# [EN] TODO: translate comment from Chinese.
# [EN] TODO: translate comment from Chinese.
def _abs(p: str | Path) -> Path:
    """
    智能地将路径转换为绝对路径。
    - 如果路径已经是绝对路径，直接返回。
    - 如果是数据文件路径，则以 BASE_DIR 为基础进行正确拼接。
    - 否则，以项目根目录为基础。
    """
    if not p: return None
    pp = Path(p)
    if pp.is_absolute():
        return pp

    # [EN] TODO: translate comment from Chinese.
    base_dir = Path(os.environ.get("BASE_DIR", "/tmp/DeepKEGG-master"))
    path_str = str(p)

    # [EN] TODO: translate comment from Chinese.
    if path_str.startswith("data/"):
        # [EN] TODO: translate comment from Chinese.
        parts = pp.parts
        if len(parts) > 1:
            disease_folder = f"{parts[1]}_data"
            file_name = "/".join(parts[2:])
            return base_dir / disease_folder / file_name
        else:
            return base_dir / pp # Fallback
            
    elif path_str.startswith("pathways/") or path_str.startswith("KEGG_pathways/"):
        return base_dir / pp
        
    else:
        # [EN] TODO: translate comment from Chinese.
        return PROJECT_ROOT / pp


def _read_csv_with_index(path_csv: str | Path, id_col: str) -> pd.DataFrame:
    read_index_col = 0 if id_col == "index" else id_col
    try:
        # [EN] TODO: translate comment from Chinese.
        df = pd.read_csv(_abs(path_csv), sep=None, engine="python", index_col=read_index_col)
    except Exception as e:
        print(f"[ERROR] Pandas 读取 '{_abs(path_csv)}' (索引: '{read_index_col}') 时失败。")
        raise e
    df.index.name = id_col
    df = df.loc[:, df.notna().any(axis=0)]
    return df

def _load_labels(path_csv: str | Path, id_col: str, y_col: str) -> pd.Series:
    read_index_col = 0 if id_col == "index" else id_col
    df = pd.read_csv(_abs(path_csv), sep=None, engine="python", index_col=read_index_col)
    if y_col not in df.columns:
        raise ValueError(f"'{_abs(path_csv)}' 中必须包含标签列 '{y_col}'。找到的列: {df.columns.tolist()}")
    df = df[[y_col]].dropna()
    y = df[y_col].astype(float)
    uniq = sorted({int(v) for v in y.unique()})
    if len(uniq) == 2 and uniq != [0, 1]:
        y = (y == max(uniq)).astype(int)
    return y.astype(int)


def _load_tables(paths: dict, modalities: list[str], id_col: str) -> pd.DataFrame:
    """按模态读取表格并以样本交集拼接；列名前加模态前缀。"""
    mats = {}
    name2path = {"mRNA": "mrna", "miRNA": "mirna", "SNV": "snv"}
    for m in modalities:
        key = name2path.get(m)
        if not key:
            continue
        p = paths.get(key)
        if not p or not _abs(p).exists():
            print(f"[WARN] Missing file for {m}: {p}")
            continue
        df = _read_csv_with_index(p, id_col=id_col).add_prefix(f"{m}::")
        mats[m] = df
    if not mats:
        raise ValueError("No omics tables available.")
    ids = None
    for df in mats.values():
        ids = set(df.index) if ids is None else (ids & set(df.index))
    ids = sorted(list(ids))
    if not ids:
        raise ValueError("Empty sample intersection across omics.")
    for k in list(mats.keys()):
        mats[k] = mats[k].loc[ids].sort_index(axis=1)
    return pd.concat([mats[m] for m in sorted(mats.keys())], axis=1)

def _mk_run_dir(output_root: Path, disease: str, tag: str) -> Path:
    """创建本次 run 目录：<runs>/<disease>_<ts>_<rid>_<tag>/"""
    now = time.strftime("%Y%m%d-%H%M%S")
    rid = os.urandom(4).hex()
    d = _abs(output_root) / f"{disease}_{now}_{rid}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _update_summary_csv(summary_csv_path: str, new_metrics_dict: dict,
                        cancer_type: str, omics_type: str, llm_provider: str,
                        model_name: str, run_dir: str):
    """将本次 metrics 追加到全局汇总 CSV。"""
    new_row = {
        "Cancer": cancer_type,
        "Omics": omics_type,
        "LLM Provider": llm_provider,
        "Model": model_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run Dir": os.path.basename(run_dir),
        **new_metrics_dict
    }
    df = pd.read_csv(summary_csv_path) if os.path.exists(summary_csv_path) else pd.DataFrame()
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    desired_order = [
        "Cancer", "Omics", "LLM Provider", "Model", "Timestamp",
        "AUC", "AUPR", "F1", "ACC", "Precision", "Recall",
        "Run Dir"
    ]
    existing_cols = [c for c in desired_order if c in updated_df.columns]
    other_cols = [c for c in updated_df.columns if c not in existing_cols]
    updated_df = updated_df[existing_cols + other_cols]
    updated_df.to_csv(summary_csv_path, index=False, float_format="%.4f")
    print(f"✅ 总表已更新: {summary_csv_path}")

# [EN] TODO: translate comment from Chinese.
def review_spec_against_paper(spec: dict, paper_methods_text: str) -> dict:
    """
    用 LLM 将生成的 estimator 规范与论文方法对齐；失败则原样返回。
    """
    print(">>> 启动基于论文的合规性审查模块...")
    try:
        spec_str = json.dumps(spec, indent=2)
        review_system_prompt = """You are a bioinformatics research assistant. Your task is to verify if a generated Python code specification for an ML model conforms to the methodology described in a scientific paper.
- The paper's methodology is the ground truth.
- If the spec conforms, return it EXACTLY as is.
- If the spec deviates, you MUST output a corrected version.
- **CRITICAL RULE: If the model mentioned in the [Code Spec] (e.g., XGBoost) is NOT mentioned in the [Paper Methodology], then the spec is considered compliant by default. In this case, you MUST return the original [Code Spec] without any changes.**
- Your output MUST be ONLY the final, pure JSON object."""
        review_user_prompt = f"""Please review the following 'Code Spec' against the 'Paper Methodology'.

[Paper Methodology]:
{paper_methods_text}

[Code Spec to Review]:
{spec_str}

Is the 'Code Spec' compliant? If not, provide the corrected JSON. If yes, return the original JSON.
"""
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key: raise RuntimeError("DEEPSEEK_API_KEY not set")
            url, model = "https://api.deepseek.com/chat/completions", "deepseek-chat"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: raise RuntimeError("OPENAI_API_KEY not set")
            url, model = "https://api.openai.com/v1/chat/completions", "gpt-4o"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": review_system_prompt},
                {"role": "user", "content": review_user_prompt},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        corrected_spec = json.loads(_strip_fences(content))
        print(">>> 论文合规性审查完成！")
        return corrected_spec
    except Exception as e:
        print(f"[ERROR] 论文合规性审查模块出错: {e}")
        print("      将返回未经审查的原始代码蓝图。")
        return spec

# [EN] TODO: translate comment from Chinese.
def llm_code_generation_with_auto_debug(user_hint: str, n_samples: int, n_features: int, pos_ratio: float) -> dict:
    """
    让 LLM 先给出 estimator 规范；若构建失败，携带报错自动反馈给 LLM 要求修复；最多 3 轮。
    """
    MAX_DEBUG_ATTEMPTS = 3
    sparsity = 0.0
    spec, last_error = {}, ""
    for attempt in range(MAX_DEBUG_ATTEMPTS):
        print(f"\n>>> [Debug Attempt {attempt+1}/{MAX_DEBUG_ATTEMPTS}]")
        if attempt == 0:
            spec = suggest_estimator(
                n_samples=n_samples, n_features=n_features,
                sparsity=sparsity, target_balance=pos_ratio, user_hint=user_hint
            )
        else:
            debug_prompt = f"""The previous spec failed with error:
{last_error}
Please return a corrected JSON spec.
[Original Spec]
{json.dumps(spec, indent=2)}
"""
            spec = suggest_estimator(
                n_samples=n_samples, n_features=n_features,
                sparsity=sparsity, target_balance=pos_ratio, user_hint=debug_prompt
            )
        if not isinstance(spec, dict) or "imports" not in spec or "init" not in spec:
            last_error = "LLM did not return a valid JSON spec."
            continue
        try:
            build_estimator_from_spec(spec)  # 只做可构建性验证
            print("    - ✅ Spec validation successful.")
            return spec
        except Exception as e:
            print(f"    - ❌ Spec validation failed: {e}")
            last_error = str(e)
    raise RuntimeError(f"Automated debugging failed after {MAX_DEBUG_ATTEMPTS} attempts. Last error: {last_error}")

# [EN] TODO: translate comment from Chinese.
def main(cfg_path: str, llm_provider: str):
    # [EN] TODO: translate comment from Chinese.
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    disease    = cfg.get("disease") or cfg.get("cancer") or "UNK"
    modalities = cfg.get("modalities") or cfg.get("omics") or ["mRNA", "miRNA", "SNV"]
    paths_cfg  = cfg.get("paths") or {}
    id_col     = str(paths_cfg.get("id_column") or "index")
    y_col      = str(paths_cfg.get("label_column") or "response")
    out_root   = Path(cfg.get("output_root") or (PROJECT_ROOT / "runs").as_posix())
    model_hint = str(cfg.get("model") or "")

    # [EN] TODO: translate comment from Chinese.
    print("[1/5] Load labels…")
    y = _load_labels(paths_cfg["clinical"], id_col=id_col, y_col=y_col)
    print("[2/5] Load & join omics…")
    X = _load_tables(paths_cfg, modalities, id_col=id_col)
    print("[3/5] Align samples…")
    idx = sorted(set(X.index) & set(y.index))
    if not idx:
        raise ValueError("No common samples between X and y.")
    X = X.loc[idx].copy()
    y = y.loc[idx].copy()
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    PREBUILT_DL_MODELS = ["DeepKEGG", "LSTM", "Transformer", "Mamba", "CNN"]
    FORCE_LLM_MODELS   = ["SVM", "LR", "XGBoost", "RandomForest", "GCN", "GAT", "PathFinder"]
    trained_model_object = None
    metrics_mean: dict = {}
    run_dir: Path
    model_name_for_summary: str

    # [EN] TODO: translate comment from Chinese.
    if model_hint == "DeepKEGG":
        print("[4/5] Preparing pathway data for DeepKEGG...")
        pathway_masks = prepare_pathway_data(X, cfg)
        print("[5/5] Cross-validated training (DeepKEGG pre-designed model)...")
        tag = "DeepKEGG"
        run_dir = _mk_run_dir(out_root, disease, tag)
        trained_model_object, metrics_mean = train_and_evaluate_deepkegg(X, y, cfg, pathway_masks, run_dir)
        model_name_for_summary = "DeepKEGG"

    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    elif (AGENT_DIR / "generated_models" / model_hint).is_dir():
        print(f"[4/5] Found dynamically generated model: '{model_hint}'. Attempting to load...")
        try:
            # [EN] TODO: translate comment from Chinese.
            model_package_path = f"agent.generated_models.{model_hint}"
            generated_model_module = importlib.import_module(model_package_path)
            
            if hasattr(generated_model_module, 'run_training'):
                print("[5/5] Executing training using the generated model's entry point...")
                tag = f"Generated_{model_hint}"
                run_dir = _mk_run_dir(out_root, disease, tag)
                
                # [EN] TODO: translate comment from Chinese.
                metrics_mean = generated_model_module.run_training(X, y, cfg, run_dir)
                
                # [EN] TODO: translate comment from Chinese.
                if not isinstance(metrics_mean, dict) or not metrics_mean:
                    print(f"[WARN] The 'run_training' function for '{model_hint}' did not return a valid metrics dictionary. Reading from file instead.")
                    metrics_path = run_dir / "metrics.csv"
                    if metrics_path.exists():
                        metrics_mean = pd.read_csv(metrics_path).iloc[0].to_dict()
                    else:
                        # [EN] TODO: translate comment from Chinese.
                        metrics_mean = {} 
                        print(f"[ERROR] Could not find metrics.csv for run {run_dir.name}. Metrics will be empty.")

                model_name_for_summary = model_hint
                trained_model_object = None # 动态生成的模型不强制返回模型对象
            else:
                raise ImportError(f"The generated model '{model_hint}' does not have a 'run_training' entry point function.")

        except Exception as e:
            print(f"\n[ERROR] Error loading or running the generated model '{model_hint}':")
            print(f"  - {e}")
            print("  - Please check the generated code in 'agent/generated_models/'.")
            return # 发生错误时提前退出
            
    elif model_hint in PREBUILT_DL_MODELS:
        print(f"[4/5] Preparing to run pre-built DL model: {model_hint}")
        tag = model_hint
        run_dir = _mk_run_dir(out_root, disease, tag)
        trained_model_object, metrics_mean = train_and_evaluate_dl_model(
            model_name=model_hint, X=X, y=y, cfg=cfg, run_dir=run_dir
        )
        model_name_for_summary = model_hint

    # [EN] TODO: translate comment from Chinese.
    elif model_hint == "INNOVATE_MODEL":
        print(">>> 检测到创新模型请求，正在启动AI创新引擎...")
        innovation_engine_script = AGENT_DIR / "step3b_innovate_and_run.py"
        command = f"python {innovation_engine_script} --cfg {cfg_path}"
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print("\n[ERROR] AI创新引擎执行失败。")
        else:
            print("\n[SUCCESS] AI创新引擎执行完毕。")
        return

    # [EN] TODO: translate comment from Chinese.
    elif model_hint in FORCE_LLM_MODELS:
        if os.getenv("USE_LLM_CODEGEN", "") not in {"1", "true", "TRUE"}:
            raise RuntimeError(f"Model '{model_hint}' requires LLM codegen. Set USE_LLM_CODEGEN=1")

        print(f"[4/5] Starting LLM code generation with auto-debug for {model_hint}...")
        n_samples, n_features = X.shape
        pos_ratio = float(y.mean())
        spec = llm_code_generation_with_auto_debug(
            user_hint=model_hint, n_samples=n_samples, n_features=n_features, pos_ratio=pos_ratio
        )

        # [EN] TODO: translate comment from Chinese.
        paper_methods_path = AGENT_DIR / "paper_methodology.txt"
        final_spec = review_spec_against_paper(
            spec, paper_methods_path.read_text(encoding="utf-8")
        ) if paper_methods_path.exists() else spec

        print("[LLM] Final estimator spec (after review):\n", json.dumps(final_spec, indent=2))
        model_obj = build_estimator_from_spec(final_spec)
        base_est, fit_params = model_obj.estimator, (model_obj.fit_params or {})

        print("[5/5] Cross-validated training (LLM-driven estimator)…")
        cv_cfg = cfg.get("cv", {})
        skf = StratifiedKFold(
            n_splits=cv_cfg.get("k", 10),
            shuffle=True,
            random_state=cv_cfg.get("seed", 42)
        )

        fold_rows, all_pred, all_prob = [], np.zeros(len(y)), np.zeros(len(y))

        # [EN] TODO: translate comment from Chinese.
        final_model_pipe = make_pipeline(StandardScaler(), deepcopy(base_est))
        final_model_pipe.fit(X.values, y.values, **fit_params)
        trained_model_object = final_model_pipe

        # [EN] TODO: translate comment from Chinese.
        for fi, (tr, te) in enumerate(tqdm(skf.split(X, y), total=skf.get_n_splits()), start=1):
            Xtr, Xte, ytr, yte = X.values[tr], X.values[te], y.values[tr], y.values[te]
            pipe = make_pipeline(StandardScaler(), deepcopy(base_est))
            try:
                pipe.fit(Xtr, ytr, **fit_params)
            except TypeError:
                pipe.fit(Xtr, ytr)

            if hasattr(pipe, "predict_proba"):
                prob = pipe.predict_proba(Xte)[:, 1]
            else:
                prob = 1.0 / (1.0 + np.exp(-pipe.decision_function(Xte)))
            ypred = pipe.predict(Xte)

            fold_rows.append({
                "fold": fi,
                "AUC": roc_auc_score(yte, prob),
                "AUPR": average_precision_score(yte, prob),
                "ACC": accuracy_score(yte, ypred),
                "F1": f1_score(yte, ypred)
            })
            all_pred[te], all_prob[te] = ypred, prob

        df_folds = pd.DataFrame(fold_rows)
        metrics_mean = df_folds.mean(numeric_only=True).dropna().to_dict()

        tag = f"LLM_{model_hint}"
        run_dir = _mk_run_dir(out_root, disease, tag)

        df_folds.to_csv(run_dir / "metrics_per_fold.csv", index=False)
        pd.DataFrame([metrics_mean]).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame({
            "sample_id": X.index, "y_true": y.values,
            "y_pred": all_pred, "y_prob": all_prob
        }).to_csv(run_dir / "predictions.csv", index=False)

        cfg_out = dict(cfg)
        cfg_out["_llm_estimator_spec"] = final_spec
        (run_dir / "run_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

        model_name_for_summary = model_hint

    else:
        raise ValueError(f"Unknown model '{model_hint}'.")

    # [EN] TODO: translate comment from Chinese.
    print(f"\n✅ Run finished! Output dir: {run_dir}")

    # [EN] TODO: translate comment from Chinese.
    if cfg.get("run_biomarker_discovery"):
        print("\n[6/6] Starting biomarker discovery...")
        if trained_model_object is None:
            print("[WARN] No trained model object available for biomarker discovery. Skipping.")
        else:
            try:
                if model_hint == "DeepKEGG":
                    find_biomarkers_for_deepkegg(
                        model=trained_model_object,
                        X_all=X, y_all=y, run_dir=run_dir,
                        top_n=cfg.get("run_biomarker_discovery", {}).get("top_n", 20)
                    )
                elif hasattr(trained_model_object.named_steps['clf'], 'feature_importances_'):
                    find_biomarkers_for_tree_models(
                        model_pipe=trained_model_object,
                        feature_names=X.columns, run_dir=run_dir,
                        top_n=cfg.get("run_biomarker_discovery", {}).get("top_n", 20)
                    )
                else:
                    print(f"[WARN] Biomarker discovery not implemented for model type: {model_hint}")
            except Exception as e:
                print(f"[WARN] Biomarker discovery failed: {e}")

    # [EN] TODO: translate comment from Chinese.
    summary_csv_path = os.path.join(out_root, "all_models_metrics.csv")
    # [EN] TODO: translate comment from Chinese.
    is_generated = (AGENT_DIR / "generated_models" / model_hint).is_dir()
    final_llm_provider = llm_provider if (model_hint not in ["DeepKEGG", "LSTM", "Transformer", "Mamba", "CNN"] and not is_generated) else "N/A"
    _update_summary_csv(
        summary_csv_path=summary_csv_path,
        new_metrics_dict=metrics_mean,
        cancer_type=disease,
        omics_type=", ".join(modalities),
        llm_provider=final_llm_provider,
        model_name=model_name_for_summary,
        run_dir=str(run_dir),
    )

# [EN] TODO: translate comment from Chinese.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single model training task based on a config file."
    )
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config JSON file.")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="Unknown",
        help="The LLM provider that generated the code (e.g., 'deepseek', 'chatgpt').",
    )
    args = parser.parse_args()
    main(cfg_path=args.cfg, llm_provider=args.llm_provider)
