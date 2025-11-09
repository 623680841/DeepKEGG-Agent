#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在单个CV折里，调用论文仓库的 DeepKEGG 模型进行训练/推理。
输入/输出（由 step3_run.py 生成并调用）：
- --workdir 目录中包含：
  train_X_mRNA.csv / train_X_miRNA.csv / train_X_SNV.csv（可能缺失）
  train_y.csv（两列: sample_id, y）
  test_X_*.csv（同上）
  kegg_gmt.gmt, kegg_map.txt
- 本脚本在同目录写出：
  y_prob.csv（两列: sample_id, y_prob）——只需 test 集的概率
"""
import os
import sys
import argparse
import pandas as pd

def load_mat(path):
    df = pd.read_csv(path)
    # [EN] TODO: translate comment from Chinese.
    df = df.set_index(df.columns[0])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--r", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wd = args.workdir
    out_csv = os.path.join(wd, "y_prob.csv")

    f_train = {
        "mRNA":  os.path.join(wd, "train_X_mRNA.csv"),
        "miRNA": os.path.join(wd, "train_X_miRNA.csv"),
        "SNV":   os.path.join(wd, "train_X_SNV.csv"),
    }
    f_test = {
        "mRNA":  os.path.join(wd, "test_X_mRNA.csv"),
        "miRNA": os.path.join(wd, "test_X_miRNA.csv"),
        "SNV":   os.path.join(wd, "test_X_SNV.csv"),
    }
    f_y   = os.path.join(wd, "train_y.csv")
    f_gmt = os.path.join(wd, "kegg_gmt.gmt")
    f_map = os.path.join(wd, "kegg_map.txt")

    if not os.path.exists(f_y):   raise SystemExit("缺少 train_y.csv")
    if not os.path.exists(f_gmt): raise SystemExit("缺少 kegg_gmt.gmt")
    if not os.path.exists(f_map): raise SystemExit("缺少 kegg_map.txt")

    # [EN] TODO: translate comment from Chinese.
    train_dict, test_dict = {}, {}
    if os.path.exists(f_train["mRNA"]):  train_dict["mRNA"]  = load_mat(f_train["mRNA"])
    if os.path.exists(f_train["miRNA"]): train_dict["miRNA"] = load_mat(f_train["miRNA"])
    if os.path.exists(f_train["SNV"]):   train_dict["SNV"]   = load_mat(f_train["SNV"])
    if os.path.exists(f_test["mRNA"]):   test_dict["mRNA"]   = load_mat(f_test["mRNA"])
    if os.path.exists(f_test["miRNA"]):  test_dict["miRNA"]  = load_mat(f_test["miRNA"])
    if os.path.exists(f_test["SNV"]):    test_dict["SNV"]    = load_mat(f_test["SNV"])
    if not test_dict:
        raise SystemExit("测试集没有任何组学矩阵")

    # [EN] TODO: translate comment from Chinese.
    y_train = pd.read_csv(f_y).set_index("sample_id")["y"]

    # [EN] TODO: translate comment from Chinese.
    if os.environ.get("PERMUTE_LABELS", "0") == "1":
        import numpy as _np
        _np.random.seed(42)
        vals = _np.array(y_train.values, copy=True)
        _np.random.shuffle(vals)
        y_train = pd.Series(vals, index=y_train.index)
    # -----------------------------------------

    # [EN] TODO: translate comment from Chinese.
    sys.path.append("/tmp/DeepKEGG-master")  # 确保能 import 到 deepkegg_api.py
    from deepkegg_api import fit_one_fold_and_predict

    # [EN] TODO: translate comment from Chinese.
    y_prob = fit_one_fold_and_predict(
        train_dict, y_train, test_dict,
        gmt_path=f_gmt, map_path=f_map,
        k=args.k, r=args.r, epochs=args.epochs, seed=args.seed
    )

    # [EN] TODO: translate comment from Chinese.
    sample_ids = next(iter(test_dict.values())).index.tolist()
    pd.DataFrame({"sample_id": sample_ids, "y_prob": y_prob}).to_csv(out_csv, index=False)
    print(f"[OK] 写出 y_prob.csv -> {out_csv}")

if __name__ == "__main__":
    main()
