# /tmp/DeepKEGG-agent/zsy_DeepKEGGAgent/agent/generated_models/DeepKEGG_v2/model_main.py

import os
import math
import time
import traceback
import datetime
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from tqdm.auto import tqdm

from torch_geometric.nn import GATv2Conv


# [EN] TODO: translate comment from Chinese.
def _ensure_node_feat_2d(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if x is None:
        raise RuntimeError("Input features to GAT is None.")
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.unsqueeze(1)
    elif x.dim() == 3:
        if x.size(0) == 1:
            x = x.squeeze(0)
        else:
            x = x.view(x.size(1), -1)
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    dtype = next(model.parameters()).dtype
    return x.to(dtype=dtype).contiguous()


def _coerce_to_pathway_mod_shape(x: torch.Tensor, n_pathways: int, n_mod: int) -> torch.Tensor:
    if x.dim() == 1:
        if x.numel() == n_pathways:
            x = x.view(n_pathways, 1).repeat(1, n_mod)
        elif x.numel() == n_mod:
            x = x.view(1, n_mod).repeat(n_pathways, 1)
        else:
            raise RuntimeError(f"Cannot coerce 1D tensor len {x.numel()} -> [{n_pathways},{n_mod}]")
        return x

    if x.dim() == 2:
        N, F = x.shape
        if N == n_pathways and F == n_mod:
            return x
        if N == n_mod and F == n_pathways:
            return x.t().contiguous()
        if N == n_pathways and F == 1 and n_mod > 1:
            return x.repeat(1, n_mod)
        if F == n_mod and N == 1 and n_pathways > 1:
            return x.repeat(n_pathways, 1)
        if F == 1 and N == n_mod and n_pathways > 1:
            return x.t().repeat(n_pathways, 1)
        raise RuntimeError(f"Bad feature shape {tuple(x.shape)}; expect [{n_pathways},{n_mod}]")

    dims = list(x.shape)
    try:
        idx_p = next(i for i, s in enumerate(dims) if s == n_pathways)
        idx_m = next(i for i, s in enumerate(dims) if s == n_mod and i != idx_p)
    except StopIteration:
        sorted_axes = sorted(range(len(dims)), key=lambda i: dims[i], reverse=True)
        idx_p, idx_m = sorted_axes[:2]
    order = [idx_p, idx_m] + [i for i in range(len(dims)) if i not in (idx_p, idx_m)]
    x = x.permute(*order).contiguous()
    if x.dim() > 2:
        x = x[..., 0]
    N, F = x.shape
    if N != n_pathways:
        if N == 1:
            x = x.repeat(n_pathways, 1)
        else:
            raise RuntimeError(f"Cannot fix N: got {N}, expect {n_pathways}")
    if F != n_mod:
        if F == 1:
            x = x.repeat(1, n_mod)
        elif F == n_pathways and N == n_mod:
            x = x.t()
        else:
            raise RuntimeError(f"Cannot fix F: got {F}, expect {n_mod}")
    return x.contiguous()


def _scalarize_logit(t: torch.Tensor, n_pathways: int) -> torch.Tensor:
    """确保模型输出是 [1,1]；若为节点级（如 [n_pathways,1] / [n_pathways]）则对节点平均。"""
    if t is None:
        raise RuntimeError("Model returned None logits.")
    if t.dim() == 0:
        return t.view(1, 1)
    if t.dim() == 1:
        if t.numel() == 1:
            return t.view(1, 1)
        return t.mean().view(1, 1)
    if t.dim() == 2:
        N, F = t.shape
        if N == 1 and F == 1:
            return t
        if N == n_pathways and F == 1:
            return t.mean(dim=0, keepdim=True)  # -> [1,1]
        if N == 1 and F != 1:
            return t.mean(dim=1, keepdim=True)
        return t.mean().view(1, 1)
    return t.view(-1, 1).mean().view(1, 1)


def _align_logits_to_targets(logits: torch.Tensor, y: torch.Tensor, n_pathways: int) -> torch.Tensor:
    """
    最后一道关：把 logits 调整成与 y 同形（[B,1]）。
    若发现 logits 是按节点展开（[B*n_pathways,1] 或 [n_pathways,1]），自动做均值聚合。
    """
    # [EN] TODO: translate comment from Chinese.
    if logits.dim() == 0:
        logits = logits.view(1, 1)
    elif logits.dim() == 1:
        logits = logits.view(-1, 1)
    elif logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)  # [?, ?] -> 尽量压

    B = y.size(0)
    if logits.size(0) == B:
        return logits  # OK

    # [EN] TODO: translate comment from Chinese.
    if logits.size(0) == B * n_pathways:
        logits = logits.view(B, n_pathways, 1).mean(dim=1, keepdim=False)  # -> [B,1]
        return logits

    # [EN] TODO: translate comment from Chinese.
    if B == 1 and logits.size(0) == n_pathways:
        logits = logits.mean(dim=0, keepdim=True)  # -> [1,1]
        return logits

    # [EN] TODO: translate comment from Chinese.
    raise ValueError(
        f"Cannot align logits to targets: logits={tuple(logits.shape)}, "
        f"targets={tuple(y.shape)}, n_pathways={n_pathways}"
    )


# [EN] TODO: translate comment from Chinese.
def parse_gmt_file(gmt_path: str) -> dict:
    pathways = {}
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                pathways[parts[0]] = set(parts[2:])
    return pathways


def build_pathway_interaction_graph(pathways: dict) -> tuple:
    print("正在构建通路交互图...")
    pathway_list = sorted(pathways.keys())
    pathway_to_id = {name: i for i, name in enumerate(pathway_list)}
    edge_list = set()
    for i in range(len(pathway_list)):
        for j in range(i + 1, len(pathway_list)):
            p1_name, p2_name = pathway_list[i], pathway_list[j]
            genes1 = pathways.get(p1_name, set())
            genes2 = pathways.get(p2_name, set())
            if not genes1.isdisjoint(genes2):
                edge_list.add(tuple(sorted((i, j))))
    edge_index = torch.tensor(list(edge_list), dtype=torch.long).t().contiguous()
    print(f"通路图构建完成: {len(pathway_list)} 个节点 (通路), {edge_index.shape[1]} 条边。")
    return edge_index, pathway_to_id, pathway_list


# [EN] TODO: translate comment from Chinese.
class OmicsToPathwayLayer(nn.Module):
    def __init__(self, modalities: list, gene_to_pathway_map: torch.Tensor):
        super().__init__()
        self.modalities = modalities
        self.register_buffer('gene_to_pathway_map', gene_to_pathway_map)
        self.mod_linears = nn.ModuleDict({mod: nn.Linear(1, 1) for mod in modalities})

    def forward(self, x_omics: dict) -> torch.Tensor:
        pathway_features_list = []
        for mod in self.modalities:
            x_mod = x_omics[mod]                               # [B, n_genes]
            x_mod_transformed = self.mod_linears[mod](x_mod.unsqueeze(-1)).squeeze(-1)
            pathway_agg = torch.matmul(x_mod_transformed, self.gene_to_pathway_map)  # [B, n_pathways]
            genes_per_pathway = self.gene_to_pathway_map.sum(dim=0) + 1e-8
            pathway_mean = pathway_agg / genes_per_pathway
            pathway_features_list.append(pathway_mean)         # [B, n_pathways]
        return torch.stack(pathway_features_list, dim=-1)      # [B, n_pathways, n_mod]


# [EN] TODO: translate comment from Chinese.
class PathwayGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, pathway_features, edge_index):
        # pathway_features: [n_pathways, in_channels]
        x = pathway_features
        if x.dim() != 2:
            raise RuntimeError(f"GAT expects [N,F], got {tuple(x.shape)}")
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)      # [N, hidden]
        x = x.mean(dim=0)                  # [hidden] —— 样本聚合
        x = self.classifier(x)             # [1] —— 样本级 logit
        return x


# [EN] TODO: translate comment from Chinese.
def run_training(X: pd.DataFrame, y: pd.Series, cfg: dict, run_dir: Path) -> dict:
    def now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def format_bytes(n):
        for u in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024:
                return f"{n:.1f}{u}"
            n /= 1024
        return f"{n:.1f}PB"

    def ensure_run_dir(rd: Path) -> Path:
        if rd is None:
            base = Path.cwd() / "runs_autogen"
            base.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            disease = str(cfg.get("disease") or "UNK")
            model = str(cfg.get("model") or "model")
            rd = base / f"{disease}_{model}_{ts}"
        rd.mkdir(parents=True, exist_ok=True)
        return rd

    run_dir = ensure_run_dir(run_dir)
    print(f"[run_dir] Outputs will be written to: {run_dir.resolve()}")

    def dump_crash(e, context: dict):
        try:
            rpt = run_dir / "crash_report.txt"
            with rpt.open("a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"[{now()}] RUNTIME ERROR\n")
                for k, v in context.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n--- TRACEBACK ---\n")
                f.write("".join(traceback.format_exc()))
                f.write("\n--- CUDA MEM ---\n")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        f.write(f"allocated={format_bytes(torch.cuda.memory_allocated())}, "
                                f"reserved={format_bytes(torch.cuda.memory_reserved())}\n")
                        f.write(torch.cuda.memory_summary() + "\n")
                    except Exception as ee:
                        f.write(f"[mem_summary_error] {ee}\n")
                f.write("\n")
        finally:
            import sys
            print("\n[FATAL] Exception raised. Full traceback:", file=sys.stderr)
            traceback.print_exc()
            print(f"[FATAL] Crash report saved to: {rpt}", file=sys.stderr)

    print(f"--- 正在运行生成的模型: {cfg.get('model')} (通路图GNN版本 - 高性能版) ---")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    # [EN] TODO: translate comment from Chinese.
    paths_cfg = cfg.get("paths", {})
    gmt_path = paths_cfg.get('kegg_gmt') or paths_cfg.get('gmt') or "DeepKEGG-agent/DeepKEGG-Agent/data/KEGG_pathways/20230205_kegg_hsa.gmt"
    pathways_dict = parse_gmt_file(gmt_path)
    edge_index, pathway_to_id, pathway_list = build_pathway_interaction_graph(pathways_dict)
    n_pathways = len(pathway_list)

    all_feature_genes = sorted(list(set([c.split("::")[1] for c in X.columns])))
    gene_to_id = {gene: i for i, gene in enumerate(all_feature_genes)}

    gene_pathway_map = torch.zeros(len(all_feature_genes), n_pathways, dtype=torch.float)
    for p_name, genes in pathways_dict.items():
        if p_name in pathway_to_id:
            p_id = pathway_to_id[p_name]
            for gene in genes:
                if gene in gene_to_id:
                    g_id = gene_to_id[gene]
                    gene_pathway_map[g_id, p_id] = 1.0

    modalities = sorted(list(set([c.split("::")[0] for c in X.columns])))
    n_mod = len(modalities)

    # [EN] TODO: translate comment from Chinese.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_omics_cpu = {}
    for mod in modalities:
        mod_cols = [c for c in X.columns if c.startswith(f"{mod}::")]
        mod_genes = [c.split("::")[1] for c in mod_cols]
        df_mod = X[mod_cols]; df_mod.columns = mod_genes
        arr = df_mod.reindex(columns=all_feature_genes, fill_value=0).values.astype(np.float32, copy=False)
        t = torch.from_numpy(arr).contiguous().clone()
        if device.type == 'cuda':
            t = t.pin_memory()
        X_omics_cpu[mod] = t

    y_np = y.values.astype(np.float32, copy=False).reshape(-1, 1)
    y_cpu = torch.from_numpy(y_np).contiguous().clone()
    if device.type == 'cuda':
        y_cpu = y_cpu.pin_memory()

    # [EN] TODO: translate comment from Chinese.
    cv_cfg = cfg.get("cv", {})
    k_folds = int(cv_cfg.get("k", 5))
    seed = int(cv_cfg.get("seed", 42))
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    train_cfg = cfg.get("train", {})
    batch_size     = int(train_cfg.get("batch_size", 16))
    epochs         = int(train_cfg.get("epochs", 30))
    use_amp        = bool(train_cfg.get("amp", True))
    hidden         = int(train_cfg.get("hidden_channels", 48))
    heads          = int(train_cfg.get("heads", 2))
    lr             = float(train_cfg.get("lr", 5e-4))
    weight_decay   = float(train_cfg.get("weight_decay", 1e-4))
    label_smoothing= float(train_cfg.get("label_smoothing", 0.01))
    max_grad_norm  = float(train_cfg.get("max_grad_norm", 1.0))

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    edge_index = edge_index.to(device, non_blocking=True)
    gene_pathway_map = gene_pathway_map.to(device, non_blocking=True)

    def make_modules():
        feature_agg_layer = OmicsToPathwayLayer(modalities, gene_pathway_map).to(device)
        model = PathwayGAT(in_channels=n_mod, hidden_channels=hidden, heads=heads).to(device)
        for m in list(feature_agg_layer.mod_linears.values()) + [model.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return feature_agg_layer, model

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    fold_metrics = []

    # [EN] TODO: translate comment from Chinese.
    n_samples = len(X)
    tmp = list(skf.split(X, y))
    train_size = len(tmp[0][0])
    steps_per_epoch = math.ceil(train_size / batch_size)
    total_train_steps = k_folds * epochs * steps_per_epoch
    total_val_steps_est = k_folds * math.ceil((n_samples - train_size) / batch_size)
    total_steps_all = total_train_steps + total_val_steps_est
    global_bar = tqdm(total=total_steps_all, desc="Total", dynamic_ncols=True, smoothing=0.1)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    start_time_all = time.perf_counter()

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=k_folds, desc="Folds", leave=False, dynamic_ncols=True)):
        feature_agg_layer, model = make_modules()
        optimizer = torch.optim.Adam(
            list(feature_agg_layer.parameters()) + list(model.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        pos = float(y_cpu[train_idx].sum().item())
        neg = float(len(train_idx) - pos)
        pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # [EN] TODO: translate comment from Chinese.
        feature_agg_layer.train(); model.train()
        n_train = len(train_idx)
        warmup_batches_t, warmup_done = [], False

        try:
            for epoch in tqdm(range(1, epochs + 1), desc=f"Fold {fold+1}/{k_folds} | Train", leave=False, dynamic_ncols=True):
                perm = torch.randperm(n_train)
                batch_bar = tqdm(total=math.ceil(n_train / batch_size), desc="Batches", leave=False, dynamic_ncols=True)
                for start in range(0, n_train, batch_size):
                    t0 = time.perf_counter()
                    idx = train_idx[perm[start: start + batch_size]]
                    X_batch = {mod: X_omics_cpu[mod][idx].to(device, non_blocking=True) for mod in modalities}
                    y_batch = y_cpu[idx].to(device, non_blocking=True)

                    if device.type == 'cuda' and use_amp:
                        try:
                            if torch.cuda.is_bf16_supported():
                                amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
                            else:
                                amp_ctx = torch.cuda.amp.autocast()
                        except Exception:
                            amp_ctx = torch.cuda.amp.autocast()
                    else:
                        amp_ctx = nullcontext()

                    optimizer.zero_grad(set_to_none=True)
                    with amp_ctx:
                        pf = feature_agg_layer(X_batch)  # [B, n_pathways, n_mod]
                        outs = []
                        for i in range(pf.size(0)):
                            xi = _coerce_to_pathway_mod_shape(pf[i], n_pathways, n_mod)
                            xi = _ensure_node_feat_2d(xi, model)
                            out = model(xi, edge_index)             # 期望 [1]
                            out = _scalarize_logit(out, n_pathways) # -> [1,1]
                            outs.append(out)
                        logits = torch.cat(outs, dim=0).view(-1, 1)  # [B,1]
                        logits = _align_logits_to_targets(logits, y_batch, n_pathways)

                        # [EN] TODO: translate comment from Chinese.
                        if label_smoothing > 0.0:
                            y_smooth = y_batch * (1.0 - label_smoothing) + 0.5 * label_smoothing
                        else:
                            y_smooth = y_batch

                        logits = logits.clamp(-30.0, 30.0)
                        loss = criterion(logits, y_smooth)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(feature_agg_layer.parameters()) + list(model.parameters()),
                        max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    # [EN] TODO: translate comment from Chinese.
                    cur_lr = optimizer.param_groups[0]["lr"]
                    if device.type == 'cuda':
                        mem_alloc = torch.cuda.memory_allocated()
                        mem_resv = torch.cuda.memory_reserved()
                        postfix = dict(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}",
                                       mem=f"{format_bytes(mem_alloc)}/{format_bytes(mem_resv)}")
                    else:
                        postfix = dict(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")
                    batch_bar.set_postfix(postfix)
                    batch_bar.update(1)
                    global_bar.update(1)

                    if not warmup_done and len(warmup_batches_t) < 10:
                        warmup_batches_t.append(t1 - t0)

                batch_bar.close()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                if not warmup_done and len(warmup_batches_t) >= 3:
                    warmup_done = True
                    avg_bt = sum(warmup_batches_t) / len(warmup_batches_t)
                    done_steps = global_bar.n
                    remain_steps = total_steps_all - done_steps
                    eta_sec = max(remain_steps * avg_bt, 0)
                    print(f"[ETA] 基于前 {len(warmup_batches_t)} 个 batch 平均 {avg_bt:.3f}s：剩余 ~{eta_sec/60:.1f} 分钟 (~{eta_sec/3600:.2f} 小时)")
        except Exception as e:
            dump_crash(e, {
                "phase": "train",
                "fold": fold,
                "epochs": epochs,
                "batch_size": batch_size,
                "use_amp": use_amp,
                "device": str(device),
            })
            raise

        # [EN] TODO: translate comment from Chinese.
        feature_agg_layer.eval(); model.eval()
        val_probs = []
        try:
            with torch.no_grad():
                n_val = len(val_idx)
                val_bar = tqdm(total=math.ceil(n_val / batch_size), desc=f"Fold {fold+1} | Valid", leave=False, dynamic_ncols=True)
                for start in range(0, n_val, batch_size):
                    idx = val_idx[start: start + batch_size]
                    Xb = {mod: X_omics_cpu[mod][idx].to(device, non_blocking=True) for mod in modalities}

                    if device.type == 'cuda' and use_amp:
                        try:
                            if torch.cuda.is_bf16_supported():
                                amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
                            else:
                                amp_ctx = torch.cuda.amp.autocast()
                        except Exception:
                            amp_ctx = torch.cuda.amp.autocast()
                    else:
                        amp_ctx = nullcontext()

                    with amp_ctx:
                        pf = feature_agg_layer(Xb)  # [B, n_pathways, n_mod]
                        logits_list = []
                        for i in range(pf.size(0)):
                            xi = _coerce_to_pathway_mod_shape(pf[i], n_pathways, n_mod)
                            xi = _ensure_node_feat_2d(xi, model)
                            out = model(xi, edge_index)
                            out = _scalarize_logit(out, n_pathways)
                            logits_list.append(out)
                        logits_b = torch.cat(logits_list, dim=0).view(-1, 1)
                        logits_b = _align_logits_to_targets(logits_b, y_cpu[val_idx].to(device), n_pathways)
                        logits_b = logits_b.clamp(-30.0, 30.0)

                    prob_batch = torch.sigmoid(logits_b).squeeze(1).cpu()
                    val_probs.append(prob_batch)
                    val_bar.update(1)
                    global_bar.update(1)
                val_bar.close()
        except Exception as e:
            dump_crash(e, {
                "phase": "valid",
                "fold": fold,
                "batch_size": batch_size,
                "device": str(device),
            })
            raise

        preds_prob = torch.cat(val_probs).numpy()
        labels_true = y_cpu[val_idx].numpy().flatten()
        if preds_prob.shape[0] != labels_true.shape[0]:
            dump_crash(RuntimeError("Pred/label length mismatch"),
                       {"phase": "metrics", "preds_prob": preds_prob.shape, "labels_true": labels_true.shape})
            raise RuntimeError(f"长度不匹配: preds_prob={preds_prob.shape}, labels_true={labels_true.shape}")
        preds_class = (preds_prob >= 0.5).astype(int)

        fold_metrics.append({
            'AUC': roc_auc_score(labels_true, preds_prob),
            'AUPR': average_precision_score(labels_true, preds_prob),
            'ACC': accuracy_score(labels_true, preds_class),
            'F1': f1_score(labels_true, preds_class)
        })

    global_bar.close()
    total_sec = time.perf_counter() - start_time_all
    print(f"[DONE] 训练与验证完成，总耗时 {total_sec/60:.1f} 分钟 (~{total_sec/3600:.2f} 小时)")

    df_metrics = pd.DataFrame(fold_metrics)
    metrics_mean = df_metrics.mean().to_dict()
    print("交叉验证完成。平均指标:", metrics_mean)
    (run_dir / "metrics.csv").write_text(pd.DataFrame([metrics_mean]).to_csv(index=False), encoding="utf-8")
    return metrics_mean
