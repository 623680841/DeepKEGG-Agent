# /tmp/DeepKEGG-agent/zsy_DeepKEGGAgent/agent/generated_models/KEGG_GAT/model_main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import combinations
from tqdm import tqdm

# [EN] TODO: translate comment from Chinese.
def build_global_graph_from_gmt(gmt_filepath: str, all_feature_genes: list) -> tuple:
    """
    从离线的GMT文件构建全局知识图。
    只包含在特征数据中存在的基因。
    """
    print(f"正在从 {gmt_filepath} 构建全局知识图...")
    
    # [EN] TODO: translate comment from Chinese.
    pathways = {}
    with open(gmt_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                pathways[parts[0]] = set(parts[2:])

    # [EN] TODO: translate comment from Chinese.
    feature_gene_set = set(all_feature_genes)
    graph_genes = sorted(list(feature_gene_set))
    gene_to_id = {gene: i for i, gene in enumerate(graph_genes)}
    
    # [EN] TODO: translate comment from Chinese.
    edge_set = set()
    for pathway_name, genes_in_pathway in pathways.items():
        # [EN] TODO: translate comment from Chinese.
        valid_genes = [gene for gene in genes_in_pathway if gene in gene_to_id]
        # [EN] TODO: translate comment from Chinese.
        for u_gene, v_gene in combinations(valid_genes, 2):
            u, v = gene_to_id[u_gene], gene_to_id[v_gene]
            edge_set.add(tuple(sorted((u, v))))

    if not edge_set:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    
    print(f"图构建完成: {len(graph_genes)} 个节点, {edge_index.shape[1]} 条边。")
    return edge_index, gene_to_id

# [EN] TODO: translate comment from Chinese.
class PatientGraphDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, gene_to_id: dict, edge_index: torch.Tensor):
        super().__init__()
        self.X = X
        self.y = y
        self.gene_to_id = gene_to_id
        self.edge_index = edge_index
        
        # [EN] TODO: translate comment from Chinese.
        self.modalities = sorted(list(set([c.split("::")[0] for c in X.columns])))
        print(f"数据集中检测到的组学: {self.modalities}")

        # [EN] TODO: translate comment from Chinese.
        self.aligned_X_modalities = {}
        for mod in self.modalities:
            mod_cols = [c for c in X.columns if c.startswith(f"{mod}::")]
            mod_genes = [c.split("::")[1] for c in mod_cols]
            df_mod = X[mod_cols]
            df_mod.columns = mod_genes
            # [EN] TODO: translate comment from Chinese.
            self.aligned_X_modalities[mod] = df_mod.reindex(columns=list(gene_to_id.keys()), fill_value=0)

    def len(self):
        return len(self.y)

    def get(self, idx):
        patient_id = self.y.index[idx]
        
        # [EN] TODO: translate comment from Chinese.
        feature_tensors = []
        for mod in self.modalities:
            patient_features = self.aligned_X_modalities[mod].loc[patient_id].values
            feature_tensors.append(torch.tensor(patient_features, dtype=torch.float).unsqueeze(1))
        
        # [EN] TODO: translate comment from Chinese.
        node_features = torch.cat(feature_tensors, dim=1)
        
        label = torch.tensor([self.y.iloc[idx]], dtype=torch.float).view(-1, 1)
        
        return Data(x=node_features, edge_index=self.edge_index, y=label)

# [EN] TODO: translate comment from Chinese.
class SimpleGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1, heads: int = 8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # [EN] TODO: translate comment from Chinese.
        x = global_mean_pool(x, batch)
        
        # [EN] TODO: translate comment from Chinese.
        x = self.classifier(x)
        
        return torch.sigmoid(x)

# [EN] TODO: translate comment from Chinese.
def run_training(X: pd.DataFrame, y: pd.Series, cfg: dict, run_dir: Path) -> dict:
    print(f"--- 正在运行生成的模型: {cfg.get('model')} ---")
    
    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    gmt_path = cfg.get("paths", {}).get("kegg_gmt")
    if not gmt_path:
        # [EN] TODO: translate comment from Chinese.
        gmt_path = "DeepKEGG-agent/DeepKEGG-Agent/data/KEGG_pathways/20230205_kegg_hsa.gmt"
        print(f"[警告] 配置文件中未找到 'kegg_gmt' 路径，使用默认路径: {gmt_path}")
    
    # [EN] TODO: translate comment from Chinese.
    all_feature_genes = sorted(list(set([c.split("::")[1] for c in X.columns])))
    edge_index, gene_to_id = build_global_graph_from_gmt(gmt_path, all_feature_genes)

    # [EN] TODO: translate comment from Chinese.
    cv_cfg = cfg.get("cv", {})
    k_folds = cv_cfg.get("k", 5)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=cv_cfg.get("seed", 42))
    
    fold_metrics = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=k_folds, desc="GAT CV")):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_dataset = PatientGraphDataset(X_train, y_train, gene_to_id, edge_index)
        val_dataset = PatientGraphDataset(X_val, y_val, gene_to_id, edge_index)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # [EN] TODO: translate comment from Chinese.
        model = SimpleGAT(in_channels=train_dataset.num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.BCELoss()

        # [EN] TODO: translate comment from Chinese.
        for epoch in range(50): # 简化训练轮数
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

        # [EN] TODO: translate comment from Chinese.
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                preds = model(data)
                all_preds.append(preds.cpu())
                all_labels.append(data.y.cpu())
        
        preds_tensor = torch.cat(all_preds).numpy().flatten()
        labels_tensor = torch.cat(all_labels).numpy().flatten()
        
        fold_auc = roc_auc_score(labels_tensor, preds_tensor)
        fold_aupr = average_precision_score(labels_tensor, preds_tensor)
        fold_metrics.append({'AUC': fold_auc, 'AUPR': fold_aupr})

    # [EN] TODO: translate comment from Chinese.
    df_metrics = pd.DataFrame(fold_metrics)
    metrics_mean = df_metrics.mean().to_dict()
    
    print("交叉验证完成。平均指标:", metrics_mean)
    pd.DataFrame([metrics_mean]).to_csv(run_dir / "metrics.csv", index=False)

    # [EN] TODO: translate comment from Chinese.
    return metrics_mean
