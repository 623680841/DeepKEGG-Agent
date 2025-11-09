# agent/deepkegg_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, activations, regularizers, backend as K
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm
import os
from pathlib import Path

# [EN] TODO: translate comment from Chinese.

# [EN] TODO: translate comment from Chinese.
class Biological_module(Layer):
    def __init__(self, units, mapp=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True,bias_initializer='zeros', bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        self.units = units
        self.activation = activation
        self.mapp = mapp
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        super(Biological_module, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        if not self.mapp is None:
            self.mapp = self.mapp.astype(np.float32)
        if self.nonzero_ind is None:
            self.nonzero_ind = np.array(np.nonzero(self.mapp)).T
        self.kernel_shape = (input_dim, self.units)
        nonzero_count = self.nonzero_ind.shape[0]
        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None
        super(Biological_module, self).build(input_shape)

    def call(self, inputs):
        if self.units == 0:
            return tf.zeros((tf.shape(inputs)[0], 0))
            
        trans = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))
        output = K.dot(inputs, trans)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def get_config(self):
        config = {
            'units': self.units, 'activation': self.activation, 'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(Biological_module, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class Self_Attention(Layer):
    def __init__(self, output_dim, W_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_regularizer = regularizers.get(W_regularizer)
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        QK = tf.matmul(WQ, WK, transpose_b=True)
        QK = QK / (tf.cast(K.shape(WV)[-1], tf.float32)**0.5)
        QK = K.softmax(QK)
        V = tf.matmul(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

# [EN] TODO: translate comment from Chinese.
def _abs(p, project_root):
    if not p: return None
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp)

# agent/deepkegg_model.py

# [EN] TODO: translate comment from Chinese.
def prepare_pathway_data(X: pd.DataFrame, cfg: dict):
    """
    根据已加载的X数据和配置，创建通路掩码。
    这个函数不再自己加载任何组学数据。
    """
    print("  - [DeepKEGG] Preparing pathway data...")
    paths = cfg.get("paths", {})
    modalities = cfg.get("modalities", [])
    base_dir = Path(os.environ.get("BASE_DIR", "/tmp/DeepKEGG-master"))

    # [EN] TODO: translate comment from Chinese.
    gmt_path_str = paths.get('kegg_gmt') or paths.get('gmt') or "KEGG_pathways/20230205_kegg_hsa.gmt"
    mirna_map_str = paths.get('kegg_map_long') or paths.get('mirna_map') or "KEGG_pathways/kegg_anano.txt"
    
    gmt_path = base_dir / gmt_path_str
    final_mirna_map_path = base_dir / mirna_map_str

    # [EN] TODO: translate comment from Chinese.
    print(f"  - Loading gene-pathway map from: {gmt_path}")
    if not gmt_path.exists(): raise FileNotFoundError(f"KEGG GMT file not found: {gmt_path}")
    with open(gmt_path, 'r', encoding='utf-8') as f:
        paways_genes_dict = {p[0]: p[2:] for p in (line.strip().split('\t') for line in f) if len(p) > 2}

    paways_mirna_dict = {}
    if "miRNA" in modalities:
        if not final_mirna_map_path.exists(): raise FileNotFoundError(f"miRNA map file not found: {final_mirna_map_path}")
        print(f"  - Loading miRNA-pathway map from: {final_mirna_map_path}")
        with open(final_mirna_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or '|' not in line or ',' not in line: continue
                parts = line.strip().split(',', 1)
                path_info, mirnas_str = parts[0], parts[1]
                path_name, path_id_num = path_info.split('|', 1)
                paways_mirna_dict[f"{path_id_num}_{path_name}"] = [m for m in mirnas_str.split(',') if m]

    # [EN] TODO: translate comment from Chinese.
    gene_pathway_keys = set(paways_genes_dict.keys())
    if "miRNA" in modalities and paways_mirna_dict:
        mirna_pathway_keys = set(paways_mirna_dict.keys())
        union_kegg = sorted(list(gene_pathway_keys.intersection(mirna_pathway_keys)))
        print(f"  - Found {len(gene_pathway_keys)} gene pathways and {len(mirna_pathway_keys)} miRNA pathways.")
        print(f"  - Using the intersection of {len(union_kegg)} pathways.")
    else:
        union_kegg = sorted(list(gene_pathway_keys.keys()))
        print(f"  - Using all {len(union_kegg)} gene pathways.")
    if not union_kegg: raise ValueError("No common pathways found.")

    # [EN] TODO: translate comment from Chinese.
    gene_pathway_bp_dfs = []
    for mod_name in ["SNV", "mRNA", "miRNA"]:
        if mod_name in modalities:
            mod_prefix = f"{mod_name}::"
            mod_cols = [c for c in X.columns if c.startswith(mod_prefix)]
            if not mod_cols: continue
            
            df_mod_features = [c.replace(mod_prefix, "") for c in mod_cols]
            
            pathway_map = np.zeros((len(union_kegg), len(df_mod_features)))
            
            source_dict = paways_genes_dict if mod_name != "miRNA" else paways_mirna_dict
            
            for p_idx, p_name in enumerate(union_kegg):
                features_in_pathway = source_dict.get(p_name, [])
                feature_indices = [df_mod_features.index(f) for f in features_in_pathway if f in df_mod_features]
                if feature_indices:
                    pathway_map[p_idx, feature_indices] = 1
            
            bp_df = pd.DataFrame(pathway_map, index=union_kegg, columns=df_mod_features)
            gene_pathway_bp_dfs.append(bp_df)
        
    print(f"  - Pathway matrices created. Total masks: {len(gene_pathway_bp_dfs)}")
    return gene_pathway_bp_dfs
    
def create_deepkegg_model(omics_shapes: dict, pathway_masks: list):
    """根据您的 Notebook 创建 Keras 模型"""
    inputs = []
    biological_layers = []
    
    mask_idx = 0
    num_pathways = pathway_masks[0].shape[0] if pathway_masks else 0

    # [EN] TODO: translate comment from Chinese.
    if "SNV" in omics_shapes:
        snv_input = Input(shape=(omics_shapes["SNV"][1],), name='snv_input')
        inputs.append(snv_input)
        h0_snv = Biological_module(num_pathways, mapp=pathway_masks[mask_idx].values.T, name='h0_snv', W_regularizer=l2(0.001))(snv_input)
        biological_layers.append(h0_snv)
        mask_idx += 1
        
    if "mRNA" in omics_shapes:
        mrna_input = Input(shape=(omics_shapes["mRNA"][1],), name='mrna_input')
        inputs.append(mrna_input)
        h0_mrna = Biological_module(num_pathways, mapp=pathway_masks[mask_idx].values.T, name='h0_mrna', W_regularizer=l2(0.001))(mrna_input)
        biological_layers.append(h0_mrna)
        mask_idx += 1

    if "miRNA" in omics_shapes:
        mirna_input = Input(shape=(omics_shapes["miRNA"][1],), name='mirna_input')
        inputs.append(mirna_input)
        h0_mirna = Biological_module(num_pathways, mapp=pathway_masks[mask_idx].values.T, name='h0_mirna', W_regularizer=l2(0.001))(mirna_input)
        biological_layers.append(h0_mirna)

    if not biological_layers or num_pathways == 0:
        raise ValueError("模型无法构建，因为没有找到共同的通路或生物学层。请检查您的通路文件。")

    concatenated = Lambda(lambda x: tf.stack(x, axis=1))(biological_layers)
    attention_output = Self_Attention(output_dim=1)(concatenated)
    flatten_output = tf.keras.layers.Flatten()(attention_output)
    
    dropout_layer = tf.keras.layers.Dropout(0.3)(flatten_output)
    output = tf.keras.layers.Dense(2, activation='softmax')(dropout_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate_deepkegg(X, y, cfg, pathway_masks, run_dir):
    """封装 DeepKEGG 模型的完整训练和评估流程"""
    cv_cfg = cfg.get("cv", {})
    kfold = cv_cfg.get("k", 5)
    seed = cv_cfg.get("seed", 42)
    modalities = cfg.get("modalities", [])
    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)

    X_dict = {}
    omics_shapes = {}
    
    # [EN] TODO: translate comment from Chinese.
    mod_name_map = {"SNV": "SNV", "mRNA": "mRNA", "miRNA": "miRNA"}
    input_names_ordered = []

    for mod in mod_name_map:
        if mod in modalities:
            mod_prefix = f"{mod}::"
            mod_cols = [c for c in X.columns if c.startswith(mod_prefix)]
            if mod_cols:
                X_mod = X[mod_cols].copy()
                X_mod.columns = [c.replace(mod_prefix, "") for c in X_mod.columns]
                
                input_name = f"{mod.lower()}_input"
                X_dict[input_name] = X_mod
                input_names_ordered.append(input_name)
                omics_shapes[mod] = X_mod.shape
    
    fold_rows = []
    all_pred = np.zeros(len(y), dtype=int)
    all_prob = np.zeros(len(y), dtype=float)
    final_trained_model = None # <-- [关键修复1] 初始化变量

    for fi, (tr_idx, te_idx) in enumerate(tqdm(skf.split(X, y), total=kfold, desc="DeepKEGG CV"), start=1):
        X_train_list = [X_dict[name].iloc[tr_idx] for name in input_names_ordered]
        y_train = y.iloc[tr_idx]
        X_test_list = [X_dict[name].iloc[te_idx] for name in input_names_ordered]
        y_test = y.iloc[te_idx]

        K.clear_session()
        
        model = create_deepkegg_model(omics_shapes, pathway_masks)
        
        # [EN] TODO: translate comment from Chinese.
        epochs = cfg.get("model_params", {}).get("epochs", 50)
        model.fit(X_train_list, y_train, epochs=epochs, batch_size=32, verbose=0)
        
        y_prob_full = model.predict(X_test_list)
        y_prob_fold = y_prob_full[:, 1]
        y_pred_fold = np.argmax(y_prob_full, axis=1)

        fold_auc = roc_auc_score(y_test, y_prob_fold)
        fold_aupr = average_precision_score(y_test, y_prob_fold)
        fold_acc = accuracy_score(y_test, y_pred_fold)
        fold_f1 = f1_score(y_test, y_pred_fold)
        
        fold_rows.append({"fold": fi, "AUC": fold_auc, "AUPR": fold_aupr, "ACC": fold_acc, "F1": fold_f1})
        all_pred[te_idx] = y_pred_fold
        all_prob[te_idx] = y_prob_fold
    
    # [EN] TODO: translate comment from Chinese.
    final_trained_model = model

    df_folds = pd.DataFrame(fold_rows)
    metrics_mean = df_folds.mean(numeric_only=True).dropna().to_dict()

    df_folds.to_csv(run_dir / "metrics_per_fold.csv", index=False)
    pd.DataFrame([metrics_mean]).to_csv(run_dir / "metrics.csv", index=False)

    out_pred = pd.DataFrame({
        "sample_id": X.index, "y_true": y.values,
        "y_pred": all_pred, "y_prob": all_prob
    })
    out_pred.to_csv(run_dir / "predictions.csv", index=False)
    
    # [EN] TODO: translate comment from Chinese.
    return final_trained_model, metrics_mean