import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras import Model, Input

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
            nonzero_ind = np.array(np.nonzero(self.mapp)).T
            self.nonzero_ind = nonzero_ind

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
                                        regularizer=self.bias_regularizer
                                        )
        else:
            self.bias = None

        super(Biological_module, self).build(input_shape)  
      

    def call(self, inputs):
        
        
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
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
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
#%%
class Self_Attention(Layer):
 
    def __init__(self, output_dim,  W_regularizer=None,**kwargs):
        self.output_dim = output_dim
        self.kernel_regularizer = regularizers.get(W_regularizer)
        super(Self_Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
 
        super(Self_Attention, self).build(input_shape)  
 
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
 

        print("K.permute_dimensions(WK.shape",(K.permute_dimensions(WK,[1,0]).shape))
 
        QK =  K.dot(K.permute_dimensions(WK,[1,0]),WQ)
    
 
        QK = QK / (64**0.5)
 
        QK = K.softmax(QK)
 
        print("QK.shape",QK.shape)
 
        V = K.dot(WV,QK)
        
        print(V.shape)
 
        return V

    def get_config(self):
        config = {
          
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),


        }
        base_config = super(Self_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
    def compute_output_shape(self, input_shape):
 
        return (input_shape[0],input_shape[1],self.output_dim)
 
#%%

# [EN] TODO: translate comment from Chinese.
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, Dropout
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras import backend as K

def _make_identity_mapping(n_features: int):
    """恒等映射 -> Biological_module 的 mapp 与 nonzero_ind"""
    mapp = np.eye(n_features, dtype=np.float32)
    nonzero_ind = np.array(np.nonzero(mapp)).T
    return mapp, nonzero_ind

def build_deepkegg_model(input_dims_present: dict,
                         k: int = 64, r: float = 1e-4,
                         use_attention: bool = False):
    """
    input_dims_present: 如 {'mRNA': 1200, 'miRNA': 500, 'SNV': 500}
    k, r: 预留给注意力与正则超参
    use_attention: 先默认 False，跑通后再接入 Self_Attention
    """
    inputs = []
    after_bio = []
    order = []

    for name, dim in input_dims_present.items():
        x = Input(shape=(dim,), name=f"inp_{name}")
        # [EN] TODO: translate comment from Chinese.
        # [EN] TODO: translate comment from Chinese.
        mapp, nz = _make_identity_mapping(dim)
        h = Biological_module(units=dim, mapp=mapp, nonzero_ind=nz,
                              activation='tanh', name=f"bio_{name}")(x)
        inputs.append(x)
        after_bio.append(h)
        order.append(name)

    if len(after_bio) == 1:
        z = after_bio[0]
    else:
        z = Concatenate(axis=-1, name="concat_omics")(after_bio)

    # [EN] TODO: translate comment from Chinese.
    # if use_attention:
    #     z = Self_Attention(output_dim=k, name="self_attn")(z)

    z = Dense(128, activation='tanh', name="fc1")(z)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='tanh', name="fc2")(z)
    out = Dense(1, activation='sigmoid', name="out")(z)

    model = Model(inputs=inputs, outputs=out, name="DeepKEGG-lite")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    return model, order
# [EN] TODO: translate comment from Chinese.
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, metrics

def parse_gmt(gmt_path):
    pathways = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set(g.upper() for g in parts[2:] if g.strip())
            pathways[name] = genes
    return pathways

def parse_kegg_map_long(map_path):
    rows = []
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\t, ]+", line)
            if len(parts) < 2:
                continue
            ent = parts[0].strip()
            pth = parts[1].strip()
            w = 1.0
            if len(parts) >= 3:
                try: w = float(parts[2])
                except: w = 1.0
            rows.append((ent, pth, w))
    return rows

def _strip_prefix(feat):
    # 'mRNA::TP53' -> 'TP53'; 'SNV::ERBB2'->'ERBB2'; 'miRNA::hsa-mir-21-5p'->'hsa-mir-21-5p'
    return feat.split("::", 1)[1] if "::" in feat else feat

def _normalize_gene(name: str) -> str:
    return name.upper()

def _normalize_mirna(name: str) -> str:
    # [EN] TODO: translate comment from Chinese.
    n = name.lower()
    n = n.replace("mirna", "mir")
    n = re.sub(r"(-[35]p)$", "", n)
    return n

def build_feature_pathway_matrix(feature_names, gmt_path, map_path, include_miRNA=True):
    """
    返回:
      M: np.ndarray, shape=[n_features, n_pathways]
      pathways: list[str]
    """
    pathways = parse_gmt(gmt_path)               # pathway -> set(genes UPPER)
    pnames = sorted(pathways.keys())
    p_index = {p:i for i,p in enumerate(pnames)}
    n_features = len(feature_names)
    M = np.zeros((n_features, len(pnames)), dtype=np.float32)

    # gene -> pathway (from gmt)
    gene_to_paths = {}
    for p, genes in pathways.items():
        for g in genes:
            gene_to_paths.setdefault(g, set()).add(p)

    # extra map (miRNA & optional gene)
    rows = parse_kegg_map_long(map_path)
    mir2p, gene_extra = {}, {}
    for ent, pth, w in rows:
        if pth not in p_index:  # 严格只保留出现在 gmt 的通路
            continue
        try: w = float(w)
        except: w = 1.0
        if ent.lower().startswith("hsa-") or "mir" in ent.lower():
            mir2p.setdefault(_normalize_mirna(ent), []).append((pth, w))
        else:
            gene_extra.setdefault(_normalize_gene(ent), []).append((pth, w))

    for i, feat in enumerate(feature_names):
        base = _strip_prefix(feat)
        if feat.startswith("mRNA::") or feat.startswith("SNV::"):
            g = _normalize_gene(base)
            # [EN] TODO: translate comment from Chinese.
            if g in gene_to_paths:
                for p in gene_to_paths[g]:
                    M[i, p_index[p]] = 1.0
            # [EN] TODO: translate comment from Chinese.
            if g in gene_extra:
                for p, w in gene_extra[g]:
                    M[i, p_index[p]] = max(M[i, p_index[p]], float(w))
        elif feat.startswith("miRNA::") and include_miRNA:
            m = _normalize_mirna(base)
            if m in mir2p:
                for p, w in mir2p[m]:
                    M[i, p_index[p]] = max(M[i, p_index[p]], float(w))
        # [EN] TODO: translate comment from Chinese.
    return M, pnames

# [EN] TODO: translate comment from Chinese.
def build_deepkegg_model_from_concat(n_features, M_feat2path, k=64, r=1e-4, use_attention=False, seed=42):
    tf.keras.utils.set_random_seed(seed)
    n_pathways = int(M_feat2path.shape[1])

    inp = layers.Input(shape=(n_features,), name="X")

    bio = Biological_module(
        units=n_pathways,
        mapp=M_feat2path.astype(np.float32),
        W_regularizer=regularizers.l2(r),
        activation='tanh',
        name="biological_projection",
    )(inp)   # (None, n_pathways)

    z = layers.Dropout(0.2)(bio)

    # [EN] TODO: translate comment from Chinese.
    # if use_attention:
    #     x3 = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(z)  # (None, n_pathways, 1)
    #     att = Self_Attention(output_dim=k, W_regularizer=regularizers.l2(r), name="self_attention")(x3)
    #     att = layers.GlobalAveragePooling1D()(att)
    #     z = layers.Concatenate()([z, att])

    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    out = layers.Dense(1, activation="sigmoid", name="prob")(z)

    model = models.Model(inputs=inp, outputs=out, name="DeepKEGG-like")
    # [EN] TODO: translate comment from Chinese.
    opt = optimizers.Adam(learning_rate=1e-3)
    try:
        auc = metrics.AUC(curve="ROC", name="AUC")
        ap  = metrics.AUC(curve="PR",  name="AUPR")
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[auc, ap])
    except Exception:
        model.compile(optimizer=opt, loss="binary_crossentropy")
    return model
