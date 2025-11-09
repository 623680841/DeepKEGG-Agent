# agent/models/self_attention.py
import torch
import torch.nn as nn
import math

class FFN(nn.Module):
    # [EN] TODO: translate comment from Chinese.
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__(); self.relu = nn.ReLU(); self.linear1 = nn.Linear(input_dim, hidden_dim); self.linear2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        residual = x; x = self.linear1(x); x = self.relu(x); x = self.linear2(x); x = residual + x; return x

class bertlayer(nn.Module):
    # [EN] TODO: translate comment from Chinese.
    def __init__(self, embeddingdim, hidden_dim, num_head = 1):
        super(bertlayer, self).__init__(); self.atte_norm = nn.LayerNorm(embeddingdim); self.ffn_norm = nn.LayerNorm(embeddingdim); self.atte = torch.nn.MultiheadAttention(embed_dim = embeddingdim, num_heads = num_head, dropout = 0.0); self.ffn = FFN(embeddingdim, hidden_dim)
    def forward(self, x, x_padding_mask = None):
        residual = x; x = self.atte_norm(x); x, _ = self.atte(x, x, x, key_padding_mask = x_padding_mask);
        x = x + residual; x = x + self.ffn(self.ffn_norm(x)); return x

class bert_feature_extraction(nn.Module):
    # [EN] TODO: translate comment from Chinese.
    def __init__(self, embeddingdim, num_layer):
        super(bert_feature_extraction, self).__init__()
        if embeddingdim > 0: self.layers = nn.ModuleList([bertlayer(embeddingdim, embeddingdim * 4) for _ in range(num_layer)])
        self.embeddingdim = embeddingdim
    def forward(self, x):
        if self.embeddingdim == 0: return torch.zeros(x.shape[0], 0).to(x.device)
        for layer in self.layers: x = layer(x)
        return x.mean(dim = -2)

class TwoLayerRegressionModel(nn.Module):
    # [EN] TODO: translate comment from Chinese.
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerRegressionModel, self).__init__(); self.fc1 = nn.Linear(input_dim, hidden_dim); self.fc2 = nn.Linear(hidden_dim, 1); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x); x = self.relu(x); output = self.fc2(x); return output

class self_attention_model(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, num_layer_a = 2, num_layer_b = 2):
        super(self_attention_model, self).__init__()
        self.layera = bert_feature_extraction(dim_a, num_layer_a)
        self.layerb = bert_feature_extraction(dim_b, num_layer_b)
        
        total_dim = dim_a + dim_b + dim_c
        hidden_dim = max(1, math.floor(total_dim / 2))
        self.header = TwoLayerRegressionModel(total_dim, hidden_dim)
        
        # [EN] TODO: translate comment from Chinese.
        self.dims = {'a': dim_a, 'b': dim_b, 'c': dim_c}

    # [EN] TODO: translate comment from Chinese.
    def forward(self, *args):
        # [EN] TODO: translate comment from Chinese.
        
        processed_features = []
        
        # [EN] TODO: translate comment from Chinese.
        if self.dims['a'] > 0:
            a_out = self.layera(args[0])
            processed_features.append(a_out)
            
        # [EN] TODO: translate comment from Chinese.
        if self.dims['b'] > 0:
            # [EN] TODO: translate comment from Chinese.
            b_idx = 1 if self.dims['a'] > 0 else 0
            b_out = self.layerb(args[b_idx])
            processed_features.append(b_out)
        
        # [EN] TODO: translate comment from Chinese.
        if self.dims['c'] > 0:
            c_idx = (1 if self.dims['a'] > 0 else 0) + (1 if self.dims['b'] > 0 else 0)
            c_tensor = args[c_idx]
            if len(c_tensor.shape) == 3:
                c_out = c_tensor.squeeze(1)
            else:
                c_out = c_tensor
            processed_features.append(c_out)

        # [EN] TODO: translate comment from Chinese.
        concatenated_features = torch.cat(processed_features, dim=-1)
        
        return torch.sigmoid(self.header(concatenated_features))
