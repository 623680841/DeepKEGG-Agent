# agent/models/lstm.py
import torch
import torch.nn as nn
import math

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerLSTM, self).__init__()
        if input_size > 0:
            self.lstm = nn.LSTM(input_size, input_size * 3, num_layers=2, batch_first=True)
            self.fc = nn.Linear(input_size * 3, input_size)
        self.input_size = input_size

    def forward(self, x):
        if self.input_size == 0:
            return torch.zeros(x.shape[0], 0).to(x.device)
        lstm_out, _ = self.lstm(x)
        mean_pooled = torch.mean(lstm_out, dim=1)
        out = self.fc(mean_pooled)
        return out

class TwoLayerRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

class lstm_model(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, num_layer_a = 2, num_layer_b = 2):
        super(lstm_model, self).__init__()
        self.layer_a = TwoLayerLSTM(dim_a)
        self.layer_b = TwoLayerLSTM(dim_b)
        
        total_dim = dim_a + dim_b + dim_c
        hidden_dim = max(1, math.floor(total_dim / 2))
        self.header = TwoLayerRegressionModel(total_dim, hidden_dim)
        
        # [EN] TODO: translate comment from Chinese.
        self.dims = {'a': dim_a, 'b': dim_b, 'c': dim_c}

    # [EN] TODO: translate comment from Chinese.
    def forward(self, *args):
        processed_features = []
        arg_idx = 0 # 用于追踪当前处理到哪个输入张量

        # [EN] TODO: translate comment from Chinese.
        if self.dims['a'] > 0:
            a_out = self.layer_a(args[arg_idx])
            processed_features.append(a_out)
            arg_idx += 1
            
        # [EN] TODO: translate comment from Chinese.
        if self.dims['b'] > 0:
            b_out = self.layer_b(args[arg_idx])
            processed_features.append(b_out)
            arg_idx += 1
        
        # [EN] TODO: translate comment from Chinese.
        if self.dims['c'] > 0:
            c_tensor = args[arg_idx]
            if len(c_tensor.shape) == 3:
                c_out = c_tensor.squeeze(1)
            else:
                c_out = c_tensor
            processed_features.append(c_out)

        # [EN] TODO: translate comment from Chinese.
        concatenated_features = torch.cat(processed_features, dim=-1)
        
        return torch.sigmoid(self.header(concatenated_features))
