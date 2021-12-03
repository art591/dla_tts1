import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, input_size, n_heads, size_per_head):
        super().__init__()
        self.n_heads = n_heads
        self.size_per_head = size_per_head
        self.queries = nn.Linear(input_size, n_heads * size_per_head)
        self.keys = nn.Linear(input_size, n_heads * size_per_head)
        self.values = nn.Linear(input_size, n_heads * size_per_head)

    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        bs, seq_len = x.shape[:2]
        x_reshaped = x.reshape(-1, x.shape[2])
        queries = self.queries(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)
        keys = self.keys(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)
        values = self.values(x_reshaped).reshape(bs, seq_len, self.n_heads, self.size_per_head).permute(0, 2, 1, 3)

        queries_scaled = queries * 1 / math.sqrt(self.size_per_head)
        before_softmax = torch.einsum('bsij, bskj->bsik', queries_scaled, keys)
        after_softmax = torch.softmax(before_softmax + attention_mask.unsqueeze(1).unsqueeze(1), dim=-1)
        context = torch.einsum('bsik, bskj->bsij', after_softmax, values)
        return context.permute(0, 2, 1, 3).flatten(2)


class IntermidiateLayer(nn.Module):
    def __init__(self, model_size, intermidiate_size, kernel_size, activation, normalization_type, dropout_prob):
        super().__init__()
        self.intermidiate = nn.Sequential(nn.Conv1d(model_size, intermidiate_size, kernel_size=kernel_size[0], padding='same'),
                                          activation(),
                                          nn.Conv1d(intermidiate_size, model_size, kernel_size=kernel_size[1], padding='same'))
        self.layer_norm = nn.LayerNorm(model_size)
        self.normalization_type = normalization_type
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        if self.normalization_type == 'pre':
            to_add = self.layer_norm(x)
            to_add = to_add.permute(0, 2, 1)
            to_add = self.dropout(self.intermidiate(to_add)).permute(0, 2, 1)
            x = x + to_add
        elif self.normalization_type == 'post':
            x = x.permute(0, 2, 1)
            to_add = self.dropout(self.intermidiate(x))
            x = (x + to_add).permute(0, 2, 1)
            x = self.layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_size,
                 intermidiate_size,
                 itermidiate_kernel_size,
                 activation,
                 n_heads,
                 size_per_head,
                 normalization_type,
                 dropout_prob):
        super().__init__()
        self.self_attention = SelfAttention(model_size, n_heads, size_per_head)
        self.context_linear = nn.Linear(n_heads * size_per_head, model_size)
        self.intermidiate_size = intermidiate_size
        self.intermediate_layer = IntermidiateLayer(model_size,
                                                    intermidiate_size,
                                                    itermidiate_kernel_size,
                                                    activation,
                                                    normalization_type,
                                                    dropout_prob)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(model_size)
        self.normalization_type = normalization_type
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        attention_output = self.self_attention(x, attention_mask)
        if self.normalization_type == 'pre':
            to_add = self.dropout(self.context_linear(self.layer_norm(attention_output)))
            x = x + to_add
        elif self.normalization_type == 'post':
            to_add = self.dropout(self.context_linear(attention_output))
            x = self.layer_norm(x + to_add)
        x = self.intermediate_layer(x)
        return x, attention_mask












