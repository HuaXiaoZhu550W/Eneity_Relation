import torch
import torch.nn as nn
import torch.nn.functional as F


# 局部注意力机制
class LocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_sizes, dropout):
        super(LocalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_sizes = window_sizes
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.window_weights = nn.Parameter(torch.ones(len(window_sizes)) / len(window_sizes))  # 初始化为均等权重

    def forward(self, query, key, value, attention_mask):
        batch_size, seq_len, _ = query.size()

        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = self.dropout(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5))

        # 初始化多尺度窗口的注意力权重
        multi_scale_weights = []

        # 遍历不同的窗口大小
        for window_size in self.window_sizes:
            mask = torch.zeros_like(attention_scores)
            for i in range(seq_len):
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                mask[:, :, i, start:end] = 1

            # 应用局部窗口掩码和attention_mask
            attention_scores = attention_scores * mask * attention_mask.unsqueeze(1).unsqueeze(2)
            attention_weights = F.softmax(attention_scores, dim=-1)
            multi_scale_weights.append(attention_weights)

        # 将所有尺度的注意力权重加权平均
        attention_weights = sum(self.window_weights[i] * multi_scale_weights[i] for i in range(len(self.window_sizes)))

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        query = query.transpose(1, 2).view(batch_size, -1, self.embed_dim)

        # 添加残差连接
        output = self.layer_norm(query + context)
        return output
