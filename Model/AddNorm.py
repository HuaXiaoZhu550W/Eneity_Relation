import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-12, elementwise_affine=True)

    def forward(self, X, Y):
        return self.layer_norm(X + self.dropout(Y))
