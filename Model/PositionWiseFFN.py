import torch.nn as nn


class PositionWiseFFN(nn.Module):
    """
    前馈神经网络
    输入 ffn_inputs (base 768, large 1024)
    中间层 ffn_hiddens (base 3072, large 4096)
    输出 ffn_outputs (base 768, large 1024)
    """

    def __init__(self, ffn_inputs, ffn_hiddens, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(in_features=ffn_inputs, out_features=ffn_hiddens)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(in_features=ffn_hiddens, out_features=ffn_inputs)

    def forward(self, X):
        return self.dense2(self.gelu(self.dense1(X)))
