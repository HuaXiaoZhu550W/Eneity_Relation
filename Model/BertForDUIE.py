import torch.nn as nn
from transformers import BertModel
from LocalAttention import LocalAttention


def init_weights(module):
    # 权重初始化
    if isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None and module.bias.requires_grad:
            nn.init.zeros_(module.bias.data)


class BertForDuie(nn.Module):
    def __init__(self, model_name, num_classes=112, dropout=0.1, window_sizes=None):
        super(BertForDuie, self).__init__()
        if window_sizes is None:
            window_sizes = [5, 10, 15]
        self.bert = BertModel.from_pretrained(model_name)
        self.local_attention = LocalAttention(embed_dim=768, num_heads=12, window_sizes=window_sizes, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.Dropout(dropout), nn.Tanh(),
                                        nn.Linear(512, 256), nn.LayerNorm(256), nn.Dropout(dropout), nn.Tanh(),
                                        nn.Linear(256, num_classes))
        # 初始化新添加的模块权重
        self.local_attention.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        # 应用局部注意力
        attended_hidden_state = self.local_attention(last_hidden_state, last_hidden_state, last_hidden_state,
                                                     attention_mask)
        # 使用注意力处理后的输出进行分类
        output = self.classifier(attended_hidden_state)
        return output
