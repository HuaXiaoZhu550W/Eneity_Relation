import torch.nn as nn
from transformers import BertModel
from .AddNorm import AddNorm
from .PositionWiseFFN import PositionWiseFFN


def init_weights(module):
    # 权重初始化
    if isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None and module.bias.requires_grad:
            nn.init.zeros_(module.bias.data)


class BertForDuie(nn.Module):
    def __init__(self, model_name, num_classes=112, dropout=0.1):
        super(BertForDuie, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.addnorm = AddNorm(normalized_shape=768, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_inputs=768, ffn_hiddens=3072)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.classifier = nn.Sequential(nn.Linear(768, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, num_classes))
        # 初始化新添加的模块权重
        self.ffn.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        output = self.addnorm(last_hidden_state, self.ffn(last_hidden_state))
        output = self.classifier(self.dropout(output))
        return output
