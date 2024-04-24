import torch
from config import opt
from Model import BertForDuie, LocalAttention
from torch.nn.functional import sigmoid
from transformers import BertTokenizerFast
from utils import convert_sample, post_process


def demo(model, tokenizer, text, max_len, threshold):
    # 将模型设置为评估模式
    model.eval()
    spo_list = None
    label = None
    # 转换输入文本为模型可以处理的格式
    input_ids, attention_mask, seq_len, ts_index, te_index, _ = convert_sample(text, spo_list, label, tokenizer,
                                                                               max_len, opt.label_map)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))

    # 解析模型的输出为实体关系列表
    preds = sigmoid(outputs).squeeze(0)
    spo_list = post_process(text, preds, opt.id2spo, seq_len, ts_index, te_index, threshold)

    return spo_list


if __name__ == "__main__":
    # 预训练模型权重
    pretrained_model = "../bert-base-chinese"
    # 模型权重
    model_path = "../checkpoint/model_weights"

    # 加载模型和权重
    model_duie = torch.load(model_path)
    # 加载分词器
    Tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

    # 示例文本
    Text = "《我的父亲是板凳》是由中国国际电视总公司出品的电视剧，黄文利和张景坤执导，王宝强、陶虹、张子枫、傅程鹏、午马等主演"
    # 执行演示
    spolist = demo(model_duie, Tokenizer, Text, max_len=128, threshold=0.5)
    print(spolist)
