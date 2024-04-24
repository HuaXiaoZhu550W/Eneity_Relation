import torch
from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from utils import read_data, convert_sample
from .dataset import DUIEDataset


def load_data(data_dir, filename, label_map, model_name, max_len, num_labels, batch_size, shuffle, num_workers):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)  # 分词器
    data = read_data(data_dir=data_dir, filename=filename)  # 读取数据
    token_ids = []  # 存储每个token的id
    attention_mask = []
    seq_len = []  # 数据的真实长度
    token_start_index = []  # token在text中起始的index
    token_end_index = []  # token在text中结束的index
    labels = []  # 实体关系联合抽取标签
    for d in tqdm(data):
        text = d['text']  # 原始文本
        spo_list = d['spo_list'] if 'spo_list' in d.keys() else None  # 当前文本中存在的spo三元组
        label = torch.zeros((max_len, num_labels))  # 实体关系联合抽取标签

        t_ids, a_mask, s_len, ts_index, te_index, label = convert_sample(text, spo_list, label, tokenizer, max_len, label_map)
        token_ids.append(t_ids)
        attention_mask.append(a_mask)
        seq_len.append(s_len)
        token_start_index.append(ts_index)
        token_end_index.append(te_index)
        labels.append(label)

    dataset = DUIEDataset(token_ids, attention_mask, seq_len, token_start_index, token_end_index, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, drop_last=True)
    return dataloader
