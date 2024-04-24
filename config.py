"""
超参数文件
"""
import os
import argparse
import json
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-max_len', type=int, default=128, help="模型处理序列的最大长度")
parser.add_argument('-dropout', type=float, default=0.1, help="dropout层的参数p")
parser.add_argument('-lr', type=float, default=5e-5, help="初始学习率")
parser.add_argument('-weight_decay', type=float, default=0.9, help="AdamW优化器的权重衰减参数")
parser.add_argument('-batch_size', type=int, default=32, help="训练批次大小")
parser.add_argument('-eval_batch', type=int, default=64, help="模型评估批次大小")
parser.add_argument('-epochs', type=int, default=5, help="训练轮次")
parser.add_argument('-warmup_steps', type=int, default=4800, help="热身训练步数")
parser.add_argument('-threshold', type=float, default=0.5, help="阈值")
parser.add_argument('-device', type=str,
                    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('-model_name', default="../bert-base-chinese",
                    help="预训练模型权重文件地址")
parser.add_argument('-data_dir', default="../DUIE/", help="数据集存储地址")
parser.add_argument('-checkpoint_path',
                    default="../checkpoint/",
                    help="模型权重保存地址")
parser.add_argument('-logs_path',
                    default="../logs/",
                    help="训练日志保存地址")

opt = parser.parse_args(args=[])

# 读取编码辅助文件
with open(os.path.join("../DUIE", "predicate2id.json"), 'r', encoding='utf8') as f:
    label_map = json.load(f)
opt.label_map = label_map
opt.num_labels = (len(label_map) - 2) * 2 + 2

# 读取解码辅助文件
with open(os.path.join("../DUIE", "id2spo.json"), 'r', encoding='utf8') as f:
    id2spo = json.load(f)
opt.id2spo = id2spo

# 模型权重文件
if not os.path.exists(opt.checkpoint_path):
    os.mkdir(opt.checkpoint_path)

# 训练日志文件
if not os.path.exists(opt.logs_path):
    os.mkdir(opt.logs_path)
