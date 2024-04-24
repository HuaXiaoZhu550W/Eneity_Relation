import os
import logging
import torch.optim as optim
from Model import BertForDuie
from config import opt
from Dataset import load_data
from Loss import BCELoss
from train import train_epoch
from eval import evaluate
from utils import load_checkpoint, save_model
from torch.utils.tensorboard import SummaryWriter
from LrScheduler import LrSchedulerWithWarmup

# 加载模型
print("======================== 加载模型,预训练模型为: bert-base-chinese ========================")
model = BertForDuie(model_name=opt.model_name, num_classes=opt.num_labels, dropout=opt.dropout,
                    window_sizes=[5, 10, 15])
model.to(opt.device)

# 加载数据集
print("========================== 加载训练数据集: duie_train.json ==============================")
train_loader = load_data(data_dir=opt.data_dir, filename='duie_train.json',
                         label_map=opt.label_map, model_name=opt.model_name,
                         max_len=opt.max_len, num_labels=opt.num_labels,
                         batch_size=opt.batch_size, shuffle=True, num_workers=0)
print("=========================== 加载验证数据集: duie_dev.json ===============================")
dev_loader = load_data(data_dir=opt.data_dir, filename='duie_dev.json',
                       label_map=opt.label_map, model_name=opt.model_name,
                       max_len=opt.max_len, num_labels=opt.num_labels,
                       batch_size=opt.eval_batch, shuffle=False, num_workers=0)

# 损失函数
loss_fn = BCELoss()

# 优化器
optimizer = optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# 学习率调整策略
lr_scheduler = LrSchedulerWithWarmup(optimizer=optimizer, num_warmup_steps=opt.warmup_steps,
                                     num_total_steps=opt.epochs * len(train_loader),
                                     warmup_ratio=5e-5, decay_ratio=1e-3)

# 配置日志
logging.basicConfig(filename=os.path.join(opt.logs_path, "training.log"), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 日志级别为信息级别

# 创建SummaryWriter实例, 记录日志信息用于TensorBorder可视化
writer = SummaryWriter(os.path.join(opt.logs_path, "train-tensorboard"))

# 检查是否存在检查点
start_epoch = 0
if os.listdir(path=opt.checkpoint_path):
    start_epoch = load_checkpoint(filename=os.path.join(opt.checkpoint_path, "model_checkpoint.pth"),
                                  model=model, optimizer=optimizer,
                                  lr_scheduler=lr_scheduler)
else:
    print("没有保存检查点(checkpoint), 需要从头训练！")

print("=========================== 开始训练 ===============================")
for epoch in range(start_epoch, opt.epochs):
    checkpoint = train_epoch(model, train_loader, optimizer, loss_fn, lr_scheduler, opt.device,
                             epoch, writer, opt.checkpoint_path)

    if (epoch + 1) % 2 == 0 or (epoch + 1) == opt.epochs:
        print("=========================== 对模型评估 ===============================")
        precision, recall, f1 = evaluate(model, dev_loader, loss_fn, opt.device, opt.threshold, opt.data_dir,
                                         opt.id2spo, 'duie_dev.json')
        save_model(model, opt.checkpoint_path)  # 模型权重保存
        print(f"precision: {precision:.2%} --||-- recall: {recall:.2%} --||-- F1: {f1:.4f}")
writer.close()
