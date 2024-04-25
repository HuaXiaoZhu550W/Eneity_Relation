import os
import torch
import logging
from tqdm import tqdm
from torch.nn.utils import clip_grad_value_


# 训练函数
def train_epoch(model, trainloader, optimizer, loss_fn, lr_scheduler, device, epoch, writer, checkpoint_path):
    model = model.to(device)
    model.train()
    total_loss = 0.0
    step = 5000
    checkpoint = {}  # 存储模型相关参数, 方便断点续训练
    iterations = len(trainloader)

    # 创建进度条
    pbar = tqdm(desc=f"epoch{epoch + 1}", total=iterations, postfix=dict, mininterval=0.4)
    for iteration, batch in enumerate(trainloader):
        inputs, attention_mask, seq_len, token_start_index, token_end_index, labels = [X.to(device) for X in batch]
        optimizer.zero_grad()  # 梯度清零
        preds = model(inputs, attention_mask)  # 前向传播
        # ['CLS'],['SEP'],['PAD']不参与损失计算
        # preds = torch.nan_to_num(preds, nan=1e-20)
        mask = (inputs != 101).logical_and((inputs != 102)).logical_and((inputs != 0))
        loss = loss_fn(preds, labels, mask)  # 计算损失
        loss.backward()
        clip_grad_value_(model.parameters(), clip_value=0.5)  # 梯度剪裁
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        # 记录当前iteration的损失和学习率
        current_lr = optimizer.param_groups[0]['lr']  # 当前学习率
        mean_loss = total_loss / (iteration + 1)  # 当前平均损失
        logging.info(f'Epoch {epoch + 1}, '
                     f'Iteration {iteration + 1}/{iterations} - Loss: {mean_loss}, '
                     f'Learning Rate: {current_lr}')

        writer.add_scalar('Loss/train', mean_loss, epoch * len(trainloader) + iteration)
        writer.add_scalar('Learning Rate', current_lr, epoch * len(trainloader) + iteration)

        pbar.set_postfix(**{'Loss': f"{total_loss / (iteration + 1):.6f}",
                            'lr': f"{current_lr:.8f}",
                            'device': f"{inputs.device}"}
                         )
        pbar.update(1)
        if (iteration+1) % step == 0 or (iteration+1) == iterations:
            # 存储模型相关参数, 方便断点续训练
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, "model_checkpoint.pth"))
    pbar.close()
    return checkpoint
