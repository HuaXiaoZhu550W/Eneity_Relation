import os
import torch
from tqdm import tqdm
from torch.nn.functional import sigmoid
from utils import read_data, decoding, write_prediction_results, get_precision_recall_f1


# 评估函数

def evaluate(model, dataloader, loss_fn, device, threshold, data_dir, id2spo, dev_file):
    # 读取验证集
    dev_data = read_data(data_dir=data_dir, filename=dev_file)

    model.to(device)
    model.eval()
    current_idx = 0  # 当前数据起始位置
    total_loss = 0.0
    formatted_outputs = []

    pbar = tqdm(desc="eval", total=len(dataloader), postfix=dict, mininterval=0.4)
    for iteration, batch in enumerate(dataloader):
        inputs, attention_mask, seq_len, token_start_index, token_end_index, labels = [X.to(device) for X in batch]
        with torch.no_grad():
            outputs = model(inputs, attention_mask)
            mask = (inputs != 101).logical_and((inputs != 102)).logical_and((inputs != 0))
            loss = loss_fn(outputs, labels, mask)
            total_loss += loss.item()
            preds = sigmoid(outputs)

            # 获得预测结果
            formatted_outputs.extend(decoding(dev_data[current_idx: current_idx + outputs.shape[0]],
                                              id2spo,
                                              preds.to("cpu").numpy(),
                                              seq_len.to("cpu").numpy(),
                                              token_start_index.to("cpu").numpy(),
                                              token_end_index.to("cpu").numpy(),
                                              threshold))
            current_idx += len(outputs)
        pbar.set_postfix(**{'Loss': f"{total_loss / (iteration + 1):.6f}"})
        pbar.update(1)

    # 将预测结果写入json文件
    predict_file_path = write_prediction_results(formatted_outputs=formatted_outputs,
                                                 file_path=os.path.join("../result/", 'predict_eval.json'))
    # 计算精准率，召回率，F1
    precision, recall, f1 = get_precision_recall_f1(predict_file=predict_file_path, dev_file=os.path.join(data_dir, dev_file))

    return precision, recall, f1
