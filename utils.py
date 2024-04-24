import os
import json
import torch
import numpy as np
from calculation_indicators import calculate_pr


def read_data(data_dir, filename):
    """ 读取文件 """
    with open(os.path.join(data_dir, filename), 'r', encoding='utf8') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    return data


def get_index(sub_obj, tokens):
    """
    获取实体在分词后的tokens中的起始index
    """
    start = 0
    indexes = []
    while sub_obj[0] in tokens[start:]:
        index = tokens.index(sub_obj[0], start)
        if tokens[index:index + len(sub_obj)] == sub_obj:
            indexes.append(index + 1)  # 返回起始索引
        start = index + 1
        if start > len(tokens) - len(sub_obj):
            break

    return indexes


def marker(sub_obj, tokens, label, label_map, predicate, flag, key=None):
    """
    实体对应的label标记为1
    """
    indexes = get_index(sub_obj, tokens)  # 寻找实体在分词结果中的索引
    if flag[0] == 1:
        for index in indexes:
            if flag[1] == 's':
                label[index, label_map[predicate + '_' + key]] = 1
            else:
                label[index, label_map[predicate + '_' + key] + len(label_map)] = 1

            label[index + 1: index + len(sub_obj), label_map["I"]] = 1
    else:
        for index in indexes:
            if flag[1] == 's':
                label[index, label_map[predicate]] = 1
            else:
                label[index, label_map[predicate] + len(label_map)] = 1
            label[index + 1: index + len(sub_obj), label_map["I"]] = 1


def create_labels(label, spo_list, tokens, tokenizer, label_map):
    """
    生成label
    """
    for spo in spo_list:
        sub = tokenizer.tokenize(spo['subject'])
        obj = [tokenizer.tokenize(v) for v in spo['object'].values() if tokenizer.tokenize(v)]  # 一个list,存储所有的object
        predicate = spo['predicate']

        if len(sub) * len(obj) > 0:  # subject和object都不为空
            if len(obj) > 1:  # 如果同一个subject存在多个obj, 就需要 predicate+object_type
                for key, value in zip(spo['object'].keys(), obj):
                    flag = [1, 's']  # 切换标记模式's', 标记subject
                    marker(sub, tokens, label, label_map, predicate, flag, key)

                    flag = [1, 'o']  # 标记完subject后, 切换为'o'标记object
                    marker(value, tokens, label, label_map, predicate, flag, key)
            else:  # 如果obj中只有一个
                if predicate in label_map.keys():
                    flag = [0, 's']  # 切换标记模式
                    marker(sub, tokens, label, label_map, predicate, flag)

                    flag = [0, 'o']  # 标记完subject后, 切换为'o'标记object
                    marker(obj[0], tokens, label, label_map, predicate, flag)
                else:  # 如果predicate不在label_map中, 例如：上映时间不在label_map中, 就需要predicate_object_type
                    for key in spo['object'].keys():
                        flag = [1, 's']  # 切换标记模式's', 标记subject
                        marker(sub, tokens, label, label_map, predicate, flag, key)

                        flag = [1, 'o']  # 标记完subject后, 切换为'o'标记object
                        marker(obj[0], tokens, label, label_map, predicate, flag, key)
    # 标记非实体O
    label[torch.all(label == 0, dim=-1), label_map["O"]] = 1

    return label


def convert_sample(text, spo_list, label, tokenizer, max_len, label_map):
    sub_text = []
    for char in text:
        if char.isalpha():
            sub_text.append(char.lower())
        else:
            sub_text.append(char)

    token_start_index = []
    token_end_index = []
    tokens = []
    text_row = ""
    for i, char in enumerate(sub_text):
        token = tokenizer.tokenize(char)
        text_row += char
        for t in token:
            token_start_index.append(len(text_row) - len(char))  # 当前char在text中的起始位置
            token_end_index.append(len(text_row) - 1)  # 当前char在text中的结束位置
            tokens.append(t)
            if len(tokens) >= max_len - 2:  # 保证["CLS"] + tokens + ["SEP"]不超过max_len
                break
        else:  # 当for循环正常结束时(没有遇到break), 则执行else中的代码
            continue
        break  # 如果tokens超过最大长度,直接跳出循环，不再继续从sub_text中拿取字符
    seq_len = len(tokens)  # tokens的真实长度

    # 生成标签
    if spo_list is not None and label is not None:
        label = create_labels(label, spo_list, tokens, tokenizer, label_map)

    # 生成token_ids
    tokens = ["[CLS]"] + tokens + ["[SEP]"] + ["[PAD]"] * (max_len - seq_len - 2)
    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    # 生成attention_mask
    attention_mask = (token_ids != 0).type(torch.int)

    # 生成token_start_index 和 token_end_index, 非文本内容的index，标记为-1
    token_start_index = torch.tensor([-1] + token_start_index + [-1] * (max_len - seq_len - 1))
    token_end_index = torch.tensor([-1] + token_end_index + [-1] * (max_len - seq_len - 1))

    # 生成seq_len
    seq_len = torch.tensor(seq_len)

    return token_ids, attention_mask, seq_len, token_start_index, token_end_index, label


def find_entity(text_raw, entity_id, predictions, token_start_index, token_end_index):
    """
    给定实体头部id,查找实体信息
    """
    entity_list = []
    for i in range(len(predictions)):
        if [entity_id] in predictions[i]:  # 遍历每个token的预测分类结果，找到当前实体头部所在位置
            j = 0  # 统计实体尾部的数量
            while i + j + 1 < len(predictions):
                if [55] in predictions[i + j + 1]:
                    j += 1
                else:
                    break

            entity = ''.join(text_raw[token_start_index[i]: token_end_index[i + j] + 1])
            entity_list.append(entity)

    return list(set(entity_list))


def decoding(dev_batch, id2spo, preds, seq_len, token_start_index, token_end_index, threshold):
    """
    将preds转成spo
    """
    formatted_outputs = []  # 收集预测结果(spo三元组)
    for i, (sample, preds, seq_len, token_start_index, token_end_index) in enumerate(
            zip(dev_batch, preds, seq_len, token_start_index, token_end_index)):

        # 去掉 "[CLS]"、"[SEP]"、"[PAD]", 并且筛选出大于threshold的
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0
        preds = preds[1: seq_len + 1]  # 去除预测结果中"[CLS]"、"[SEP]"、"[PAD]"
        token_start_index = token_start_index[1: seq_len + 1]
        token_end_index = token_end_index[1: seq_len + 1]

        predictions = []
        for token in preds:  # 遍历每一个token(shape:[112, ]),如果存在等于1的, 就记录下该位置的类别
            predictions.append([[i] for item in np.argwhere(token == 1).tolist() for i in item])

        # 将模型的预测结果转换成与数据集示例中相同格式的spo
        formatted_instance = {}
        text_raw = sample['text']
        cr_label = [2, 9, 15, 23, 32]  # 复杂关系标签
        cra_label = [3, 10, 16, 24, 25, 26, 33]  # 复杂关系中的隶属关系的标签

        # 将模型的预测结果拉平并从中检索所有有效的主体(subject)
        flatten_predictions = [i for ids in predictions for item in ids for i in item]

        subject_id = []
        for cls_label in list(set(flatten_predictions)):
            if 0 < cls_label < 56 and (cls_label + 57) in flatten_predictions:  # 收集subject的头部id
                subject_id.append(cls_label)
        subject_id = list(set(subject_id))

        # 通sub_id, 查找所有spo
        spo_list = []
        for sub_id in subject_id:
            if sub_id in cra_label:  # 如果是复杂关系, 暂不处理, 在else中处理
                continue
            if sub_id not in cr_label:  # 如果不是复杂关系的隶属关系, 就直接获取subject和object
                subjects = find_entity(text_raw, sub_id, predictions,
                                       token_start_index, token_end_index)
                objects = find_entity(text_raw, sub_id + 57, predictions,
                                      token_start_index, token_end_index)
                for sub in subjects:
                    for obj in objects:
                        spo_list.append({"predicate": id2spo['predicate'][sub_id],
                                         "object_type": {
                                             '@value': id2spo['object_type'][sub_id]
                                         },
                                         "subject_type": id2spo['subject_type'][sub_id],
                                         "object": {'@value': obj},
                                         "subject": sub
                                         })
            else:  # 遍历所有复杂关系并查找相应的附属对象
                subjects = find_entity(text_raw, sub_id, predictions,
                                       token_start_index, token_end_index)
                objects = find_entity(text_raw, sub_id + 57, predictions,
                                      token_start_index, token_end_index)
                for sub in subjects:
                    for obj in objects:
                        object_dict = {'@value': obj}
                        object_type_dict = {'@value': id2spo['object_type'][sub_id].split('_')[0]}

                        if sub_id in [2, 9, 15, 32] and sub_id + 1 in subject_id:
                            # 复杂关系有2个附属object对象
                            sub_ida = sub_id + 1
                            object_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                find_entity(text_raw, sub_ida + 57, predictions,
                                            token_start_index, token_end_index)[0]
                            object_type_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                id2spo['object_type'][sub_ida].split('_')[0]

                        elif sub_id == 23:
                            # 复杂关系有4个附属object对象
                            for sub_ida in [24, 25, 26]:
                                if sub_ida in subject_id:
                                    object_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                        find_entity(text_raw, sub_ida + 57, predictions,
                                                    token_start_index, token_end_index)[0]
                                    object_type_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                        id2spo['object_type'][sub_ida].split('_')[0]

                        spo_list.append({"predicate": id2spo['predicate'][sub_id],
                                         "object_type": object_type_dict,
                                         "subject_type": id2spo['subject_type'][sub_id],
                                         "object": object_dict,
                                         "subject": sub
                                         })

        formatted_instance['text'] = sample['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def post_process(text, preds, id2spo, seq_len, token_start_index, token_end_index, threshold):
    """ 后处理函数, 将preds转成spo """

    # 去掉 "[CLS]"、"[SEP]"、"[PAD]", 并且筛选出大于threshold的
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    preds = preds[1: seq_len + 1]  # 去除预测结果中"[CLS]"、"[SEP]"、"[PAD]"
    token_start_index = token_start_index[1: seq_len + 1]
    token_end_index = token_end_index[1: seq_len + 1]

    predictions = []
    for token in preds:  # 遍历每一个token(shape:[112, ]),如果存在等于1的, 就记录下该位置的类别
        predictions.append([[i] for item in np.argwhere(token == 1).tolist() for i in item])

    # 将模型的预测结果转换成与数据集示例中相同格式的spo
    formatted_instance = {}
    cr_label = [2, 9, 15, 23, 32]  # 复杂关系标签
    cra_label = [3, 10, 16, 24, 25, 26, 33]  # 复杂关系中的隶属关系的标签

    # 将模型的预测结果拉平并从中检索所有有效的主体(subject)
    flatten_predictions = [i for ids in predictions for item in ids for i in item]
    subject_id = []
    for cls_label in list(set(flatten_predictions)):
        if 0 <= cls_label < 56 and (cls_label + 57) in flatten_predictions:  # 收集subject的头部id
            subject_id.append(cls_label)
    subject_id = list(set(subject_id))

    # 通sub_id, 查找所有spo
    spo_list = []
    for sub_id in subject_id:
        if sub_id in cra_label:  # 如果是复杂关系, 暂不处理, 在else中处理
            continue
        if sub_id not in cr_label:  # 如果不是复杂关系的隶属关系, 就直接获取subject和object
            subjects = find_entity(text, sub_id, predictions,
                                   token_start_index, token_end_index)
            objects = find_entity(text, sub_id + 57, predictions,
                                  token_start_index, token_end_index)
            for sub in subjects:
                for obj in objects:
                    spo_list.append({"predicate": id2spo['predicate'][sub_id],
                                     "object_type": {
                                         '@value': id2spo['object_type'][sub_id]
                                     },
                                     "subject_type": id2spo['subject_type'][sub_id],
                                     "object": {'@value': obj},
                                     "subject": sub
                                     })
        else:  # 遍历所有复杂关系并查找相应的附属对象
            subjects = find_entity(text, sub_id, predictions,
                                   token_start_index, token_end_index)
            objects = find_entity(text, sub_id + 57, predictions,
                                  token_start_index, token_end_index)
            for sub in subjects:
                for obj in objects:
                    object_dict = {'@value': obj}
                    object_type_dict = {'@value': id2spo['object_type'][sub_id].split('_')[0]}

                    if sub_id in [2, 9, 15, 32] and sub_id + 1 in subject_id:
                        # 复杂关系有2个附属object对象
                        sub_ida = sub_id + 1
                        object_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                            find_entity(text, sub_ida + 57, predictions,
                                        token_start_index, token_end_index)[0]
                        object_type_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                            id2spo['object_type'][sub_ida].split('_')[0]

                    elif sub_id == 23:
                        # 复杂关系有4个附属object对象
                        for sub_ida in [24, 25, 26]:
                            if sub_ida in subject_id:
                                object_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                    find_entity(text, sub_ida + 57, predictions, token_start_index,
                                                token_end_index)[0]
                                object_type_dict[id2spo['object_type'][sub_ida].split('_')[1]] = \
                                    id2spo['object_type'][sub_ida].split('_')[0]

                    spo_list.append({"predicate": id2spo['predicate'][sub_id],
                                     "object_type": object_type_dict,
                                     "subject_type": id2spo['subject_type'][sub_id],
                                     "object": object_dict,
                                     "subject": sub
                                     })

        formatted_instance['text'] = text
        formatted_instance['spo_list'] = spo_list

    return formatted_instance


def write_prediction_results(formatted_outputs, file_path):
    """将预测结果写入.json中"""
    with open(file_path, 'w', encoding='utf-8') as f:  # 将预测结果写入到文件中
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
    return file_path


def get_precision_recall_f1(predict_file, dev_file):
    """ 计算指标"""
    alias_filename = ""  # 别名字典地址
    ret_info = calculate_pr(predict_file, dev_file, alias_filename)

    # 从ret_info中获取precision, recall, f1_score
    if "data" in ret_info.keys():
        metrics = {item['name']: item['value'] if item['value'] is not None else 0 for item in ret_info['data']}
    else:
        metrics = {'precision': 0, "recall": 0, "f1-score": 0}

    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1-score']

    return precision, recall, f1_score


def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None, device="hpu"):
    """ 加载检查点 """
    checkpoint = torch.load(filename, map_location=device)  # 使用模型所在的设备
    start_epoch = checkpoint['epoch']
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])  # 加载学习率调度器
    return start_epoch


def save_model(model, model_path):
    # 确保模型处于评估模式，以便正确保存
    model_cpu = model.cpu()
    model_cpu.eval()
    # 保存模型
    torch.save(model_cpu, os.path.join(model_path, "model_weights"))
    print(f"Model saved to {model_path}")
