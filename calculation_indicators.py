import json
import os


"""
用于计算预测结果的精确度、召回率和 f1分数的结果
"""

SUCCESS = 0
FILE_ERROR = 1
ENCODING_ERROR = 2
JSON_ERROR = 3
SCHEMA_ERROR = 4
ALIAS_FORMAT_ERROR = 5

CODE_INFO = {
    SUCCESS: 'success',
    FILE_ERROR: 'file is not exists',
    ENCODING_ERROR: 'file encoding error',
    JSON_ERROR: 'json parse is error',
    SCHEMA_ERROR: 'schema is error',
    ALIAS_FORMAT_ERROR: 'alias dict format is error'
}


def remove_book_titles(entity_name):
    """ 删除实体中的书名号 """
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name


def check_format(line):
    """检查line是否格式错误, 并且尝试解析line"""
    ret_code = SUCCESS
    json_info = {}
    line = line.strip()
    try:
        json_info = json.loads(line)
    except json.JSONDecodeError:
        ret_code = JSON_ERROR
        return ret_code, json_info

    # 检查必需的键: text 和 spo_list
    if 'text' not in json_info or 'spo_list' not in json_info:
        ret_code = SCHEMA_ERROR
        return ret_code, json_info

    required_key_list = ['subject', 'predicate', 'object']
    for spo_item in json_info['spo_list']:
        # 确保spo_item是一个字典
        if not isinstance(spo_item, dict):
            ret_code = SCHEMA_ERROR
            return ret_code, json_info

        # 确保所有必需的键都在spo_item中
        if not all(required_key in spo_item for required_key in required_key_list):
            ret_code = SCHEMA_ERROR
            return ret_code, json_info

        # 确保subject是字符串类型，object是字典类型
        if not isinstance(spo_item['subject'], str) or not isinstance(spo_item['object'], dict):
            ret_code = SCHEMA_ERROR
            return ret_code, json_info

    return ret_code, json_info


def parse_structured_value(json_info):
    """解析predict_result 和 test_dataset 的结构化数据, 保证两者spo格式一致"""
    spo_result = [{"predicate": item['predicate'],
                   "subject": remove_book_titles(item['subject']).lower(),
                   "object": {o_key: remove_book_titles(o_value).lower() for o_key, o_value in item['object'].items()}}
                  for item in json_info["spo_list"]]
    return spo_result


def load_dataset(filename):
    """ 加载一个JSON文件, 并返回一个包含句子和对应的结构化输出值的字典 """
    dataset_dict = {}  # 初始化一个空的字典来存储句子和结构化输出值
    ret_code = SUCCESS  # 初始化返回码为成功

    # 检查文件是否存在
    if not os.path.exists(filename):
        ret_code = FILE_ERROR  # 如果不存在，设置返回码为文件错误，并返回空的字典
        return ret_code, dataset_dict

    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            ret_code, json_info = check_format(line)  # 检查line格式，并解析

            if ret_code != SUCCESS:  # 如果返回码不是SUCCESS, 返回出错的返回码和空字典
                return ret_code, dataset_dict

            sent = json_info['text']
            spo_result = parse_structured_value(json_info)
            dataset_dict[sent] = spo_result
    # 返回成功码和填充好的字典
    return ret_code, dataset_dict


def load_alias_dict(alias_filename):
    """ 加载别名字典 """
    alias_dict = {}
    ret_code = SUCCESS  # 初始化返回码为成功

    if not alias_filename:  # 如果文件路径为空，直接返回成功码和空的字典
        return ret_code, alias_dict

    if not os.path.exists(alias_filename):  # 如果文件不存在，设置返回码为文件错误，并返回空的字典
        ret_code = FILE_ERROR
        return ret_code, alias_dict

    with open(alias_filename, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()  # 去除行首和行尾的空白字符
            try:
                words = line.split('\t')
                main_word = words[0].lower()  # 获取主词，并将其转换为小写
                aliases = set(word.lower() for word in words[1:])  # 获取别名列表，并将其中的每个单词转换为小写
                alias_dict.setdefault(main_word, set()).update(aliases)  # 如果主词已经在字典中，将其对应的别名集合合并
            except ALIAS_FORMAT_ERROR:
                ret_code = ALIAS_FORMAT_ERROR  # 设置返回码为文件错误
                return ret_code, alias_dict
    return ret_code, alias_dict


def remove_duplicate(spo_list, alias_dict):
    """ 删除预测结果中的同义词三元组 """
    new_spo_list = []  # 存储去重后的spo
    for spo in spo_list:
        if not is_spo_in_list(spo, new_spo_list, alias_dict):
            new_spo_list.append(spo)
    return new_spo_list


def is_spo_in_list(target_spo, spo_list, alias_dict):
    """ target_spo是否在spo_list中 """
    if target_spo in spo_list:
        return True
    t_subject = target_spo["subject"]
    t_predicate = target_spo["predicate"]
    t_object = target_spo["object"]
    t_sub_alias_set = alias_dict.get(t_subject, set())  # 获取主词t_subject的别名集合
    t_sub_alias_set.add(t_subject)  # 添加主词到别名集合中
    for spo in spo_list:
        sub = spo["subject"]
        predicate = spo["predicate"]
        obj = spo["object"]
        if predicate != t_predicate:
            continue
        if sub in t_sub_alias_set and is_equal_obj(obj, t_object, alias_dict):  # 如果subject和object都相同，就返回True
            return True
    return False


def is_equal_obj(obj_a, obj_b, alias_dict):
    """ 比较两个object是否相同 """
    for key_a, value_a in obj_a.items():  # 如果obj_a的某个键不在obj_b中，则obj_a, obj_b不相同
        if key_a not in obj_b:
            return False

        value_a_alias_set = alias_dict.get(value_a, set())  # 获取obj_a的值的别名集合
        value_a_alias_set.add(value_a)
        if obj_b[key_a] not in value_a_alias_set:  # 如果obj_b中的值不在obj_a值的别名集合中，则obj_a, obj_b不相同
            return False

    for key_b, value_b in obj_b.items():
        if key_b not in obj_a:
            return False
        value_b_alias_set = alias_dict.get(value_b, set())
        value_b_alias_set.add(value_b)
        if obj_a[key_b] not in value_b_alias_set:
            return False
    return True


def calculate_pr(predict_filename, dev_filename, alias_filename):
    """ 计算预测结果的准确率、召回率和F1分数 """
    ret_info = {}

    # 加载别名字典
    ret_code, alias_dict = load_alias_dict(alias_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info

    # 加载验证数据集
    ret_code, dev_dict = load_dataset(dev_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info

    # 加载预测结果
    ret_code, predict_result = load_dataset(predict_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info

    correct_sum = 0.0  # 包含了模型正确预测的所有三元组，无论这些三元组是否在真实数据中
    predict_sum = 0.0  # 预测结果中的全部三元组数量
    recall_sum = 0.0  # 验证集数据中全部的三元组数量
    recall_correct_sum = 0.0  # 只包含模型正确预测的三元组中那些确实存在于真实数据中的部分

    for sent in dev_dict:
        dev_spo_list = remove_duplicate(dev_dict[sent], alias_dict)
        predict_spo_list = predict_result.get(sent, list())
        new_predict_spo = remove_duplicate(predict_spo_list, alias_dict)  # 去除预测结果中的同义词三元组

        predict_sum += len(new_predict_spo)
        recall_sum += len(dev_spo_list)
        for spo in new_predict_spo:
            if is_spo_in_list(spo, dev_spo_list, alias_dict):
                correct_sum += 1
        for dev_spo in dev_spo_list:
            if is_spo_in_list(dev_spo, predict_spo_list, alias_dict):
                recall_correct_sum += 1

    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = recall_correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['errorCode'] = SUCCESS
    ret_info['errorMsg'] = CODE_INFO[SUCCESS]
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info
