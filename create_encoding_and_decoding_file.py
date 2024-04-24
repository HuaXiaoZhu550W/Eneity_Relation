import os
import json
from utils import read_data


def get_predicate2id(data_dir, filename):
    """
    1.生成关系类别文件, 假设有N中关系类别, 那么最终分类为2N+2
    (N乘2是因为对于所有实体都可以是头实体或者尾实体, 加2是要添加'I'标签和'O'标签)
    2.获取数据集中predicate, subject_type, object_type, 用于模型预测生成与数据标记相同格式的spo三元组
    """
    data = read_data(data_dir, filename)
    # 对于同一个predicate下对应多个object_type的, 使用predicate_object_type, 例: '配音_inWork'和'配音_@value'
    predicate = [d['predicate'] if len(d['object_type'].keys()) < 2 else d['predicate'] + '_' + key for d in data
                 for key in d['object_type'].keys()]
    predicate2id = {k: v for v, k in enumerate(predicate)}

    # 添加'I'标签和'O'标签
    predicate2id['I'] = len(predicate)
    predicate2id['O'] = len(predicate) + 1

    with open(os.path.join(data_dir, 'predicate2id.json'), 'w', encoding='utf8') as f:
        json.dump(predicate2id, f, ensure_ascii=False, indent=4)
    print('The file predicate2id.json has been generated.')

    id2spo = {"predicate": [d['predicate'] for d in data for _ in d['object_type'].keys()],
              "subject_type": [d['subject_type'] for d in data for _ in d['object_type'].keys()],
              "object_type": [value if len(d['object_type'].keys()) < 2 else value + '_' + key for d in data for key, value in d['object_type'].items()]}

    for value in id2spo.values():  # 添加标签“I”, “O” 对应的spo
        value.extend(["empty", "empty"])

    with open(os.path.join(data_dir, 'id2spo.json'), 'w', encoding='utf8') as f:
        json.dump(id2spo, f, ensure_ascii=False, indent=4)
    print('The file id2spo.json has been generated.')


if __name__ == "__main__":
    get_predicate2id("../DUIE", 'duie_schema.json')
