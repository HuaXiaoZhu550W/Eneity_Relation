# 实体关系抽取任务

本项目是一个基于DUIE数据集的实体关系抽取任务。我们使用了一种创新的模型结构，在分类器前额外添加一个前馈神经网络和残差链接结构来提高模型性能。

## 项目结构

- `main.py`: 模型的训练和评估主脚本。
- `create_encoding_and_decoding_file.py`: 生成 `predicate2id.json` 和 `id2spo.json` 文件的脚本。
- `calculation_indicators.py`: 用于计算评估指标。
- `Model/`: 包含模型定义的模块。
- `Dataset/`: 包含数据加载和预处理的模块。
- `Loss.py`: 包含损失函数定义。
- `eval.py`: 包含模型评估函数。
- `train.py`: 包含模型训练函数。
- `utils.py`: 包含通用工具函数。
- `config.py`: 包含项目配置参数。
- `logs/`: 存储训练日志。
- `checkpoint/`: 存储训练过程中的检查点。

## 模型结构

本项目的模型基于BERT架构，并针对实体关系抽取任务进行了定制。模型结构定义在 `Model/BertForDuie.py` 中。特别地，采用了在分类器前面添加一个前馈神经网络和残差链接结构，从而提高了模型对实体关系理解的准确性。

## 安装依赖

在运行代码之前，请确保您的Python版本并且已安装以下Python库：

- torch 
- transformers
- python 3.8 +

## 如何运行

在运行 `main.py` 之前，您需要首先运行 `create_encoding_and_decoding_file.py` 脚本来生成 `predicate2id.json` 和 `id2spo.json` 文件。这些文件包含了关系类别到ID的映射以及用于模型预测的额外信息。

1. 克隆或下载本项目到本地。
2. 在项目根目录下，运行 `python create_encoding_and_decoding_file.py`。这将生成所需的 `predicate2id.json` 和 `id2spo.json` 文件。
3. 使用 `python main.py` 启动模型的训练和评估过程。

## 额外说明

- 需要注意在运行 `python create_encoding_and_decoding_file.py`时要修改脚本中'duie_schema.json'文件的地址。
- 您可以通过修改 `config.py` 中的参数来调整模型配置。
- 实验证明，添加了前馈神经网络和残差链接结构的模型比预训练模型直接连接分类器效果要好，F1分数提升2%~3%。
