# Fake-news-detection

虚假新闻检测-南开大学《python语言程序设计》大作业，数据集来源[yaqingwang/WeFEND-AAAI20: Dataset for paper "Weak Supervision for Fake News Detection via Reinforcement Learning" published in AAAI'2020. (github.com)](https://github.com/yaqingwang/WeFEND-AAAI20)

本项目遵守[AGPL-3.0开源协议](LICENSE)，你可以自由地使用、修改、传播源代码，但是你必须遵守[AGPL-3.0许可证](LICENSE)的规定，即如果使用或修改我的源代码，需要注明作者并给出项目链接，并且在公共平台开源（例如在GitHub或GitLab等），否则我有权力进行追责。

***如果您觉得本项目还不错，请给出您的 star，球球了。另外，我以后可能还会将其他课程或项目的相关代码开源在 Github，欢迎 follow me。***

如果对此项目或其他学习有疑问也可以根据我主页或博客上面的联系方式联系我，我会尽可能帮助你。

[![Star History Chart](https://api.star-history.com/svg?repos=lazy-forever/Fake-news-detection&type=Timeline)](https://star-history.com/#lazy-forever/Fake-news-detection&Timeline)

## 构建

确保你的电脑上已经装了pytorch

```shell
pip install -r requirements.txt
mkdir model
```

下载[ymcui/Chinese-BERT-wwm: Pre-Training with Whole Word Masking for Chinese BERT（中文BERT-wwm系列模型）](https://github.com/ymcui/Chinese-BERT-wwm)中的`RoBERTa-wwm-ext-large`PyTorch版本，[SeanLee97/xmnlp: xmnlp](https://github.com/SeanLee97/xmnlp)中的`xmnlp-onnx-models-v5.zip`存入model文件夹中。

下载[Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors 上百种预训练中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)中的`sgns.sogounews.bigram-char`存入data文件夹中。

## 运行

### 数据预处理

```shell
python web.py
python construct_test.py
python construct_train.py
```

### 训练模型+预测

**注**：请手动调整模型的各种数值以达到最优，本项目中默认数值结果不一定为最优。

#### Word2Vec+CNN

```shell
python cnn_model_sen.py
```

#### Bert+CNN

```shell
python bert_model_sen.py
```

#### Bert+LSTM

```shell
python lstm_model.py
```

### 机器学习

```shell
python ML_model.py
```

### 结果处理

**注**：请手动调节分隔数值以达到最优解，本项目中默认数值结果不一定为最优。

```shell
python construct.py
```

## 分数计算

```shell
python calc_AUC.py
```

