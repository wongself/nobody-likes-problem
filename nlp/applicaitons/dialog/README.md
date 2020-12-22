# 对话模块
## 安装部署

### 模块测试
- [对话模块测试: 2335](http://101.124.42.4:2335)
### 环境依赖
- torch(tested with version 1.3.1)
- tensorboardX(tested with version 1.9)
- pytorch-pretrained-bert(tested with version 0.6.1)
- tqdm (tested with version 4.36.1)

### 环境安装
1. 位于CMAML目录中，输入命令`cp -r /data/sfs/nobody-likes-problem-main/nlp/applicaitons/dialog/CMAML/save/cmaml` 来导入对话模块运行所需的模型，`cp -r /data/sfs/nobody-likes-problem-main/nlp/applicaitons/dialog/CMAML/data`来导入对话模块运行所需的数据集，`cp -r /data/sfs/nobody-likes-problem-main/nlp/applicaitons/dialog/CMAML/vectors`来导入对话模块运行所需的词向量，`cp -r /data/sfs/nobody-likes-problem-main/nlp/applicaitons/dialog/CMAML/tmp`来导入对话模块运行所需的预训练模型。

2. 上一步仅限 V100 服务器用户操作。`./nlp/applicaitons/dialog/CMAML/data`中ConvAI2的数据请在[此处](https://pan.baidu.com/s/1AapbsWLtzv3adRatPmINSw)下载，提取码mb62,nli_model中的模型请在[此处](https://pan.baidu.com/s/1-9VLRDfy-Bf-pbGFRER0eA)下载，取码`8dw4`

   `./nlp/applicaitons/dialog/CMAML/save/cmaml`中的模型请在[此处](https://pan.baidu.com/s/1bTeaItGW0ScD5T4VgcSR_g)下载，提取码：`tgzk`，

   `./nlp/applicaitons/dialog/CMAML/save/vectors`中的词向量请在[此处](http://nlp.stanford.edu/data/glove.6B.zip),

   `./nlp/applicaitons/dialog/CMAML/tmp`中的预训练模型请在[此处](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)和[此处](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.vocab.txt)下载。

3. `./nlp/applicaitons/dialog/CMAML`文件夹中需要导入的文件结构如下。

```
./nlp/applicaitons/dialog/CMAML
├── save
│   └── cmaml
│       ├── model_55_45.7215_0.0000_0.0000_0.0000_1.1000
│       ├── model_55_45.7215_0.0000_0.0000_0.0000_1.1000dataset.p
│       ├── dataset.p
├── tmp
│   └── bert-base-uncased-vocab.txt
│   └── bert-base-uncased-tar.gz
├── vectors
│   └── glove.6B.300d.txt
└── data
    └── ConvAI2
    └── nli_model
```


## 项目开发

### 项目调试
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/dialog/server.py`来启动对话模块，随后在浏览器中输入本机网址及端口`2335`，来测试模块是否启动成功。若页面出现出现`Nobody Likes Problem`，则表明模块启动成功。
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/dialog/server.py`来启动对话系统模块，随后在浏览器中输入本机网址及端口`2335`，来测试模块是否启动成功。若页面出现出现`Nobody Likes Problem`，则表明模块启动成功。
### 项目维护
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/dialog/server.py`来启动对话系统模块。

