# 阅读理解模块

## 安装部署

### 模块测试
- [阅读理解模块测试: 2338](http://101.124.42.4:2338)

### 环境依赖
- torch (tested with version 1.6.0)
- absl (tested with version 0.11.0)
- spacy (tested with version 2.3.5)
- tqdm (tested with version 4.50.2)

### 环境安装
1. 位于服务根目录，输入命令`cp -r /data/zhq/nobody-likes-problem/nlp/applicaitons/mrc/data ./`来导入阅读理解模块运行所需的预训练模型、外部数据等必要资料。
2. 上一步仅限 V100 服务器用户操作。不过，可以手动在[此处](https://www.jianguoyun.com/p/DUHmaH0Qm6q_CBiJ6dMD)下载如下`data`文件夹，并将其放置在服务根目录。

```
./data
├── char_emb_dict.json
├── char_emb.json
├── char_emb.pkl
├── model.pt
├── word2id.json
├── word_emb_dict.json
├── word_emb.json
└── word_emb.pkl
```

## 项目开发

### 项目调试
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动阅读理解模块，随后在浏览器中输入本机网址及端口`2338`，来测试模块是否启动成功。若页面出现出现`Nobody Likes Problem`，则表明模块启动成功。

### 项目维护
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动阅读理解模块。

