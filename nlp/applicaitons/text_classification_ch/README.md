# 中文新闻分类模块

## 安装部署

### 模块测试
- [信息抽取模块测试: 2335](http://101.124.42.4:2335)

### 环境依赖
- numpy (tested with version 1.19.2)
- torch (tested with version 1.6.0)
- tqdm (tested with version 4.50.2)
- pytorch_pretrained_bert (tested with version 0.6.2)

### 环境安装
1. 位于服务根目录，输入命令`cp -r /data/zj/LanguageInformationProcessing/git-try/nobody-likes-problem/nlp/applicaitons/text_classification_ch/THUCNews ./THUCNews`来导入中文新闻分类模块运行所需的预训练模型、外部数据等必要资料。
2. 上一步仅限 V100 服务器用户操作。不过，可以手动创建如下`THUCNews`文件夹，并将其放置在服务根目录。其中，`data`文件夹请在百度网盘[此处](https://pan.baidu.com/s/1wra1vlmnUQWYf4m245N2rg)下载, 提取码:xxph;`saved_dict`文件夹请在百度网盘[此处](https://pan.baidu.com/s/1mXay8JYSWrUhFQpfxi23DQ)下载,提取码：rm9j。 
```
./THUCNews
├── data
│   ├── class.txt
│   └── test.txt(不需要提前构建，运行服务时会自动生成)
└── saved_dict
    └── ERNIE.ckpt
```

## 项目开发
中文新闻分类模块
### 项目调试
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/text_classification_ch/server.py`来启动中文新闻分类模块，随后在浏览器中输入本机网址及端口`2335`，来测试模块是否启动成功。若页面出现出现`NLP in Your Area`，则表明模块启动成功。

### 项目维护
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/text_classification_ch/server.py`来启动中文新闻分类模块。

