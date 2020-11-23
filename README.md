# nobody-likes-problem

## 安装部署

### 展示网址
- [展示网址: 2444](http://101.124.42.4:2444)

### 模块测试
- [信息抽取模块测试: 2334](http://101.124.42.4:2334)

### 环境依赖

#### 网站架构
- python 3.7+ (test with version 3.7.8)
- django 3.1+ (test with version 3.1.3)
- flask (tested with version 1.1.2)
- gunicorn (test with version 20.0.4)
- waitress (test with version 1.4.4)
- whitenoise (test with version 5.2.0)

#### 信息抽取模块
- nltk (tested with version 3.5)
- numpy (tested with version 1.19.2)
- scikit-learn (tested with version 0.23.2)
- torch 1.1.0+ (tested with version 1.6.0)
- tqdm (tested with version 4.50.2)
- transformers 2.2.0+ (tested with version 3.4.0)

### 环境安装

1. 位于用户根目录，输入命令`git clone git@github.com:wongself/nobody-likes-problem.git`来下载该仓库。
2. 进入项目`nobody-likes-problem`的目录，输入命令`pip install -r requirements.txt`来下载环境依赖，推荐在 Anaconda 创建的虚拟环境中安装所需依赖。
3. 位于项目根目录，输入命令`cp -r /data/wsf/nobody-likes-problem/data ./data`来导入信息抽取模块运行所需的预训练模型、外部数据等必要资料。
4. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`来测试 Django 架构是否安装成功。

> `data`文件结构如下所示。

```
./data
├── datasets
│   └── scierc
│       ├── scierc_dev.json
│       ├── scierc_test.json
│       ├── scierc_train_dev.json
│       ├── scierc_train.json
│       └── scierc_types.json
├── log
└── models
    └── scierc
        ├── config.json
        ├── pytorch_model.bin
        └── vocab.txt
```

## 项目开发

### 项目调试

1. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`来生成 Django 架构运行所需的必要文件。
2. 位于项目根目录，输入命令`python manage.py collectstatic --no-input`来更新网站所需的静态文件（CSS、JS、HTML、IMG...）。仅当第一次启动项目或者位于`nobody-likes-problem/nlp/static`的静态文件被修改时才需要执行这项操作。
3. 位于项目根目录，输入命令`python manage.py runserver 0:2444`来启动网站，随后在浏览器中输入本机网址及端口`2444`。其中，`0`代表`0.0.0.0`，而`2444`代表网站的默认端口，你可以将端口改为`1024~65535`中的任意一个数值。但需要注意的是，不要设置重复的端口号。
4. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/spert/spert.py`来启动信息抽取模块，随后在浏览器中输入本机网址及端口`2334`，来测试模块是否启动成功。若页面出现出现`NLP in Your Area`，则表明模块启动成功。~~ 信息抽取模块已经在端口`2334`启动成功。

### 项目维护

1. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`和`python manage.py collectstatic --no-input`来生成网站运行所需的必要文件。
2. 位于项目根目录，~~输入命令`gunicorn nlp_in_your_area.wsgi -w 4 -k gthread -b 0.0.0.0:2444`来启动网站。~~ 输入命令`python manage.py runserver 0:2444`来启动项目。
3. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/spert/spert.py`来启动信息抽取模块。~~ 信息抽取模块已经在端口`2334`启动成功。
4. 若有附加功能需要添加，可以将 Python 代码放置于`nobody-likes-problem/nlp/applicaitons`中，在`nobody-likes-problem/nlp/urls.py`和`nobody-likes-problem/nlp/views.py`设置相应的链接跳转和消息处理，并在`nobody-likes-problem/nlp/templates`和`nobody-likes-problem/nlp/static/nlp`中修改相应的前端代码和 AJAX 相应代码。
