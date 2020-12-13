# nobody-likes-problem

## 安装部署

### 展示网址
- [展示网址: 2444](http://101.124.42.4:2444)

### 模块测试
- [信息抽取模块测试: 2334](http://101.124.42.4:2334)
- [中文新闻分类模块测试: 2335](http://101.124.42.4:2335)

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

#### 中文新闻分类模块
- numpy (tested with version 1.19.2)
- torch (tested with version 1.6.0)
- tqdm (tested with version 4.50.2)
- pytorch_pretrained_bert (tested with version 0.6.2)

### 环境安装

1. 位于用户根目录，输入命令`git clone git@github.com:wongself/nobody-likes-problem.git`来下载该仓库。
2. 进入项目`nobody-likes-problem`的目录，输入命令`pip install -r requirements.txt`来下载环境依赖，推荐在 Anaconda 创建的虚拟环境中安装所需依赖。
3. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`来测试 Django 架构是否安装成功。

> 👇信息抽取模块
4. 位于项目根目录，输入命令`cp -r /data/wsf/nobody-likes-problem/data ./data`来导入信息抽取模块运行所需的预训练模型、外部数据等必要资料。若需手动构建，请参考 [README](https://github.com/wongself/nobody-likes-problem/blob/main/nlp/applicaitons/spert/README.md) 文件。

> 👇中文新闻分类模块
5. 位于服务根目录，输入命令`cp -r /data/zj/LanguageInformationProcessing/git-try/nobody-likes-problem/nlp/applicaitons/text_classification_ch/THUCNews ./`来导入中文新闻分类模块运行所需的预训练模型、外部数据等必要资料。若需手动构建，请参考 [README](https://github.com/wongself/nobody-likes-problem/tree/ZJ/nlp/applicaitons/text_classification_ch/README.md) 文件。

## 项目开发

### 项目调试

1. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`来生成 Django 架构运行所需的必要文件。
2. 位于项目根目录，输入命令`python manage.py collectstatic --no-input`来更新网站所需的静态文件（CSS、JS、HTML、IMG...）。仅当第一次启动项目或者位于`nobody-likes-problem/nlp/static`的静态文件被修改时才需要执行这项操作。
3. 位于项目根目录，输入命令`python manage.py runserver 0:2444`来启动网站，随后在浏览器中输入本机网址及端口`2444`。其中，`0`代表`0.0.0.0`，而`2444`代表网站的默认端口，你可以将端口改为`1024~65535`中的任意一个数值。但需要注意的是，不要设置重复的端口号。

> 👇信息抽取模块
4. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动信息抽取模块，随后在浏览器中输入本机网址及端口`2334`，来测试模块是否启动成功。若页面出现出现`NLP in Your Area`，则表明模块启动成功。~~ 信息抽取模块已经在端口`2334`启动成功。

> 👇中文新闻分类模块
5. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/text_classification_ch/server.py`来启动中文新闻分类模块，随后在浏览器中输入本机网址及端口`2335`，来测试模块是否启动成功。若页面出现出现`NLP in Your Area`，则表明模块启动成功。~~ 中文新闻分类模块已经在端口`2335`启动成功。

### 项目维护

1. 位于项目根目录，先后输入命令`python manage.py makemigrations`、`python manage.py migrate`和`python manage.py collectstatic --no-input`来生成网站运行所需的必要文件。
2. 位于项目根目录，~~输入命令`gunicorn nlp_in_your_area.wsgi -w 4 -k gthread -b 0.0.0.0:2444`来启动网站。~~ 输入命令`python manage.py runserver 0:2444`来启动项目。
3. 若有附加功能需要添加，可以将 Python 代码放置于`nobody-likes-problem/nlp/applicaitons`中，在`nobody-likes-problem/nlp/urls.py`和`nobody-likes-problem/nlp/views.py`设置相应的链接跳转和消息处理，并在`nobody-likes-problem/nlp/templates`和`nobody-likes-problem/nlp/static/nlp`中修改相应的前端代码和 AJAX 相应代码。

> 👇信息抽取模块
4. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动信息抽取模块。~~ 信息抽取模块已经在端口`2334`启动成功。

> 👇中文新闻分类模块
5. ~~位于项目根目录，输入命令`python ./nlp/applicaitons/text_classification_ch/server.py`来启动中文新闻分类模块。~~ 中文新闻分类模块已经在端口`2335`启动成功。

### 如何构建你的服务
1. 为了创建类似于信息抽取模块（已占用端口号`2334`）的服务，以便于 Django 框架中的视图层向创建的服务发送请求并接收响应，你可以参考`nobody-likes-problem/nlp/applicaitons/server_template`中的文件格式，自行在`nobody-likes-problem/nlp/applicaitons`中创建一个新的文件夹（例如`/dialog`），并参考`/server_template`中的`server.py`文件，将其改写为你需要的形式。
2. 在服务根目录，输入命令`python ./server.py`来启动你刚刚创建的模块。随后，在浏览器中输入本机网址及对应端口，来测试模块是否启动成功。若页面出现出现`NLP in Your Area`，则表明模块启动成功。

### 如何展示你的服务
1. 为了在页面成功展示模块的输出结果，需要自行创建对应页面并设置路径，详见下方说明。
2. 首先，在`nobody-likes-problem/nlp/templates/template.html`中展示了一个页面模板，你也可以在`2444`端口直接点击导航栏中的`模板`一项查看模板的[展示效果](http://101.124.42.4:2444/template/)。现在，为了创建模块对应的页面，需要在对应文件夹内拷贝一份`template.html`并重命名为模块对应的名称（推荐使用模块功能对应的核心单词，如`dialog.html`，后续均用`dialog`代表模块名称）。随后，在`dialog.html`中将所有跟`template`有关的名称重命名为`dialog`，如`trigger_template->trigger_dialog`、`template.js->dialog.js`等。
3. 接着，参考`nobody-likes-problem/nlp/templates/base.html`中`nav_template`字段所在行的注释，创建一个导航`<li>`块。需要注意的是，上述两个文件中的`template`字段均需要重命名为`dialog`。
4. 然后，复制模块页面脚本，即拷贝一份`nobody-likes-problem/nlp/static/nlp/js/custom/template.js`到对应文件夹并重命名为`dialog.js`，接着在`dialog.html`中重命名倒数第5行`{% static 'nlp/js/custom/template.js' %}`中的`template.js`引用。
5. 接着，参考`nobody-likes-problem/nlp/static/nlp/js/custom/dialog.js`中`复制该段`字段所在行的注释，创建一个`case`块。需要注意的是，文件中的`template`字段均需要重命名为`dialog`。
6. 然后，复制模块页面样式，即拷贝一份`nobody-likes-problem/nlp/static/nlp/css/custom/template.css`到对应文件夹并重命名为`dialog.css`，接着在`dialog.html`中重命名第14行`{% static 'nlp/css/custom/template.css' %}`中的`template.css`引用。
7. 页面相关文件创建成功后，参考`nobody-likes-problem/nlp/urls.py`中的注释，创建一行页面重定向语句和一行查询重定向语句。然后，参考`nobody-likes-problem/nlp/views.py`中的注释，创建一段页面调用函数和一段查询调用语句。需要注意的是，上述两个文件中的`template`字段均需要重命名为`dialog`。

> 若脚本、样式修改没有效果，请确保文件引用正确，且网站运行在调试模式下。
