# 情感分析模块

## 安装部署

### 模块测试
- [情感分析模块测试: 2339](http://101.124.42.4:2339)

### 环境依赖
- paddlepaddle (tested with version 1.8.5)
- Senta (tested with version 2.0.0)

### 环境安装
1. 位于项目根目录，首次运行`python ./nlp/applicaitons/sana/server.py`即可安装情感分析模块运行所需的预训练模型、外部数据等必要资料，但需要等待较长一段时间直到senta文件夹下生成model_files目录。

## 项目开发

### 项目调试
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/sana/server.py`来启动情感分析模块，随后在浏览器中输入本机网址及端口`2339`，来测试模块是否启动成功。若页面出现出现`Nobody Likes Problem`，则表明模块启动成功。

### 项目维护
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/sana/server.py`来启动情感分析模块。
