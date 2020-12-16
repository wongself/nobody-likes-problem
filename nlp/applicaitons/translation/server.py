from flask import Flask, jsonify, request
import nltk
import torch
import datetime
import dill
from waitress import serve
import translation 
your_global_variable = None  # 不用的话记得删掉

# 初始化参数设置
UNK = 0  # 未登录词的标识符对应的词典id
PAD = 1  # padding占位符对应的词典id
BATCH_SIZE = 64  # 每批次训练数据数量
# EPOCHS = 20  # 训练轮数
EPOCHS = 1  # 训练轮数
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
H_NUM = 8  # multihead attention hidden个数
D_MODEL = 256  # embedding维数
D_FF = 1024  # feed forward第一个全连接层维数
DROPOUT = 0.1  # dropout比例
MAX_LENGTH = 60  # 最大句子长度

SAVE_FILE = 'save/model(1).pt'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_file = 'nmt/en-cn/test.txt'
# with open('data.pkl', 'rb') as f:
# 	data = dill.load(f)

with open('cndict', 'rb') as f:
	cndict = dill.load(f)
with open('endict', 'rb') as f:
	endict = dill.load(f)

src_vocab = len(endict[0])
tgt_vocab = len(cndict[0])

app = Flask(__name__)
app.config.from_object('configure')


def init_server():
    global your_global_variable  # 要修改全局变量的话，需要保留这句
    your_global_variable = '不用的话记得删掉'


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_translation', methods=['POST'])
def query_translation():
    if request.method == 'POST':
        source = request.form['source']
        # 你可以不需要分词
        # jsentences = nltk.sent_tokenize(source)
        # jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]
        # source = source + ' .'
        jserver = translation.test(source)

        # 分词完之后，你要做的在这里，若最终前端无任何结果，可能是因为JSON格式的问题
        # jserver = 'Nobody Likes Problem'

        return jsonify({'jserver': jserver})
    return jsonify({'jserver': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2345, debug=False)
    serve(app, host="0.0.0.0", port=2377)  # 请在2335~2400之间选择一个端口
