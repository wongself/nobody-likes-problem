from flask import Flask, jsonify, request
import nltk
from waitress import serve

your_global_variable = None  # 不用的话记得删掉

app = Flask(__name__)
app.config.from_object('configure')


def init_server():
    global your_global_variable  # 要修改全局变量的话，需要保留这句
    your_global_variable = '不用的话记得删掉'


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_server', methods=['POST'])
def query_server():
    if request.method == 'POST':
        source = request.form['source']
        # 你可以不需要分词
        jsentences = nltk.sent_tokenize(source)
        jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]

        # 分词完之后，你要做的在这里，若最终前端无任何结果，可能是因为JSON格式的问题
        jserver = 'Nobody Likes Problem'

        return jsonify({'jserver': jserver})
    return jsonify({'jserver': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2345, debug=False)
    serve(app, host="0.0.0.0", port=2345)  # 请在2335~2400之间选择一个端口
