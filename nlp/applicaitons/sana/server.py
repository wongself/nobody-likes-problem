from flask import Flask, jsonify, request
# import nltk
from waitress import serve

# import paddlehub as hub
from senta import Senta

# senta = hub.Module(name="senta_bilstm")
senta = Senta()
use_cuda = False

app = Flask(__name__)
app.config.from_object('configure')


def init_server():
    global senta, use_cuda  # 要修改全局变量的话，需要保留这句
    senta.init_model(model_class="ernie_1.0_skep_large_ch",
                     task="sentiment_classify",
                     use_cuda=use_cuda)


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_server', methods=['POST'])
def query_server():
    if request.method == 'POST':
        source = request.form['source']

        # jsentences = nltk.sent_tokenize(source)
        # jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]

        result = senta.predict(source)
        print('result', result)
        jsana = '未知'

        if result[0][1] == 'positive':
            jsana = '积极'
        elif result[0][1] == 'negative':
            jsana = '消极'

        return jsonify({'jserver': jsana})
    return jsonify({'jserver': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2339, debug=False)
    serve(app, host="0.0.0.0", port=2339)  # 请在2335~2400之间选择一个端口
