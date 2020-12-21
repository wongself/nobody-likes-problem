from flask import Flask, jsonify, request
from waitress import serve
from model.main import Demo as mrc
from model.config import config as mrc_config

app = Flask(__name__)
app.config.from_object('configure')
mrc_model = None


def init_server():
    global mrc_model
    mrc_model = mrc(mrc_config)
    # global your_global_variable  # 要修改全局变量的话，需要保留这句
    # your_global_variable = '不用的话记得删掉'


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/mrc', methods=['POST'])
def query_server():
    if request.method == 'POST':
        source = request.form['source']
        question, article = source.split('\n', 1)
        ans = mrc_model.predict(article, question)
        return jsonify({'jmrc': ans})
    return jsonify({'jmrc': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2338, debug=False)
    serve(app, host="0.0.0.0", port=2338)  # 请在2335~2400之间选择一个端口
