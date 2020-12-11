from flask import Flask, jsonify, request
import nltk
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
        question,article = source.split('\n',1)
        # article = "The MuRel network is a Machine Learning model learned end-to-end to answer questions about images. It relies on the object bounding boxes extracted from the image to build a complitely connected graph where each node corresponds to an object or region. The MuRel network contains a MuRel cell over which it iterates to fuse the question representation with local region features, progressively refining visual and question interactions. Finally, after a global aggregation of local representations, it answers the question using a bilinear model. Interestingly, the MuRel network doesn't include an explicit attention mechanism, usually at the core of state-of-the-art models. Its rich vectorial representation of the scene can even be leveraged to visualize the reasoning process at each step." 
        # question = 'How The MuRel network answer the question'
        ans = mrc_model.predict(article,question)
        return jsonify({'jmrc': ans})
    return jsonify({'jmrc': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2345, debug=False)
    serve(app, host="0.0.0.0", port=2345)  # 请在2335~2400之间选择一个端口
