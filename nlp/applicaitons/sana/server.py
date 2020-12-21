from flask import Flask, jsonify, request
import nltk
from waitress import serve

import datetime
#import paddlehub as hub
from senta import Senta

#senta = hub.Module(name="senta_bilstm")
senta = Senta()
use_cuda = False

app = Flask(__name__)
app.config.from_object('configure')


def init_server():
    global senta, use_cuda  # 要修改全局变量的话，需要保留这句
    senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_server', methods=['POST'])
def query_server():
    if request.method == 'POST':
        source = request.form['source']

        jsentences = nltk.sent_tokenize(source)
        #jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]

        #Data preprocess
        #input_dict = {}
        #for jsentence in jsentences:
            #input_dict[jsentence] = jsentence
        #input_dict = {"text": jsentences}


        #Predict
        #jextract = senta.sentiment_classify(data=input_dict)
        result = senta.predict(jsentences)
        jextract = {}
        for r in result:
            jextract[r[0]]=r[1]
        #jextract = senta.predict(data=input_dict)



        return jsonify({'jserver': jextract})
    return jsonify({'jserver': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2345, debug=False)
    serve(app, host="0.0.0.0", port=2347)  # 请在2335~2400之间选择一个端口
