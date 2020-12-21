from flask import Flask, jsonify, request
import nltk
from waitress import serve
from CMAML.utils.data_reader import Personas,Dataset,collate_fn
from CMAML.utils import config
from CMAML.model.seq2spg import Seq2SPG
from CMAML.test_s import evaluate
import torch
p = None  
model = None
val = None

app = Flask(__name__)
app.config.from_object('configure')


def init_server():
    global p
    global model
    p = Personas()
    print("Test model",config.model)
    print(config.save_path)
    model = Seq2SPG(p.vocab,model_file_path=config.save_path,is_eval=False)


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_dialog', methods=['POST'])
def query_dialog():
    if request.method == 'POST':
        source = request.form['source']
        # 你可以不需要分词
        #jsentences = nltk.sent_tokenize(source)
        #jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]
        print(source)
        val = [[[],[" "],0,[" "]]]
        val[0][0].append(source)
        print(val)
        dataset_valid = Dataset(val,p.vocab)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        collate_fn=collate_fn)

        res = evaluate(model, data_loader_val, model_name=config.model,ty="test",verbose=False)

        # 分词完之后，你要做的在这里，若最终前端无任何结果，可能是因为JSON格式的问题
        jdialog = 'Nobody Likes Problem'

        return jsonify({'jdialog': res})
    return jsonify({'jdialog': ''})


if __name__ == "__main__":
    init_server()
    # app.run(host='0.0.0.0', port=2345, debug=False)
    serve(app, host="0.0.0.0", port=2335)  # 请在2335~2400之间选择一个端口
