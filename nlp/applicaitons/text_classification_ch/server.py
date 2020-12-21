from flask import Flask, jsonify, request
from waitress import serve

import os
import time
import torch
import numpy as np
from importlib import import_module
from utils_for_test import build_dataset, build_iterator, get_time_dif

app = Flask(__name__)
app.config.from_object('configure')


@app.route('/')
def hello_world():
    return 'Nobody Likes Problem'


def test(config, model, test_iter):
    # test
    test_res = evaluate(config, model, test_iter, test=True)

    class_path = os.path.join(os.getcwd(), './THUCNews/data/class.txt')
    f = open(class_path, 'r', encoding='utf-8')
    lines = f.readlines()
    classes = []
    for line in lines:
        classes.append(line.strip())

    print(classes[test_res])
    return str(classes[test_res])


def evaluate(config, model, data_iter, test=False):
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    return predict_all[0]


@app.route('/query_server', methods=['POST'])
def query_server():
    if request.method == 'POST':
        source = request.form['source']
        # 将source写入test.txt
        test_path = os.path.join(os.getcwd(), 'THUCNews/data/test.txt')
        print(os.getcwd(), test_path)
        f = open(test_path, 'w', encoding='utf-8')
        f.write(str(source) + '\t1\n')
        f.close()

        # 开始处理
        dataset = 'THUCNews'  # 数据集+
        model_name = 'ERNIE'
        x = import_module('models.' + model_name)
        config = x.Config(dataset)

        start_time = time.time()
        print("Loading data...")
        test_data = build_dataset(config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        model = x.Model(config)
        model.load_state_dict(torch.load(
            config.save_path, map_location='cpu'))  # , map_location={'cuda:0'}
        model.eval()
        model = model.to(config.device)
        jserver = test(config, model, test_iter)
        ###

        return jsonify({'jserver': jserver})
    return jsonify({'jserver': ''})


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=2336, debug=False)
    serve(app, host="0.0.0.0", port=2336)  # 请在2335~2400之间选择一个端口
