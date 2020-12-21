import json
import math
import os
import pickle
import random
import re
import string
from collections import Counter

import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from absl import app
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from tqdm import tqdm
from .models import QANet

from .config import config, device
from .preproc import (build_features_for_Demo, gen_word2id, get_char2id,
                     get_word2id, preproc, preproc_predict,
                     process_file_for_Demo)

# writer = SummaryWriter(log_dir='./log1')
'''
Some functions are from the official evaluation script.
'''
os.environ["PATH"] = os.getcwd() + '/nlp/applicaitons/mrc/model' 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
class SQuADDataset(Dataset):
    def __init__(self, npz_file, batch_size):
        data = np.load(npz_file)
        self.context_idxs = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs = data["ques_idxs"]
        self.ques_char_idxs = data["ques_char_idxs"]
        self.y1s = data["y1s"]
        self.y2s = data["y2s"]
        self.ids = data["ids"]  
        self.num = len(self.ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.context_idxs[idx],self.context_char_idxs[idx], self.ques_idxs[idx],self.ques_char_idxs[idx],self.y1s[idx],self.y2s[idx],self.ids[idx]




class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1+num_updates)/(10+num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, ids = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    ids = torch.from_numpy(np.array(ids)).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, ids

def get_loader(npz_file, batch_size):
    dataset = SQuADDataset(npz_file, batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=5,
                                              collate_fn=collate)
    return data_loader

def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        #exclude.update('，', '。', '、', '；', '「', '」')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def train(model, optimizer, scheduler, dataset, dev_dataset, dev_eval_file, start, ema):
    model.train()
    losses = []
    print(f'Training epoch {start}')
    for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
        optimizer.zero_grad()
        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        y1, y2 = y1.to(device), y2.to(device)
        p1 = F.log_softmax(p1, dim=1)
        p2 = F.log_softmax(p2, dim=1)
        loss1 = F.nll_loss(p1, y1)
        loss2 = F.nll_loss(p2, y2)
        loss = (loss1 + loss2)
        writer.add_scalar('data/loss', loss.item(), i+start*len(dataset))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        ema(model, i+start*len(dataset))

        scheduler.step()
        if (i+1) % config.checkpoint == 0 and (i+1) < config.checkpoint*(len(dataset)//config.checkpoint):
            ema.assign(model)
            metrics = test(model, dev_dataset, dev_eval_file, i+start*len(dataset))
            ema.resume(model)
            model.train()
        for param_group in optimizer.param_groups:
            #print("Learning:", param_group['lr'])
            writer.add_scalar('data/lr', param_group['lr'], i+start*len(dataset))
        print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
    loss_avg = np.mean(losses)
    print("STEP {:8d} Avg_loss {:8f}\n".format(start, loss_avg))

def my_test(model,features,convert_file):
    print("\nPredict")
    model.eval()
    with torch.no_grad():
        for each_key in features.keys():
            features[each_key] = features[each_key].to(device)
        P1,P2 = model(features['Cwid'],features['Ccid'],features['Qwid'],features['Qcid'])
        p1 = F.softmax(P1,dim=1)
        p2 = F.softmax(P2,dim=1)
        outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
        for j in range(outer.size()[0]):
            outer[j] = torch.triu(outer[j])
        a1, _ = torch.max(outer, dim=2)
        a2, _ = torch.max(outer, dim=1)
        ymin = torch.argmax(a1, dim=1)
        ymax = torch.argmax(a2, dim=1)
        ans = []
        for each_start,each_end,context_id in zip(ymin,ymax,convert_file):
            ans.append(convert_file[context_id]['context'][convert_file[context_id]['spans'][each_start.item()][0]:convert_file[context_id]['spans'][each_end.item()][1]])
        return ans
        # answer_dict_, ans = convert_tokens(convert_file, ids.tolist(), ymin.tolist(), ymax.tolist())
        

def test(model, dataset, eval_file, test_i):
    print("\nTest")
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = config.val_num_batches
    with torch.no_grad():
        for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)


            P1, P2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            p1 = F.log_softmax(P1, dim=1)
            p2 = F.log_softmax(P2, dim=1)
            loss1 = F.nll_loss(p1, y1)
            loss2 = F.nll_loss(p2, y2)
            loss = torch.mean(loss1 + loss2)
            losses.append(loss.item())

            p1 = F.softmax(P1, dim=1)
            p2 = F.softmax(P2, dim=1)

            #ymin = []
            #ymax = []
            outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            for j in range(outer.size()[0]):
                outer[j] = torch.triu(outer[j])
                #outer[j] = torch.tril(outer[j], config.ans_limit)
            a1, _ = torch.max(outer, dim=2)
            a2, _ = torch.max(outer, dim=1)
            ymin = torch.argmax(a1, dim=1)
            ymax = torch.argmax(a2, dim=1)

            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)
            print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
            if((i+1) == num_batches):
                break
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/answers.json", "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print("EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))
    if config.mode == "train":
        writer.add_scalar('data/test_loss', loss, test_i)
        writer.add_scalar('data/F1', metrics["f1"], test_i)
        writer.add_scalar('data/EM', metrics["exact_match"], test_i)
    return metrics

def train_entry(config):
    

    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = get_loader(config.train_record_file, config.batch_size)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)

    lr = config.learning_rate
    base_lr = 1
    lr_warm_up_num = config.lr_warm_up_num

    model = QANet(word_mat, char_mat).to(device)

    ema = EMA(config.decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8, params=parameters)
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr)
    best_f1 = 0
    best_em = 0
    patience = 0
    unused = False
    for iter in range(config.num_epoch):
        train(model, optimizer, scheduler, train_dataset, dev_dataset, dev_eval_file, iter, ema)
        ema.assign(model)
        metrics = test(model, dev_dataset, dev_eval_file, (iter+1)*len(train_dataset))
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(config.save_dir, "model.pt")
        torch.save(model, fn)
        ema.resume(model)

class Demo():
    def __init__(self,config):
        with open(config.word_emb_file, "rb") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(config.char_emb_file, "rb") as fh:
            char_mat = np.array(json.load(fh), dtype=np.float32)
        self.qa_model = QANet(word_mat,char_mat).to(device)
        self.qa_model.load_state_dict(torch.load(config.save_dir+'/model.pt', map_location='cpu'))
        self.config = config
    
    def predict(self,aritcle,question):
        word_counter, char_counter = Counter(), Counter()
        test_examples,test_eval_examples = process_file_for_Demo(question,aritcle,"test",word_counter,char_counter)
        word2idx_dict = get_word2id(word_counter,config.word_index)
        char2idx_dict = get_char2id(word_counter)
        data = build_features_for_Demo(self.config,test_examples,word2idx_dict,char2idx_dict)
        context_idxs = torch.from_numpy(data["context_idxs"]).long()
        context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        features = {'Cwid':context_idxs,'Ccid':context_char_idxs,'Qwid':ques_idxs,'Qcid':ques_char_idxs}
        ans = my_test(self.qa_model,features,test_eval_examples)
        return ans

def test_entry(config):
    fn = os.path.join(config.save_dir, "model.pt") 
    model = torch.load(fn, map_location='cpu')  # , map_location={'cuda:1':'cuda:0'}
    # ,map_location=device
    with open(config.test_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

        # for each_id in dev_eval_file:
        #     dev_eval_file[each_id]['span'] = []
        #     last_i = 0
        #     for i in range(len(dev_eval_file[each_id]['context'])):
        #         if(dev_eval_file[each_id]['context'][i] == ' '):
        #             dev_eval_file[each_id]['span'].append([last_i,i])
        #             last_i = i+1
    # dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    # test(model,dev_dataset,dev_eval_file,0)
    data = np.load(config.test_record_file)
    context_idxs = torch.from_numpy(data["context_idxs"]).long()
    context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
    ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
    ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
    features = {'Cwid':context_idxs,'Ccid':context_char_idxs,'Qwid':ques_idxs,'Qcid':ques_char_idxs}
    my_test(model, features, dev_eval_file)


def main(_):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "word2id":
        word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
        gen_word2id(word_emb_file,vec_size=config.glove_dim,save_file='./data/word2id.json')
    elif config.mode == "predict":
        preproc_predict(config)
    elif config.mode == "debug":
        config.batch_size = 2
        config.num_steps = 32
        config.val_num_batches = 2
        config.checkpoint = 2
        config.period = 1
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)

def demo(_):
    context = "The MuRel network is a Machine Learning model learned end-to-end to answer questions about images. It relies on the object bounding boxes extracted from the image to build a complitely connected graph where each node corresponds to an object or region. The MuRel network contains a MuRel cell over which it iterates to fuse the question representation with local region features, progressively refining visual and question interactions. Finally, after a global aggregation of local representations, it answers the question using a bilinear model. Interestingly, the MuRel network doesn't include an explicit attention mechanism, usually at the core of state-of-the-art models. Its rich vectorial representation of the scene can even be leveraged to visualize the reasoning process at each step." 
    question = 'How The MuRel network answer the question'
    demo = Demo(config)
    ans = demo.predict(context,question)

if __name__ == '__main__':
    app.run(demo)
    
