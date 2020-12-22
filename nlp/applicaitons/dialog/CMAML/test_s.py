import matplotlib
matplotlib.use('Agg')
from CMAML.utils.data_reader import Personas
from CMAML.model.seq2spg import Seq2SPG
from CMAML.model.common_layer import NoamOpt
#from model.common_layer import evaluate
from CMAML.utils.data_reader import Dataset,collate_fn
from CMAML.utils.beam_omt import Translator
import pickle
from CMAML.utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import numpy as np 
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import math

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def evaluate(model, data, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", verbose=False, log=False, result_file="results/results_our.txt", ref_file="results/ref_our.txt", case_file="results/case_our.txt"):
    if log:
        f1 = open(result_file, "a")
        f2 = open(ref_file, "a")
    dial,ref, hyp_b, per= [],[],[], []
    t = Translator(model, model.vocab)

    l = []
    p = []
    ent_b = []
    
    pbar = tqdm(enumerate(data),total=len(data))
    for j, batch in pbar:
        #print(len(batch["input_batch"]))
        #print(len(batch["target_batch"]))
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
        if((j<3 and ty != "test") or ty =="test"): 

            sent_b, _ = t.translate_batch(batch)

            for i in range(len(batch["target_txt"])):
                new_words = []
                for w in sent_b[i][0]:
                    if w==config.EOS_idx:
                        break
                    new_words.append(w)
                    if len(new_words)>2 and (new_words[-2]==w):
                        new_words.pop()
                
                sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
                hyp_b.append(sent_beam_search)
    return sent_beam_search
                


'''

p = Personas()
# Build model, optimizer, and set states
print("Test model",config.model)
model = Seq2SPG(p.vocab,model_file_path=config.save_path,is_eval=False)
#fine_tune = []
#iter_per_task = []
#iterations = 26
#weights_original = deepcopy(model.state_dict())
#tasks = p.get_personas('test')
val = [[["hi , how are you doing today ?"],[" "],0,[" "]]]
dataset_valid = Dataset(val,p.vocab)
data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                batch_size=1,
                                                shuffle=False,
                                                collate_fn=collate_fn)

res = evaluate(model, data_loader_val, model_name=config.model,ty="test",verbose=False)
print(res)
'''
if __name__ == '__main__':
    p = Personas()
    # Build model, optimizer, and set states
    print("Test model",config.model)
    print(config.save_path)
    model = Seq2SPG(p.vocab,model_file_path=config.save_path,is_eval=False)
    que = "do you like to listen to music very much ?"
    val = [[[],[" "],0,[" "]]]
    val[0][0].append(que)
    print(val)
    dataset_valid = Dataset(val,p.vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=collate_fn)

    res = evaluate(model, data_loader_val, model_name=config.model,ty="test",verbose=False)
    print(res)
        


