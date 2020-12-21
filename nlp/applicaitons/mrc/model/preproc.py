from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np
from codecs import open

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
'''

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def process_file_for_Demo(question,article,data_type, word_counter, char_counter):
    eval_examples = {}
    article = article.replace("''", '" ').replace("``", '" ')
    context_tokens = word_tokenize(article)
    context_chars = [list(token) for token in context_tokens]
    spans = convert_idx(article, context_tokens)
    for token in context_tokens:
        word_counter[token] += 1 # 问题的个数
        for char in token:
            char_counter[char] += 1
    question = question.replace("''", '" ').replace("``", '" ')
    ques_tokens = word_tokenize(question)
    ques_chars = [list(token) for token in ques_tokens]
    for token in ques_tokens:
        word_counter[token] += 1
        for char in token:
            char_counter[char] += 1
    total = 1
    y1s, y2s = [], []
    answer_texts = []
    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
    eval_examples[str(total)] = {
                        "context": article, "spans": spans, "answers": answer_texts, "uuid": 1}
    return [example],eval_examples

def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    if(data_type != 'test'):
                        for answer in qa["answers"]:
                            answer_text = answer["text"]
                            answer_start = answer['answer_start']
                            answer_end = answer_start + len(answer_text)
                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                            y1, y2 = answer_span[0], answer_span[-1]
                            y1s.append(y1)
                            y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def build_features_for_Demo(config,examples,word2idx_dict,char2idx_dict):
    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit
    def filter_func(example, is_test=False):
        if is_test is True:
            return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit
        else:
            return len(example["context_tokens"]) > para_limit or \
                len(example["ques_tokens"]) > ques_limit or \
                (example["y2s"][0] - example["y1s"][0]) > ans_limit
    total = 0
    total_ = 0
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    ids = []
    for example in examples:
        total_ += 1

        if filter_func(example, True):
            continue
        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)
        
        ids.append(example["id"])
    result = {}
    result['context_idxs'] = np.array(context_idxs)
    result['context_char_idxs'] = np.array(context_char_idxs)
    result['ques_idxs'] = np.array(ques_idxs)
    result['ques_char_idxs'] = np.array(ques_char_idxs)
    result['ids'] = np.array(ids)
    return result

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        if is_test is True:
            return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit
        else:
            return len(example["context_tokens"]) > para_limit or \
                len(example["ques_tokens"]) > ques_limit or \
                (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    N = len(examples)
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    if is_test is False:
        y1s = []
        y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)
        
        if is_test is False:
            start, end = example["y1s"][-1], example["y2s"][-1]
            y1s.append(start)
            y2s.append(end)
        ids.append(example["id"])
    if is_test is False:
        np.savez(out_file, context_idxs=np.array(context_idxs), context_char_idxs=np.array(context_char_idxs),
                ques_idxs=np.array(ques_idxs), ques_char_idxs=np.array(ques_char_idxs), y1s=np.array(y1s),
                y2s=np.array(y2s), ids=np.array(ids))
    else:
        np.savez(out_file, context_idxs=np.array(context_idxs), context_char_idxs=np.array(context_char_idxs),
                ques_idxs=np.array(ques_idxs), ques_char_idxs=np.array(ques_char_idxs), ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def gen_word2id(emb_file=None, vec_size=None,save_file=None):
    NULL = "--NULL--"
    OOV = "--OOV--"
    r_file = {NULL:0,OOV:1}
    cnt = 2
    if(emb_file is not None and save_file is not None):
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                # if(word in counter and counter[word] > limit):
                r_file[word] = cnt
                cnt += 1
        save(save_file,r_file,"word2id")
    else:
        print("ERROR with emb_file!")

    # return r_file

def get_word2id(counter = None,index_file = None):
    NULL = "--NULL--"
    OOV = "--OOV--"
    word2idx_dict = [{'key':NULL,'value':0},{'key':OOV,'value':1}]
    result = {NULL:0,OOV:1}
    if(counter is not None and index_file is not None):
        filtered_elements = [k for k, v in counter.items() if v > -1]
        with open(index_file,'r',encoding='utf-8') as fh:
            w2id = json.load(fh)
            for each_token in filtered_elements:
                if(each_token not in word2idx_dict and each_token in w2id) :
                    word2idx_dict.append({'key':each_token,'value':w2id[each_token]})
            word2idx_dict.sort(key = lambda x:x['value'])   
            for i, each_token in enumerate(word2idx_dict):
                if(each_token['key'] not in result):
                    result[each_token['key']] = i
    else:
        print("counter or index ERROR")
    return result

def get_char2id(counter = None):
    NULL = "--NULL--"
    OOV = "--OOV--"
    char2id =  {NULL:0,OOV:1}
    cnt =2
    filtered_elements = [k for k, v in counter.items() if v > -1]
    for each_element in filtered_elements:
        for each_char in each_element:
            if(each_char not in char2id):
                char2id[each_char] = cnt 
                cnt +=1 
    return char2id

def preproc_predict(config):
    word_counter, char_counter = Counter(), Counter()
    test_examples,test_eval = process_file(config.test_file,"test",word_counter,char_counter)
    word2idx_dict = get_word2id(word_counter,config.word_index)
    char2idx_dict = get_char2id(word_counter)
    # word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    # char_emb_file = config.glove_char_file if config.pretrained_char else None
    # char_emb_size = config.glove_char_size if config.pretrained_char else None
    # char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim
    # word2idx_dict = record_file(word_counter,emb_file=word_emb_file,vec_size=config.glove_dim)
    # char2idx_dict = record_file(char_counter,emb_file=char_emb_file,vec_size=char_emb_dim)
    # word_emb_mat, word2idx_dict = get_embedding(
    #     word_counter, "word", emb_file=word_emb_file, vec_size=config.glove_dim)
    # char_emb_mat, char2idx_dict = get_embedding(
    #     char_counter, "char", emb_file=char_emb_file, vec_size=char_emb_dim)
    # save("./data/word_emb_dict.json",word2idx_dict,message="word_dict")
    # save("./data/char_emb_dict.json",char2idx_dict,message="char_dict")
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.test_meta, test_meta, message="test meta")

def preproc(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(config.dev_file, "dev", word_counter, char_counter)
    # test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, vec_size=char_emb_dim)

    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict)
    # test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    # save(config.test_eval_file, test_eval, message="test eval")
    save(config.word2idx_file, word2idx_dict, message="word dictionary")
    save(config.char2idx_file, char2idx_dict, message="char dictionary")
    save(config.dev_meta, dev_meta, message="dev meta")
    # save(config.test_meta, test_meta, message="test meta")
