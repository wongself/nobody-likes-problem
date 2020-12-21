import os
import absl.flags as flags
import torch
import sys
import torch.backends.cudnn as cudnn

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/config.py
'''
home = os.path.expanduser(".")
real_path = os.getcwd()
sub_path = '/nlp/applicaitons/mrc/'
target_dir = real_path + sub_path + "data"
event_dir = real_path + sub_path + "log"
save_dir = real_path + sub_path + "data"
word_emb_file = os.path.join(target_dir, "word_emb.pkl")
char_emb_file = os.path.join(target_dir, "char_emb.pkl")
# word_index = os.path.join(target_dir,"word2id.json")
word2idx_file = os.path.join(target_dir, "word2id.json")
# answer_file = os.path.join(answer_dir, "answer.json")

flags.DEFINE_string("mode", "test", "train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("event_dir", event_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
# flags.DEFINE_string("load_dir", load, "")
flags.DEFINE_string("word_index", word2idx_file, "")

flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("char_emb_file", char_emb_file, "")

flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
# flags.DEFINE_integer("test_para_limit", 400, "Limit length for paragraph in test file")
# flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
#flags.DEFINE_integer("num_steps", 0, "Number of steps")
flags.DEFINE_integer("num_epoch", 40, "Number of epoch")
flags.DEFINE_integer("checkpoint", 900, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 500, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("dropout_char", 0.05, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 10.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_integer("lr_warm_up_num", 1000, "Number of warm-up steps of learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
#flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("early_stop", 50, "Checkpoints for early stop")
flags.DEFINE_integer("connector_dim", 96, "Dimension of connectors of each layer")
flags.DEFINE_integer("num_heads", 1, "Number of heads in multi-head attention")

flags.DEFINE_boolean("print_weight", False, "Print weights of some layers")

# Extensions (Uncomment corresponding line in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained char embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

config = flags.FLAGS
config(sys.argv)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

cudnn.enabled = False
