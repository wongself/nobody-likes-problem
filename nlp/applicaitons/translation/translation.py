import dill
import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable

# 初始化参数设置
UNK = 0  # 未登录词的标识符对应的词典id
PAD = 1  # padding占位符对应的词典id
BATCH_SIZE = 64  # 每批次训练数据数量
# EPOCHS = 20  # 训练轮数
EPOCHS = 1  # 训练轮数
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
H_NUM = 8  # multihead attention hidden个数
D_MODEL = 256  # embedding维数
D_FF = 1024  # feed forward第一个全连接层维数
DROPOUT = 0.1  # dropout比例
MAX_LENGTH = 60  # 最大句子长度

SAVE_FILE = 'save/model(1).pt'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

# with open('data.pkl', 'rb') as f:
# 	data = dill.load(f)
with open('cndict', 'rb') as f:
	cndict = dill.load(f)
with open('endict', 'rb') as f:
	endict = dill.load(f)

src_vocab = len(endict[0])
tgt_vocab = len(cndict[0])
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)




"""
获取每一个单词的词向量
"""
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		# Embedding层
		self.lut = nn.Embedding(vocab, d_model)
		# Embedding维数
		self.d_model = d_model

	def forward(self, x):
		# 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
		return self.lut(x) * math.sqrt(self.d_model)


# 导入依赖库
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
		# 来存放所有小于这个长度位置对应的porisional embedding
		pe = torch.zeros(max_len, d_model, device=DEVICE)
		# 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
		"""
		形式如：
		tensor([[0.],
				[1.],
				[2.],
				[3.],
				[4.],
				...])
				
		PE(pos,2i)=sin(pos/10000^(2i/dmodel)) PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
		pe[:, 0::2] = torch.sin(position * div_term) 偶数0,2,4,6,8,10
		pe[:, 1::2] = torch.cos(position * div_term) 奇数1,3,5,7,9,11
		"""
		position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1) #[max_len, 1]
		# 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
		div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model)) #[d_model / 2]
		# div_term = div_term.unsqueeze(0)
		# div_term_numpy = div_term.cpu().numpy()

		# TODO: 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
		test_a = position * div_term
		test_a = test_a.cpu().numpy()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		# 加1个维度，使得pe维度变为：1×max_len×embedding维度
		# (方便后续与一个batch的句子所有词的embedding批量相加)

		"""
		[1,5000,256]
		"""
		pe = pe.unsqueeze(0)
		# 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
		self.register_buffer('pe', pe)

	def forward(self, x):
		# 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
		# (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
		# self.encoder(self.src_embed(src), src_mask)

		test_1 = x.size(1)
		test_2 = self.pe[:, :x.size(1)]
		test_2 = test_2.cpu().numpy()
		"""
		x.size(1):9 可以看做是句子的长度
		self.pe[:, :x.size(1)]:  从5000个句子中取9行 -> [1,9,256] 
		
		[64 9 256] + [1,9, 256] //每个句子都加入PosEmbedding信息
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		来自self.encoder(self.src_embed(src), src_mask)	
		"""

		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)

# 可见，这里首先是按照最大长度max_len生成一个位置，而后根据公式计算出所有的向量，在forward函数中根据长度取用即可，非常方便。
#
#	 注意要设置requires_grad=False，因其不参与训练。
#
# 下面画一下位置嵌入，可见纵向观察，随着embedding dimensionembedding dimension增大，位置嵌入函数呈现不同的周期变化。

# pe = PositionalEncoding(16, 0, 100)
# positional_encoding = pe.forward(Variable(torch.zeros(1, 100, 16, device=DEVICE)))
# plt.figure(figsize=(10,10))
# sns.heatmap(positional_encoding.squeeze().cpu().numpy())
# plt.title("Sinusoidal Function")
# plt.xlabel("hidden dimension")
# plt.ylabel("sequence length")
#
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe.forward(Variable(torch.zeros(1, 100, 20,device=DEVICE))).cpu()
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])


def attention(query, key, value, mask=None, dropout=None):
	# 将query矩阵的最后一个维度值作为d_k
	d_k = query.size(-1)

	"""
	d_k:32
	query:		[64,8,9,32]
	key.transpose:[64,8,32,9]
	torch.matmul(query, key.transpose(-2, -1)):[64,8,9,9]
	
	mask:torch.Size([64, 1, 1, 9])
	如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
	
	torch.matmul(p_attn, value): [64,8,9,9]x[64,8,9,32] = [64,8, 9,32]
	"""

	# TODO: 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

	# 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)

	# TODO: 将mask后的attention矩阵按照最后一个维度进行softmax
	p_attn = F.softmax(scores, dim = -1)

	# 如果dropout参数设置为非空，则进行dropout操作
	if dropout is not None:
		p_attn = dropout(p_attn)
	# 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadedAttention, self).__init__()

		"""
		d_model: 256
		h:	   8
		self.d_k = d_model // h = #256 / 8 = 32 ->得 到一个head的attention表示维度
		定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵 为什么是4个?
		
		第四个矩阵是用来做线性变换self.linears[-1](x)
		"""
		# 保证可以整除
		assert d_model % h == 0
		# 得到一个head的attention表示维度
		self.d_k = d_model // h #256 / 8 = 32
		# head数量
		self.h = h
		# 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"""
		tag_mask:原本为[64,17,17]
		经过mask.unsqueeze(1) -> [64,1,17,17]
		"""
		if mask is not None:
			mask = mask.unsqueeze(1)

		"""
		mask:	torch.Size([64, 1, 1, 9])
		nbatches: batch_size
		query, key, value 计算前: [64,9, 256]
		query, key, value :[64(batch_size),8(head数量),9(单词的数量),32(每个head的维度)]
		交换位置是为了方便计算单词与单词之间的attention
		
		将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
		"""

		# query的第一个维度值为batch size
		nbatches = query.size(0)
		# 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
		# 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
							 for l, x in zip(self.linears, (query, key, value))]
		# 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
		# 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		# 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
		return self.linears[-1](x)

def clones(module, N):
	"""
	克隆模型块，克隆的模型块参数不共享
	"""
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# FeedForwardFeedForward（前馈网络）层其实就是两层线性映射并用激活函数激活
class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# TODO: 请利用init中的成员变量实现Feed Forward层的功能
		"""
		self.w_1(x):  [64, 9, 1024] = [batch_size,src_len,d_model] * [d_model, d_ff] = [batch_size,src_len,d_ff]
		self.w_2(self.dropout(F.relu(self.w_1(x)))) -> [batch_size, src_len, d_models]
		"""
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
	# layer = EncoderLayer
	# N = 6
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		"""
		6个Encoder单元 
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
		layer.size: dmodel
		"""
		# 复制N个encoder layer
		self.layers = clones(layer, N)
		# Layer Norm
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""
		使用循环连续eecode N次(这里为6次)
		这里的Eecoderlayer会接收一个对于输入的attention mask处理

		def encode(self, src, src_mask):
			return self.encoder(self.src_embed(src), src_mask)
		"""
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()

		"""
		self.size是dmodel
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
		"""
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		# SublayerConnection的作用就是把multi和ffn连在一起
		# 只不过每一层输出之后都要先做Layer Norm再残差连接
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		# d_model
		self.size = size

	def forward(self, x, mask):
		# 将embedding层进行Multi head Attention
		"""
		return x + self.dropout(sublayer(self.norm(x)))
		"""
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		# 注意到attn得到的结果x直接作为了下一层的输入
		return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		# 初始化α为全1, 而β为全0
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		# 平滑项
		self.eps = eps

	def forward(self, x):
		# TODO: 请利用init中的成员变量实现LayerNorm层的功能

		"""
		y = \gamma \;(\frac{x-\mu(x)}{\sigma(x)}) + \beta\\
		"""
		# 按最后一个维度计算均值和方差
		mean = x.mean(-1, keepdim = True)
		std = x.std(-1, keepdim = True)

		# TODO: 返回Layer Norm的结果
		return self.a_2 * ( x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
	只不过每一层输出之后都要先做Layer Norm再残差连接

	SublayerConnection(size, dropout)
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		# TODO: 请利用init中的成员变量实现LayerNorm和残差连接的功能
		# 返回Layer Norm和残差连接后结果
		return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		# TODO: 参照EncoderLayer完成成员变量定义

		# 复制N个encoder layer
		self.layers = clones(layer, N)
		# Layer Norm
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""
		使用循环连续decode N次(这里为6次)
		这里的Decoderlayer会接收一个对于输入的attention mask处理
		和一个对输出的attention mask + subsequent mask处理
		"""
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

class DecoderLayer(nn.Module):
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		# Self-Attention
		self.self_attn = self_attn
		# 与Encoder传入的Context进行Attention
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		# 用m来存放encoder的最终hidden表示结果
		m = memory

		"""
		比Encoder多了一个Encoder-Decoder Attention
		DecodeAttention里面d的q,v矩阵是Encoder的输出值
		"""
		# TODO: 参照EncoderLayer完成DecoderLayer的forwark函数
		# Self-Attention：注意self-attention的q，k和v均为decoder hidden
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		# Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
	"Mask out subsequent positions."
	# 设定subsequent_mask矩阵的shape
	attn_shape = (1, size, size)

	# TODO: 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵

	"""
	subsequent_mask
	[[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
	 [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
	 [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
	 [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
	 [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
	 [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
	 [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
	 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
	 [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
	 [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
	 [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
	 [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
	 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
	 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
	 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
	"""
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

	# test_1 = torch.from_numpy(subsequent_mask) == 0

	# TODO: 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
	return torch.from_numpy(subsequent_mask) == 0

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()

class Transformer(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

	def forward(self, src, tgt, src_mask, tgt_mask):
		"""
		self.encode
		self.decode
		"""
		# encoder的结果作为decoder的memory参数传入，进行decode
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
	# vocab: tgt_vocab
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		# decode后的结果，先进入一个全连接层变为词典大小的向量
		self.proj = nn.Linear(d_model, vocab)

	"""
	decode后的结果，先进入一个全连接层变为词典大小的向量
	然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
	"""
	def forward(self, x):

		return F.log_softmax(self.proj(x), dim=-1)

"""
src_vocab： 原语料库的大小(单词的数量)
tgt_vocab： 目标语料库的大小
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
D_MODEL = 256  # embedding维数
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 8  # multihead attention hidden个数
DROPOUT = 0.1  # dropout比例
"""

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"""
	克隆对象
	"""
	c = copy.deepcopy

	"""
	PositionwiseFeedForward: forward 得到的是[batch_size, src_len, d_model]
	
	decode后的结果，先进入一个全连接层变为词典大小的向量
	然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
	
	"""

	# 实例化Attention对象
	attn = MultiHeadedAttention(h, d_model).to(DEVICE)
	# 实例化FeedForward对象
	ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
	# 实例化PositionalEncoding对象
	position = PositionalEncoding(d_model, dropout).to(DEVICE)
	# 实例化Transformer模型对象
	model = Transformer(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
		nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
		Generator(d_model, tgt_vocab)).to(DEVICE)

	# This was important from their code.
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			# 这里初始化采用的是nn.init.xavier_uniform
			nn.init.xavier_uniform_(p)
	return model.to(DEVICE)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
	"""
	传入一个训练好的模型，对指定数据进行预测

	srcs:
	tensor([[  2, 684, 219,   4,   3]], device='cuda:0')

	src_mask:
	tensor([[[True, True, True, True, True]]], device='cuda:0')

	memory.shape:[1,5,256]
	tensor([[[-0.8779,  0.5413, -0.4980,  ...,  1.2255, -0.5578,  0.4383],
		 [-0.8189,  0.3560, -0.3527,  ...,  1.1341, -0.5992,  0.3934],
		 [-0.6386,  0.2663, -0.3175,  ...,  1.0623, -0.5792,  0.3122],
		 [-0.9151,  0.3019, -0.5100,  ...,  1.1343, -0.5595,  0.3246],
		 [-1.0434,  0.2334, -0.6855,  ...,  1.0713, -0.4478,  0.4140]]],
	   device='cuda:0')

	ys
	tensor([[2]], device='cuda:0')

	test_1
	tensor([[[1]]], device='cuda:0')

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
	"""

	# 先用encoder进行encode
	memory = model.encode(src, src_mask)
	# 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
	ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

	# 遍历输出的长度下标
	for i in range(max_len-1):

		test_1 = subsequent_mask(ys.size(1)).type_as(src.data)


		# decode得到隐层表示
		out = model.decode(memory,
						   src_mask,
						   Variable(ys),
						   Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

		"""
		test_2 = out[:, -1] -> [1,256]
		prob = [1,256]x[256,3124] = [1,3124]
		"""
		test_2 = out[:, -1]

		# 将隐藏表示转为对词典各词的log_softmax概率分布表示
		prob = model.generator(out[:, -1])
		# 获取当前位置最大概率的预测词id
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		# 将当前位置预测的字符id与之前的预测内容拼接起来
		ys = torch.cat([ys,
						torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

def test(string):
	with torch.no_grad():
		# 在data的英文数据长度上遍历下标
		# import pdb;
		# pdb.set_trace()
		
		model = make_model(
							src_vocab,
							tgt_vocab,
							LAYERS,
							D_MODEL,
							D_FF,
							H_NUM,
							DROPOUT
						)
		model.load_state_dict(torch.load(SAVE_FILE))

		# string = input() #输入
		en = []
		en.append(['BOS'] + word_tokenize(string) + ['EOS']) #分词


		"""
		将翻译前(英文)数据和翻译后(中文)数据都转换为id表示的形式
		en_dict.get(w, 0) 默认值为0
		以英文句子长度排序的(句子下标)顺序为基准
		"""
		out_en_ids = [[endict[0].get(w, 0) for w in sent] for sent in en]
		for i in range(len(out_en_ids)):
			# TODO: 打印待翻译的英文句子
			en_sent = " ".join([endict[2][w] for w in  out_en_ids[i]])
			print("\n" + en_sent)

			# # TODO: 打印对应的中文句子答案
			# cn_sent =" ".join([data.cn_index_dict[w] for w in  data.dev_cn[i]])
			# print("".join(cn_sent))

			# 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
			src = torch.from_numpy(np.array(out_en_ids[i])).long().to(DEVICE)
			# 增加一维
			src = src.unsqueeze(0)
			# 设置attention mask
			src_mask = (src != 0).unsqueeze(-2)
			# 用训练好的模型进行decode预测
			out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=cndict[0]["BOS"])
			# 初始化一个用于存放模型翻译结果句子单词的列表
			translation = []
			# 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
			for j in range(1, out.size(1)):
				# 获取当前下标的输出字符
				sym = cndict[2][out[0, j].item()]
				# 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
				if sym != 'EOS':
					translation.append(sym)
				# 否则终止遍历
				else:
					break
			# 打印模型翻译输出的中文句子结果
			print("translation: %s" % " ".join(translation))
	return translation


# model = make_model(
# 					src_vocab,
# 					tgt_vocab,
# 					LAYERS,
# 					D_MODEL,
# 					D_FF,
# 					H_NUM,
# 					DROPOUT
# 				)
# model.load_state_dict(torch.load(SAVE_FILE))

# 开始预测
# print(">>>>>>> start evaluate")
# evaluate_start  = time.time()
# # evaluate(data,model)
# string = input() #输入
# test(string)
# print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")
