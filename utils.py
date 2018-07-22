from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as random
import torch
import pdb

class preprocess:

	def __init__(self):
		self.train_triples = []
		self.valid_triples = []
		self.test_triples = []
		self.maxlen_s = 0
		self.maxlen_o = 0
		self.word_vec_map = {}
		self.word_id_map = {}
		self.embedding_weights = {}

	def read_train_triples(self, filename):
		self.train_triples = [line.strip().split('\t') for line in open(filename,'r')]

	def read_valid_triples(self, filename):
		self.valid_triples = [line.strip().split('\t') for line in open(filename,'r')]

	def read_test_triples(self, filename):
		self.test_triples = [line.strip().split('\t') for line in open(filename,'r')]

	def embedding_matrix(self):
		matrix = np.zeros((len(self.embedding_weights), self.embedding_dim))
		for key,value in self.embedding_weights.items():
			matrix[key,:] = value
		return matrix
		
	def load_embedding(self, filename):
		with open(filename,'r') as f:
			for i,line in enumerate(f):
				pair = line.split()
				word = pair[0]
				embedding = np.array(pair[1:], dtype='float32')
				self.word_vec_map[word] = embedding
				self.word_id_map[word] = i+1
			self.embedding_dim = len(embedding)

	def pretrained_embeddings(self, filename='embeddings.txt'):
		self.load_embedding(filename)
		for word, id_ in self.word_id_map.items():
			self.embedding_weights[id_] = self.word_vec_map[word]
		self.word_id_map['PAD'] = 0		
		self.embedding_weights[self.word_id_map['PAD']] = np.zeros(self.embedding_dim)

	def sentence2idx(self, sentence):
		return [self.word_id_map[word] if word in self.word_id_map else self.word_id_map['UUUNKKK'] for word in sentence]
			
	def	triple_to_index(self, triples, dev=False):
		triple2idx = []
		for triple in triples:
			if dev:
				p, s, o, label = triple[0].split(' '), triple[1].split(' '), triple[2].split(' '), int(triple[3])
				triple2idx.append([self.sentence2idx(p),self.sentence2idx(s),self.sentence2idx(o), label])
			else:
				p, s, o = triple[0].split(' '), triple[1].split(' '), triple[2].split(' ')
				triple2idx.append([self.sentence2idx(p),self.sentence2idx(s),self.sentence2idx(o)])
		return triple2idx			

	def get_max_len(self, tripleidx):
		for triple in tripleidx:
			self.maxlen_s = max(self.maxlen_s, len(triple[1]))
			self.maxlen_o = max(self.maxlen_o, len(triple[2]))

	def pad_idx_data(self, tripleidx, dev=False):
		all_s, all_p, all_o = [], [], []
		if dev:
			label =[]
		for triple in tripleidx:
			pad_s = triple[1] + [self.word_id_map['PAD']]*(self.maxlen_s - len(triple[1]))
			all_s.append(pad_s)

			pad_o = triple[2] + [self.word_id_map['PAD']]*(self.maxlen_o - len(triple[2]))
			all_o.append(pad_o)

			all_p.append(triple[0])
			if dev:
				label.append(triple[3])
		if dev:
			return np.array(all_s), np.array(all_o), np.array(all_p), np.array(label)	
		
		return np.array(all_s), np.array(all_o), np.array(all_p)


class TripleDataset(Dataset):
	
	def __init__(self, data, dev=False):
		self.dev = dev
		if self.dev:
			self.s, self.o, self.p, self.label = data[0], data[1], data[2], data[3]
		else:
			self.s, self.o, self.p = data[0], data[1], data[2]
		self.len = len(self.s)
	
	def __getitem__(self, index):
		if self.dev:
			return self.s[index], self.o[index], self.p[index], self.label[index]	
		return self.s[index], self.o[index], self.p[index]

	def __len__(self):
		return self.len

def stats(values):
    return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

def sample_negatives(data, type='RAND',sampling_factor=10):
	s_data, o_data, p_data = data[0], data[1], data[2]
	data_len = len(s_data)
	neg_s_data = []
	neg_o_data = []
	neg_p_data = []
	while(sampling_factor):
		for i in range(len(s_data)):
			idx_s = random.randint(0,data_len-1)
			idx_o = random.randint(0,data_len-1)
			idx_p = random.randint(0,data_len-1)
			if idx_s != idx_o:
				neg_s_data.append(s_data[idx_s].numpy())
				neg_o_data.append(o_data[idx_o].numpy())
				neg_p_data.append(p_data[idx_p].numpy())
		sampling_factor -= 1
	return torch.from_numpy(np.array(neg_s_data)), torch.from_numpy(np.array(neg_o_data)), torch.from_numpy(np.array(neg_p_data))