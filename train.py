from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess, stats
from sklearn.utils import shuffle
import torch.optim
import numpy as np
from model import DistMult
from torch.autograd import Variable

import pdb

np.random.seed(4086)
torch.manual_seed(4086)
torch.cuda.manual_seed(4086)

checkpoint_dir = 'models/'
checkpoint_dir = '{}/ConceptNet'.format(checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/model1.bin'.format(checkpoint_dir)

preprocessor = preprocess()
preprocessor.read_train_triples('data/ConceptNet/train100k.txt')
preprocessor.pretrained_embeddings(filename='data/ConceptNet/embeddings.txt')

train_triples = preprocessor.train_triples
train_idx = preprocessor.triple_to_index(train_triples)
preprocessor.get_max_len(train_idx)
train_data = preprocessor.pad_idx_data(train_idx)

pretrained_weights = preprocessor.embedding_matrix()
embedding_dim = preprocessor.embedding_dim

train_loader = DataLoader(TripleDataset(train_data), batch_size=20,shuffle=True, num_workers=4)

preprocessor.read_valid_triples('data/ConceptNet/dev1.txt')
valid_triples = preprocessor.valid_triples
valid_idx = preprocessor.triple_to_index(valid_triples, dev=True)
valid_data = preprocessor.pad_idx_data(valid_idx, dev=True)
valid_label = valid_data[3]
valid_data = valid_data[0], valid_data[1], valid_data[2]
gpu = True
model = DistMult(embedding_dim=embedding_dim, weights=pretrained_weights, gpu=gpu)

epochs = 100
lr = 0.1
#lr_decay_every = 10
lr_decay = 1e-4
weight_decay = 1e-4
sampling_factor = 5
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
for epoch in range(epochs):
	epoch_loss = []
	#lr_n = lr * (0.5 ** (epoch // lr_decay_every))
	for i, train_positive_data in enumerate(train_loader,0):
		optimizer.zero_grad()
		
		train_s, train_o, train_p = train_positive_data
		train_negative_data = sample_negatives(train_positive_data, sampling_factor=sampling_factor)
		train_neg_s, train_neg_o, train_neg_p = train_negative_data
		train_label = np.concatenate((np.ones(len(train_s)), np.zeros(len(train_neg_s))))
		train_s = np.vstack([train_s, train_neg_s])
		train_o = np.vstack([train_o, train_neg_o])
		train_p = np.vstack([train_p, train_neg_p])
		train_s, train_o, train_p, train_label = shuffle(train_s, train_o, train_p, train_label, random_state=4086)

		
		score = model.forward(train_s, train_o, train_p)
		train_label = Variable(torch.from_numpy(train_label).type(torch.FloatTensor))
		train_label = train_label.cuda() if gpu else train_label 
		loss = model.loss(score, train_label)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		epoch_loss.append(loss)

		# Do evaluation on Dev Set
	if epoch%10 == 0:
		print('Epoch {0}\tLoss value: {1}'.format(epoch, stats(epoch_loss)))
	pdb.set_trace()
	torch.save(model.state_dict(), checkpoint_path)





