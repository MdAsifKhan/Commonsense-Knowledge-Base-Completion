from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
from sklearn.utils import shuffle
import torch.optim
import numpy as np
from model import DistMult
from torch.autograd import Variable
from evaluation import get_accuracy, auc, find_clf_threshold, stats
import json
import pdb

np.random.seed(4086)
torch.manual_seed(4086)
torch.cuda.manual_seed(4086)

checkpoint_dir = 'models/'
checkpoint_dir = '{}/ConceptNet'.format(checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/model1.bin'.format(checkpoint_dir)

preprocessor = preprocess()

preprocessor.read_train_triples('data/ConceptNet/train100k.txt')
preprocessor.read_relations('data/ConceptNet/rel.txt')
preprocessor.pretrained_embeddings(filename='data/ConceptNet/embeddings.txt')

n_r = preprocessor.n_rel
train_triples = preprocessor.train_triples
train_idx = preprocessor.triple_to_index(train_triples)
preprocessor.get_max_len(train_idx)
train_data = preprocessor.pad_idx_data(train_idx)

pretrained_weights = preprocessor.embedding_matrix()
embedding_dim = preprocessor.embedding_dim
word_id_map = preprocessor.word_id_map
rel_id_map = preprocessor.rel
with open('data/word_id_map.dict','w') as f:
	j = json.dumps(word_id_map)
	f.write(j)
with open('data/rel_id_map.dict','w') as f:
	j = json.dumps(rel_id_map)
	f.write(j)


batch_size = 100
# Embedding Normalization
lw = 1e-3 
train_loader = DataLoader(TripleDataset(train_data), batch_size=batch_size,shuffle=True, num_workers=4)

preprocessor.read_valid_triples('data/ConceptNet/dev1.txt')
valid_triples = preprocessor.valid_triples
valid_idx = preprocessor.triple_to_index(valid_triples, dev=True)
valid_data = preprocessor.pad_idx_data(valid_idx, dev=True)
valid_label = valid_data[3]
valid_data = valid_data[0], valid_data[1], valid_data[2]


preprocessor.read_test_triples('data/ConceptNet/dev2.txt')
test_triples = preprocessor.test_triples
test_idx = preprocessor.triple_to_index(test_triples, dev=True)
test_data = preprocessor.pad_idx_data(test_idx, dev=True)
test_label = test_data[3]
test_data = test_data[0], test_data[1], test_data[2]
gpu = True
embedding_rel_dim = 150
model = BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=0.2, gpu=gpu)
epochs = 1000
lr = 0.01
#lr_decay_every = 10
lr_decay = 1e-4
weight_decay = 1e-3
sampling_factor = 3
thresh = 0.5
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
		train_p = np.concatenate((train_p, train_neg_p))
		train_s, train_o, train_p, train_label = shuffle(train_s, train_o, train_p, train_label, random_state=4086)
		score = model.forward(train_s, train_o, train_p)
		loss = model.bce_loss(score, train_label, average=True)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		epoch_loss.append(loss.cpu().data.numpy())

	if epoch%10 == 0:
		pred_score = model.predict(score, sigmoid=True)
		score = score.cpu().data.numpy() if gpu else score.data.numpy()
		train_auc_score = auc(score, train_label)
		print('Epoch {0}\tTrain Loss value: {1}'.format(epoch, stats(epoch_loss)))
		print('Epoch {0}\tTraining AUC Score: {1}'.format(epoch, train_auc_score))

		# Do evaluation on Dev Set		
		valid_s, valid_o, valid_p = valid_data
		score_val = model.forward(valid_s, valid_o, valid_p)
		score_val = score_val.cpu().data.numpy() if gpu else score_val.data.numpy()
		val_acc, thresh = find_clf_threshold(score_val)
		
		print('Threshold {0}'.format(thresh))
		val_auc_score = auc(score_val, valid_label)

		print('Epoch {0}\tValidation Accuracy: {1}'.format(epoch, val_acc))
		print('Epoch {0}\tValidation AUC Score: {1}'.format(epoch, val_auc_score))	

		# Do evaluation on Dev Set 2
		test_s, test_o, test_p = test_data
		score_test = model.forward(test_s, test_o, test_p)
		score_test = score_test.cpu().data.numpy() if gpu else score_test.data.numpy()
		test_acc = get_accuracy(score_test, thresh)

		#score_test = score_test.cpu().data.numpy() if gpu else score_test.data.numpy()
		test_auc_score = auc(score_test, test_label)

		print('Epoch {0}\tTest Accuracy: {1}'.format(epoch, test_acc))
		print('Epoch {0}\tTest AUC Score: {1}'.format(epoch, test_auc_score))	


	torch.save(model.state_dict(), checkpoint_path)