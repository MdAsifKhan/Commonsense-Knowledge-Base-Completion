from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
from sklearn.utils import shuffle
import torch.optim
import numpy as np
from model import BilinearModel, LSTM_BilinearModel, Avg_DistMult, LSTM_DistMult, LSTM_ERMLP, ERMLP_avg
from torch.autograd import Variable
from evaluation import get_accuracy, auc, find_clf_threshold, stats
import json
import argparse
import os
import pdb


np.random.seed(4086)
torch.manual_seed(4086)

parser = argparse.ArgumentParser(
	description='Train Models for Commonsense KB Reasoning: Bilinear Averaging Model, Bilinear LSTM Model, \
		DistMult Averaging Model, DistMult LSTM Model, ER-MLP Averaging Model, ER-MLP LSTM Model'
)
parser.add_argument('--model', default='BilinearAvg', metavar='',
					help='model to run: {BilinearAvg, BilinearLstm, DistMultAvg, DistMultLstm, \
						ErmlpLstm, ErmlpAvg} (default: BilinearAvg)')
parser.add_argument('--train_file', default='train100k.txt', metavar='',
					help='training dataset to be used: {train100k.txt, train300k.txt, train600k.txt} (default: train100k.txt)')
parser.add_argument('--valid_file', default='dev1.txt', metavar='',
					help='validation dataset to be used: {dev1.txt, dev2.txt} (default: dev1.txt)')
parser.add_argument('--test_file', default='dev2.txt', metavar='',
					help='test dataset to be used: {test.txt, dev2.txt} (default: dev2.txt)')
parser.add_argument('--rel_file', default='rel.txt', metavar='',
					help='file containing ConceptNet relation')
parser.add_argument('--pretrained_weights_file', default='embeddings.txt', metavar='', \
						help='name of pretrained weights file')
parser.add_argument('--k', type=int, default=150, metavar='',
					help='embedding relation dim (default: 150)')
parser.add_argument('--dropout_p', type=float, default=0.2, metavar='',
					help='Probability of dropping out neuron (default: 0.2)')
parser.add_argument('--mlp_hidden', type=int, default=100, metavar='',
					help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--mb_size', type=int, default=50, metavar='',
					help='size of minibatch (default: 50)')
parser.add_argument('--negative_samples', type=int, default=3, metavar='',
					help='number of negative samples per positive sample  (default: 3)')
parser.add_argument('--nm_epoch', type=int, default=1000, metavar='',
					help='number of training epoch (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='',
					help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=1e-3, metavar='',
					help='decaying learning rate every n epoch (default: 1e-3)')
parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='',
					help='L2 weight decay (default: 1e-3)')
parser.add_argument('--embeddings_lambda', type=float, default=1e-2, metavar='',
					help='prior strength for embeddings. Constraints embeddings norms to at most one  (default: 1e-2)')
parser.add_argument('--normalize_embed', default=False, type=bool, metavar='',
					help='whether to normalize embeddings to unit euclidean ball (default: False)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
					help='directory to save model checkpoint, saved every epoch (default: models/)')
parser.add_argument('--use_gpu', default=False, action='store_true',
					help='whether to run in the GPU')

args = parser.parse_args()

if args.use_gpu:
	torch.cuda.manual_seed(4086)

	
checkpoint_dir = args.checkpoint_dir
checkpoint_dir = '{}/ConceptNet'.format(checkpoint_dir.rstrip('/'))
checkpoint_path = '{}/model.bin'.format(checkpoint_dir)

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)


DATA_ROOT = 'data/ConceptNet/'
# 
train_file = DATA_ROOT+args.train_file
valid_file = DATA_ROOT+args.valid_file
test_file = DATA_ROOT+args.test_file
rel_file = DATA_ROOT+args.rel_file
pretrained_file = DATA_ROOT+args.pretrained_weights_file

# Prepare Training DataSet
preprocessor = preprocess()
preprocessor.read_train_triples(train_file)
preprocessor.read_relations(rel_file)
preprocessor.pretrained_embeddings(filename=pretrained_file)
n_r = preprocessor.n_rel
train_triples = preprocessor.train_triples
train_idx = preprocessor.triple_to_index(train_triples)
preprocessor.get_max_len(train_idx)
train_data = preprocessor.pad_idx_data(train_idx)

pretrained_weights = preprocessor.embedding_matrix()
embedding_dim = preprocessor.embedding_dim
word_id_map = preprocessor.word_id_map
rel_id_map = preprocessor.rel
np.save(DATA_ROOT + 'pretrained_weights.npy',pretrained_weights)

with open(DATA_ROOT + 'word_id_map.dict','w') as f:
	j = json.dumps(word_id_map)
	f.write(j)
with open(DATA_ROOT + 'rel_id_map.dict','w') as f:
	j = json.dumps(rel_id_map)
	f.write(j)

# Prepare Validation DataSet
preprocessor.read_valid_triples(valid_file)
valid_triples = preprocessor.valid_triples
valid_idx = preprocessor.triple_to_index(valid_triples, dev=True)
valid_data = preprocessor.pad_idx_data(valid_idx, dev=True)
valid_label = valid_data[3]
valid_data = valid_data[0], valid_data[1], valid_data[2]

# Prepare Test DataSet
preprocessor.read_test_triples(test_file)
test_triples = preprocessor.test_triples
test_idx = preprocessor.triple_to_index(test_triples, dev=True)
test_data = preprocessor.pad_idx_data(test_idx, dev=True)
test_label = test_data[3]
test_data = test_data[0], test_data[1], test_data[2]

# Parameters of Model
batch_size = args.mb_size
# Embedding Normalization
lw = args.embeddings_lambda
gpu = args.use_gpu
epochs = args.nm_epoch
sampling_factor = args.negative_samples
lr = args.lr
lr_decay = args.lr_decay
embedding_rel_dim = args.k
weight_decay = args.weight_decay
mlp_hidden = args.mlp_hidden
dropout_p = args.dropout_p
normalize_embed = args.normalize_embed
thresh = 0.5
#Batch Data Loader
train_loader = DataLoader(TripleDataset(train_data), batch_size=batch_size,shuffle=True, num_workers=4)
lstm_ = False
batch_size = batch_size + (batch_size*2)*sampling_factor
# Model
if args.model == 'BilinearAvg':
	model = BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'BilinearLstm':
	lstm_ = True
	model = LSTM_BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'DistMultAvg':
	model = Avg_DistMult(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)	
elif args.model == 'DistMultLstm':
	model = LSTM_DistMult(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
	lstm_ = True
elif args.model == 'ErmlpLstm':
	model = LSTM_ERMLP(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, mlp_hidden=mlp_hidden, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
	lstm_ = True
elif args.model == 'ErmlpAvg':
	model = ERMLP_avg(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, mlp_hidden=mlp_hidden, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
else:
	raise Exception('Unknown model!')	


optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)


for epoch in range(epochs):
	epoch_loss = []
	#lr_n = lr * (0.5 ** (epoch // lr_decay_every))
	for i, train_positive_data in enumerate(train_loader,0):
		model.zero_grad()
		if lstm_:
			model.init_hidden()
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
		if normalize_embed:
			model.normalize_embeddings()
		epoch_loss.append(loss.cpu().data.numpy())

	if epoch%10 == 0:
		pred_score = model.predict_proba(score)
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
