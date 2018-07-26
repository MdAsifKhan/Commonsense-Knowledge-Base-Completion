from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
import numpy as np
from model import DistMult
from torch.autograd import Variable
from evaluation import get_accuracy, auc, find_clf_threshold, stats

import pdb

np.random.seed(4086)
torch.manual_seed(4086)
torch.cuda.manual_seed(4086)

# Read and Prepare DataSet
preprocessor = preprocess()
preprocessor.read_test_triples('data/ConceptNet/dev2.txt')
test_triples = preprocessor.test_triples
test_idx = preprocessor.triple_to_index(test_triples, dev=True)
test_data = preprocessor.pad_idx_data(test_idx, dev=True)
test_label = test_data[3]
test_data = test_data[0], test_data[1], test_data[2]
gpu = True
embedding_rel_dim = 150

# Load Trained Model
model = BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=0.2, gpu=gpu)
model_name = 'models/ConceptNet/model1.bin'
state = torch.load(model_name, map_location=lambda storage, loc: storage)
model.load_state_dict(state)

# Test Model
test_s, test_o, test_p = test_data
score_test = model.forward(test_s, test_o, test_p)
score_test = score_test.cpu().data.numpy() if gpu else score_test.data.numpy()
test_acc = get_accuracy(score_test, thresh)
test_auc_score = auc(score_test, test_label)

print('Test Accuracy: {0}'.format(test_acc))
print('Test AUC Score: {0}'.format(test_auc_score))