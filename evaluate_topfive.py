from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
import numpy as np
from model import DistMult
from torch.autograd import Variable
from evaluation import evaluate_model, stats

import pdb

np.random.seed(4086)
torch.manual_seed(4086)
torch.cuda.manual_seed(4086)

gpu = True
embedding_rel_dim = 150
embedding_dim = 200

sub_ = ''
pred_ = ''
eval_type = 'topfive'

# Load Trained Model
model = BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=0.2, gpu=gpu)
model_name = 'models/ConceptNet/model1.bin'
state = torch.load(model_name, map_location=lambda storage, loc: storage)
model.load_state_dict(state)

'''
relations = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
'NotHasA','NotIsA','NotHasProperty','NotCapableOf']
'''

with open('data/word_id_map.dict','r') as f:
	word_id_map = json.load(f)

with open('data/rel_id_map.dict','r') as f:
	rel_id_map = json.load(f)


evaluate_model(sub_, obj_, word_id_map, rel_id_map, eval_type='topfive')