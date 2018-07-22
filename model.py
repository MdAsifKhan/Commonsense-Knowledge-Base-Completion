import torch
from torch.nn import functional as F
from torch.autograd import Variable
import pdb

class DistMult(torch.nn.Module):
    def __init__(self, embedding_dim, weights, input_dropout=0.2, gpu=True):
        super(DistMult, self).__init__()
        self.embed = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.init(weights)
        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed.weight.data.copy_(torch.from_numpy(weights))

    def forward(self, s, o, p):
        freq_s = Variable(torch.from_numpy(s.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_o = Variable(torch.from_numpy(o.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_s = freq_s.cuda() if self.gpu else freq_s
        freq_o = freq_o.cuda() if self.gpu else freq_o

        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = s.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed(s).sum(dim=1)
        s_embedded = s_embedded.mul(freq_s.unsqueeze(1))
        o_embedded = self.embed(o).sum(dim=1)
        o_embedded = o_embedded.mul(freq_o.unsqueeze(1))
        p_embedded = self.embed(p)

        s_embedded = self.inp_drop(s_embedded)
        o_embedded = self.inp_drop(o_embedded)
        p_embedded = self.inp_drop(p_embedded).squeeze(1)

        pred = torch.sum(s_embedded * p_embedded * o_embedded, 1)
        pred = F.sigmoid(pred)
        return pred