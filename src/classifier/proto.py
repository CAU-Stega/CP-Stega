import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from torch.autograd import Variable
from scipy import *
from sklearn import metrics
import numpy as np

class PROTO(BASE):
    '''
        PROTOTIPICAL NETWORK FOR FEW SHOT LEARNING
    '''
    def __init__(self, ebd_dim, args):
        super(PROTO, self).__init__(args)
        self.ebd_dim = ebd_dim

        if args.embedding == 'meta':
            self.mlp = None
            print('No MLP')
        else:
            self.mlp = self._init_mlp(
                    self.ebd_dim, self.args.proto_hidden, self.args.dropout)

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        if self.mlp is not None:
            XS = self.mlp(XS)
            XQ = self.mlp(XQ)

        YS, YQ = self.reidx_y(YS, YQ)

        prototype = self._compute_prototype(XS, YS)

        pred = -self._compute_l2(prototype, XQ)

        #print("pred:", pred)
        #print("YQ:", YQ)
        loss_ce = F.cross_entropy(pred, YQ)
        loss_margin = F.multi_margin_loss(pred, YQ) 

        result = torch.argmax(pred, dim=1)
        result = result.float()
        result = Variable(result.float(), requires_grad=False)
        tar = Variable(YQ.float(), requires_grad=True)
        loss_kl = F.kl_div(result.softmax(dim=-1).log(), tar.softmax(dim=-1), reduction='sum')

        loss =  loss_kl + 0.5 * loss_margin
        #print('p')
        if self.args.mode=='test':
            temp_a = np.ones(self.args.query * self.args.way)
            temp_b = np.zeros(self.args.query * self.args.way)
            a = torch.tensor(temp_a, device='cuda:1')
            b = torch.tensor(temp_b, device='cuda:1')
            result = torch.where(result > 0, a, b)
            true = torch.where(YQ > 0, a, b)
            labels = true.cpu().numpy()
            r = result.cpu().detach().numpy()
            precious = metrics.precision_score(labels, r, average='macro')
            recall = metrics.recall_score(labels, r, average='macro')
            # print("precious:",precious)
            # print("recall:", recall)
            acc = BASE.compute_acc(pred, YQ, self.args.way, self.args.query)
            return acc,precious,recall
        else:
            acc = BASE.compute_acc(pred, YQ,self.args.way,self.args.query)

            return acc, loss
