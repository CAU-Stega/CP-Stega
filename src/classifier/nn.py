import torch
from classifier.base import BASE
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.base_bpw import BASE_BPW
from torch.autograd import Variable
from scipy import *
from sklearn import metrics
import numpy as np


class NN(BASE):
    '''
        Nearest neighbour classifier
    '''
    def __init__(self, ebd_dim, args):
        super(NN, self).__init__(args)
        self.ebd_dim = ebd_dim

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return None (a placeholder for loss)
        '''
        if self.args.nn_distance == 'l2':
            dist = self._compute_l2(XS, XQ)
        elif self.args.nn_distance == 'cos':
            dist = self._compute_cos(XS, XQ)
        else:
            raise ValueError("nn_distance can only be l2 or cos.")

        # 1-NearestNeighbour
        nn_idx = torch.argmin(dist, dim=1)
        pred = YS[nn_idx]

        temp = np.ones(self.args.query * self.args.way)
        temp1 = np.zeros(self.args.query * self.args.way)
        a = torch.tensor(temp, device='cuda:1')
        b = torch.tensor(temp1, device='cuda:1')

        result_2 = torch.where(pred > 0, a, b)
        true_2 = torch.where(YQ > 0, a, b)
        result_2 = Variable(result_2.float(), requires_grad=True)
        true_2 = Variable(true_2.float(), requires_grad=True)

        if self.args.mode=='test':
            temp_a = np.ones(self.args.query * self.args.way)
            temp_b = np.zeros(self.args.query * self.args.way)
            a = torch.tensor(temp_a, device='cuda:1')
            b = torch.tensor(temp_b, device='cuda:1')
            result = torch.where(pred > 0, a, b)
            true = torch.where(YQ > 0, a, b)
            labels = true.cpu().numpy()
            r = result.cpu().detach().numpy()
            precious = metrics.precision_score(labels, r, average='macro')
            recall = metrics.recall_score(labels, r, average='macro')
            # print("precious:",precious)
            # print("recall:", recall)
            acc = torch.mean((result_2 == true_2).float()).item()
            return acc,precious,recall
        else:

            acc = torch.mean((result_2 == true_2).float()).item()
        #print("pred:",pred)
        #print("YQ:",YQ)
        #print("YS",YS)
        return acc, None
