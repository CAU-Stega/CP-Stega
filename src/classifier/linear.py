import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.base_bpw import BASE_BPW
from torch.autograd import Variable
from scipy import *
from sklearn import metrics
import numpy as np

class LINEAR(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(LINEAR, self).__init__(args)
        self.ebd_dim = ebd_dim
        self.args=args
        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = torch.inverse( XS.t() @ XS) @ XS.t()@YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta
        label=YQ
        #print("pred:",pred)
        #print("YQ:", YQ)
        temp = np.ones(self.args.query * self.args.way)
        temp1 = np.zeros(self.args.query * self.args.way)
        a = torch.tensor(temp, device='cuda:1')
        b = torch.tensor(temp1, device='cuda:1')

        #YQ = YQ.float()
        result = torch.argmax(pred, dim=1)
        result_2 = torch.where(result > 0, a, b)
        true_2 = torch.where(YQ > 0, a, b)
        result_2 = Variable(result_2.float(), requires_grad=True)
        true_2 = Variable(true_2.float(), requires_grad=True)

        loss = F.cross_entropy(pred, YQ)+\
               0.5*F.kl_div(result_2.softmax(dim=-1).log(), true_2.softmax(dim=-1), reduction='sum')

        #asymmetricKL=sum(pred * log(pred / YQ))
        #asymmetricKL_=sum(pred * log(YQ / pred))

        #loss = F.cross_entropy(pred, YQ)
        #print("pred:",pred)
        #print("YQ:",YQ)
        #loss=(asymmetricKL+asymmetricKL_)/2.00
        '''temp = np.ones(self.args.query * self.args.way)
        temp1 = np.zeros(self.args.query * self.args.way)
        a = torch.tensor(temp, device='cuda:1')
        b = torch.tensor(temp1, device='cuda:1')

        YQ=YQ.float()
        result = torch.argmax(pred, dim=1)
        result_2 = torch.where(result > 0, a, b)
        true_2 = torch.where(YQ > 0, a, b)
        result_2=Variable(result_2.float(), requires_grad=True)
        true_2 = Variable(true_2.float(), requires_grad=True)
        result_multi=result.float()
        result_multi = Variable(result_multi.float(), requires_grad=True)
        tar_multi = Variable(YQ.float(), requires_grad=True)
        loss = 0.5*F.kl_div(result_multi.softmax(dim=-1).log(), tar_multi.softmax(dim=-1), reduction='sum')\
               +F.kl_div(result_2.softmax(dim=-1).log(), true_2.softmax(dim=-1), reduction='sum')
        #print("loss:",loss)
        #print("loss_n:",F.kl_div(result_multi.softmax(dim=-1).log(), tar_multi.softmax(dim=-1), reduction='sum'))
        #print("loss_2:", F.kl_div(result_2.softmax(dim=-1).log(), true_2.softmax(dim=-1), reduction='sum'))'''
        '''result = torch.argmax(pred.float(), dim=1)

        result = Variable(result.float(), requires_grad=True).long()
        tar = Variable(YQ.float(), requires_grad=True).long()
        result=self._label2onehot(result)
        tar=self._label2onehot(tar)
        loss=F.kl_div(result.softmax(dim=-1).log(), tar.softmax(dim=-1), reduction='sum')'''

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
         if self.args.bpw:
            #print("BASE1")
            acc = BASE_BPW.compute_acc(pred, label)
         else:
            acc = BASE.compute_acc(pred, YQ,self.args.way,self.args.query)

        #print("loss:",loss)
         return acc, loss


'''def asymmetricKL(P, Q):
    return sum(P * log(P / Q))  # calculate the kl divergence between P and Q


def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00'''

