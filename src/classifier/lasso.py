import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.base_bpw import BASE_BPW
from torch.autograd import Variable
from scipy import *
from sklearn import metrics
import numpy as np
import copy
from sklearn import linear_model
#from classifier.lasso.linear import dict_learning, sparse_encode

class LASSO(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(LASSO, self).__init__(args)
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
        '''
                    Compute the W matrix of ridge regression
                    @param XS: support_size x ebd_dim
                    @param YS_onehot: support_size x way

                    @return W: ebd_dim * way
                '''

        W = XS.t() @ torch.inverse(
            XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot
       # W = torch.inverse( XS.t() @ XS) @ XS.t()@YS_onehot
        '''m = XS.shape[0]
        xMat = np.mat(XS.cpu())
        yMat = np.mat(YS_onehot.reshape(-1, 1).cpu())

        w = np.ones(XS.shape[1].cpu()).reshape(-1, 1)

        for n in range(self.args.way*self.args.shot):

            out_w = copy.copy(w)
            for i, item in enumerate(w):
                # 在每一个W值上找到使损失函数收敛的点
                for j in range(self.args.way*self.args.shot):
                    h = xMat * w
                    gradient = xMat[:, i].T * (h - yMat) / m + self.lam * np.sign(w[i])
                    w[i] = w[i] - gradient * self.args.leaening
                    if abs(gradient) < 1e-3:
                        break
            out_w = np.array(list(map(lambda x: abs(x) < 1e-3, out_w - w)))
            if out_w.all():
                break'''

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

        #dictionary, losses = dict_learning(XS, n_components=50, alpha=0.5, algorithm='ista')
        #coeffs = sparse_encode(XS, dictionary, alpha=0.2, algorithm='interior-point')
        #print(YQ)
        # 训练 Lasso 模型
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import LinearRegression
        YS, YQ = self.reidx_y(YS, YQ)
        # 加载Lasso算法
        #alpha = 0.5
        # 设置惩罚参数
        #lasso = Lasso(alpha=alpha)
        # 获得Lasso全部参数

        #训练样本后，预测
        #X_testr2_score_lasso = r2_score(y_test, y_pred_lasso)
        # 计算拟合优度

        #print("r^2 on test data : %f" % r2_score_lasso)
        #YS, YQ = self.reidx_y(YS, YQ)
        Lasso=LinearRegression()

        Lasso.fit(XS.cpu().detach().numpy(), YS.cpu().detach().numpy())
        pred1=Lasso.predict(XQ.cpu().detach().numpy())
        #W1=Lasso.coef_
        #pred=torch.from_numpy(pred)
        print(pred1)


        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)
        #print("w1=",len(pred1))
        #print("w=", W.shape)

        #pred2 = (10.0 ** self.alpha) * XQ @ W + self.beta
        #print(pred2)
        label=YQ
        #print("pred:",pred)
        #print("YQ:", YQ)
        for i in range(0,len(pred1)):
            if 0<pred1[i] and pred1[i]<=0.5:
                pred1[i]=0
            elif 0.5<pred1[i] and pred1[i]<=1:
                pred1[i] = 1
            elif 1<pred1[i] and pred1[i]<=1.5:
                pred1[i] = 1
            else :
                pred1[i] = 2
        pred=torch.tensor(pred1,device='cuda:1')


        temp = np.ones(self.args.query * self.args.way)
        temp1 = np.zeros(self.args.query * self.args.way)
        a = torch.tensor(temp, device='cuda:1')
        b = torch.tensor(temp1, device='cuda:1')

        #YQ = YQ.float()
        #result = torch.argmax(pred, dim=1)
        result_2 = torch.where(pred > 0, a, b)
        true_2 = torch.where(YQ > 0, a, b)
        result_2 = Variable(result_2.float(), requires_grad=True)
        true_2 = Variable(true_2.float(), requires_grad=True)
        print(YQ)

        loss = F.cross_entropy(pred, YQ)+0.5*F.kl_div(result_2.softmax(dim=-1).log(), true_2.softmax(dim=-1), reduction='sum')

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

