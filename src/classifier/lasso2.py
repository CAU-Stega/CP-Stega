import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.base_bpw import BASE_BPW
from torch.autograd import Variable
from scipy import *
from sklearn import metrics
import numpy as np

class LASSO(BASE):
    def __init__(self, ebd_dim, args):
        super(LASSO, self).__init__(args)
        self.ebd_dim = ebd_dim
        self.args = args
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
        pass

    '''def prepare_data(self):
        data = np.genfromtxt('./example.dat', delimiter=',')
        x = data[:, 0:100]
        y = data[:, 100].reshape(-1, 1)
        X = np.column_stack((np.ones((x.shape[0], 1)), x))
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]
        return X_train, y_train, X_test, y_test'''

    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def l1_loss(self, X, y, w, b, alpha):
        num_train = X.shape[0]
        num_feature = X.shape[1]

        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(alpha * abs(w))
        dw = np.dot(X.T, (y_hat - y)) / num_train + alpha * np.vectorize(self.sign)(w)
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    def lasso_train(self, X, y, learning_rate, epochs):
        loss_list = []
        w, b = self.initialize_params(X.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.l1_loss(X, y, w, b, 0.1)
            w += -learning_rate * dw
            b += -learning_rate * db
            loss_list.append(loss)

            if i % 300 == 0:
                print('epoch %d loss %f' % (i, loss))

            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss, loss_list, params, grads

    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b
        return y_pred


if __name__ == '__main__':
    lasso = Lasso()
    X_train, y_train, X_test, y_test = lasso.prepare_data()
    loss, loss_list, params, grads = lasso.lasso_train(X_train, y_train, 0.01, 3000)
    print(params)
    y_pred = lasso.predict(X_test, params)
    print(r2_score(y_test, y_pred))

