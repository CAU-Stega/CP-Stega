import os
import time
import datetime
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from train.utils import named_grad_param, grad_param, get_norm

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from ipywidgets import interact, fixed
from matplotlib.ticker import NullFormatter


X, y = make_circles(100, factor=0.1, noise=.1)


def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_3D(X, y, elev=30, azim=30):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='rainbow')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()

def draw(x, y, args):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
    clf = SVC(kernel="linear").fit(x, y)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
    plot_svc_decision_function(clf)
    r = np.exp(-(x ** 2).sum(1))
    rlim = np.linspace(min(r), max(r), 0.2)
    interact(plot_3D(x, y, elev=30, azim=30), elev=[0, 30], azip=(-180, 180), X=fixed(x), y=fixed(y))
    path = args.embedding + '_' + "Twitter3D"
    plt.savefig(path)

def plot_embedding_3d(X, y, args, title):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    color = ['lightcoral', 'tan', 'greenyellow', 'burlywood', 'crimson', 'palegreen']
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=plt.cm.Set1(np.argmax(y[:], axis=1) / 10.0))
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        if y[i]==0:
            ax.text(X[i, 0], X[i, 1], X[i, 2],str(y[i]) ,c='cornflowerblue', fontdict={'weight': 'bold', 'size': 4})
        else:
          #print(y[i])
          ax.text(X[i, 0], X[i, 1], X[i, 2],str(y[i]) ,c=color[y[i]-7], fontdict={'weight': 'bold', 'size': 4})

    elevs = [ 20, 30]
    azims = [0, 30, 45, 60,75]
    for i, theta1 in enumerate(elevs):
        for j, theta2 in enumerate(azims):
            ax.view_init(elev=theta1,  # 仰角
                         azim=theta2  # 方位角
                         )
            # ax.set_title( f' {args.embedding} Twitter 仰角：{theta1}  方位角：{theta2}')
            path = "./image/YY/3D/News/"+args.embedding + '_' + "N" + str(theta1) + str(theta2)
            plt.savefig(path)  

def tsne(x, y, args):
    # fig=plt.figure(figsize=(5,5))
    tsne = TSNE(n_components=3, init='pca', random_state=0).fit(x)
    X_tsne_3d = tsne.fit_transform(x)

    ##y=float(y)
    #x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
    #embedded = tsne / (x_max - x_min)

    plot_embedding_3d(X_tsne_3d[:,0:3], y, args, "t-SNE 3D")
    # point3d(ax, theta1, theta2)

    # plot_embedding_3d(X_tsne_3d[:, 0:3], y, args, "t-SNE 3D")

def tsne2(x, y, args):   ################2D
    
    tsne2 = TSNE(n_components=2, init='pca', random_state=0).fit(x)
    Y = tsne2.fit_transform(x)

    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(Y, axis=0), np.max(Y, axis=0)
    Y = (Y - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    color = ['lightcoral', 'tan', 'greenyellow', 'burlywood', 'crimson', 'palegreen']

    #plt.scatter(Y[:, 0], Y[:, 1], c= color, cmap= plt.cm.Spectral)
    for i in range(Y.shape[0]):
        if y[i]==0:
            ax.text(Y[i, 0], Y[i, 1], str(y[i]) ,c='cornflowerblue', fontdict={'weight': 'bold', 'size': 4})
        else:
            #print(y[i])
            ax.text(Y[i, 0], Y[i, 1], str(y[i]) ,c=color[y[i]-7], fontdict={'weight': 'bold', 'size': 4})

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    path = "./image/YY/2D/"+args.embedding + '_' + "Twitter_margin"
    plt.savefig(path)


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()
    print('ttttt')
    acc = []
    precious = []
    recall = []

    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))
    num = 0

    for task in sampled_tasks:
        support, query = task
        if num == 0:
            XQ = model['ebd'](query)
            YQ = query['label']
        elif num < 20:
            XQ = torch.cat((XQ, model['ebd'](query)), 0)
            YQ = torch.cat((YQ, query['label']), 0)
            # print(YQ)
        num += 1
    tsne2(XQ.cpu().detach().numpy(), YQ.cpu().detach().numpy(), args)
    print("******plot success*****")
    # print(num)
    # print(YQ.size())

    for task in sampled_tasks:
        a, p, r = test_one(task, model, args)
        acc.append(a)
        precious.append(p)
        recall.append(r)

    acc = np.array(acc)
    precious = np.array(precious)
    recall = np.array(recall)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            colored("acc mean", "blue"),
            np.mean(acc),
            colored("std", "blue"),
            np.std(acc)

        ), flush=True)

    print("precious", np.mean(precious))
    print("recall", np.mean(recall))

    return np.mean(acc), np.std(acc)


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task

    # Embedding the document
    XS = model['ebd'](support)
    YS = support['label']

    XQ = model['ebd'](query)
    YQ = query['label']
    # Apply the classifier
    acc, precious, recall = model['clf'](XS, YS, XQ, YQ)

    return acc, precious, recall
