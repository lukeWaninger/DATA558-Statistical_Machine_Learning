from homework.H3_LogisticRegression import MyLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from multiprocessing import Process
import numpy as np
import os
import pandas as pd
import pickle
import statsmodels.api as sm


def datapath(filename):
    return os.getcwd() + '\\kaggle\\data\\' + filename

x_train = np.load(datapath('train_features.npy'))
y_train = np.load(datapath('train_labels.npy'))
x_val = np.load(datapath('val_features.npy'))
y_val = np.load(datapath('val_labels.npy'))

scalar = MinMaxScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_val = scalar.transform(x_val)

# train using a one vs. all approach
y_train_sets, cvs = [], []
# try:
#     data = None
#     with open(datapath('classifiers.pickle'), 'rb') as f:
#         data = pickle.load(f)
#
#     y_train_sets = data['training_sets']
#     cvs = data['classifiers']
# except FileNotFoundError:


def train_one(idx):
    # set class labels
    y = np.copy(y_train)
    y[y != idx] = -1
    y[y == idx] =  1

    # train this class vs the rest
    print("fitting %s vs rest on pid %s" % (i, os.getpid()))
    cv = MyLogisticRegression(X_train=x_train, y_train=y, lamda=.1, eps=0.0001)
    cv = cv.fit(algo='fgrad', init_method='normal')
    cvs.append(cv)

    # pickle them out so I don't have to wait again
    to_dump = {
        'training_sets': y_train_sets,
        'classifiers': cvs
    }
    with open(datapath('classifier_%svr.pickle' % idx), 'wb') as f:
        pickle.dump(to_dump, f, pickle.HIGHEST_PROTOCOL)


[Process(target=train_one, args=(i,)).start() for i in np.unique(y_train)]

# make predictions (train/val) and plot per iteration
ppi = []
for i in range(len(cvs[0]._betas)):
    predictions = np.array([cv.predict_proba(x_train, cv._betas[i]) for cv in cvs])
    ppi.append(np.argmax(predictions, axis=0))

err = []
for i in range(len(ppi)):
    acc = np.sum([1 if yh == yt else 0 for yh, yt in zip(ppi[i], y_train)])/5000
    err.append(1-acc)


