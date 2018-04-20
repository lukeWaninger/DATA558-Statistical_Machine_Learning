from DATA558_StatisticalML.kaggle.mlogreg import MyLogisticRegression
from DATA558_StatisticalML.kaggle import models
from DATA558_StatisticalML.kaggle import settings
from multiprocessing import Process, Queue
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import statsmodels.api as sm
import time


Base = declarative_base()


class Context:
    def __init__(self):
        self.__engine = create_engine(URL(**settings.DATABASE))
        self.Session = sessionmaker()
        self.Session.configure(bind=self.__engine)
        models.Base.metadata.create_all(self.__engine)


def datapath(filename):
    return os.getcwd() + '/data/' + filename


x_train = np.load(datapath('train_features.npy'))
y_train = np.load(datapath('train_labels.npy'))
x_val = np.load(datapath('val_features.npy'))
y_val = np.load(datapath('val_labels.npy'))

scalar = MinMaxScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_val = scalar.transform(x_val)


def train_one(idx, log_queue):
    # set class labels
    y = np.copy(y_train)
    y[y != idx] = -1
    y[y == idx] =  1

    # train this class vs the rest
    print("fitting %s vs rest on pid %s" % (idx, os.getpid()))
    cv = MyLogisticRegression(X_train=x_train, y_train=y, lamda=.05, eps=0.001, idx=idx, log_queue=log_queue)
    cv = cv.fit(algo='fgrad', init_method='zeros')

    # pickle them out
    with open(datapath('classifier_%svr.pickle' % idx), 'wb') as f:
        pickle.dump(cv, f, pickle.HIGHEST_PROTOCOL)

    print("%s finished %s vs rest" % (os.getpid(), idx))


qu = Queue()
processes = [Process(target=train_one, args=(i, qu)) for i in np.unique(y_train)]

for process in processes:
    time.sleep(4)
    process.start()


def child_running():
    for proc in processes:
        if proc.is_alive():
            return True
    return False


context = Context()
session = context.Session()

while True:
    m = qu.get()

    if isinstance(m, models.LogMessage):
        session.add(m)
        session.commit()

    if not child_running() and qu.empty():
        break
