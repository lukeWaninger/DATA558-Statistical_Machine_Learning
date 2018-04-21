from kaggle.mlogreg import MyLogisticRegression
import kaggle.models as models
from multiprocessing import Manager, Queue, Process
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import statsmodels.api as sm
import time


class OVR:
    def __init__(self):
        self.__log_queue = Queue()
        self.x_train = np.load('kaggle/data/train_features.npy')
        self.y_train = np.load('kaggle/data/train_labels.npy')
        self.x_val = np.load('kaggle/data/val_features.npy')
        self.y_val = np.load('kaggle/data/val_labels.npy')

        scalar = MinMaxScaler().fit(self.x_train)
        self.x_train = scalar.transform(self.x_train)
        self.x_val = scalar.transform(self.x_val)

        self.manager = Manager()
        self.qu = self.manager.Queue()
        self.cvs = []
        self.dumps = []

        self.best_lamdas = []

    def train_one(self, idx, log_queue):
        # set class labels
        y = np.copy(self.y_train)
        y[y != idx] = -1
        y[y == idx] =  1

        # train this class vs the rest
        print("fitting %s vs rest on pid %s" % (idx, os.getpid()))
        cv = MyLogisticRegression(X_train=self.x_train, y_train=y, lamda=self.best_lamdas[idx][1], eps=0.001, idx=idx, log_queue=log_queue)
        cv = cv.fit(algo='fgrad', init_method='zeros')
        self.cvs.append(cv)

        to_dump = {
            'class':    cv.pos_class_,
            'betas':    cv.betas[-1],
            'eps':      cv.eps,
            'lambda':   cv.lamda,
            'max_iter': cv.max_iter,
        }

        self.dumps.append(to_dump)

        # pickle them out
        # with open(('kaggle/data/ovrc_%s.pickle' % idx), 'wb') as f:
        #   pickle.dump(to_dump, f, pickle.HIGHEST_PROTOCOL)

        #log_queue.put("%s finished %s vs rest" % (os.getpid(), idx))

    def train(self):
        workers = [Process(target=self.train_one, args=(i, self.qu)) for i in np.unique(self.y_train)]

        for worker in workers:
            worker.start()
            time.sleep(3)

        # running = len(np.unique(self.y_train))
        # while True:
        #     m = self.qu.get()
        #     if isinstance(m, models.LogMessage):
        #         print(str(m))
        #         with open('/mnt/hgfs/descent_logs/descent_log.csv', 'a+') as f:
        #             f.writelines(str(m) + "\n")
        #     else:
        #         print(m)
        #
        #     for worker in workers:
        #         if not worker.is_alive():
        #             worker.terminate()
        #             worker.join()
        #             running -= 1
        #
        #     if running == 0:
        #         break

    def best_lambdas_per_class(self):
        for idx in np.unique(self.y_train):
            y = np.copy(self.y_train)
            y[y != idx] = -1
            y[y == idx] = 1

            cv = LogisticRegression(fit_intercept=False, max_iter=5000)
            parameters = {'C': np.linspace(.001, 2.0, 20)}
            gs = GridSearchCV(cv, parameters, scoring='neg_log_loss', n_jobs=-1).fit(self.x_train, y)
            self.best_lamdas.append((idx, gs.best_estimator_.C))

    def load_classifiers(self):
        cvs = []
        for i in np.unique(self.y_train):
            with open(('kaggle/data/ovrc_%s.pickle' % i), 'wb') as f:
                data = pickle.load(f)

            cv = MyLogisticRegression(self.x_train,
                                      self.y_train,
                                      lamda=data['lambda'],
                                      eps=data['eps'],
                                      max_iter=data['max_iter'],
                                      idx=data['class'])
            cv.betas = data['betas']
            cvs.append(cv)

        self.cvs = cvs

    def predict(self):
        predictions = []
        for i, cv in enumerate(self.cvs):
            predictions.append(cv.predict_proba(self.x_val))

        return np.array(predictions)


ovr = OVR()
ovr.best_lambdas_per_class()
#ovr.train()

for i in range(10):
    idx = i
    y = np.copy(ovr.y_train)
    y[y != idx] = -1
    y[y == idx] = 1
    cv = MyLogisticRegression(X_train=ovr.x_train, y_train=y, lamda=.1, eps=0.001, idx=idx)
    cv.fit()
    ovr.cvs.append(cv)

predictions = []

x_test = np.load('kaggle/data/test_features.npy')
for cv in ovr.cvs:
    pre = cv.predict_proba(x_test)
    predictions.append(pre)



