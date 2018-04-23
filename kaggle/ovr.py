import datetime
from mlogreg import MyLogisticRegression
from my_classifier import MyClassifier
import multiprocessing
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import os
import pandas as pd
import time


class MultiClassifier(MyClassifier):
    def __init__(self, x_train, y_train, x_val=None, y_val=None, method='ovr', max_iter=500,
                 lamda=None, eps=0.001, init_method='zeros', scale_method='minmax', n_jobs=-1):

        super().__init__(x_train, y_train, x_val, y_val, task='%s multiclass' % method)
        self.eps = eps
        self.init_method = init_method
        self.lamda = lamda
        self.max_iter = max_iter

        cpu_count = multiprocessing.cpu_count()
        if n_jobs == -1 or n_jobs >= cpu_count:
            self.__available_procs = cpu_count
        else:
            self.__available_procs = n_jobs

        self.__manager = multiprocessing.Manager()
        self.__log_queue = self.__manager.Queue()
        self.__snd, self.__rcv = multiprocessing.Pipe()

        self.__scalar = None
        self._x = self.__scale_data(x_train, scale_method)
        self._y = y_train

        if x_val is not None:
            self._x_val = self.__scale_data(x_val, scale_method)

        self.__method = method

        if not self.__load_classifiers():
            self.cvs = self.__build_classifiers()

    def fit(self):
        log_manager = multiprocessing.Process(target=self.__log_manager,
                                              args=(self.__log_queue,
                                                    self.__snd))
        log_manager.start()

        workers = []
        for cv in self.cvs:
            cv.set_log_queue(self.__log_queue)

            if self.__available_procs == 0:
                self.__rcv.recv()
                self.__available_procs += 1

            worker = multiprocessing.Process(target=self.__train_one,
                                             args=(cv, self.__snd))
            workers.append(worker)
            worker.start()
            time.sleep(3)

            self.__available_procs -= 1

        print('all classifiers queued for processing..\n')

        for worker in workers:
            worker.join()

        log_manager.terminate()

    def output_predictions(self, x):
        predictions = self.predict(x)
        pd.DataFrame(predictions).to_csv('kaggle_predictions.csv')

    def predict(self, x, beta=None):
        if self.__method == 'ovr':
            predictions = []
            for cv in self.cvs:
                predictions.append(cv.predict_proba(x))

            predictions = np.array(predictions).T
            predictions = [np.argmax(p) for p in predictions]
            return predictions

        elif self.__method in ['all_pairs', 'both']:
            predictions = []
            for cv in self.cvs:
                if 'rest' in cv.task:
                    continue

                pre = cv.predict(x)
                pos, drop, neg = cv.task.split(' ')
                predictions.append([int(pos) if pi == 1 else int(neg) for pi in pre])
            predictions = [mode(pi).mode for pi in np.array(predictions).T]

            # break ties at random unless ovr classifiers have been fitted
            ties = [(i, pi) for i, pi in enumerate(predictions) if len(pi) > 1]
            if len(ties) > 0:
                break_at_random = not self.__load_classifiers(method='ovr')
                if not break_at_random:
                    breaker_idx = np.unique(np.concatenate(np.array([pi for i, pi in ties])))
                    breakers = {}
                    tasks = [(i, '%s vs rest' % i) for i in breaker_idx]

                    for i, task in tasks:
                        c = [c for c in self.cvs if c.task == task][0]
                        breakers[str(i)] = c

                    if len(breakers) == 0:
                        break_at_random = True

                    else:
                        for idx, tie in ties:
                            probas = dict([(c.task.split()[0], c.predict_proba([x[idx]]))
                                           for c in [breakers[str(i)] for i in tie]])
                            predictions[idx] = list(probas.keys())[np.argmin(probas)[0]]

                if break_at_random:
                    for idx, pi in ties: predictions[idx] = np.random.choice(pi)

            predictions = [int(i) for i in predictions]
            return predictions

        else:
            raise Exception('classification method not found')

    def __build_classifiers(self, method=None):
        class_sets = self.__get_training_sets(method)

        cvs, x_idx, x_idx_v, y_v = [], [], [], []
        for pos, neg in class_sets:
            # set class labels
            if neg == 'rest':
                pos_idx = np.where(self._y == int(pos))
                y = self._y**0*-1
                y[pos_idx] = 1
                x_idx = [i for i in range(len(y))]

                if self._x_val is not None and self._y_val is not None:
                    v_pos_idx = np.where(self._y_val == int(pos))
                    y_v = self._y_val**0*-1
                    y_v[v_pos_idx] = 1
                    x_idx_v = [i for i in range(len(y_v))]

            else:
                y = np.copy(self._y)

                pos_idx = np.where(y == int(pos))
                neg_idx = np.where(y == int(neg))
                y = y**0*-1
                y[pos_idx] = 1

                x_idx = np.concatenate((pos_idx[0], neg_idx[0]), axis=0)
                y = y[x_idx]

                if self._x_val is not None and self._y_val is not None:
                    y_v = np.copy(self._y_val)
                    v_pos_idx = np.where(y_v == int(pos))
                    v_neg_idx = np.where(y_v == int(neg))

                    y_v = y_v**0*-1
                    y_v[v_pos_idx] = 1

                    x_idx_v = np.concatenate((v_pos_idx[0], v_neg_idx[0]), axis=0).flatten()
                    y_v = y_v[x_idx_v]

            if self.lamda is None:
                lamda = self.__find_best_lamda(self._x[x_idx], y)
            else:
                lamda = self.lamda

            classifier = MyLogisticRegression(x_train=self._x[x_idx], y_train=y,
                                              x_val=self._x_val[x_idx_v], y_val=y_v,
                                              lamda=lamda, max_iter=self.max_iter,
                                              eps=self.eps, task='%s vs %s' % (pos, neg))
            cvs.append(classifier)
        return cvs

    def __find_best_lamda(self, x, y):
        cv = LogisticRegression(fit_intercept=False, max_iter=5000)

        parameters = {'C': np.linspace(.001, 2.0, 20)}
        gs = GridSearchCV(cv, parameters, scoring='neg_log_loss', n_jobs=self.__available_procs).fit(x, y)

        return gs.best_estimator_.C

    def __get_training_sets(self, method=None):
        classes = [str(c) for c in np.unique(self._y)]
        pairs = []

        if method is not None:
            m = method
        else:
            m = self.__method

        if m in ['ovr', 'both']:
            [pairs.append((c, 'rest')) for c in classes]

        if m in ['all_pairs', 'both']:
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    pairs.append((classes[i], classes[j]))

        return pairs

    def __load_classifiers(self, path='', method=None):
        try:
            if not hasattr(self, 'cvs'):
                self.cvs = []

            tasks = ['%s vs %s' % (a, b) for a,  b in self.__get_training_sets(method)]
            for task in tasks:
                cv = MyLogisticRegression(self._x, self._y,
                                          self._x_val, self._y_val,
                                          task=task)
                cv = cv.load_from_disk(path)
                self.cvs.append(cv)
            return True
        except Exception as e:
            print(e.args)
            return False

    def __log_manager(self, log_queue, conn):
        start = time.time()

        path = '/mnt/hgfs/descent_logs/descent_log_%s_%s.csv' % (self.task, str(int(time.time())))
        header = 'timestamp,pid,task,test_acc,test_err,test_precision,test_recall,test_f1,' \
                 'test_fpr,test_tpr,test_specificity,val_acc,val_err,val_precision,' \
                 'val_recall,val_f1,val_fpr,val_tpr,val_specificity\n'

        with open(path, 'w+') as f:
            f.writelines(header)

        running = len(self.cvs)
        while True:
            message = log_queue.get()

            if message == 'END_FLAG':
                running -= 1

            else:
                print(message)
                try:
                    with open(path, 'a+') as f:
                        f.writelines(message + "\n")
                except Exception as e:
                    print(e.args)

            if running == 0:
                break

        time_delta = time.time()-start
        print('training completed in %s\n' % time_delta)
        conn.send(True)

    def __scale_data(self, data, method):
        if self.__scalar is None:
            if method == 'standardize':
                self.__scalar = StandardScaler().fit(data)
            else:
                self.__scalar = MinMaxScaler().fit(data)

        return self.__scalar.transform(data)

    def __train_one(self, cv, conn):
        start = time.time()

        print("starting %s on pid %s" % (cv.task, os.getpid()))
        cv = cv.fit(algo='fgrad', init_method=self.init_method)
        cv.write_to_disk('')

        print('%s finished %s in %s seconds' %
              (os.getpid(), cv.task, time.time() - start))

        self.__log_queue.put('END_FLAG')
        conn.send(True)


try:
    x_train = np.load('kaggle/data/train_features.npy')
    y_train = np.load('kaggle/data/train_labels.npy')
    x_val = np.load('kaggle/data/val_features.npy')
    y_val = np.load('kaggle/data/val_labels.npy')
    x_test = np.load('kaggle/data/test_features.npy')
except:
    x_train = np.load('data/train_features.npy')
    y_train = np.load('data/train_labels.npy')
    x_val = np.load('data/val_features.npy')
    y_val = np.load('data/val_labels.npy')
    x_test = np.load('data/test_features.npy')

x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))

i, n_splits = 1, 7
kf = KFold(n_splits=n_splits)
for train_idx, test_idx in kf.split(x):
    print("\n\nROUND %s of %s: %s \n\n" % (i, n_splits, datetime.datetime.now()))
    # MultiClassifier(x[train_idx], y[train_idx],
    #                 x[test_idx],  y[test_idx],
    #                 eps=0.001, n_jobs=-1, lamda=0.001,
    #                 max_iter=5000, method='ovr').fit()
    MultiClassifier(x[train_idx], y[train_idx],
                    x[test_idx], y[test_idx],
                    eps=0.001, n_jobs=-1, lamda=0.001,
                    max_iter=5000, method='all_pairs').fit()


