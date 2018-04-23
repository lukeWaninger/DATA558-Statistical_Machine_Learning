from abc import ABC, abstractmethod
import numpy as np
import datetime
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MetricSet:
    def __init__(self, acc=0., err=1., pre=0., rec=0.,
                 f1=0., fpr=1., tpr=0., specificity=0.):
        self.accuracy    = acc
        self.error       = err
        self.precision   = pre
        self.recall      = rec
        self.f1_measure  = f1
        self.fpr         = fpr
        self.tpr         = tpr
        self.specificity = specificity

    def __str__(self):
        return "%s,%s,%s,%s,%s,%s,%s,%s" % \
               (self.accuracy, self.error, self.precision, self.recall,
                self.f1_measure, self.fpr, self.tpr, self.specificity)


class TrainingSplit:
    def __init__(self, n, d, train_idx, val_idx, lamda=0.1, betas=None):
        self.n = n
        self.d = d
        self.train_idx = train_idx
        self.val_idx   = val_idx
        self.lamda     = lamda
        self.betas     = betas if betas is not None else []
        self.train_metrics = None
        self.val_metrics   = None

    def as_dict(self):
        return {
            'train_idx':  self.train_idx,
            'val_idx':  self.val_idx,
            'lamda':  self.lamda,
            'betas':  self.betas,
            'train_metrics': str(self.train_metrics),
            'val_metrics':   str(self.val_metrics)
        }


class MyClassifier(ABC):
    def __init__(self, x_train, y_train, x_val=None, y_val=None,
                 lamda=None, cv_splits=1, log_queue=None, task=None,
                 scale_method=None):
        self.task = task

        self.__x, self.__y, self.__cv_splits = \
            self.__generate_splits(x_train, y_train, x_val, y_val,
                                   cv_splits, lamda, scale_method)

        self.__current_split = -1
        self.__finished = cv_splits-1
        self.__log_queue = log_queue

    @property
    def coef_(self):
        return self.__cv_splits[self.__current_split].betas

    @property
    def _d(self):
        return self.__cv_splits[self.__current_split].d

    @property
    def _n(self):
        return self.__cv_splits[self.__current_split].n

    @property
    def _x(self):
        return self.__x[self.__cv_splits[self.__current_split].train_idx]

    @property
    def _x_val(self):
        return self.__x[self.__cv_splits[self.__current_split].val_idx]

    @property
    def _y(self):
        return self.__y[self.__cv_splits[self.__current_split].train_idx]

    @property
    def _y_val(self):
        return self.__y[self.__cv_splits[self.__current_split].val_idx]

    @property
    def betas(self):
        return self.__cv_splits[self.__current_split].betas

    @property
    def _lamda(self):
        return self.__cv_splits[self.__current_split].lamda

    def _set_betas(self, betas):
        self.__cv_splits[self.__current_split].betas = betas

    def fit(self):
        if self.__current_split == self.__finished:
            return False

        splits, idx = self.__cv_splits, self.__current_split
        splits[idx].train_metrics = self.__compute_metrics(self._x, self._y)
        splits[idx].val_metrics = self.__compute_metrics(self._x_val, self._y_val)

        self.write_to_disk('')

        self.__current_split += 1
        return True

    def load_from_disk(self, path):
        with open('%s%s.pk' % (path, self.task), 'rb') as f:
            data = pickle.load(f)

            self.__cv_splits = data['splits']
            self.__x = data['x']
            self.__y = data['y']

        return self

    def log_metrics(self, args, prediction_func=None):
        row  = '%s,p%s,%s,s%s,' % (datetime.datetime.now(), os.getpid(), self.task, self.__current_split) + \
               str(self.__compute_metrics(self._x, self._y, prediction_func)) + ',' + \
               str(self.__compute_metrics(self._x_val, self._y_val, prediction_func)) + ',' + \
               ','.join([str(a) for a in args])

        if self.__log_queue is not None:
            self.__log_queue.put(row)
        else:
            print(row)

    @abstractmethod
    def predict(self, x, beta):
        pass

    @abstractmethod
    def predict_proba(self, x, beta):
        pass

    def set_log_queue(self, queue):
        self.__log_queue = queue

    def write_to_disk(self, path):
        dict_rep = {
            'task':   self.task,
            'splits': [split.as_dict() for split in self.__cv_splits],
            'x':      self._x,
            'y':      self._y
        }

        with open('%s%s.pk' % (path, self.task), 'wb') as f:
            pickle.dump(dict_rep, f, pickle.HIGHEST_PROTOCOL)

    def __compute_metrics(self, x, y, prediction_func=None):
        if x is None and y is None:
            return MetricSet()

        if prediction_func is None:
            pre = self.predict(x, self.coef_)
        else:
            pre = prediction_func(x, self.coef_)

        num_classes = len(np.unique(y))
        if num_classes > 2:
            acc = (np.sum([yh == yt for yh, yt in zip(pre, y)]) / len(y))[0]
            return MetricSet(acc=acc, err=1 - acc)

        p  = np.sum(y ==  1)
        n  = np.sum(y == -1)
        tp = np.sum([yh ==  1 and yt ==  1 for yh, yt in zip(pre, y)])
        tn = np.sum([yh == -1 and yt == -1 for yh, yt in zip(pre, y)])
        fp = np.sum([yh ==  1 and yt == -1 for yh, yt in zip(pre, y)])

        # accuracy, error, recall, tpr, fpr
        if p == 0 or n == 0:
            acc = rec = tpr = fpr = 0
        else:
            acc = (tp+tn)/(p+n)
            rec = tp / p
            tpr = rec
            fpr = fp / n
        err = 1-acc
        specificity = 1-fpr

        # precision
        if tp == 0 or fp == 0:
            prec = 0
        else:
            prec = tp/(tp+fp)

        # f1 measure
        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1  = 2/(prec**-1 + rec**-1)

        return MetricSet(acc=acc, err=err, pre=prec, rec=rec,
                         f1=f1, fpr=fpr, tpr=tpr, specificity=specificity)

    def __generate_splits(self, x_train, y_train, x_val, y_val,
                          cv_splits, lamda, scale_method):
        x = np.concatenate((x_train, x_val))
        x = self.__scale_data(x, scale_method)

        y = np.concatenate((y_train, y_val))
        splits = []

        if cv_splits > 1:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(x, y):
                splits.append(
                    TrainingSplit(
                        n=len(train_idx),
                        d=x_train.shape[1],
                        train_idx=train_idx,
                        val_idx=val_idx,
                        lamda=lamda
                    ))
        else:
            splits.append(
                TrainingSplit(
                    n=x_train.shape[0],
                    d=x_train.shape[1],
                    train_idx=[i for i in range(len(y_train))],
                    val_idx=[i for i in range(len(y_train), len(y_train)+len(y_val))]
                )
            )

        return x, y, splits

    @staticmethod
    def __scale_data(data, method):
        if method == 'standardize':
            return StandardScaler().fit_transform(data)
        elif method == 'minmax':
            return MinMaxScaler().fit_transform(data)
        else:
            return data
