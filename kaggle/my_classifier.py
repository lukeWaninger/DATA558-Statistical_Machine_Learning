from abc import ABC, abstractmethod
import numpy as np
import datetime
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
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

    def as_dict(self):
        return {
            'accuracy':    self.accuracy,
            'error':       self.error,
            'precision':   self.precision,
            'recall':      self.recall,
            'f1_measure':  self.f1_measure,
            'fpr':         self.fpr,
            'tpr':         self.tpr,
            'specificity': self.specificity
        }

    def from_dict(self, data):
        self.accuracy    = data['accuracy']
        self.error       = data['error']
        self.precision   = data['precision']
        self.recall      = data['recall']
        self.f1_measure  = data['f1_measure']
        self.fpr         = data['fpr']
        self.tpr         = data['tpr']
        self.specificity = data['specificity']

        return self


class TrainingSplit:
    def __init__(self, n=0, d=0, train_idx=None, val_idx=None, parameters=None, betas=None):
        self.n = n
        self.d = d
        self.train_idx  = train_idx
        self.val_idx    = val_idx
        self.parameters = parameters if parameters is not None else {}
        self.betas      = betas if betas is not None else []
        self.train_metrics = None
        self.val_metrics   = None

    def as_dict(self):
        return {
            'n': self.n,
            'd': self.d,
            'train_idx':   self.train_idx if not None else [],
            'val_idx':     self.val_idx if not None else [],
            'parameters':  self.parameters,
            'betas': self.betas,
            'train_metrics': self.train_metrics.as_dict() if self.train_metrics is not None else 'none',
            'val_metrics':   self.val_metrics.as_dict() if self.val_metrics is not None else 'none'
        }

    def from_dict(self, data):
        self.betas = data['betas']
        self.d = data['d']
        self.train_metrics = MetricSet().from_dict(data['train_metrics'])
        self.val_metrics   = MetricSet().from_dict(data['val_metrics'])
        self.n = data['n']
        self.parameters = data['parameters']
        self.train_idx  = data['train_idx']
        self.val_idx    = data['val_idx']

        return self


class MyClassifier(ABC):
    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None,
                 preprocessor=None, task=None, log_path=None, log_queue=None):
        self.task = task

        self.__parameters = parameters
        self.__x, self.__y, self.__cv_splits = \
            self.__generate_splits(x_train, y_train, x_val, y_val)

        self.__current_split = -1

        try:
            self.__log_path = parameters['log_path']
        except:
            self.__log_path = None

        try:
            self.__log_queue = log_queue
        except:
            self.__log_queue = None

    @property
    def coef_(self):
        return self.__cv_splits[self.__current_split].betas

    # protected attributes
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

    # protected methods
    @abstractmethod
    def _simon_says_fit(self):
        pass

    def _param(self, parameter):
        return self.__cv_splits[self.__current_split].parameters[parameter]

    def _set_betas(self, betas):
        self.__cv_splits[self.__current_split].betas = betas

    # public methods
    def fit(self):
        if self.__current_split == len(self.__cv_splits)-1:
            return False

        if self.__log_path is not None:
            self.write_to_disk(self.__log_path)

        self.__current_split += 1
        return True

    def load_from_disk(self, path):
        self.__cv_splits = None
        self.__x = None
        self.__y = None

        try:
            with open('%s%s.pk' % (path, self.task), 'rb') as f:
                data = pickle.load(f)

                self.task = data['task']
                self.__cv_splits = [TrainingSplit().from_dict(d) for d in data['splits']]
                self.__x = data['x']
                self.__y = data['y']

                return self
        except Exception as e:
            print('could not load model from disk: %s' % ', '.join([str(a) for a in e.args]))

    def log_metrics(self, args, prediction_func=None, include='all'):
        split = self.__cv_splits[self.__current_split]
        pstr = ','.join([v for k, v in split.parameters])

        row = '%s,p%s,%s,%s,' % \
              (datetime.datetime.now(), os.getpid(), self.task, self.__current_split) + pstr

        if include in ['all', 'reduced']:
            train_metrics = self.__compute_metrics(self._x, self._y, prediction_func)
            val_metrics   = self.__compute_metrics(self._x_val, self._y_val, prediction_func)

            split.train_metrics = train_metrics
            split.val_metrics = val_metrics

            if include == 'all':
                row += '%s,%s,%s' % (str(train_metrics), str(val_metrics), ','.join([str(a) for a in args]))
            else:
                row += '%s,%s,%s' % (train_metrics.error, val_metrics.error, ','.join([str(a) for a in args]))

        elif include == 'minimal':
            row += ','.join([str(a) for a in args])

        else:
            return

        print(row)
        if self.__log_queue is not None:
            self.__log_queue.put(row)
        else:
            if self.__log_path is not None:
                with open('%s%s.csv' % (self.__log_path, self.task), 'a+') as f:
                    f.writelines(row + '\n')

    @abstractmethod
    def predict(self, x, beta=None):
        pass

    @abstractmethod
    def predict_proba(self, x, beta=None):
        pass

    def predict_with_best_fold(self, x, metric='accuracy', beta=None):
        splits = self.__cv_splits
        idx = np.argmax([s.val_metrics.as_dict()[metric] for s in splits])[0]

        best_split = TrainingSplit(
            n=self.__x.shape[0],
            d=self.__x.shape[1],
            train_idx=np.arange(self.__x.shape[0]),
            parameters=self.__cv_splits[idx]
        )

        self.__cv_splits.append(best_split)

        print('training with all features ' + str(best_split.parameters))
        self._simon_says_fit()

        return self.predict(x, beta)

    def set_log_queue(self, queue):
        self.__log_queue = queue

    def set_split(self, split_number):
        if not 0 < split_number < len(self.__cv_splits):
            print('split not calculated')
        else:
            self.__current_split = split_number

        return self

    def write_to_disk(self, path=None):
        try:
            dict_rep = {
                'task':   self.task,
                'splits': [split.as_dict() for split in self.__cv_splits],
                'x':      self._x,
                'y':      self._y
            }

            with open('%s%s.pk' % (path, self.task), 'wb') as f:
                pickle.dump(dict_rep, f, pickle.HIGHEST_PROTOCOL)

            print('%s [%s] written to disk' % (self.task, self.__parameters))
        except Exception as e:
            print('could not write %s to disk: %s' % (self.task, [str(a) for a in e.args]))

    # private methods
    def __compute_metrics(self, x, y, prediction_func=None):
        if x is None and y is None:
            return MetricSet()

        if prediction_func is None:
            pre = self.predict(x)
        else:
            pre = prediction_func(x)

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

    def __generate_splits(self, x_train, y_train, x_val, y_val):
        if x_val is not None:
            x = np.concatenate((x_train, x_val))
        else:
            x = x_train

        if y_val is not None:
            y = np.concatenate((y_train, y_val))
        else:
            y = y_train

        splits, idx_set = [], {}
        cv_splits = 1

        for key, value in self.__parameters.items():
            cv_splits *= len(value)
            idx_set[key] = 0

        def set_parameter_idx():
            for k, v in self.__parameters.items():
                if idx_set[k] == len(v) - 1:
                    idx_set[k] = 0
                else:
                    idx_set[k] += 1
                    break

        # create the splits
        if cv_splits > 1:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

            first = True
            for train_idx, val_idx in kf.split(x, y):
                if not first:
                    set_parameter_idx()
                else:
                    first = False

                parameters = {}
                for key, value in self.__parameters.items():
                    parameters[key] = value[idx_set[key]]

                splits.append(
                    TrainingSplit(
                        n=len(train_idx),
                        d=x_train.shape[1],
                        train_idx=train_idx,
                        val_idx=val_idx,
                        parameters=parameters
                    )
                )
        else:
            parameters = {}
            for key, value in self.__parameters:
                parameters[key] = value[0]

            splits.append(
                TrainingSplit(
                    n=x_train.shape[0],
                    d=x_train.shape[1],
                    train_idx=[i for i in range(len(y_train))],
                    val_idx=[i for i in range(len(y_train), len(y_train)+len(y_val))],
                    parameters=parameters
                )
            )

        return x, y, splits
