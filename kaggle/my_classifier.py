from abc import ABC, abstractmethod
import numpy as np
import datetime
import os
import pickle
from sklearn.model_selection import KFold


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
        return ','.join([str(round(val, 4)) for key, val in self.dict().items()])

    def dict(self):
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

    def dict(self):
        return {
            'n': self.n,
            'd': self.d,
            'train_idx':   self.train_idx if not None else [],
            'val_idx':     self.val_idx if not None else [],
            'parameters':  self.parameters,
            'betas': self.betas,
            'train_metrics': self.train_metrics.dict() if self.train_metrics is not None else 'none',
            'val_metrics':   self.val_metrics.dict() if self.val_metrics is not None else 'none'
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
                 task=None, log_queue=None, log_path='', logging_level='none', dict_rep=None):
        if dict_rep is not None:
            self.__load_from_dict(dict_rep)
        else:
            self.task = task

            self.__parameters = parameters
            self.__x, self.__y, self.__cv_splits = self.__generate_splits(x_train, y_train, x_val, y_val)

        self.__current_split = -1

        self.__logging_level = logging_level
        self.__log_path = log_path
        self.__log_queue = log_queue

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

    def _set_param(self, param, value):
        self.__cv_splits[self.__current_split].parameters[param] = value

    # public methods
    def dict(self):
        return {
            'task': self.task,
            'x': self.__x,
            'y': self.__y,
            'parameters': self.__parameters,
            'splits': [s.dict() for s in self.__cv_splits]
        }

    def __load_from_dict(self, dict_rep):
        self.task = dict_rep['task']
        self.__x = dict_rep['x']
        self.__y = dict_rep['y']
        self.__parameters = dict_rep['parameters']
        self.__cv_splits = [TrainingSplit().from_dict(d) for d in dict_rep['splits']]

    def fit(self):
        if self.__current_split == len(self.__cv_splits)-1:
            return False

        # if self.__log_path is not None:
        #     self.write_to_disk(self.__log_path)

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

    def log_metrics(self, args, prediction_func=None):
        if self.__logging_level == 'none':
            return

        pstr = ','.join([str(v) for k, v in self.__cv_splits[self.__current_split].parameters.items()])
        arg_str = ','.join(['%.7f' % a if not float(a).is_integer() else str(a) for a in args])

        row = '%s,p%s,%s,%s,' % \
              (datetime.datetime.now(), os.getpid(), self.task, self.__current_split) + pstr

        if self.__logging_level in ['all', 'reduced']:
            train_metrics = self.__compute_metrics(self._x, self._y, prediction_func)
            val_metrics   = self.__compute_metrics(self._x_val, self._y_val, prediction_func)

            self.__cv_splits[self.__current_split].train_metrics = train_metrics
            self.__cv_splits[self.__current_split].val_metrics = val_metrics

            if self.__logging_level == 'all':
                row += ',%s,%s,%s' % (str(train_metrics), str(val_metrics), arg_str)
            else:
                row += ',%.4f,%.4f,%s' % (round(train_metrics.error, 4), round(val_metrics.error, 4), arg_str)

        elif self.__logging_level == 'minimal':
            row += arg_str

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

    def predict_with_best_fold(self, x, metric='error', beta=None):
        splits = self.__cv_splits
        idx = np.argmin([s.val_metrics.dict()[metric] for s in splits])

        best_split = TrainingSplit(
            n=self.__x.shape[0],
            d=self.__x.shape[1],
            train_idx=np.arange(self.__x.shape[0]),
            parameters=splits[idx].parameters
        )

        self.__cv_splits.append(best_split)

        print('training with all features ' + str(best_split.parameters))
        self.__current_split = len(self.__cv_splits)-2
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
                'splits': [split.dict() for split in self.__cv_splits],
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

        # calculate how many splits we need to create
        for key, value in self.__parameters.items():
            cv_splits *= len(value)
            idx_set[key] = 0

        # adjust which parameter we need for which split
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
