import numpy as np
import datetime
import os


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


class MyClassifier:
    def __init__(self, x_train, y_train, x_val=None, y_val=None, log_queue=None, task=None):
        self.task = task
        self._x = x_train
        self._y = y_train
        self._n, self._d = x_train.shape
        self._x_val = x_val
        self._y_val = y_val

        self.__log_queue = log_queue

    @property
    def coef_(self):
        return None

    def load_from_disk(self, path):
        pass

    def log_metrics(self, args, prediction_func=None):
        row  = '%s,%s,%s,' % (datetime.datetime.now(), os.getpid(), self.task) + \
               str(self.__compute_metrics(self._x, self._y, prediction_func)) + ',' + \
               str(self.__compute_metrics(self._x_val, self._y_val, prediction_func)) + ',' + \
               ','.join([str(a) for a in args])

        if self.__log_queue is not None:
            self.__log_queue.put(row)
        else:
            print(row)

    def predict(self, x, beta):
        pass

    def predict_proba(self, x, beta):
        pass

    def set_log_queue(self, queue):
        self.__log_queue = queue

    def write_to_disk(self, path):
        pass

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
        fn = np.sum([yh == -1 and yt == -1 for yh, yt in zip(pre, y)])

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
            pre = 0
        else:
            pre = tp/(tp+fp)

        # f1 measure
        if pre == 0 or rec == 0:
            f1 = 0
        else:
            f1  = 2/(pre**-1 + rec**-1)

        return MetricSet(acc=acc, err=err, pre=pre, rec=rec,
                         f1=f1, fpr=fpr, tpr=tpr, specificity=specificity)

