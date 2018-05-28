import numpy as np


class MetricSet(object):
    def __init__(self, y_true=None, y_hat=None, data=None):
        if data is not None:
            self.__from_dict(data)

        elif y_true is not None and y_hat is not None:
            self.__compute_metrics(y_true, y_hat)

        else:
            self.accuracy = 0
            self.error = 1
            self.precision = 0
            self.recall = 0
            self.f1_measure = 0
            self.fpr = 0
            self.tpr = 0
            self.specificity = 0

    def __str__(self):
        return ','.join([str(round(val, 4)) for key, val in self.dict.items()])

    @property
    def dict(self):
        return {
            'accuracy': self.accuracy,
            'error': self.error,
            'precision': self.precision,
            'recall': self.recall,
            'f1_measure': self.f1_measure,
            'fpr': self.fpr,
            'tpr': self.tpr,
            'specificity': self.specificity
        }

    def __compute_metrics(self, yt, yh):
        p = np.sum(yt == 1)
        n = np.sum(yt == -1)
        tp = np.sum([yh == 1  and yt == 1  for yh, yt in zip(yh, yt)])
        tn = np.sum([yh == -1 and yt == -1 for yh, yt in zip(yh, yt)])
        fp = np.sum([yh == 1  and yt == -1 for yh, yt in zip(yh, yt)])

        # accuracy, error, recall, tpr, fpr
        if p == 0 or n == 0:
            acc = rec = tpr = fpr = 0
        else:
            acc = (tp + tn) / (p + n)
            rec = tp / p
            tpr = rec
            fpr = fp / n
        err = 1 - acc
        specificity = 1 - fpr

        # precision
        if tp == 0 or fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)

        # f1 measure
        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1 = 2 / (prec ** -1 + rec ** -1)

        self.accuracy = acc
        self.error = err
        self.precision = prec
        self.recall = rec
        self.f1_measure = f1
        self.fpr = fpr
        self.tpr = tpr
        self.specificity = specificity

        return self

    def __from_dict(self, data):
        if data == 'none':
            return self

        self.accuracy = data['accuracy']
        self.error = data['error']
        self.precision = data['precision']
        self.recall = data['recall']
        self.f1_measure = data['f1_measure']
        self.fpr = data['fpr']
        self.tpr = data['tpr']
        self.specificity = data['specificity']

        return self

