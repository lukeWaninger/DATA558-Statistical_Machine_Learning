from abc import ABC, abstractmethod
import datetime
import numpy as np
import pickle
from sklearn.model_selection import KFold
import threading
import time


class MyClassifier(ABC):
    """my classifier base class"""

    def __init__(self, x_train, y_train, parameters, *args, **kwargs):
        """

        Args:
            x_train (ndarray): nXd array of input samples
            y_train (ndarray: nX1 array of training labels
            parameters (dictionary): child class parameters
            *args:
            **kwargs:
        """
        # load this classifier from the dict representation if provided
        if 'dict_rep' in args[1].keys():
            self.__load_from_dict(args[1]['dict_rep'])

        # otherwise begin loading parameters
        else:
            if 'x_val' in args[1].keys() and 'y_val' in args[1].keys():
                x_val = args[1]['x_val']
                y_val = args[1]['y_val']
            else:
                x_val, y_val = None, None

            self.task = args[1]['task']
            self.__parameters = parameters
            self.__x, self.__y, self.__cv_splits = self.__generate_splits(x_train, y_train, x_val, y_val)

        # setup cross-validation and logging features
        self.__thread_split_map = {}
        self.__logging_level = args[1]['logging_level'] if 'logging_level' in args[1].keys() else 'none'
        self.__log_path = args[1]['log_path'] if 'log_path' in args[1].keys() else ''
        self.__write_to_disk = args[1]['write_to_disk'] if 'write_to_disk' in args[1].keys() else False
        self.__log_queue = args[1]['loq_queue'] if 'loq_queue' in args[1].keys() else None

    @abstractmethod
    def _compute_grad(self, beta):
        yield

    @abstractmethod
    def _objective(self, beta):
        yield

    @abstractmethod
    def predict(self, x, beta=None):
        yield

    @abstractmethod
    def predict_proba(self, x, beta=None):
        yield

    def fit(self):
        """fit the classifier

        proceeds through each cross validation split, training each using the
        algorithm defined by the 'algo' key provided in parameters

        Returns
            trained classifier
        """
        for i in range(len(self.__cv_splits)):
            # setup running thread
            thread = threading.Thread(target=self.fit_one)
            thread_name = f'split_{str(i)}'
            thread.setName(thread_name)

            self.__thread_split_map[thread_name] = i
            thread.start()

        while len(self.__thread_split_map.keys()) != 0:
            time.sleep(1)

        # write self to disk
        if self.__log_path is not None and self.__write_to_disk:
            self.write_to_disk(self.__log_path)

        return self

    def fit_one(self):
        """fit a single training fold"""
        self._set_betas(np.zeros(self._d))

        algo = self.__cv_splits[self.__current_split].parameters['algo']
        if algo == 'grad':
            self.__grad_descent()
        elif algo == 'fgrad':
            self.__fast_grad_descent()
        elif algo == 'random_cd':
            raise NotImplemented('random coordinate descent not implemented')
        elif algo == 'cyclic_cd':
            raise NotImplemented('cyclic coordinate descent not implemented')

        thread_name = threading.currentThread().getName()
        del self.__thread_split_map[thread_name]

    def predict_with_best_fold(self, x, metric='error', proba=False):
        """predict the given labels

        uses the provided metric to determine which fold to extract parameters.
        once identified, a new fold is generated and trained using those parameters.

        Args
            x (ndarray): nXd array of samples to predict
            metric (str) - optional: metric to use for finding the optimal

        Returns
            [({-1, 1})]: list of predictions

        Raises
            ValueError: if provided metric is not defined
        """
        splits = self.__cv_splits
        pretrained = False
        func = np.argmin
        metrics = []

        # determine the best metric to use
        if metric in ['fpr', 'error']:
            func = np.argmin
        elif metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1_measure', 'tpr']
            func = np.argmax
        else:
            raise ValueError('provided metric is not defined')

        # get all those metrics, unless one without validation metrics
        # is found. In that case, the best params have already been trained
        for s in splits:
            if s.val_metrics is None:
                pretrained = True
                break
            else:
                metrics.append(s.val_metrics.dict[metric])

        # if the best params haven't been trained, train
        if not pretrained:
            idx = func(metrics)

            # create a new split with parameters from the best
            new_split = TrainingSplit(
                n=self.__x.shape[0],
                d=self.__x.shape[1],
                train_idx=np.arange(self.__x.shape[0]),
                parameters=splits[idx].parameters
            )
            self.__cv_splits.append(new_split)

            # train with the identified parameter set
            print('training with all features %s: %s' % (self.task, str(splits[-1].parameters)))

            thread_name = threading.current_thread().getName()
            self.__thread_split_map[thread_name] = len(self.__cv_splits) - 1

            self._set_param('eta', 1.)
            self.fit_one()
        else:
            pass

        # set the correct thread for prediction
        thread_name = threading.current_thread().getName()
        self.__thread_split_map[thread_name] = len(self.__cv_splits) - 1

        # return labels or probabilities
        return self.predict(x) if not proba else self.predict_proba(x)

    def __compute_metrics(self, x, y, prediction_func=None):
        """compute metrics at the current classifier state

        Args
            x (ndarray): nXd array of input samples
            y (ndarray): nX1 of true labels
            prediction_func (function) - optional: prediction function

        Returns
            MetricSet: containing classifier performance metrics
        """
        if x is None and y is None:
            return MetricSet()

        if prediction_func is None:
            pre = self.predict(x)
        else:
            pre = prediction_func(x)

        p = np.sum(y == 1)
        n = np.sum(y == -1)
        tp = np.sum([yh == 1 and yt == 1 for yh, yt in zip(pre, y)])
        tn = np.sum([yh == -1 and yt == -1 for yh, yt in zip(pre, y)])
        fp = np.sum([yh == 1 and yt == -1 for yh, yt in zip(pre, y)])

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

        return MetricSet(acc=acc, err=err, pre=prec, rec=rec,
                         f1=f1, fpr=fpr, tpr=tpr, specificity=specificity)

    def __backtracking(self, beta):
        """backtracking line search

        use the provided beta values to determine optimum learning rate for
        current iteration

        Args
            beta (ndarray): 1xD array of weight coefficients

        Return
            Float: learning rate
        """
        p = self._param
        a, t, t_eta, max_iter = p('alpha'), p('eta'), p('t_eta'), p('bt_max_iter')

        gb = self._compute_grad(beta)
        n_gb = np.linalg.norm(gb)

        found_t, i = False, 0
        while not found_t and i < max_iter:
            lh = self._objective(beta - t * gb)
            rh = self._objective(beta) - a * t * n_gb ** 2
            if lh < rh:
                found_t = True
            elif i == max_iter - 1:
                break
            else:
                t *= t_eta
                i += 1

        self._set_param('eta', t)
        return t

    def __grad_descent(self, beta=None):
        """gradient descent

        Args
            beta (ndarray) - optional: 1Xd array of weight coefficients
        """
        x, y, eta, max_iter = self._x, self._y, self._param('eta'), self._param('max_iter')
        if beta is None:
            beta = self.coef_
        grad_x = self._compute_grad(beta)

        i = 0
        while i < max_iter:
            beta = beta - eta * grad_x
            grad_x = self._compute_grad(beta)

            i += 1
            self._set_betas(beta)
            self.log_metrics([i, self._objective(beta)])

    def __fast_grad_descent(self):
        """fast-gradient descent

        perform gradient descent with the fast-gradient algorithm
        """
        eps, max_iter = self._param('eps'), self._param('max_iter')

        b0 = self.coef_
        theta = np.copy(b0)
        grad = self._compute_grad(theta)

        i = 0
        while i < max_iter:
            t = self.__backtracking(b0)

            b1 = theta - t * grad
            theta = b1 + (i / (i + 3)) * (b1 - b0)
            grad = self._compute_grad(theta)
            b0 = b1

            i += 1
            self._set_betas(b0)
            self.log_metrics([self._param('lambda'), i, t, self._objective(b0)])

            if np.isclose(0, t):
                break

    # ---------------------------------------------------------------
    # miscellaneous methods
    # ---------------------------------------------------------------
    def load_from_disk(self, path):
        """load the classifier from disk

        Args
            path (str): path to classifier file location. name
            should match current classifier task
        """
        try:
            with open('%s%s.pk' % (path, self.task), 'rb') as f:
                data = pickle.load(f)
                self.__load_from_dict(data)
                return self
        except Exception as e:
            print('could not load model from disk: %s' % ', '.join([str(a) for a in e.args]))

    def log_metrics(self, args=None, prediction_func=None):
        """metrics logging

        Args
            args ([obj]) - optional: list of additional metrics to print
            prediction_func (function) - optional: prediction function

        Returns
            None
        """
        split = self.__current_split

        train_metrics = self.__compute_metrics(self._x, self._y, prediction_func)
        self.__cv_splits[split].train_metrics = train_metrics

        if self.__cv_splits[split].val_idx is not None:
            val_metrics = self.__compute_metrics(self._x_val, self._y_val, prediction_func)
            self.__cv_splits[split].val_metrics = val_metrics

        if self.__logging_level == 'none':
            return

        # write basic classifier state
        pstr = ', '.join([str(v) for k, v in self.__cv_splits[self.__current_split].parameters.items()])
        arg_str = ', '.join(['%.7f' % a if not float(a).is_integer() else str(a) for a in args])

        # write row by current logging level
        no_val_set = 'no_val_set'
        if self.__logging_level == 'minimal':
            row = arg_str

        elif self.__logging_level == 'reduced':
            row = f'{self.task}, ' \
                  f'{round(train_metrics.error, 4)}, ' \
                  f'{round(val_metrics.error, 4) if val_metrics is not None else no_val_set}, ' \
                  f'{arg_str}'

        elif self.__logging_level == 'verbose':
            row = f'{datetime.datetime.now()}, ' \
                  f'{self.task}, ' \
                  f'{pstr}, ' \
                  f'{str(train_metrics)}, ' \
                  f'{str(val_metrics)}, ' \
                  f'{arg_str}'

        else:
            return

        # if there's a log queue managing multiple classifiers, forward the message
        if self.__log_queue is not None:
            self.__log_queue.put(row)

        # otherwise print it to the console and output to csv
        else:
            print(row)
            if self.__log_path is not None:
                with open('%s%s.csv' % (self.__log_path, self.task), 'a+') as f:
                    f.writelines(row + '\n')

    def set_log_queue(self, queue):
        """set queue for logging metrics

        Args
            queue (multiprocessing.Queue): set the managing log queue
        """
        self.__log_queue = queue

    def write_to_disk(self, path=None):
        """write current classifier state to disk

        Args
            path (str) - optional: file path to store calassifier
        """
        try:
            with open('%s%s.pk' % (path, self.task), 'wb') as f:
                pickle.dump(self.dict, f, pickle.HIGHEST_PROTOCOL)

            print('%s [%s] written to disk' % (self.task, self.__parameters))
        except Exception as e:
            print('could not write %s to disk: %s' % (self.task, [str(a) for a in e.args]))

    def _param(self, parameter):
        """retrieve parameter from current training split

        Args
            parameter (str): key into dictionary of parameters

        Raises
            KeyError: if key is not defined

        Returns
            object: parameter
        """
        try:
            return self.__cv_splits[self.__current_split].parameters[parameter]
        except KeyError as e:
            print([str(a) for a in e.args])
            return None

    def _set_betas(self, betas):
        """set beta values for current training split

        Args
            betas (ndarray): 1Xd array of beta values to set

        Returns
            None
        """
        self.__cv_splits[self.__current_split].betas = betas

    def _set_param(self, param, value):
        """set parameter of current training split

        Adrgs
            param (str): dictionary key to set
            value (object): value to associate with provided key
        """
        self.__cv_splits[self.__current_split].parameters[param] = value

    def __generate_splits(self, x_train, y_train, x_val, y_val):
        """generate training splits

        Args
            x_train: nXd ndarray, training samples
            y_train: 1Xn ndarray, true training labels
            x_val: nXd ndarray, validation samples
            y_val: 1Xn ndarray, true validation labels

        Returns
            (x, y, [TrainingSplit])
        """
        # if
        if x_val is not None and y_val is not None:
            x = np.concatenate((x_train, x_val))
            y = np.concatenate((y_train, y_val))
        else:
            x = x_train
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
            for key, value in self.__parameters.items():
                parameters[key] = value[0]

            splits.append(
                TrainingSplit(
                    n=x_train.shape[0],
                    d=x_train.shape[1],
                    train_idx=[i for i in range(len(y_train))],
                    val_idx=[i for i in range(len(y_train), len(y_train) + len(y_val))],
                    parameters=parameters
                )
            )

        return x, y, splits

    def __load_from_dict(self, dict_rep):
        """load classifier from provided dictionary representation

        Args
            dict_rep (dict): dictionary to load keys from
        """
        self.task = dict_rep['task']
        self.__x = dict_rep['x']
        self.__y = dict_rep['y']
        self.__logging_level = dict_rep['logging_level']
        self.__log_path = dict_rep['log_path']
        self.__parameters = dict_rep['parameters']
        self.__cv_splits = [TrainingSplit().from_dict(d) for d in dict_rep['splits']]

    @property
    def coef_(self):
        return self.__cv_splits[self.__current_split].betas

    @property
    def dict(self):
        return {
            'task': self.task,
            'x': self.__x,
            'y': self.__y,
            'logging_level': self.__logging_level,
            'log_path': self.__log_path,
            'parameters': self.__parameters,
            'splits': [s.dict for s in self.__cv_splits]
        }

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
    def __current_split(self):
        """retrieve split number

        matches thread name to dictionary map of training splits

        Returns
            int: training split

        Raises
            Exception: if the thread doesn't have an associated split number
            ThreadError: re raised
        """
        try:
            tid = threading.current_thread()
            split = self.__thread_split_map.get(tid.getName(), -1)

            if split == -1:
                raise Exception('split not found')

            return split
        except threading.ThreadError as e:
            print([str(a) for a in e.args])
            raise threading.ThreadError()


# ---------------------------------------------------------------
# helper classes
# ---------------------------------------------------------------
class MetricSet:
    def __init__(self, acc=0., err=1., pre=0., rec=0.,
                 f1=0., fpr=1., tpr=0., specificity=0.):
        self.accuracy = acc
        self.error = err
        self.precision = pre
        self.recall = rec
        self.f1_measure = f1
        self.fpr = fpr
        self.tpr = tpr
        self.specificity = specificity

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

    def from_dict(self, data):
        self.accuracy = data['accuracy']
        self.error = data['error']
        self.precision = data['precision']
        self.recall = data['recall']
        self.f1_measure = data['f1_measure']
        self.fpr = data['fpr']
        self.tpr = data['tpr']
        self.specificity = data['specificity']

        return self


class TrainingSplit:
    def __init__(self, n=0, d=0, train_idx=None, val_idx=None, parameters=None, betas=None):
        self.n = n
        self.d = d
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.parameters = parameters if parameters is not None else {}
        self.betas = betas if betas is not None else []
        self.train_metrics = None
        self.val_metrics = None

    @property
    def dict(self):
        return {
            'n': self.n,
            'd': self.d,
            'train_idx': self.train_idx if not None else [],
            'val_idx': self.val_idx if not None else [],
            'parameters': self.parameters,
            'betas': self.betas,
            'train_metrics': self.train_metrics.dict if self.train_metrics is not None else 'none',
            'val_metrics': self.val_metrics.dict if self.val_metrics is not None else 'none'
        }

    def from_dict(self, data):
        self.betas = data['betas']
        self.d = data['d']
        self.train_metrics = MetricSet().from_dict(data['train_metrics'])
        self.val_metrics = MetricSet().from_dict(data['val_metrics'])
        self.n = data['n']
        self.parameters = data['parameters']
        self.train_idx = data['train_idx']
        self.val_idx = data['val_idx']

        return self
