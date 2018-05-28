from abc import ABC, abstractmethod
import datetime
from myml.metrics import MetricSet
from myml.cvsplit import TrainingSplit
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
        self.__log_path      = args[1]['log_path']      if 'log_path'      in args[1].keys() else ''
        self.__log_queue     = args[1]['loq_queue']     if 'loq_queue'     in args[1].keys() else None
        self.__write_to_disk = args[1]['write_to_disk'] if 'write_to_disk' in args[1].keys() else False

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

        # wait for threads to finish
        while len(self.__thread_split_map.keys()) != 0:
            time.sleep(1)

        # find best params and refit
        if len(self.__cv_splits) > 1:
            splits, metrics = self.__cv_splits, []
            val_errors = [s.val_metrics.dict['error'] for s in splits if s.val_metrics is not None]
            val_errors = [e for e in val_errors if e > 0]
            idx = np.argmin(val_errors) if len(val_errors) > 0 else 0

            # create a new split with parameters from the best
            n, d = self.__x.shape
            new_split = TrainingSplit(
                n=n, d=d,
                train_idx=np.arange(n),
                parameters=splits[idx].parameters
            )
            new_split.parameters['eta'] = 1.

            # remove the old splits to free some memory
            del self.__cv_splits
            self.__cv_splits = [new_split]

            # train with the identified parameter set
            thread = threading.Thread(target=self.fit_one)
            name = thread.getName()
            self.__thread_split_map[name] = len(self.__cv_splits) - 1
            thread.start()

            while thread.isAlive():
                time.sleep(.5)

        # free more memory
        self.__x = []
        self.__y = []
        self.__cv_splits[0].train_idx = []
        self.__cv_splits[0].val_idx = []

        # write self to disk
        if self.__log_path is not None and self.__write_to_disk:
            self.write_to_disk(self.__log_path)

        return self

    def fit_one(self):
        """fit a single training fold"""
        split = self.__current_split

        # initialize optimization coefficients
        if self.__current_split.has_kernel:
            self._set_param('betas', np.zeros(self._n))
        else:
            self._set_param('betas', np.zeros(self._d))

        algo = split.parameters['algo']
        if algo == 'grad':
            self.__grad_descent()
        elif algo == 'fgrad':
            self.__fast_grad_descent()

        thread_name = threading.currentThread().getName()
        del self.__thread_split_map[thread_name]

    # ---------------------------------------------------------------
    # optimization methods
    # ---------------------------------------------------------------
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
            self._set_param('betas', beta)
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
            self._set_param('betas', b0)
            self.log_metrics([self._param('regularization'), i, t, self._objective(b0)])

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

    def log_metrics(self, args=None):
        """metrics logging

        Args
            args ([obj]) - optional: list of additional metrics to print
            prediction_func (function) - optional: prediction function

        Returns
            None
        """
        if self.__logging_level == 'none':
            return

        split = self.__current_split
        y_hat = self.predict(self.__x)

        train_metrics = MetricSet(y_hat=y_hat, y_true=self._y)
        split.train_metrics = train_metrics

        if split.val_idx is not None:
            y_hat = self.predict(self._x_val)

            val_metrics = MetricSet(y_hat=y_hat, y_true=self._y_val)
            split.val_metrics = val_metrics
        else:
            val_metrics = None

        # write basic classifier state
        pstr = ', '.join([str(v) for k, v in split.parameters.items()])
        arg_str = ', '.join(['%.7f' % a if isinstance(a, float) and not float(a).is_integer() else str(a) for a in
                             args])

        # write row by current logging level
        no_val_set = 'no_val_set'
        if self.__logging_level == 'minimal' or train_metrics is None or val_metrics is None:
            row = f'{self.task}, {arg_str}'

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
        return self.__current_split.get_param(parameter)

    def _set_param(self, param, value):
        """set parameter of current training split

        Args
            param (str): dictionary key to set
            value (object): value to associate with provided key
        """
        self.__current_split.set_param(param, value)

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
            if not isinstance(value, list):
                value = [value]

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

        # if there are no cross-val splits seen in the classifier param set
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
        if len(self.__cv_splits) == 1:
            return self.__cv_splits[0].betas
        else:
            return self.__current_split.betas

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
        return self.__current_split.d

    @property
    def _n(self):
        return self.__current_split.n

    @property
    def _x(self):
        if self.__current_split.has_kernel:
            return self.__current_split.kernel
        else:
            return self.__x[self.__current_split.train_idx]

    @property
    def _x_val(self):
        return self.__x[self.__current_split.val_idx]

    @property
    def _y(self):
        return self.__y[self.__current_split.train_idx]

    @property
    def _y_val(self):
        return self.__y[self.__current_split.val_idx]

    @property
    def __current_split(self):
        """retrieve current  training split

        matches thread name to dictionary map of training splits

        Returns
            int: training split

        Raises
            Exception: if the thread doesn't have an associated split number
            ThreadError: re raised
        """
        return self.__cv_splits[self.__current_split_idx]

    @property
    def __current_split_idx(self):
        try:
            tname = threading.currentThread().getName()
            split = self.__thread_split_map.get(tname, -1)

            if split == -1:
                raise Exception('split not found')

            return split

        except threading.ThreadError as e:
            print([str(a) for a in e.args])
            raise e
