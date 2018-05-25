from myml.mlassoreg import MyLASSORegression
from myml.mlogreg import MyLogisticRegression
from myml.mridgereg import MyRidgeRegression
from myml.msvm import MyLinearSVM
import multiprocessing
import numpy as np
import os
import pandas as pd
from scipy.stats import mode
import time


class MultiClassifier(object):
    """classifier for multi-class classification"""

    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None, task='',
                 classification_method='ovr', n_jobs=-1, log_path='', logging_level='none'):
        """class constructor

        Args:
            x_train (ndarry): nXd array of input samples
            y_train (ndarry: nX1 array of true labels
            parameters (dictionary):
            x_val (ndarry): nXd arrray of validation samples
            y_val (ndarray): nXd array of validation labels
            task (str): name of task for writing
            classification_method (str): {ovr, all_pairs}
            n_jobs (int): number of processors to use
            log_path (path): path to save log files
            logging_level (path): {minimal, reduced, verbose}
        """
        # calculate and set designated number of processes or max if n_jobs is -1
        cpu_count = multiprocessing.cpu_count()
        if n_jobs == -1 or n_jobs >= cpu_count:
            self.__available_procs = cpu_count*2
        else:
            self.__available_procs = n_jobs

        self.__classification_method = classification_method
        self.__parameters = parameters
        self.__x = x_train
        self.__x_val = x_val
        self.__y = y_train
        self.__y_val = y_val
        self.__to_train = self.__get_training_sets()

        self.__manager = multiprocessing.Manager()
        self.__log_queue = self.__manager.Queue()
        self.__log_path = log_path
        self.__logging_level = logging_level
        self.__completion_queue = self.__manager.Queue()
        self.__snd, self.__rcv = multiprocessing.Pipe()

        self.task = task
        self.cvs = []

    def fit(self):
        """fit the classifier

        Returns
            self
        """
        # create and start a process to manage logging across all child classifiers
        log_manager = multiprocessing.Process(target=self.__log_manager,
                                              args=(self.__log_queue,
                                                    self.__snd))
        log_manager.start()

        # start consumers
        consumers = []
        consumption_queue = multiprocessing.Queue()

        for i in range(self.__available_procs):
            p = multiprocessing.Process(target=self.__consumer,
                                        args=(consumption_queue, self.__log_queue, self.__completion_queue))
            consumers.append(p)
            p.start()

        # act as producer and dish out classifiers
        while len(self.__to_train) > 0:
            consumption_queue.put(self.__to_train.pop())
            self.__empty_completion_queue()

        [consumption_queue.put('END_FLAG') for c in consumers]

        def is_running():
            for c in consumers:
                if c.is_alive():
                    return True
            return False

        while is_running():
            self.__empty_completion_queue()

        # wait for each process to finish training
        while len(consumers) > 0:
            consumers[0].join()
            consumers[0].terminate()
            consumers.remove(consumers[0])

        log_manager.terminate()
        return self

    def output_predictions(self, x):
        """predict and output predictions to file

        Args
            x (ndarray): nXd array of samples to predict

        Returns
            None
        """
        predictions = self.predict(x)
        predictions = [(i, p) for i, p in enumerate(predictions)]
        pd.DataFrame(predictions).to_csv(f'{self.__log_path}/{self.task}.csv',
                                         index=None,
                                         header=['Id', 'Category'])

    def predict(self, x):
        """predict labels for provided samples

        Args
            x (ndarray): nXd array of samples to predict

        Raises
            ValueError: if current classification method is not defined

        Returns
            [({-1, 1})]: list of predicted labels
        """
        # predict using one-versus-rest algorithm
        if self.__classification_method == 'ovr':
            predictions = []
            for cv in self.cvs:
                predictions.append(cv.predict_proba(x))

            predictions = np.array(predictions).T
            predictions = [np.argmax(p) for p in predictions]
            return predictions

        # predict using all-pairs algorithm
        elif self.__classification_method in ['all_pairs']:
            predictions = []
            for cv in self.cvs:
                if 'rest' in cv.task:
                    continue

                pre = cv.predict(x)
                e = cv.task.split(' ')
                pos, neg = e[0], e[2]
                predictions.append([int(pos) if pi == 1 else int(neg) for pi in pre])
            predictions = [mode(pi).mode for pi in np.array(predictions).T]

            # break ties at random
            ties = [(i, pi) for i, pi in enumerate(predictions) if len(pi) > 1]
            if len(ties) > 0:
                for idx, pi in ties:
                    predictions[idx] = np.random.choice(pi)

            predictions = [int(i) for i in predictions]
            return predictions

        else:
            raise ValueError('classification method not found')

    def __build_classifiers(self, class_set):
        """build the classifiers for either one-vs-rest or all-pairs

        Returns
            ([MyClassifier]): list of classifiers

        Raises
            ValueError: if designated classifier model is not found
        """
        cvs, x_idx, x_idx_v, y_v, x_v = [], [], [], [], []
        pos, neg = class_set

        x, x_v = self.__x, self.__x_val
        y, y_v = self.__y, self.__y_val

        # set class labels and parse indices
        if neg == 'rest':
            neg_idx = np.where(y != int(pos))[0]
            pos_idx = np.where(y == int(pos))[0]

            sub_idx = np.random.choice(neg_idx, int(len(neg_idx)))
            sub_idx = np.concatenate((sub_idx, pos_idx))

            x = x[sub_idx, :]
            y = y[sub_idx]

            y = y**0*-1

            pos_idx = np.where(y == int(pos))[0]
            y[pos_idx] = 1

            if len(x_v) > 0 and len(y_v) > 0:
                v_pos_idx = np.where(y_v == int(pos))[0]
                y_v = y_v**0*-1
                y_v[v_pos_idx] = 1

        else:
            pos_idx = np.where(y == int(pos))[0]
            neg_idx = np.where(y == int(neg))[0]
            y = y**0*-1
            y[pos_idx] = 1

            x_idx = np.concatenate((pos_idx, neg_idx))
            y = y[x_idx]
            x = x[x_idx]

            if len(x_v) > 0 and len(y_v) > 0:
                v_pos_idx = np.where(y_v == int(pos))[0]
                v_neg_idx = np.where(y_v == int(neg))[0]

                y_v = y_v**0*-1
                y_v[v_pos_idx] = 1

                x_idx_v = np.concatenate((v_pos_idx, v_neg_idx))
                y_v = y_v[x_idx_v]
                x_v = x_v[x_idx_v]

        task = '%s vs %s' % (pos, neg)
        for cv in self.__parameters['classifiers']:
            cv_type = cv['type']

            if cv_type == 'linear_svm':
                classifier = MyLinearSVM(x_train=x, y_train=y, x_val=x_v, y_val=y_v,
                                         parameters=cv['parameters'],
                                         task=task + " [linear_svm]",
                                         logging_level=self.__logging_level,
                                         log_queue=self.__log_queue)

            else:
                raise ValueError('classifier model not found')

            cvs.append(classifier)
        return cvs

    def __empty_completion_queue(self):
        while not self.__completion_queue.empty():
            cv_d = self.__completion_queue.get()

            tasks = cv_d['task']
            if 'linear_svm' in tasks:
                cv_d = MyLinearSVM(x_train=None, y_train=None, parameters=None, dict_rep=cv_d)

            else:
                cv_d = None

            if cv_d is not None:
                self.cvs.append(cv_d)

    def __get_training_sets(self):
        """generate training sets

        generates training sets based on the defined classification method

        Returns
            [(a,b)], list of tuples
        """
        classes = [str(c) for c in np.unique(self.__y)]
        pairs = []

        m = self.__classification_method
        if m in ['ovr', 'both']:
            [pairs.append((c, 'rest')) for c in classes]

        if m in ['all_pairs', 'both']:
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    pairs.append((classes[i], classes[j]))

        return pairs

    def __log_manager(self, log_queue, conn):
        """log manager

        controlling process for child classifiers to route log messages through

        Args
            log_queue (multiprocessing.Queue): queue to receive child log messages
            conn (multiprocessing.Pipe): pipe to send termination message to parent process

        Returns
            None
        """
        start = time.time()
        path = f'{self.__log_path}/{self.task}.csv'

        # continue until no child processes are running
        k = len(np.unique(self.__y))
        running = k*(k-1)/2
        while True:
            message = log_queue.get()

            if message == 'END_FLAG':
                running -= 1

            else:
                #print(message)
                try:
                    with open(path, 'a+') as f:
                        f.writelines(message + "\n")
                except Exception as e:
                    print(e.args)

            if running == 0:
                break

        time_delta = time.time()-start
        print(f'training completed in {round(time_delta, 2)} seconds.\n')
        conn.send(True)

    def __consumer(self, consumption_queue, log_queue, completion_queue):
        while True:
            to_train = consumption_queue.get()

            if to_train == 'END_FLAG':
                log_queue.put('END_FLAG')
                break

            cvs = self.__build_classifiers(to_train)

            for cv in cvs:
                start = time.time()
                print(f'starting {cv.task} on pid {os.getpid()}')

                cv.set_log_queue(log_queue)
                cv = cv.fit()

                print(f'{os.getpid()} finished {cv.task} in {round(time.time() - start, 2)} seconds')
                completion_queue.put(cv.dict)