from my_classifier import MyClassifier
import numpy as np

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression(MyClassifier):
    def __init__(self, x_train, y_train, x_val=None, y_val=None,
                 lamda=.01, max_iter=1000, cv_splits=1,
                 log_queue=None, task=None):

        super().__init__(x_train, y_train, x_val, y_val, lamda,
                         cv_splits, log_queue, task)

        self.max_iter = max_iter

        self.__betas = self.coef_
        self.__eta = 1. #self.__calc_t_init()
        self.__objective_vals = None
        self.__thetas = None

    # public methods
    def fit(self, algo='coord_descent', init_method='zeros'):
        def init(method):
            if method == 'ones':
                b = [np.ones(self._d)]
            elif method == 'zeros':
                b = [np.zeros(self._d)]
            elif method == 'normal':
                b = [np.random.normal(0, 1, self._d)]
            else:
                raise Exception('init method not defined')
            return b

        while super().fit():
            self.__betas = self.coef_

            if len(self.__betas) == 0:
                self.__betas = init(init_method)
                self._set_betas(self.__betas[0])

            self.__objective_vals = None

            if algo == 'coord_descent':
                self.__graddescent()

            else:
                raise Exception("algorithm <%s> is not available" % algo)

            self._set_betas(self.__betas[-1])
        return self

    def predict(self, x, betas=None):
        if betas is not None:
            b = betas
        elif len(self.__betas) > 0:
            b = self.__betas[-1]
        else:
            b = self.coef_

        return None

    def predict_proba(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return None

    def __objective(self):
        x, y, n, l, b = self._x, self._y, self._n, self._lamda, self.__betas[-1]

        return (1/n)*norm(y-x.T @ b)**2 + l*norm(b)

    def __cyclic_coordinate_descent(self):
        pass

    def __pick_coordinate(self):
        pass

    def __random_coordinate_descent(self):
        pass
