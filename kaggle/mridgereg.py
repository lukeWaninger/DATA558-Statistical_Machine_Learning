from kaggle.my_classifier import MyClassifier
import numpy as np


# misc setup for readability
norm = np.linalg.norm
rand = np.random
log  = np.log
rand.seed(42)


class MyRidgeRegression(MyClassifier):
    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None,
                 log_queue=None, task=None):

        super().__init__(x_train, y_train, parameters, x_val, y_val, log_queue, task)

        self.eps = parameters['eps']
        self.max_iter = parameters['max_iter']

        self.__betas = self.coef_
        self.__eta = 1.
        self.__objective_vals = None
        self.__thetas = None

    # public methods
    def fit(self, algo='grad', init_method='zeros'):
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

            if algo == 'grad':
                self.__grad_descent()
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

        return [1 if xi @ b.T > 0 else -1 for xi in x]

    def predict_proba(self, x, betas=None):
        return None

    def __objective(self):
        x, y, l, beta = self._x, self._y, self._lamda, self.__betas[-1]

        return 2/len(y) * (norm((y-x@beta)**2)+l*norm(beta)**2)

    def __compute_grad(self):
        x, y, l, beta = self._x, self._y, self._lamda, self.__betas[-1]

        return 2/len(y) * (x.T@x@beta + l*beta - x.T@y)

    def __grad_descent(self):
        x, y, l, beta, eta = self._x, self._y, self._lamda, self.__betas[-1], self.__eta

        i, xvals = 0, []
        grad_x = self.__compute_grad()

        while i < self.max_iter:
            beta = beta - eta * grad_x
            xvals.append(self.__objective())
            grad_x = self.__compute_grad()

            i += 1

        self.__betas.append(beta)
