from my_classifier import MyClassifier
import numpy as np
import pickle

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression(MyClassifier):
    def __init__(self, x_train, y_train, x_val=None, y_val=None,
                 lamda=.01, max_iter=500, eps=0.001, cv_splits=1,
                 log_queue=None, task=None):

        super().__init__(x_train, y_train, x_val, y_val, lamda,
                         cv_splits, log_queue, task)

        self.eps = eps
        self.max_iter = max_iter

        self.__betas = self.coef_
        self.__eta = 1. #self.__calc_t_init()
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
                self.__graddescent()
            elif algo == 'fgrad':
                self.__betas.append(self.__betas[-1])
                self.__thetas = init(init_method)[0]
                self.__fastgradalgo()
                self.__betas = self.__betas[1:]
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
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [exp(xi@b)/(1 + exp(xi@b)) for xi in x]

    # private methods
    def __backtracking(self, beta, t_eta=0.5, alpha=0.5):
        l, t = self._lamda, self.__eta

        gb = self.__computegrad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < self.max_iter:
            if self.__objective(beta - t*gb) < self.__objective(beta) - alpha * t * n_gb**2:
                found_t = True
            elif i == self.max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        self.__eta = t
        return self.__eta

    def __calc_t_init(self):
        x, l, n = self._x, self._lamda, self._n

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self._x, self._y, self._lamda, self._n

        p = (1 + exp(y * (x @ b))) ** -1
        return 2 * l * b - (x.T @ np.diag(p) @ y) / n

    def __graddescent(self):
        grad_x = self.__computegrad(self.__betas[-1])

        i = 0
        while norm(grad_x) > self.eps and i < self.max_iter:
            b0 = self.__betas[-1]
            t = self.__backtracking(b0)

            self.__betas.append(b0 - t * grad_x)
            grad_x = self.__computegrad(b0)

            i += 1
            self.log_metrics([i, t, norm(grad_x), norm(b0), self.__objective(b0), self._lamda])

    def __fastgradalgo(self):
        theta = self.__thetas
        grad = self.__computegrad(theta)

        i = 0

        while norm(grad) > self.eps and i < self.max_iter:
            b0 = self.__betas[-1]
            t  = self.__backtracking(b0)
            grad = self.__computegrad(theta)

            b1 = theta - t*grad
            self.__betas.append(b1)

            theta = b1 + (i/(i+3))*(b1-b0)
            i += 1

            self.log_metrics([i, t, norm(grad), norm(b0), self.__objective(b0), self._lamda])

    def __objective(self, beta):
        x, y, n, l = self._x, self._y, self._n, self._lamda

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2

    def __repr__(self):
        return "MyLogisticRegression(task=%s, lambda=%s)" % (self.task, self._lamda)

