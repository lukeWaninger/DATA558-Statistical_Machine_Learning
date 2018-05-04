from kaggle.my_classifier import MyClassifier
import numpy as np

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression(MyClassifier):
    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None,
                 log_queue=None, logging_level='none', log_path='', task=None,
                 dict_rep=None):

        super().__init__(x_train=x_train, y_train=y_train, parameters=parameters,
                         x_val=x_val, y_val=y_val, log_queue=log_queue,
                         logging_level=logging_level, log_path=log_path, task=task,
                         dict_rep=dict_rep)

        self.__betas = self.coef_
        self.__objective_vals = None
        self.__thetas = None

    def _simon_says_fit(self):
        return self.fit()

    # public methods
    def fit(self):
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

            init_method = self._param('init_method')
            if len(self.__betas) == 0:
                self.__betas = init(init_method)
                self._set_betas(self.__betas[0])

            self.__objective_vals = None

            algo = self._param('algo')
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
    def __backtracking(self, beta):
        p = self._param
        a, t, t_eta, max_iter = p('alpha'), p('eta'), p('t_eta'), p('bt_max_iter')

        gb = self.__computegrad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < max_iter:
            if self.__objective(beta - t*gb) < self.__objective(beta) - a * t * n_gb**2:
                found_t = True
            elif i == max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        self._set_param('eta', t)
        return t

    def __calc_t_init(self):
        x, n, l = self._x, self._n, self._param('lambda')

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, n, l = self._x, self._y, self._n, self._param('lambda')

        p = (1 + exp(y * (x @ b))) ** -1
        return 2 * l * b - (x.T @ np.diag(p) @ y) / n

    def __graddescent(self):
        eps, max_iter = self._param('eps'), self._param('max_iter')
        grad_x = self.__computegrad(self.__betas[-1])

        i = 0
        while norm(grad_x) > eps and i < max_iter:
            b0 = self.__betas[-1]
            t = self.__backtracking(b0)

            self.__betas.append(b0 - t * grad_x)
            grad_x = self.__computegrad(b0)

            i += 1
            self.log_metrics([i, t, norm(grad_x), norm(b0), self.__objective(b0)])

    def __fastgradalgo(self):
        eps, max_iter = self._param('eps'), self._param('max_iter')

        theta = self.__thetas
        grad = self.__computegrad(theta)

        i = 0
        while norm(grad) > eps and i < max_iter:
            b0 = self.__betas[-1]
            t  = self.__backtracking(b0)
            grad = self.__computegrad(theta)

            b1 = theta - t*grad
            self.__betas.append(b1)

            theta = b1 + (i/(i+3))*(b1-b0)
            i += 1

            self.log_metrics([i, t, self.__objective(b0), norm(grad)])

    def __objective(self, beta):
        x, y, n, l = self._x, self._y, self._n, self._param('lambda')

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2

    def __repr__(self):
        return "MyLogisticRegression(task=%s, lambda=%s)" % (self.task, self._param('lambda'))
