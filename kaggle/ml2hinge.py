from kaggle.my_classifier import MyClassifier
import numpy as np


# misc setup for readability
norm = np.linalg.norm


class MyL2Hinge(MyClassifier):
    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None,
                 log_queue=None, logging_level='none', log_path='', task=None,
                 dict_rep=None):

        super().__init__(x_train=x_train, y_train=y_train, parameters=parameters,
                         x_val=x_val, y_val=y_val, task=task, log_queue=log_queue,
                         log_path=log_path, logging_level=logging_level, dict_rep=dict_rep)

        self.__betas = self.coef_

    def _simon_says_fit(self):
        return self.fit()

    # public methods
    def fit(self):
        def init(method):
            if method == 'ones':
                b = np.ones(self._d)
            elif method == 'zeros':
                b = np.zeros(self._d)
            elif method == 'normal':
                b = np.random.normal(0, 1, self._d)
            else:
                raise Exception('init method not defined')
            return b

        while super().fit():
            self.__betas = self.coef_

            if len(self.__betas) == 0:
                self.__betas = init(self._param('init_method'))
                self._set_betas(self.__betas)

            self.__grad_descent()

            self._set_betas(self.__betas)
        return self

    def predict(self, x, beta=None):
        if beta is None:
            beta = self.__betas

        return [1 if xi @ beta.T > 0 else -1 for xi in x]

    def predict_proba(self, x, beta=None):
        return None

    def __objective(self, beta):
        x, y, l = self._x, self._y, self._param('lambda')

        return 1/len(y) * np.sum([np.maximum(0, 1-yi*xi @ beta)**2 for xi, yi in zip(x, y)])+l*norm(beta)**2

    def __backtracking(self, beta):
        p = self._param
        a, t, t_eta, max_iter = 0.1, p('eta'), 0.5, 100

        gb = self.__compute_grad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < max_iter:
            m = self.__objective((beta - t*gb))
            n = self.__objective(beta) - a*t*n_gb**2

            if m <= n:
                found_t = True
            elif i == max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        self._set_param('eta', t)
        return t

    def __compute_grad(self, beta):
        x, y, n, l = self._x, self._y, self._n, self._param('lambda')

        reg = 2*l*norm(beta)
        inner = np.array([yi*(xi.T@beta) for xi, yi in zip(x, y)])
        db = np.sum([-1.*yi*xi*(1-i) for xi, yi, i in zip(x, y, inner) if i <= 1 ], axis=0)

        return 2/n * db + reg

    def __grad_descent(self):
        x, y, beta, eta = self._x, self._y, self.__betas, self._param('eta')
        grad_x = self.__compute_grad(beta)

        i = 0
        while norm(grad_x) > 0.3 or i == 0 and i < 1000:
            beta -= eta * grad_x
            grad_x = self.__compute_grad(beta)

            i += 1
            self.log_metrics([i, eta, self.__objective(beta), norm(grad_x)])
        self.__betas = beta
