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

            algo = self._param('algo')
            if algo == 'grad':
                self.__grad_descent()
            elif algo == 'fgrad':
                self.__fast_grad_descent()
            else:
                raise Exception("algorithm <%s> is not available" % algo)

            self._set_betas(self.__betas)
        self.log_metrics([self.__objective(self.coef_)])
        return self

    def predict(self, x, beta=None):
        if beta is None:
            beta = self.__betas

        return [1 if xi @ beta.T > 0 else -1 for xi in x]

    def predict_proba(self, x, beta=None):
        return None

    def __objective(self, beta):
        x, y, l, n = self._x, self._y, self._param('lambda'), self._n

        return 1/n * np.sum([np.maximum(0, 1-yi*(xi.T@beta))**2 for xi, yi in zip(x, y)])+l*norm(beta)**2

    def __backtracking(self, beta):
        p = self._param
        a, t, t_eta, max_iter, eps = p('alpha'), p('eta'), p('t_eta'), p('bt_max_iter'), p('eps')

        gb = self.__compute_grad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < max_iter:
            if self.__objective(beta - t*gb) <= self.__objective(beta) - a*t*n_gb or \
                    t*norm(gb) < 0.0001:
                found_t = True
            elif i == max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        return t

    def __compute_grad(self, beta):
        x, y, n = self._x, self._y, self._n,
        reg = 2*self._param('lambda')*norm(beta)

        # m = [i for i, xy in enumerate(zip(x, y)) if 1-xy[1]*xy[0].T@beta > 0]
        # xm, ym = x[m, :], y[m]
        #
        # return -2/n*(ym@xm - xm.T@(xm@beta)) + reg

        inner = np.array([yi*xi.T@beta for xi, yi in zip(x, y)])
        db = np.sum([yi*xi*(1-i) for xi, yi, i in zip(x, y, inner) if i <= 1], axis=0)

        return -2./n*db + reg

        # db = np.zeros(self._d)
        # for xi, yi in zip(x, y):
        #     if 1-yi*xi@beta > 0:
        #         db += yi*xi*(1-yi*(xi@beta))
        # return -2/n*db + reg

    def __grad_descent(self):
        x, y, beta, eta = self._x, self._y, self.__betas, self._param('eta')
        grad_x = self.__compute_grad(beta)

        i = 0
        while (norm(grad_x) > 0.3 or i == 0) and i < 1000:
            beta = beta - eta*grad_x
            grad_x = self.__compute_grad(beta)

            i += 1
            if i % 100 == 0:
                self.log_metrics([i])
        self.__betas = beta

    def __fast_grad_descent(self):
        eps, max_iter = self._param('eps'), self._param('max_iter')

        b0 = self.__betas
        theta = np.copy(self.__betas)
        grad = self.__compute_grad(theta)

        i = 0
        while norm(grad) > eps and i < max_iter:
            t  = self.__backtracking(b0)

            b1 = theta - t*grad
            theta = b1 + (i/(i+3))*(b1-b0)
            grad = self.__compute_grad(theta)

            b0 = b1
            i += 1
            self.log_metrics([i, self.__objective(theta)])
