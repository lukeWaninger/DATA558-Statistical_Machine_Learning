from kaggle.my_classifier import MyClassifier
import numpy as np
import pickle

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression(MyClassifier):
    def __init__(self, x_train, y_train, x_val=None, y_val=None,
                 lamda=.01, max_iter=500, eps=0.001, log_queue=None,
                 task=None):
        super().__init__(x_train, y_train, x_val, y_val, log_queue, task)

        self.betas = None
        self.eps = eps
        self.lamda = lamda
        self.max_iter = max_iter

        self.__eta = self.__calc_t_init()
        self.__objective_vals = None
        self.__thetas = None

    @property
    def coef_(self):
        return self.betas[-1] if self.betas is not None else []

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

        self.betas = init(init_method)
        self.__objective_vals = None

        if algo == 'grad':
            self.__graddescent()
        elif algo == 'fgrad':
            self.betas.append(self.betas[-1])
            self.__thetas = init(init_method)[0]
            self.__fastgradalgo()
            self.betas = self.betas[1:]
        else:
            raise Exception("algorithm <%s> is not available" % algo)

        self.betas = self.betas[1:]
        return self

    def load_from_disk(self, path):
        with open('%s%s.pk' % (path, self.task), 'rb') as f:
            data = pickle.load(f)

            self.betas    = [data['coef']]
            self.eps      = data['eps']
            self.lamda    = data['lamda']
            self.max_iter = data['max_iter']
            self.__eta    = data['eta']
            self._x       = data['x']
            self._x_val   = data['x_val']
            self._y       = data['y']
            self._y_val   = data['y_val']

        return self

    def predict(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [1 if xi @ b.T > 0 else -1 for xi in x]

    def predict_proba(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [exp(xi@b)/(1 + exp(xi@b)) for xi in x]

    def write_to_disk(self, path):
        dict_rep = {
            'task':     self.task,
            'coef':     self.betas[-1],
            'eps':      self.eps,
            'lamda':    self.lamda,
            'max_iter': self.max_iter,
            'eta':      self.__eta,
            'x':        self._x,
            'x_val':    self._x_val,
            'y':        self._y,
            'y_val':    self._y_val
        }

        with open('%s%s.pk' % (path, self.task), 'wb') as f:
            pickle.dump(dict_rep, f, pickle.HIGHEST_PROTOCOL)

    # private methods
    def __backtracking(self, beta, t_eta=0.5, alpha=0.5):
        l, t = self.lamda, self.__eta

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
        x, l, n = self._x, self.lamda, self._n

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self._x, self._y, self.lamda, self._n

        p = (1 + exp(y * (x @ b))) ** -1
        return 2 * l * b - (x.T @ np.diag(p) @ y) / n

    def __graddescent(self):
        grad_x = self.__computegrad(self.betas[-1])

        i = 0
        while norm(grad_x) > self.eps and i < self.max_iter:
            b0 = self.betas[-1]
            t = self.__backtracking(b0)

            self.betas.append(b0 - t * grad_x)
            grad_x = self.__computegrad(b0)

            i += 1
            self.log_metrics([i, t, norm(grad_x), norm(b0), self.__objective(b0)])

    def __fastgradalgo(self):
        theta = self.__thetas
        grad = self.__computegrad(theta)

        i = 0

        while norm(grad) > self.eps and i < self.max_iter:
            b0 = self.betas[-1]
            t  = self.__backtracking(b0)
            grad = self.__computegrad(theta)

            b1 = theta - t*grad
            self.betas.append(b1)

            theta = b1 + (i/(i+3))*(b1-b0)
            i += 1

            self.log_metrics([i, t, norm(grad), norm(b0), self.__objective(b0)])

    def __objective(self, beta):
        x, y, n, l = self._x, self._y, self._n, self.lamda

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2

    def __repr__(self):
        return "MyLogisticRegression(C=%s, eps=%s, max_iter=%s)" % (self.lamda, self.eps, self.max_iter)

