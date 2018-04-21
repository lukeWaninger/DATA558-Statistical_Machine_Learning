from kaggle.models import LogMessage
import numpy as np
import os

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression:
    def __init__(self, X_train, y_train, lamda=2, max_iter=500, eps=0.001, idx=0, log_queue=None):
        self.betas = None

        self.eps = eps
        self.lamda = lamda
        self.max_iter = max_iter

        self.__log_queue = log_queue
        self.__idx = idx
        self.__x = X_train
        self.__y = y_train
        self.__n, self.__d = X_train.shape

        self.__eta = 1. # self.__calc_t_init()i
        self.__objective_vals = None
        self.__training_errors = None
        self.__thetas = None

    @property
    def coef_(self):
        return self.betas[-1].reshape(1, self.__d) if self.betas is not None else []

    @property
    def pos_class_(self):
        return self.__idx

    @property
    def objective_vals_(self):
        if self.__objective_vals is not None:
            return self.__objective_vals
        else:
            self.__objective_vals = [self.__objective(b) for b in self.betas] if self.betas is not None else []
            return self.__objective_vals

    @property
    def training_errors_(self):
        def accuracy(pre):
            return np.sum([yh == yt for yh, yt in zip(pre, self.__y)]) / self.__n

        if self.__training_errors is not None and len(self.__training_errors) == len(self.betas):
            return self.__training_errors

        elif self.__training_errors is not None:
            acc = accuracy(self.predict(self.__x))
            self.__training_errors.append(1 - acc)

        else:
            self.__training_errors = []

            for b in self.betas:
                pre = self.predict(self.__x, b)
                acc = accuracy(pre)
                self.__training_errors.append(1 - acc)
        return self.__training_errors

    # public methods
    def fit(self, algo='grad', init_method='zeros'):
        def init(method):
            if method == 'ones':
                b = [np.ones(self.__d)]
            elif method == 'zeros':
                b = [np.zeros(self.__d)]
            elif method == 'normal':
                b = [np.random.normal(0, 1, self.__d)]
            else:
                raise Exception('init method not defined')
            return b

        self.betas = init(init_method)
        self.__objective_vals = None
        self.__training_errors = None

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

    def objective(self, coef):
        return self.__objective(coef)

    def predict(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [1 if xi.T @ b.T > 0 else -1 for xi in x]

    def predict_proba(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [exp(xi@b)/(1 + exp(xi@b)) for xi in x]

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
        x, l, n = self.__x, self.lamda, self.__n

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self.__x, self.__y, self.lamda, self.__n

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

            # self.__log_queue.put(LogMessage(task='%s vs rest' % self.__idx,
            #                                 pid=str(os.getpid()),
            #                                 iteration=i,
            #                                 eta=t,
            #                                 norm_grad=norm(grad),
            #                                 norm_beta=norm(b0),
            #                                 objective=self.__objective(b0),
            #                                 training_error=self.training_errors_[-1],
            #                                 accuracy=1-self.training_errorg_errors_[-1]))

    def __objective(self, beta):
        x, y, n, l = self.__x, self.__y, self.__n, self.lamda

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2

    def __repr__(self):
        return "MyLogisticRegression(C=%s, eps=%s, max_iter=%s)" % (self.lamda, self.eps, self.max_iter)

