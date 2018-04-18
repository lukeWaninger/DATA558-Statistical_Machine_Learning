from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression:
    def __init__(self, X_train, y_train, X_test, y_test, lamda=0.01, max_iter=500):
        # private
        self._x = X_train
        self._y = y_train
        self._xtest = X_test
        self._ytest = y_test
        self._eps = 0.001
        self._lamda = lamda
        self._max_iter = max_iter
        self._n, self._d = X_train.shape
        self._eta = self.__calc_t_init()
        self._betas = None

        # public
        self.coef_ = self.__betas()
        self.objective_vals_ = self.__objective_vals()

    # public methods
    def fit(self, algo='grad'):
        self._betas = [np.ones(self._d)]

        if algo == 'grad':
            self.__graddescent()
        elif algo == 'fgrad':
            self._betas.append(np.ones(self._d))
            self.__fastgradalgo()
        else:
            raise Exception("algorithm <%s> is not available" % algo)

        return self

    # getters
    def __objective_vals(self):
        return [self.__objective(b) for b in self._betas[-1]]

    def __betas(self):
        return self._betas[-1] if self._betas is not None else []

    # private methods
    def __backtracking(self, beta, t_eta=0.5, alpha=0.5):
        l, t = self._lamda, self._eta

        gb = self.__computegrad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < self._max_iter:
            if self.__objective(beta - t * gb) < self.__objective(beta) - alpha * t * n_gb ** 2:
                found_t = True
            elif i == self._max_iter - 1:
                raise Exception("max number of backtracking iterations reached")
            else:
                t *= t_eta
                i += 1

        self._eta = t
        return self._eta

    def __calc_t_init(self):
        x, l, n = self._x, self._lamda, self._n

        m = np.max(1 / n * np.linalg.eigvals(x @ x.T)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self._x, self._y, self._lamda, self._n

        p = (1 + exp(y * (x @ b))) ** -1
        return 2 * l * b - (x.T @ np.diag(p) @ y) / n

    def __graddescent(self):
        grad_x = self.__computegrad(self._betas[-1])

        i = 0
        while norm(grad_x) > self._eps and i < self._max_iter:
            b0 = self._betas[-1]
            t = self.__backtracking(b0)

            self._betas.append(b0 - t * grad_x)
            grad_x = self.__computegrad(b0)

            i += 1

    def __fastgradalgo(self):
        theta = np.ones(self._d)
        grad = self.__computegrad(theta)

        i = 0
        while norm(grad) > self._eps and i < self._max_iter:
            b0 = self._betas[-1]
            t = self.__backtracking(b0)
            grad = self.__computegrad(theta)

            b1 = theta - t * grad
            self._betas.append(b1)
            theta = b1 + (i / (i + 3)) * (b1 - b0)

            i += 1

    def __objective(self, beta):
        x, y, n, l = self._x, self._y, self._n, self._lamda

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2


def exercise_1():
    try:
        spam = pd.read_csv('data/spam.csv', sep=',').dropna()
    except FileNotFoundError as e:
        spam = pd.read_csv('spam.csv', sep=',').dropna()

    X = spam.loc[:, spam.columns != 'type']
    y = spam.loc[:, 'type'].values
    y[y == 0] = -1

    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    cv_grad = MyLogisticRegression(X_train, y_train, X_test, y_test).fit()
    cv_fgrad = MyLogisticRegression(X_train, y_train, X_test, y_test).fit(algo='fgrad')
    cv_scikt = LogisticRegression(fit_intercept=False, C=.1667, solver='saga', max_iter=500).fit(X_train, y_train)

    return cv_grad, cv_fgrad, cv_scikt


cv_grad, cv_fgrad, cv_scikt = exercise_1()

if __name__ == '__main__':
    exercise_1()
