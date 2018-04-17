from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# misc setup for readability
norm = np.linalg.norm
exp  = np.exp
log  = np.log


class LogisticRegression:
    def __init__(self, X_train, y_train, lamda=0.01):
        # private
        self._x = X_train
        self._y = y_train
        self._eps = 0.001
        self._lamda = lamda
        self._n, self._d = X_train.shape
        self._eta = self.__calc_t_init()

        # public
        self.betas = None

        # function variables
        self.__vsum = np.vectorize(self.__inner_sum, signature='(x),(y),(z)->()')

    def fit(self, algo='grad'):
        self.betas = [np.ones(self._n)]

        if algo == 'grad':
            self.__graddescent()
        elif algo == 'fgrad':
            self.__fastgradalgo()
        else:
            raise Exception("algorithm <%s> is not avalailable" % algo)
        return self

    def objective_vals(self):
        return [self.__objective(b) for b in self.betas]

    @staticmethod
    def __inner_sum(x, y, beta):
        return log(1 + exp(-y[0] * x.T @ beta))

    @staticmethod
    def __print_status(status):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(status)

    def __calc_t_init(self):
        x, l, n = self._x, self._lamda, self._n

        m = np.max(1/n * np.linalg.eigvals(x @ x.T)) + l
        return 1/np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self._x, self._y, self._lamda, self._n

        p = (1 + exp(y * (x @ b))) ** -1
        return 2 * l * b - (x.T @ np.diag(p) @ y) / n ** 2

    def __objective(self, beta):
        sum, x, y, n, l = self.__vsum, self._x, self._y, self.n, self._lamda

        vec = sum(x=x, y=y, beta=beta)
        return (np.sum(vec) / n) + l * norm(beta) ** 2

    def __backtracking(self, beta, t_eta=.5, alpha=1.5, max_iter=50):
        l, t = self._lamda, self._eta

        gb   = self.__computegrad(beta)
        n_gb = norm(gb)
        o_gb = self.__objective(beta)

        found_t, i = False, 0
        while not found_t and i < max_iter:
            a = self.__objective(beta=self.__computegrad(b=beta - t * gb))
            b = o_gb - alpha * t * n_gb ** 2

            print("%s %s %s" % (a, b, n_gb))
            if a < b:
                found_t = True
            elif i == max_iter-1:
                raise Exception("max number of backtracking iterations reached")
            else:
                t *= t_eta
                i += 1

        self._eta = t
        return self._eta

    def __fastgradalgo(self):
        theta = np.ones(self._n)
        grad_x = self.__computegrad(self.betas[-1])

        i = 0
        while norm(grad_x) > self._eps and i < 500:
            b0, b1 = self.betas[-1], self.betas[-2]

            t = self.__backtracking(b0)

            self.betas.append(theta - t * grad_x)
            theta = b0 + (i/(i+3)) * (b0 - b1)
            grad_x = self.__computegrad(b0)

            o_val = self.__objective(b0)
            self.__print_status("i: %s \tt: %s \t obj: %s" % (i, t, o_val))

    def __graddescent(self):
        grad_x = self.__computegrad(self.betas[-1])

        i = 0
        while norm(grad_x) > self._eps and i < 500:
            b0 = self.betas[-1]
            t = self.__backtracking(b0)

            self.betas.append(b0 - t * grad_x)
            grad_x = self.__computegrad(b0)

            o_val = self.__objective(b0)
            self.__print_status("i: %s \tt: %s \t obj: %s" % (i, t, o_val))
            i += 1


def exercise_1():
    try:
        spam = pd.read_csv('data/spam.csv', sep=',').dropna()
    except:
        spam = pd.read_csv('spam.csv', sep=',').dropna()

    X = spam.loc[:, spam.columns != 'type']
    y = spam.loc[:, 'type'].values
    y[y == 0] = -1

    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    cv = LogisticRegression(X_train, y_train).fit()


if __name__ == '__main__':
    exercise_1()