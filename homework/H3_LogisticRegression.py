from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# misc setup for readability
norm = np.linalg.norm
exp = np.exp
log = np.log


class MyLogisticRegression:
    def __init__(self, X_train, y_train, lamda=0.1, max_iter=500, eps=0.001):
        self._betas = None
        self._eps = eps
        self._lamda = lamda
        self._max_iter = max_iter

        self.__x = X_train
        self.__y = y_train
        self.__n, self.__d = X_train.shape

        self.__eta = self.__calc_t_init()
        self.__objective_vals = None
        self.__training_errors = None
        self.__thetas = None

    @property
    def coef_(self):
        return self._betas[-1].reshape(1, 57) if self._betas is not None else []

    @property
    def objective_vals_(self):
        if self.__objective_vals is not None:
            return self.__objective_vals
        else:
            self.__objective_vals = [self.__objective(b) for b in self._betas] if self._betas is not None else []
            return self.__objective_vals

    @property
    def training_errors_(self):
        if self.__training_errors is not None:
            return self.__training_errors
        else:
            self.__training_errors = []

            for b in self._betas:
                pre = self.predict(self.__x, b)
                acc = np.sum([1 if yh == yt else 0 for yh, yt in zip(pre, self.__y)]) / self.__n
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

        self._betas = init(init_method)
        self.__objective_vals = None
        self.__training_errors = None

        if algo == 'grad':
            self.__graddescent()
        elif algo == 'fgrad':
            self._betas.append(self._betas[-1])
            self.__thetas = init(init_method)[0]
            self.__fastgradalgo()
            self._betas = self._betas[1:]
        else:
            raise Exception("algorithm <%s> is not available" % algo)

        self._betas = self._betas[1:]
        return self

    def predict(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return [1 if xi @ b > 0 else -1 for xi in x]

    def objective(self, coef):
        return self.__objective(coef)

    # private methods
    def __backtracking(self, beta, t_eta=0.5, alpha=0.5):
        l, t = self._lamda, self.__eta

        gb = self.__computegrad(beta)
        n_gb = norm(gb)

        found_t, i = False, 0
        while not found_t and i < self._max_iter:
            if self.__objective(beta - t*gb) < self.__objective(beta) - alpha * t * n_gb**2:
                found_t = True
            elif i == self._max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        self.__eta = t
        return self.__eta

    def __calc_t_init(self):
        x, l, n = self.__x, self._lamda, self.__n

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __computegrad(self, b):
        x, y, l, n = self.__x, self.__y, self._lamda, self.__n

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
        theta = self.__thetas
        grad = self.__computegrad(theta)

        i = 0
        while norm(grad) > self._eps and i < self._max_iter:
            b0 = self._betas[-1]
            t  = self.__backtracking(b0)
            grad = self.__computegrad(theta)

            b1 = theta - t*grad
            self._betas.append(b1)

            theta = b1 + (i/(i+3))*(b1-b0)
            i += 1

    def __objective(self, beta):
        x, y, n, l = self.__x, self.__y, self.__n, self._lamda

        return np.sum([log(1 + exp(-yi * xi.T @ beta)) for xi, yi in zip(x, y)]) / n + l * norm(beta) ** 2

    def __repr__(self):
        return "MyLogisticRegression(C=%s, eps=%s, max_iter=%s)" % (self._lamda, self._eps, self._max_iter)


def exercise_12():
    try:
        spam = pd.read_csv('data/spam.csv', sep=',').dropna()
    except FileNotFoundError as e:
        spam = pd.read_csv('spam.csv', sep=',').dropna()

    X = spam.loc[:, spam.columns != 'type']
    y = spam.loc[:, 'type'].values
    y[y == 0] = -1

    # scale the data
    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)

    # train classifiers using both algorithms and scikit's
    method = 'zeros'
    cv_grad  = MyLogisticRegression(X_train, y_train).fit(init_method=method)
    cv_fgrad = MyLogisticRegression(X_train, y_train).fit(algo='fgrad', init_method=method)
    cv_scikt = LogisticRegression(fit_intercept=False,
                                  C=10/X_train.shape[0],
                                  solver='sag',
                                  max_iter=1000).fit(X_train, y_train)

    def visualize_objective_comparison(grad, fgrad):
        plt.clf()
        plt.figure(figsize=(8, 6))

        pgrad,  = plt.plot(grad, label='grad')
        pfgrad, = plt.plot(fgrad, label='fgrad')
        plt.legend(handles=[pgrad, pfgrad], fontsize=14)
        plt.title('Gradient vs FastGradient Descent', fontsize=16)
        plt.ylabel('objective value', fontsize=14)
        plt.xlabel('iteration', fontsize=14)

    visualize_objective_comparison(cv_grad.objective_vals_, cv_fgrad.objective_vals_)

    def visualize_beta_comparison(betas):
        i = 1
        for beta, label, color in betas:
            plt.bar(np.arange(len(beta)), beta, width=.75, alpha=.6, label=label, color=color)
            i += 1

        plt.ylabel(r'$\beta$', fontsize=14)
        plt.xlabel(r'$\beta_i$', fontsize=14)
        plt.title(r'$\beta_T$ vs. $\beta^*$', fontsize=16)
        plt.legend(fontsize=14)
        plt.axis([0, 57, min([min(b[0]) for b in betas])-.05, max([max(b[0]) for b in betas]) + 0.05])
        plt.xticks(np.arange(0, 57, step=5))

    visualize_beta_comparison([(cv_scikt.coef_[0], 'scikit', '#3B3A38'),
                               (cv_fgrad.coef_[0], 'fgrad', '#FFAD00')])

    # grid search to find the optimal regularization coefficient
    cv = LogisticRegression(fit_intercept=False, max_iter=5000)
    parameters = {'C': np.linspace(.001, 2.0, 20)}
    gs = GridSearchCV(cv, parameters, scoring='neg_log_loss', n_jobs=-1).fit(X_train, y_train)
    print(gs.best_estimator_)

    # retrain my own classifiers with the optimal regularization coefficient
    cv_grad = MyLogisticRegression(X_train, y_train, lamda=gs.best_estimator_.C).fit(init_method=method)
    cv_fgrad = MyLogisticRegression(X_train, y_train, lamda=gs.best_estimator_.C).fit(algo='fgrad', init_method=method)
    visualize_objective_comparison(cv_grad.objective_vals_, cv_fgrad.objective_vals_)

    # plot training error vs iteration
    plt.clf()
    plt.plot(cv_grad.training_errors_, label='grad')
    plt.plot(cv_fgrad.training_errors_, label='fgrad')
    plt.title('Training error vs. iteration', fontsize=18)
    plt.ylabel('error', fontsize=16)
    plt.xlabel('iteration', fontsize=16)
    plt.legend()

    # plot test error vs iteration
    pred_grad  = [cv_grad.predict(X_test, b) for b in cv_grad._betas]
    pred_fgrad = [cv_grad.predict(X_test, b) for b in cv_fgrad._betas]
    err_grad  = [1-np.sum([1 if yh == yt else 0 for yh, yt in zip(p, y_test)])/X_test.shape[0] for p in pred_grad]
    err_fgrad = [1-np.sum([1 if yh == yt else 0 for yh, yt in zip(p, y_test)])/X_test.shape[0] for p in pred_fgrad]

    plt.clf()
    plt.plot(err_grad, label='grad')
    plt.plot(err_fgrad, label='fgrad')
    plt.title('Testing missclassification error vs. iteration', fontsize=18)
    plt.ylabel('error', fontsize=16)
    plt.xlabel('iteration', fontsize=16)
    plt.legend()

    # EXERCISE TWO
    lamdas = np.linspace(0, 2.5, 100)
    cvs_train = [MyLogisticRegression(X_train, y_train, lamda=l).fit(init_method='normal') for l in lamdas]
    errors = [c.training_errors_[-1] for c in cvs_train]

    # plot missclassification training error as lambda increases
    plt.clf()
    plt.scatter(lamdas, errors, alpha=0.3)
    lowess = sm.nonparametric.lowess(errors, lamdas, frac=1./5)
    plt.plot(lamdas, lowess[:, 1], label='train')
    plt.grid()
    plt.title('Train/test missclassification error steadily increases with $\lambda$', fontsize=18)
    plt.xlabel('$\lambda$', fontsize=16)
    plt.ylabel('error', fontsize=16)

    # plot for testing errors
    pred = [c.predict(X_test, c.coef_[0]) for c in cvs_train]
    errors = [1-np.sum([1 if yh == yt else 0 for yh, yt in zip(p, y_test)])/X_test.shape[0] for p in pred]
    plt.scatter(lamdas, errors, alpha=0.3)
    lowess = sm.nonparametric.lowess(errors, lamdas, frac=1./5)
    plt.plot(lamdas, lowess[:, 1], label='test')
    plt.grid()
    plt.legend(fontsize=16)
