from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# misc setup for readability
norm = np.linalg.norm
exp  = np.exp
log  = np.log


#def exercise_1():
spam = pd.read_csv('spam.csv', sep=',').dropna()
X = spam.loc[:, spam.columns != 'type']
y = spam.loc[:, 'type'].values
y[y == 0] = -1

scalar = StandardScaler().fit(X)
X = scalar.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def inner_sum(x, y, beta):
    return log(1 + exp(-y[0] * x.T @ beta))-1
vsum = np.vectorize(inner_sum, signature='(x),(y),(z)->()')


def objective(x, y, beta, l):
    vec = vsum(x=x, y=y, beta=beta)
    return np.sum(vec)/len(y) + l*norm(beta)**2


def print_status(status):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(status)


def computegrad(x, y, l, b):
    u = np.diag([(1+exp(yi * xi @ b))**-1 for xi, yi in zip(x, y)])
    return 2*l*norm(b) - (x.T @ u @ y)/len(y)


def backtracking(x, y, beta, l, t=1., eta_t=0.5, alpha=0.5, max_iter=1000):
    grad_x = computegrad(x, y, l, beta)
    found_t, i = False, 0

    while not found_t and i < max_iter:
        a = objective(x + t * grad_x, y, beta, l)
        b = objective(x, y, beta, l) + alpha * t * norm(grad_x)**2

        if a < b:
            found_t = True
        elif i == max_iter-1:
            raise Exception("max number of backtracking iterations reached")
        else:
            t *= eta_t
            i += 1
    return t


def fastgradalgo(x, y, t_init, b_init, l=None, eps=0.001):
    if l is None:
        l = 1 / len(x)

    betas = [np.zeros(x.shape[1])]
    thtas = [np.zeros(x.shape[1])]
    grads = [computegrad(x, y, l, betas[-1])]
    xvals = [objective(x, y, betas[-1], l)]

    i = 0
    while norm(grads[-1]) > eps:
        t = backtracking(xvals[-1], betas[-1], l, b_init, t_init)

        betas.append(thtas[-1] - t * grads[-1])
        thtas.append(betas[-1] + (i/(i+3)) * (betas[-1] - betas[-2]))
        grads.append(computegrad(x, y, l, betas[-1]))

        xvals.append(objective(xvals[-1], y, betas[-1], l))
        print_status(xvals[-1])

    return xvals, betas[-1]


def graddescent(x, y, t_init, l=None, eps=0.001):
    if l is None:
        l = 1/len(x)

    betas = [np.ones(x.shape[1])]
    grads = [computegrad(x, y, l, betas[0])]
    xvals = [x]

    i = 0
    while norm(grads[-1]) > eps:
        #t = backtracking(x=xvals[-1], y=y, beta=betas[-1], l=l, t=t_init)
        t=.1
        betas.append(betas[-1] - t * grads[-1])
        grads.append(computegrad(xvals[-1], y, l, betas[-1]))

        #xvals.append(x - t * grads[-1])
        print_status("i: %s \tt: %s \t obj: %s" % (i, t, objective(xvals[-1], y, betas[-1], l)))
        i+=1
    return xvals, betas[-1]


def calc_t_init(x, l):
    return np.float(max(np.linalg.eig(1/len(x) * x@x.T + l)[0]))

# TODO: there is a lower bound of 1 appearing during training. what in the math hell is causing this
# TODO: this could be linked to the backtracking method shooting towards infinity
# TODO: and it isn't t because t only seems to change the rate at witch the event occurs

t_init = 1. #calc_t_init(X_train, .01)
fx, bt = graddescent(X_train, y_train, t_init, l=.05)
#x = X_train;y=y_train;l=0.01;b_init=.5;t=34.;beta=np.ones(x.shape[1]);alpha=0.5;eta_t=.5
