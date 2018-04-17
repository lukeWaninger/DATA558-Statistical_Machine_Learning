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


#def exercise_1():
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


class LogisticRegression:
    def __init__(self, X_train, y_train, lamda=0.1):
        self.__x = X_train
        self.--y = y_train


def inner_sum(x, y, beta):
    return log(1 + exp(-y[0] * x.T @ beta))
vsum = np.vectorize(inner_sum, signature='(x),(y),(z)->()')


def objective(beta, l, x=X_train, y=y_train):
    vec = vsum(x=x, y=y, beta=beta)
    return np.sum(vec)/len(y) + l*norm(beta)**2


def print_status(status):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(status)


def computegrad(l, b, x=X_train, y=y_train):
    p = (1+exp(y * (x @ b)))**-1
    return 2*l*b - (x.T @ np.diag(p) @ y)/len(y)**2


def calc_t_init(lamda=0.01, x=X_train):
    m = np.max(1/x.shape[0] * np.linalg.eigvals(x@x.T)) + lamda
    return 1/np.float(m)


def backtracking(beta, lamda=0.01, t=1., eta=.5, alpha=1.5, max_iter=50):
    grad_beta = computegrad(lamda, beta)
    norm_grad_beta = norm(grad_beta)
    obj_grad_beta = objective(beta, lamda)
    found_t, i = False, 0

    while not found_t and i < max_iter:
        a = objective(beta=computegrad(l=lamda, b=beta - t * grad_beta), l=lamda)
        b = obj_grad_beta - alpha * t * norm_grad_beta ** 2

        print("%s %s %s" % (a, b, norm_grad_beta))
        if a < b:
            found_t = True
        elif i == max_iter-1:
            raise Exception("max number of backtracking iterations reached")
        else:
            t *= eta
            i += 1
    return t


def fastgradalgo(t, lamda=.01, eps=0.001, x=X_train):
    betas = [np.ones(x.shape[1])]
    thtas = [np.ones(x.shape[1])]
    grads = [computegrad(lamda, betas[0])]
    objective_vals = [objective(betas[-1], lamda)]

    i = 0
    while norm(grads[-1]) > eps:
        t = backtracking(beta=betas[-1], lamda=lamda, t=t)

        betas.append(thtas[-1] - t * grads[-1])
        thtas.append(betas[-1] + (i/(i+3)) * (betas[-1] - betas[-2]))
        grads.append(computegrad(lamda, betas[-1]))

        objective_vals.append(objective(betas[-1], lamda))
        print_status("i: %s \tt: %s \t obj: %s" % (i, t, objective_vals[-1]))

    return objective_vals, betas[-1]


def graddescent(t, lamda=.01, eps=0.001, x=X_train):
    betas = [np.ones(x.shape[1])]
    grads = [computegrad(lamda, betas[0])]
    objective_vals = [objective(betas[-1], lamda)]

    i = 0
    while norm(grads[-1]) > eps:
        t = backtracking(beta=betas[-1], lamda=lamda, t=t)

        betas.append(betas[-1] - t * grads[-1])
        grads.append(computegrad(lamda, betas[-1]))
        objective_vals.append(objective(betas[-1], lamda))

        print_status("i: %s \tt: %s \t obj: %s" % (i, t, objective_vals[-1]))
        i += 1
    return objective_vals, betas[-1]


t_init = 1.5 #.167 #calc_t_init(0.1)
#fx, bt = graddescent(t_init)
ffx, fbt = fastgradalgo(t_init)
#x = X_train;y=y_train;lamda=0.01;b_init=.5;beta=np.ones(x.shape[1]);alpha=0.5;eta_t=.5;b=beta;max_iter=100

plt.plot(fx, c='green')
plt.plot(ffx, c='blue')