from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# misc setup for readability
norm = np.linalg.norm
exp  = np.exp
log  = np.log


#def exercise_1():
spam = pd.read_csv('spam.csv', sep=',').dropna()
X = pd.get_dummies(spam, drop_first=True)
y = spam.type

scalar = StandardScaler.fit(X)
X = scalar.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def objective(x, y, beta, l):
    return 1/len(y) * (sum([log(1+exp(-yi*x.T@beta)) + l*norm(beta)**2 for yi in y]))


def computegrad(x, y, l, beta):
    p = 1-(1/(1+exp(y@x.T@beta)))
    return 1/len(y) * x@p@y + 2*l*beta


def backtracking(beta, beta1, t):
    return beta1+(t/(t+3))*(beta1-beta)


def fastgradalgo():
    pass


def graddescent(x, y, l, eta, eps):
    betas  = [np.zeros(x.shape[1])]
    thetas = [np.zeros(x.shape[1])]
    grads  = [computegrad(x, y, l, betas[-1])]
    xvals  = [objective(x, y, betas[-1], l)]
    t = 0

    while norm(grads[-1]) > eps:
        betas.append(thetas[-1]-eta * grads[x])
        xvals.append(objective(x, y, betas[-1], l))

        thetas.append(backtracking(betas[-2], betas[-1], t))
        t += 1



