from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np


hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0).dropna()
X = pd.get_dummies(hitters, drop_first=True)
y = hitters.Salary

# standardize and split the data
x_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y.values.reshape(-1,1))

X = x_scaler.transform(X)
y = y_scaler.transform(y.values.reshape((-1, 1))).reshape((-1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def objective(x, y, beta, l):
    return 1/len(y) * (np.linalg.norm((y-x.dot(beta))**2)+l*np.linalg.norm(beta)**2)


def compute_grad(x, y, beta, l):
    return 1/len(y) * -2*x.T.dot((y+x.dot(beta)))+2*l*beta


def grad_descent(x, y, l, eta, max_iter):
    beta = np.zeros(x.shape[1])
    i, x_vals = 0, [x]
    grad_x = compute_grad(x, y, beta, l)

    while i < max_iter:
        beta = beta + eta * grad_x
        x_vals.append(objective(grad_x, y, beta, l))
        grad_x = compute_grad(x, y, beta, l)
        i += 1

    return x_vals


fx = grad_descent(X_train, y_train, l=0.1, eta=0.1, max_iter=1000)

v = [i for i in range(1, len(fx)+1)]
fig = plt.figure()
plt.plot(v, fx)
plt.show()