from sklearn.model_selection import train_test_split
from sklearn.linear_model.ridge import Ridge
from sklearn.preprocessing import *
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np


# misc setup
norm = np.linalg.norm
log  = np.log
np.random.seed(42)


def exercise_one():
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
        return 2/len(y) * (norm((y-x@beta)**2)+l*norm(beta)**2)

    def compute_grad(x, y, beta, l):
        return 2/len(y) * (x.T@x@beta + l*beta - x.T@y)

    def grad_descent(x, y, l, eta, max_iter):
        beta = np.zeros(x.shape[1])
        i, xvals = 0, []
        grad_x = compute_grad(x, y, beta, l)

        while i < max_iter:
            beta = beta - eta * grad_x
            xvals.append(objective(x, y, beta, l))
            grad_x = compute_grad(x, y, beta, l)
            i += 1

        return xvals, beta

    fx, bt = grad_descent(X_train, y_train, l=0.1, eta=0.1, max_iter=1000)

    # compare with sklearn's ridge
    clf = Ridge(alpha=0.1, max_iter=1000, solver='saga').fit(X_train, y_train)

    # plot the object function vs iteration number
    plt.plot(fx)
    plt.title("Objective function rapidly decreases")
    plt.xlabel("iterations (t)")
    plt.ylabel(r'$F(\beta)$')

    # calculate the difference in objective values between sklearn's and my own descent
    objective(X_train, y_train, bt, 0.1) - objective(X_train, y_train, clf.coef_, 0.1)
    # -1.21634123961e-05

    # visualize the comparison
    def visualize(betas, sklb):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

        plt.subplot(ax1)
        plt.bar(np.arange(len(betas)), betas, width=.5)
        plt.bar(np.arange(len(betas))+0.5, sklb, width=.5)
        plt.ylabel(r'$\beta$')
        plt.xlabel(r'$\beta_i$')
        plt.axis([0,20,-.05,max(max(betas), max(sklb))+0.1])
        plt.xticks(np.arange(0, 20, step=2))

        plt.subplot(ax2)
        plt.bar(np.arange(len(betas)), (betas-sklb))
        plt.ylabel(r'$\Delta \, \beta$')
        plt.xlabel(r'$\beta_i$')
        plt.xticks(np.arange(0, 20, step=2))

        st = plt.suptitle(r'Resulting $\beta$ values are quite similar', fontsize=16)
        st.set_y(.95)
        fig.subplots_adjust(wspace=.3, top=.85)

    visualize(bt, clf.coef_)

    runs = [(1/10**n, grad_descent(X_train, y_train, l=0.1, eta=1/10**n, max_iter=1000)) for n in np.linspace(1, 5, 10)]
    plt.clf()
    [plt.plot(r[1][0]) for r in runs]
    plt.title("Objective values per iteration")

    best_idx = np.argmin([min(r[1][0]) for r in runs])
    bt = runs[best_idx][1][1]
    visualize(bt, clf.coef_)

    # calculate the difference in objective values between sklearn's and my own descent
    objective(X_train, y_train, bt, 0.1) - objective(X_train, y_train, clf.coef_, 0.1)
    # -1.22808378832e-05


def exercise_two():
    auto = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Auto.csv',
                       sep=',', header=0, na_values='?').dropna()

    model = ols(formula='mpg ~ horsepower', data=auto).fit()
    model.summary()

    fig, ax = plt.subplots()
    sm.graphics.plot_fit(model, 1, ax=ax)

    plt.clf()
    plt.scatter(model.fittedvalues, model.resid, alpha=.7, c='#2F3336')
    plt.title('Residuals vs. fitted values', fontsize=16)
    plt.ylabel("fitted value", fontsize=14)
    plt.xlabel("residual", fontsize=14)


def exercise_three():
    auto = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Auto.csv',
                           sep=',', header=0, na_values='?').dropna()

    pd.scatter_matrix(auto)
    auto.corr()

    formula = 'mpg ~' + ' + '.join(auto.columns[1:-1])
    model = ols(formula=formula, data=auto).fit()

    plt.clf()
    plt.scatter(model.fittedvalues, model.resid, alpha=.7, c='#2F3336')
    plt.title('Residuals vs. fitted values', fontsize=16)
    plt.ylabel("fitted value", fontsize=14)
    plt.xlabel("residual", fontsize=14)

    model = ols(formula='year ~ mpg + weight + cylinders + mpg*weight + mpg*displacement + mpg*cylinders',
                data=auto).fit()
    model.summary()

    auto['mpg_log_t'] = log(auto.mpg)
    formula = 'mpg_log_t ~' + ' + '.join(auto.columns[1:-1])
    model = ols(formula=formula, data=auto).fit()
    model.summary()

    fig, ax = plt.subplots()
    sm.graphics.plot_fit(model, 1, ax=ax)

    plt.clf()
    plt.scatter(model.fittedvalues, model.resid, alpha=.7, c='#2F3336')
    plt.title('Residuals vs. fitted values (log transformed response)', fontsize=16)
    plt.ylabel("fitted value", fontsize=14)
    plt.xlabel("residual", fontsize=14)


def exercise_four():
    pass