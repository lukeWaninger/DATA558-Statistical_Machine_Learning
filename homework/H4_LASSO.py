from kaggle.mlassoreg import MyLASSORegression
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv',
                      sep=',', header=0).dropna()
x = pd.get_dummies(hitters, drop_first=True)
y = hitters.Salary

# standardize and split the data
x_scalar = StandardScaler().fit(x)
y_scalar = StandardScaler().fit(y.values.reshape(-1,1))

x = x_scalar.transform(x)
y = y_scalar.transform(y.values.reshape((-1, 1))).reshape((-1))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


cv = Lasso(fit_intercept=False)
parameters = {'alpha': np.linspace(0.0001, 5., 100)}
gscv = GridSearchCV(cv, parameters, n_jobs=-1).fit(x_train, y_train)
cv = Lasso(alpha=gscv.best_estimator_.alpha, fit_intercept=False).fit(x_train, y_train)

cyclic = MyLASSORegression(x_train, y_train, x_test, y_test,
                           lamda=gscv.best_estimator_.alpha,
                           max_iter=1000,
                           task='cyclic_cd',
                           expected_betas=cv.coef_,
                           log_path='homework/logs/').fit(algo='cyclic')
random = MyLASSORegression(x_train, y_train, x_test, y_test,
                           lamda=gscv.best_estimator_.alpha,
                           task='random_cd',
                           max_iter=1000,
                           expected_betas=cv.coef_,
                           log_path='homework/logs/').fit(algo='random')
