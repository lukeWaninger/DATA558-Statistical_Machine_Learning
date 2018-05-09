from myml.mlassoreg import MyLASSORegression
from myml.my_multiclassifier import MultiClassifier
import numpy as np
import os
import pandas as pd
import re
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#
# # EXERCISE 1
# hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv',
#                       sep=',', header=0).dropna()
# x = pd.get_dummies(hitters, drop_first=True)
# y = hitters.Salary
#
# # standardize and split the data
# x_scalar = StandardScaler().fit(x)
# y_scalar = StandardScaler().fit(y.values.reshape(-1,1))
#
# x = x_scalar.transform(x)
# y = y_scalar.transform(y.values.reshape((-1, 1))).reshape((-1))
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
#
# cv = Lasso(fit_intercept=False)
# parameters = {'alpha': np.linspace(0.0001, 5., 100)}
# gscv = GridSearchCV(cv, parameters, n_jobs=-1).fit(x_train, y_train)
# cv = Lasso(alpha=gscv.best_estimator_.alpha, fit_intercept=False).fit(x_train, y_train)
#
# log_path = 'homework/logs/ex1log.csv'
# np.random.seed(5)
# if os.path.exists(log_path):
#     os.remove(log_path)
#
# cyclic = MyLASSORegression(x_train, y_train, x_test, y_test,
#                            {
#                                'alpha': [gscv.best_estimator_.alpha],
#                                'max_iter': [1000],
#                                'algo': ['cyclic'],
#                                'log_path': ''
#                            },
#                            expected_betas=cv.coef_
#                            ).fit()
# random = MyLASSORegression(x_train, y_train, x_test, y_test,
#                            {
#                                'alpha': [gscv.best_estimator_.alpha],
#                                'max_iter': [1000],
#                                'algo': ['random'],
#                                'log_path': ''
#                            },
#                            expected_betas=cv.coef_
#                            ).fit()
#
#
# # EXERCISE 2
# def ex2(x, y):
#     x = np.array([[xi**p for p in range(10)] for xi in x])
#     x = StandardScaler().fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
#
#     cv = Lasso(fit_intercept=False)
#     alpha_space = np.linspace(.001, 5., 50)
#     parameters = {'alpha': alpha_space}
#     gscv = GridSearchCV(cv, parameters, scoring='neg_mean_squared_error', n_jobs=-1).fit(x_train, y_train)
#
#     pd.DataFrame(gscv.cv_results_).to_csv('homework/logs/ex2bresults.csv', index=None)
#     print(gscv.best_estimator_.alpha)
#     print(gscv.best_estimator_.coef_)
#
#
# # ex2 a, b
# x = np.random.normal(0, 1, 100)
# e = np.random.normal(0, .2, 100)
# b = [0, -.5, .5, .75]
# y = np.array([b[0] + b[1]*xi + b[2]*xi**2 + b[3]*xi**3 + ei for xi, ei in zip(x, e)])
# ex2(x, y)
#
# x = np.random.normal(0, 1, 100)
# e = np.random.normal(0, .2, 100)
# b = [0, -.5, .5, .75, -.75, .25, -.25, .82]
# y = np.array([b[0] + b[6]*xi**7 for xi in x])
# ex2(x, y)
#

# EXERCISE 3
p1 = re.sub(r'(homework)|(myml)|(data)+', '', os.getcwd()) + '/myml/data/'

x_train = np.load(p1 + 'train_features.npy')
y_train = np.load(p1 + 'train_labels.npy')
x_val = np.load(p1 + 'val_features.npy')
y_val = np.load(p1 + 'val_labels.npy')
x_test = np.load(p1 + 'test_features.npy')

log_path ='/mnt/hgfs/descent_logs/'
num_splits = 3
parameters = {
    'classifiers': [
        {
            'type': 'LASSO',
            'parameters': {
                'alpha': [2**i*1.0 for i in range(2, 2+num_splits)],
                'max_iter': [1000],
                'algo': ['random'],
                'log_path': log_path
            }
        }
    ],
}

scalar  = StandardScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_val   = scalar.transform(x_val)
x_test  = scalar.transform(x_test)

clf = MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                      x_val=x_val, y_val=y_val, n_jobs=5,
                      classification_method='all_pairs',
                      log_path=log_path,
                      logging_level='reduced')
clf.fit()
clf.output_predictions(x_test)
