from myml.my_multiclassifier import MultiClassifier
from myml.ml2hinge import MyL2Hinge
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ex1():
    try:
        spam = pd.read_csv('data/spam.csv', sep=',').dropna()
    except FileNotFoundError as e:
        spam = pd.read_csv('spam.csv', sep=',').dropna()

    x = spam.loc[:, spam.columns != 'type']
    y = spam.loc[:, 'type'].values
    y[y == 0] = -1

    # scale the data
    scalar = StandardScaler().fit(x)
    x = scalar.transform(x)

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=0)

    # noinspection PyDictCreation
    parameters = {
        'algo': ['fgrad'],
        'alpha': [0.5],
        'bt_max_iter': [50],
        'eps': [.001],
        'eta': [1.],
        'lambda': [1.],
        'max_iter': [100],
        't_eta': [0.8]
    }

    cv = MyL2Hinge(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test,
                   parameters=parameters, log_path='', logging_level='reduced',
                   task='ex1a')
    cv = cv.fit()
    del cv

    parameters['lambda'] = list(np.linspace(0.001, 1., 5))
    cv = MyL2Hinge(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test,
                   parameters=parameters, log_path='', logging_level='reduced',
                   task='ex1b')
    cv.fit()


def ex2():
    p1 = re.sub(r'(homework)|(myml)|(data)+', '', os.getcwd()) + '/myml/data/'

    x_train = np.load(p1 + 'train_features.npy')
    y_train = np.load(p1 + 'train_labels.npy')
    x_val = np.load(p1 + 'val_features.npy')
    y_val = np.load(p1 + 'val_labels.npy')
    x_test = np.load(p1 + 'test_features.npy')

    log_path = './'
    parameters = {
        'classifiers': [
            {
                'type': 'hinge',
                'parameters': {
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': [2 ** i + 0.001 for i in range(-5, 4, 1)],
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            }
        ]
    }

    ## filter classes
    # train_idx, val_idx = [], []
    # for k in [1, 2]:
    #     train_idx = np.concatenate((train_idx, np.where(y_train == k)[0]))
    #     val_idx = np.concatenate((val_idx, np.where(y_val == k)[0]))
    #
    # train_idx = [int(i) for i in train_idx]
    # val_idx = [int(i) for i in val_idx]
    #
    # x_train = x_train[train_idx, :]
    # y_train = y_train[train_idx]
    #
    # x_val = x_val[val_idx, :]
    # y_val = y_val[val_idx]

    # scale data
    scalar  = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_val   = scalar.transform(x_val)
    x_test  = scalar.transform(x_test)

    # train
    clf = MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                          x_val=x_val, y_val=y_val, n_jobs=-1,
                          classification_method='all_pairs',
                          log_path=log_path,
                          logging_level='reduced')
    clf.fit()
    clf.output_predictions(x_test)


if __name__ == '__main__':
    ex1()
    #ex2()