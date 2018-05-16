from myml.my_multiclassifier import MultiClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ex1():
    train = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train')
    test  = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test')

    x_train = train.iloc[:, 2:]
    x_test  = test.iloc[:, 2:]

    y_train = train.iloc[:, 1]
    y_test  = test.iloc[:, 1]

    scalar  = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test  = scalar.transform(x_test)

    parameters = {
        'classifiers': [
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss':  ['smoothed_hinge'],
                    'h':     [0.5],
                    'algo':  ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps':   [.001],
                    'eta':   [1.],
                    'lambda':   [1.],
                    'max_iter': [100],
                    't_eta':    [0.8]
                }
            }
        ]
    }

    MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                    x_val=x_test, y_val=y_test, n_jobs=-1,
                    classification_method='all_pairs', task='ex1a',
                    log_path='.', logging_level='reduced').fit()

    # use cross val to find the optimal value of lambda
    parameters['classifiers'][0]['parameters']['lambda'] = list(np.linspace(0.001, 1., 10))
    MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                    x_val=x_test, y_val=y_test, n_jobs=-1,
                    classification_method='all_pairs', task='ex1b',
                    log_path='.', logging_level='reduced').fit()


if __name__ == '__main__':
    ex1()