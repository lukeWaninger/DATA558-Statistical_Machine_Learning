from myml.my_multiclassifier import MultiClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ex1():
    train = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train')
    test  = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test')

    train = train.sample(frac=1, random_state=1).reset_index(drop=True)
    test  = test.sample(frac=1, random_state=1).reset_index(drop=True)

    # filter classes
    # train = train.loc[(train['y'] == 1) | (train['y'] == 2) | (train['y'] == 3)]
    # test  = test.loc[(test['y'] == 1) | (test['y'] == 2) | (test['y'] == 3)]

    val_idx = int(test.shape[0]*.7)

    x_train = train.iloc[:, 2:]
    x_test  = test.iloc[:val_idx, 2:]
    x_val   = test.iloc[val_idx:, 2:]

    y_train = train.iloc[:, 1]
    y_test  = test.iloc[:val_idx, 1]
    y_val   = test.iloc[val_idx:, 1]

    scalar  = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test  = scalar.transform(x_test)
    x_val   = scalar.transform(x_val)

    ex1_parameters = {
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

    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=ex1_parameters,
                         x_val=x_test, y_val=y_test, n_jobs=-1,
                         classification_method='all_pairs', task='ex1a',
                         log_path='.', logging_level='reduced').fit()

    predictions = cv.predict(x_val)
    error = 1-np.mean(predictions == y_val)
    with open('ex1.txt', 'a+') as f:
        f.write(f'ex1a error: {str(error)}\n')

    # use cross val to find the optimal value of lambda
    pset51 = {
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
                    'lambda':   list(np.linspace(0.001, 1., 25)),
                    'max_iter': [100],
                    't_eta':    [0.8]
                }
            }
        ]
    }

    pset46 = {
        'classifiers': [
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.001, .01, 10)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            },
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.01, .1, 10)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            },
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.1, 1., 10)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            }
        ]
    }

    pset60 = {
        'classifiers': [
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.001, 1., 25)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            },
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['squared_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.001, 1., 25)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            }
        ]
    }

    ex1b_parameters = {
        'classifiers': [
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.001, .1, 25)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            },
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': list(np.linspace(0.1, 1., 25)),
                    'max_iter': [100],
                    't_eta': [0.8]
                }
            }
        ]
    }

    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=ex1b_parameters,
                         x_val=x_test, y_val=y_test, n_jobs=-1,
                         classification_method='all_pairs', task='ex1b',
                         log_path='.', logging_level='reduced').fit()

    predictions = cv.predict(x_val)
    error = np.mean(predictions == y_val)

    with open('ex1.txt', 'a+') as f:
        f.write(f'cv: {str(error)}\n')


def ex2():
    x_train = np.load('./data/h6/train_features.npy')
    y_train = np.load('./data/h6/train_labels.npy')
    x_val   = np.load('./data/h6/val_features.npy')
    y_val   = np.load('./data/h6/val_labels.npy')
    x_test  = np.load('./data/h6/test_features.npy')

    # select a small subset of classes for proof of concept
    test_classes = [0, 1, 2]
    idx = np.isin(y_train, test_classes)

    x_train = x_train[idx, :]
    y_train = y_train[idx]

    idx   = np.isin(y_val, test_classes)
    x_val = x_val[idx, :]
    y_val = y_val[idx]

    ex2a_parameters = {
        'classifiers': [
            {
                'type': 'linear_svm',
                'parameters': {
                    'loss': ['smoothed_hinge'],
                    'h': [0.5],
                    'algo': ['fgrad'],
                    'alpha': [0.5],
                    'bt_max_iter': [50],
                    'eps': [.001],
                    'eta': [1.],
                    'lambda': [1.],
                    'max_iter': [2],
                    't_eta': [0.8]
                }
            }
        ]
    }

    task = 'ex2a'
    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=ex2a_parameters,
                         x_val=x_val, y_val=y_val, n_jobs=3,
                         classification_method='all_pairs', task=task,
                         log_path='.', logging_level='reduced').fit()

    cv.output_predictions(x_test)

    predictions = cv.predict(x_val)
    error = np.mean(predictions == y_val)

    with open('ex2.txt', 'a+') as f:
        f.write(f'{task}: {str(error)}\n')


if __name__ == '__main__':
    #ex1()
    ex2()