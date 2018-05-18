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
    error = 1-np.mean(predictions == y_val)

    with open('ex1.txt', 'a+') as f:
        f.write(f'cv: {str(error)}\n')


def ex2_data(classes=None):
    x_train = np.load('./data/h6/train_features.npy')
    y_train = np.load('./data/h6/train_labels.npy')
    x_val = np.load('./data/h6/val_features.npy')
    y_val = np.load('./data/h6/val_labels.npy')
    x_test = np.load('./data/h6/test_features.npy')

    if classes is not None:
        idx = np.isin(y_train, classes)
        x_train = x_train[idx, :]
        y_train = y_train[idx]

        idx = np.isin(y_val, classes)
        x_val = x_val[idx, :]
        y_val = y_val[idx]

    scalar = StandardScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)
    x_val = scalar.transform(x_val)

    return x_train, y_train, x_val, y_val, x_test


def ex2a_ap():
    from sklearn.svm import LinearSVC
    from scipy.stats import mode

    x_train, y_train, x_val, y_val, x_test = ex2_data()

    # generate pairs
    labels = np.unique(y_train)
    pairs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            pairs.append((labels[i], labels[j]))

    # fit and predict for each pair
    predictions = []
    for pair in pairs:
        cv = LinearSVC()
        idx = np.isin(y_train, pair)

        cv = cv.fit(x_train[idx, :], y_train[idx])
        predictions.append(cv.predict(x_val))

    # take the modes
    predictions = np.array(predictions).T
    predictions = [mode(pi).mode[0] for pi in predictions]
    error = 1-np.mean(predictions == y_val)

    with open('ex2a.txt', 'a+') as f:
        f.write(f'all pairs: {str(error)}\n')


def ex2a_ovr():
    from sklearn.svm import LinearSVC

    x_train, y_train, x_val, y_val, x_test = ex2_data()
    cv = LinearSVC()
    cv = cv.fit(x_train, y_train)
    predictions = cv.predict(x_val)

    error = 1 - np.sum(1 for yh, yt in zip(predictions, y_val) if yh == yt) / len(predictions)
    with open('ex2a.txt', 'a+') as f:
        f.write(f'ovr: {str(error)}\n')


def ex2b_ap():
    x_train, y_train, x_val, y_val, x_test = ex2_data([28, 56])

    ex2b_ap_params = {
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
                    'lambda': [2048., 4096., 8192., 16384.],
                    'max_iter': [100],
                    't_eta': [0.8],
                }
            }
        ]
    }

    task = 'ex2b_ap'
    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=ex2b_ap_params,
                         x_val=x_val, y_val=y_val, n_jobs=5,
                         classification_method='all_pairs', task=task,
                         log_path='.', logging_level='none').fit()
    cv.output_predictions(x_test)

    predictions = cv.predict(x_val)
    error = 1-np.sum(1 for yh, yt in zip(predictions, y_val) if yh == yt)/len(predictions)
    with open('ex2.txt', 'a+') as f:
        f.write(f'{task}: {str(error)}\n')


def ex2b_ovr():
    x_train, y_train, x_val, y_val, x_test = ex2_data()

    ex2b_ovr_params = {
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
                    'lambda': [2048., 4096., 8192., 16384.],
                    'max_iter': [100],
                    't_eta': [0.8],
                }
            }
        ]
    }

    task = 'ex2b_ovr'
    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=ex2b_ovr_params,
                         x_val=x_val, y_val=y_val, n_jobs=3,
                         classification_method='all_pairs', task=task,
                         log_path='.', logging_level='none').fit()
    cv.output_predictions(x_test)

    predictions = cv.predict(x_val)
    error = 1 - np.sum(1 for yh, yt in zip(predictions, y_val) if yh == yt) / len(predictions)
    with open('ex2.txt', 'a+') as f:
        f.write(f'{task}: {str(error)}\n')


if __name__ == '__main__':
    #ex1()
    ex2a_ap()
    ex2a_ovr()
    #ex2b_ap()
    #ex2b_ovr()