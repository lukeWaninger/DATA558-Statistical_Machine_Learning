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

    # parameters = {
    #     'classifiers': [
    #         {
    #             'type': 'linear_svm',
    #             'parameters': {
    #                 'loss':  ['smoothed_hinge'],
    #                 'h':     [0.5],
    #                 'algo':  ['fgrad'],
    #                 'alpha': [0.5],
    #                 'bt_max_iter': [50],
    #                 'eps':   [.001],
    #                 'eta':   [1.],
    #                 'lambda':   [1.],
    #                 'max_iter': [100],
    #                 't_eta':    [0.8]
    #             }
    #         }
    #     ]
    # }
    #
    # cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
    #                      x_val=x_test, y_val=y_test, n_jobs=-1,
    #                      classification_method='all_pairs', task='ex1a',
    #                      log_path='.', logging_level='reduced').fit()
    #
    # predictions = cv.predict(x_val)
    # error = 1-np.mean(predictions == y_val)
    # with open('ex1.txt', 'a+') as f:
    #     f.write(f'validation error when labmda = 1: {str(error)}\n')

    # use cross val to find the optimal value of lambda
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
                    'lambda':   list(np.linspace(0.001, 1., 35)),
                    'max_iter': [100],
                    't_eta':    [0.8]
                }
            }
        ]
    }

    cv = MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                         x_val=x_test, y_val=y_test, n_jobs=-1,
                         classification_method='all_pairs', task='ex1b',
                         log_path='.', logging_level='reduced').fit()

    predictions = cv.predict(x_val)
    error = np.mean(predictions == y_val)

    with open('ex1.txt', 'a+') as f:
        f.write(f'validation error when lambda is found through cross validation: {str(error)}\n')


if __name__ == '__main__':
    ex1()