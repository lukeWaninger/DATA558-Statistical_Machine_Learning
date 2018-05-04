from kaggle.my_multiclassifier import MultiClassifier
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler

p1 = re.sub(r'(homework)|(kaggle)|(data)+', '', os.getcwd()) + '/kaggle/data/'

x_train = np.load(p1 + 'train_features.npy')
y_train = np.load(p1 + 'train_labels.npy')
x_val = np.load(p1 + 'val_features.npy')
y_val = np.load(p1 + 'val_labels.npy')
x_test = np.load(p1 + 'test_features.npy')

log_path ='/mnt/hgfs/descent_logs/'
num_splits = 6
parameters = {
    'classifiers': [
        {
            'type': 'hinge',
            'parameters': {
                'lambda': list(np.linspace(0.0001, 0.01, 5)),
                'eta': [.001],
                'init_method': ['zeros'],
            }
        },
        # {
        #     'type': 'lasso',
        #     'parameters': {
        #         'alpha': [0.25, 0.5, 2.0, 4.0],
        #         'max_iter': [1000],
        #         'algo': ['random']
        #     }
        # },
        # {
        #     'type': 'ridge',
        #     'parameters': {
        #         'init_method': ['zeros'],
        #         'lambda': [10**-i for i in range(5)],
        #         'eta': [1.],
        #         'max_iter': [500]
        #     }
        # },
        # {
        #     'type': 'logistic',
        #     'parameters': {
        #         'init_method': ['zeros'],
        #         'lambda': [0.01, 0.1],
        #         'eps': [0.001],
        #         'max_iter': [500],
        #         'algo': ['fgrad'],
        #         'alpha': [0.5],
        #         'eta': [1.],
        #         't_eta': [0.5],
        #         'bt_max_iter': [50]
        #     }
        # }
    ],
}

# filter classes
train_idx, val_idx = [], []
for k in [1, 2, 3]:
    train_idx = np.concatenate((train_idx, np.where(y_train == k)[0]))
    val_idx = np.concatenate((val_idx, np.where(y_val == k)[0]))

train_idx = [int(i) for i in train_idx]
val_idx = [int(i) for i in val_idx]

x_train = x_train[train_idx, :]
y_train = y_train[train_idx]

x_val = x_val[val_idx, :]
y_val = y_val[val_idx]

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
