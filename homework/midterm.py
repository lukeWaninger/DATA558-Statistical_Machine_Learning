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
                'lambda': [0.0001, 0.001, 0.01],
                'eta': [.001],
                'init_method': ['zeros'],
            }
        }
    ],
}

# filter classes
# train_idx = np.concatenate((np.where(y_train == 1)[0], np.where(y_train == 2)[0]))
# x_train = x_train[train_idx, :]
# y_train = y_train[train_idx]
#
# val_idx = np.concatenate((np.where(y_val == 1)[0], np.where(y_val == 2)[0]))
# x_val = x_val[val_idx, :]
# y_val = y_val[val_idx]

scalar  = StandardScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_val   = scalar.transform(x_val)
x_test  = scalar.transform(x_test)

clf = MultiClassifier(x_train=x_train, y_train=y_train, parameters=parameters,
                      x_val=x_val, y_val=y_val, n_jobs=1,
                      classification_method='all_pairs',
                      log_path=log_path,
                      logging_level='reduced')
clf.fit()
clf.output_predictions(x_test)
