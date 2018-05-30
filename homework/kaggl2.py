import datetime as dt
import numpy  as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


np.random.seed(42)
x_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
x_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
x_test = np.load('test_features.npy')


def log_it(message, preds, cvname):
    with open('log.txt', 'a+') as f:
        f.write(f'{dt.datetime.now().isoformat()} {message}\n')

    pd.DataFrame(preds).to_csv(f'{cvname}.csv', index=None)


# standardize
scalar  = StandardScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_val   = scalar.transform(x_val)
x_test  = scalar.transform(x_test)

# first run PCA because I don't have time to use all the features
pca = PCA(n_components=1000).fit(x_train)
x_sub = pca.transform(x_train)
v_sub = pca.transform(x_val)


# parameters to test
rbf_params = [
    {
        'base_estimator__kernel': ['rbf'],
        'base_estimator__degree': [1e-3, 1e-5],
        'base_estimator__C': [100]
    }
]

poly_params = [
    {
        'base_estimator__kernel': ['poly'],
        'base_estimator__degree': [1, 2],
        'base_estimator__C': [100]
    }
]

sig_params = [
    {
        'base_estimator__kernel': ['sigmoid'],
        'base_estimator__gamma': [1e-3, 1e-5],
        'base_estimator__C': [100]
    },
]

base_est = SVC()

# RBF
print(f'\n\n\nfitting radial rbf\n\n\n')
rbf_bag = BaggingClassifier(base_estimator=base_est, max_samples=1/10, n_estimators=15, n_jobs=-1)
rbf_cv  = GridSearchCV(rbf_bag, rbf_params, verbose=10)
rbf_cv.fit(x_sub, y_train)
rbf_pred = rbf_cv.predict(v_sub)
rbf_acc = accuracy_score(y_val, rbf_pred)
log_it(f'rbf: {rbf_acc}\n-----------{rbf_cv.best_estimator_}\n{rbf_cv.best_params_}',
       rbf_pred, 'radial_bias')

# POLY
print(f'\n\n\nfitting polynomial rbf\n\n\n')
poly_bag = BaggingClassifier(base_estimator=base_est, max_samples=1/10, n_estimators=15, n_jobs=-1)
poly_cv  = GridSearchCV(poly_bag, poly_params, verbose=10)
poly_cv.fit(x_sub, y_train)
poly_pred = poly_cv.predict(v_sub)
poly_acc  = accuracy_score(y_val, poly_pred)
log_it(f'rbf: {rbf_acc}\n-----------{rbf_cv.best_estimator_}\n{rbf_cv.best_params_}',
       poly_pred, 'poly')

# SIGMOID
print(f'\n\n\nfitting sigmoid rbf\n\n\n')
sig_bag = BaggingClassifier(base_estimator=base_est, max_samples=1/10, n_estimators=15, n_jobs=-1)
sig_cv  = GridSearchCV(sig_bag, sig_params, verbose=10)
sig_cv.fit(x_sub, y_train)
sig_pred = sig_cv.predict(v_sub)
sig_acc  = accuracy_score(y_val, sig_pred)
log_it(f'rbf: {rbf_acc}\n-----------{rbf_cv.best_estimator_}\n{rbf_cv.best_params_}',
       sig_pred, 'sigmoid')

# WEIGHTED VOTING
weights = [rbf_acc, poly_acc, sig_acc]
weights = weights/np.sum(weights)
voter   = VotingClassifier(estimators=[rbf_cv, poly_cv, sig_cv], voting='soft', weights=weights, n_jobs=-1)
voter_pred = voter.predict(x_val)
voter_acc  = accuracy_score(y_val, voter_pred)
log_it(f'voting before refit: {voter_acc}', voter_pred, 'voting_pre_refit')

# RETRAIN FOR TEST PREDICTIONS
x_full = np.concatenate((x_sub, v_sub))
y_full = np.concatenate((y_train, y_val))

rbf_cv  = rbf_cv.fit(x_full, y_full, refit=True)
poly_cv = poly_cv.fit(x_full, y_full, refit=True)
sig_cv  = sig_cv.fit(x_full, y_full, refit=True)
voter   = VotingClassifier(estimators=[rbf_cv, poly_cv, sig_cv], voting='soft', weights=weights, n_jobs=-1)

kaggle_predictions = voter.predict(x_test)
log_it(f'voting before after: unk', kaggle_predictions, 'voting_post_refit')
