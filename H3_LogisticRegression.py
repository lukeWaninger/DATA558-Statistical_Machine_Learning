from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# misc setup for readability
norm = np.linalg.norm
exp  = np.exp
log  = np.log


#def exercise_1():
spam = pd.read_csv('spam.csv', sep=',').dropna()
X = pd.get_dummies(spam, drop_first=True)
y = spam.type

scalar = StandardScaler.fit(X)
X = scalar.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def objective(x, y, b, l):
    return 1/len(y) * (sum([log(1+exp(-yi*x.T@b)) + l*norm(b)**2 for yi in y]))


def compute_grad():
    pass


def backtrack():
    pass


def fastgradalgo():
    pass


def graddescent():
    pass

