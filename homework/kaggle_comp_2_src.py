import datetime as dt
import itertools as it
import numpy  as np
from multiprocessing.dummy import Pool as threadPool
import multiprocessing
import re
from scipy.special import comb
from scipy.stats import mode
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


exp   = np.exp
na    = np.newaxis
norm  = np.linalg.norm
ident = np.identity
np.random.seed(42)


x_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
x_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
x_test = np.load('test_features.npy')


