import datetime as dt
import itertools as it
import numpy  as np
from multiprocessing.dummy import Pool as threadPool
import multiprocessing
import plotly.graph_objs as go
import plotly.offline as py
import re
from scipy.special import comb
from scipy.stats import mode
from tqdm import tqdm


exp   = np.exp
na    = np.newaxis
norm  = np.linalg.norm
ident = np.identity
np.random.seed(42)
py.init_notebook_mode(connected=True)


def backtracking(k, y, beta, l, eta, grad, obj, a=0.5, t_eta=0.8, max_iter=5):
    """backtracking line search using armijo stopping condition

    Args:
        k nXn (ndarray): compute kernel matrix
        y 1Xn (ndarray): true labels
        beta 1Xd (ndarray): weight coefficients
        l (float): regularization coefficientj
        eta (float): learning rate
        grad (func): method to compute gradient descent
        obj (func): method to compute objective function
        a (float): [optional, 0 < a < 1] tune stopping condition
        t_eta (float): [optional, 0 < t_eta < 1] learning rate for eta
        max_iter (int): [optional, max_iter > 1] maximum number of training iterations

    Returns:
        float: optimum learning rate
    """
    gb = grad(k, y, beta, l)
    n_gb = norm(gb)

    found_t, i = False, 0
    while not found_t and i < max_iter:
        lh = obj(k, y, l, beta - eta * gb)
        rh = obj(k, y, l, beta) - a * eta * n_gb ** 2
        if lh < rh:
            found_t = True
        elif i == max_iter - 1:
            break
        else:
            eta *= t_eta
            i += 1

    return eta


def cv(x, y, estimator, eargs, nfolds=3):
    """cross validation

    Args:
        x nXd (ndarray): input observations
        y 1Xn (ndarray): true labels
        estimator (Estimator): classifier
        eargs (dict): dictionary of hyperparameters
        nfolds (int): [optional] number of folds

    Returns:
        fitted estimator
    """
    pbar = track_bar(track_total(estimator, eargs, y), desc=f'{nfolds}-Fold CV: {estimator}')
    step = int(x.shape[0] / nfolds)

    for arg in eargs:
        tidx = np.random.choice(np.arange(len(y)), 2 * step)
        vidx = list(set(np.arange(len(y))) - set(tidx))
        xa, ya, xva, yva = x[tidx, :], y[tidx], x[vidx, :], y[vidx]
        estimator = estimator.fit(xa, ya, xva, yva, arg, pbar)

    [pbar.put(f) for f in [1, 'END_FLAG']]

    return estimator


def pca_plot(pca, n_obs, n_klass):
    """plot principle components

    Args:
        pca (nXm ndarray): principle components, after transformation
        n_obs (int): number of observations for each class
        n_klass (int): number of classes

    Returns:

    """
    title   = 'Top 2 Principle Components'
    colors  = ['#A7A37E', '#046380', '#BA2309']
    symbols = ['circle', 'square']
    algos   = ['mykit', 'scikit']
    data = [
        go.Scatter(
            name=f'{algos[j]} class {i}',
            x=pc[n_obs*i:n_obs*i+n_obs, 0],
            y=pc[n_obs*i:n_obs*i+n_obs, 1],
            mode='markers',
            marker=dict(
                size=12,
                line=dict(width=1),
                color=colors[i],
                symbol=symbols[j],
                opacity=1.
            )
        )
        for i in range(n_klass)
        for j, pc in enumerate(pca)
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='x',
            gridwidth=1
        ),
        yaxis=dict(
            title='y',
            gridwidth=1
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


def timeit(func):
    """simple timing function

    Examples:
        timeit(lambda x: 1+2)
        timeit(someFunction())

    Args:
        func: (Function) to perform

    Returns:
        obj resulting from function being applied
    """
    t = dt.datetime.now()
    v = func()
    print(f'{dt.datetime.now() - t}')
    return v


def track_bar(total, desc=''):
    """progress bar to track parallel computations

    Examples:
        pbar = track_bar(100, 'a random procedure')
        pbar.put(1) # will update the bar

    Notes:
        The subprocess managing completed tasks must be
        terminated by putting an 'END_FLAG' into the queue

        pbar.put('END_FLAG')

    Args:
        total (float): number of tasks to be performed
        desc (string): [optional] description for display

    Returns:
        Queue, put element to signal completed task
    """
    def track_it(total, trackq):
        pbar = tqdm(total=total, desc=desc)
        while True:
            update = trackq.get()

            if update == 'END_FLAG':
                break
            else:
                pbar.update(1)

    trackq = multiprocessing.Queue()
    multiprocessing.Process(target=track_it, args=(total, trackq)).start()
    return trackq


def track_total(estimator, eargs, y):
    """calculate total number of classifiers

    Args:
        estimator (Estimator): classifier
        eargs (dict): dictionary of hyperparameters to train with
        y 1Xn (ndarray): true labels

    Returns:
        int
    """
    if isinstance(estimator, OVR):
        total = len(np.unique(y)) * len(eargs)
    elif isinstance(estimator, OVO):
        total = comb(len(np.unique(y)), 2) * len(eargs)
    else:
        total = len(eargs)
    return int(total)


class Estimator(object):
    """tiny base class for polymorphic classifiers"""
    def fgrad(self):
        yield

    def predict(self):
        yield

    def predict_proba(self):
        yield


class Kernel(object):
    """even tinier base class for SVM kernels"""
    def compute(self):
        yield


class Multiclass(object):
    """multilabel classifier"""
    def __init__(self, eargmap, estimator, n_jobs):
        """
        Args:
            eargmap (dict): mapping hyperparameters to validation scores
            estimator (Estimator): classifier
            n_jobs (int): number of processing cores to utilize
        """
        self.eargmap = eargmap
        self.err = 0.
        self.estimator = estimator
        self.n_jobs = n_jobs if n_jobs != 0 else 1

    @property
    def kl_args(self):
        yield

    def _map(self, func, vals):
        """parallel mapping function

        Args:
            func (Function): to apply
            vals ([object]): list of values to apply to function

        Returns:
            ([object]) list of return values
        """
        cpuc = multiprocessing.cpu_count()
        pool = threadPool(cpuc if self.n_jobs <= -1 or self.n_jobs >= cpuc else self.n_jobs)

        vals = pool.map(func, vals)

        pool.close()
        pool.join()
        return vals

    def _split(self):
        yield

    def fit(self, x, y, xv, yv, earg, pbar=None):
        yield

    def predict(self, xpre, xtrain, labels):
        yield

    def predict_proba(self):
        yield

    def update_eargmap(self, fitr, force=False):
        """update argument map with best params

        Args:
            fitr (dict): containing all classes being trained
            force (bool): optional, force the update

        Returns:
            None
        """
        klass, err = fitr['klass'], fitr['err']

        if klass not in self.eargmap.keys():
            self.eargmap[klass] = fitr
        else:
            if self.eargmap[klass]['err'] > err or force:
                self.eargmap[klass] = fitr
            else:
                pass


class OVR(Multiclass):
    """One vs Rest classifier"""
    def __init__(self, estimator, n_jobs=1, eargmap={}):
        """
        Args:
            estimator (Estimator): base classifier
            n_jobs (int): [optional] number of processing cores to utilize
            eargmap (dict): [optional] mapping hyperparameters to validation scores
        """
        super().__init__(eargmap, estimator, n_jobs)

    def __repr__(self):
        return f'<OVR(estimator={self.estimator} err={self.err})>'

    @property
    def kl_args(self):
        """return string representation of regularization coefficients"""
        return '\n'.join([f'{k}VR lambda={v["lambda"]}' for k, v in self.eargmap.items()])

    def fit(self, x, y, xv, yv, earg, pbar=None):
        """fit the classifier, in parallel

        Args:
            x nXd (ndarray): observations
            y 1Xn (ndarray): true labels
            xv mXd (ndarray): observations for validation performance
            yv 1Xm (ndarray): true labels for validation performance
            earg (str): key to dictionary hyperparam to use
            pbar (Queue): [optional] to update progress bar

        Returns:
            dict: { klass: str, lambda: float, err: float, beta 1Xd ndarray) }
        """
        kt = self.estimator.kernel.compute(x)
        kp = self.estimator.kernel.compute(xv, x)

        def compute(args):
            """fit one child classifier"""
            kli, earg, pbar = args

            # extract the hyperparam and class observations to fit
            earg = self.eargmap[str(kli)]['lambda'] if earg == 'best' else earg
            yti, yvi = self._split(kli, y), self._split(kli, yv)

            # fit, predict, update progress bar, and return validation error
            beta = self.estimator.fgrad(kt, yti, earg)
            yhat = self.estimator.predict(kp, beta)
            pbar.put(1)
            return {
                'klass': str(kli),
                'lambda': earg,
                'err': np.mean(yhat != yvi),
                'beta': beta
            }

        # extract unique labels to train
        pears = np.unique(y)

        # setup a progress bar if none was given
        pb = track_bar(len(pears), f'fitting {self}') if pbar is None else pbar

        # fit each child classifier
        fitr = self._map(compute, [(yi, earg, pb) for yi in pears])

        # terminate the progress bar
        if pbar is None:
            pb.put('END_FLAG')

        # update the mapping with each childs optimal hyperparams
        [self.update_eargmap(fr, (earg == 'best')) for fr in fitr]

        # calculate the classifier's combined validation error
        self.err = np.mean([kli[1]['err'] for kli in self.eargmap.items()])
        return self

    def predict(self, xpre, xtrain, labels):
        """predict

        Args:
            xpre (mXd ndarray): observations to predict
            xtrain (nXd ndarray): observations used to train the classifier
            labels (1Xn ndarray):

        Returns:
            1Xm ndarray of predicted labels
        """
        kp = self.estimator.kernel.compute(xpre, xtrain)

        def compute(kli):
            """compute predictions for one label"""
            beta = self.eargmap[str(kli)]['beta']
            return kli, self.estimator.predict_proba(kp, beta)

        # comppute predictions for all child classifiers in parallel
        predictions = np.array(self._map(compute, labels))

        # select the class with the max probability for each observation
        predictions = np.apply_along_axis(lambda row: row[np.argmax([r[1] for r in row])][0], axis=0, arr=predictions)
        return predictions

    @staticmethod
    def _split(klp, y):
        """split the observations and labels

        Args:
            klp (str): positive class label
            y (1Xn ndarray): true labels

        Returns:
            1Xn ndarray of true labels {-1, 1} for associated classifier
        """
        neg = np.where(y != int(klp))[0]

        yt = y**0
        yt[neg] = -1

        return yt


class OVO(Multiclass):
    """One vs One multiclassifier"""
    def __init__(self, estimator, n_jobs=1, eargmap={}):
        """
       Args:
           estimator (Estimator): base classifier
           n_jobs (int): [optional] number of processing cores to utilize
           eargmap (dict): [optional] mapping hyperparameters to validation scores
       """
        super().__init__(eargmap, estimator, n_jobs)

    def __repr__(self):
        return f'<OVO(estimator={self.estimator} err={self.err})>'

    @property
    def kl_args(self):
        """return string representation of regularization coefficients"""
        return '\n'.join([' '.join([
            re.sub(r'\.', 'vs', k), f'lambda={round(self.eargmap[k]["lambda"], 4)}',
            f'err={round(self.eargmap[k]["err"], 4)}'])
            for k in self.eargmap.keys()])

    def fit(self, x, y, xv, yv, earg, pbar=None):
        """fit the classifier, in parallel

        Args:
            x nXd (ndarray): observations
            y 1Xn (ndarray): true labels
            xv mXd (ndarray): observations for validation performance
            yv 1Xm (ndarray): true labels for validation performance
            earg (str): key to dictionary hyperparam to use
            pbar (Queue): [optional] to update progress bar, will be generated if None

        Returns:
            dict: { klass: str, lambda: float, err: float, beta 1Xd ndarray) }
        """
        def compute(args):
            """fit one child classifier"""
            klp, kln, earg, pbar = args

            # extract the hyperparam and class observations to fit
            earg = self.eargmap[f'{klp}.{kln}']['lambda'] if earg == 'best' else earg
            xti, yti = self._split(klp, kln, x, y)
            xvi, yvi = self._split(klp, kln, xv, yv)

            # compute the kernel matrix for training and prediction
            kt = self.estimator.kernel.compute(xti)
            kp = self.estimator.kernel.compute(xvi, xti)

            # fit the classifier using fast gradient descent
            beta = self.estimator.fgrad(kt, yti, earg)

            # predict, update progress bar, and return findings
            yhat = self.estimator.predict(kp, beta)
            pbar.put(1)
            return {
                'klass': f'{klp}.{kln}',
                'lambda': earg,
                'err': np.mean(yhat != yvi),
                'beta': beta
            }

        # compute all pairs
        pairs = list(it.combinations(np.unique(y), 2))

        # initialize a progress bar, and train each child classifier in parallel
        pb = track_bar(len(pairs), f'fitting {self}') if pbar is None else pbar
        fitr = self._map(compute, [(s[0], s[1], earg, pb) for s in pairs])

        # terminate the progress bar
        if pbar is None:
            pb.put('END_FLAG')

        # update the eargmap with each optimal parameter
        [self.update_eargmap(fr, (earg == 'best')) for fr in fitr]

        # compute combined validation error
        self.err = np.mean([kli[1]['err'] for kli in self.eargmap.items()])
        return self

    def predict(self, xpre, xtrain, labels):
        """predict

        Args:
           xpre (mXd ndarray): observations to predict
           xtrain (nXd ndarray): observations used to train the classifier
           labels (1Xn ndarray):

        Returns:
           1Xm ndarray of predicted labels
        """
        kp = self.estimator.kernel.compute(xpre, xtrain)

        def compute(args):
            """compute predictions for one label"""
            pkl, nkl = args
            klkey = f'{pkl}.{nkl}'
            beta = self.eargmap[klkey]['beta']

            return klkey, self.estimator.predict_proba(kp, beta)

        # find all pairs and predict in parallel
        pairs = it.combinations(np.unique(labels), 2)
        predictions = np.array(self._map(compute, pairs))

        # choose the label that was predicted most frequently
        predictions = np.apply_along_axis(lambda row: np.random.choice(mode(row).mode, 1), axis=0, arr=predictions)
        return predictions

    @staticmethod
    def _split(klp, kln, x, y):
        """split the observations and labels

        Args:
            klp (str): positive class label
            y (1Xn ndarray): true labels

        Returns:
            1Xm ndarray of true labels {-1, 1} for associated classifier
        """
        pos = np.where(y == int(klp))[0]
        neg = np.where(y != int(kln))[0]

        yt = y**0
        yt[neg] = -1

        idx = np.concatenate((pos, neg))
        xt, yt = x[idx, :], yt[idx]
        return xt, yt
