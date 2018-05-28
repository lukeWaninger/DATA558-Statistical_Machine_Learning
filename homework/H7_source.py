import datetime as dt
import itertools as it
import numpy  as np
from multiprocessing.dummy import Pool as threadPool
import multiprocessing
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
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
py.init_notebook_mode(connected=True)


def backtracking(k, y, beta, l, eta, grad, obj, a=0.5, t_eta=0.8, max_iter=5):
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
    t = dt.datetime.now()
    v = func()
    print(f'{dt.datetime.now() - t}')
    return v


def track_bar(total, desc):
    def track_it(total, trackq):
        pbar = tqdm(total=total, desc=desc)
        while True:
            update = trackq.get()
            pbar.update(1)

            if update == 'END_FLAG':
                break

    trackq = multiprocessing.Queue()
    multiprocessing.Process(target=track_it, args=(total, trackq)).start()
    return trackq


def track_total(estimator, eargs, y):
    if isinstance(estimator, OVR):
        total = len(np.unique(y)) * len(eargs)
    elif isinstance(estimator, OVO):
        total = comb(len(np.unique(y)), 2) * len(eargs)
    else:
        total = len(eargs)
    return int(total)


class Estimator(object):
    def fgrad(self):
        yield

    def predict(self):
        yield

    def predict_proba(self):
        yield


class Kernel(object):
    def compute(self):
        yield


class Multiclass(object):
    def __init__(self, eargmap, estimator, n_jobs):
        self.eargmap = eargmap
        self.err = 0.
        self.estimator = estimator
        self.n_jobs = n_jobs if n_jobs != 0 else 1

    @property
    def kl_args(self):
        yield

    def _map(self, func, vals):
        cpuc = multiprocessing.cpu_count()
        pool = threadPool(cpuc if self.n_jobs <= -1 or self.n_jobs >= cpuc else self.n_jobs)

        err = pool.map(func, vals)

        pool.close()
        pool.join()
        return err

    def _split(self):
        yield

    def fit(self):
        yield

    def predict(self):
        yield

    def predict_proba(self):
        yield

    def update_eargmap(self, fitr, force=False):
        klass, err = fitr['klass'], fitr['err']

        if klass not in self.eargmap.keys():
            self.eargmap[klass] = fitr
        else:
            if self.eargmap[klass]['err'] > err or force:
                self.eargmap[klass] = fitr
            else:
                pass


class OVR(Multiclass):
    def __init__(self, estimator, n_jobs=1, eargmap={}):
        super().__init__(eargmap, estimator, n_jobs)

    def __repr__(self):
        return f'<OVR(estimator={self.estimator} err={self.err})>'

    @property
    def kl_args(self):
        return '\n'.join([f'{k}VR lambda={v["lambda"]}' for k, v in self.eargmap.items()])

    def fit(self, x, y, xv, yv, earg, pbar=None):
        kt = self.estimator.kernel.compute(x)
        kp = self.estimator.kernel.compute(xv, x)

        def compute(args):
            kli, earg, pbar = args
            earg = self.eargmap[str(kli)]['lambda'] if earg == 'best' else earg
            yti, yvi = self._split(kli, y), self._split(kli, yv)

            beta = self.estimator.fgrad(kt, yti, earg)
            yhat = self.estimator.predict(kp, beta)
            pbar.put(1)
            return {
                'klass': str(kli),
                'lambda': earg,
                'err': np.mean(yhat != yvi),
                'beta': beta
            }

        pears = np.unique(y)
        pb = track_bar(len(pears), f'fitting {self}') if pbar is None else pbar
        fitr = self._map(compute, [(yi, earg, pb) for yi in pears])
        if pbar is None: pb.put('END_FLAG')

        [self.update_eargmap(fr, (earg == 'best')) for fr in fitr]
        self.err = np.mean([kli[1]['err'] for kli in self.eargmap.items()])
        return self

    def predict(self, xpre, xtrain, labels):
        kp = self.estimator.kernel.compute(xpre, xtrain)

        def compute(kli):
            beta = self.eargmap[str(kli)]['beta']
            return kli, self.estimator.predict_proba(kp, beta)

        predictions = np.array(self._map(compute, labels))
        predictions = np.apply_along_axis(lambda row: row[np.argmax([r[1] for r in row])][0], axis=0, arr=predictions)
        return predictions

    @staticmethod
    def _split(klp, y):
        neg = np.where(y != int(klp))[0]

        yt = y ** 0
        yt[neg] = -1

        return yt


class OVO(Multiclass):
    def __init__(self, estimator, n_jobs=1, eargmap={}):
        super().__init__(eargmap, estimator, n_jobs)

    def __repr__(self):
        return f'<OVO(estimator={self.estimator} err={self.err})>'

    @property
    def kl_args(self):
        return '\n'.join([' '.join([
            re.sub(r'\.', 'vs', k), f'lambda={round(ovo.eargmap[k]["lambda"], 4)}',
            f'err={round(ovo.eargmap[k]["err"], 4)}'])
            for k in ovo.eargmap.keys()])

    def fit(self, x, y, xv, yv, earg, pbar=None):
        def compute(args):
            klp, kln, earg, pbar = args
            earg = self.eargmap[f'{klp}.{kln}']['lambda'] if earg == 'best' else earg
            xti, yti = self._split(klp, kln, x, y)
            xvi, yvi = self._split(klp, kln, xv, yv)

            kt = self.estimator.kernel.compute(xti)
            kp = self.estimator.kernel.compute(xvi, xti)

            beta = self.estimator.fgrad(kt, yti, earg)
            yhat = self.estimator.predict(kp, beta)
            pbar.put(1)
            return {
                'klass': f'{klp}.{kln}',
                'lambda': earg,
                'err': np.mean(yhat != yvi),
                'beta': beta
            }

        pairs = it.combinations(np.unique(y), 2)
        pb = track_bar(len(pairs), f'fitting {self}') if pbar is None else pbar
        fitr = self._map(compute, [(s[0], s[1], earg, pb) for s in pairs])
        if pbar is None: pb.put('END_FLAG')

        [self.update_eargmap(fr, (earg == 'best')) for fr in fitr]
        self.err = np.mean([kli[1]['err'] for kli in self.eargmap.items()])
        return self

    def predict(self, xpre, xtrain, labels):
        def compute(args):
            klp, kln = args
            klkey = f'{klp}.{kln}'
            earg = self.eargmap[klkey]['lambda']
            beta = self.eargmap[klkey]['beta']

            xti = self._split(klp, kln, x, y)[0]
            kp = self.estimator.kernel.compute(xpre, xti)

            return klkey, self.estimator.predict_proba(kp, beta)

        pairs = it.combinations(np.unique(y), 2)
        predictions = np.array(self._map(compute, pairs))
        predictions = np.apply_along_axis(lambda row: np.choice(mode(row).mode, 1), axis=0, arr=predictions)
        return predictions

    @staticmethod
    def _split(klp, kln, x, y):
        pos = np.where(y == int(klp))[0]
        neg = np.where(y != int(kln))[0]

        yt = y ** 0
        yt[neg] = -1

        idx = np.concatenate((pos, neg))
        xt, yt = x[idx, :], yt[idx]
        return xt, yt
