import datetime as dt
import itertools as it
import numpy  as np
from multiprocessing.dummy import Pool as threadPool
import multiprocessing
import pandas as pd
import re
from scipy.special import comb
from scipy.stats import mode
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def test_grad(eps=1e-6):
    n = 5
    for i in range(n):
        np.random.seed(0)
        beta = -np.random.normal(size=n)
        x = np.random.randn(n, n)
        k = pairwise.rbf_kernel(x, x)
        y = np.random.choice([0, 1], size=5)
        l = 0.5

        f1 = objective(k, y, l, beta)
        beta[i] = beta[i] + eps
        f2 = objective(k, y, l, beta)

        grad = gradient(k, y, beta, l)[i]
        print(f'Estimated and calculated values of beta[{i}]: {(f2-f1)/eps}, {grad}')

        assert np.isclose((f2 - f1) / eps, grad), \
            f'Estimated gradient {str((f2-f1)/eps)} is not approximately equal to the computed gradient {str(grad)}'

    print('Test passed')


def timeit(func):
    t = dt.datetime.now()
    v = func()
    print(f'{dt.datetime.now() - t}')
    return v


def track_total(estimator, eargs):
    if isinstance(estimator, OVR):
        total = len(np.unique(y)) * len(eargs)
    elif isinstance(estimator, OVO):
        total = comb(len(np.unique(y)), 2) * len(eargs)
    else:
        total = len(eargs)
    return int(total)


def track_bar(total, desc):
    def track_it(total, trackq):
        pbar = tqdm(total=total, desc=desc)
        while True:
            update = trackq.get()

            if update == 'END_FLAG':
                break
            else:
                pbar.update()

    trackq = multiprocessing.Queue()
    multiprocessing.Process(target=track_it, args=(total, trackq)).start()
    return trackq


def backtracking(k, y, n, beta, l, eta, grad, obj, a=0.5, t_eta=0.8, max_iter=5):
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



def ex2():
    def gen_ki(m, s):
        return np.random.normal(m, s, 1)

    def gen_k(n, m):
        means = np.arange(0, n*m, 2)
        np.random.shuffle(means)

        return np.array([[gen_ki(mi, 1) for mi in means] for i in range(n)]).reshape((n, m))

    def oja(x, eta=1, max_iter=50):
        n, d = x.shape
        w, w1 = np.random.normal(size=d)/norm(d), None

        for i in range(max_iter):
            w  = w + eta*((x.T @ x) @ w)
            w /= norm(w)
            eta /= i+1

        for i in range(2, max_iter):
            w1 = w1 + eta*(x @ x.T @ (ident(d) - w @ w.T)) @ w1
            w1  /= norm(w1)
            eta /= i+1

        pc = np.array([w @ x.T, w1 @ x.T]).T
        return pc

    n_obs, n_feat, n_k = 30, 60, 3
    x = np.array([gen_k(n_obs, n_feat) for i in range(n_k)])

    mykit_pca = oja(x)
    scikt_pca = PCA().fit_transform(X=x)
