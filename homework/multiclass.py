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

    def fit(self, x, y, xv, yv, earg, pbar=None):
        yield

    def predict(self, xpre, xtrain, labels):
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
            re.sub(r'\.', 'vs', k), f'lambda={round(self.eargmap[k]["lambda"], 4)}',
            f'err={round(self.eargmap[k]["err"], 4)}'])
            for k in self.eargmap.keys()])

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

        pairs = list(it.combinations(np.unique(y), 2))
        pb = track_bar(len(pairs), f'fitting {self}') if pbar is None else pbar
        fitr = self._map(compute, [(s[0], s[1], earg, pb) for s in pairs])
        if pbar is None: pb.put('END_FLAG')

        [self.update_eargmap(fr, (earg == 'best')) for fr in fitr]
        self.err = np.mean([kli[1]['err'] for kli in self.eargmap.items()])
        return self

    def predict(self, xpre, xtrain, labels):
        kp = self.estimator.kernel.compute(xpre, xtrain)

        def compute(args):
            pkl, nkl = args
            klkey = f'{pkl}.{nkl}'
            beta = self.eargmap[klkey]['beta']

            return klkey, self.estimator.predict_proba(kp, beta)

        pairs = it.combinations(np.unique(labels), 2)
        predictions = np.array(self._map(compute, pairs))
        predictions = np.apply_along_axis(lambda row: np.random.choice(mode(row).mode, 1), axis=0, arr=predictions)
        return predictions

    @staticmethod
    def _split(klp, kln, x, y):
        pos = np.where(y == int(klp))[0]
        neg = np.where(y != int(kln))[0]

        yt = y**0
        yt[neg] = -1

        idx = np.concatenate((pos, neg))
        xt, yt = x[idx, :], yt[idx]
        return xt, yt