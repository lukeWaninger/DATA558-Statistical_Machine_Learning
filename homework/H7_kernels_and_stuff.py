import numpy as np
from sklearn.decomposition import PCA

norm, ident = np.linalg.norm, np.identity


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



if __name__ == '__main__':
    ex2()