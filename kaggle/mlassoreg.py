from kaggle.my_classifier import MyClassifier
import numpy as np

# misc setup for readability
norm = np.linalg.norm


class MyLASSORegression(MyClassifier):
    def __init__(self, x_train, y_train, parameters, x_val=None, y_val=None,
                 expected_betas=None, log_queue=None, logging_level='none', task=None):

        super().__init__(x_train=x_train, y_train=y_train, parameters=parameters,
                         x_val=x_val, y_val=y_val, log_queue=log_queue,
                         logging_level=logging_level, task=task)

        self.__betas = self.coef_
        self.__exp_betas = expected_betas
        self.__seen = dict()
        self.__available_idx = []

    def _simon_says_fit(self):
        return self.fit()

    # public methods
    def fit(self):
        while super().fit():
            self.__betas = np.zeros(self._d)
            self.__seen = dict()
            self.__available_idx = list(np.arange(self._d))

            algo = self._param('algo')
            if algo == 'random':
                self.__random_coordinate_descent()

            elif algo == 'cyclic':
                self.__cyclic_coordinate_descent()

            else:
                raise Exception("algorithm <%s> is not available" % algo)

            self._set_betas(self.__betas)
        return self

    def predict(self, x, betas=None):
        if betas is not None:
            b = betas
        elif len(self.__betas) > 0:
            b = self.__betas
        else:
            b = self.coef_

        return [1 if xi.T @ b > 0 else -1 for xi in x]

    def predict_proba(self, x, betas=None):
        if betas is not None:
            b = betas
        else:
            b = self.coef_

        return None

    def __beta_str(self):
        return ','.join([str(b) for b in self.__betas])

    def __correct_beta_percentage(self):
        return np.sum([1 for b, be in zip(self.__betas, self.__exp_betas) if np.isclose(b, be)])/self._d

    def __compute_beta(self, j):
        n, a = self._n, self._param('alpha')
        b = np.concatenate((self.__betas[:j], self.__betas[j+1:]))
        x = np.concatenate((self._x[:, :j], self._x[:, j+1:]), axis=1)

        r = self._y - x @ b
        z = np.sum(self._x[:, j]**2)

        switch = 2/self._n * norm(r)

        if abs(switch) >= a and z != 0:
            bj = (-1*np.sign(switch)*a + (2/n)*self._x[:, j].T@r)/((2/n)*z)
        else:
            bj = 0
        return bj

    def __cyclic_coordinate_descent(self, idx=0, max_iter=None):
        if not max_iter:
            max_iter = self._param('max_iter')

        t, seen = 0, []
        while t < max_iter:
            for i in range(self._d):
                j = idx % self._d

                b0 = self.__compute_beta(j)
                self.__betas[j] = b0

                idx += 1
                self.__update_available(j)
            self.log_metrics([t, self.__objective()])
            t += 1

    @staticmethod
    def __min_n1d1(x, y, l, b):
        return norm((y-x*b)) + l*abs(b)

    def __objective(self):
        x, y, n = self._x, self._y, self._n
        a, b = self._param('alpha'), self.__betas

        return (1/n)*norm(y-x @ b)**2 + a*np.sum(np.abs(b))

    def __pick_coordinate(self):
        return np.random.choice(self.__available_idx, 1)[0]

    def __random_coordinate_descent(self, max_iter=None):
        if not max_iter:
            max_iter = self._param('max_iter')

        t = 0
        while t < max_iter:
            for i in range(self._d):
                j = self.__pick_coordinate()

                b0 = self.__compute_beta(j)
                self.__betas[j] = b0

                if self.__update_available(j):
                    break

            self.log_metrics([t, self.__objective()])
            t += 1

    def __update_available(self, j):
        if self.__betas[j] != 0:
            return False

        j = str(j)
        if j not in self.__seen:
            self.__seen[j] = 0

        elif self.__seen[j] <= 3:
            self.__seen[j] += 1
            return False

        else:
            self.__available_idx.remove(int(j))

            if len(self.__available_idx) == 0:
                return True 
