import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

class RRR(BaseEstimator):
    # reduced-rank regression on mean-centred datasets
    def __init__(self, rank):
        self.rank = rank
        self.B_rrr = None

    def fit(self, X, Y):
        self.ols_model = linear_model.LinearRegression().fit(X, Y)
        self.Y_hat_ols = self.ols_model.predict(X)
        self.pc = PCA(n_components=self.rank).fit(self.Y_hat_ols)
        self.V = self.pc.components_.T
        self.B_ols = self.ols_model.coef_.T
        self.B_rrr = self.B_ols @ self.V @ self.V.T
        return self

    def predict(self, X):
        assert self.B_rrr is not None, 'Not fitted!'
        return X @ self.B_rrr

def myr2_score(estim, X, Y):
    return 1 - np.mean((Y - estim.predict(X))**2) / np.var(Y)
