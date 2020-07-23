import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

def ReducedRankRegress(Y, X, dim, ridgeinit=False, scale=False, qopt=None):
    """  B = ReducedRankRegress(Y, X, dim) fits a Reduced Rank Regression model,
      with number of predictive dimensions given by dim, to target variables Y
      and source variables X, returning the mapping matrix B (which includes
      an intercept). If dim is a vector containing multiple numbers of
      predictive dimensions to be tested, ReducedRankRegress(Y, X, dim)
      returns an extended mapping matrix, containing the mapping matrices
      corresponding to each number of predictive dimensions tested.
     Arguments:
     Y np.ndarray: target data matrix (NxK)
     X np.ndarray: data matrix (Nxp)
     dim np.ndarray: vector with numbers of dimension to be tested (1xnumDims)
     ridgeinit bool : not sure yet
     scale bool : not sure yet either
     Output:
     B np.ndarray : Weights for each corresponding dimension (p+1 x K*numDims)
     B_ np.ndarray: OLS weights multiplied with V (eigenvector matrix) (p x K)
     V np.ndarray : colums are the eigenvectors
    """

    # Exclude neurons with 0 variance
    m = X.mean(0)
    s = X.std(0)
    p = X.shape[1]
    # idxs = np.argwhere(np.abs(s) < np.sqrt(np.spacing(np.double(1))))
    idxs = np.where(np.logical_not(np.isclose(s, 0)))[0]
    X = X[:, idxs]
    (n, K) = Y.shape

    Z = X - m[idxs]

    # here we need to put the optimization lambda stuff
    ridge = Ridge(alpha=0) # with 0 is equivalent to ols
    ridge.fit(Z, Y)
    Bfull = ridge.coef_.T

    ridge.intercept_ = 0
    Yhat = ridge.predict(Z)

    V = PCA().fit(Yhat).components_.T

    B_ = Bfull @ V

    if idxs.size != p:
        Bnew = B_.copy()
        B_ = np.zeros((p, K))
        B_[idxs, :] = Bnew

    B = np.zeros((p, K*dim.size))
    for i, j in enumerate(dim):
        B[idxs, K*i:K*(i+1)] = Bfull @ V[:, :j] @ V[:, :j].T

    B = np.row_stack([np.repeat(Y.mean(0), dim.size) - m @ B, B])

    return B, B_, V
