import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

def PrincipalComponentRegress(Y, X, dim, qopt=None, alpha=0):
    """B = PrincipalComponentRegress(Y, X, q) fits a Principal Component
     Regression model, with dimensionality given by q, to target variables
     Y and source variables X, returning the mapping matrix B (which includes
     an intercept). If q is a vector containing multiple dimensionalities to
     be tested, PrincipalComponentRegress(Y, X, dim) returns an extended
     mapping matrix, containing the mapping matrices corresponding to each
     dimensionality tested.
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
    # idxs = np.argwhere(np.abs(s) < np.sqrt(np.spacing(np.double(1))))
    idxs = np.where(np.isclose(s, 0))[0]
    if idxs.size > 0:
        s[idxs] = 1

    (n, K) = Y.shape

    Z = (X-m) / s
    if idxs.size > 0:
        Z[:, idxs] = 1
    V = PCA().fit(Z).components_.T

    p = X.shape[1]
    B = np.zeros((p, K*dim.size))
    for i, j in enumerate(dim):
        # import ipdb;ipdb.set_trace()
        x,_,_,_ = np.linalg.lstsq(Z @ V[:, :j], Y)
        B[:, K*i:K*(i+1)] = V[:, :j] @ x

    # import ipdb;ipdb.set_trace()
    B = B / np.repeat(X.std(0), dim.size * K).reshape(B.shape[0], B.shape[1])
    B = np.row_stack([np.repeat(Y.mean(0), dim.size) - m @ B, B])

    return B
