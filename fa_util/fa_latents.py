import numpy as np
from scipy.io import loadmat
import scipy

def ExtractFaLatents(X, q=0):
    #C_CV_NUM_FOLDS = 10;
    #C_CV_OPTIONS = statset('crossval');

    [n,p] = X.shape

    if q == 0:
        q = np.arange(p)

    '''if len(q) > 1:
        qOpt = FactorAnalysisModelSelect(CrossValFa(X, q, C_CV_NUM_FOLDS, C_CV_OPTIONS), q);
    else:
        qOpt = q;'''
    qOpt = q
    Sigma = np.cov(X,rowvar=False)

    [L, psi] = FactorAnalysis(Sigma, qOpt);

    Psi = np.diag(psi)

    C = L@L.T + Psi
    [U, S, V] = scipy.linalg.svd(L, 0);

    Q = matlab_mldivide(C,L) @ V @ S.T

    m = np.mean(X,axis=0)
    M = np.tile(m,(n,1))

    Z = (X - M) @ Q;
    return Z, Q, U


def FactorAnalysis(S, q, *args):
    varargin = args
    C_TOL = 1e-8;#	Stopping criterion for EM
    C_MAX_ITER = 1e6;#	Maximum number of EM iterations
    C_MIN_FRAC_VAR = .01; # Fraction of overall data variance for each

    method = 'FA'

    #for i in range(1,len(varargin),2):
    #   if (varargin[i].upper() == 'METHOD'):
    #       method = varargin[i+1]
    # Excludes neurons with 0 variability from the analysis.
    s = np.diag(S)
    idxs = s[np.abs(s) < np.sqrt(np.spacing(np.double(1)))]
    if len(idxs)>0:
        S = s[:, idxs]

    [n,p] = S.shape
    if np.linalg.matrix_rank(S) == p:

        scale = np.exp(2 * np.sum(np.log(np.diag(np.linalg.cholesky(S)))) / p);
    else:
        r = np.linalg.matrix_rank(S);
        d = np.sort(np.linalg.eig(S), 'descend');
        scale = scipy.stats.gmean(d[1:r])

    L = np.random.randn(p, q) * np.sqrt(scale / q);
    psi = np.diag(S);

    varFloor = C_MIN_FRAC_VAR * psi;

    I = np.identity(q);
    c = -p / 2 * np.log(2 * np.pi);
    logLike = 0;
    import time
    start_time = time.time()
    for i in range(np.int(C_MAX_ITER)):
        if (i % 10000) == 0:
            print((time.time() - start_time))
        invPsi = np.diag(1. / psi)
        invPsiTimesL = invPsi @ L
        a_short = invPsiTimesL
        b_short = (I + L.T@invPsiTimesL )
        c_short = np.linalg.lstsq(b_short.T,a_short.T)[0].T
        invC = invPsi - c_short@L.T @ invPsi

        V = invC @ L
        StimesV = S @ V
        EZZ = I - V.T@L + V.T @ StimesV

        prevLogLike = logLike
        ldm = np.sum( np.log(np.diag(np.linalg.cholesky(invC))) )
        logLike = c + ldm - .5 * np.sum(np.sum(invC* S))

        if i <= 2:
            baseLogLike = logLike
        #elseif (logLike-baseLogLike) < (1+C_TOL) * (prevLogLike-baseLogLike):
        #   break

        L = np.linalg.lstsq(EZZ.T,StimesV.T)[0].T

        psi = np.diag(S) - np.sum(StimesV * L, axis=1)
    return L, psi

def matlab_mldivide(A,b):
    import numpy as np
    from itertools import combinations

    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    if rank == num_vars:
        sol = np.linalg.lstsq(A, b)[0]  # not under-determined
    else:
        for nz in combinations(range(num_vars), rank):  # the variables not set to zero
            try:
                sol = np.zeros((num_vars, 1))
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))

            except np.linalg.LinAlgError:
                pass  # picked bad variables, can't solve
    return sol