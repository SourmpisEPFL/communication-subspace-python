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

    [L, psi] = FactorAnalysis(Sigma, qOpt)

    Psi = np.diag(psi)

    C = L@L.T + Psi
    [U, S, V] = scipy.linalg.svd(L, 0)

    Q = matlab_mldivide(C,L) @ V @ S.T

    m = np.mean(X,axis=0)
    M = np.tile(m,(n,1))

    Z = (X - M) @ Q
    return Z, Q, U


def FactorAnalysis(S, q, *args):
    varargin = args
    C_TOL = 1e-8#	Stopping criterion for EM
    C_MAX_ITER = 1e8#	Maximum number of EM iterations
    C_MIN_FRAC_VAR = .01 # Fraction of overall data variance for each

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

        scale = np.exp(2 * np.sum(np.log(np.diag(np.linalg.cholesky(S)))) / p)
    else:
        r = np.linalg.matrix_rank(S)
        d = np.sort(np.linalg.eig(S), 'descend')
        scale = scipy.stats.gmean(d[1:r])

    L = np.random.randn(p, q) * np.sqrt(scale / q)
    psi = np.diag(S)

    varFloor = C_MIN_FRAC_VAR * psi

    I = np.identity(q)
    c = -p / 2 * np.log(2 * np.pi)
    logLike = 0
    import time
    start_time = time.time()
    for i in range(np.int(C_MAX_ITER)):
        if (i % 10000) == 0:
            print(i)
            print((time.time() - start_time))
        invPsi = np.diag(1. / psi)
        invPsiTimesL = invPsi @ L
        a_short = invPsiTimesL
        b_short = (I + L.T@invPsiTimesL )
        c_short = np.linalg.lstsq(b_short.T,a_short.T)[0].T
        #c_short = np.linalg.inv(b_short.T @ b_short) @ b_short @ a_short




        invC = invPsi - c_short@L.T @ invPsi

        V = invC @ L
        StimesV = S @ V
        EZZ = I - V.T@L + V.T @ StimesV

        prevLogLike = logLike
        ldm = np.sum( np.log(np.diag(np.linalg.cholesky(invC))) )
        logLike = c + ldm - .5 * np.sum(np.sum(invC* S))

        if i <= 2:
            baseLogLike = logLike
        elif (logLike-baseLogLike) < (1+C_TOL) * (prevLogLike-baseLogLike):
            break


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

def FactorRegress(Y, X, q, qOpt = []):
    [n, K] = Y.shape
    [_,p] = X.shape

    if qOpt == []:
        '''qFactorAnalysis = np.arange(p)
        qOpt = FactorAnalysisModelSelect(...
        CrossValFa(X, qFactorAnalysis), qFactorAnalysis)'''

    q0pt = 11
    m = np.mean(X,axis=0)
    M = np.tile(m,(n,1))
    q= [x for x in q if x <= q0pt]
    if len(q) < 1:
        q = qOpt

    Sigma = np.cov(X, rowvar=False)
    s = np.diag(Sigma)

    idxs = s[np.abs(s) < np.sqrt(np.spacing(np.double(1)))]
    if len(idxs)>0:
        Sigma = Sigma[:, idxs]
        X = X[:, idxs]
        _,auxP = X.shape
        auxIdxs = np.arange(1,auxP).T
        auxIdxs[idxs] =[];



    [L, psi] = FactorAnalysis(Sigma, q0pt)

    Psi = np.diag(psi)

    C = L@L.T + Psi
    [_, sdiag, V] = scipy.linalg.svd(L, 0)
    S = np.zeros((q0pt, q0pt))
    np.fill_diagonal(S, sdiag)
    Q = matlab_mldivide(C,L) @ V @ S.T
    if q[0] == 0:
        B = np.zeros(p, K)
    else:
        EZ = (X - M) @ Q[:, 0: q[0]]

        B = Q[:, 0: q[0]] @ matlab_mldivide(EZ,Y)

    numDims = len(q)
    if numDims > 1:
        B_n,B_m = B.shape
        zeropad = np.zeros((B_n,K*numDims-B_m))
        B = np.append(B, zeropad, axis=1)
        for i in range(1,numDims):
            EZ = (X - M) @ Q[:, 0: q[i]]
            B[:,K*(i-1):K*i] = Q[:, 0: q[i]] @ matlab_mldivide(EZ, Y)
    if len(idxs)>0:
        auxB = B
        B = np.zeros(auxP, K * numDims)
        B[auxIdxs,:] = auxB

        auxM = m
        m = np.zeros(1, auxP)
        m[auxIdxs] = auxM
    B = np.vstack((np.tile(np.mean(Y,axis=0),[1, numDims]) - m @B,B))

    return B