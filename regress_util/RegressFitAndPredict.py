import numpy as np
from regress_util.RegressPredict import RegressPredict

def RegressFitAndPredict(regressFun, Ytrain, Xtrain, Ytest, Xtest, dim, lossmeasure='NSE', qopt=None, alpha=0):
    """
    Wrappers in order to run the convoluted cv thingy
    """

    B = regressFun(Ytrain, Xtrain, dim, qopt=qopt, alpha=alpha)

    # if B is tuple take only first
    if type(B) == type(()):
        B = B[0]

    loss, Yhat = RegressPredict(Ytest, Xtest, B, lossmeasure)

    return loss
