import numpy as np
from regress_util.RegressPredict import RegressPredict

def RegressFitAndPredict(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, lossmeasure='NSE', qopt=None):
    """
    Wrappers in order to run the convoluted cv thingy
    """

    B = regressFun(Ytrain, Xtrain, alpha, qopt=qopt)

    # if B is tuple take only first
    if type(B) == type(()):
        B = B[0]

    loss, Yhat = RegressPredict(Ytest, Xtest, B, lossmeasure)

    return loss
