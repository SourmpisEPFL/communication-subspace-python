import numpy as np
from regress_util.Losses import MeanSquaredError, NormalizedSquaredError

def RegressPredict(Y, X, B, lossmeasure='NSE'):
    """
    Wrapper for inference
    """

    Yhat = np.column_stack((np.ones((X.shape[0],1)), X)) @ B

    if lossmeasure == 'MSE':
        loss = MeanSquaredError(Y, Yhat)
    elif lossmeasure == 'NSE':
        loss = NormalizedSquaredError(Y, Yhat)
    else:
        assert True, 'invalid lossmeasure name'

    return loss, Yhat
