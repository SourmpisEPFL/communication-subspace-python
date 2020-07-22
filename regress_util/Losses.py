import numpy as np

def MeanSquaredError(Ytest, Yhat):
    """
    mean squared error for multiple models
    """

    n, K = np.shape(Ytest)
    numModels = int(Yhat.shape[1] / K)
    mse = np.array( [ ((Ytest - Yhat[:, i*K: (i+1)*K])**2).mean() for i in range(numModels) ] )
    return mse

def NormalizedSquaredError(Ytest, Yhat):
    """
    normalized squared error for multiple models
    """

    n, K = np.shape(Ytest)
    numModels = int(Yhat.shape[1] / K)

    sse = np.array( [ ((Ytest - Yhat[:, i*K: (i+1)*K])**2).sum() for i in range(numModels) ] )
    tss = np.array( [ ((Ytest - np.mean(Ytest))**2).sum() for i in range(numModels) ] )

    return sse / tss
