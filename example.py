from scipy.io import loadmat
from regress_util.RegressFitAndPredict import RegressFitAndPredict
from regress_methods.ReducedRankRegress import ReducedRankRegress
from fa_util.fa_latents import FactorRegress
from fa_util.fa_latents import ExtractFaLatents
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = loadmat('../communication-subspace/mat_sample/sample_data.mat')

X = data['X']
Y_V1 = data['Y_V1']
Y_V2  = data['Y_V2']

_, B_, _ = ReducedRankRegress(Y_V2, X, np.array([1]))

# q=30
# Z, U, Q = ExtractFaLatents(X, q)
numDimsForPrediction = np.arange(10).astype(int) + 1

numFolds = 10
k = KFold(n_splits=numFolds, shuffle=True)

cvLoss = []
for train_ind, test_ind in k.split(X):
    Xtrain, Ytrain = X[train_ind], Y_V2[train_ind]
    Xtest, Ytest = X[test_ind], Y_V2[test_ind]

    cvLoss += [RegressFitAndPredict(ReducedRankRegress,
    Ytrain, Xtrain, Ytest, Xtest,
    numDimsForPrediction, lossmeasure='NSE')]

cvLoss = np.vstack(cvLoss)
plt.figure()
plt.errorbar(numDimsForPrediction, 1 - cvLoss.mean(0), yerr=cvLoss.std(0) / np.sqrt(numFolds))

#########################################################33

# Factor Regression
numDimsUsedForPrediction = np.arange(10) + 1

k = KFold(n_splits=numFolds, shuffle=True)

qopt = 11 # this is taken from matlab, be caruful magic number
cvLoss = []
for train_ind, test_ind in k.split(X):
    Xtrain, Ytrain = X[train_ind], Y_V2[train_ind]
    Xtest, Ytest = X[test_ind], Y_V2[test_ind]

    cvLoss += [RegressFitAndPredict(FactorRegress,
    Ytrain, Xtrain, Ytest, Xtest,
    numDimsUsedForPrediction, lossmeasure='NSE', qopt=qopt)]

cvLoss = np.vstack(cvLoss)
plt.figure()
plt.errorbar(numDimsUsedForPrediction, 1 - cvLoss.mean(0), yerr=cvLoss.std(0) / np.sqrt(numFolds))
