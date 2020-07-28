%% SET_CONSTS
addpath('../communication-subspace/fa_util')
addpath('../communication-subspace/mat_sample')
addpath('../communication-subspace/regress_methods')
addpath('../communication-subspace/regress_util')
% load('mat_sample/sample_data.mat')
load('data/temp_data.mat')
X = double(X);
Y_V1 = double(Y_V1);
Y_V2 = double(Y_V2);
