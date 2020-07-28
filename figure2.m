function [cvlossSelf, optRankSelf, cvlossCross, optRankCross] = figure2(X, Y_V1, Y_V2, dimensions)
    %  [cvlossSelf, optRankSelf, cvlossCross, optRankCross] = figure2(dimensions)

    %% SET_CONSTS
    addpath('../communication-subspace/fa_util')
    addpath('../communication-subspace/mat_sample')
    addpath('../communication-subspace/regress_methods')
    addpath('../communication-subspace/regress_util')
    % load('mat_sample/sample_data.mat')
    % load('data/temp_data.mat')
    % dimensions = double(dimensions);
    % X = double(X);
    % Y_V1 = double(Y_V1);
    % Y_V2 = double(Y_V2);


    % Vector containing the interaction dimensionalities to use when fitting
    % RRR. 0 predictive dimensions results in using the mean for prediction.
    % dimensions = 1:10;

    % Number of cross validation folds.
    cvNumFolds = 10;

    % Initialize default options for cross-validation.
    cvOptions = statset('crossval');

    % If the MATLAB parallel toolbox is available, uncomment this line to
    % enable parallel cross-validation.
    % cvOptions.UseParallel = true;

    % Regression method to be used.
    regressMethod = @ReducedRankRegress;

    cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    	dimensions, 'LossMeasure', 'NSE');

    % Cross-validation routine.
    cvl = crossval(cvFun, Y_V1, X, ...
    	  'KFold', cvNumFolds, ...
    	'Options', cvOptions);

    % Stores cross-validation results: mean loss and standard error of the
    % mean across folds.
    cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

    % To compute the optimal dimensionality for the regression model, call
    % ModelSelect:
    optDimReducedRankRegress = ModelSelect...
    	(cvLoss, dimensions);

    % Plot Reduced Rank Regression cross-validation results
    x = dimensions;
    y = 1-cvLoss(1,:);
    e = cvLoss(2,:);

    errorbar(x, y, e, 'o--', 'MarkerSize', 10)

    xlabel('Number of predictive dimensions')
    ylabel('Predictive performance')

    disp(optDimReducedRankRegress)
    cvlossSelf = cvl;
    optRankSelf = optDimReducedRankRegress;
    %%

    % Cross-validation routine.
    cvl = crossval(cvFun, Y_V2, X, ...
    	  'KFold', cvNumFolds, ...
    	'Options', cvOptions);

    % Stores cross-validation results: mean loss and standard error of the
    % mean across folds.
    cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

    % To compute the optimal dimensionality for the regression model, call
    % ModelSelect:
    optDimReducedRankRegress = ModelSelect...
    	(cvLoss, dimensions);

    % Plot Reduced Rank Regression cross-validation results
    x = dimensions;
    y = 1-cvLoss(1,:);
    e = cvLoss(2,:);

    cvlossCross = cvl;
    optRankCross = optDimReducedRankRegress;
    hold on
    errorbar(x, y, e, 'o--', 'MarkerSize', 10)

    xlabel('Number of predictive dimensions')
    ylabel('Predictive performance')

end
