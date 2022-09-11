% File: findBestParams
% ------------------------------------------------------------------
% This function will use 5-fold cross validation to determine the best
% parameters for the SVM.
%
% Input:
% - dataSamples: matrix of 'numSamples' x 'numFeatures' containing the data
%   that will be used for training the SVM.
% - dataLabels: vector of length 'numSamples' that contains the class of
%   every sample in the matrix 'dataSamples'.
% - folds: Contains the fold number of every sample in dataSamples. This
%   vector is meant to be used in the cross validation process. For example,
%   in the ith iteration all the samples whose entry in folds == i will be
%   part of the test set, while the rest will be used for trainig the model.
% - kernel: String that specifies the kernel to use. It can take the
%   following values:
%       + lin_primal:   for the linear kernel using the primal.
%       + lin_dual:     for the linear kernel using the dual.
%       + rbf:          for using the RBF kernel
%       + poly:         for using the polynomial kernel
% Output:
% - bestParams: Structure that contains the following fields:
%       + bestParams.C:     Soft margin parameter C.
%       + bestParams.Sigma: Width of the Gaussian kernel.
%       + bestParams.D:     Degree of the polynomial.
%   Note that not all the kernels require the same parameters, if a
%   parameter is not required it can be left blank.

% ----------------------------------------------------------------------
%                            IMPORTANT
% ----------------------------------------------------------------------
% The vector folds assign every sample into a fold for performing 5-fold
% cross validation. This means that in every iteration you will have to
% create a new training set a test set. In the first iteration, the entries
% with a 1 in the vector 'folds' are part of the TEST set. In the second
% iteration, the TEST set are the entries with a 2 in the vector 'folds'
% and so on.
% ----------------------------------------------------------------------


function bestParams = findBestParams(dataSamples, dataLabels, folds, kernel)
    % You need to try all possible combinations of the following
    % parameters, and choose the combination that produces the highest
    % cross-validation accuracy. In case of ties you need to select the
    % lowest possible value of C (among the ties), and then the lowest
    % value of sigma/degree (among the ties).
    C = [2^-5, 2^-3, 2^-1, 2, 4, 1, 5, 2^3, 2^5];
    Sigma = [2^-3, 2^-1, 2, 2^3, 0.2, 0.5, 1, 1.5, 10, 25, 200];
    Degree = [2, 3, 4, 5, 6, 7, 8, 10];

accuracy = 0;
newAccuracy = 0;
field1 = 'C';  value1 = 0;
field2 = 'Sigma';  value2 = 0;
field3 = 'D';  value3 = 0;
bestParams = struct(field1,value1,field2,value2,field3,value3);
params = struct(field1,value1,field2,value2,field3,value3);
    
for i = 1:5    
    testSet = dataSamples(find(folds == i),:);
    trainSet = dataSamples(find(folds ~= i),:);
    testSet_Labels = dataLabels(find(folds == i),:);
    trainSet_Labels = dataLabels(find(folds ~= i),:);
    for j = 1:length(C)
        params.C = C(j);
        if strcmp(kernel, 'lin_primal') || strcmp(kernel, 'lin_dual')
            [alphas, w, b, sv, sv_labels] = trainSVM_model(trainSet, trainSet_Labels, kernel, params);
            predictions = predictUsingSVM(testSet,w,b,sv,sv_labels,...
                            kernel,params);
            sumCorrect = 0;
            for k = 1:length(predictions)
                if predictions(k) == trainSet_Labels(k)
                    sumCorrect = sumCorrect+1;
                end
            end 
            newAccuracy = sumCorrect / length(predictions);
            if newAccuracy > accuracy
                accuracy = newAccuracy;
                bestParams.C = C(j);
            end
            
        elseif strcmp(kernel, 'rbf')
            for s = 1:length(Sigma)
                params.Sigma = Sigma(s);
        
                [alphas, w, b, sv, sv_labels] = trainSVM_model(trainSet, trainSet_Labels, kernel, params);
                predictions = predictUsingSVM(testSet,w,b,sv,sv_labels,...
                            kernel,params);
                sumCorrect = 0;
                for k = 1:length(predictions)
                    if predictions(k) == trainSet_Labels(k)
                        sumCorrect = sumCorrect+1;
                    end
                end 
                newAccuracy = sumCorrect / length(predictions);
                if newAccuracy > accuracy
                    accuracy = newAccuracy;
                    bestParams.C = C(j);
                    bestParams.Sigma = Sigma(s);
                end
            end
        else %poly
            for d = 1:length(Degree)
                params.D = Degree(d);
        
                [alphas, w, b, sv, sv_labels] = trainSVM_model(trainSet, trainSet_Labels, kernel, params);
                predictions = predictUsingSVM(testSet,w,b,sv,sv_labels,...
                            kernel,params);
                sumCorrect = 0;
                for k = 1:length(predictions)
                    if predictions(k) == trainSet_Labels(k)
                        sumCorrect = sumCorrect+1;
                    end
                end 
                newAccuracy = sumCorrect / length(predictions);
                if newAccuracy > accuracy
                    accuracy = newAccuracy;
                    bestParams.C = C(j);
                    bestParams.D = Degree(d);
                end
            end
        end    
    end
end



end