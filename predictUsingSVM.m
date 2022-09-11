% File: predictUsingSVM.m
% ------------------------------------------------------------
% This function will take the values 'w' and 'b' of a trained SVM model and
% it will use them to predict the class of every sample in the matrix
% dataSamples.
%
% Inputs:
%   - dataSamples: Matrix of 'numSamples' by 'numFeatures' containing the
%     feature vector of the samples whose class will be predicted.
%   - w: weight vector of length 'numFeatures' for the case of the primal,
%     and length 'numSupportVectors' for the case of the dual.
%   - b: bias term (Scalar).
%   - sv: Matrix of numSupportVectors x numFeatures that contains the 
%     support vectors. This value should be empty for the case of 
%     the primal.
%   - sv_label: Vector of numSupportVectors x1 that contains the label of the
%       support vectors. This value should be empty for the case of 
%       the primal.
%   - kernel: String that specifies the kernel to use. It can take the
%     following values:
%       + lin_primal:   for the SVM using the primal.
%       + lin_dual:     for the linear kernel using the dual.
%       + rbf:          for using the RBF kernel
%       + poly:         for using the polynomial kernel
% - params: It is a structure that contains the parameters needed for
%   different versions of the SVM. It contains three values:
%       + params.C:     Soft margin parameter C. (scalar)
%       + params.Sigma: Width of the Gaussian kernel. (scalar)
%       + params.D:     Degree of the polynomial. (discrete)
%
% Outputs:
%   - predictions: column vector of length 'numSamples' that contains a
%     value of -1 or 1 indicating the predicted class of every sample.
function predictions = predictUsingSVM(dataSamples,w,b,sv,sv_labels,...
    kernel,params)

predictions = ones(size(dataSamples, 1), 1);

if strcmp(kernel, 'lin_primal')
    
    %find predictions
    for i = 1:size(dataSamples, 1)
        predictions(i) = sign( (dataSamples(i,:)* w )+b); 
    end
    
elseif strcmp(kernel, 'lin_dual')
    %find predictions
    for i = 1:size(dataSamples, 1)
        predictions(i) = sign( sum((w.*sv_labels)'*(sv*dataSamples(i)'))+b); 
    end
    
elseif strcmp(kernel, 'rbf')
    %find predictions
    K = gaussKern(sv, dataSamples, params.Sigma);
    for i = 1:size(dataSamples, 1)
        predictions(i) = sign( sum((w.*sv_labels)'*(K(:,i)))+b); 
    end
    

else % polynomial
    %find predictions
    K = quadKern(sv, dataSamples, params.C, params.D);
    for i = 1:size(dataSamples, 1)
        predictions(i) = sign( sum((w.*sv_labels)'*(K(:,i))+b)); 
    end
      
end    
        
end