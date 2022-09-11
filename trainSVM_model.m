% This function will estimate the weights vector and the bias term needed
% for making predictions with a support vector machine.
%
% Inputs:
% - dataSamples: matrix of 'numSamples' x 'numFeatures' containing the data
%   that will be used for training the SVM.
% - dataLabels: vector of length 'numSamples' that contains the class of
%   every sample in the matrix 'dataSamples'.
% - kernel: String that specifies the kernel to use. It can take the
%   following values:
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
% - alpha: Vector of length numSamples that contains the value of the
%       Lagrange multipliers (you will use them for the graphs). It can be
%       empty for the primal.
% - w:  Weight vector of length 'numFeatures' for the case of the primal.
%       Alpha vector of length 'numSupportVectors' for the case of the dual.
% - b:  Bias term (scalar).
% - sv: Matrix of numSupportVectors x numFeatures that contains the 
%       support vectors. This value should be empty for the case of 
%       the primal.
% - sv_labels: Vector of numSupportVectors x1 that contains the label of the
%       support vectors. This value should be empty for the case of 
%       the primal. 
%
% ----------------------------------------------------------------------
%             Note about the Support Vectors (for the dual)
% ----------------------------------------------------------------------
%       Remember that the support vectors are those whose value
%       of alpha is > 0 (In practice, you will not get a value of exactly
%       zero, so consider as support vectors those whos value of alpha is >
%       .0001).
%
%       For computing the vale of the bias term 'b', you can take the
%       average over the values of 'b' obtained using the alpha value and
%       label of every support vector.
% ----------------------------------------------------------------------

function [alpha,w,b,sv,sv_labels] = trainSVM_model(dataSamples, dataLabels, ...
    kernel, params)

%initialize vectors for quadprog and outputs
%   create vector of zeroes: remember 
%       w -> size is number of features
%       xi, alpha -> size is number samples 
%   
numS = size(dataSamples, 1);
numF = size(dataSamples, 2);

%determine kernal being used 
if strcmp(kernel, 'lin_primal')
    %from assignment notes
    w = zeros(numF, 1);
    xi = zeros(numS, 1);
    b = 0;   
    z = [w; b; xi];
    %create quadprog H matrix -> z by z 
    H = [eye(numF) zeros(numF, numS+1); zeros(numS+1, numF+numS+1)];
    %create quadprog f
    f = params.C * [zeros(numF, 1); 0; ones(numS, 1)];
    %create quadprog A
    A = -1 * [dataSamples .* dataLabels, dataLabels, eye(numS)];
    %upper bound
    ub = -1 * ones(numS, 1);  
    %lower bound on xi's
    lb = [ -inf * ones(numF, 1) ; zeros(numS, 1), ; -inf ];    
    %Find svms
    [z, fval, exitval, output, alpha] = quadprog(H, f, A, ub, [], [], lb, []);
    %Final results
    b = z(numF+1);
    w = z(1:numF);
    sv = [];
    sv_labels = [];
    
elseif strcmp(kernel, 'lin_dual')
    beq = 0;
    K = dataSamples * dataSamples';
    %create quadprog H matrix
    H = dataLabels * dataLabels' .* K;
    %create quadprog f 
    f = - params.C * ones(numS,1);  
    %create quadprog Aeq
    Aeq = dataLabels';

    lb = zeros(numS,1);
    ub = ones(numS,1);
    %use a convex function
    options.Display = 'off';

    alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub,0,options);
    
    sv_labels = dataLabels(find(alpha > 0.0001));
    sv = dataSamples(find(alpha > 0.0001),:);
    w = alpha(find(alpha > 0.0001));
    b=0;
    for i = 1:size(alpha)
        if alpha(i) > 0.0001
            b= b + dataLabels(i) - alpha' * (dataLabels.*K(:,i));
        end
    end
    b=b/(size(sv_labels, 1));
    
elseif strcmp(kernel, 'rbf')
    beq = 0;
    K = gaussKern(dataSamples, dataSamples, params.Sigma);
    %create quadprog H matrix
    H = dataLabels * dataLabels' .* K;
    %create quadprog f 
    f = - params.C * ones(numS,1);  
    %create quadprog Aeq
    Aeq = dataLabels';

    lb = zeros(numS,1);
    ub = ones(numS,1);
    %use a convex function
    options.Display = 'off';

    alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub,0,options);
    %correct this
    sv_labels = [];
    sv = [];
    w = [];
    b=0;
    for i = 1:size(alpha)
        if alpha(i) > 0.0001
            sv_labels = [sv_labels; dataLabels(i)];
            sv = [sv; dataSamples(i, :)];
            w = [w; alpha(i)];
            b= b + (alpha' * (dataLabels.*K(:,i))) - dataLabels(i);
        end
    end
    b=b/(size(sv_labels, 1));
    
else %polynomial kernel
    beq = 0;
    K = quadKern(dataSamples, dataSamples, params.C, params.D);
    %create quadprog H matrix
    H = dataLabels * dataLabels' .* K;
    %create quadprog f 
    f = - params.C * ones(numS,1);  
    %create quadprog Aeq
    Aeq = dataLabels';

    lb = zeros(numS,1);
    ub = ones(numS,1);
    %use a convex function
    options.Display = 'off';

    alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub,0,options);
    
    sv_labels = dataLabels(find(alpha > 0.0001));
    sv = dataSamples(find(alpha > 0.0001),:);
    w = alpha(find(alpha > 0.0001));
    b=0;
    for i = 1:size(alpha)
        if alpha(i) > 0.0001
            b= b + (alpha' * (dataLabels.*K(:,i))) - dataLabels(i);
        end
    end
    b=b/(size(sv_labels, 1));
    
end


end