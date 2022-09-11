% File:generateModelsToBeGraded.m
% ------------------------------------------------------------------
% This function will save the outputs that we require to evaluate your
% implementation of SVM. In this function you are expected to do the
% following:
% 1. Using the dataset, you need to find the best parameters for every
% classifier that you will create (Linear SVM primal, Linear SVM Dual, RBF,
% and Polynomial kernel).
% 2. Use those parameters and the entire dataset to train your final
% classifiers.
% 3. You can use the outputs of your SVM model to create the graphs (Check
% the files inside the folder 'SVCplot_demo').
%
% Please pay attention to the name that you have to give to the variables
% that you create.
%
% Note that you already created all the code for performing the steps 1 and
% 2 of this function, so this part should be simple to do.
%
% Do not modify the names of the .mat files, nor the name of the
% variables that will be saved on those .mat files.


function generateModelsToBeGraded()
global p1
% For the Survival Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios

% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;
% ------------------------------------------------
% YOUR CODE HERE

%Load Data
fprintf('\nLoading Survival data...\n');
data = load('survivaldatatrain.mat');
foldData = load('Folds_Survival.mat');

X_train = data.X_train; 
Y_train = data.Y_train;

external_fold = foldData.external_fold;
internal_fold = foldData.internal_fold; 
fprintf('Paused. Press enter to continue.\n');
pause;

%parameters for SVM 
field1 = 'C';  value1 = 0.125;
field2 = 'Sigma';  value2 = 50;
field3 = 'D';  value3 = 8;

bP_Survival = struct(field1,value1,field2,value2,field3,value3);

%initialize returned alpha fields
linP_alphas = [];
linD_alphas = [];
rbf_alphas = [];
poly_alphas = [];

%train Linear Primal SVM
fprintf('Training primal form linear SVM...\n');
bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, 'lin_primal')
[linP_alphas, w_lin_primal, b_lin_primal, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_primal', bestParam_lin_primal);
%Visualize results -routing borrowed from Coursera SVM plotting
pos_labels = find(Y_train == 1);
neg_labels = find(Y_train == -1);
x = 30:65;
y = (-w_lin_primal(1) * x - b_lin_primal)/w_lin_primal(2);
plot(X_train(pos_labels, 1), X_train(pos_labels, 2),'+', ...
     X_train(neg_labels, 1), X_train(neg_labels, 2),'o', x, y);

fprintf('Paused. Press enter to continue.\n');
pause;
%
bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, 'lin_dual')
[linD_alphas, w_lin_dual, b_lin_dual, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_dual', bestParam_lin_dual);
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf')
[rbf_alphas, w_rbf, b_rbf, sv_rbf, sv_labels_rbf] = trainSVM_model(X_train, Y_train, 'rbf', bestParam_rbf);
bestParam_poly = findBestParams(X_train, Y_train, external_fold, 'poly')
[poly_alphas, w_poly, b_poly, sv_poly, sv_labels_poly] = trainSVM_model(X_train, Y_train, 'poly', bestParam_poly);
%plot SVM result

svcplot(X_train,Y_train,'linear',linD_alphas,b_lin_dual);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_rbf.Sigma;
svcplot(X_train,Y_train,'rbf',rbf_alphas,b_rbf);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_poly.D;
svcplot(X_train,Y_train,'poly',poly_alphas,0);
fprintf('Paused. Press enter to continue.\n');
pause;

% ------------------------------------------------
load('survivaldatatrain.mat');
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', ...
   external_fold, internal_fold);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual', ...
    external_fold, internal_fold);
acc_rbf = expectedAccuracy(X_train, Y_train, 'rbf', ...
    external_fold, internal_fold);
acc_poly = expectedAccuracy(X_train, Y_train, 'poly', ...
    external_fold, internal_fold);

save('SurvivalModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');

clear all;

% For the Chess Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios

% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;
% ------------------------------------------------
% YOUR CODE HERE

%Load Data
fprintf('\nLoading Chess data...\n');
data = load('chessboarddatatrain.mat');
foldData = load('Folds_Chess.mat');

X_train = data.X_train; 
Y_train = data.Y_train;

external_fold = foldData.external_fold;
internal_fold = foldData.internal_fold; 
fprintf('Paused. Press enter to continue.\n');
pause;

%parameters for SVM 
field1 = 'C';  value1 = 0.125;
field2 = 'Sigma';  value2 = 50;
field3 = 'D';  value3 = 8;

bP_Survival = struct(field1,value1,field2,value2,field3,value3);

%initialize returned alpha fields
linP_alphas = [];
linD_alphas = [];
rbf_alphas = [];
poly_alphas = [];

%train Linear Primal SVM
fprintf('Training primal form linear SVM...\n');
bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, 'lin_primal')
[linP_alphas, w_lin_primal, b_lin_primal, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_primal', bestParam_lin_primal);
%Visualize results -routing borrowed from Coursera SVM plotting
pos_labels = find(Y_train == 1);
neg_labels = find(Y_train == -1);
x = 30:65;
y = (-w_lin_primal(1) * x - b_lin_primal)/w_lin_primal(2);
plot(X_train(pos_labels, 1), X_train(pos_labels, 2),'+', ...
     X_train(neg_labels, 1), X_train(neg_labels, 2),'o', x, y);
fprintf('Paused. Press enter to continue.\n');
pause;


bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, 'lin_dual')
[linD_alphas, w_lin_dual, b_lin_dual, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_dual', bestParam_lin_dual);
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf')
[rbf_alphas, w_rbf, b_rbf, sv_rbf, sv_labels_rbf] = trainSVM_model(X_train, Y_train, 'rbf', bestParam_rbf);
bestParam_poly = findBestParams(X_train, Y_train, external_fold, 'poly')
[poly_alphas, w_poly, b_poly, sv_poly, sv_labels_poly] = trainSVM_model(X_train, Y_train, 'poly', bestParam_poly);
%plot SVM result


svcplot(X_train,Y_train,'linear',linD_alphas,b_lin_dual);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_rbf.Sigma;
%svcplot(X_train,Y_train,'rbf',rbf_alphas,b_rbf);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_poly.D;
%svcplot(X_train,Y_train,'poly',poly_alphas,b_poly);
fprintf('Paused. Press enter to continue.\n');
pause;

% ------------------------------------------------
load('chessboarddatatrain.mat');
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', ...
    external_fold, internal_fold);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual', ...
    external_fold, internal_fold);
acc_rbf = expectedAccuracy(X_train, Y_train, 'rbf', ...
    external_fold, internal_fold);
acc_poly = expectedAccuracy(X_train, Y_train, 'poly', ...
    external_fold, internal_fold);

save('ChessModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');
clear all


% For the Iris Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios

% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;
% ------------------------------------------------
% YOUR CODE HERE

%Load Data
fprintf('\nLoading Iris data...\n');
data = load('iris1_v24.mat');
foldData = load('Folds_Iris.mat');

X_train = data.X_train; 
Y_train = data.Y_train;

external_fold = foldData.external_fold;
internal_fold = foldData.internal_fold; 
fprintf('Paused. Press enter to continue.\n');
pause;

%parameters for SVM 
field1 = 'C';  value1 = 0.125;
field2 = 'Sigma';  value2 = 50;
field3 = 'D';  value3 = 8;

bP_Survival = struct(field1,value1,field2,value2,field3,value3);

%initialize returned alpha fields
linP_alphas = [];
linD_alphas = [];
rbf_alphas = [];
poly_alphas = [];

%train Linear Primal SVM
fprintf('Training primal form linear SVM...\n');
bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, 'lin_primal')
[linP_alphas, w_lin_primal, b_lin_primal, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_primal', bestParam_lin_primal);
%Visualize results -routing borrowed from Coursera SVM plotting
pos_labels = find(Y_train == 1);
neg_labels = find(Y_train == -1);
x = 30:65;
y = (-w_lin_primal(1) * x - b_lin_primal)/w_lin_primal(2);
plot(X_train(pos_labels, 1), X_train(pos_labels, 2),'+', ...
     X_train(neg_labels, 1), X_train(neg_labels, 2),'o', x, y);

fprintf('Paused. Press enter to continue.\n');
pause;
%
bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, 'lin_dual')
[linD_alphas, w_lin_dual, b_lin_dual, sv_lin, sv_labels_lin] = trainSVM_model(X_train, Y_train, 'lin_dual', bestParam_lin_dual);
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf')
[rbf_alphas, w_rbf, b_rbf, sv_rbf, sv_labels_rbf] = trainSVM_model(X_train, Y_train, 'rbf', bestParam_rbf);
bestParam_poly = findBestParams(X_train, Y_train, external_fold, 'poly')
[poly_alphas, w_poly, b_poly, sv_poly, sv_labels_poly] = trainSVM_model(X_train, Y_train, 'poly', bestParam_poly);
%plot SVM result

svcplot(X_train,Y_train,'linear',linD_alphas,b_lin_dual);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_rbf.Sigma;
%svcplot(X_train,Y_train,'rbf',rbf_alphas,b_rbf);
fprintf('Paused. Press enter to continue.\n');
pause;
p1 = bestParam_poly.D;
%svcplot(X_train,Y_train,'poly',poly_alphas,b_poly);
fprintf('Paused. Press enter to continue.\n');
pause;

% ------------------------------------------------
load('iris1_v24.mat');
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', ...
    external_fold, internal_fold);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual', ...
    external_fold, internal_fold);
acc_rbf = expectedAccuracy(X_train, Y_train, 'rbf', ...
    external_fold, internal_fold);
acc_poly = expectedAccuracy(X_train, Y_train, 'poly', ...
    external_fold, internal_fold);

save('IrisModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');
clear all
end