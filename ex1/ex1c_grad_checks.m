%
% Check gradients for softmax regression example
%

addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
train.y = train.y+1; % make labels 1-based.
n=size(train.X,1);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
% We only use num_classes-1 columns, since the last column is always assumed 0.
theta = rand(n,num_classes-1)*0.001;

% Check that the programmed gradient is similar to the numerical estimate:
% The "err" column should be near 0, basically just rounding error
grad_check(@softmax_regression_vec, theta, 10, train.X, train.y)

