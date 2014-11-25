%
% Check gradients for linear and logistic regression examples
%

addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

%%% LINEAR REGRESSION %%%

% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];

% Shuffle examples.
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1,1:400);
train.y = data(end,1:400);

% Initialize the coefficient vector theta to random values.
n=size(train.X,1);
theta = rand(n,1);

% Check that the programmed gradient is similar to the numerical estimate:
% The "err" column should be near 0, basically just rounding error
grad_check(@linear_regression_vec, theta, 10, train.X, train.y)


%%% LOGISTIC REGRESSION %%%

% Load the MNIST data for this exercise.
binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];

% Initialize theta to some small random values.
n=size(train.X,1);
theta = rand(n,1)*0.001;

% Check that the programmed gradient is similar to the numerical estimate:
grad_check(@logistic_regression_vec, theta, 10, train.X, train.y)


