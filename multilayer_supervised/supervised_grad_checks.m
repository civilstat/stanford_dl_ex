%
% Check gradients for supervised multilayer neural net example
%

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath ../ex1; % also add ex1 directory for grad_check function


%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for L2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';
% Whether output is categorical (each sample's y is in 1:k if there are k categories)
% or continuous (each sample's y is a vector in (0,1) range)
ei.output_type = 'categorical';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);


% Check that the programmed gradient is similar to the numerical estimate:
% The "err" column should be near 0, basically just rounding error
grad_check(@supervised_dnn_cost, params, 10, ei, data_train, labels_train)



% also try checking if modified function works for squared error loss for autoencoder
ei.output_type = 'continuous';
ei.output_dim = ei.input_dim;
ei.layer_sizes = [256, ei.output_dim];
stack = initialize_weights(ei);
params = stack2params(stack);
grad_check(@supervised_dnn_cost, params, 10, ei, data_train, data_train)
% yes! it seems to work :)