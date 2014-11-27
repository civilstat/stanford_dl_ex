% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

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
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;  % just do a few iters for testing, else set to 1e6
options.Method = 'lbfgs';

%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);

% Woohoo! I got it to work with 100% training accuracy, 96.88% test accuracy
% (converging in 101 iterations)
% with the suggested setup:
%   ei.layer_sizes = [256, ei.output_dim];
%   ei.lambda = 0;

if false
	% If instead I set
	%   ei.lambda = 1;
	% I get the same results,
	% so this seems to be an insignificantly small lambda.
	% To confirm that lambda does have an effect, let's retry with
	%   ei.lambda = 1000;
	% but let's restart from the last optimal param values
	% to hopefully converge a bit faster.
	ei.lambda = 1000;
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		opt_params,options,ei, data_train, labels_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	% It took longer to converge: stopped after 500 steps
	% but with train acc 87%, test acc 88%, so not too bad.
	
	% Should we try again with intermediate lambda?
	ei.lambda = 10;
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		opt_params,options,ei, data_train, labels_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	% Again, stopped after max of 500 steps
	% but with train acc 99%, test acc 98%, so much better.
	
	% OK, enough of playing with lambda; seems like it's working properly.
	% Reset to lambda=0;
	ei.lambda = 0;


	% Does it also run and work well with fewer units in the hidden layer?
	'Trying a smaller hidden layer of size 100 instead of 256'
	'Using new random-init params'
	ei.layer_sizes = [100, ei.output_dim];
	stack = initialize_weights(ei);
	params = stack2params(stack);
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		params,options,ei, data_train, labels_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	% Yes: converged in 116 steps, train acc 100%, test acc 96%
	
	
	% What about with more hidden layers?
	'Trying two smaller hidden layers, both of size 100'
	'Using new random-init params'
	ei.layer_sizes = [100, 100, ei.output_dim];
	stack = initialize_weights(ei);
	params = stack2params(stack);
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		params,options,ei, data_train, labels_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	% Yes! It works! Awesome :)
	% Converged in 146 steps, train acc 100%, test acc 96%
	% So the extra layer didn't really help it,
	% BUT at least the code clearly works with the extra layer.
	
	% Awesome. Now can we try autoencoder?
	% AH NO, the code below won't work,
	% because our softmax regression assumes that y is a vector, not a matrix:
	% for example e, y(e) = k = which class example e belongs to,
	% but X(:,e) is a continuous vector with values in (0,1) range.
	% TODO: we'll have to modify the kind of outputs we can handle
	% before we can run the code below. 
	% But we're getting closer!
	
	
	% The simplest interesting DL things to try would be:
	% Run optimization once with X as both the inputs AND outputs,
	%   then use those estimates as the initial param values
	%   and rerun optimization with Y as the output.
	% How does it affect train and test accuracy?
	% How does it change with nr and sizes of hidden layers?
	'Trying autoencoder: Can we get X back with one hidden layer of size 100?'
	ei.output_dim = ei.input_dim;
	ei.layer_sizes = [100, ei.output_dim];
	stack = initialize_weights(ei);
	params = stack2params(stack);
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		params,options,ei, data_train, data_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	'Saving the opt_params from this autoencoder'
	opt_params_autoencoder = opt_params;
	
	'Now trying to use those parameter estimates from P(X|X)'
	'as intial values for supervised learning of P(Y|X)'
	ei.output_dim = 10;
	ei.layer_sizes = [100, ei.output_dim];
	stack = initialize_weights(ei);
	opt_stack_autoencoder = params2stack(opt_stack_autoencoder);
	% TODO: how to modify the weights in this stack
	%       so that the bottom-most weights are the ones
	%       trained by the autoencoder of X?
	stack{1} = opt_stack_autoencoder{1};
	params = stack2params(stack);
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		params,options,ei, data_train, labels_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);

end
