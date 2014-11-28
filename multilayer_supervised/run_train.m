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
% Whether output is categorical (each sample's y is in 1:k if there are k categories)
% or continuous (each sample's y is a vector in (0,1) range)
ei.output_type = 'categorical'; % or 'continuous';

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
	% (Had to modify the supervised_dnn_cost.m code
	%  because our softmax regression assumes that y is a vector, not a matrix:
	%  for example e, y(e) = k = which class example e belongs to,
	%  but X(:,e) is a continuous vector with values in (0,1) range.
	%  But that's done now!)
	
	
	% The simplest interesting DL things to try would be:
	% Run optimization once with X as both the inputs AND outputs,
	%   then use those estimates as the initial param values
	%   and rerun optimization with Y as the output.
	% How does it affect train and test accuracy?
	% How does it change with nr and sizes of hidden layers?
	'Trying autoencoder: Can we get X back with one hidden layer of size 100?'
	ei.output_type = 'continuous';
	ei.output_dim = ei.input_dim;
	ei.layer_sizes = [100, ei.output_dim];
	stack = initialize_weights(ei);
	params = stack2params(stack);
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
		params,options,ei, data_train, data_train);
	% It didn't finish converging -- just ran to max of 500 iterations
	[~, ~, pred_train] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
	[~, ~, pred_test] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
	train_rms=sqrt(mean(mean((pred_train - data_train).^2)));
	fprintf('RMS training error: %f\n', train_rms);
	test_rms=sqrt(mean(mean((pred_test - data_test).^2)));
	fprintf('RMS testing error: %f\n', test_rms);
	% RMS train and test error are both around 0.062
	% while mean of train and test data are around 0.13
	% so the prediction RMS isn't TOTALLY hopeless.
	'Saving the opt_params from this autoencoder'
	opt_params_autoencoder = opt_params;
	opt_stack_autoencoder = params2stack(opt_params_autoencoder, ei);
	
	% Try to visualize the inputs and predictions:
	figure(1); imagesc(reshape(data_train(:,1), 28, 28))
	figure(2); imagesc(reshape(pred_train(:,1), 28, 28))
	figure(3); imagesc(reshape(data_test(:,1), 28, 28))
	figure(4); imagesc(reshape(pred_test(:,1), 28, 28))
	% Yep, the reconstructions look pretty decent,
	% even though it didn't finish converging.
	
	'Now trying to use those parameter estimates from P(X|X)'
	'as intial values for supervised learning of P(Y|X)'
	ei.output_type = 'categorical';
	ei.output_dim = 10;
	ei.layer_sizes = [100, ei.output_dim];
	stack = initialize_weights(ei);
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
	% Converged in 141 steps, train acc 100%, test acc 95.51%
	% so the pre-training did NOT help the test accuracy at all,
	% but it DID help the train acc over the 100-unit version
	% (though it's no better than the default 256-unit version at the top)
	
	
	% So now we have opt_params from the pre-trained AND supervised network,
	% as well as opt_params_autoencoder from the pre-training run alone.
	% Can we visualize the inputs that maximize each of the hidden layer units?
	%   The tutorial says to rescale the weights, but it seems like by a sum which is constant over pixels...?
	%   Yes, but it's a different scaling factor for each UNIT,
	%   so we might as well rescale them for simultaneous display's sake.
	% Let's show 2 images from the autoencoder,
	% and 2 from the fully trained network:
	opt_stack = params2stack(opt_params, ei);
	autoencW = zeros(100,784);
	fulltrainW = zeros(100,784);
	% Row: which unit?
	% Col: which pixel within that unit?
	for i=1:100
		autoencW(i,:) = opt_stack_autoencoder{1}.W(i,:);
		autoencW(i,:) = autoencW(i,:) / sum(autoencW(i,:).^2);
		fulltrainW(i,:) = opt_stack{1}.W(i,:);
		fulltrainW(i,:) = fulltrainW(i,:) / sum(fulltrainW(i,:).^2);
	end
	autoencMin = min(min(autoencW));
	autoencMax = max(max(autoencW));
	fulltrainMin = min(min(fulltrainW));
	fulltrainMax = max(max(fulltrainW));
	for i=1:100
		figure(1)
		subplot(10,10,i)
		imagesc(reshape(autoencW(i,:), 28, 28))
		set(gca,'xtick',[]); set(gca,'ytick',[]); caxis([autoencMin autoencMax])
		figure(2)
		subplot(10,10,i)
		imagesc(reshape(fulltrainW(i,:), 28, 28))
		set(gca,'xtick',[]); set(gca,'ytick',[]); caxis([fulltrainMin fulltrainMax])
	end
	% Fascinating! When we look at the autoencoder in Fig 1,
	% a few look like real signal and the rest look like white noise.
	% Then, looking at the fully-trained Fig 2,
	% all those that looked good in Fig 1 still look the same,
	% while those that looked like noise in Fig 1 look much less noisy
	% (though still not as clear as the original "good" ones).
	% ...
	% although that's when we let each subplot scale to its own min and max.
	% When we scale all subplots within a figure to the same min and max,
	% the units in Fig 2 that looks like white noise in Fig 1
	% now just look like almost nothing at all --
	% they've been smoothed spatially, but also dampened a lot towards 0 overall.
	% ...
	% What if instead of min and max, we use +/- of max(abs(min,max))?
	% Then at least we know that green is neutral.
	%   autoencMM = max(abs(autoencMin), abs(autoencMax));
	%   fulltrainMM = max(abs(fulltrainMin), abs(fulltrainMax));
	%   caxis([-autoencMM autoencMM]);
	%   caxis([-fulltrainMM fulltrainMM]);
	% Nope, that doesn't look great either. Oh well.
	% ...
	% I need to find a better way to plot these:
	% have a neutral color at 0, and diverging colors towards the + and - extremes,
	% such that the "good" plots look similar in BOTH figures.
	% But that's tough, since the max in figure 1 is in a white-noise subplot, not a "good" subplot,
	% so we can't just use the max of the data to scale things. Argh.
	
	
	% ANYWAY! The autoencoder seems to work, and seems to do something productive :)
	% 
	% Last step at this stage should be to rerun the training WITHOUT pretraining,
	% and visualize THOSE hidden units too, and see if any of them look
	% as good as 'the good ones' from the pretrained network
	% (on the same set of train/test data, just with new random-init params).
	% 
	opt_params_fulltrain = opt_params;
	opt_stack_fulltrain = opt_stack;
	'Now retrying supervised learning of P(Y|X), with NO pretraining'
	ei.output_type = 'categorical';
	ei.output_dim = 10;
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
	% Converged in 122 steps, train acc 100%, test acc 96.3%
	opt_params_justtrain = opt_params;
	opt_stack_justtrain = params2stack(opt_params_justtrain, ei);
	justtrainW = zeros(100,784);
	% Row: which unit?
	% Col: which pixel within that unit?
	for i=1:100
		justtrainW(i,:) = opt_stack_justtrain{1}.W(i,:);
		justtrainW(i,:) = justtrainW(i,:) / sum(justtrainW(i,:).^2);
	end
	for i=1:100
		figure(3)
		subplot(10,10,i)
		imagesc(reshape(justtrainW(i,:), 28, 28))
		set(gca,'xtick',[]); set(gca,'ytick',[]);
	end
	% OK, it's really hard to tell (without good colormap)
	% whether this is just noise or actually something useful. Argh.
	% But I *THINK* it looks different at least in the sense that
	% there are none of those "good"-looking units that we got in pretraining.
	% Maybe the pretrainer just needs to run longer?
	
	% FINALLY, we could also redo the sq.err. loss and the gradient
	% into KL-divergence loss, which requires changing the cost penalty
	% and also the delta terms in the gradient...


	% Then we can move to the next stage, of trying several different network architectures,
	% with and without pretraining, so we can test the hypotheses from the paper.
	% Also try SUPERVISED pretraining and see if it differs much.
end
