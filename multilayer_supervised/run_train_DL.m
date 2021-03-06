% Train multilayer neural networks on MNIST data
% with various network architectures, with and without pre-training,
% modifying:
% - number of hidden layers (1:4)
% - number of hidden units per layer (8^2, 10^2, 16^2, 28^2)
% - whether pretraining is supervised or unsupervised
% and saving the results:
% - training and testing accuracies
% - weights at each layer of the neural network
% so we can investigate:
% - whose final training and/or test error is lower
% - what the maximal-activation inputs look like for all sets of units

% So we have 3 experimental factors to manipulate
% (UnsupPretrain vs NoPretrain; NrsOfHiddenLayers; LayerSizes)
% for a total of 2x4x4 = 32 combinations.
% Ideally we should do several simulation runs at each combination of the factors,
% but it is so slow that right now I am only doing one simulation run for each setting.

% To match the paper exactly, we ought to use denoising autoencoders,
% with a different cost function than sq.err.loss,
% and with what seems to be dropout noise on the X's?...
% but that would all require substantial modification,
% so I hope that what we have will be good enough for now.

% Also, the paper mentions "the number of ... passes through the data (epochs) is 50 ... per layer"
% so are they stopping the gradient descent after 50 iterations?
% Googling suggests that an "epoch" is indeed what I think of as an iteration.
% For the sake of runtime I will stop mine after 50 iters,
% since otherwise it will take forever to run anything,
% but in that case we should note that our simulation results use "early stopping".



%%% SETUP THE ENVIRONMENT / EXPERIMENT INFO %%%

LabelDim = 10; % nr of classes, i.e. nr of units in last (unhidden) layer
NrsOfHiddenLayers = 1:4; % for nrHiddenLayers in this set...
LayerSizes = [8 10 16 28].^2; % for layerSize in this set...
Outcomes.NoPretrain = cell(length(NrsOfHiddenLayers), length(LayerSizes));
Outcomes.UnsupPretrain = cell(length(NrsOfHiddenLayers), length(LayerSizes));
% In the Outcomes object, we'll store the following info
% (where <Whether> is either No or Unsup):
%   outcomes.<Whether>Pretrain{nrHiddenLayers,layerSize}.trainAcc,
%   outcomes.<Whether>Pretrain{nrHiddenLayers,layerSize}.testAcc,
%   outcomes.<Whether>Pretrain{nrHiddenLayers,layerSize}.optStack,
%   outcomes.UnsupPretrain{nrHiddenLayers,layerSize}.autoencoderStack
% so that we can plot the train and test accuracies,
% and we can display the Weight heatmaps for the optStacks
% (and compare them to before and after the supervised 'post-training')



% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 50;  % Erhan et al. paper seems to stop after 50 iters per layer
options.Method = 'lbfgs';

% a struct containing network layer sizes etc
ei = [];
%% populate ei with the network architecture to train:
% dimension of input features
ei.input_dim = 784;  % input images are always 28x28 pixels
% scaling parameter for L2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% (currently we've implemented support for only the logistic sigmoid function)
ei.activation_fun = 'logistic';
% For each run of the experiment, we'll also need to specify
% ei.output_dim, ei.layer_sizes, and ei.output_type;
% please see examples below.
% We'll also need the fully-specified ei for each run
% to compute the predictions for getting the accuracy of each model.



%%% RUN THE EXPERIMENTS AND SAVE RESULTS %%%

for nrHiddenLayers = NrsOfHiddenLayers
  for layerSizeID = 1:length(LayerSizes)
    layerSize = LayerSizes(layerSizeID);
	
	% Set up the overall random initial values,
	% first for supervised stack and then for autoencoder
	
	% Number of output classes:
	eiSup = ei;
	eiSup.output_dim = LabelDim;
	% sizes of all hidden layers and the output layer
	eiSup.layer_sizes = [repmat(layerSize, 1, nrHiddenLayers), eiSup.output_dim];
	% Whether output is categorical (each sample's y is in 1:k if there are k categories)
	% or continuous (each sample's y is a vector in (0,1) range)
	eiSup.output_type = 'categorical'; % or 'continuous';
	%% setup random initial weights
	stackInit = initialize_weights(eiSup);
	paramsInit = stack2params(stackInit);
	
	% Copy the above randomly-initialized weights stack for the full autoencoder,
	% redoing the weight-initialization just to get a top layer of the correct size
	% and saving that new top layer in the initial params for the autoencoder
	eiUnsupPretrain = ei;
	eiUnsupPretrain.output_dim = eiUnsupPretrain.input_dim;
	eiUnsupPretrain.layer_sizes = [repmat(layerSize, 1, nrHiddenLayers), eiUnsupPretrain.output_dim];
	eiUnsupPretrain.output_type = 'continuous';
	stackTemp = initialize_weights(eiUnsupPretrain);
	stackInitAutoencoder = stackInit;
	stackInitAutoencoder{nrHiddenLayers+1} = stackTemp{nrHiddenLayers+1};
	paramsInitAutoencoder = stack2params(stackInitAutoencoder);
		
	% So, both the pre-trained and non-pre-trained have the same init values where possible,
	% to make their outputs more comparable.
	
    % Also create a placeholder for where the pretrained weights will be stored
	stackTrainedAutoencoder = stackInit;

	% FIRST, TRAIN THE SUPERVISED MODEL WITHOUT ANY PRETRAINING
	disp(['FIRST, WITHOUT PRETRAINING: nrHiddenLayers=', num2str(nrHiddenLayers), ...
	     ', layerSize=', num2str(layerSize)])
	%% run training
	tic
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost, ...
		paramsInit, options, eiSup, data_train, labels_train);
	toc
	%% compute accuracy on the train and test sets
	[~, ~, pred] = supervised_dnn_cost( opt_params, eiSup, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, eiSup, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	%% Store outcomes
	Outcomes.NoPretrain{nrHiddenLayers, layerSizeID}.trainAcc = acc_train;
	Outcomes.NoPretrain{nrHiddenLayers, layerSizeID}.testAcc = acc_test;
	Outcomes.NoPretrain{nrHiddenLayers, layerSizeID}.optStack = params2stack(opt_params, eiSup);

	
	% SECOND, RUN UNSUPERVISED PRETRAINING AND THEN REFIT THE MODEL
	disp(['SECOND, WITH UNSUPERVISED PRETRAINING: nrHiddenLayers=', num2str(nrHiddenLayers), ...
	     ', layerSize=', num2str(layerSize)])
	disp('Pretrain with unsupervised autoencoder')
	% Greedy layer-by-layer pretraining:
	% First, set up and run unsup. pretraining from the data to the 1st hidden layer
	eiLayer = eiUnsupPretrain;
	eiLayer.layer_sizes = [layerSize, eiLayer.output_dim]; % just the first hidden layer and the output
	stackLayer = cell(2,1);
	stackLayer{1} = stackInitAutoencoder{1};
	stackLayer{2} = stackInitAutoencoder{nrHiddenLayers+1};
	paramsLayer = stack2params(stackLayer);
	tic
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost, ...
		paramsLayer, options, eiLayer, data_train, data_train); % let the labels be just the data again
	toc
	stackTemp = params2stack(opt_params, eiLayer);
	stackTrainedAutoencoder{1} = stackTemp{1};
	predLayer = logsig(bsxfun(@plus, stackTrainedAutoencoder{1}.W*data_train, stackTrainedAutoencoder{1}.b));
	
	% Do the next layers, if there are others
	if nrHiddenLayers > 1
		eiLayer.input_dim = layerSize; % the next inputs will be outputs from the prev hidden layer
		for layer = 2:nrHiddenLayers
			% Leave stackLayer{2} at the same random inits as before (to connect to the output)
			% but change stackLayer{1} to be the random-init weights into the current layer:
			stackLayer{1} = stackInitAutoencoder{layer};
			paramsLayer = stack2params(stackLayer);
			% Train on the predictions from previous training,
			% and again let the labels be just the data
			tic
			[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost, ...
				paramsLayer, options, eiLayer, predLayer, data_train);
			toc
			% Save the trained weights for later use
			stackTemp = params2stack(opt_params, eiLayer);
			stackTrainedAutoencoder{layer} = stackTemp{1};
			% compute new predictions, using prev layer's predictions as inputs
			predLayer = logsig(bsxfun(@plus, stackTrainedAutoencoder{layer}.W*predLayer, ...
				stackTrainedAutoencoder{layer}.b));
		end
	end
	
	% Should there be an extra step, of training just a mini-model,
	% from the last hidden layer to the real labels?
	% (That would just be a simple softmax regression, right?)
	% Or is it OK to revert back to the full model without pretraining those weights too?
	% Let's skip doing the final supervised layer separately, and just do it all together.

	% So, by now we have trained each layer of stackTrainedAutoencoder
	% (except the last, i.e. the supervised layer whose outputs are the labels).
	% We can feed them all together into one full supervised multilayer network:
	% Revert to supervised setup with these stackTrainedAutoencoder params as initial weights,
	% and retrain the full model at once (not greedily layer-by-layer)
	disp('Train supervised model starting with previously-found weights')
	paramsTrainedAutoencoder = stack2params(stackTrainedAutoencoder);
	tic
	[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost, ...
		paramsTrainedAutoencoder, options, eiSup, data_train, labels_train);
	toc
	[~, ~, pred] = supervised_dnn_cost( opt_params, eiSup, data_train, [], true);
	[~,pred] = max(pred);
	acc_train = mean(pred'==labels_train);
	fprintf('train accuracy: %f\n', acc_train);
	[~, ~, pred] = supervised_dnn_cost( opt_params, eiSup, data_test, [], true);
	[~,pred] = max(pred);
	acc_test = mean(pred'==labels_test);
	fprintf('test accuracy: %f\n', acc_test);
	Outcomes.UnsupPretrain{nrHiddenLayers, layerSizeID}.trainAcc = acc_train;
	Outcomes.UnsupPretrain{nrHiddenLayers, layerSizeID}.testAcc = acc_test;
	Outcomes.UnsupPretrain{nrHiddenLayers, layerSizeID}.autoencoderStack = stackTrainedAutoencoder;	
	Outcomes.UnsupPretrain{nrHiddenLayers, layerSizeID}.optStack = params2stack(opt_params, eiSup);	
	
	
	% We could also add SUPERVISED pretraining here at the end,
	% for comparison with NO and SUPERVISED pretraining.
	
	% Save the outputs as we go
	save(['Outcomes_' date '.mat'], 'Outcomes');
	
  end
end

