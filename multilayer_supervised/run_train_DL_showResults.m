%%% SETUP %%%

load('Outcomes_28-Nov-2014.mat');

% In the run loaded above,
% we hadn't finished running the largest layer-size for 3 hidden layers,
% or any simulations for 4 hidden layers.
% At some point we should run the rest,
% or at least the smaller layer-sizes for 4 hidden layers.

% All runs used early stopping at 50 batch gradient descent iterations per layer,
% rather than running to convergence.

% Parameters used in the simulation run loaded above:
LabelDim = 10; % nr of classes, i.e. nr of units in last (unhidden) layer
% We have runs for nrHiddenLayers in this set:
NrsOfHiddenLayers = 1:4;
% and for layerSize in this set:
LayerSizes = [8 10 16 28].^2;



%%% COMPARE TEST ACCURACIES ACROSS THE EXPERIMENT %%%

% Make simple matrices of the test accuracies
NoPretrainTestAcc = zeros(4,4);
UnsupPretrainTestAcc = zeros(4,4);
for nrLayers=1:4
	for layerSizeID=1:4
		NoPretrainTestAcc(nrLayers,layerSizeID) = Outcomes.NoPretrain{nrLayers,layerSizeID}.testAcc;
		UnsupPretrainTestAcc(nrLayers,layerSizeID) = Outcomes.UnsupPretrain{nrLayers,layerSizeID}.testAcc;
	end
end
% Look at the results so far
NoPretrainTestAcc
UnsupPretrainTestAcc
% For 1 or 2 hidden layers (rows 1:2),
%   NoPretraining always has similar or better test accuracy as UnsupPretraining.
% But for 3 hidden layers, at the smaller layer sizes (64 and 100),
%   UnsupPretraining DOES improve accuracy substantially over NoPretraining,
%   though it's worse again for the larger layer size (256).
% These results are unlike what we saw in the paper.
% However, all of this may be substantially affected by the early-stopping after 50 iterations.



%%% EXAMPLE CODE FOR VISUALIZING THE WEIGHTS %%%

% Ideally, for each unit in each layer,
% we should plot the input that maximizes the activation of that unit.
% But as a first attempt, we just plot the weights (for the first layer),
% or the product of the weights up to that layer (for higher layers).
% This looks somewhat OK, but not as good as what's in the paper.

% Choose a combination of layer size and nr of layers to plot
layerSizeID = 1; % Display results for experiment with this layer size
nrLayers = 3; % Display results for experiment with this number of layers
layerSize = LayerSizes(layerSizeID);
stackTemp = 1;
% Make a new figure for each layer
for layer=1:nrLayers
  figure(layer)
  colormap(gray)
  % For higher hidden layers, multiply weights by previous layers' weights for display
  stackTemp = Outcomes.UnsupPretrain{nrLayers,layerSizeID}.optStack{layer}.W * stackTemp;
  % In the current layer's figure, make a new subplot for each unit
  for unit=1:layerSize
	subplot(sqrt(layerSize),sqrt(layerSize),unit)
	weightsTemp = stackTemp(unit,:);
	% Standardize the weights within each unit, so they are comparable across units
	% (but then also need to ensure that MATLAB uses the same colorbar for each subplot)
	weightsTemp = weightsTemp / sum(weightsTemp.^2);
	sqrtWeightLength = sqrt(length(weightsTemp));
	imagesc(reshape(weightsTemp, sqrtWeightLength, sqrtWeightLength))
	set(gca,'xtick',[]); set(gca,'ytick',[]);
	minTemp = min(min(weightsTemp));
	maxTemp = max(max(weightsTemp));
  end
  % Give each unit the same colorbar
  for unit=1:layerSize
	subplot(sqrt(layerSize),sqrt(layerSize),unit)
	caxis([minTemp maxTemp])
  end
  suptitle(['UnsupPretrain: ' num2str(nrLayers) ' layers, ' num2str(layerSize) ' units: layer ' num2str(layer)])
end

% In the code above,
% replace UnsupPretrain with NoPretrain
% to visualize the weights without pretraining