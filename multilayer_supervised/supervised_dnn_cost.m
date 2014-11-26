function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%
% For layer l = 2:nrLayers
%   (to be consistent with the notes:
%   with a single hidden layer,
%   X is layer 1, hidden is layer 2, output is layer 3)
a = data; % a(1)=X;
for l = 2:(numel(ei.layer_sizes)+1)
  % But the actual indexing in the code runs 1:(nrLayers-1)
  d = l-1; % d = depth level?
  % z = W*a + b, for each sample,
  % where b is col.vector with same nr rows as W
  z = bsxfun(@plus, stack{d}.W*a, stack{d}.b);
  hAct{d} = f(z);
  a = hAct{d};
end
% Normalize final layer, so each sample's outputs sum to 1,
% so that they are actual prediction probabilities.
% Final layer outputs are in a, with nrClasses rows and nrSamples cols,
% so we need something like (a ./ sum(a,1))...
%pred_prob = bsxfun(@rdivide, a, sum(a,1));
% Actually, this is not strictly necessary,
% since we'll just use the highest value's index as the prediction,
% so it's OK to skip the normalization
pred_prob = a;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


% Helper subfunctions:

% Let f be the sigmoid, logistic, or inverse-logit function.
% We can apply it element-wise.
function h=f(a)
  h=logsig(a);
end