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
depth = numel(ei.layer_sizes);
n_l = depth + 1;
numHidden = depth - 1;
hAct = cell(depth, 1);
gradStack = cell(depth, 1);

%% forward prop
%%% YOUR CODE HERE %%%
% For layer l = 2:nrLayers
%   (to be consistent with the notes:
%   with a single hidden layer,
%   X is layer 1, hidden is layer 2, output is layer 3;
%   and that's a depth of 2 = nr of layers BESIDES input)
a = data; % a(1)=X;
for l = 2:n_l
  % The actual indexing in the code runs 1:(nrLayers-1) i.e. 1:depth
  d = l-1; % d = depth level

  % z = W*a + b, for each sample,
  % where b is col.vector with same nr rows as W
  z = bsxfun(@plus, stack{d}.W*a, stack{d}.b);
  hAct{d} = f(z);
  a = hAct{d};
end
% Normalize final layer, so each sample's outputs sum to 1,
% so that they are actual prediction probabilities.
% Final layer outputs are in a, with nrClasses rows and nrSamples cols,
% so we need something like (a ./ sum(a,1)), arranged right...
pred_prob = bsxfun(@rdivide, a, sum(a,1));
% This is not strictly necessary so far,
% since we'll just use the highest value's index as the prediction...
% but we'll need the normalization anyway
% for computing the loss function later.

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;


%% compute cost
%%% YOUR CODE HERE %%%

% Find linear equivalents of matrix indices (i,j)
% where, in j'th sample (column), i=y(j)
% i.e. the row ID is the class of that column
IDs = sub2ind(size(pred_prob), labels, 1:length(labels));
% Compute binary matrix where IDs are 1 and rest are 0
y_matrix = zeros(size(pred_prob));
y_matrix(IDs) = 1;

% Cost function f=J(theta)
cost = -sum(log(pred_prob(IDs)));


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% For layers 2:nrLayers, we will need a delta vector
deltaStack = cell(depth, 1);
% The last layer's delta vector is just the residuals,
% apparently summed over samples? Double-check this...
% It does sound like we want one delta per output unit;
% but from softmax example, it seems like we should be
% multiplying the residuals by X or by a(depth-1) before summing...
deltaStack{depth} = sum(pred_prob - y_matrix, 2);



%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


% Helper subfunctions:

% Let f be the sigmoid, logistic, or inverse-logit function.
% We can apply it element-wise.
function h=f(a)
  h = logsig(a);
end
% The derivative of sigmoid f is f*(1-f)
function h=fPrime(a)
  g = f(a);
  h = f(a).*(1-f(a));
end