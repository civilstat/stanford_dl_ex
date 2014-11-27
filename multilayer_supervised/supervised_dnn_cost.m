function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
% SUPERVISED_DNN_COST Cost function for supervised neural network
%   Does all the work of cost / gradient computation
%   Returns: total cost;
%     gradient for each W and b at each layer (as a stack);
%     and prediction probabilities for each class for each sample

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
% stack{1} has weights W and biases b that combine with data X
%   to form inputs to 1st hidden layer;
% stack{end} has W and b that combine with last hidden layer
%   to form output pred_probs (unnormalized)
stack = params2stack(theta, ei);
nrLayers = numel(ei.layer_sizes) + 1;
stackDepth = nrLayers - 1;
a = cell(nrLayers, 1); % 1st layer is data, last is output (unnormalized)
gradStack = cell(stackDepth, 1);
nrSamples = size(data, 2);

%% forward prop
%%% YOUR CODE HERE %%%
% For layer l = 2:nrLayers
%   (to be consistent with the tutorial notes:
%   with a single hidden layer,
%   X is layer 1, hidden is layer 2, output is layer 3;
%   so that's a stack depth of 2 = nr of layers BESIDES output)
%
a{1} = data; % a(1)=X;
for l = 2:nrLayers
  % z(l) = [W*a + b](l-1), for each sample,
  % where b is col.vector with same nr rows as W
  z = bsxfun(@plus, stack{l-1}.W*a{l-1}, stack{l-1}.b);
  if l == nrLayers
    a{l} = exp(z); % for last layer don't do full sigmoid, just exp, to calculate softmax below in pred_prob?
  else
    a{l} = f(z); % for other layers DO calculate sigmoid?
  end
end
% Normalize final layer, so each sample's outputs sum to 1,
% so that they are actual prediction probabilities.
% Final layer outputs are in a, with nrClasses rows and nrSamples cols,
% so we need something like (a ./ sum(a,1)), arranged right...
pred_prob = bsxfun(@rdivide, a{nrLayers}, sum(a{nrLayers},1));
% So far, this is not strictly necessary,
% since we'll just use the highest value's index as the prediction...
% but we'll need the normalization anyway
% for computing the loss function later.

%% return here if only predictions desired.
if po
  cost = -1;
  grad = [];  
  return;
end;


%% compute cost
%%% YOUR CODE HERE %%%

% Find linear equivalents of matrix indices (i,j)
% where, in j'th sample (column), i=y(j)
% i.e. the row ID is the class of that column
IDs = sub2ind(size(pred_prob), labels', 1:nrSamples);
% Compute binary matrix where IDs are 1 and rest are 0
y_matrix = zeros(size(pred_prob));
y_matrix(IDs) = 1;

% Cost function f=J(theta), NOT yet including L2 penalty on the weights
cost = -sum(log(pred_prob(IDs)));


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% For layers 2:nrLayers, we will need a delta vector
% (so 1st element of deltaStack will just stay empty)
deltaStack = cell(nrLayers, 1);
% The last layer's delta vector is just the residuals,
% NOT summed over samples like the tutorial mistakenly says.
% We want one delta per output unit AND per sample.
deltaStack{nrLayers} = pred_prob - y_matrix;

% Backpropagate to get deltas for previous layers
for l=((nrLayers-1):-1:2)
  % delta{l} = [W{l}' * delta{l+1}] .* [deriv of f wrt z{l}]
  % TODO: if generalizing to different function f,
  % replace the last product a.*(1-a) with the appropriate derivative of f at z
  deltaStack{l} = (stack{l}.W'*deltaStack{l+1}) .* (a{l}.*(1-a{l}));
end

% Compute gradient (sum of contributions of each sample)
%   at each layer, for W and b
%   (still WITHOUT the L2 penalty on the weights)
% NOTE: Take SUMS not MEANS since our loss function is a sum not a mean
% (unlike sq.err. loss example in tutorial)
for l=1:stackDepth
  % Grad of W{l}(i,j) is matrix of sums of delta{l+1}(i)*a{l}(j) across samples
  gradStack{l}.W = deltaStack{l+1} * a{l}'; %' % why doesn't it recognize the 1st ' as transpose?
  % For grad of b, take sum within rows of delta{l+1} i.e. across samples
  gradStack{l}.b = sum(deltaStack{l+1}, 2);
end


%% compute L2 weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
if ei.lambda ~= 0
  % For each layer with weights W in it...
  for l=1:stackDepth
    % Add to the cost: sum of squared weights multiplied by lambda/2
    cost = cost + (ei.lambda/2)*sum(sum(stack{l}.W .^ 2));
	% Add to the gradient: weight times lambda
	gradStack{l}.W = gradStack{l}.W + ei.lambda.*stack{l}.W;
  end
end


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


% Helper subfunctions:

% Let f be the sigmoid, logistic, or inverse-logit function.
% We can apply it element-wise.
function h=f(z)
  h = logsig(z);
end
% The derivative of sigmoid f is f*(1-f)
% (although we don't use this, since we store a=f(z) already)
function h=fPrime(z)
  g = f(z);
  h = f(z).*(1-f(z));
end