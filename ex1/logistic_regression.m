function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

  n=size(X,1);
  for j=1:m  % m = nr of samples
    % For each sample, compute the predictor h_theta(x) i.e. p.hat
	h_j = 1/(1+exp(-theta'*X(:,j)));
    % For each sample, add its contribution to the penalty function f=J(theta)
    f = f - (y(j)*log(h_j) + (1-y(j))*log(1-h_j));
	for i=1:n  % n = nr of parameters
	  % For each parameter, add this sample's contribution to its gradient g=J'(theta)
	  g(i) = g(i) + X(i,j)*(h_j - y(j));
	end
  end
  
  % This unvectorized version took 1 hour to run on my laptop.

end % of the whole function