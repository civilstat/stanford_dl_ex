function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %           theta is a column vector.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %       y is a row vector.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%

  y_hat = theta'*X; % y_hat is a row vector
  resid = y_hat - y;
  f = 0.5*resid*resid'; % penalty function f=J(theta)
  g = X*resid'; % gradient g=J'(theta)

  % This vectorized version took .1 sec to run on my laptop.

end