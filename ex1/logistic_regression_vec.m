function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %       y is a row vector.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%

  h = 1./(1+exp(-theta'*X));
  resid = h - y;
  f = -(y*log(h)' + (1-y)*log(1-h)'); % penalty function f=J(theta)
  g = X*resid'; % gradient g=J'(theta)

  % This vectorized version took 4 sec to run on my laptop: 800x faster than looping!

end