function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %       y is a row vector.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x (num_classes-1).
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  % m = nr of samples
  % n = nr of parameters
  % num_classes = nr of categories
  
  % Compute numerator of each class's contribution to p_hat, for each sample.
  % Include the last category as well: its thetas are all 0, so exp(0)=1.
  expThetaX = [exp(theta'*X); ones(1,m)];  % matrix of size num_classes * m
  
  % Sum the numerators to get denominator (normalizing constant), for each sample
  denom = sum(expThetaX,1);  % row vector of length m
  
  % Compute fraction that is each class's probability, for each sample
  p_hat = bsxfun(@rdivide, expThetaX, denom); % matrix of size num_classes * m
    
  % Store binary matrix where, in j'th sample (column),
  % the y(j)'th element is 1 and rest are 0
  IDs = sub2ind(size(p_hat), y, 1:m);
  y_matrix = zeros(size(p_hat));
  y_matrix(IDs) = 1;
  
  % Penalty function f=J(theta)
  f = -sum(log(p_hat(IDs)));

  % Compute "residuals": for each class and j'th sample, we get 
  %   p_hat if j'th sample is not that class (y(j) = 0), or
  %   (p_hat-1) if j'th sample is that class (y(j) = 1),
  %   so should be small if p_hat is good
  resid = p_hat - y_matrix;
  
  % Gradient g=J'(theta)
  g = X*resid'; % matrix of size n * num_classes

  % Drop last row of g
  % (since we assume last theta is 0, so don't need a gradient for it)
  g = g(:, 1:(num_classes-1));
  % and make gradient a vector for minFunc
  g=g(:);

  % This vectorized version took 90 sec to run on my laptop
  % and did not converge but ran the full 200 iterations
  % though seemed to get good performance
  % (around 90%-95% classification accuracy on both train and test)
  % ...though when I let it run longer, it does converge in ~950 iters,
  % but with about the same train and test performance.

end