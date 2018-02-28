function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h0 = sigmoid(X*theta);
theta1 = theta(2:end);
reg = (lambda/(2*m))*sum(theta1*theta1);


J = (1/m)*sum((-y*log(h0)-(1-y)*log(1-h0))) + reg

theta_reg = theta;
theta_reg(1)= 0;

 grad = (X' * (h0 - y) ./ m) + theta_reg * lambda / m;




% =============================================================

end
