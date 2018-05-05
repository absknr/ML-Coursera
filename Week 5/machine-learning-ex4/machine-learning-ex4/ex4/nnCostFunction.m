function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1), X];

% calculate a2 - units of 2nd layer
a2 = sigmoid(Theta1 * X');

% Add ones to the a2 data matrix
a2 = [ones(1, m); a2];

% calculate a3 = units of 3rd layer 
h = sigmoid(Theta2 * a2);

% calculating cost(unregularized)
for i = 1 : m
	
	y_i = zeros(num_labels , 1);
	y_i(y(i)) = 1;
	
	J = J + sum((y_i .* log(h(:, i))) + ((ones(num_labels, 1) - y_i) .* log(ones(num_labels, 1) - h(:, i))));

end

J = (-1 / m) * J;

% calculating cost(regularized)

% Theta1 and Theta2 without the bias units. Theta(unbiased)
Theta1_u = Theta1(: , 2 : input_layer_size + 1);
Theta2_u = Theta2(: , 2 : hidden_layer_size + 1);

J = J + (lambda / (2 * m)) * (sum(sum(Theta1_u .^ 2 , 2)) + sum(sum(Theta2_u .^ 2 , 2)));

% Backprop

for i = 1 : m
	
	% Forward pass
	
	
	% units of 2nd layer for i'th training example
	a2_i = sigmoid(Theta1 * X(i , :)'); % (25 X 1) = (25 X 401) * (401 X 1)

	% Add 1
	a2_i = [1; a2_i];  % (26 X 1)
	
	% hypothesis for i'th training example
	h_i = sigmoid(Theta2 * a2_i); % (10 X 1) = (10 X 26) * (26 X 1)

	% Backpropogation
	
	% generating the vector form of label for each training example (10 X 1)
	y_i = zeros(num_labels , 1);
	y_i(y(i)) = 1;  
	
	% delta vector of output layer
	delta_3 = h_i - y_i; % (10 X 1)
	
	% delta vector of hidden layer 
	delta_2 = (Theta2' * delta_3) .* (a2_i .* (ones(size(a2_i)) - a2_i));  %(26 X 1) = ((26 X 10) * (10 X 1)) .* (26 X 1)
	
	Theta1_grad = Theta1_grad + delta_2(2 : end) * X(i , :);  % (25 X 401) = (25 X 401) + ((25 X 1) * (1 X 401))
	
	Theta2_grad = Theta2_grad + delta_3 * a2_i';  % (10 X 26) = (10 X 26) + (10 X 1) * (1 X 26)

end

% unregularized gradients

Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;


% regularized gradients
Theta1_grad(: , 2 : end) = Theta1_grad(: , 2 : end) + (lambda / m) * Theta1(: , 2 : end);
Theta2_grad(: , 2 : end) = Theta2_grad(: , 2 : end) + (lambda / m) * Theta2(: , 2 : end);










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
