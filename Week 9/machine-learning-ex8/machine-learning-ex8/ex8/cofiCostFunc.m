function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Calculating J and Theta_grad 
for j = 1 : num_users
	
	% find the movies which have been rated by j'th user
	rated_j = find(R(:, j));
	
	% subset of Y and X for j'th user
	Y_j = Y(rated_j, j);
	X_j = X(rated_j, :);
	
	% Predicting ratings for j'th user
	H_X_j = X_j * Theta(j, :)';
	
	% Difference b/w prediction and actual rating
	diff = H_X_j - Y_j;
	
	% Calculating summation term for J
	J = J + sum(diff .^ 2);
	
	% Calculating Theta_grad for j'th user
	Theta_grad(j, :) = Theta_grad(j, :) + diff' * X_j + lambda * Theta(j, :);
	
	
end

% calculating J
J = (1 / 2) * J + (lambda / 2) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));


% X_grad calculation
for i = 1 : num_movies
	
	% find the users who have rated i'th movie
	rated_i = find(R(i, :));
	
	% subset of Y and Theta for i'th movie
	Y_i = Y(i, rated_i);
	Theta_i = Theta(rated_i, :);
	
	% Predicting the ratings for i'th movie
	H_X_i = X(i, :) * Theta_i';
	
	% Difference b/w prediction and actual rating
	diff = H_X_i - Y_i;
	
	% Calculating X_grad for i'th movie
	X_grad(i, :) = X_grad(i, :) + diff * Theta_i + lambda * X(i, :);
	
end





















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
