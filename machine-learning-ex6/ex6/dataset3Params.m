function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

param = [C sigma];
error_init = 1;
x1 = [0 1 0];
x2 = [0 -1 0];
x1 = x1(:);
x2 = x2(:);
t = [0.01,0.03,0.1,0.3,1,3,10,30];
l = length(t);
for i = 1:l
	for j = 1:l
		C = t(i);
		sigma = t(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if error < error_init
			error_init = error;
			param = [C sigma];
		end
	end

end

C = param(1)
sigma = param(2)



% =========================================================================

end
