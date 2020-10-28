% Problem 7b

% eigCustom(A) finds the the eigenvalues&-vectors of the real symmetric
% matrix A by power iteration and Hotelling's deflation.

function [lambda_vec, eig_matrix] = eigCustom(A)
dim = length(A);  % Mind that A is symmetric, dim: NxN for some N
eig_matrix = zeros(dim,dim);
lambda_vec = zeros(dim,1);

for i = 1:dim
    [lambda_vec(i), eig_matrix(1:end,i)] = eigLargest(A);
    A = A - lambda_vec(i)*(eig_matrix(1:end,i) * eig_matrix(1:end,i)');
end
end