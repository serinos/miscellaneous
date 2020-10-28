% Problem 7a

% eigLargest(A) finds the largest in absolute value eigenvalue and its
% eigenvector for the real symmetric matrix A by power iteration method

function [lambda, v] = eigLargest(A)
v_guess = ones(length(A),1)/sqrt(length(A));  % Tinker with if better guess available
v = v_guess;
tol = 1e-12;
iteration_cap = 1000;

% Finding the eigenvector assoc w/ largest |eigval_i|
for i = 1:iteration_cap
    A_v = A*v;
    v_next = A_v/norm(A_v);
    v_diff = norm(v_next-v);
    v = v_next;
    if v_diff < tol  % This may be troublesome if there exists a region
        break        % of slow change other than near the convergence point
    end
end

% Finding lambda assoc with eigvec v
A_v = A*v;
lambda = norm(A_v);
if abs(A_v - lambda*v) > tol  % Checking if lambda pos or neg
    lambda = -lambda;
end

end