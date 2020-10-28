% Problem 5d

% linSolveGaussSeidel(A,b) implements Gauss Seidel method for solving the
% linear system Ax=b. Returns approx solution x when |Ax-b|<=1e-10

function x = linSolveGaussSeidel(A,b)
x = b*0;  % Initial guess, let us use the zero vector
A_x_min_b = A*x-b;
M1 = inv(tril(A));
M2 = A - tril(A);

while norm(A_x_min_b) > 1e-10
    x = M1*(b-M2*x);
    A_x_min_b = A*x-b;
end
end