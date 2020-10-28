% Problem 5c

% linSolveGradientDescent(A,b) implements gradient descent for solving the
% linear system Ax=b. Returns approx solution x when |Ax-b|<=1e-10

function x = linSolveGradientDescent(A,b)
x = b*0;  % Initial guess, let us use the zero vector
A_x_min_b = A*x-b;

while norm(A_x_min_b) > 1e-10
    tau = (A_x_min_b.' * A_x_min_b)/ (A_x_min_b.' * (A*A_x_min_b));
    x = x - tau*A_x_min_b;
    A_x_min_b = A*x-b;
end
end