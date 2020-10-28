% Problem 6a

% newton(fun, dfun, x_init) implements the Newton-Raphson method to solve
% fun(x)=0. x_init is the initial guess, dfun() is the derivative of fun()

function root = newton(fun, dfun, x_init)
root = x_init;
tol = 1e-15;
max_iter = 1000;  % Capping to prevent inf loops
for i=1:max_iter
    if norm(fun(root))<tol
        break
    end
    root = root - fun(root)./dfun(root);
end
end