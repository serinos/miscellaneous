% Problem 6b

% halley(fun, dfun, ddfun, x_init) implements Halley's method to solve
% fun(x)=0. x_init is the initial guess, dfun() is the derivative of fun()
% ddfun() second derivative of fun()

function root = halley(fun, dfun, ddfun, x_init)
root = x_init;
tol = 1e-15;
max_iter = 1000;  % Capping to prevent inf loops
for i=1:max_iter
    if norm(fun(root))<tol
        break
    end
    fun_eval = fun(root);
    dfun_eval = dfun(root);
    ddfun_eval = ddfun(root);
    
    root = root - (2*fun_eval.*dfun_eval)./(2*(dfun_eval.^2)-fun_eval.*ddfun_eval);
end
end
