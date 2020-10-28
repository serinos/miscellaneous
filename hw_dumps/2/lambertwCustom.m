% Problem 6

% lambertwCustom(x, method) calculates the Lambert W (principal branch) 
% function defined as W(xe^x)=y if y=xe^x, using the method specified.
% For Halley's, method='h'
% For Newton-Raphson, method='n'; defaults to Newton-Raphson otherwise

function y = lambertwCustom(x, method)
funShifted = @(z)(z.*exp(z)-x);
funDeriv = @(z)((z+1).*exp(z));
funDeriv2 = @(z)((z+2).*exp(z));
x_init=1;

if nargin>1 && method=='h'
    y = halley(funShifted, funDeriv, funDeriv2, x_init);
else
    y = newton(funShifted, funDeriv, x_init);
end
end