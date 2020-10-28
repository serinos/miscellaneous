% Problem 1.c - func0101c()

% Implements the function f(x) = cot(x^2) - 1/(x^2)
% Precision is kept above 1e-8
% This is achieved by using a Taylor series approximation for small x.

% METHODOLOGY
% The 3rd term of the Taylor expansion of f(x) is -(x^2)/3 while the first
% two are 0.
% For |x|<<1 we would like to disregard -(x^2)/3 if its order is less than 1e-8.
% This is sufficed when |x|<1e-4 and we can simply return 0, and mind that
% thinking about the orders, 1e-16(double err) over 1e-8(tay. approx of f(1e-4))
% yields roughly 1e-8 relative error.

% USAGE
% func0101c(x) accepts doubles and vectors of doubles,
% returns 0 for |x|<1e-4, otherwise cot(x^2) - 1/(x^2)

function y = func0101c(x)
y = (abs(x) > 1e-4).*(cot(x.*x) - 1./(x.*x));
end