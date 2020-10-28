% Problem 1.b - func0101b()

% Implements the function f(x) = 1 - (x^2 + 1)^(1/2)/(x^2/2 + 1)
% Precision is kept above 1e-8
% This is achieved by using a Taylor series approximation for small x.

% METHODOLOGY
% The fifth term of the Taylor expansion of f(x) is (x^4)/8 while the first
% four are 0. This can be computed by taylor(y,'Order',5)
% For |x|<<1 we would like to disregard (x^4)/8 if it is less than 1e-8.
% This is sufficed when |x|<1e-2 and we can simply return 0, and mind that
% thinking about the orders, 1e-16(double err) over 1e-8(tay. approx of f(1e-2))
% yields roughly 1e-8 relative error.

% USAGE
% func0101b(x) accepts doubles and vectors of doubles,
% returns 0 for |x|<1e-2, otherwise 1 - (x^2 + 1)^(1/2)/(x^2/2 + 1)

function y = func0101b(x)
y = (abs(x) > 1e-2).*(1 - sqrt(x.*x + 1)./((x.*x)/2 + 1));
end