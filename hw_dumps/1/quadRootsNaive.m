% Problem 2 - quadRootsNaive()

% Finds two real roots of quadratic eqn ax^2+bx+c=0 using the commonplace
% discriminant formula, no input sanity checks

function [r1, r2] = quadRootsNaive(a,b,c)
discriminant = sqrt(b^2-4*a*c);
r1 = (-b+discriminant)/(2*a);
r2 = (-b-discriminant)/(2*a);
end