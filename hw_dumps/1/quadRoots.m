% Problem 2 - quadRoots()

% Finds two roots of quadratic eqn ax^2+bx+c=0, intended for real roots.

% METHODOLOGY
% quadRoots() uses two formulae for finding roots:
% (-b +- sqrt(b^2-4*a*c))/(2*a)
% 2*c/(-b +- sqrt(b^2-4*a*c)) which follows directly from the above
% The same signed variants of these formulae map to different roots
% For b>=0, use minus variants, otherwise plus variants
% This way, b^2>>4*a*c cases cause less problem, as b-(b+eps) kind of
% operation does not take place

function [r1, r2] = quadRoots(a,b,c)
discriminant = sqrt(b^2-4*a*c);

if b>=0
    r1 = 2*c/(-b-discriminant);
    r2 = (-b-discriminant)/(2*a);
else
    r1 = 2*c/(-b+discriminant);
    r2 = (-b+discriminant)/(2*a);
end

end