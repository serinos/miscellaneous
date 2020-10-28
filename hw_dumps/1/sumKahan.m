% Problem 4 - sumKahan()

% Implements Kahan summation for addition of elements of a vector

function [vec_sum] = sumKahan(vec)
vec_sum = 0;
compensator = 0;  % Compensates for lost low order bits in loop

for i = 1:length(vec)
    x = vec(i) - compensator;  % Low bit leftovers from prev. ite. are compensated
    total = vec_sum + x;
    compensator = (total - vec_sum) - x;  % The low bit leftovers are stored
    vec_sum = total;
end

end