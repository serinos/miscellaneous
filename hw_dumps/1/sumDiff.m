% Problem 3 - sumDiff()

% sumDiff returns the difference between the summation of the elements of a
% vector through index 1 to n and the summation done in reverse order.

function diff = sumDiff(vec)
diff = abs(sum(vec)-sum(vec(end:-1:1)));  % Mind that sum() adds consecutively wrt indices
end