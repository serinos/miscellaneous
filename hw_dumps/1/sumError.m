% Problem 3 - sumError()

% sumError(n) returns the average of 100 instances of sumDiff(rand(1,n))
% The aim is to see the average error caused by summation in different orders

function average = sumError(n)
trial_count = 100;
average = 0;

for instance_no = 1:trial_count
    average = average + sumDiff(rand(1,n));
end

average = average/trial_count;
end