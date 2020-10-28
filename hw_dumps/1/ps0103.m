% Problem 3 - ps0103(n_vec)

% The function accepts a vector of integers n_vec with entries referring to
% amount of doubles to be used in sumError() to get an average roundoff error
% due to different directions of addition of that many doubles that are
% selected uniform randomly from [0,1]. Plots log-log of number of doubles
% and the respective average roundoff errors for investigation of the
% effect of number count.

%{
function ps0103(n_vec)
ave_error_vec = zeros(1,length(n_vec));
for i = 1:length(n_vec)
    ave_error_vec(i) = sumError(int32(n_vec(i)));  % Beware: int32 casting
end

loglog(n_vec, ave_error_vec)
grid
end
%}

% KAHAN SUMMATION VARIANT:
% Instead of using the vector summation implemented in MATLAB, uses
% sumKahan(), the aim is to again visualise the change in roundoff error
% caused by direction of addition while increasing the amount of random
% doubles to be added together.
% The numbers are again selected randomly from a uniform dist. btw [0,1].
% Mind that the plot is NOT log-log this time.

%%{
function ps0103(n_vec)
trial_count = 10;  % Random trial count for each average roundoff value
ave_error_vec = zeros(1,length(n_vec));

for i = 1:length(n_vec)
    average = 0;  % Will hold average error due to different direc
    for instance_no = 1:trial_count
        rand_vec = rand(1, int32(n_vec(i)));  % Mind the int32 casting
        error = abs(sumKahan(rand_vec)-sumKahan(rand_vec(end:-1:1)));
        average = average + error;
    end
    ave_error_vec(i) = average/trial_count; % Ave abs error

plot(n_vec, ave_error_vec)
grid
end
%}
