% Problem 6

% Aim:
% Comparison of Newton Raphson and Halley methods with respect to speed.
% Comparison of builtin lambertw() output and outputs of hand-made funcs
% that use the aforementioned methods.

% Test parameters:
test_interval = [-0.99/exp(1) 10];  % Almost the same one in the problem set
% This interval is better for visualization, as around 1/exp(1) the order
% of magnitude of error is too high compared to the rest and clipping that
% part away results in more explanatory plots.
test_point_count = 1000;
test_points = linspace(test_interval(1), test_interval(2), test_point_count);


% Will compare output of custom ones with the builtin, plot the abs diff:
Lw_points_newton = lambertwCustom(test_points, 'n');
Lw_points_halley = lambertwCustom(test_points, 'h');
Lw_points_builtin = lambertw(test_points);

Lw_abs_difference_newton_builtin = abs(Lw_points_newton - Lw_points_builtin);
Lw_abs_difference_halley_builtin = abs(Lw_points_halley - Lw_points_builtin);

plot(test_points, Lw_abs_difference_newton_builtin), hold on
plot(test_points, Lw_abs_difference_halley_builtin)
xlabel('Inputs')
ylabel('Abs error wrt builtin')
legend('Newton-Raphson','Halley')
grid, hold off


% Will compare the speed of the three functions of interest by taking the
% average of 10 instances of execution time for each of them using
% test_points as initialized above:
instance_count = 10;
time_Lw_newton = zeros(instance_count,1);
time_Lw_halley = zeros(instance_count,1);
time_Lw_builtin = zeros(instance_count,1);

for i=1:instance_count
   tic;
   lambertwCustom(test_points, 'n');
   time_Lw_newton(i) = toc;
   tic;
   lambertwCustom(test_points, 'h');
   time_Lw_halley(i) = toc;
   tic;
   lambertw(test_points);
   time_Lw_builtin(i) = toc;
end

time_ave_Lw_newton = sum(time_Lw_newton)/instance_count;
time_ave_Lw_halley = sum(time_Lw_halley)/instance_count;
time_ave_Lw_builtin = sum(time_Lw_builtin)/instance_count;

% Displaying the average time data:
%%{
fprintf('Interval: [%f , %f]\n',test_interval(1),test_interval(2))
fprintf('Average values for completion of %i evaluations\n', test_point_count)
fprintf('Average time for Newton: %f sec\n', time_ave_Lw_newton)
fprintf('Average time for Halley: %f sec\n', time_ave_Lw_halley)
fprintf('Average time for Builtin: %f sec\n', time_ave_Lw_builtin)
fprintf('Average values for completion of one evaluation\n')
fprintf('Average time for Newton: %.10f sec\n', time_ave_Lw_newton/test_point_count)
fprintf('Average time for Halley: %.10f sec\n', time_ave_Lw_halley/test_point_count)
fprintf('Average time for Builtin: %.10f sec\n', time_ave_Lw_builtin/test_point_count)
%}