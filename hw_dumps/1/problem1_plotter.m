% Problem 1.b)
% By using func0101b(x) and func0101c(x), plots 50 results from the interval 1e-10<x<1
% Selected x values are logarithmically equidistant.

points = logspace(-10,-1);

fig_1b = loglog(points,func0101b(points));
legend('func0101b()','Location','northwest');
saveas(fig_1b,'problem_1b_plot.png');

fig_1c = loglog(points,func0101c(points));
legend('func0101c()','Location','northwest');
saveas(fig_1c,'problem_1c_plot.png');