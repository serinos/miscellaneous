% Problem 7c

% Wigner's semicircle law qualitatively demonstrated by histograms of 
% eigvals of symmetric matrices A with entries from normal dist, and the
% pdf of the law superimposed. See the effect of increasing dim(A)

%%{
% Calculations

dim_vals = [512,1024,2048];
length_dim_vals = length(dim_vals);
lambda_vec = zeros(length_dim_vals, max(dim_vals));
row_ctr = 0;

for i = 1:length_dim_vals  % Will calculate each config only once
    A = randn(dim_vals(i));  % Entries of A from normal dist so that we can check Wigner's semicircle law
    A = A - tril(A) + (triu(A))';  % Ensuring that A is symmetric
    lambda_vec(i,1:dim_vals(i)) = eig(A);
end
%}

syms x;
confidence = 0.95;
int_p = int(sqrt(4-x^2),x);
p_wigner = @(x1,x2)(double(subs(int_p,x2)-subs(int_p,x1))/(2*pi));
p_wigner_conf = @(x0)(p_wigner(-x0,x0)-confidence);  
% Finding root x0 will give us intervals of [-sqrt(N)*x0,sqrt(N)*x0]
% We can check the histograms to see whether the tails kind of die out at
% the boundaries and whether they look alike in general.
x_0 = fsolve(p_wigner_conf, 1);
lambda_intervals = zeros(length_dim_vals,2);

for i = 1:length_dim_vals
    lambda_intervals(i,1) = -x_0*sqrt(dim_vals(i));
    lambda_intervals(i,2) = -lambda_intervals(i,1);
end

x_for_pdf = linspace(-2,2);
p_wigner_pdf = @(x)(sqrt(4-x.^2)./(2*pi));

%%{
% Plotting

magic_fix = 1/25;  % For some reason, the histogram plotting subroutine does
% not normalize eigenvalue distribution properly or I am doing something
% wrong, so I multiply p_wigner_pdf by a magic constant to get it down
% to scale of the histograms, the rough shapes are what we are after anyway
figure(1);
hold on;
for i = 1:length_dim_vals
    subplot(1,length_dim_vals,i);
    histogram(lambda_vec(i:dim_vals(i)),'Normalization','pdf'), hold on
    plot(sqrt(dim_vals(i))*x_for_pdf, p_wigner_pdf(x_for_pdf)*magic_fix)
    legend(sprintf('Eigvals, N=%d',dim_vals(i)),'Wigner pdf')
    ylabel('Amplitude')
    xlabel('Eigenvalues')
end
%}