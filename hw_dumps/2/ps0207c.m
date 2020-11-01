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
    A = tril(A)+tril(A,-1).';  % Ensuring that A is symmetric
    lambda_vec(i,1:dim_vals(i)) = eig(A);
end
%}

wigner_pdf = @(x)(sqrt(4-x.^2)./(2*pi));
wigner_x_vals = linspace(-2,2,200);
wigner_eval = wigner_pdf(wigner_x_vals);

%%{
% Plotting
figure(1);
hold on;
for i = 1:length_dim_vals
    subplot(1,length_dim_vals,i);
    histogram(lambda_vec(i,1:dim_vals(i)),'Normalization','pdf'), hold on
    plot(sqrt(dim_vals(i))*wigner_x_vals, wigner_eval/sqrt(dim_vals(i)))
    legend(sprintf('Eigvals, N=%d',dim_vals(i)),'Wigner pdf')
    ylabel('Amplitude')
    xlabel('Eigenvalues')
end
%}
