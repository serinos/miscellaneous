% Problem 7b

% Plotting time comparison of eig() and eigCustom() for various matrices

dim_vals = 2:40;
elapsed_builtin = zeros(1,dim_vals(end));
elapsed_powerit = zeros(1,dim_vals(end));

for i = dim_vals  % Will calculate each config only once
    A = randn(i);
    A = A - tril(A) + (triu(A))';  % Ensuring that A is symmetric
    tic;
    eig(A);
    elapsed_builtin(i) = toc;
    tic;
    eigCustom(A);
    elapsed_powerit(i) = toc;
end

plot(dim_vals, elapsed_builtin(dim_vals(1):end)), hold on
plot(dim_vals, elapsed_powerit(dim_vals(1):end))
xlabel('Dimension of A')
ylabel('Time (sec)')
legend('Built-in','Power Iter. & Hot. Deflation','Location','northwest')
grid