% Problem 5c,d,e

% ps0205(N_vec)
% Plots the time elapsed until an answer was found for each solution
% technique outlined in the problem set for Ax=b where the dimension of A
% is N by N, N values are taken in as a vector N_vec.

% Matrix A is the one on the PS, b some random vector that is kept the same
% for all calculations besides cropping for matching dimensions.

function ps0205(N_vec)

N_count = length(N_vec);
N_largest = max(N_vec);
N_trial_count = 3;  % Will take the average of N_trial_count computations per datapoint

% Matrices holding all the data calculated
elapsed_GradientDescent = zeros(N_trial_count,N_count);
elapsed_GaussSeidel = zeros(N_trial_count,N_count);
elapsed_SparsedBuiltin = zeros(N_trial_count,N_count);
elapsed_Builtin_NotSparse = zeros(N_trial_count,N_count);

% Initializing A and b
A = diag(1*ones(1,N_largest)) + diag((-1/2)*ones(1,N_largest-1),1) + diag((-1/2)*ones(1,N_largest-1),-1);
b = rand(N_largest,1);

for j = 1:N_trial_count
    for i = 1:N_count
        tmp_A = A(1:N_vec(i),1:N_vec(i));
        tmp_b = b(1:N_vec(i));
        tic;
        tmp = tmp_A\tmp_b;
        elapsed_Builtin_NotSparse(j,i) = toc;
        tic;
        tmp = linSolveGaussSeidel(tmp_A,tmp_b);
        elapsed_GaussSeidel(j,i) = toc;
        tic;
        tmp = linSolveGradientDescent(tmp_A,tmp_b);
        elapsed_GradientDescent(j,i) = toc;
    end
end

% Reinitializing A as a sparse matrix and solving Ax=b with built-in funcs
A = sparse(A);
for j = 1:N_trial_count
    for i = 1:N_count
        tmp_A = A(1:N_vec(i),1:N_vec(i));
        tmp_b = b(1:N_vec(i));
        tic;
        tmp = tmp_A\tmp_b;
        elapsed_SparsedBuiltin(j,i) = toc;
    end
end

% Averaging out the results
results_GradientDescent = sum(elapsed_GradientDescent,1)/N_trial_count;
results_GaussSeidel = sum(elapsed_GaussSeidel,1)/N_trial_count;
results_Builtin_NotSparse = sum(elapsed_Builtin_NotSparse,1)/N_trial_count;
results_SparsedBuiltin = sum(elapsed_SparsedBuiltin,1)/N_trial_count;

% Plotting code:
%%{
plot(N_vec, results_GradientDescent), hold on
plot(N_vec, results_GaussSeidel)
plot(N_vec, results_Builtin_NotSparse)
plot(N_vec, results_SparsedBuiltin)
legend({'Gradient Descent','Gauss Seidel','Builtin (Non-Sparse)','Builtin (Sparse)'},'Location','northwest')
xlabel('N value')
ylabel('Time (sec)')
grid, hold off
%}
end