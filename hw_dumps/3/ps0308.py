'''
Problem 8
Functions:

--lin_solve_gradient_descent(A,b) implements Gradient Descent for solving the
linear system Ax=b. Returns approx solution x when |Ax-b|<=1e-10

--lin_solve_gauss_seidel(A,b) implements Gauss Seidel method for solving the
linear system Ax=b. Returns approx solution x when |Ax-b|<=1e-10

--mytests(N_vec) outputs the plots requested in the problem set
N_vec is an array that contains

Explanation regarding the contents of mytest():
###
Plots the time elapsed until an answer was found for each solution
technique outlined in the problem set for Ax=b where the dimension of A
is N by N, N values are taken in as a vector N_vec.

Matrix A is the one on the PS, b some random vector that is kept the same
for all calculations besides cropping for matching dimensions.
###
'''

import time
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def lin_solve_gradient_descent(A: np.array,b: np.array):
    x = b*0  # Initial guess, let us use the zero vector
    A_x_min_b = A@x - b
    tol = 1e-10

    while linalg.norm(A_x_min_b) > tol:
        tau = (A_x_min_b.T @ A_x_min_b)/(A_x_min_b.T @ (A@A_x_min_b))
        x = x - tau*A_x_min_b
        A_x_min_b = A@x - b

    return x


def lin_solve_gauss_seidel(A: np.array, b: np.array):
    x = b*0  # Initial guess, let us use the zero vector
    A_x_min_b = A@x - b
    P_A, L_A, U_A = linalg.lu(A)
    
    M1 = linalg.inv(L_A)
    M2 = A - L_A
    tol = 1e-10

    while linalg.norm(A_x_min_b) > tol:
        x = M1@(b - M2@x)
        A_x_min_b = A@x-b

    return x


def mytests(N_vec = np.array([i for i in range(2,32)])):
    N_count = len(N_vec)
    N_largest = max(N_vec)
    N_trial_count = 3  # Will take the average of N_trial_count computations per datapoint

    data = [[[0 for k in range(N_count)] for j in range(N_count)]\
            for i in range(4)] # Four tables, one for each method

    # Initializing tridiagonal A described in PS
    A = np.diag(np.ones(N_largest)) \
        + np.diag((-1/2)*np.ones(N_largest-1),1) \
        + np.diag((-1/2)*np.ones(N_largest-1),-1)
    b = np.random.rand(N_largest,1)    

    for j in range(N_trial_count):
        for i in range(N_count):
            tmp_A = A[:N_vec[i],:N_vec[i]]
            tmp_A_sp = sparse.csr_matrix(tmp_A)
            tmp_b = b[:N_vec[i]]
            
            timer = time.time()
            tmp = linalg.solve(tmp_A, tmp_b)  # Built-in NonSparse [0]
            data[0][j][i] = time.time() - timer

            timer = time.time()
            tmp = sparse.linalg.spsolve(tmp_A_sp,tmp_b)  # Built-in Sparse [1]
            data[1][j][i] = time.time() - timer

            timer = time.time()
            tmp = lin_solve_gradient_descent(tmp_A, tmp_b)  # Grad Desc [2]
            data[2][j][i] = time.time() - timer

            timer = time.time()
            tmp = lin_solve_gauss_seidel(tmp_A, tmp_b)  # Gauss Seidel [3]
            data[3][j][i] = time.time() - timer

    average_data = [np.sum(np.array(data[i]),0)/N_trial_count for i in range(4)]
    
    # Plotting
    plt.plot(N_vec, average_data[0], label="Built-in (Non-Sparse)")
    plt.plot(N_vec, average_data[1], label="Built-in (Sparse)")
    plt.plot(N_vec, average_data[2], label="Gradient Descent")
    plt.plot(N_vec, average_data[3], label="Gauss Seidel")
    plt.xlabel("N value")
    plt.ylabel("Time (sec)")
    plt.title("Time comparison for solving lin sys")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mytests()
