'''
Problem 10
Functions:
--eig_largest(A) finds the largest in absolute value eigenvalue and its
eigenvector for the real symmetric matrix A by power iteration method
Returns eigenvalue, eigenvector

--eig_custom(A) finds the the eigenvalues&-vectors of the real symmetric
matrix A by power iteration and Hotelling's deflation.
Returns eigenvalue vector, matrix with cols consisting of eigenvectors
--mytests() does the following:
*Plotting time comparison of eig() and eigCustom() for various matrices
*Wigner's semicircle law qualitatively demonstrated by histograms of 
 eigvals of symmetric matrices A with entries from normal dist, and the
 pdf of the law superimposed. See the effect of increasing dim(A)
'''

import time
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def eig_largest(A):
    v_guess = np.ones((A.shape[0],1))/np.sqrt(A.shape[0])  # Tinker with if better guess available
    v = v_guess
    tol = 1e-10

    A_v = A@v
    lambda_val = (v.T @ A_v)/(v.T @ v)

    while True:
        A_v = A@v
        v = A_v/linalg.norm(A_v)
        A_v = A@v
        v = A_v/linalg.norm(A_v)  # Each time 2 iterations without checks
        lambda_prev = lambda_val
        lambda_val = (v.T @ A_v)/(v.T @ v)
        if abs(lambda_val-lambda_prev) < tol:
            break

    return lambda_val,v


def eig_custom(A):
    dim = A.shape[0]  # Mind that A is symmetric, dim: NxN for some N
    eig_matrix = [0 for i in range(dim)]
    lambda_vec = [0 for i in range(dim)]

    for i in range(dim):
        lambda_vec[i], eig_matrix[i] = eig_largest(A)
        A = A - lambda_vec[i]*(eig_matrix[i] @ np.transpose(eig_matrix[i]));

    eig_matrix = np.array(eig_matrix)
    eig_matrix.shape = (dim,dim)
    return np.array(lambda_vec), eig_matrix


def mytests():
    ### Part b:
    np.random.seed(69316)
    dim_vals = [i for i in range(2,24)]
    data = [[0 for j in range(len(dim_vals))] for i in range(2)]
    data_2 = []

    for i in range(len(dim_vals)):  # Will calculate each config only once
        A = np.random.randn(dim_vals[i],dim_vals[i])
        A = np.tril(A) + np.tril(A, -1).T  # Ensuring that A is symmetric

        timer = time.time()
        tmp = linalg.eig(A)
        data[0][i] = time.time() - timer  # Built-in timing

        timer = time.time()
        tmp = eig_custom(A)
        data[1][i] = time.time() - timer  # Power iteration timing

    plt.plot(dim_vals, data[0], label = "Built-in")
    plt.plot(dim_vals, data[1], label = "Power iteration")
    plt.xlabel("Dimension of A")
    plt.ylabel("Time (sec)")
    plt.title("Time comparison for solving eigval problem")
    plt.legend()
    plt.show()

    ### Part c:
    dim_vals = [256,512,1024,2048]
    len_dim_vals = len(dim_vals)
    lambda_data = [0 for i in range(len_dim_vals)]
    row_ctr = 0

    for i in range(len_dim_vals):  # Will calculate each config only once
        A = np.random.randn(dim_vals[i],dim_vals[i])
        A = np.tril(A) + np.tril(A, -1).T  # Ensuring that A is symmetric
        lambda_data[i] = linalg.eigvals(A)

    # Collecting some points on the interval x=[-2,2] from Wigner pdf
    wigner_pdf = lambda x: np.sqrt(4-x**2)/(2*np.pi)
    wigner_x_vals = np.linspace(-2,2,100)
    wigner_eval_array = wigner_pdf(wigner_x_vals)

    # Plot with 4 panels
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(lambda_data[0], density=True)
    axes[0, 0].plot(wigner_x_vals*np.sqrt(dim_vals[0]), wigner_eval_array/np.sqrt(dim_vals[0]))
    axes[0, 0].set_title(f"N={dim_vals[0]}")

    axes[0, 1].hist(lambda_data[1], density=True)
    axes[0, 1].plot(wigner_x_vals*np.sqrt(dim_vals[1]), wigner_eval_array/np.sqrt(dim_vals[1]))
    axes[0, 1].set_title(f"N={dim_vals[1]}")

    axes[1, 0].hist(lambda_data[2], density=True)
    axes[1, 0].plot(wigner_x_vals*np.sqrt(dim_vals[2]), wigner_eval_array/np.sqrt(dim_vals[2]))
    axes[1, 0].set_title(f"N={dim_vals[2]}")

    axes[1, 1].hist(lambda_data[3], density=True)
    axes[1, 1].plot(wigner_x_vals*np.sqrt(dim_vals[3]), wigner_eval_array/np.sqrt(dim_vals[3]))
    axes[1, 1].set_title(f"N={dim_vals[3]}")

    for i in axes.flat:
        i.set(xlabel="Eigenvalues", ylabel="Amplitude")
        i.label_outer()
    plt.show()


if __name__ == "__main__":
    mytests()
