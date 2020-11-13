'''
Problem 12

--runges_function(x) Evaluates $R(x)=\frac{1}{1+25x^2}$

--E_n_calc_polyn(polyn) Evaluates E(n), which is:
$E(n)=[\frac{1}{2} \int_{-1}^1 |R(x)-R_n(x)|^2]^{1/2}$
where R_n(x) is the n-1 deg polynomial that fits n points exactly
Takes in polyfit outputs of coeff arrays

--E_n_calc(func) Evaluates E(n), but takes R_n to be some func

--chebyshev_nodes(n) Returns array with elements as follows:
$x_j = \cos(\frac{2j+1}{2n}\pi)$

--mytests() Solves questions in the PS
'''

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d


def runges_function(x):
    value = 1/(1+25*(x**2))
    return value


def E_n_calc_polyn(polyn):
    integrand = lambda x: np.power(np.abs(runges_function(x)-polyval(x,polyn)),2)
    E_n = np.sqrt(0.5*quad(integrand,-1,1)[0])
    return E_n


def E_n_calc(func):
    integrand = lambda x: np.power(np.abs(runges_function(x)-func(x)),2)
    E_n = np.sqrt(0.5*quad(integrand,-1,1)[0])
    return E_n


def chebyshev_nodes(n):
    cheb_array = [np.cos((2*i+1)*np.pi/(2*n)) for i in range(n)]
    return np.array(cheb_array)


def mytests():
    # Part A -----
    # Polynomial fits for Runge's function on [-1,1] and plot
    n = [4,8,16,32,64]
    bound = [-1, 1]
    plot_bound = bound
    plot_x = np.linspace(plot_bound[0], plot_bound[1], 250)
    runges_eval, polyn_list, polyn_eval, sample_x_list = [],[],[],[]

    for i in range(len(n)):
        sample_x_list.append(np.linspace(bound[0],bound[1],n[i]))
        runges_eval.append(runges_function(sample_x_list[i]))
        polyn_list.append(polyfit(sample_x_list[i], runges_eval[i], n[i]-1))
        polyn_eval.append(polyval(plot_x, polyn_list[i]))
        # Using n[i]-1 degree polyns to fit exactly
        plt.plot(plot_x, polyn_eval[i], label=f"{n[i]} pts, {n[i]-1} deg")
    plt.plot(sample_x_list[-1], runges_eval[-1], '.')
    plt.grid()
    plt.legend()
    plt.title("Polynomial fits of Runge's fn, range [-1,1]")
    plt.show()

    # E(n) vs n plot
    E_n_vals = []
    for polyn in polyn_list:
        E_n_vals.append(E_n_calc_polyn(polyn))
    plt.plot(n, E_n_vals, 'bo')
    plt.plot(n, E_n_vals, 'g--')
    plt.xlabel("n value")
    plt.ylabel("E(n)")
    plt.grid()
    plt.title("E(n) vs n for Polynomial fits")
    plt.show()
    
    #######
    #######
    
    # Part B -----
    # Repeating Part A with splines
    spline_lin_list, spline_cub_list = [],[]
    for i in range(len(n)):
        spline_lin_list.append(interp1d(sample_x_list[i], runges_eval[i], kind='linear'))
        spline_cub_list.append(interp1d(sample_x_list[i], runges_eval[i], kind='cubic'))
        plt.plot(plot_x, spline_lin_list[i](plot_x), '-.', label=f"{n[i]} pts, linear")
        plt.plot(plot_x, spline_cub_list[i](plot_x), '--', label=f"{n[i]} pts, cubic")
    plt.plot(sample_x_list[-1], runges_eval[-1], '.')
    plt.grid()
    plt.legend()
    plt.title("Spline fits of Runge's fn, range [-1,1]")
    # Would not draw all the stuff on the same plot, but that seems to be the instruction?
    plt.show()

    # E(n) vs n plot
    E_n_vals_spl_lin, E_n_vals_spl_cub = [], []
    for func in spline_lin_list:
        E_n_vals_spl_lin.append(E_n_calc(func))
    for func in spline_cub_list:
        E_n_vals_spl_cub.append(E_n_calc(func))
    plt.plot(n, E_n_vals_spl_lin, 'bo')
    plt.plot(n, E_n_vals_spl_lin, 'b--', label="Linear Splines")
    plt.plot(n, E_n_vals_spl_cub, 'go')
    plt.plot(n, E_n_vals_spl_cub, 'g--', label="Cubic Splines")
    plt.xlabel("n value")
    plt.ylabel("E(n)")
    plt.grid()
    plt.legend()
    plt.title("E(n) vs n for Splines")
    plt.show()
    
    #######
    #######
    
    # Part C -----
    # Repeating A with Chebyshev nodes
    runges_eval, polyn_list, polyn_eval = [],[],[]
    sample_x_list = [chebyshev_nodes(j) for j in n]
    for i in range(len(n)):
        runges_eval.append(runges_function(sample_x_list[i]))
        polyn_list.append(polyfit(sample_x_list[i], runges_eval[i], n[i]-1))
        polyn_eval.append(polyval(plot_x, polyn_list[i]))
        # Using n[i]-1 degree polyns to fit exactly
        plt.plot(plot_x, polyn_eval[i], label=f"{n[i]} pts, {n[i]-1} deg")
    plt.grid()
    plt.legend()
    plt.title("Polynomial fits of Runge's fn, range [-1,1], points Chebyshev nodes")
    plt.show()

    E_n_vals_cheb = []
    for polyn in polyn_list:
        E_n_vals_cheb.append(E_n_calc_polyn(polyn))
    plt.plot(n, E_n_vals_cheb, 'bo')
    plt.plot(n, E_n_vals_cheb, 'g--')
    plt.xlabel("n value")
    plt.ylabel("E(n)")
    plt.grid()
    plt.title("E(n) vs n for Polynomial fits using Chebyshev nodes")
    plt.show()
    
    #######
    #######

    # Part D -----
    n = [16,64,256,1024]
    runges_eval, polyn_list, polyn_eval, sample_x_list = [],[],[],[]

    for i in range(len(n)):
        sample_x_list.append(np.linspace(bound[0],bound[1],n[i]))
        runges_eval.append(runges_function(sample_x_list[i]))
        sqr_deg = int(np.sqrt(n[i]))  # Degree of polyns: floored sqrt(n[i])
        polyn_list.append(polyfit(sample_x_list[i], runges_eval[i],sqr_deg))
        polyn_eval.append(polyval(plot_x, polyn_list[i]))
        plt.plot(plot_x, polyn_eval[i], label=f"{n[i]} pts, {sqr_deg} deg")
    plt.grid()
    plt.legend()
    plt.title("Polynomial fits of Runge's fn, range [-1,1]")
    plt.show()

    E_n_vals_sqr = []
    for polyn in polyn_list:
        E_n_vals_sqr.append(E_n_calc_polyn(polyn))
    plt.plot(n, E_n_vals_sqr, 'bo')
    plt.plot(n, E_n_vals_sqr, 'g--')
    plt.xlabel("n value")
    plt.ylabel("E(n)")
    plt.grid()
    plt.title("E(n) vs n for Polynomial fits of deg=floor(sqrt(n))")
    plt.show()

    
if __name__ == "__main__":
    mytests()
