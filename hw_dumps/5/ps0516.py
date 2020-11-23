'''
Problem 16

Functions:
--read_hubble_data()  Reads z, dL, err data from file, outputs three arrays
correspoding to those

--numerical_derivative_irregular(fx, x)  Uses irregular finite differencing
to take the derivative of a function f(x) that is sampled at some x values
Input fx,x are arrays, array x should be strictly increasing, output is
an array d_fx that corresponds to evaluations of f'(x) at vals of array x

--numerical_integral_irregular(fx, x)  Uses Simpsons method for irregularly
spaced steps to take the integral of a function f(x) that is sampled at some
x values. Trapezoid method is used for the final two points.
Input fx,x are arrays, array x should be strictly increasing, returns a double

--mytests()  Solves the questions in the problem set
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from time import time
from scipy.interpolate import CubicSpline
from ps0513 import *  # Will use findiff()


def read_hubble_data(filename="hubble_var.dat"):
    z_arr, dL_arr, err_arr = [],[],[]
    with open(f"{filename}", 'r') as fh:
        for line in fh:
            (col1, col2, col3) = line.split()
            z_arr.append(np.float(col1))
            dL_arr.append(np.float(col2))
            err_arr.append(np.float(col3))
    return np.array(z_arr), np.array(dL_arr), np.array(err_arr)


def numerical_derivative_irregular(fx, x):
    pts_count = len(fx)
    d_fx = []

    # Taking care of x_0
    x_0_list = [fx[0],fx[1],fx[2]]
    stencil = findiff([0,(x[1]-x[0]),(x[2]-x[1])],1)
    d_fx.append(np.sum(stencil*x_0_list))

    # Going over the middle points
    for i in range(1,pts_count-1):
        x_i_list = [fx[i-1],fx[i],fx[i+1]]
        stencil = findiff([(x[i-i]-x[i]),0,(x[i+1]-x[i])],1)
        d_fx.append(np.sum(stencil*x_i_list))

    # Taking care of x_fin
    x_fin_list = [fx[-3],fx[-2],fx[-1]]
    stencil = findiff([(x[-3]-x[-2]), (x[-2]-x[-1]), 0],1)
    d_fx.append(np.sum(stencil*x_fin_list))

    return np.array(d_fx)


def numerical_integral_irregular(fx, x):
    pts_count = len(fx)
    int_slices = []
    h = np.diff(x)

    for i in range(0,pts_count//2 -1):  # Simpson's Rule - Irreg Spaced
        h_2i_1 = h[2*i+1]
        h_2i = h[2*i]
        alpha = (2*(h_2i_1**3) - h_2i**3 + 3*h_2i*(h_2i_1**2))/(6*h_2i_1*(h_2i_1+h_2i))
        beta = (h_2i_1**3 + h_2i**3 + 3*h_2i_1*h_2i*(h_2i_1+h_2i))/(6*h_2i_1*h_2i)
        eta = (2*(h_2i**3) - h_2i_1**3 + 3*h_2i_1*(h_2i**2))/(6*h_2i*(h_2i_1+h_2i))
        int_slices.append(alpha*fx[2*i+2]+beta*fx[2*i+1]+eta*fx[2*i])
    if pts_count%2==0:  # Handling edge case arising from pts_count being even
                        # Using Trapezoid Rule for the last two points
        int_slices.append(h[-1]*(fx[-1]+fx[-2])/2)

    result = np.sum(int_slices)
    return result


def mytests():
    # Part A
    x,fx,err = read_hubble_data()
    x_sample = np.arange(0.014,0.83,0.01)
    fx_cub = CubicSpline(x,fx)
    d_fx_cub = fx_cub.derivative()
    plt.plot(x, fx, ".-", label="Function")
    plt.plot(x, numerical_derivative_irregular(fx,x), ".-", label="Fin Diff Derivative")
    plt.plot(x_sample, d_fx_cub(x_sample), "--", label="Cub Spl Derivative")
    plt.title("Derivative of d_L")
    plt.legend()
    plt.grid()
    plt.show()

    # Part B
    time_man, time_bin = [],[]
    dL_int_evals_man, dL_int_evals_bin,\
                      dL_int_evals_cub = [],[],[]  # Taking integral from z[0] to z[i], i={1,2,...,pts_count}
    for i in range(2,len(x)):
        ctr = time()
        dL_int_evals_man.append(numerical_integral_irregular(fx[0:i],x[0:i]))
        time_man.append(time()-ctr)
        ctr_2 = time()
        dL_int_evals_bin.append(simps(fx[0:i],x[0:i]))
        time_bin.append(time()-ctr_2)
        dL_int_evals_cub.append(fx_cub.integrate(x[0],x[i]))
    print(f"[Eval] Total time for built-in simps: {np.sum(time_bin)}")
    print(f"[Eval] Total time for hand-made simps: {np.sum(time_man)}")
    plt.plot(x[1:-1], dL_int_evals_man, ".-", label="Hand-made Simps")
    plt.plot(x[1:-1], dL_int_evals_bin, "o", label="Built-in Simps")
    plt.plot(x[1:-1], dL_int_evals_bin, ".-", label="Cubic Spl")
    plt.legend()
    plt.title("Integral of d_L")
    plt.grid()
    plt.show()   
        

if __name__ == "__main__":
    mytests()
