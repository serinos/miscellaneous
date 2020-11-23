'''
Problem 14

Functions:
--numerical_derivative(fx,a,b) Calculates the derivative of a function
represented as fx, an array of doubles, the first entry corresponding to f(a)
and the last entry corresponding to f(b), spacings between points are assumed
to be equal. 3-point stencils are used for numerical differentiation

--numerical_integral(fx,a,b,method="simpson") Calculates numerical integral
for given fx array of doubles, the first entry corresponding to f(a)
and the last entry corresponding to f(b), spacings between points are assumed
to be equal. Available methods are "simpson" and "trapezoid"

--mytests() Solves the questions in the problem set
'''

import numpy as np
import matplotlib.pyplot as plt
from ps0513 import *  # Will use findiff() 


def numerical_derivative(fx, a, b):
    pts_count = len(fx)
    h = (b-a)/(pts_count-1)
    d_fx = []
    stencil_3pt_fwd = findiff([0,1,2],1)
    stencil_3pt_bwd = findiff([-2,-1,0],1)
    stencil_3pt_ctr = findiff([-1,0,1],1)
    # We will factor out 1/h at the end instead of putting *h to stencils

    # Taking care of x=a
    x_0_list = [fx[0],fx[1],fx[2]]
    d_fx.append(np.sum(stencil_3pt_fwd*x_0_list)/h)

    # Going over the middle points
    for i in range(1,pts_count-1):
        x_i_list = [fx[i-1],fx[i],fx[i+1]]
        d_fx.append(np.sum(stencil_3pt_ctr*x_i_list)/h)

    # Taking care of x=b
    x_fin_list = [fx[-3],fx[-2],fx[-1]]
    d_fx.append(np.sum(stencil_3pt_bwd*x_fin_list)/h)

    return np.array(d_fx)


def numerical_integral(fx, a, b, method="simpson"):
    pts_count = len(fx)
    h = (b-a)/(pts_count-1)
    slice_area = []
    if method=="trapezoid":  # Trapezoidal Rule
        result = (np.sum(fx) - fx[0]/2 -fx[-1]/2)*h  # The formula
    else:  # Simpson's Rule implemented here
        even_ind, odd_ind = [],[]
        for i in range(1,pts_count-1,2):  # Tried to do masking with (0,1,0,1,...) instead,
                                          # but it took even longer to initialize the masks
            odd_ind.append(fx[i])
            even_ind.append(fx[i+1])  # Not adding up areas directly in the loop
                                      # which could result in catastrophic cancellation
                                      # np.sum() uses pairwise summation, opted for that
        if pts_count%2==1:  # Appending missing fx[-2] to even_ind in case N is odd
            even_ind.append(fx[-2])
        result = (fx[0]+fx[-1]+2*np.sum(even_ind)+4*np.sum(odd_ind))*h/3  # The formula

    return result


def mytests():
    func = lambda x: 1+np.sin(np.pi*x + np.pi/4)
    d_func = lambda x: np.cos(np.pi*x + np.pi/4)*np.pi  # Analyt. der. of func
    interval = [-1,1]

    # Part A - Comparison of Finite Differencing and Analytical Deriv
    sample1_h = 2**(-14)
    x_pts_1 = np.arange(interval[0],interval[1],sample1_h)
    func_sample_1 = func(x_pts_1)
    d_func_sample_an = d_func(x_pts_1)
    d_func_sample_1 = numerical_derivative(func_sample_1,interval[0], interval[1])

    plt.plot(x_pts_1, func_sample_1, label="Function")
    plt.plot(x_pts_1, d_func_sample_1, '--g', label="Num. Deriv")
    plt.plot(x_pts_1, d_func_sample_an, '-.r', label="Analyt. Deriv")
    plt.title("Comparison of Num Deriv w/ 3pts stencil vs Analytical Deriv")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.plot(x_pts_1,\
             100*(d_func_sample_1-d_func_sample_an)/d_func_sample_an, '-.',)
    plt.title("Error of Num Deriv w/ 3pts stencil wrt Analytical Deriv")
    plt.ylabel("Error pc")
    plt.grid()
    plt.show()

    # Part A - Halving h=2**-15 step by step until roundoff error dominates
    interval_2 = [-0.03125,0.03125] # Will work on this interval only, my RAM is not enough
    two_to_what = [-15,-20]  # Tinker with this to get the fluctuations
    part_A_trials = [2**i for i in range(two_to_what[0],two_to_what[1],-1)]
    d_func_collec = []
    plot_skip_step = 1  # A const to decrease no of pts on the plot at the end
    for i in range(len(part_A_trials)):
        print(f"[Init] h = {part_A_trials[i]} is being calculated")
        x_pts_tmp = np.arange(interval_2[0],interval_2[1],part_A_trials[i])
        func_sample_tmp = func(x_pts_tmp)
        d_func_sample_tmp = numerical_derivative(func_sample_tmp,interval_2[0], interval_2[1])
        d_func_collec.append(np.array(d_func_sample_tmp[::2**(i+plot_skip_step)]))
        print(f"[Fin] h = {part_A_trials[i]} has been completed")

    x_pts_plot = np.arange(interval_2[0],interval_2[1],part_A_trials[0])[::2**plot_skip_step]
    print("[Info] Number of data points on the plot for each strip: ", len(x_pts_plot))
    for i in range(len(d_func_collec)-1):
        plt.plot(x_pts_plot, (2**(i))*(d_func_collec[i]-d_func_collec[i+1]),\
                 label=f"f'_2^{two_to_what[0]-i} - f'_2^{two_to_what[0]-1-i}")
    plt.grid()
    plt.legend()
    plt.title("3pt-Stencil Num Derivs, h_val Comparison")
    plt.show()
        

    # Part B - Simple Eval
    print(f"[Info] h = {sample1_h} for the following two evaluations of the integral in the PS")
    print("[Eval] Simpsons: ", numerical_integral(func_sample_1,interval[0],interval[1],method="simpson"))
    print("[Eval] Trapezoid: ", numerical_integral(func_sample_1,interval[0],interval[1],method="trapezoid"))
    print("[Info] Analytically it is exactly 2.")
    

    # Part B - Comparison with Analytical Eval & Roundoff Chasing
    interval_3 = [-1,1]  # Do not change this
    print(f"[Init] Integral evaluations in {interval_3} using Simpsons & Trapezoid")
    two_to_what_2 = [-19,-23]
    part_A_trials = [2**i for i in range(two_to_what_2[0],two_to_what_2[1],-1)]
    int_func_collec_sim, int_func_collec_tra = [],[]
    subreg = 200  # Will partition the interval into 200 subregs
    # Simpsons and Trapezoid rules evaled here:
    for i in range(len(part_A_trials)):
        x_pts_tmp = np.arange(interval_3[0],interval_3[1],part_A_trials[i])
        func_sample_tmp = func(x_pts_tmp)
        total_len = len(x_pts_tmp)
        tmp_col_sim, tmp_col_tra = [],[]
        for j in range(subreg):  
            func_sample_tmp_n = func_sample_tmp[int(total_len*i/subreg):int(total_len*(i+1)/subreg)]
            new_interv_0 = interval_3[0]+i/(subreg/2)
            new_interv_1 = interval_3[0]+(i+1)/(subreg/2)
            int_func_sample_tmp = numerical_integral(func_sample_tmp_n,new_interv_0,new_interv_1,method="simpson")
            tmp_col_sim.append(int_func_sample_tmp)
            int_func_sample_tmp_2 = numerical_integral(func_sample_tmp_n,new_interv_0,new_interv_1,method="trapezoid")
            tmp_col_tra.append(int_func_sample_tmp_2)
        int_func_collec_sim.append(np.array(tmp_col_sim))
        int_func_collec_tra.append(np.array(tmp_col_tra))
    # Analytical eval for subregs here:
    int_f = lambda x: x+1/4 - np.cos(np.pi*x+np.pi/4)/np.pi
    analytical_collec = []
    for j in range(subreg):  
        new_interv_0 = interval_3[0]+j/(subreg/2)
        new_interv_1 = interval_3[0]+(j+1)/(subreg/2)
        analytical_collec.append(int_f(new_interv_1)-int_f(new_interv_0))
    analytical_collec = np.array(analytical_collec)
    print(f"[Eval] Analytical result calculated by the computer: {np.sum(analytical_collec)}")
    
    # Plotting
    x_pts_2 = np.arange(interval_3[0], interval_3[1], 1/(subreg/2))
    # Plotting pc error of Simpsons and Trapezoid at subregs wrt analyt.
    plt.plot(x_pts_2, 100*(int_func_collec_sim[0]-analytical_collec)/analytical_collec,'-.',\
                      label="Simpsons")
    plt.plot(x_pts_2, 100*(int_func_collec_tra[0]-analytical_collec)/analytical_collec,'-.',\
                      label="Trapezoid")
    plt.grid()
    plt.legend()
    plt.ylabel("Error pc")
    plt.title("Comparison of Simpsons & Trapezoid wrt Analytical at Regs")
    plt.show()
    # Plotting plausible fluctuation region:
    for j in [int_func_collec_sim, int_func_collec_tra]:
        for i in range(len(j)-1):
            plt.plot(x_pts_2, j[i]-j[i+1], label=f"F_2^{two_to_what_2[0]-i} - F_2^{two_to_what_2[0]-i-1}")
        plt.grid()
        plt.legend()
        plt.title(r"Integ. diffs @neighborhoods for different h vals")
        plt.show()
    

if __name__== "__main__":
    mytests()
