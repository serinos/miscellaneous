'''
Problem 23-C
(Proofs of Part A,B in the solutions pdf)
Comparison of Numerov's method with RK23 & RK45 using an undamped
harmonic oscillator with a sinusoidal driving force:
Eqn: x''(t) + x(t) = 2cos(3t), init vals x=0, x'=0
Analytical sln:  x(t) = sin^2(t)cos(t)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import time


def numerov_eval(gx, sx, h, end):
    # Implements Numerov's method, starts at x=0, init conds y=0, y'=0
    # Also assumes that y_1 = y_0 by approximating y'_0 to (y_1-y_0)/h
    # Returns an array of evaluations in the interval [0,end], (end/h entries)
    # Tweaked to decrease catastrophic cancellation
    y_vals = [0,0]
    h_2 = h**2
    iter_count = int(end/h)
    iter_array = np.linspace(0,end,iter_count)
    gx_vals = gx(iter_array)
    sx_vals = sx(iter_array)
    for i in range(2,iter_count):
        div_p = 1+(gx_vals[i]/12)*h_2
        part_1 = (2 -(5/6)*gx_vals[i-1]*h_2)*y_vals[i-1]
        part_2 = -(1+(gx_vals[i-2]/12)*h_2)*y_vals[i-2]
        part_3 = ((sx_vals[i]+10*sx_vals[i-1]+sx_vals[i-2])/12)*h_2
        y_new = (part_1+part_2+part_3)/div_p
        y_vals.append(y_new)
    return np.array(y_vals)


def mytests():
    func_sln = lambda t: np.cos(t)*(np.sin(t)**2)
    clock = []
    # Numerov
    gx = lambda t: np.ones(len(t))
    sx = lambda t: 2*np.cos(3*t)
    h = 0.001
    end = 20
    timer = time()
    y_numerov = numerov_eval(gx,sx,h,end)
    clock.append(time()-timer)
    # RK4 and RK45
    x_array = np.linspace(0,end,int(end/h))
    rhs = lambda t,y: [y[1], -y[0]+2*np.cos(3*t)]
    init_vals = [0,0]
    timer2 = time()
    rk23 = solve_ivp(rhs,[0,end],init_vals,t_eval=x_array, method="RK23")
    clock.append(time()-timer2)
    timer3 = time()
    rk45 = solve_ivp(rhs,[0,end],init_vals,t_eval=x_array, method="RK45")
    clock.append(time()-timer3)
    # Plot
    analytical = func_sln(x_array)
    plt.plot(x_array,analytical,label="Analytical")
    plt.plot(x_array,y_numerov,'--',label="Numerov")
    plt.plot(x_array,rk23.y[0],'--',label="RK23")
    plt.plot(x_array,rk45.y[0],'--',label="RK45")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amp")
    plt.grid()
    plt.title("x''(t) + x(t) = 2cos(3t), x(0)=x'(0)=0")
    plt.show()
    # Diff plot
    plt.plot(x_array,y_numerov-analytical, label="Numerov")
    plt.plot(x_array,rk23.y[0]-analytical, label="RK23")
    plt.plot(x_array,rk45.y[0]-analytical, label="RK45")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amp")
    plt.grid()
    plt.title("Difference wrt analytical result")
    plt.show()
    # Timings
    print(f"[Info] Time interval: [0,{end}], h={h}")
    print(f"[Info] Numerov:\t{clock[0]} sec")
    print(f"[Info] RK23:\t{clock[1]} sec")
    print(f"[Info] RK45:\t{clock[2]} sec")


if __name__ == "__main__":
    mytests()
