'''
Problem 17

Functions:
--quad_custom(fun, n=10)  Uses Gauss-Legendre Quadrature to evaluate the
integral of function fun in [-1,1], n refers to n of P_n Legendre polyn
to be used

--mytests() Solves the questions in the problem set
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg
from ps0514 import numerical_integral


def quad_custom(fun,n=10):
    P_n = [0 for i in range(1,n)]+[1]
    P_n_roots = leg.legroots(P_n)
    d_P_n = leg.legder(P_n)
    d_P_n_vals = leg.legval(P_n_roots, d_P_n)
    weights = 2/((1-np.power(P_n_roots,2))*np.power(d_P_n_vals,2))
    result = np.sum(weights*fun(P_n_roots))
    return result


def mytests():
    # Part A
    print("\nPart A-----\n")
    func_A = lambda x: 1 + np.sin(np.pi*x + np.pi/4)
    for i in range(10,21):
        res = quad_custom(func_A,i)
        print(f"[Eval] n={i}: ", res, f"\tPc Error: {100*(res-2)/2}")

    # Calculating the h required for a similar error when n=20 above, using Simpson's method
##    wanted_error_pc_abs = 8.881e-14
##    for i in range(12):  # Capped
##        sample_h = 2**(-14-i)
##        x_pts = np.arange(-1,1,sample_h)
##        func_sample = func_A(x_pts)
##        res = numerical_integral(func_sample,-1,1)
##        if np.abs(100*(res-2)/2) < wanted_error_pc_abs:
##            print(f"\n[Info] Found: h=2^{-14-i}")
##            break
##    # I could have use Cahan summation instead in numerical_integral(),
##    # memory req. is too much for keeping all infinitesimal areas for pairwise summation at the end

    # Part B
    print("\nPart B-----\n")
    func_B = lambda u: 2*np.exp((u+1)/(u-1))*np.power(u-1,-2)
    # Substituted x with (-u-1)/(u-1) to switch the boundary to [-1,1]
    print("[Eval] Result: ", quad_custom(func_B,20))


if __name__ == "__main__":
    mytests()
