'''
Problem 13
Arbitrary Finite Distance Coefficient Calculator
Proof of the formula is in the solutions pdf

--findiff(pts,deg) finds the coefficients a_i corresponding to f(x+pts[i]*h)
for approximating the deg-th degree derivative of f(x), and returns a_i vals
'''

import numpy as np
from math import factorial as fact


def findiff(pts, deg=1):
    pts_len = len(pts)

    ##### Some sanity checks
    if deg >= pts_len:
        raise Exception(r"Degree too large for # of given pts")
    else:
        for i in range(pts_len):
            for j in range(i+1,pts_len):
                if pts[i]==pts[j]:
                    raise Exception(r"Some points recur")
    #####

    A = np.array([np.power(pts,i) for i in range(pts_len)])
    b = np.where(np.arange(0,pts_len)==deg, 1, 0)*fact(deg)
    soln = np.linalg.solve(A,b)

    return soln
