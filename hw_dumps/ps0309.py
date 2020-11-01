'''
Problem 9  # TODO: PART C STILL NOT IMPLEMENTED
Functions:

--newton(fun, dfun, x_init) implements the Newton-Raphson method to solve
fun(x)=0. x_init is the initial guess, dfun() is the derivative of fun()

--halley(fun, dfun, ddfun, x_init) implements Halley's method to solve
fun(x)=0. x_init is the initial guess, dfun() is the derivative of fun()
ddfun() second derivative of fun()

--lambertw_custom(x, method) calculates the Lambert W (principal branch) 
function defined as W(xe^x)=y if y=xe^x, using the method specified.
For Halley's, method='h'
For Newton-Raphson, method='n'; defaults to Newton-Raphson otherwise

--koc_lambertw(x) implements the inverse function of f(x^2e^(x^2))=y

--mytests() performs the required tests, its aims are:
* Comparison of Newton Raphson and Halley methods with respect to speed.
* Comparison of builtin lambertw() output and outputs of hand-made funcs
  that use the aforementioned methods.
'''

import time
import numpy as np
import scipy.linalg as linalg
import scipy.special as special
import matplotlib.pyplot as plt


def lambertw_custom(x, method='n'):
    x = np.array(x)
    fun_shifted = lambda z: z*np.exp(z)-x
    fun_deriv = lambda z: (z+1)*np.exp(z)
    fun_deriv2 = lambda z: (z+2)*np.exp(z)
    x_init = np.where(x<np.e,1,np.log(x))  # x=ye^y --> lnx =y+lny, lnx approx y for large y

    if method=='h':
        y = halley(fun_shifted, fun_deriv, fun_deriv2, x_init)
    else:
        y = newton(fun_shifted, fun_deriv, x_init)

    return y


def lambertw_custom_npvec(x, method='n'):  # np.vectorize() variant
    x = np.array(x)
    fun_shifted = np.vectorize(lambda z: z*np.exp(z)-x)
    fun_deriv = np.vectorize(lambda z: (z+1)*np.exp(z))
    fun_deriv2 = np.vectorize(lambda z: (z+2)*np.exp(z))
    x_init = np.where(x<np.e,1,np.log(x))  # x=ye^y --> lnx =y+lny, lnx approx y for large y

    if method=='h':
        y = halley(fun_shifted, fun_deriv, fun_deriv2, x_init)
    else:
        y = newton(fun_shifted, fun_deriv, x_init)

    return y


def newton(fun, dfun, x_init):
    root = x_init
    tol = 1e-15
    max_iter = 10000  # Capping to prevent inf loops
    
    for i in range(max_iter):
        if linalg.norm(fun(root)) < tol:
            break
        root = root - fun(root)/dfun(root);

    return root


def halley(fun, dfun, ddfun, x_init):
    root = x_init
    tol = 1e-15
    max_iter = 10000  # Capping to prevent inf loops

    for i in range(max_iter):
        if linalg.norm(fun(root))<tol:
            break
        fun_eval = fun(root)
        dfun_eval = dfun(root)
        ddfun_eval = ddfun(root)
        root = root - (2*fun_eval*dfun_eval)/(2*(dfun_eval**2)-fun_eval*ddfun_eval)

    return root


def koc_lambertw(x):
    ans = np.sqrt(special.lambertw(x))
    return ans
    

def mytests():
    # Test parameters:
    test_interval = [-0.99/np.e, 1e20]
    test_point_count = 250
    test_points = np.linspace(test_interval[0], test_interval[1], test_point_count)

    # Will compare output of custom ones with the builtin, plot the abs diff:
    Lw_points_newton = lambertw_custom(test_points, 'n')
    Lw_points_halley = lambertw_custom(test_points, 'h')
    Lw_points_builtin = np.real(special.lambertw(test_points))
    
    Lw_abs_diff_newton_builtin = np.abs(Lw_points_newton - Lw_points_builtin)
    Lw_abs_diff_halley_builtin = np.abs(Lw_points_halley - Lw_points_builtin)

    # Plot
    plt.plot(test_points, Lw_abs_diff_newton_builtin, label=r"Newton")
    plt.plot(test_points, Lw_abs_diff_halley_builtin, label=r"Halley")
    plt.xlabel("Inputs")
    plt.ylabel("Abs error wrt builtin")
    plt.legend()
    plt.title(r"Abs error of Newton, Halley methods wrt Built-in")
    plt.show()

    # Timing comparisons:
    # Will compare the speed of the three functions of interest by taking the
    # average of 10 instances of execution time for each of them using
    # test_points as initialized above. Also will compare the effect of np.vectorize
    # vs built-in element-wise operators on timing
    instance_count = 10
    time_data = [[0 for j in range(instance_count)] for i in range(5)]
    
    for i in range(instance_count):
       timer = time.time()
       tmp = lambertw_custom(test_points, 'n')
       time_data[0][i] = time.time() - timer  # Newton
       
       timer = time.time()
       tmp = lambertw_custom(test_points, 'h')
       time_data[1][i] = time.time() - timer  # Halley
       
       timer = time.time()
       tmp = special.lambertw(test_points)
       time_data[2][i] = time.time() - timer  # Built-in
       
       timer = time.time()
       tmp = lambertw_custom(test_points, 'n')
       time_data[3][i] = time.time() - timer  # Newton, w/ np.vectorize
       
       timer = time.time()
       tmp = lambertw_custom(test_points, 'h')
       time_data[4][i] = time.time() - timer  # Halley, w/ np.vectorize
    
    time_data = np.sum(np.array(time_data),1)
    print(f"Interval: [{test_interval[0]} , {test_interval[1]}]")
    print(f"Average values for completion of {test_point_count} evaluations")
    print(f"Average time for Newton: {time_data[0]} sec")
    print(f"Average time for Halley: {time_data[1]} sec")
    print(f"Average time for Builtin: {time_data[2]} sec")
    print(f"Average time for Newton with np.vectorize: {time_data[3]} sec")
    print(f"Average time for Halley with np.vectorize: {time_data[4]} sec")

    # Plotting real part of koc_lambertw in [1e-20,1e20]:
    koc_lw_interval = np.logspace(-20,20,10000)
    koc_lw_eval = np.real(koc_lambertw(koc_lw_interval))
    plt.loglog(koc_lw_interval, koc_lw_eval)
    plt.title("Koc Lambert w")
    plt.show()


if __name__ == "__main__":
    mytests()
