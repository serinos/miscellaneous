'''
Problem 11 - Finding $\Omega_M$ by Hubble Telescope Data

Functions:
--read_hubble_data(filename) Returns z, dL, error vals in 3 separate arrays

--hubble_integral(z, omega_m) Takes the following integral:
$\int_0^z\frac{dz'}{\sqrt(\Omega_M(1+z')^3+(1-\Omega_M))}$

--hubble_integral_del_z(z, omega_m) Partial deriv of int wrt z

--hubble_integral_del_O_M(z, omega_m) Partial deriv of int wrt $\Omega_M$

--lin_hubble_fit(df_hubble) Linear fit for z<0.1 data

--manual_nonlin_hubble_fit(z_arr, dL_arr) See the description inside func

--built_in_nonlin_hubble_fit(z_arr, dL_arr) Uses curve_fit of scipy.optimize

--nonlin_fit_with_err(z_arr, dL_arr, err_arr, method, count) does the following:
Returns dH list and O_M list calculated by method 'builtin' or 'manual' using
Hubble z and np.random.normal(d_L,err) vals. Thus we can use the returned lists
in histograms to see the distribution of d_H and O_M caused by deviation from the mean
data represented in col2 of Hubble data

--mytests() Solves questions in the PS
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.stats import linregress
from scipy.optimize import root
from scipy.optimize import curve_fit


def read_hubble_data(filename="hubble.dat"):
    z_arr, dL_arr, err_arr = [],[],[]
    with open(f"{filename}", 'r') as fh:
        for line in fh:
            (col1, col2, col3) = line.split()
            z_arr.append(np.float(col1))
            dL_arr.append(np.float(col2))
            err_arr.append(np.float(col3))
    return np.array(z_arr), np.array(dL_arr), np.array(err_arr)


def hubble_integral(z, omega_m):
    I_fn = lambda x: 1/np.sqrt(omega_m*np.power(1+x, 3) + 1 - omega_m)
    result = quad(I_fn,0,z)
    return result[0]  # Only evaluation passed


def hubble_integral_del_z(z, omega_m):
    I_fn = lambda x: 1/np.sqrt(omega_m*np.power(1+x, 3) + 1 - omega_m)
    result = I_fn(z)-I_fn(0)  # By fund thm of calc
    return result


def hubble_integral_del_O_M(z, omega_m):
    I_2_fn = lambda x: (np.power(x+1,3) - 1)/(2*np.power(omega_m*np.power(x+1,3)-omega_m+1,3/2))
    result = quad(I_2_fn,0,z)
    return result[0]  # Only evaluation passed


def manual_nonlin_hubble_fit(z_arr, dL_arr):
    # Solves Part C, returns best fit for d_L and O_M
    # To assist nonlin fit, take d_H around magnitude of slope in lin fit, and 0<O_M<1
    # d_L(z, d_H, O_M) is to be fitted with fitting parameters d_H, O_M
    # d_L(z) = (1+z)*d_H*hubble_integral(z,O_M)
    # Would like to minimize RMS of d_L(z) - d_L_data_at_z.
    # E_2_del_dH and E_2_del_OM are to be zero

    data_len = len(z_arr)
    x_0 = (4500,0.5)  # Initial guess for (d_H, O_M)
    
    E_2_del_dH_single = lambda z_k,dH,O_M,D_k:\
            2*(dH*(z_k+1)*hubble_integral(z_k, O_M) - D_k)*(z_k+1)*(hubble_integral(z_k, O_M))
    E_2_del_OM_single = lambda z_k,dH,O_M,D_k:\
            2*(dH*(z_k+1)*hubble_integral(z_k, O_M) - D_k)*dH*(z_k+1)*hubble_integral_del_O_M(z_k, O_M)

    E_2_del_dH = lambda dH,O_M: np.sum([ E_2_del_dH_single(z_arr[i],dH,O_M,dL_arr[i]) for i in range(data_len) ])
    E_2_del_OM = lambda dH,O_M: np.sum([ E_2_del_OM_single(z_arr[i],dH,O_M,dL_arr[i]) for i in range(data_len) ])
    E_2_total = lambda x: (E_2_del_dH(x[0],x[1]), E_2_del_OM(x[0],x[1]))

    return root(E_2_total, x_0, method="hybr")


def built_in_nonlin_hubble_fit(z_arr, dL_arr):
    dL_func = np.vectorize(lambda z,dH,O_M: (1+z)*dH*hubble_integral(z,O_M))
    x_0 = [4500,0.5]  # Initial guess, same as manual_nonlin_hubble_fit()
    boundaries = ((3000,0.01),(6000,1))  # Used 3000<d_H<6000 and 0.01<O_M<1 as bounds
    results = curve_fit(dL_func, z_arr, dL_arr, p0=x_0, bounds=boundaries)
    return results


def lin_hubble_fit(df_hubble):
    # Linear fit for d_L(z)=az+b using z<0.1 data
    z_small_arr = np.array(df_hubble[df_hubble['z']<0.1]['z'])
    dL_small_arr = np.array(df_hubble[df_hubble['z']<0.1]['dL'])
    
    linear_dL_data = linregress(z_small_arr, dL_small_arr)
    x_vals = np.linspace(z_small_arr[0], z_small_arr[-1],30)
    linear_dL_points = (x_vals*linear_dL_data[0]) + linear_dL_data[1]
    return linear_dL_data, z_small_arr, dL_small_arr, x_vals, linear_dL_points


def nonlin_fit_with_err(z_arr, dL_arr, err_arr, method='builtin', count=20):
    dH_dist, O_M_dist = [],[]
    if method == 'manual':
        index = 0
        while True:
            dL_arr_new = np.random.normal(dL_arr, err_arr)
            part_result = manual_nonlin_hubble_fit(z_arr, dL_arr_new)
            if part_result.success == True:  # Drop calc if tol not sufficed,
                            # Could introduce bias together with init val selection
                dH_dist.append(part_result.x[0])
                O_M_dist.append(part_result.x[1])
                print(f"Instance[{index}]: Found d_H = {part_result.x[0]} and O_M = {part_result.x[1]}")
                count -= 1
                if count == 0:
                    break
            else:
                print(f"Instance[{index}]: Failed to find solutions at desired tolerance")
            index += 1
    else:
        for i in range(count):
            dL_arr_new = np.random.normal(dL_arr, err_arr)
            part_result = built_in_nonlin_hubble_fit(z_arr, dL_arr_new)
            dH_dist.append(part_result[0][0])
            O_M_dist.append(part_result[0][1])
            print(f"Instance[{i}]: Found d_H = {part_result[0][0]} and O_M = {part_result[0][1]}")
            
    return dH_dist, O_M_dist


def mytests():
    # Initializing datapoints
    z_arr, dL_arr, err_arr = read_hubble_data()
    
    df_hubble = pd.DataFrame([[z_arr[i], dL_arr[i], err_arr[i]] \
                             for i in range((len(z_arr)))],\
                             columns=['z', 'dL', 'error'])
    
    # Part A -----
    # Plotting Hubble z vs luminosity data with error bars
    plt.plot(z_arr, dL_arr, 'bo')
    plt.errorbar(z_arr, dL_arr, err_arr, ls='')
    plt.xlabel("z")
    plt.ylabel("d_L (Mpc)")
    plt.title("z vs luminosity distance")
    plt.grid()
    plt.show()

    # Part B -----
    linear_dL_data, z_small_arr, dL_small_arr, x_vals, linear_dL_points = lin_hubble_fit(df_hubble)
    plt.plot(z_small_arr, dL_small_arr, 'bo', label="Data")
    plt.plot(x_vals, linear_dL_points,'r-', label="Linear Fit")
    plt.xlabel("z")
    plt.ylabel("d_L (Mpc)")
    plt.title("z vs luminosity distance")
    plt.legend()
    plt.grid()
    plt.show()

    # Part C -----
    # Manual nonlinear fit for given function d_L(z, d_H, O_M)
    part_c_result = manual_nonlin_hubble_fit(z_arr, dL_arr)
    print("Part C -----\nCalculating...")
    print(part_c_result)
    print(f"\nThus d_H = {part_c_result.x[0]} and O_M = {part_c_result.x[1]}")
    print("-----")

    # Part D -----
    # We will now see how d_H and O_M turn out by handwritten nonlin fit when..
    # we take into account the error in measurements, will introduce noise by
    # using the standard error given in col3 of data and Gaussian distr. Then
    # we can plot a histogram of d_H and O_M
    part_d_count = 20  # Let us eval 20 instances
    print(f"Part D -----\nGathering {part_d_count} data points - Please be patient")
    dH_dist, O_M_dist = nonlin_fit_with_err(z_arr, dL_arr, err_arr,\
                                            method='manual', count=part_d_count)
    
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(dH_dist, density=True)
    axes[0].set_title("d_H")
    axes[1].hist(O_M_dist, density=True)
    axes[1].set_title("Omega_M")
    plt.show()

    # Part E -----
    # Redoing parts C,D with built-in curve_fit of scipy.optimize
    part_e_results = built_in_nonlin_hubble_fit(z_arr, dL_arr)
    print("Part E -----\nCalculating without considering error margins...")
    print(f"Found: d_H = {part_e_results[0][0]} and O_M = {part_e_results[0][1]}")
    print(f"Part E with errors taken into account -----")
    
    part_e_count = 30
    print("Gathering {part_e_count} data points - Please be patient")
    dH_dist2, O_M_dist2 = nonlin_fit_with_err(z_arr, dL_arr, err_arr,\
                                              method='builtin', count=part_e_count)
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(dH_dist2, density=True)
    axes[0].set_title("d_H")
    axes[1].hist(O_M_dist2, density=True)
    axes[1].set_title("Omega_M")
    plt.show()

    
if __name__ == "__main__":
    mytests()
