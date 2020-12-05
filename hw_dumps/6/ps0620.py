'''
Problem 20
Numerical investigation of a simple fireball model,
and a perturbed harmonic oscillator
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.integrate import solve_ivp
from time import time
from ps0619 import time_step


##### Initializations for Part B, fireball
#rho=r/r_0
#tau=t/t_0
#r_0 = c_2/c_3
#t_0 = c_3/c_2**2
#So, dr/dt = c_2*r**2 - c_3*r**3 --> drho/dtau = rho**2 - rho**3

# rho(tau) analytical solution:  (delta := rho(0))
rho_analytical = lambda tau,delta:\
                 np.real(1/(lambertw(((1/delta)-1)*np.exp(((1/delta)-1)-tau))+1))
# We will work on the ODE numerically:
rho_rhs = lambda rho,dummy: rho**2 - rho**3  # drho/dtau = rho**2 - rho**3
jac = lambda rho,dummy: [[2*rho - 3*rho**2]]  # 1x1 Jacobian
######


def part_c(dt,L): # Perturbed harmonic potential
    # Trying to see what happens when we lower L, and the effect of dt of choice
    L_2 = L**2
    rhs = lambda x,t: -x + x*np.exp(-(x**2)/2*L_2)/L_2
    cursor_particle = [np.array([1]),np.array([0])]
    pos_particle = [cursor_particle[0]]
    vel_particle = [cursor_particle[1]]
    t = 0  # Working in natural time scale t <-- wt
    t_tot = 10
    t_vals = [0]
    while t<t_tot:
        cursor_particle = time_step(cursor_particle,t,rhs,dt,method=4)
        pos_particle.append(cursor_particle[0])
        vel_particle.append(cursor_particle[1])
        t += dt
        t_vals.append(t)
    plt.plot(t_vals,pos_particle)
    plt.ylabel("Position")
    plt.xlabel("Natural time")
    plt.title(f"dt={dt}, l={L}")
    plt.grid()
    plt.show()
    # Now let us use solve_ivp RK45, which will use adaptive time stepping
    res = solve_ivp(rhs,[0,t_tot],pos_particle[0])
    print(f"[Info] RK45 successful: {res.success}, nfev={res.nfev}")
    plt.plot(res.t,res.y[0])
    plt.ylabel("Position")
    plt.xlabel("Natural time")
    plt.title(f"Built-in, l={L}")
    plt.grid()
    plt.show()


def mytests():
    # Plotting num. slns for fireball:
    delta = 2**(-8)
    for method in ["RK45","LSODA","BDF"]:
        if method=="BDF":
            res = solve_ivp(rho_rhs,[0,2/delta],[delta],method=method, jac=jac)
        else:
            res = solve_ivp(rho_rhs,[0,2/delta],[delta],method=method)
        print(f"[Info] {method} successful: {res.success}, nfev={res.nfev}")
        plt.plot(res.t,res.y[0],label=f"{method}")
    plt.title(f"Numerical Methods vs Analytical result")
    analytical = [rho_analytical(i,delta) for i in res.t]
    plt.plot(res.t,analytical,label="Analytical")
    plt.legend()
    plt.grid()
    plt.show()
    # Perturbed harmonic potential:
    part_c(dt=1/100,L=2**(-8)) # Goes to neg when no adaptive time
    part_c(dt=1/1500,L=2**(-8))  # Come-backs seem inevitable when no adaptive time
    part_c(dt=1/3000,L=2**(-12)) # Need to decrease dt to not go to neg decreasing L

        
if __name__ == "__main__":
    mytests()
