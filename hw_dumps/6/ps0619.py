'''
Problem 19
Numerical investigation of the Sun & Jupiter 2-body system
Comparing different numerical methods
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from time import time


### Constants, Units
G = 6.674e-11  # m^3kg^-1s^-2
v_p = 1.372e4  # ms^-1  Speed of Jupiter at perihelion
r_p = 7.405e11 # m    Distance between the Sun and Jupiter at perihelion
m_j = 1.898e27 # kg   Mass of Jupiter
m_s = 1.988e30 # kg   Mass of the Sun
mu_red = m_s*m_j/(m_s+m_j)  # kg  Reduced mass
mu_J = m_j/mu_red  # Reduced mass of J
mu_S = m_s/mu_red  # Reduced mass of S
t_0 = np.sqrt(r_p**3 /(G*(m_s+m_j)))  # Natural time scale
v_p_nat = v_p * t_0 / r_p
r_p_nat = 1
###


def time_step(vec,t,rhs,dt,method=2):
    # Solves 2nd order ODEs of form r''=F(r,t), vec is [r_pos,r_vel],
    # the dimensions of the two elements of vec must be equal
    # RHS is a function that depends on vec[0] and t
    # Solves from t to t+dt, 1 step
    # Methods: Forward Euler (1), Symplectic Euler (2),
    # Implicit Euler (3), RK4 (4)
    r_pos = vec[0]
    r_vel = vec[1]
    
    if method==1 :
        r_vel_new = r_vel + dt*rhs(r_pos,t)
        r_pos_new = r_pos + dt*r_vel
        
    elif method==2 :
        r_vel_new = r_vel + dt*rhs(r_pos,t)
        r_pos_new = r_pos + dt*r_vel_new

    elif method==3 :
        fn_imp = lambda r_vel_new: r_vel_new-dt*rhs(r_pos+dt*r_vel_new,t+dt)-r_vel
        fn_root = root(fn_imp,r_vel)
        r_vel_new = fn_root.x
        r_pos_new = r_pos+dt*r_vel  # Using r_vel_new here results in some interesting behavior

    else:
        # RK4 subroutine
        f_1 = rhs(r_pos,t)
        f_2 = rhs(r_pos+(dt*f_1/2),t+dt/2)
        f_3 = rhs(r_pos+(dt*f_2/2),t+dt/2)
        f_4 = rhs(r_pos+dt*f_3,t+dt)
        r_vel_new = r_vel + (dt/6)*(f_1+2*f_2+2*f_3+f_4)
        z_1 = r_vel_new
        z_2 = r_vel_new+dt*z_1/2
        z_3 = r_vel_new+dt*z_2/2
        z_4 = r_vel_new+dt*z_3
        r_pos_new = r_pos + (dt/6)*(z_1+2*z_2+2*z_3+z_4)

    return np.array([r_pos_new,r_vel_new])
    

def sun_jup_iterate(t_tot, dt_val, method_no=2):
    cursor_sun = [np.array([0,0]),np.array([0,0])]
    cursor_jup = [np.array([1,0]), np.array([0,v_p_nat])]
##    cursor_3rd=[np.array([np.cos(np.pi/6),np.sin(np.pi/6)]),np.array([0,0])]
##    mass_3rd = 1e10
##    mu_3rd = mass_3rd/mu_red  # Converting from kg to reduced mass

    pos_sun, pos_jup, vel_sun, vel_jup, pos_3rd, vel_3rd = [],[],[],[],[],[]
    t = 0
    pos_sun.append(cursor_sun[0])
    pos_jup.append(cursor_jup[0])
    vel_sun.append(cursor_sun[1])
    vel_jup.append(cursor_jup[1])
##    pos_3rd.append(cursor_3rd[0])
##    vel_3rd.append(cursor_3rd[1])
    
    while t <= t_tot:
        r_S = cursor_sun[0]
        r_J = cursor_jup[0]
        rhs_sun = lambda r,t: -(r-r_J)/(mu_S*(np.sum((r-r_J)**2))**(3/2))
        rhs_jup = lambda r,t: -(r-r_S)/(mu_J*(np.sum((r-r_S)**2))**(3/2))
        cursor_sun = time_step(cursor_sun,t,rhs_sun,dt_val,method_no)
        cursor_jup = time_step(cursor_jup,t,rhs_jup,dt_val,method_no)
##        if is_3_body==True:  # Reduced 3 body
##            r_3rd = cursor_3rd[0]
##            rhs_3rd = lambda r,t: -(r-r_J)/(mu_3rd*(np.sum((r-r_J)**2))**(3/2))\
##                                  -(r-r_S)/(mu_3rd*(np.sum((r-r_S)**2))**(3/2))
##            cursor_3rd = time_step(cursor_3rd,t,rhs_3rd,dt_val,method_no)
##            pos_3rd.append(cursor_3rd[0])
##            vel_3rd.append(cursor_3rd[1])
##        else:
##            pass
        pos_sun.append(cursor_sun[0])
        pos_jup.append(cursor_jup[0])
        vel_sun.append(cursor_sun[1])
        vel_jup.append(cursor_jup[1])
        t += dt_val  
##    if is_3_body==True:
##        return pos_sun,pos_jup,vel_sun,vel_jup,pos_3rd,vel_3rd
    return pos_sun,pos_jup,vel_sun,vel_jup


def mytests():
    # Calculating and plotting Sun&Jupiter orbits, using all the methods
    # To be plotted: Orbitals, Energies
    t_tot = 100  # in modif. units
    dt_val = 1/64
    for method_no in range(1,5):
        timer = time()
        pos_sun,pos_jup,vel_sun,vel_jup = sun_jup_iterate(t_tot,dt_val,method_no)
        timer = time() - timer
        print(f"[Info] Method {method_no} took {timer} sec")
        # Plotting the orbits
        jup_x = [i[0] for i in pos_jup]
        jup_y = [i[1] for i in pos_jup]
        sun_x = [i[0] for i in pos_sun]
        sun_y = [i[1] for i in pos_sun]
        plt.plot(jup_x,jup_y, label="Jupiter")
        plt.plot(sun_x,sun_y, label="The Sun")
        plt.plot()
        plt.grid()
        plt.legend()
        plt.title(f"Sun&Jupiter, Method {method_no}")
        plt.show()
        # Plotting L_tot
        data_len = len(pos_jup)
        plot_x_range = t_0*np.arange(0, t_tot+2*dt_val, dt_val)
        jup_ang_mom = mu_J*np.array([np.cross(pos_jup[i],vel_jup[i]) for i in range(data_len)])
        sun_ang_mom = mu_S*np.array([np.cross(pos_sun[i],vel_sun[i]) for i in range(data_len)])
        tot_ang_mom = (jup_ang_mom + sun_ang_mom)*mu_red*(r_p**2)/t_0
        plt.plot(plot_x_range, tot_ang_mom)
        plt.grid()
        plt.ylabel("L (kgm^2s^-1)")
        plt.xlabel("Time (s)")
        plt.title("L (z^)")
        plt.show()
        # Plotting E_tot
        mu_for_pot = 1/mu_J + 1/mu_S
        f_pot = lambda pos_1,pos_2: mu_for_pot*(-1/np.linalg.norm(pos_1-pos_2))
        jup_T = (1/2)*mu_J*np.array([np.sum(vel_jup[i]**2) for i in range(data_len)])
        sun_T = (1/2)*mu_S*np.array([np.sum(vel_sun[i]**2) for i in range(data_len)])
        pot_vals = [f_pot(pos_jup[i],pos_sun[i]) for i in range(data_len)]
        tot_E = (jup_T + sun_T + pot_vals)*mu_red*(r_p**2)/(t_0**2)
        plt.plot(plot_x_range, tot_E)
        plt.grid()
        plt.ylabel("Energy (J)")
        plt.xlabel("Time (s)")
        plt.title("E_tot")
        plt.show()
        # Plotting P
        jup_P = mu_J*np.array(vel_jup)
        sun_P = mu_S*np.array(vel_sun)
        tot_P = (jup_P + sun_P)*mu_red*r_p/t_0
        tot_P_sca = np.array([np.linalg.norm(tot_P[i]) for i in range(data_len)])
        plt.plot(plot_x_range, tot_P_sca)
        plt.grid()
        plt.ylabel("P (kgms^-1)")
        plt.xlabel("Time (s)")
        plt.title("Total P")
        plt.show()
        
        
if __name__ == "__main__":
    mytests()
