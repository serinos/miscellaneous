'''
May 4 2020
Misc tester for beam.py library - Finding Effective Jsat

Note that you need to have beam.py in the same directory to run the code

--Definitions:
Effective Jsat := Esat/(pi*(25**2)) where Esat := E_p such that deltaΕ/Ε = initial_E_fraction/2

--Conclusion after the test:
E_sat definition cannot be satisfied for some E_p values, might not be satisfiable for any.
'''

from beam import *


def tester(cursor):  # Takes E_p, outputs deltaE_over_E after
                     # passing the beam with E_p and w=25um through
                     # a zebra graphene layer (pattern:5um ablated&5um graphene)
                     # Use the output to compare for yourself with the
                     # initial delta_E_over_E to see whether
                     # (1/2)*delta_E_over_E is feasible to be achieved
    ee = beam_initialize(res=15, threshold=10**-10, w=25, Ep=cursor)
    mm = mask_initialize(beam=ee, shape='lines',width=5,thickness=5)
    ss = mask_apply(ee,mm)
    Eval = integrate_for_energy(ee)
    tmp_dE_E = (Eval - integrate_for_energy(ss))/Eval
    print(f"E_p: {Eval}uJ, fraction: {tmp_dE_E}")

# The following beam is the one that we would like to find the effective Jsat for:
Einput = np.pi*(25**2)*0.00000015*10  # Einput will be 10-fold of [Jsat*Area]
                                      # Jsat is given as 0.00000015 uJ/um2
ee = beam_initialize(res=15, threshold=10**-10, w=25, Ep=Einput)
Ein = integrate_for_energy(ee)
mm = mask_initialize(beam=ee, shape='lines',width=5,thickness=5)
ss = mask_apply(beam=ee,mask=mm)
initEfrac_over2 = ((Ein-integrate_for_energy(ss))/Ein)/2
print(f"Objective: Achieve {initEfrac_over2} by changing E_p (={Ein}uJ)")
# We would like to achieve (1/2)*delta_E_over_E with tinkering with E_p

# The following code compares the delta_E_over_E of some E_p values
# with initEfrac_over2
print("Results:")
tester(Ein*0.5)
tester(Ein*5)  # 5 times initial E_p and so on...
tester(Ein*15)
tester(Ein*100)
tester(Ein*1000)
# As you can see, there seems to be an upper bound, and we cannot achieve
# half of the initial delta_E_over_E
