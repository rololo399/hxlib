# -*- coding: utf-8 -*-
"""
Table C.1 
One dimensional steady state solutions to the Heat Equation for 
plane, cylinder and spherical walls with uniform generation and 
asymmetrical surface conditions
"""

import numpy as np

pi = np.pi

def plane_temperature(x, L, T1, T2, qdot, k):
    d1 = qdot*L**2/(2*k)
    d2 = 1 - (x/L)**2
    d3 = (T2 - T1)/2
    d4 = (T2 + T1)/2
    T = d1*d2 + d3*x/L + d4
    return T

def plane_heatflux(x, L, T1, T2, qdot, k):
    qf = qdot*x - k/(2*L)*(T2 - T1)
    return qf

def cylinder_temperature(r, r1, r2, T1, T2, qdot, k):
    d1 = qdot*r2**2/(4*k)
    d2 = 1 - (r/r2)**2
    d3 = 1 - (r1/r2)**2
    d4 = np.log(r2/r)/np.log(r2/r1)
    T = T2 + d1*d2 - (d1*d3 + (T2-T1))*d4    
    return T

def cylinder_heatflux(r, r1, r2, T1, T2, qdot, k):
    d1 = qdot*r/2
    d2 = qdot*r2**2/(4*k)
    d3 = 1 - (r1/r2)**2
    d4 = r*np.log(r2/r1)
    qf = d1 - k*(d2*d3 + (T2-T1))/d4
    return qf

def cylinder_heatrate(r, r1, r2, T1, T2, qdot, k, L):
    d1 = qdot*r2**2/(4*k)
    d2 = 2*pi*L*k/np.log(r2/r1)
    d3 = 1 - (r1/r2)**2
    q = qdot*pi*L*r**2 - d2*(d1*d3 + (T2-T1))
    return q

def sphere_temperature(r, r1, r2, T1, T2, qdot, k):
    d1 = qdot*r2**2/(6*k)
    d2 = (1/r -1/r2)/(1/r1 - 1/r2)
    d3 = 1 - (r1/r2)**2
    T = T2 + d1*(1-(r/r2)**2) - (d1*d3 + (T2-T1))*d2    
    return T

def sphere_heatflux(r, r1, r2, T1, T2, qdot, k):
    d1 = qdot*r2**2/(6*k)
    d3 = 1 - (r1/r2)**2
    qf = qdot*r/3 - k*(d1*d3 + (T2-T1))/(r**2*(1/r1 - 1/r2))
    return qf
def sphere_heatrate(r, r1, r2, T1, T2, qdot, k):
    d1 = qdot*r2**2/(6*k)
    d2 = qdot**4*np.pi*r**3/3
    d3 = 1 - (r1/r2)**2    
    q = d2 - 4*np.pi*k*(d1*d3 + (T2-T1))/(1/r1 - 1/r2)
    return q


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    from scipy.optimize import newton
    
    k = 5
    r1, r2 = 20e-3, 25e-3
    qdot = 5e6
    Tinf = 80
    h = 110
    Q = pi*(r2**2-r1**2)*qdot
    T1 = Tinf + Q/(2*pi*r1*h)

    def fun(T2):
        qf = cylinder_heatflux(r2, r1, r2, T1, T2, qdot, k)
        return qf
    T2 = newton(fun, 400)
    print(T1, T2)
    
    r = np.linspace(r1, r2)
    T = cylinder_temperature(r, r1, r2, T1, T2, qdot, k)
    q = cylinder_heatrate(r, r1, r2, T1, T2, qdot, k, 1)
    
    plt.plot(r, T)
    plt.show()
    
    plt.plot(r, q)
    plt.show()