# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:17:36 2021

@author: user
"""
import numpy as np
from scipy.special import j0, j1
from heatlib import fzeros

def eigen_planewall(Bi, n=1):
    beta = fzeros(lambda x: x*np.sin(x) - Bi*np.cos(x), n=n)
    C = 4*np.sin(beta)/(2*beta + np.sin(2*beta))
    return beta, C

def eigen_cylinder(Bi, n=1):
    beta = fzeros(lambda x: x*j1(x) - Bi*j0(x), n=n)
    C = 2*j1(beta)/(beta*(j0(beta)**2 + j1(beta)**2))
    return beta, C

def eigen_sphere(Bi, n=1):
    beta = fzeros(lambda x: (1-Bi)*np.sin(x) - x*np.cos(x), n=n)
    C = 4*(np.sin(beta) - beta*np.cos(beta))/(2*beta - np.sin(2*beta))
    return beta, C

def planewall(beta, C, Fo, xi, n=1):
    u = np.sum(np.cos(beta*xi) * C * np.exp(-beta**2*Fo))    
    return u

def cylinder(beta, C, Fo, r, n=1):
    u = np.sum(j0(beta*r) * C * np.exp(-beta**2*Fo))
    return u

def sphere(beta, C, Fo, r, n=1):
    ss = 1 if r == 0 else np.sin(beta*r)/(beta*r)    
    u = np.sum(ss * C * np.exp(-beta**2*Fo))
    return u

def oneterm(Bi, Fo, xi, geom='planewall'):
    if geom == 'planewall':
        beta, C = eigen_planewall(Bi, n=1)
        return planewall(beta, C, Fo, xi)
    elif geom == 'cylinder':
        beta, C = eigen_cylinder(Bi, n=1)
        return cylinder(beta, C, Fo, xi)
    elif geom == 'sphere':
        beta, C = eigen_sphere(Bi, n=1)
        return sphere(beta, C, Fo, xi)
    else:
        print('__ name error __')
        return None
        
def term(Fo):
    if    Fo > 3e-1: n = 1
    elif  Fo > 2e-1: n = 2
    elif  Fo > 1e-1: n = 4
    elif  Fo > 5e-2: n = 6
    elif  Fo > 1e-2: n = 10
    elif  Fo > 5e-3: n = 15
    else:            n = 30    
    return n

def transient(Bi, Fo, xi, geom='planewall'):
    """
    Bi : scalar
    xi : 1-D array
    Fo : 1-D array
    """
    Fo = np.atleast_1d(Fo)
    xi = np.atleast_1d(xi)
    nFo, nxi = len(Fo), len(xi) 
    u = np.zeros((nFo, nxi))
    if   geom == 'planewall': beta, C = eigen_planewall(Bi, n=30)
    elif geom == 'cylinder':  beta, C = eigen_cylinder(Bi, n=30)
    elif geom == 'sphere':    beta, C = eigen_sphere(Bi, n=30)
        
    for i in range(nFo):
        for j in range(nxi):
            if   geom == 'planewall':  
                u[i,j] = planewall(beta, C, Fo[i], xi[j], n=term(Fo[i]))
            elif geom == 'cylinder': 
                u[i,j] = cylinder(beta, C, Fo[i], xi[j], n=term(Fo[i]))
            elif geom == 'sphere':
                u[i,j] = sphere(beta, C, Fo[i], xi[j], n=term(Fo[i]))
    return u


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    from heatlib import  secant
    
    D = 0.04
    ro = D/2
    rho, cp = 1200, 1250
    
    Ti, Tinf = 100, 25
    
    # a 
    h = 55
    t = 1136
    T0 = 40
    
    def fun(k):
        alpha = k/(rho*cp)
        Bi = h*ro/k
        Fo = alpha*t/ro**2
        u = oneterm(Bi, Fo, 0, geom='cylinder')
        T00 = Tinf + (Ti - Tinf)*u
        return T00 - T0
    
    k = secant(fun, 0.1)
    print(k)
    
    # b
    C = 55/6.8**(0.618)
    alpha = k/(rho*cp)
    Fo = alpha*t/ro**2
    V_ = np.linspace(3,20)
    for V in V_:    
        h = C*V**0.618    
        Bi = h*ro/k    
        u = oneterm(Bi, Fo, 0, geom='cylinder')
        T0 = Tinf + (Ti - Tinf)*u
        plt.plot(V, T0, '.')
    plt.title('at t = 1136 s')
    plt.xlabel('V')
    plt.ylabel('centerline temp.')
    plt.show()
    
    #%%
    t = np.geomspace(10,1500, 100)
    Fo = alpha*t/ro**2
    for V in [3, 10, 20]: 
        h = C*V**0.618    
        Bi = h*ro/k    
        T = transient(Bi, Fo, [0, 1])
        plt.semilogx(Fo, T, label=V)
        # plt.plot(Fo, T, label=V)
    plt.legend()
    plt.xlabel('Fo')
    plt.ylabel('centerline temp.')
    plt.show()