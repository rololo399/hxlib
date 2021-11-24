"""
Incropera's Principles of Heat and Mass Transfer, Global Edition
Chapter 5 Transient Conduction

Created on Mon Apr 19 15:17:36 2021
"""

import numpy as np
from scipy.special import j0, j1, erf, erfc
from hxlib.heatlib import fzeros

def eigenvalue(Bi, n=1, geom='planewall'):
    if   geom == 'planewall':
        beta = fzeros(lambda x: x*np.sin(x) - Bi*np.cos(x), n=n)
        C = 4*np.sin(beta)/(2*beta + np.sin(2*beta))    
    elif geom == 'cylinder':
        beta = fzeros(lambda x: x*j1(x) - Bi*j0(x), n=n)
        C = 2*j1(beta)/(beta*(j0(beta)**2 + j1(beta)**2))        
    elif geom == 'sphere':
        beta = fzeros(lambda x: (1-Bi)*np.sin(x) - x*np.cos(x), n=n)
        C = 4*(np.sin(beta) - beta*np.cos(beta))/(2*beta - np.sin(2*beta))
    else: 
        return None 
    if n == 1:
        return beta[0], C[0]
    else:
        return beta, C

def temperature(beta, C, Fo, xi, geom='planewall'):
    if   geom == 'planewall':
        u = np.sum(np.cos(beta*xi) * C * np.exp(-beta**2*Fo))
    elif geom == 'cylinder':
        u = np.sum(j0(beta*xi) * C * np.exp(-beta**2*Fo))
    elif geom == 'sphere':
        ss = 1 if xi <= 1e-5 else np.sin(beta*xi)/(beta*xi)    
        u = np.sum(ss * C * np.exp(-beta**2*Fo))        
    else: 
        return None 
    return u

def oneterm(Bi, Fo, xi, geom='planewall'):
    beta, C = eigenvalue(Bi, n=1, geom=geom)
    u = temperature(beta, C, Fo, xi, geom=geom)
    return u

def oneterm_qq0(Bi, Fo, geom='planewall'):
    beta, C = eigenvalue(Bi, n=1, geom=geom)
    u0 = C*np.exp(-beta**2*Fo)
    if   geom == 'planewall':        
        qq0 = 1 - np.sin(beta)/beta * u0
    elif geom == 'cylinder':
        qq0 = 1 - j1(beta)/beta * 2*u0
    elif geom == 'sphere':
        qq0 = 1 - 3*u0/beta**3 *(np.sin(beta) - beta*np.cos(beta))
    else:
        return None
    return qq0    
    
def term(Fo):
    if    Fo > 5e-1: n = 1
    elif  Fo > 4e-1: n = 2
    elif  Fo > 3e-1: n = 4
    elif  Fo > 2e-2: n = 8
    elif  Fo > 1e-2: n = 16
    else           : n = 32    
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
    beta, C = eigenvalue(Bi, n=30, geom=geom)
        
    for i in range(nFo):
        n = term(Fo[i])
        for j in range(nxi):
            u[i,j] = temperature(beta[:n], C[:n], Fo[i], xi[j], geom=geom)
    return u

def semiinf_Ts(x, t, alpha, k, Ti, Ts):
    # eq 5-60, 5-61
    eta = x/(2*np.sqrt(alpha*t))
    T = Ts + (Ti - Ts) * erf(eta)
    # qs = k*(Ts - Ti)/np.sqrt(np.pi*alpha*t)
    # Q = 2*k*(Ts-Ti)*np.sqrt(t/(np.pi*alpha))
    return T
def semiinf_qs(x, t, alpha, k, Ti, qs):
    T = ( Ti + 2*qs*np.sqrt(alpha*t/np.pi)/k*np.exp(-x**2/(4*alpha*t)) - 
          qs*x/k*erfc(x/2/np.sqrt(alpha*t)) )
    return T
def semiinf_convection(x, t, alpha, k, Ti, Tinf, h):
    at = alpha*t
    at2  = np.sqrt(at)
    d1 = erfc(x/2/at2)
    d2 = np.exp(h*x/k + h**2*at/k**2)
    d3 = erfc(x/2/at2 + h*at2/k)
    T = Ti + (Tinf - Ti)*(d1 - d2*d3)
    return T

def qstar_Ts(Fo, type=1):
    # qstar = qs'' * Lc / k (Ts - Ti)
    # table 5.2.a
    if type == 0: # semi-infinite
        return 1/np.sqrt(np.pi*Fo)
    elif type == 1: # plane wall, interior
        if Fo < 0.2:
            return 1/np.sqrt(np.pi*Fo)
        else:
            return 2*np.exp(-(np.pi/2)**2*Fo)
    elif type == 2: # cylinder, interior
        if Fo < 0.2:
            return 1/np.sqrt(np.pi*Fo) - 0.5 - 0.65*Fo
        else:
            beta = 2.4050
            return 2*np.exp(-beta**2*Fo)
    elif type == 3: # sphere, interior
        if Fo < 0.2:
            return 1/np.sqrt(np.pi*Fo) - 1.0
        else:
            beta = np.pi
            return 2*np.exp(-beta**2*Fo)
    elif type == 4: # sphere, exterior
        return 1/np.sqrt(np.pi*Fo) + 1.0
        

def qstar_qs(Fo, type=1):
    # table 5.2.b
    if type == 0: # semi-infinite 
        return 1/2*np.sqrt(np.pi/Fo)
    elif type == 1: # plane wall
        if Fo < 0.2:
            return 1/2*np.sqrt(np.pi/Fo)
        else:
            return 1/(Fo + 1/3)
    elif type == 2: # cylinder
        if Fo < 0.2:
            return 1/2*np.sqrt(np.pi/Fo) - np.pi/8
        else:
            return 1/(2*Fo + 1/4)
    elif type == 3: # sphere
        if Fo < 0.2:
            return 1/2*np.sqrt(np.pi/Fo) - np.pi/4
        else:
            return 1/(3*Fo + 1/5)
    elif type == 4: # sphere, exterior
        return 1/(1 - np.exp(Fo)*erfc(np.sqrt(Fo)))

def eq_0572(x, t, omega, alpha):
    # periodic heating, p. 301
    # (T(x,t) - Ti) / dT
    xx = x*np.sqrt(omega/(2*alpha))
    return np.exp(-xx)*np.sin(omega*t - xx)

def eq_0573(t, omega, alpha, k, dT):
    # heat flux at the surface, p. 302
    return k*dT*np.sqrt(omega/alpha)*np.sin(omega*t - np.pi/4)
    
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