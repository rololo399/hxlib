"""
Table 3.4 Temperature distribution and heat rates 
for fins of uniform cross section
"""

import numpy as np

def fun_m(h,P,k,A):
    return np.sqrt((h*P)/(k*A))

def fun_M(h,P,k,A,theta_b):
    return np.sqrt(h*P*k*A)*theta_b

def Case_A_temperature(x, m, L, k, h):    
    # convection
    d1 = np.cosh(m*(L-x)) + (h/(m*k))*np.sinh(m*(L-x))
    d2 = np.cosh(m*L) + (h/(m*k))*np.sinh(m*L)
    return d1/d2

def Case_A_heatrate(M, m, L, k, h):    
    # convection
    d1 = np.sinh(m*L) + (h/(m*k))*np.cosh(m*L)
    d2 = np.cosh(m*L) + (h/(m*k))*np.sinh(m*L)
    return M*d1/d2

def Case_B_temperature(x, m, L):    
    # adiabatic
    return np.cosh(m*(L-x))/np.cosh(m*L)

def Case_B_heatrate(M, m, L):    
    # adiabatic
    return M*np.tanh(m*L)

def Case_C_temperature(x, m, L, theta_L, theta_b):    
    # prescribed temperature
    d1 = (theta_L/theta_b)*np.sinh(m*x) + np.sinh(m*(L-x))    
    return d1/np.sinh(m*L)

def Case_C_heatrate(M, m, L, theta_L, theta_b):    
    # prescribed temperature
    q0 = M/np.sinh(m*L) * (np.cosh(m*L) - (theta_L/theta_b))
    qL = M/np.sinh(m*L) * (1 - np.cosh(m*L)*(theta_L/theta_b))
    return q0, qL

def Case_D_temperature(x, m):
    # infinite fin
    return np.exp(-m*x)

def Case_D_heatrate(M):
    # infinite fin
    return M
