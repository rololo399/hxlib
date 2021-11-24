# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:17:27 2021

@author: user
"""
import numpy as np
from CoolProp.CoolProp import PropsSI

g = 9.81

# =============================================================================
# external flow
# =============================================================================
def prop_02(fluid, T, P=101325):
    k   = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu  = PropsSI('viscosity',    'T', T, 'P', P, fluid)
    rho = PropsSI('D',            'T', T, 'P', P, fluid)
    Pr  = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    return k, mu, rho, Pr

def Reynolds(fluid, T, V, L, P=101325):
    k   = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu  = PropsSI('viscosity',    'T', T, 'P', P, fluid)
    rho = PropsSI('D',            'T', T, 'P', P, fluid)
    Pr  = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    Re = rho*V*L/mu
    return Re, Pr, k

########## plate    
def external_flow_plate_nd(Re, Pr, Re_c = 5e5):
    """ 
    Eq. 7.33
    Nusselt number for a flow over a flat plate of finite length.
    """    
    if Re < Re_c:
        Nu_x = 0.3387*Re**(1/2)*Pr**(1/3)/((1+(0.0468/Pr)**(2/3))**(1/4))
        # Churchill and Ozoe, valid Pe = Re*Pr > 100 
        Nu = 2*Nu_x
    else:
        A = 0.037*Re_c**(4/5) - 0.664*Re_c**(1/2)
        Nu = (0.037*Re**(4/5) - A)*Pr**(1/3)  # Eq. 7.38
    return Nu

def external_flow_plate(fluid, Ts, Tinf, uinf, L, Re_c=5e5, P=101325):
    """
    Nusselt number for a flow over a flat plate of finite length.
    The properties are evaluated at the film temperature.
    
    Parameters
    ----------
    fluid : string
        DESCRIPTION.
    Ts : TYPE
        DESCRIPTION.
    Tinf : TYPE
        DESCRIPTION.
    uinf : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    Re_c : TYPE, optional
        DESCRIPTION. The default is 5e5.
    P : TYPE, optional
        DESCRIPTION. The default is 101325.

    Returns
    -------
    h : TYPE
        DESCRIPTION.
    Nu : TYPE
        DESCRIPTION.
    Re : TYPE
        DESCRIPTION.

    """
        
    Tf = (Ts+Tinf)/2 # film temperature
    Re, Pr, k = Reynolds(fluid, Tf, uinf, L, P=P)    
    Nu = external_flow_plate_nd(Re, Pr, Re_c=Re_c)
    h = Nu*k/L
    return h, Nu, Re        

########## cylinder    
def external_flow_cylinder_nd(Re, Pr):
    """ Eq. 7.54 Churchill and Bernstein 
    for external flow past an infinite cylinder.  """
    d1 = 0.62*Re**(1/2)*Pr**(1/3)
    d2 = (1 + (0.4/Pr)**2/3)**(1/4)
    d3 = (1 + (Re/282000)**(5/8))**(4/5)
    Nu = 0.3 + d1/d2*d3
    return Nu

def external_flow_cylinder(fluid, Ts, Tinf, uinf, D, P=101325):
    """
    Eq. 7.54 Churchill and Bernstein 
    average heat transfer coefficient
    for external flow past an infinite cylinder.  
    The properties are evaluated at the film temperature.
    """
    Tf = (Ts+Tinf)/2 # film temperature
    Re, Pr, k = Reynolds(fluid, Tf, uinf, D, P=P)    
    Nu = external_flow_cylinder_nd(Re, Pr)
    h = Nu*k/D
    return h, Nu, Re

def Hilpert_nd(Re, Pr, C, m, n=1/3):
    """ for external flow past an infinite cylinder.  """
    Nu = C*Re**m*Pr**n
    return Nu

def Zukauskas_nd(Re, Pr, Prs):
    """ Zukauskas eq. 7.53 for external flow past an infinite cylinder.  """
    if Pr < 10:
        n = 0.37
    else:
        n = 0.36
        
    if 0.4 < Re < 4:      C, m = 0.989, 0.330
    elif     Re < 4e1:    C, m = 0.911, 0.385
    elif     Re < 4e3:    C, m = 0.683, 0.466
    elif     Re < 4e4:    C, m = 0.193, 0.618
    elif     Re < 4e5:    C, m = 0.027, 0.805
    else: return None
    
    Nu = C*Re**m*Pr**n*(Pr/Prs)**(1/4)
    return Nu

def Zukauskas(fluid, Ts, Tinf, V, D, P=101325):
    """for external flow past an infinite cylinder.  """
    Re, Pr, k = Reynolds(fluid, Tinf, V, D, P=P)
    Prs = PropsSI('Prandtl', 'T', Ts, 'P', P, fluid)
    Nu = Zukauskas_nd(Re, Pr, Prs)
    h = Nu*k/D
    return h, Nu, Re

########## sphere

def external_flow_sphere_nd(Re, Pr, mu_ratio=1):
    """ Whitaker Eq. 7.56 """
    Nu = 2 + (0.4*Re**(1/2) + 0.06*Re**(2/3))*Pr**(0.4)*mu_ratio**(1/4)
    return Nu

def external_flow_sphere(fluid, Ts, Tinf, uinf, D, P=101325):
    """
    average heat transfer coefficient
    for external flow past a sphere
    The properties are evaluated at Tinf
    """
    Re, Pr, k = Reynolds(fluid, Tinf, uinf, D, P=P)    
    Nu = external_flow_sphere_nd(Re, Pr)
    mu = PropsSI('viscosity', 'T', Tinf, 'P', P, fluid)
    mus = PropsSI('viscosity', 'T', Ts, 'P', P, fluid)
    Nu = external_flow_sphere_nd(Re, Pr, mu_ratio=mu/mus)
    h = Nu*k/D
    return h, Nu, Re

def Ranz_Marshall_nd(Re, Pr):
    Nu = 2 + 0.6*Re**(1/2)*Pr**(1/3)
    return Nu

def External_Flow_Inline_Bank(
        fluid, T, T_in, T_out, T_s,  P, u_inf, N_L, D, S_T, S_L):
    
    N_LL = np.array([1, 2, 3, 4, 5, 7, 10, 13, 16, 20]) 
    C22 = np.array([0.7, 0.8, 0.86, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1])
    f_C2 = interpd(N_LL, C22)
    
    Tm =(T_in + T_out)/2
    Vmax = S_T/(S_T - D)*u_inf
    Re, Pr, k = Reynolds(fluid, Tm, Vmax, D)
    Prs = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    Nu = External_Flow_Inline_Bank_nd(Re, Pr, Prs)
    
    if N_L < 20:
        Nu = f_C2(N_L)*Nu
    
    h = Nu*k/D    
    return h, Nu, Re


def External_Flow_Inline_Bank_nd(Re, Pr, Prs):
    """
    N_L > 20, 0.7 < Pr < 500, 10 < Re < 2e6    
    """
    if 10 < Re < 1e2 : 
        C1, m = 0.80, 0.40
    elif 1e2 <=  Re <= 1e3 : 
        Nu = Zukauskas_nd(Re, Pr, Prs) 
        return Nu
    elif  Re < 2e5 : 
        C1, m = 0.27, 0.63
    elif  Re < 2e6 : 
        C1, m = 0.021, 0.84
    else:
        return None    
    Nu = C1*Re**m*Pr**0.36*(Pr/Prs)**(1/4)
    return Nu
