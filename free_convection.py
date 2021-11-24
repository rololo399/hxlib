# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:16:19 2021

@author: user
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

g = 9.81
 
# =============================================================================
# free convection
# =============================================================================
def prop_01(fluid, T, P=101325):
    k    = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu   = PropsSI('viscosity',    'T', T, 'P', P, fluid)
    rho  = PropsSI('D',            'T', T, 'P', P, fluid)
    cp   = PropsSI('C',            'T', T, 'P', P, fluid)
    Pr   = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    beta = PropsSI('isobaric_expansion_coefficient', 
                                   'T', T, 'P', P, fluid)
    return k, mu, rho, cp, Pr, beta

def Grashof(fluid, T1, T2, L, P=101325.0):    
    T = (T1+T2)/2
    k    = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu   = PropsSI('viscosity',    'T', T, 'P', P, fluid)
    rho  = PropsSI('D',            'T', T, 'P', P, fluid)
    Pr   = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    beta = PropsSI('isobaric_expansion_coefficient', 
                                    'T', T, 'P', P, fluid)    
    nu = mu/rho
    Gr = g*beta*np.abs(T1-T2)*L**3/nu**2
    return Gr, Pr, k

def Rayleigh(fluid, T1, T2, L, P=101325.0):    
    Gr, Pr, k = Grashof(fluid, T1, T2, L, P=P)
    Ra = Gr*Pr
    return Ra, Pr, k

def fc_vertical_plate_nd(Ra, Pr):    
    if Ra < 1e9: # Eq 9.27
        Nu = 0.68 + 0.670*Ra**(1/4)/(1+(0.492/Pr)**(9/16))**(4/9)
    else: # Eq 9.26 Churchill and Chu
        Nu = (0.825 + 0.387*Ra**(1/6)/(1 + (0.492/Pr)**(9/16))**(8/27))**2        
    return Nu

def fc_vertical_plate(fluid, Ts, Tinf, L, P=101325):
    Ra, Pr, k = Rayleigh(fluid, Ts, Tinf, L, P=P)
    Nu = fc_vertical_plate_nd(Ra, Pr)
    h = Nu*k/L
    return h, Nu, Ra

def fc_horizontal_cylinder_nd(Ra, Pr):
    Nu = (0.6 + (0.387*Ra**(1/6))/(1+(0.559/Pr)**(9/16))**(8/27))**2
    return Nu
  
def fc_horizontal_cylinder(fluid, Ts, Tinf, D, P=101325):
    """ 
    External Free Convection, long horizontal cylinder
    Heat Transfer, Incropera, chap. 9, Eq. 9.34 Churchill and Chu
    """
    Ra, Pr, k = Rayleigh(fluid, Ts, Tinf, D, P=P)    
    Nu = fc_horizontal_cylinder_nd(Ra, Pr)
    h = Nu*k/D
    return h, Nu, Ra

def fc_sphere_nd(Ra, Pr):
    Nu = 2 + (0.589*Ra**(1/4))/(1+(0.469/Pr)**(9/16))**(4/9)
    return Nu
  
def fc_sphere(fluid, Ts, Tinf, D, P=101325):
    """ 
    External Free Convection, long horizontal cylinder
    Heat Transfer, Incropera, chap. 9, Eq. 9.34 Churchill and Chu
    """
    Ra, Pr, k = Rayleigh(fluid, Ts, Tinf, D, P=P)
    Nu = fc_sphere_nd(Ra, Pr)
    h = Nu*k/D
    return h, Nu, Ra

def fc_plate_horizontal1_nd(Ra, Pr):
    if 1e4 < Ra < 1e7 and Pr >= 0.7:
        Nu = 0.54*Ra**(1/4)
    elif 1e7 <= Ra < 1e11:
        Nu = 0.15*Ra**(1/3)
    else:
        print('error : fc_plate_horizontal1_nd')
        return None
    return Nu

def fc_plate_horizontal1(fluid, Ts, Tinf, L, P=101325):
    """ 
    External Free Convection, 
    upper surface of hot plate
    lower surface of cold plate
    Heat Transfer, Incropera, chap. 9, Eq. 9.30
    L = Ac/P
    """
    Ra, Pr, k = Rayleigh(fluid, Ts, Tinf, L, P=P)
    Nu = fc_plate_horizontal1_nd(Ra, Pr)
    h = Nu*k/L
    return h, Nu, Ra

def fc_plate_horizontal2_nd(Ra, Pr):
    # Eq. 9.32  1e4 < Ra < 1e9, Pr > 0.7
    Nu = 0.52*Ra**(1/5)
    return Nu

def fc_plate_horizontal2(fluid, Ts, Tinf, L, P=101325):
    """ 
    External Free Convection, 
    lower surface of hot plate
    upper surface of cold plate
    Heat Transfer, Incropera, chap. 9, Eq. 9.32
    L = Ac/P
    """
    Ra, Pr, k = Rayleigh(fluid, Ts, Tinf, L, P=P)
    Nu = fc_plate_horizontal2_nd(Ra, Pr)
    h = Nu*k/L
    return h, Nu, Ra
    
def fc_horizontal_enclosure(fluid, T1, T2, L, P=101325):    
    """
    Eq. 9.49 Globe and Dropkin 
    fc_horizontal_cavity
    """
    Ra, Pr, k = Rayleigh(fluid, T1, T2, L, P=P)
    Nu = 0.069*Ra**(1/3)*Pr**(0.074)
    h = Nu*k/L
    return h, Nu, Ra

def fc_vertical_cavity(fluid, T1, T2, L, H, P=101325):
    """Eq. 9.53  """
    Ra, Pr, k = Rayleigh(fluid, T1, T2, L, P=P)
    Nu = 0.42*Ra**(1/4)*Pr**(0.012)*(H/L)**(-0.3)
    h = Nu*k/L
    return h, Nu, Ra


    
def fc_concentric_cylinder(fluid, Ti, To, ri, ro, L=1, P=101325):
    """Eq. 9.58 """    
    Lc = 2*(np.log(ro/ri))**(4/3)/(ri**(-3/5) + ro**(-3/5))**(5/3)
    Ra, Pr, k = Rayleigh(fluid, Ti, To, Lc, P=P)
    k_eff = k*0.386*(Pr/(0.861 + Pr))**(1/4)*Ra**(1/4)
    if k_eff < k:
        k_eff = k        
    q = 2*np.pi*L*k_eff*(Ti-To)/np.log(ro/ri)
    return q, k_eff, Ra

def fc_concentric_sphere(fluid, Ti, To, ri, ro, P=101325):
    """ 9.8.3 concentric sphere """
    Lc = (1/ri-1/ro)**(4/3)/(2**(1/3)*(ri**(-7/5)+ro**(-7/5))**(5/3))
    Ra, Pr, k = Rayleigh(fluid, Ti, To, Lc, P=P)
    k_eff = k*0.74*(Pr/(0.861 + Pr))**(1/4)*Ra**(1/4)
    if k_eff < k:
        k_eff = k        
    q = 4*np.pi*k_eff*(Ti-To)/(1/ri-1/ro)
    return q, k_eff, Ra
    
def fc_tilted_cavity_nd(Ra, tau):
    """
    Eq. 9.54 for H/L > 12, 0 < tau < tau_star
    """
    a1 = 1 - 1708/(Ra*np.cos(tau))
    if a1 < 0: a1 = 0
    a2 = 1 - 1708*(np.sin(1.8*tau))**(1.6)/(Ra*np.cos(tau))
    a3 = (Ra*np.cos(tau)/5830)**(1/3) - 1
    if a3 < 0: a3 = 0
    Nu = 1 + 1.44*a1*a2 + a3
    return Nu

def fc_tilted_cavity(fluid, T1, T2, L, tau, P=101325):
    Ra, Pr, k = Rayleigh(fluid, T1, T2, L, P=P)
    Nu = fc_tilted_cavity_nd(Ra, tau)
    h = Nu*k/L
    return h, Nu, Ra