# -*- coding: utf-8 -*-
"""
Dimensionless Group
"""
import numpy as np

g = 9.81

from CoolProp.CoolProp import PropsSI

def Reynolds(fluid, T, V, L, P=101325):
    mu = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    Re = rho*V*L/mu
    return Re

def Prandtl(fluid, T, P):
    Pr = PropsSI('Prandtl', 'T', T, 'P', P, fluid)
    return Pr

def Grashof(fluid, T1, T2, L, P=101325.0):    
    T = (T1 + T2)/2
    mu   = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho  = PropsSI('D', 'T', T, 'P', P, fluid)
    beta = PropsSI('isobaric_expansion_coefficient',
                   'T', T, 'P', P, fluid)    
    nu = mu/rho
    Gr = g*beta*np.abs(T1-T2)*L**3/nu**2
    return Gr

def Rayleigh(fluid, T1, T2, L, P):    
    Gr = Grashof(fluid, T1, T2, L, P)
    T = (T1 + T2)/2
    Pr = Prandtl(fluid, T, P=P)
    Ra = Gr*Pr
    return Ra

def Peclet(fluid, T, V, L, P=101325.0):
    Re = Reynolds(fluid, T, V, L, P=P)
    Pr = Prandtl(fluid, T, P=P)
    Pe = Re*Pr
    return Pe


# from moody import moody
# def friction_factor(fluid, T, V, D, epsilon=0, P=101325):
#     Re = Reynolds(fluid, T, V, D, P=P)
#     RR = epsilon/D
#     f = moody(Re, RR)
#     return f


# if __name__ == "__main__":
    
    # f = friction_factor('water', 25+273, 1, 0.3, 0.01e-3)
    # print(f)
    
