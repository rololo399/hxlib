"""
Table C.3
One-Dimensional, Steady-State solutions to the heat equation for 
Uniform Generation in a Plane Wall with One Adiabatic Surface, 
a Solid Cylinder, and a Solid Sphere 
"""

# import numpy as np

def plane_temp(x, L, qdot, k, Ts): # C.22
    return qdot*L**2/(2*k)*(1-(x/L)**2) + Ts

def plane_heatflux(qdot, x):
    return qdot*x

def circular_temp(r, ro, qdot, k, Ts): # C.23
    return qdot*ro**2/(4*k)*(1-(r/ro)**2) + Ts

def circular_heatflux(qdot, r):
    return qdot*r/2

def sphere_temp(r, ro, qdot, k, Ts): # C.24
    return qdot*ro**2/(6*k)*(1-(r/ro)**2) + Ts

def sphere_heatflux(qdot, r):
    return qdot*r/3


