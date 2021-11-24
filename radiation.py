# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:59:49 2021

@author: user
"""
import numpy as np
from hxlib.heatlib import bisect
from scipy.constants import pi, sigma
from scipy.integrate import quad

# radiation
# pi = np.pi
C1 = 3.741771852192757e8
C2 = 1.438776877503933e4
Wien = 2897.771955


def deg2rad(d):
    return d/180*np.pi  # degree to radian


def rad2deg(r):
    return r*180/np.pi  # degree to radian


def qij(I1, r, thetai, thetaj, Ai, Aj):
    thetai, thetaj = thetai/180*np.pi, thetaj/180*np.pi
    omega = Aj*np.cos(thetaj)/r**2
    q = I1*Ai*np.cos(thetai)*omega
    return q, omega


def planck(x, T):  # Eq. 12.30
    # x : lambda
    if x*T < 200:
        return 0
    E = C1/(x**5*np.exp(C2/(x*T))-1)
    return E

# def band(b): # Eq. 12.34
#     # b = lambda * T
#     if b < 200: return 0.0
#     if b > 1e5: return 1.0
#     fun = lambda x: C1/(sigma*x**5)/(np.exp(C2/x)-1)
#     F = quad(fun, 200, b)[0]
#     return F


def band(xT):
    y = C2/(xT)
    f = 0.0
    for i in range(1, 11):
        df = np.exp(-i*y)*(y**3 + 3.0*y**2/i + 6.0*y/i**2 + 6.0/i**3)/i
        f += df
    F = 15.0*f/np.pi**4
    return F


def blackbody(T, x1, x2=None):
    """
    Blackbody returns the fraction of the blackbody emissive power
    that is emitted between wavelengths x1 and x2
    """
    # C2 = 14387.768775039336

    def fun(x):
        y = C2/(x*T)
        f = 0.0
        for i in range(1, 11):
            df = np.exp(-i*y)*(y**3 + 3.0*y**2/i + 6.0*y/i**2 + 6.0/i**3)/i
            f += df
        F = 15.0*f/np.pi**4
        return F

    if x1 < 1e-4:
        x1 = 1e-4
    if x2 == None:
        return fun(x1)
    if x2 < 1e-4:
        x2 = 1e-4
    return np.abs(fun(x1) - fun(x2))


def invband(F):
    b = bisect(lambda b: band(b) - F, 200, 1e5)
    return b


def intensity(b):
    # intensity / (sigma T^5)
    # b = lambda * T
    I2 = C1/sigma/b**5/(np.exp(C2/b) - 1)/pi
    b1 = 2898
    I1 = C1/sigma/b1**5/(np.exp(C2/b1) - 1)/pi
    return I2, I2/I1
