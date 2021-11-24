# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:23:56 2021

@author: user
"""
import numpy as np


def moody(Re, RR=0, tol=1e-6):
    """
    This function returns the Darcy friction factor (f) for internal flow
    given inputs of Reynolds number (Re) and the Relative Roughness (RR).

    The head loss and pressure drop can then be found from:
        head loss  = DELTAp/(rho*g) = f * (L/D )* (V^2/(2*g))
    """

    if Re < 2300.0:
        f = 64/Re
        return f
    a, b = 2.51/Re, RR/3.7
    y = 7.0
    while 1:
        x = -2.0*np.log10(b + a*y)
        y = -2.0*np.log10(b + a*x)
        if np.abs((x-y)/y) < tol:
            f = 1/y**2
            return f


def headloss(V, D, L, rho, mu, epsilon, g=9.81):
    Re = rho*V*D/mu
    RR = epsilon/D
    f = moody(Re, RR)
    h = f*L/D*V**2/(2*g)
    return h


if __name__ == "__main__":

    Re = 1e5
    RR = 1e-5
    f = moody(Re, RR)
    print(f)

