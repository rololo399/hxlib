"""
Table 4.1 
Conduction shape factors and dimensionless conduction heat rates
"""

import numpy as np

def Case_01(D, z):
    """ isothermal sphere buried in a semi-infinite medium """    
    return 2*np.pi*D/(1 - D/(4*z))

def Case_02(D, z, L=1):
    """ horizontal isothermal cylinder of length L buried 
        in a semi-infinite medium """
    if z > 3*D/2:
        S = 2*np.pi*L/np.log(4*z/D)
    else:
        S = 2*np.pi*L/np.arccosh(2*z/D)
    return S

def Case_03(L, D):
    """ vertical cylinder in a semi-infinite medium """
    return 2*np.pi*L/np.log(4*L/D)

def Case_06(D, w, L=1):
    """ circular cylinder in a square soild """
    return 2*np.pi*L/np.log(1.08*w/D)

def Case_07(D, d, z, L=1):
    """ eccentric circular cylinder in a large cylinder """
    return 2*np.pi/np.arccosh((D**2 + d**2 - 4*z**2)/(2*D*d))

def Case_11(W, w, L=1):
    """ square channel """
    if W/w < 1.4:
        S = 2*np.pi*L/(0.785*np.log(W/w))
    else:
        S = 2*np.pi*L/(0.930*np.log(W/w) - 0.050)
    return S