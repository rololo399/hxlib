"""
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.constants import sigma
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

pi = np.pi

def aligned_parallel_rectangles(X, Y, L):
    """ table 13-2 View factors for three-dimensional geometries """
    x, y = X/L, Y/L
    x1, y1 = 1 + x**2, 1 + y**2
    d1 = np.log(np.sqrt((x1*y1)/(x1 + y1 - 1)))
    d2 = x*np.sqrt(y1)*np.arctan(x/np.sqrt(y1))
    d3 = y*np.sqrt(x1)*np.arctan(y/np.sqrt(x1))
    d4 = x*np.arctan(x) + y*np.arctan(y)
    F = 2/(np.pi*x*y)*(d1 + d2 + d3 - d4)
    return F

def coaxial_parallel_disks(r1, r2, L):
    """ table 13-2 View factors for three-dimensional geometries """
    R1, R2 = r1/L, r2/L
    S = 1 + (1+R2**2)/R1**2
    F12 = 1/2*(S - np.sqrt(S**2 - 4*(R2/R1)**2))    
    return F12

def coaxial_parallel_ring_shape_disks(D1, D1in, D2, D2in, L):
    """
    two parallel coaxial ring shaped disks 
    see problem 13.5
    """
    D3 = D1in
    D4 = D2in
    
    A3 = pi*D3**2/4
    A1 = pi*D1**2/4 - A3
    A4 = pi*D4**2/4
    
    F_13_24 = coaxial_parallel_disks(D1/2, D2/2, L)
    F_3_24  = coaxial_parallel_disks(D3/2, D2/2, L)
    F_4_13  = coaxial_parallel_disks(D4/2, D1/2, L)
    F_4_3   = coaxial_parallel_disks(D4/2, D3/2, L)
    
    F12 = 1/A1*((A1+A3)*F_13_24 - A3*F_3_24 - A4*(F_4_13 - F_4_3))
    
    return F12

def crossed_string(w, c1, c2, nc1, nc2):
    """
    w : length of 1
    ac, bd : length of crossed string
    ad, bc :  length of non-crossed string
    """
    return ((c1 + c2) - (nc1 + nc2))/(2*w)    

def cylinder_and_parallel_rectangle(r, L, s1, s2):
    """ 
    Table 13.1
    cylinder (j) and parallel rectangle (i)    
    """
    Fij = r/(s1-s2)*(np.arctan(s1/L) - np.arctan(s2/L))
    return Fij


def inclined_parallel_plates_of_equal_width_and_a_common_edge(alpha):
    """ table 13.1 View factors for two-dimensional geometries """
    return 1 - np.sin(alpha/2)

def infinite_plane_and_row_of_cylinders(D, s):
    """ Table 13.1 """
    d1 = np.sqrt((s**2 - D**2)/D**2)
    d2 = (D/s)*np.arctan(d1)
    d3 = 1 - np.sqrt(1 - (D/s)**2)
    Fij = d3 + d2
    return Fij

def parallel_cylinder_of_different_radii(ri, rj, s):
    """ table 13.1 View factors for two-dimensional geometries """
    R = rj/ri
    S = s/ri
    C = 1 + R + S
    a1 = np.sqrt(C**2 - (R+1)**2)
    a2 = np.sqrt(C**2 - (R-1)**2)
    a3 = (R-1)*np.arccos(R/C - 1/C)
    a4 = (R+1)*np.arccos(R/C + 1/C)
    Fij = 1/(2*np.pi)*(np.pi + a1 - a2 + a3 - a4)
    return Fij

def parallel_plates_with_midlines_connected_by_perpendicular(wi, wj, L):
    """ table 13.1 View factors for two-dimensional geometries """
    Wi, Wj = wi/L, wj/L
    a1 = np.sqrt((Wi + Wj)**2 + 4)
    a2 = np.sqrt((Wj - Wi)**2 + 4)
    Fij = (a1 - a2)/(2*Wi)    
    return Fij

def parallel_paltes(x1,x2,x3, y1, y2, y3, c):   
    """ Modest Eq. 4.41 Fig. 4.15 b"""
    def f(x, y):
        return aligned_parallel_rectangles(x, y, c)    
    A1 = x1*y1
    d1 = f(x3, y3) - f(x3, y2) - f(x3, y3-y1) + f(x3, y2-y1)
    d2 = f(x2, y3) - f(x2, y2) - f(x2, y3-y1) + f(x2, y2-y1) 
    d3 = f(x3-x1, y3) - f(x3-x1, y2) - f(x3-x1, y3-y1) + f(x3-x1, y2-y1)
    d4 = f(x2-x1, y3) - f(x2-x1, y2) - f(x2-x1, y3-y1) + f(x2-x1, y2-y1)
    F12 = (d1 - d2 - d3 + d4)/(4*A1)
    return F12

def perpendicular_rectangles_with_common_edge(X,Y,Z):
    """ table 13-2 View factors for three-dimensional geometries """
    h = Z/X
    w = Y/X
    w2 = w*w
    h2 = h*h
    d1 = w*np.arctan(1/w)
    d2 = h*np.arctan(1/h)
    d31 = np.sqrt(h2 + w2)
    d32 = np.arctan(1/d31)
    d3 = d31*d32    
    d41 = (1+w2)*(1+h2)/(1+w2+h2)
    d42 = (w2*(1+w2+h2)/(1+w2)/(w2+h2))**w2
    d43 = (h2*(1+w2+h2)/(1+h2)/(w2+h2))**h2
    d4 = 1/4*np.log(d41*d42*d43)
    
    F12 = 1/(np.pi*w)*(d1 + d2 - d3 + d4)
    return F12

def perpendicular_plates(x1,x2, y1,y2, z1,z2,z3):
    """ Modest Eq. 4.40 Fig. 4.15 a """
    def f(w,h,l):
        return perpendicular_rectangles_with_common_edge(l,h,w)
    A1 = (x2-x1)*z3
    d1 = f(x2,y2,z3) - f(x2,y1,z3) - f(x1,y2,z3) + f(x1,y1,z3)
    d2 = f(x1,y2,z2) - f(x1,y1,z2) - f(x2,y2,z2) + f(x2,y1,z2)
    d3 = -f(x2,y2,z3-z1) + f(x2,y1,z3-z1) + f(x1,y2,z3-z1) - f(x1,y1,z3-z1)
    d4 = f(x2,y2,z2-z1) - f(x2,y1,z2-z1) - f(x1,y2,z2-z1) + f(x1,y1,z2-z1)
    F12 = (d1 + d2 + d3 + d4)/(2*A1)
    return F12

def sphere_to_coaxial_disk(r, a):
    """ Modest, Appendix D,  # 48, p. 792 """
    R = r/a
    F12 = 0.5*(1 - 1/np.sqrt(1 + R**2))
    return F12



def viewfactor_completion(A, F):
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]    
    # reciprocity relation
    for i in range(n):
        if np.sum(F[i]) != 1.0:
            for j in range(n):
                if F[i,j] == 0.0:
                    F[i,j] = A[j]*F[j,i]/A[i]
    # summation rule                    
    for i in range(n):
        if np.sum(F[i]) != 1.0:
            F[i,i] = 1.0
            for j in range(n):
                if i != j:
                    F[i,i] -= F[i,j]
    return F
