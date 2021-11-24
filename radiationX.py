# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:00:45 2021

@author: user
"""
import numpy as np
from scipy.constants import sigma


def net_radiation_blackbody(T, F, A):
    """
    13.2 Blackbody Radiation Exchange, Eq. 13.14

    IN :
        T : surface temperature, K
        F : view factor
        A : area, m2
    OUT :
        q : net radiative exchange, W
    """
    A = np.asfarray(A)
    F = np.asfarray(F)
    T = np.asfarray(T)

    n = len(T)
    Eb = sigma*T**4
    q = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            AF = A[i]*F[i, j] if F[i, j] != 0 else A[j]*F[j, i]
            q[i] += AF*(Eb[i] - Eb[j])
    return q


def blackbodyX(A1, F, T):
    """ Eq 13.14 """
    F = np.asfarray(F)
    T = np.asfarray(T)
    n = len(T)
    Eb = sigma*T**4
    q = 0
    for i in range(1, n):
        q += A1*F[i]*(Eb[0] - Eb[i])
    return q


def net_radiation(A, F, T, e):
    """
    13.3.2 Radiation Exchange Between Surfaces
    Eq. 13.21, 13.22

    input
        A : area
        F : view factor
        Eb : sigma*T**4
        e : emissivity
    output
        J : radiosity
        q : net rate of heat transfer
    """

    A = np.asfarray(A)
    F = np.asfarray(F)
    T = np.asfarray(T)
    e = np.asfarray(e)

    n = len(T)
    B = np.zeros((n, n))
    b = np.zeros(n)

    Eb = sigma*T**4

    for i in range(n):
        if e[i] == 1:
            B[i, i] = 1
            b[i] = Eb[i]
            continue

        R = (1 - e[i])/(e[i]*A[i])
        B[i, i] = 1/R
        b[i] = Eb[i]/R

        for j in range(n):
            if j == i:
                continue
            AF = A[i]*F[i, j] if F[i, j] != 0 else A[j]*F[j, i]
            B[i, i] += AF
            B[i, j] = -AF

    J = np.linalg.solve(B, b)

    q = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            AF = A[i]*F[i, j] if F[i, j] != 0 else A[j]*F[j, i]
            q[i] += AF*(J[i] - J[j])

    return q, J, B, b


def net_radiation_qT(qT, A, F, e):
    """
    v : 'T', T is given
        'q', q is given
    """

    def AF(i, j):
        return A[i]*F[i, j] if F[i, j] != 0 else A[j]*F[j, i]

    n = len(qT)
    B = np.zeros((n, n))
    b = np.zeros(n)
    T = np.zeros(n, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    v = []
    for i in range(n):
        v1, v2 = qT[i]
        v.append(str(v1).capitalize()[0])
        if v[i] == 'T':
            T[i] = v2
        else:
            q[i] = v2

    Eb = sigma*T**4

    for i in range(n):
        R = (1 - e[i])/(e[i]*A[i])
        if v[i] == 'T':  # T is given
            if R == 0:
                B[i, i], b[i] = 1, Eb[i]
                continue
            else:
                B[i, i], b[i] = 1/R, Eb[i]/R
        else:  # q is given
            b[i] = q[i]

        for j in range(n):
            if j != i:
                B[i, i] += AF(i, j)
                B[i, j] = -AF(i, j)

    J = np.linalg.solve(B, b)

    for i in range(n):
        R = (1 - e[i])/(e[i]*A[i])

        if v[i] == 'T':  # T is given
            if R == 0:
                q[i] = 0
                for j in range(n):
                    if j != i:
                        q[i] += AF(i, j)*(J[i] - J[j])
            else:
                q[i] = (Eb[i] - J[i])/R

        else:  # q is given
            if e[i] == 1:
                Eb[i] = J[i]
            else:
                Eb[i] = J[i] + q[i]*R

            T[i] = (Eb[i]/sigma)**(1/4)

    return T, q, J, B, b


def two_surface_enclosure(T1, T2, e1, e2, A1, A2, F12):
    """
    이 함수는 2면 복사 열전달량을 구한다.
    출처 : Incropera 열전달 8판, Eq. 13.23 
    
    Parameter :
        T1, T2 : 표면 온도, (K))
        e1, e2 : emissivity
        A1, A2 : surface area (m^2)
        F12 : view factor
    
    Output :
        q : heat transfer ratre (W)
        
    ed. 2021-08-18    
    """
    Eb1, Eb2 = sigma*T1**4, sigma*T2**4
    q = (Eb1 - Eb2)/((1-e1)/(e1*A1) + 1/(A1*F12) + (1-e2)/(e2*A2))
    return q


def reradiation(T1, T2, A1, A2, e1, e2, F12, F13, F23):
    """
    13.3.5 The Reradiating Surface, Eq. 13.30
    -----------------------------------------

    Parameters
    ----------
    T1, T2 : float, K
        temperature of surface 1 and 2
    A1, A2 : float, m2
        area of surface 1 and 2
    e1, e2 : float
        emissivity of surface 1 and 2
    F12, F13, F23 : float
        view factors, surface 3 is adiabatic q3 = 0

    Returns
    -------
    q1 : float, W
        rate of radiation heat transfer, q1 = -q2
    T3 : float, K
        temperature of surface 3
    """

    Eb1, Eb2 = sigma*T1**4, sigma*T2**4

    R1 = (1 - e1)/(e1*A1)
    R2 = (1 - e2)/(e2*A2)
    R12, R13, R23 = 1/(A1*F12), 1/(A1*F13), 1/(A2*F23)
    R3 = 1/(1/R12 + 1/(R13 + R23))

    q1 = (Eb1 - Eb2)/(R1 + R2 + R3)
    q2 = -q1

    J1 = Eb1 if e1 == 1 else Eb1-q1*R1
    J2 = Eb2 if e2 == 1 else Eb2-q2*R2

    J3 = (J1*R23 + J2*R13)/(R13 + R23)
    T3 = (J3/sigma)**(1/4)

    return q1, T3
