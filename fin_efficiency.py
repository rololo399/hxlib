"""
Table 3.5 Fin efficiency
"""
import numpy as np
from scipy.special import i0, i1, k0, k1
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


pi = np.pi


def overall_effeciency(eta_f, N, Af, At, Rtc=0, h=0, Ab=0):
    if Rtc == 0:
        C1 = 1
    else:
        C1 = 1 + eta_f*h*Af*Rtc/Ab
    return 1 - N*Af/At*(1 - eta_f/C1)


def straight_rectangular(k, h, L, t, w=1, is_convection_tip=True):
    if is_convection_tip: 
        Lc = L + t/2
    else:
        Lc = L
    m = np.sqrt((2*h)/(k*t))
    Af = 2*w*Lc
    Ap = t*L
    eta = np.tanh(m*Lc)/(m*Lc)
    return eta, m, Af, Ap


def straight_triangular(k, h, L, t, w=1):
    m = np.sqrt((2*h)/(k*t))
    Af = 2*w*np.sqrt(L**2 + (t/2)**2)
    Ap = t*L/2
    eta = 1/(m*L)*i1(2*m*L)/i0(2*m*L)
    return eta, m, Af, Ap


def straight_parabolic(k, h, L, t, w=1):
    m = np.sqrt((2*h)/(k*t))
    C1 = np.sqrt(1 + (t/L)**2)
    Af = w*(C1*L + L**2/t*np.log(t/L + C1))
    Ap = t/3*L
    eta = 2/(np.sqrt(4*(m*L)**2 + 1) + 1)
    return eta, m, Af, Ap


def annular_fin(k, h, r1, r2, t, is_convection_tip=True):
    if is_convection_tip:
        r2c = r2 + t/2
    else:
        r2c = r2
    m = np.sqrt((2*h)/(k*t))
    Af = 2*pi*(r2c**2 - r1**2)
    V = pi*(r2**2 - r1**2)*t
    C2 = (2*r1/m)/(r2c**2 - r1**2)
    d1 = k1(m*r1)*i1(m*r2c) - i1(m*r1)*k1(m*r2c)
    d2 = i0(m*r1)*k1(m*r2c) + k0(m*r1)*i1(m*r2c)
    eta = C2*d1/d2
    return eta, m, Af, V


def pin_fin_rectangular(k, h, L, D, is_convection_tip=True):
    if is_convection_tip:
        Lc = L + D/4
    else:
        Lc = L
    m = np.sqrt((4*h)/(k*D))
    Af = pi*D*Lc
    V = pi*D**2/4 * L
    eta = np.tanh(m*Lc)/(m*Lc)
    return eta, m, Af, V


if __name__ == "__main__":

    k = 100
    h = 10
    t = 1e-3

    L = np.linspace(1e-3, 250e-3)
    x = np.empty_like(L)
    eta = np.empty_like(L)

    for i in range(len(L)):
        eta[i], m, Af, Ap = straight_rectangular(k, h, L[i], t)
        Lc = L[i] + t/2
        x[i] = Lc**(3/2)*np.sqrt(h/(k*Ap))
    plt.plot(x, eta, label='rectangular')

    for i in range(len(L)):
        eta[i], m, Af, Ap = straight_triangular(k, h, L[i], t)
        x[i] = L[i]**(3/2)*np.sqrt(h/(k*Ap))
    plt.plot(x, eta, label='triangular')

    for i in range(len(L)):
        eta[i], m, Af, Ap = straight_parabolic(k, h, L[i], t)
        x[i] = L[i]**(3/2)*np.sqrt(h/(k*Ap))
    plt.plot(x, eta, label='parabolic')

    plt.xlim([0, 2.5])
    plt.legend()
    plt.xlabel(r'$L_c^{3/2} (h/k A_p)^{1/2}$')
    plt.ylabel(r'$\eta$')
    plt.show()
