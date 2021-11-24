"""
Empirical correlation
"""
import numpy as np
from scipy.interpolate import interp1d
from CoolProp.CoolProp import PropsSI

g = 9.81

# =============================================================================
# free convection
# =============================================================================


def prop_01(fluid, T, P=101325):
    k = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    cp = PropsSI('C', 'T', T, 'P', P, fluid)
    Pr = PropsSI('Prandtl', 'T', T, 'P', P, fluid)
    beta = PropsSI('isobaric_expansion_coefficient',
                   'T', T, 'P', P, fluid)
    return k, mu, rho, cp, Pr, beta


def Grashof(fluid, T1, T2, L, P=101325.0):
    T = (T1+T2)/2
    k = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    Pr = PropsSI('Prandtl', 'T', T, 'P', P, fluid)
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
    if Ra < 1e9:  # Eq 9.27
        Nu = 0.68 + 0.670*Ra**(1/4)/(1+(0.492/Pr)**(9/16))**(4/9)
    else:  # Eq 9.26 Churchill and Chu
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
    if a1 < 0:
        a1 = 0
    a2 = 1 - 1708*(np.sin(1.8*tau))**(1.6)/(Ra*np.cos(tau))
    a3 = (Ra*np.cos(tau)/5830)**(1/3) - 1
    if a3 < 0:
        a3 = 0
    Nu = 1 + 1.44*a1*a2 + a3
    return Nu


def fc_tilted_cavity(fluid, T1, T2, L, tau, P):
    Ra, Pr, k = Rayleigh(fluid, T1, T2, L, P)
    Nu = fc_tilted_cavity_nd(Ra, tau)
    h = Nu*k/L
    return h, Nu, Ra

# =============================================================================
# external flow
# =============================================================================


def prop_02(fluid, T, P=101325):
    k = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    Pr = PropsSI('Prandtl', 'T', T, 'P', P, fluid)
    return k, mu, rho, Pr


def Reynolds(fluid, T, V, L, P=101325):
    k = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu = PropsSI('viscosity', 'T', T, 'P', P, fluid)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)
    Pr = PropsSI('Prandtl', 'T', T, 'P', P, fluid)
    Re = rho*V*L/mu
    return Re, Pr, k


# plate
def external_flow_plate_nd(Re, Pr, Re_c=5e5):
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
    Tf = (Ts + Tinf)/2  # film temperature
    Re, Pr, k = Reynolds(fluid, Tf, uinf, L, P=P)
    Nu = external_flow_plate_nd(Re, Pr, Re_c=Re_c)
    h = Nu*k/L
    return h, Nu, Re


# cylinder
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
    Tf = (Ts + Tinf)/2  # film temperature
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

    if 0.4 < Re < 4:
        C, m = 0.989, 0.330
    elif Re < 4e1:
        C, m = 0.911, 0.385
    elif Re < 4e3:
        C, m = 0.683, 0.466
    elif Re < 4e4:
        C, m = 0.193, 0.618
    elif Re < 4e5:
        C, m = 0.027, 0.805
    else:
        return None

    Nu = C*Re**m*Pr**n*(Pr/Prs)**(1/4)
    return Nu


def Zukauskas(fluid, Ts, Tinf, V, D, P=101325):
    """for external flow past an infinite cylinder.  """
    Re, Pr, k = Reynolds(fluid, Tinf, V, D, P=P)
    Prs = PropsSI('Prandtl', 'T', Ts, 'P', P, fluid)
    Nu = Zukauskas_nd(Re, Pr, Prs)
    h = Nu*k/D
    return h, Nu, Re


# sphere
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
        fluid, T_in, T_out, T_s,  P, u_inf, N_L, D, S_T, S_L):
    N_LL = np.array([1, 2, 3, 4, 5, 7, 10, 13, 16, 20])
    C22 = np.array([0.7, 0.8, 0.86, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1])
    f_C2 = interp1d(N_LL, C22)

    Tm = (T_in + T_out)/2
    Vmax = S_T/(S_T - D)*u_inf
    Re, Pr, k = Reynolds(fluid, Tm, Vmax, D)
    Prs = PropsSI('Prandtl', 'T', T_s, 'P', P, fluid)
    Nu = External_Flow_Inline_Bank_nd(Re, Pr, Prs)

    if N_L < 20:
        Nu = f_C2(N_L)*Nu

    h = Nu*k/D
    return h, Nu, Re


def External_Flow_Inline_Bank_nd(Re, Pr, Prs):
    """
    N_L > 20, 0.7 < Pr < 500, 10 < Re < 2e6
    """
    if 10 < Re < 1e2:
        C1, m = 0.80, 0.40
    elif 1e2 <= Re <= 1e3:
        Nu = Zukauskas_nd(Re, Pr, Prs)
        return Nu
    elif Re < 2e5:
        C1, m = 0.27, 0.63
    elif Re < 2e6:
        C1, m = 0.021, 0.84
    else:
        return None
    Nu = C1*Re**m*Pr**0.36*(Pr/Prs)**(1/4)
    return Nu

# =============================================================================
# pipe flow, internal flow
# =============================================================================


def moody(Re, RR=0, tol=1e-6):
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

# def Reynolds(fluid, T, V, L, P=101325):
#     k   = PropsSI('conductivity', 'T', T, 'P', P, fluid)
#     mu  = PropsSI('viscosity',    'T', T, 'P', P, fluid)
#     rho = PropsSI('D',            'T', T, 'P', P, fluid)
#     Pr  = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
#     Re = rho*V*L/mu
#     return Re, Pr, k


def Graetz(fluid, T, V, D, x, P=101325):
    Re, Pr, k = Reynolds(fluid, T, V, D, P=P)
    Gz = (D/x)*Re*Pr
    return Gz, Re, Pr, k


def thermal_entry_nd(Gz):
    Nu = 3.66 + 0.0668*Gz/(1 + 0.04*Gz**(2/3))
    return Nu


def thermal_entry(fluid, T, V, D, x, P=101325):
    Gz, Re, Pr, k = Graetz(fluid, T, V, D, x, P=P)
    Nu = thermal_entry_nd(Gz)
    h = Nu*k/D
    return h, Nu, Gz, Re, Pr


def combined_entry_nd(Gz, Pr):
    a1 = 3.66/np.tanh(2.264*Gz**(-1/3) + 1.7*Gz**(-2/3))
    a2 = 0.0499*Gz*np.tanh(1/Gz)
    a3 = np.tanh(2.432*Pr**(1/6)*Gz**(-1/6))
    Nu = (a1 + a2)/a3
    return Nu


def combined_entry(fluid, T, V, D, x, P=101325):
    Gz, Re, Pr, k = Graetz(fluid, T, V, D, x, P=P)
    Nu = combined_entry_nd(Gz, Pr)
    h = Nu*k/D
    return h, Nu, Gz, Re, Pr


def Dittus_Boelter_nd(Re, Pr, n=0.4):
    """
    all properties are evaluated at Tm, Eq. 8.60
    turbulent flow in circular tube
    0.6 < Pr < 160, Re > 10000, L/D > 10
    n = 0.4 for heating (Ts > Tm), 0.3 for cooling (Ts < Tm)
    """
    Nu = 0.023*Re**(4/5)*Pr**n
    return Nu


def Dittus_Boelter(fluid, Tm, Ts, Dh, V, P=101325):
    """ all properties are evaluated at Tm, Eq. 8.60 """
    Re, Pr, k = Reynolds(fluid, Tm, V, Dh, P=P)
    n = 0.4 if Ts > Tm else 0.3
    Nu = Dittus_Boelter_nd(Re, Pr, n=n)
    h = k*Nu/Dh
    return h, Nu, Re


def Dittus_Boelter_mdot(fluid, Tm, Ts, Dh, Ac, mdot, P=101325):
    rho = PropsSI('D', 'T', Tm, 'P', P, fluid)
    V = mdot/(rho*Ac)
    Re, Pr, k = Reynolds(fluid, Tm, V, Dh, P=P)
    n = 0.4 if Ts > Tm else 0.3
    Nu = Dittus_Boelter_nd(Re, Pr, n=n)
    h = k*Nu/Dh
    return h, Nu, Re


def Sieder_Tate_nd(Re, Pr, mu_over_mus):
    """ turbulent flow in circular tube Eq. 8.61 """
    Nu = 0.027*Re**(4/5)*Pr**(1/3)*mu_over_mus**(0.14)
    return Nu


def Sieder_Tate(fluid, Tm, Ts, D, V, P=101325):
    Re, Pr, k = Reynolds(fluid, Tm, V, D, P=P)
    mu = PropsSI('viscosity', 'T', Tm, 'P', P, fluid)
    mus = PropsSI('viscosity', 'T', Ts, 'P', P, fluid)
    Nu = Sieder_Tate_nd(Re, Pr, mu/mus)
    h = k*Nu/D
    return h, Nu, Re


def Gnielinski_nd(Re, Pr, f):
    """ Eq. 8.62 """
    a1 = (f/8)*(Re - 1000)*Pr
    a2 = 1 + 12.7*(f/8)**(1/2)*(Pr**(2/3) - 1)
    Nu = a1/a2
    return Nu


def Gnielinski(fluid, Tm, Ts, D, V, epsilon=0, P=101325):
    Re, Pr, k = Reynolds(fluid, Tm, V, D, P=P)
    RR = epsilon/D
    f = moody(Re, RR=RR)
    Nu = Gnielinski_nd(Re, Pr, f)
    h = k*Nu/D
    return h, Nu, Re
