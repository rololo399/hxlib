
import numpy as np
from CoolProp.CoolProp import PropsSI

g = 9.81

def Reynolds(fluid, T, V, L, P=101325):
    k   = PropsSI('conductivity', 'T', T, 'P', P, fluid)
    mu  = PropsSI('viscosity',    'T', T, 'P', P, fluid)
    rho = PropsSI('D',            'T', T, 'P', P, fluid)
    Pr  = PropsSI('Prandtl',      'T', T, 'P', P, fluid)
    Re = rho*V*L/mu
    return Re, Pr, k

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
    rho  = PropsSI('D', 'T', Tm, 'P', P, fluid)
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
    mu  = PropsSI('viscosity', 'T', Tm, 'P', P, fluid)
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
