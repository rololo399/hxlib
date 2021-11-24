"""
Created on Mon Aug 12 18:32:42 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

pi = np.pi

def printt(p, time, T, fmt="{:10.3f}"):
    # fmt = '{:10.3f}'
    print("{:5d}".format(p), end='')
    print(fmt.format(time), end='')
    for i in range(len(T)):
        print(fmt.format(T[i]), end='')
    print()

def mygeomspace(x1, x2, n, s=0.1, increase=True):
    y = np.geomspace(s, 1+s, n) - s
    dy = np.diff(y)
    if increase:
        x = np.r_[x1, x1 + (x2 - x1)*np.cumsum(dy)]
    else:
        x = np.r_[x1, x1 + (x2 - x1)*np.cumsum(dy[::-1])]
    return x
    
def gauss_seidel(fun, x, kmax=1000, tol=1e-6):
    for kk in range(kmax):
        xold = x.copy()
        x = fun(x)
        err = np.linalg.norm(x-xold)/np.linalg.norm(x)
        if err < tol: 
            # print('iteration : ', kk)
            return x
       
def tdma(tri):
    c,d,e,r = tri.T
    n = len(d)
    for k in range(1,n):
        factor = c[k]/d[k-1]
        d[k] -= factor*e[k-1]
        r[k] -= factor*r[k-1]
    r[-1] = r[-1]/d[-1]
    for k in reversed(range(n-1)):
        r[k] = (r[k] - e[k]*r[k+1])/d[k]
    return r

def secant(func, x1, x2=None, tol=1e-6, kmax=100):    
    if x2 == None: x2 = 1.1*x1
    f1, f2 = func(x1), func(x2)
    for k in range(kmax):
        x3 = x2 - f2*(x1 - x2)/(f1 - f2)
        f3 = func(x3)
        if np.abs(x2 - x3) < tol*np.max([1, np.abs(x3)]):
            return x3
        x1, f1, x2, f2 = x2, f2, x3, f3
    return None

def bisect(fun, x1, x2, tol=1e-6):
    f1, f2 = fun(x1), fun(x2)
    while 1:
        if np.abs(x1 - x2) < tol*np.max([1, np.abs(x1)]):
            return x1
        x3 = (x1 + x2)/2
        f3 = fun(x3)
        if f1*f3 < 0:
            x2, f2 = x3, f3
        else:
            x1, f1 = x3, f3

def fzeros(fun, x1=0, h=0.05, n=10):
    x = np.empty(n)
    f1 = fun(x1)
    i = 0
    while i < n:
        x2 = x1 + h; f2 = fun(x2)
        if f1*f2 < 0:
            x[i] = bisect(fun, x1, x2)
            i += 1
        x1, f1 = x2, f2
    return x

def inner(x, y1, y2=1, w=1): 
    return trapz(y1*y2*w,x)

def trapz(y, x):
    n = len(x)
    s = 0
    for i in range(n-1):
        s += 0.5*(y[i+1] + y[i])*(x[i+1] - x[i])
    return s      

def oderk4(odefun, y0, t):
    def rk4(odefun, y, t, h):
        k1 = odefun(y, t)
        k2 = odefun(y + h*1/2*k1, t + 1/2*h)
        k3 = odefun(y + h*1/2*k2, t + 1/2*h)
        k4 = odefun(y + h*k3, t + h)
        return y + h*(k1 + 2*k2 + 2*k3 + k4)/6
    n = len(t)
    y0 = np.atleast_1d(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        y[i+1] = rk4(odefun, y[i], t[i], t[i+1]-t[i])    
    return y    
