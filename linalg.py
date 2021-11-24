import numpy as np

def gauss(A, b):
    n = A.shape[0]
    A = A.copy()
    b = b.copy()
    
    for k in range(n-1):
        for i in range(k+1,n):
            factor = A[i,k]/A[k,k]
            A[i, k+1:] -= factor*A[k, k+1:]
            b[i] -= factor*b[k]
            A[i,k] = factor
            
    for k in reversed(range(n)):
        b[k] = (b[k] - np.dot(A[k,k+1:], b[k+1:]))/A[k,k]
    return b

def gauss_pivot(A, b):
    n = A.shape[0]
    A = A.copy()
    b = b.copy()
    for k in range(n-1):
        
        i = k + np.argmax(abs(A[k:,k]))
        if i != k:
            A[[k,i]] = A[[i,k]]
            b[[k,i]] = b[[i,k]]
        
        for i in range(k+1,n):
            factor = A[i,k]/A[k,k]
            A[i, k+1:] -= factor*A[k, k+1:]
            b[i] -= factor*b[k]
            A[i,k] = factor
            
    for k in reversed(range(n)):
        b[k] = (b[k] - np.dot(A[k,k+1:], b[k+1:]))/A[k,k]
        
    return b

def tdma(tri):
    e,f,g,r = tri.copy().T
    n = len(f)
    for k in range(1,n):
        factor = e[k]/f[k-1]
        f[k] -= factor*g[k-1]
        r[k] -= factor*r[k-1]    
    r[-1] /= f[-1]
    for k in reversed(range(n-1)):
        r[k] = (r[k] - g[k]*r[k+1])/f[k]
    return r

def LUdec(A):
    n = A.shape[0]
    A = A.copy()
    for k in range(n-1):
        for i in range(k+1,n):
            factor = A[i,k]/A[k,k]
            A[i, k+1:] -= factor*A[k, k+1:]
            A[i,k] = factor
    return A

def LUsol(A,b):
    n = A.shape[0]
    b = b.copy()
    for k in range(1,n):
        b[k] -= np.dot(A[k,:k], b[:k])
    for k in reversed(range(n)):
        b[k] = (b[k] - np.dot(A[k,k+1:], b[k+1:]))/A[k,k]
    return b


if __name__ == "__main__":
    A = np.array([[3, -0.1, -0.2],[0.1, 7., -0.3],[0.3, -0.2, 10]])
    b = np.array([7.85, -19.3, 71.4])
    #n = 5
    #A = np.random.random((n,n))
    #b = np.random.random(n)
    
    x = np.linalg.solve(A,b)
    print(np.round(A@x-b,2))
    
    x = gauss(A,b)
    print(np.round(A@x-b,2))
    
    x = gauss_pivot(A,b)
    print(np.round(A@x-b,2))
    
    B = LUdec(A)
    x = LUsol(B,b)
    print(np.round(A@x-b,2))