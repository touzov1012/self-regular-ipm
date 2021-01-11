# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:53:55 2021

@author: alex
"""

import numpy as np
    
def Y(t, p, q):
    """
    Self-regular proximity function class kernel p,q >= 1
    """
    
    if q == 1:
        return (t**(p+1) - 1) / (p * (p+1)) - np.log(t) / q + (p - 1) / p * (t-1)
    else:
        return (t**(p+1) - 1) / (p * (p+1)) + (t**(1-q) - 1) / (q*(q-1)) + (p - q) / (p * q) * (t-1)

def Yp(t, p, q):
    """
    Self-regular proximity gradient kernel p,q >= 1
    """
    
    return (t**p - 1) / p + (1-t**(-q)) / q

def Phi(v, p, q):
    """
    Self-regular proximity for solution v = sqrt(xs/mu) using Y(p,q) kernel
    """
    
    return np.sum([Y(t,p,q) for t in v])

def Phip(v, p, q):
    """
    Self-regular proximity gradient
    """
    
    return np.array([Yp(t,p,q) for t in v])


def SelfDualNewtonSystem(A, b, c):
    """
    Create the self dual embedding with a known interior point for the problem
    
    min cx
        Ax = b
        x >= 0
        
    The embedding will be of the form
    
    min -c*x
        A*x* = 0
    -A*'y + Cx* - S = c*
    x,S >= 0
    
    with known solution x=S=1, y=0
    
    The returned quantity is the right hand side of the linear system
    solved during a newton step, that is the optimality conditions
    xs = mu are appended in linearized form
    """
    
    n = A.shape[1]
    m = A.shape[0]
    e = np.ones(n, dtype = int)
    
    b_bar = b - np.matmul(A,e)
    c_bar = c - e
    alpha = 1 + np.dot(c, e)
    beta = n + 2
    
    A_star = np.c_[A,-b,b_bar]
    C = np.zeros((n+2,n+2))
    C[0:n,n] = c
    C[n,0:n] = -C[0:n,n].T
    C[0:n,n+1] = -c_bar
    C[n+1,0:n] = -C[0:n,n+1].T
    C[n,n+1] = alpha
    C[n+1,n] = -C[n,n+1].T
    
    yA = np.r_[np.zeros((m,m)), -A_star.T, np.zeros((n+2, m))]
    xA = np.r_[A_star, C, np.eye(n+2)]
    sA = np.r_[np.zeros((m, n+2)), -np.eye(n+2), np.eye(n+2)]
    
    return np.c_[yA, xA, sA]
    

def FindSearchDirection(nA, nb, x, s, mu, p, q):
    """
    Find the search direction (newton step)  solving
    
    yA dy + xA dx + sA ds = 0
    vdx + vds = -H(v)
    
    where H is the gradient of the proximity measure and v are v space
    quantities
    """
    
    n = len(x)
    m = nA.shape[0] - 2 * n
    
    np.fill_diagonal(nA[-n:,m:(m+n)], s)
    np.fill_diagonal(nA[-n:,(m+n):], x)
    
    grad = Phip(np.sqrt(x * s / mu), p, q)
    
    nb[0:] = 0
    nb[-n:] = -np.sqrt(mu * s * x) * grad
    
    v = np.linalg.solve(nA, nb)
    
    dy = v[0:m]
    dx = v[m:(m+n)]
    ds = v[(m+n):]
    
    alpha = min(1/(3*p+1),1/(6*q+4)) * np.linalg.norm(grad) ** (-(q+1)/q)
    
    return [dy, dx, ds, alpha]
    
    
def PDSolve(A, b, c, tau, eps, theta, p, q):
    """
    Solve a LP using self-regular proximities in IPM
    """
    
    nA = SelfDualNewtonSystem(A, b, c)
    nb = np.zeros(nA.shape[0])
    
    m = A.shape[0]
    n = A.shape[1] + 2
    
    y = np.zeros(m)
    x = np.ones(n)
    s = np.ones(n)
    mu = 1
    
    while n * mu >= eps:
        mu = (1 - theta) * mu
        
        while Phi(np.sqrt(x * s / mu), p, q) >= tau:
            [dy, dx, ds, alpha] = FindSearchDirection(nA, nb, x, s, mu, p, q)
            
            y += alpha * dy
            x += alpha * dx
            s += alpha * ds
            
    if x[-2] > s[-2] and x[-2] > eps:
        """
        P optimal
        """
        x = x[0:-2] / x[-2]
        print("Optimal Value: " + str(np.dot(c, x)) + " with x = " + str(x))
        return x
    elif s[-2] > x[-2] and s[-2] > eps:
        """
        P/D inf
        """
        print("Primal or Dual Infeasible")
        return None
    else:
        """
        Unknown
        """
        print("Unknown, check conditioning")
        return None
    
    
    
    
    
    
    
    