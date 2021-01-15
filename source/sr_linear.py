# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:53:55 2021

@author: alex
"""

from sr_utility import *
import numpy as np
    

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
    
    v = np.sqrt(x * s / mu)
    
    grad = Phip(v, p, q)
    
    nb[0:] = 0
    nb[-n:] = -np.sqrt(mu * s * x) * grad
    
    sol = np.linalg.solve(nA, nb)
    
    dy = sol[0:m]
    dx = sol[m:(m+n)]
    ds = sol[(m+n):]
    
    alpha = min(1/(3*p+1),1/(6*q+4)) * np.linalg.norm(grad) ** (-(q+1)/q)
    
    return [dy, dx, ds, alpha]
    
    
def PDSolve(A, b, c, tau, eps, theta, p, q, search_steps):
    """
    Solve a LP using self-regular proximities in IPM
    Data: (A, b, c)
    tau: proximity ball size, update central path target after entering
    eps: target duality gap
    theta: central path shrink coefficient, how much do we shift mu each update
    p, q: proximity function parameters
    search_steps: number of line search iterations
    """
    
    nA = SelfDualNewtonSystem(A, b, c, np.ones(A.shape[1]))
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
            
            ls_alpha = LineSearchXS(lambda u,w: Phi(np.sqrt(u.clip(min=0) * w.clip(min=0) / mu), p, q), x, s, dx, ds, alpha, 2, search_steps)
            
            #print(str(ls_alpha) + " " + str(alpha))
            
            y += ls_alpha * dy
            x += ls_alpha * dx
            s += ls_alpha * ds
    
    
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
    
    
    
def TestLP():
    """
    Generate a test LP problem in standard form
    opt: 6450
    x = (700, 700, 50)
    """
    
    A = np.array([[0.4, 0.30, 0.2, -1,  0,  0, 0, 0, 0],
                  [0.4, 0.35, 0.2,  0, -1,  0, 0, 0, 0],
                  [0.2, 0.35, 0.6,  0,  0, -1, 0, 0, 0],
                  [  1,    0,   0,  0,  0,  0, 1, 0, 0],
                  [  0,    1,   0,  0,  0,  0, 0, 1, 0],
                  [  0,    0,   1,  0,  0,  0, 0, 0, 1]])
    
    b = np.array([500, 300, 300, 700, 700, 700])
    
    c = np.array([5, 4, 3, 0, 0, 0, 0, 0, 0])
    
    return [A, b, c]
    
    
    
    
    
    
    
    