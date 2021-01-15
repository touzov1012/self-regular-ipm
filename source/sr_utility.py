# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:29:55 2021

@author: alex
"""

import numpy as np

def sqrtmatsym(A):
    """
    matrix square root
    """
    
    [evals, evecs] = np.linalg.eigh(A)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T

def sqrtmatsyminv(A):
    """
    matrix square root
    """
    
    [evals, evecs] = np.linalg.eigh(A)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

def TriLengthToSquare(k):
    """
    Get the width of a square matrix for which the upper triangle is size k
    """
    
    return int(np.rint((np.sqrt(8 * k + 1) - 1) / 2))

def svec(A):
    """
    Flatten out symmetric A, while preserving usual inner product
    """
    
    n = A.shape[0]
    B = A.copy()
    B[np.triu_indices(n, 1)] = B[np.triu_indices(n, 1)] * np.sqrt(2)
    return B[np.triu_indices(n)]

def smat(v):
    """
    Opposite of svec() operation
    """
    
    k = len(v)
    n = TriLengthToSquare(k)
    
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = v
    A[np.triu_indices(n,1)] *= 2 / np.sqrt(2)
    return (A + A.T) / 2

def Rotate(A, T):
    """
    Apply rotation A := T'AT
    """
    
    return np.matmul(np.matmul(T.T, A), T)

def svecRotate(v, T):
    """
    Rotate a matrix in svec form
    """
    
    return svec(Rotate(smat(v), T))

def Y(t, p, q):
    """
    Self-regular proximity function class kernel p,q >= 1
    """
    
    if t <= 0:
        return float('inf')
    
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

def LineSearchXS(F, x, s, dx, ds, L, U, iterates):
    """
    Line search minimum value of alpha for F(x0 + d * alpha)
    """
    
    L_val = F(x + dx * L, s + ds * L)
    U_val = F(x + dx * U, s + ds * U)
    
    if iterates <= 0:
        if L_val < U_val:
            return L
        else:
            return U
        
    
    if L_val < U_val:
        return LineSearchXS(F, x, s, dx, ds, L, (U + L) / 2, iterates - 1)
    else:
    	return LineSearchXS(F, x, s, dx, ds, (U + L) / 2, U, iterates - 1)


def SelfDualNewtonSystem(A, b, c, e):
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
    
    e is the vector for initializing interior point, e = 1 for LP, e = svec(I) for SDP
    """
    
    n = A.shape[1]
    m = A.shape[0]
    
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