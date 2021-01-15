# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:45:45 2021

@author: alex
"""

from sr_utility import *
import numpy as np

def NTScaling(X, S):
    """
    Calculate D, the scaling transformation
    """
    
    Xhalf = sqrtmatsym(X)
    return sqrtmatsym(Xhalf @ sqrtmatsyminv(Xhalf @ S @ Xhalf) @ Xhalf)

def Prox(x, s, mu, p, q):
    """
    Evaluate the proximity at a vectorized x, s, which have additional 2 dim
    """
    
    X = smat(x[0:-2])
    S = smat(s[0:-2])
    
    xl = min(np.linalg.eigvals(X))
    sl = min(np.linalg.eigvals(S))
    
    if xl <= 0 or sl <= 0 or x[-2] <= 0 or x[-1] <= 0 or s[-2] <= 0 or s[-1] <= 0:
        return float('inf')
    
    D = NTScaling(X, S)
    d = np.sqrt(x[-2:] / s[-2:])
    
    V = Rotate(S, D) / np.sqrt(mu)
    [evals, evecs] = np.linalg.eigh(V)
    phi0 = Phi(evals, p, q)
    phi1 = Phi(d * s[-2:] / np.sqrt(mu), p, q)
    
    return phi0 + phi1

def FindSearchDirection(nA, nb, x, s, mu, p, q):
    """
    Find the search direction (newton step)  solving
    
    yA dy + xA dx + sA ds = 0
    vdx + vds = -H(v)
    
    where H is the gradient of the proximity measure and v are v space
    quantities
    """
    
    len_vec_var = len(x)
    trm_vec_var = len_vec_var - 2
    const_cnt = nA.shape[0] - 2 * len_vec_var
    wdt_mat_var = TriLengthToSquare(trm_vec_var)
    
    X = smat(x[0:trm_vec_var])
    S = smat(s[0:trm_vec_var])
    
    D = NTScaling(X, S)
    Dinv = np.linalg.inv(D)
    d = np.sqrt(x[-2:] / s[-2:])
    
    # set the rotated data
    A = nA.copy()
    A[0:const_cnt, const_cnt:(const_cnt+trm_vec_var)] = np.apply_along_axis(lambda u: svecRotate(u, D), 1, A[0:const_cnt, const_cnt:(const_cnt+trm_vec_var)])
    A[0:const_cnt, const_cnt+trm_vec_var] *= d[0]
    A[0:const_cnt, const_cnt+trm_vec_var+1] *= d[1]
    A[const_cnt:(const_cnt+len_vec_var), 0:const_cnt] = -A[0:const_cnt, const_cnt:(const_cnt+len_vec_var)].T
    A[const_cnt:(const_cnt+trm_vec_var), const_cnt+trm_vec_var] = svecRotate(A[const_cnt:(const_cnt+trm_vec_var), const_cnt+trm_vec_var], D)
    A[const_cnt:(const_cnt+trm_vec_var), const_cnt+trm_vec_var+1] = svecRotate(A[const_cnt:(const_cnt+trm_vec_var), const_cnt+trm_vec_var+1], D)
    A[(const_cnt+trm_vec_var):(const_cnt+len_vec_var), const_cnt:(const_cnt+trm_vec_var)] = -A[const_cnt:(const_cnt+trm_vec_var), (const_cnt+trm_vec_var):(const_cnt+len_vec_var)].T
    A[const_cnt+trm_vec_var, const_cnt+trm_vec_var+1] *= d[1]
    A[const_cnt+trm_vec_var+1, const_cnt+trm_vec_var] = -A[const_cnt+trm_vec_var, const_cnt+trm_vec_var+1]
    
    
    V = Rotate(S, D) / np.sqrt(mu)
    [evals, evecs] = np.linalg.eigh(V)
    pvals = Phip(evals, p, q)
    grad = evecs @ np.diag(pvals) @ evecs.T
    
    nb[0:] = 0
    nb[-len_vec_var:-2] = -np.sqrt(mu) * svec(grad)
    nb[-2:] = -np.sqrt(mu) * Phip(d * s[-2:] / np.sqrt(mu), p, q)
    
    sol = np.linalg.solve(A, nb)
    
    dy = sol[0:const_cnt]
    dx = sol[const_cnt:(const_cnt+len_vec_var)]
    ds = sol[(const_cnt+len_vec_var):]
    
    dx[0:-2] = svecRotate(dx[:-2], D)
    dx[-2:] *= d
    ds[0:-2] = svecRotate(ds[:-2], Dinv)
    ds[-2:] /= d
    
    alpha = min(1/(3*p+1),1/(6*q+4)) * np.linalg.norm(pvals) ** (-(q+1)/q)
    
    return [dy, dx, ds, alpha]


def PDSolve(A, b, c, tau, eps, theta, p, q, search_steps):
    """
    Solve a SDP using self-regular proximities in IPM
    Data: (A, b, c)
    tau: proximity ball size, update central path target after entering
    eps: target duality gap
    theta: central path shrink coefficient, how much do we shift mu each update
    p, q: proximity function parameters
    search_steps: number of line search iterations
    """
    
    m = A.shape[0]
    nt = A.shape[1]
    
    A = np.array([svec(A[i]) for i in range(0,m)])
    c = svec(c)
    
    nA = SelfDualNewtonSystem(A, b, c, svec(np.eye(nt)))
    nb = np.zeros(nA.shape[0])
    
    
    y = np.zeros(m)
    x = np.append(svec(np.eye(nt)), [1,1])
    s = np.append(svec(np.eye(nt)), [1,1])
    mu = 1
    
    n = nt + 2
    
    while n * mu >= eps:
        mu = (1 - theta) * mu
        
        while Prox(x, s, mu, p, q) >= tau:
            [dy, dx, ds, alpha] = FindSearchDirection(nA, nb, x, s, mu, p, q)
            
            ls_alpha = LineSearchXS(lambda u,w: Prox(u, w, mu, p, q), x, s, dx, ds, alpha, 2, search_steps)
            
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
        Pathology encountered
        """
        print("SDP Pathology")
        return None