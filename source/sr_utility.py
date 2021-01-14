# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:29:55 2021

@author: alex
"""

import numpy as np

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

    