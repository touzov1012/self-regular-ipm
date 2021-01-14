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

def LineSearchMin(F, x_0, d, L, U, eps):
    """
    Line search minimum value of alpha for F(x0 + d * alpha)
    """
    
    if abs(U - L) < eps:
        return (U + L) / 2
    
    L_val = F(x_0 + (3/4 * L + 1/4 * U) * d)
    U_val = F(x_0 + (1/4 * L + 3/4 * U) * d)
    
    
    if L_val < U_val:
        return LineSearchMin(F, x_0, d, L, 1/4 * L + 3/4 * U, eps)
    else:
    	return LineSearchMin(F, x_0, d, 3/4 * L + 1/4 * U, U, eps)

    