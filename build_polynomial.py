# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    X=np.zeros((x.shape[0],degree*x.shape[1]+1))
    for i in range(degree+1) :
        if i == 0 :
         X[:,i] = x[:,0]**i
        else  :
         X[:,(i-1)*x.shape[1]+1:i*x.shape[1]+1]=x**i
    return X