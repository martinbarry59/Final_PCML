# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
from math import *

def build_poly(x, degree):
    n = x.shape[1]
    dcross = np.int(factorial(n)/((factorial(n-2))*(factorial(2))))
    X=np.zeros((x.shape[0],(degree+1)*x.shape[1]+1+2*dcross))
    for i in range(degree+1) :
        if i == 0 :
         X[:,i] = x[:,0]**i
        else  :
         X[:,(i-1)*x.shape[1]+1:i*x.shape[1]+1]=x**i
    #print(degree*x.shape[1]+1)    
    #print((degree+1)*x.shape[1]+1)
    X[:,degree*x.shape[1]+1:(degree+1)*x.shape[1]+1]  = np.absolute(x)**0.5
    m=0
    for i in range (n) :
        for j in range(i+1,n) :
            m += 1
            X[:,(degree+1)*x.shape[1]+m]  = x[:,i]*x[:,j]
            m += 1
            X[:,(degree+1)*x.shape[1]+m]  = np.absolute(x[:,i]*x[:,j])**0.5

    return X