# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-np.dot(tx,w)
    J = 0.5/y.shape[0]*np.dot(e,e)
    #J  = 1/y.shape[0]*np.sum(np.abs(e))
    
    #print(J)
    #raise NotImplementedError
    return J

