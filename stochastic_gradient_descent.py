# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
import numpy as np
import costs as co
def compute_stoch_gradient(y, tx, w):
    return -np.dot((y-np.dot(tx,w)),tx)/y.shape[0]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,max_epochs):
            
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            loss = co.compute_loss(y,tx, w)
        
            w=w-gamma*grad
        # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)
    
            n_iter+=1
    print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=max_epochs - 1, ti=max_epochs - 1, l=loss))        
    return losses, ws
