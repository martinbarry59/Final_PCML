# -*- coding: utf-8 -*-

import numpy as np
import costs as co


def backtracing(y,tx,w,gradient,beta):
    p =pow((np.linalg.norm(gradient)),2)
    t = 1
    loss = co.compute_loss(y,tx,w)
    loss_mod = co.compute_loss(y,tx,w-t*gradient)
    while(loss_mod > loss - t/2 *p):
        loss_mod = co.compute_loss(y,tx,w-t*gradient)
        t = beta*t
    return t

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=(y-np.dot(tx,w))
    
    return -1/y.shape[0]*np.dot(e,tx) 
    #return -1/y.shape[0]*np.dot(np.sign(e),tx) 


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        loss = co.compute_loss(y, tx, w)
        gamma = backtracing(y,tx,w,grad,0.3)
        print(gamma)
        w=w-gamma*grad
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        #if (n_iter+1)%max_iters==0 :
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter , ti=max_iters - 1, l=loss))

    return losses, ws
