# -*- coding: utf-8 -*-

import numpy as np
from math import *

"""Custom helper functions"""

"""implement a polynomial basis function."""
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

def formating(tX) :
    tX_final =tX    
    number = np.zeros(tX.shape[0])
    index = []
    
    for j in range(tX.shape[1]) :
        r = [index for index,value in enumerate(tX[:,j]) if value != -999.]
        x = tX[r,j]
        median = np.median(x)
        #print(median)
        if median == median :
            index.append(j)
        for i in range(tX.shape[0]) :
            if tX[i][j]!=tX[i][j] :
                number[i]+=1
                tX_final[i,j]= median    
    print(index)
    tX_final = tX_final[:,index]
    return tX_final , index

def backtracing(y,tx,w,gradient,beta):
    p =pow((np.linalg.norm(gradient)),2)
    t = 1
    loss = compute_loss(y,tx,w)
    loss_mod = compute_loss(y,tx,w-t*gradient)
    while(loss_mod > loss - t/2 *p):
        loss_mod = compute_loss(y,tx,w-t*gradient)
        t = beta*t
    return t

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=(y-np.dot(tx,w))
    
    return -1/y.shape[0]*np.dot(e,tx) 
    #return -1/y.shape[0]*np.dot(np.sign(e),tx) 

def compute_loss(y, tx, w):
    """Calculate the loss using mse or mae."""
    e=y-np.dot(tx,w)
    J = 0.5/y.shape[0]*np.dot(e,e)
    #J  = 1/y.shape[0]*np.sum(np.abs(e))
    
    return J

def compute_rmse(y, tx, w):
    return np.sqrt(2 * compute_loss(y, tx, w))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    
"""ML algorithms"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        loss = compute_loss(y, tx, w)
        gamma = backtracing(y,tx,w,grad,0.3)
        print(gamma)
        w=w-gamma*grad
       
        #if (n_iter+1)%max_iters==0 :
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter , ti=max_iters - 1, l=loss))

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    w = initial_w
    n_iter = 0
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,max_iters):
            
            grad = compute_gradient(minibatch_y,minibatch_tx,w)
            loss = compute_loss(y,tx, w)
        
            w=w-gamma*grad
    
            n_iter+=1
    print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=max_iters - 1, ti=max_iters - 1, l=loss))        
    return w, loss

def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.transpose(),tx),np.dot(tx.transpose(),y))
    rmse = compute_rmse(y, tx, w)
    return w, rmse

"""Helper functions for ridge regression"""
def grid_search(y,tx,lambda_,degree):
    los = np.zeros((len(lambda_),len(degree)))
    for i in range(0,len(los[0])):
        for j in range(0,len(los[1])):
            weight,losses = ridge_regression_demo(y, tx,lambda_[i],degree[j])
            los[i,j] = losses
    return los
def get_best_parameters(lambda_, degree, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], lambda_[min_row], degree[min_col]


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    #w = np.dot(np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)+np.dot(lamb.transpose(),lamb)),tx.transpose()),y)
    L = lambda_*np.identity(tx.shape[1])*2*y.shape[0]
    w = np.linalg.solve(np.dot(tx.transpose(),tx)+L,np.dot(tx.transpose(),y))
    rmse = compute_rmse(y, tx, w)
    return w, rmse

"""Logistic regression functions"""

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    #return  -np.dot(np.ones(y.shape[0]),(y*np.log(sigmoid(np.dot(tx,w)))+(1-y)*np.log((1-sigmoid(np.dot(tx,w))))))/y.shape[0]
    return sum(np.log(1+np.exp(np.dot(tx,w)))-np.multiply(y,(np.dot(tx,w))))

def calculate_gradient(y, tx, w):
    """compute the gradient using the sigmoid function"""
    return  np.dot(tx.T,sigmoid(np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y, tx, w)
    #print(gradient)
    w = w - gamma * gradient
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters = 200, gamma = 1e-12):
    # init parameters
    threshold = 0.1
    losses = []
    degree = 1
    w = initial_w
    y = (1+y)/2
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss

"""Reg Logistic Regression"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    y = (1+y)/2
    losses = []
    threshold = 0.1
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    norm = sum(w**2)
    cost = w + lambda_*norm/(2*np.shape(w)[0])
    return w, cost
