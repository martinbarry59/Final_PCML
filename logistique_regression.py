import numpy as np


def backtracing(y,tx,w,gradient,beta):
    p =pow((np.linalg.norm(gradient)),2)
    t = 1
    loss = calculate_loss(y,tx,w)
    loss_mod = calculate_loss(y,tx,w-t*gradient)
    while(loss_mod > loss - t/2 *p):
        loss_mod = calculate_loss(y,tx,w-t*gradient)
        t = beta*t
    return t

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    #return  -np.dot(np.ones(y.shape[0]),(y*np.log(sigmoid(np.dot(tx,w)))+(1-y)*np.log((1-sigmoid(np.dot(tx,w))))))/y.shape[0]
    return sum(np.log(1+np.exp(np.dot(tx,w)))-np.multiply(y,(np.dot(tx,w))))

def calculate_gradient(y, tx, w):
    return  np.dot(tx.T,sigmoid(np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, alpha):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y, tx, w)
    #print(gradient)
    w = w-alpha*gradient
    return loss, w
