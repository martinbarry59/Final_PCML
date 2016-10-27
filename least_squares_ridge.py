import numpy as np
def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    #w = np.dot(np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)+np.dot(lamb.transpose(),lamb)),tx.transpose()),y)
    L = lamb*np.identity(tx.shape[1])*2*y.shape[0]
    return np.linalg.solve(np.dot(tx.transpose(),tx)+L,np.dot(tx.transpose(),y))
  