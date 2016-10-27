import numpy as np
def least_squares(y, tx):
  """calculate the least squares solution."""
  #w = np.dot(np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)),tx.transpose()),y)
  
  return np.linalg.solve(np.dot(tx.transpose(),tx),np.dot(tx.transpose(),y))