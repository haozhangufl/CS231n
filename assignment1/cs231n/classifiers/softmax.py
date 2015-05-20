import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(X.shape[1]):
      f = np.dot(W,X[:,i])
      f -= np.max(f)
      q = np.exp(f)/np.sum(np.exp(f))
      loss += -np.log(q[y[i]])
  
      for c in xrange(W.shape[0]):
            if c == y[i]:
                dW[c,:] += X[:,i].T*(q[c] - 1)
            else:
                dW[c,:] += X[:,i].T*q[c]
        
  loss = loss/float(X.shape[1])
  loss += 0.5*reg*np.sum(W * W)

  dW = dW/float(X.shape[1])
  dW += reg*W        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  score = np.dot(W,X)         # C x N matrix store all the scores
  score -= np.max(score)
  F = np.exp(score)
  sF = np.sum(F,axis=0)   # 1 x N vector store the sum for each sample  
  Q = F/sF
  cQ = np.choose(y,Q)
  loss = -np.sum(np.log(cQ))/(float(X.shape[1]))
  loss += 0.5*reg*np.sum(W * W)
    
  M = np.zeros_like(Q)
  M[y,xrange(M.shape[1])] = 1
  D = Q - M
  dW = np.dot(D,X.T)/(float(X.shape[1]))
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
