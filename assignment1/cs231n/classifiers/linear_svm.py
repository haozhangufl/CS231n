import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1;
  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in xrange(num_train):
    for j in xrange(num_classes):
        if j == y[i]:
            s = 0
            for k in xrange(num_classes):
                if k!= j:
                    if W[k,:].dot(X[:,i])-W[j,:].dot(X[:,i])+delta > 0:
                        s = s + 1
            grad = -s*X[:,i]            
            dW[j,:] = dW[j,:] + grad
        else:
            s = W[j,:].dot(X[:,i]) - W[y[i],:].dot(X[:,i])+delta > 0
            grad = s*X[:,i]
            dW[j,:] = dW[j,:] + grad                                            

  dW = dW/float(num_train) + reg* W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1
  num_train = X.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  Score = np.matrix(W)*np.matrix(X)   # C x N matrix store all the scores
  Correct_Score = np.choose(y,Score)  # 1 x N vector store the correct score
  Matrix_L = np.maximum(Score - Correct_Score+delta,0)
  L = np.sum(Matrix_L,axis=0)
  loss = np.sum(L - delta)/float(num_train)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  Matrix_I = 1*(Matrix_L>0)   # C x N matrix as indicator 
  Matrix_I[y,range(num_train)] = - (np.sum(Matrix_I,axis=0) - delta)
  dW = (np.matrix(Matrix_I)*(np.matrix(X).T))/float(num_train) + reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
