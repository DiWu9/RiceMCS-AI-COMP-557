import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    patchSize = patches.shape[0]
    numPatches = patches.shape[1]
    centroids = np.random.randn(patchSize,k)

    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)

        # find closest centroid for each patch
        patchSum = np.zeros((patchSize, k)) # sum of patch values that shares the closest centroid for centroids 1-k
        patchNum = np.zeros(k) # num of patches that shares the closest centroid for centroids 1-k
        for ithPatch in range(numPatches):
              patch = patches[:,ithPatch]
              dist = np.sqrt(np.sum((np.transpose(centroids) - patch) ** 2, axis=1))
              closest = np.argmin(dist) # index of the closest centroid
              patchSum[:,closest] += patch
              patchNum[closest] += 1

        # recalculate cluster mean
        for ithCentroid in range(k):
              centroids[:,ithCentroid] = patchSum[:,ithCentroid] / patchNum[ithCentroid]
        
        # END_YOUR_CODE

    return centroids

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for ithPatch in range(numPatches):
          patch = patches[:,ithPatch]
          dist = np.sqrt(np.sum((np.transpose(centroids) - patch) ** 2, axis=1))
          avgDist = np.sum(dist) / k
          for ithCentroid in range(k):
                centroid = centroids[:,ithCentroid]
                aPatchCentroid = max(0, avgDist - np.sqrt(np.sum((patch - centroid) ** 2)))
                features[ithPatch][ithCentroid] = aPatchCentroid
    # END_YOUR_CODE
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    yy = y * 2 - 1
    power = yy * (-np.matmul(featureVector, np.transpose(theta)))
    return - featureVector * yy * math.exp(power) / (1 + math.exp(power))
    # END_YOUR_CODE

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    yy = y * 2 - 1
    loss = 1 - np.matmul(featureVector, np.transpose(theta)) * yy
    if loss >= 0:
          return -featureVector * yy
    else:
          return featureVector * 0
    # END_YOUR_CODE

