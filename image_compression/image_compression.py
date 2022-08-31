""" 
Using K-means to compress a 128x128 pixel image
author: Ramon Cardias (2022)
"""

import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    """
    Returns the 'idx' which are the closest centroids of points 'X'
    """

    # Set K
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)
    m = range(X.shape[0])
    for i in range(X.shape[0]):
        #print(i)
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = idx[i] = np.argmin(distance)
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    """
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    
    for i in range(K):
        points = X[idx==i]
        centroids[i] = np.mean(points, axis = 0)
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    return centroids



if __name__ == '__main__':

   # Load an image
   original_img = plt.imread('poney.png')

   # Visualizing the image
   plt.imshow(original_img)
   # Divide by the shape of the figure so that all values are in the range 0 - 1
   original_img = original_img / original_img.shape[0]

   # Reshape the image into an m x 3 matrix where m = number of pixels
   # (in this case m = 128 x 128 = 16384)
   # Each row will contain the Red, Green and Blue pixel values
   # This gives us our dataset matrix X_img that we will use K-Means on.
   X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], original_img.shape[2]))
 
   # Run your K-Means algorithm on this data
   # You should try different values of K and max_iters here
   K = 8                        
   max_iters = 10               
  
   # Using the function you have implemented above. 
   initial_centroids = kMeans_init_centroids(X_img, K) 
  
   # Run K-Means
   centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

   # Represent image in terms of indices
   X_recovered = centroids[idx, :] 

   # Reshape recovered image into proper dimensions
   X_recovered = np.reshape(X_recovered, original_img.shape) 

   # Display original image
   fig, ax = plt.subplots(1,2, figsize=(8,8))
   plt.axis('off')
  
   ax[0].imshow(original_img*original_img.shape[0])
   ax[0].set_title('Original')
   ax[0].set_axis_off()
  
  
   # Display compressed image
   ax[1].imshow(X_recovered*original_img.shape[0])
   ax[1].set_title('Compressed with %d colours'%K)
   ax[1].set_axis_off()
   plt.show()
