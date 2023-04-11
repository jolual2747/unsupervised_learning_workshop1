import numpy as np
from scipy.linalg import svd
import logging
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from numpy.linalg import norm
from random import normalvariate
from math import sqrt

# PCA from scratch
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)
        return X_transformed    
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class svd_scratch:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, X):
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.sigma = np.diag(sigma)[0:self.n_components,:self.n_components]
        self.VT = VT[:self.n_components, :]
        
    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.U @ self.sigma @ self.VT
        return X_transformed
    
    def transform(self, X):
        X_transformed = self.U @ self.sigma @ self.VT
        return X_transformed

# plot some images from numpy arrays
def plotImages(img_arrays, n_images):
  if img_arrays.shape[0] >= n_images:
    fig, axes = plt.subplots(1, n_images, figsize = (15,5))
  else:
    fig, axes = plt.subplots(1, img_arrays.shape[0], figsize = (15,5))    
  axes = axes.flatten()
  fig.suptitle('Plotting '+str(n_images)+ ' images', fontsize=16)
  for img, ax in zip(img_arrays, axes):
    ax.imshow(img, cmap = 'gray')
    ax.axis('off')
  plt.tight_layout()
  plt.show()


def read_one_image(path):
    img = io.imread(path, as_gray = True)
    img_resized = resize(img, output_shape = (256, 256))
    img_scaled = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized)) # image scaling 
    return img_scaled

def read_some_images(path_list):
    imgs = []
    for path in path_list:
        img = read_one_image(path)
        imgs.append(img)
    return np.array(imgs)

