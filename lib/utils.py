import numpy as np
from scipy.linalg import svd
import logging
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

# PCA from scratch
class PCA:
    def __init__(self, n_components, solver = 'svd') -> None:
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X, y = None):
        self.mean = np.mean(X, axis = 1)
        self._decompose(X)
    
    def _decompose(self, X):
        # mean centering
        X = X.copy()
        X -= self.mean

        if self.solver == 'svd':
            _, s, Vh = svd(X, full_matrices = True)
        elif self.solver == 'eigen':
            s, Vh = np.linalg.eig(np.cov(X.T))
            Vh = Vh.T
        
        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        logging.info('Explained variance ratio: %s' % (variance_ratio[0:self.n_components]))
        self.components = Vh[0: self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def _predict(self, X=None):
        self.transform(X)

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

