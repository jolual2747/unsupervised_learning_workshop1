import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from numpy.linalg import norm
from random import normalvariate
from math import sqrt

# PCA from scratch
class PCA:
    def __init__(self, n_components):
        """Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : 'eigen'
        """
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
        """
        SVD (Singular Value Decomposition): is a factorization of a matrix into 
        three matrices. U, S, VT.

        It's used as a data reduction method in machine learning
        Parameters
        ----------
        n_components: int
            number of sigular values to decompose
        """
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
    
class tsne:
    def __init__(self, n_components=2, perplexity=15.0, max_iter=50, momentum = 1.0, learning_rate=10,random_state=1234):
        """
        T-SNE: A t-Distributed Stochastic Neighbor Embedding implementation. Built based on https://github.com/nlml/tsne_raw
        It's a tool to visualize high-dimensional data. It converts
        similarities between data points to joint probabilities and tries
        to minimize the Kullback-Leibler divergence between the joint
        probabilities of the low-dimensional embedding and the
        high-dimensional data. 
        Parameters:
        ----------
        max_iter : int, default 300
        perplexity : float, default 15.0
        n_components : int, default 2
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter    
        self.momentum = momentum
        self.lr = learning_rate
        self.seed=random_state

    def fit(self, X):
        self.Y = np.random.RandomState(self.seed).normal(0., 0.0001, [X.shape[0], self.n_components])
        self.Q, self.distances = self.q_tsne()
        self.P=self.p_joint(X)

    def transform(self, X):
        if self.momentum:
            Y_m2 = self.Y.copy()
            Y_m1 = self.Y.copy()

        for i in range(self.max_iter):

            # Get Q and distances (distances only used for t-SNE)
            self.Q, self.distances = self.q_tsne()
            # Estimate gradients with respect to Y
            grads = self.tsne_grad()

            # Update Y
            self.Y = self.Y - self.lr * grads

            if self.momentum:  # Add momentum
                self.Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = self.Y.copy()
        return self.Y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def p_joint(self, X):
        """Given a data matrix X, gives joint probabilities matrix.
        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        def p_conditional_to_joint(P):
            """Given conditional probabilities matrix P, return
            approximation of joint distribution probabilities."""
            return (P + P.T) / (2. * P.shape[0])
        def calc_prob_matrix(distances, sigmas=None, zero_index=None):
            """Convert a distances matrix to a matrix of probabilities."""
            if sigmas is not None:
                two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                return self.softmax(distances / two_sig_sq, zero_index=zero_index)
            else:
                return self.softmax(distances, zero_index=zero_index)
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas()
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        self.P = p_conditional_to_joint(p_conditional)
        return self.P


    def find_optimal_sigmas(self):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                        lower=1e-20, upper=1000.):
            """Perform a binary search over input values to eval_fn.
            # Arguments
                eval_fn: Function that we are optimising over.
                target: Target value we want the function to output.
                tol: Float, once our guess is this close to target, stop.
                max_iter: Integer, maximum num. iterations to search for.
                lower: Float, lower bound of search range.
                upper: Float, upper bound of search range.
            # Returns:
                Float, best input value to function found during search.
            """
            for i in range(max_iter):
                guess = (lower + upper) / 2.
                val = eval_fn(guess)
                if val > target:
                    upper = guess
                else:
                    lower = guess
                if np.abs(val - target) <= tol:
                    break
            return guess
        def calc_perplexity(prob_matrix):
            """Calculate the perplexity of each row
            of a matrix of probabilities."""
            entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
            perplexity = 2 ** entropy
            return perplexity

        def perplexity(distances, sigmas, zero_index):
            """Wrapper function for quick calculation of
            perplexity over a distance matrix."""
            def calc_prob_matrix(distances, sigmas=None, zero_index=None):
                """Convert a distances matrix to a matrix of probabilities."""
                if sigmas is not None:
                    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                    return self.softmax(distances / two_sig_sq, zero_index=zero_index)
                else:
                    return self.softmax(distances, zero_index=zero_index)
            return calc_perplexity(
                calc_prob_matrix(distances, sigmas, zero_index))
        sigmas = []
        # For each row of the matrix (each point in our dataset)
        for i in range(self.distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: \
                perplexity(self.distances[i:i+1, :], np.array(sigma), i)
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = binary_search(eval_fn, self.perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)


    def tsne_grad(self):
        """t-SNE: Estimate the gradient of the cost with respect to Y."""
        pq_diff = self.P - self.Q  # NxN matrix
        pq_expanded = np.expand_dims(pq_diff, 2)  # NxNx1
        y_diffs = np.expand_dims(self.Y, 1) - np.expand_dims(self.Y, 0)  # NxNx2
        # Expand our distances matrix so can multiply by y_diffs
        distances_expanded = np.expand_dims(self.distances, 2)  # NxNx1
        # Weight this (NxNx2) by distances matrix (NxNx1)
        y_diffs_wt = y_diffs * distances_expanded  # NxNx2
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
        return grad

    def neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X
        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D


    def softmax(self, X, diag_zero=True, zero_index=None):
        """Compute softmax values for each row of matrix X."""

        # Subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        # We usually want diagonal probailities to be 0.
        if zero_index is None:
            if diag_zero:
                np.fill_diagonal(e_x, 0.)
        else:
            e_x[:, zero_index] = 0.

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability

        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def q_tsne(self):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self.neg_squared_euc_dists(self.Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances


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

def l2_distance(X):
    sum_X = np.sum(X * X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X