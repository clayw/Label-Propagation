#!/usr/bin/env python2

import numpy as np

sigma = 26

two_sigma_sq = 2 * sigma ** 2

eps = np.finfo(np.double).eps

def gaussian_kernel(x1, x2):
    """
    computes the gaussian kernel between two input vectors
    """
    return np.exp( -np.linalg.norm(x1 - x2) ** 2 / two_sigma_sq )

def compute_affinity_matrix(X, diagonal=1):
    """
    affinity matrix from input matrix (representing a graph)
    """
    height = X.shape[0]
    aff_mat = np.zeros((height,height)) # square matrix
    
    for i in xrange(height):
        aff_mat[i,i] = diagonal
        for j in xrange(i+1, height):
            aff = gaussian_kernel(X[i], X[j])
            aff_mat[i,j] = aff
            aff_mat[j,i] = aff

    return aff_mat

def not_converged(y, y_hat, threshold=1e-3):
    cvg = sum(sum(abs(np.asarray(y-y_hat))))
    return cvg > threshold

class LabelPropagation(object):
    def __init__(self, max_iters=1000, convergence_threshold=1e-3):
        self.max_iters, self.convergence_threshold = max_iters, convergence_threshold

    def predict(self, X):
        return [np.argmax(self.predict_proba(x)) for x in X]

    def predict_proba(self, x):
        s = 0
        for xj, yj in zip(self.X, self.Y):
            Wx = gaussian_kernel(xj, x)
            s += Wx * yj / (Wx + eps)
        return s

    def fit(self, X, Y):
        """
        Fit a semi-supervised label propagation model based on input data 
        matrix X and corresponding label matrix Y. 


        @param X numpy matrix where Xi is represents the ith datapoint

        @param Y numpy matrix where Yi corresponds to the label of Xi


        """

#        print X
#        print Y

        affinity_matrix = compute_affinity_matrix(X)
        degree_matrix = map(sum, affinity_matrix) * np.identity(affinity_matrix.shape[0])
        deg_inv = np.linalg.inv(degree_matrix)
        self.affinity_matrix = affinity_matrix
#        print affinity_matrix
#        print deg_inv

        aff_ideg = deg_inv * np.matrix(affinity_matrix)

        self.Y, self.X = Y, X

        lbls = Y.shape[0]

        y_hat_init = np.zeros((X.shape[0], Y.shape[1]))
        y_hat_init[:lbls] = Y

        y_hat = y_hat_init
        y_hat_n = aff_ideg * y_hat

        max_iters = self.max_iters
        while not_converged(y_hat, y_hat_n, self.convergence_threshold) and max_iters > 0:
            y_hat = y_hat_n
            y_hat_n = aff_ideg * y_hat
            y_hat_n[:lbls] = Y
            max_iters -= 1

        self.Y = y_hat_n
        return self

class LabelSpreading(object):
    def __init__(self, alpha=0.5, max_iters=1000, convergence_threshold=1e-3):
        self.alpha, self.max_iters, self.convergence_threshold = alpha, max_iters, convergence_threshold

    def predict(self, X):
        return [np.argmax(self.predict_proba(x)) for x in X]

    def predict_proba(self, x):
        s = 0
        for xj, yj in zip(self.X, self.Y):
            Wx = gaussian_kernel(xj, x)
            s += Wx * yj / (Wx + eps)
        return s

    def fit(self, X, Y):
        """
        Fit a semi-supervised label propagation model based on input data 
        matrix X and corresponding label matrix Y. 


        @param X numpy matrix where Xi is represents the ith datapoint

        @param Y numpy matrix where Yi corresponds to the label of Xi


        """


        affinity_matrix = compute_affinity_matrix(X, diagonal=0)
        degree_matrix = map(sum, affinity_matrix) * np.identity(affinity_matrix.shape[0])
        deg_invsq = np.sqrt(np.linalg.inv(degree_matrix))

        laplacian = deg_invsq * np.matrix(affinity_matrix) * deg_invsq

        self.Y, self.X = Y, X

        lbls = Y.shape[0]

        y_hat_init = np.zeros((X.shape[0], Y.shape[1]))
        y_hat_init[:lbls] = Y

        y_hat = y_hat_init
        y_hat_n = self.alpha * laplacian * y_hat + (1 - self.alpha) * y_hat_init

        max_iters = self.max_iters
        while not_converged(y_hat, y_hat_n, self.convergence_threshold) and max_iters > 0:
            y_hat = y_hat_n
            y_hat_n = self.alpha * laplacian * y_hat + (1 - self.alpha) * y_hat_init
            y_hat_n[:lbls] = Y
            max_iters -= 1

        self.Y = y_hat_n
        return self
        

