import numpy as np
from scipy.sparse import eye, lil_matrix

def get_l(n, d):
    """
    Computes the discrete approximation L to the derivative operator of order d on a regular grid with n points.
    Also computes W, an orthonormal basis for the null space of L.

    Parameters:
    n : int
        Number of grid points.
    d : int
        Order of the derivative (must be non-negative).

    Returns:
    L : scipy.sparse.lil_matrix
        The (n-d)-by-n discrete derivative operator.
    W : numpy.ndarray
        Orthonormal basis for the null space of L (if requested).
    """
    if d < 0:
        raise ValueError('Order d must be nonnegative')

    # Zero'th derivative case.
    if d == 0:
        L = eye(n, format='csr')
        W = np.zeros((n, 0))
        return L, W

    # Compute L.
    c = np.concatenate(([-1, 1], np.zeros(d - 1)))
    nd = n - d
    for i in range(2, d + 1):
        c = np.concatenate(([0], c[:d])) - np.concatenate((c[:d], [0]))

    L = lil_matrix((nd, n))
    for i in range(d + 1):
        L[np.arange(nd), np.arange(nd) + i] = c[i]

    # If required, compute the null vectors W via modified Gram-Schmidt.
    W = None
    if d > 0:
        W = np.zeros((n, d))
        W[:, 0] = np.ones(n)
        for i in range(1, d):
            W[:, i] = W[:, i - 1] * np.arange(1, n + 1)
        
        # Modified Gram-Schmidt orthonormalization
        for k in range(d):
            W[:, k] /= np.linalg.norm(W[:, k])
            for j in range(k + 1, d):
                W[:, j] -= np.dot(W[:, k], W[:, j]) * W[:, k]
    

    return L, W
