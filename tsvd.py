import numpy as np

def tsvd(U, s, V, b, k):
    # Truncated SVD regularization.

    # Parameters:
    # U : numpy.ndarray
    #     Left singular vectors from the SVD.
    # s : numpy.ndarray
    #     Singular values from the SVD.
    # V : numpy.ndarray
    #     Right singular vectors from the SVD.
    # b : numpy.ndarray
    #     Right-hand side vector.
    # k : int or list of int
    #     Truncation parameter. If k is a list, the output will be a matrix.

    # Returns:
    # x_k : numpy.ndarray
    #     Truncated SVD solution.
    # rho : numpy.ndarray
    #     Residual norms.
    # eta : numpy.ndarray
    #     Solution norms.

    # Initialization
    n, p = V.shape
    # print('k:', k)
    if isinstance(k, int):
        # print('k is int')
        k = [k]
    lk = len(k)
    print('lk:', lk, 'k:', k)
    if min(k) < 0 or max(k) > p:
        raise ValueError('Illegal truncation parameter k')

    x_k = np.zeros((n, lk))
    eta = np.zeros(lk)
    rho = np.zeros(lk)
    beta = U.T @ b
    xi = beta / s

    # Treat each k separately
    for j in range(lk):
        i = k[j]
        if i > 0:
            x_k[:, j] = V[:, :i] @ xi[:i]
            eta[j] = np.linalg.norm(xi[:i])
            rho[j] = np.linalg.norm(beta[i:p])

    if len(b) > p:
        rho = np.sqrt(rho**2 + np.linalg.norm(b - U @ beta)**2)

    return x_k, rho, eta
