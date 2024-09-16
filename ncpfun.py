import numpy as np

def ncpfun(lambda_, s, beta, U, dsvd=False):
    """
    Auxiliary routine for NCP (Normalized Cumulative Periodogram).
    
    Parameters:
    lambda_ : The regularization parameter.
    s : Singular values (or diagonal elements in case of dsvd).
    beta : U.T @ b (transformed right-hand side vector).
    U : Matrix from SVD.
    dsvd : Boolean flag indicating whether dsvd is used (default False).
    
    Returns:
    dist : The distance between the cumulative periodogram and a uniform distribution.
    cp : The cumulative periodogram.
    """

    # Compute the regularization filter.
    if not dsvd:
        f = (lambda_**2) / (s**2 + lambda_**2)
    else:
        f = lambda_ / (s + lambda_)

    # Compute r = U * (f .* beta)
    r = U @ (f * beta)
    m = len(r)

    # Compute the FFT-based periodogram.
    if np.isrealobj(beta):
        q = m // 2
    else:
        q = m - 1

    D = np.abs(np.fft.fft(r))**2
    D = D[1:q+1]  # MATLAB 1-based indexing is converted to Python's 0-based
    v = np.arange(1, q+1) / q

    # Cumulative sum and normalized cumulative periodogram.
    cp = np.cumsum(D) / np.sum(D)

    # Compute the distance.
    dist = np.linalg.norm(cp - v)

    return dist, cp
