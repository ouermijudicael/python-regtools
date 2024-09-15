import numpy as np

def gcvfun(lmbda, s2, beta, delta0, mn, dsvd=None):
    """
    Auxiliary function for GCV.
    
    Parameters:
    lmbda : float
        Regularization parameter.
    s2 : array-like
        Squared singular values.
    beta : array-like
        Transformed right-hand side vector.
    delta0 : float
        Intrinsic residual.
    mn : int
        Degrees of freedom (m - n).
    dsvd : int, optional
        If provided, determines if dsvd is being used. Default is None.
    
    Returns:
    G : float
        GCV function value.
    """
    if dsvd is None:
        # Tikhonov regularization (1 - filter factors)
        f = (lmbda**2) / (s2 + lmbda**2)
    else:
        # Damped SVD
        f = lmbda / (s2 + lmbda)

    # Compute the GCV function value
    G = (np.linalg.norm(f * beta)**2 + delta0) / (mn + np.sum(f))**2
    
    return G
