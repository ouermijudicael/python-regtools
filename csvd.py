import numpy as np

def csvd(A, tst=None):
    """
    Compact singular value decomposition.
    
    Parameters:
    A : array_like
        Input matrix.
    tst : str, optional
        If provided, the full U and V are returned.
    
    Returns:
    U : ndarray
        Left singular vectors.
    s : ndarray
        Singular values.
    V : ndarray
        Right singular vectors.
    """
    m, n = A.shape
    if tst is None:
        if m >= n:
            U, s, V = np.linalg.svd(A, full_matrices=False)
            # s = s[:, np.newaxis]
        else:
            V, s, U = np.linalg.svd(A.T, full_matrices=False)
            # s = np.diag(s)
            # s = s[:, np.newaxis]
    else:
        U, s, V = np.linalg.svd(A)
        # s = np.diag(s)
        # s = s[:, np.newaxis]
    
    return U, s, V