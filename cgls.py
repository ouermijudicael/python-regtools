def cgls(A, b, k, reorth=0, s=None):
# CGLS Conjugate gradient algorithm applied implicitly to the normal equations.

# [X,rho,eta,F] = cgls(A,b,k,reorth,s)

# Performs k steps of the conjugate gradient algorithm applied
# implicitly to the normal equations A'*A*x = A'*b.

# The routine returns all k solutions, stored as columns of
# the matrix X.  The corresponding solution and residual norms
# are returned in the vectors eta and rho, respectively.

# If the singular values s are also provided, cgls computes the
# filter factors associated with each step and stores them
# columnwise in the matrix F.

# Reorthogonalization of the normal equation residual vectors
# A'*(A*X(:,i)-b) is controlled by means of reorth:
#    reorth = 0 : no reorthogonalization (default),
#    reorth = 1 : reorthogonalization by means of MGS.

# References: A. Bjorck, "Numerical Methods for Least Squares Problems",
# SIAM, Philadelphia, 1996.
# C. R. Vogel, "Solving ill-conditioned linear systems using the
# conjugate gradient method", Report, Dept. of Mathematical
# Sciences, Montana State University, 1987.


    import numpy as np

    # The fudge threshold is used to prevent filter factors from exploding.
    fudge_thr = 1e-4

    # Initialization.
    if k < 1:
        raise ValueError('Number of steps k must be positive')

    if reorth < 0 or reorth > 1:
        raise ValueError('Illegal reorth')
    
    if (s is None):
        raise ValueError('Too few input arguments')
    
    m, n = A.shape
    X = np.zeros((n, k))

    if reorth == 1:
        ATr = np.zeros((n, k + 1))

    eta = np.zeros(k)
    rho = np.zeros(k)

    F = np.zeros((n, k))
    Fd = np.zeros(n)
    s2 = s**2

    # Prepare for CG iteration.
    x = np.zeros(n)
    d = A.T @ b
    r = b
    normr2 = d @ d
    if reorth == 1:
        ATr[:, 0] = d / np.linalg.norm(d)

    # Iterate.
    for j in range(k):  
        # Update x and r vectors.
        Ad = A @ d
        alpha = normr2 / (Ad.T @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        s = A.T @ r

        # Reorthogonalize s to previous s-vectors, if required.
        if reorth == 1:
            for i in range(j):
                s = s - (ATr[:, i] @ s) * ATr[:, i]
            ATr[:, j + 1] = s / np.linalg.norm(s)

        # Update d vector.
        normr2_new = s @ s
        beta = normr2_new / normr2
        normr2 = normr2_new
        d = s + beta * d
        X[:, j] = x

        # Compute norms, if required.
        rho[j] = np.linalg.norm(r)
        eta[j] = np.linalg.norm(x)

        # Compute filter factors, if required.
        if s is not None:
            if j == 1:
                F[:, 0] = alpha * s2
                Fd = s2 - s2 * F[:, 0] + beta * s2
            else:
                F[:, j] = F[:, j - 1] + alpha * Fd
                Fd = s2 - s2 * F[:, j] + beta * Fd
            if j > 2:
                f = np.where((np.abs(F[:, j-1] - 1) < fudge_thr) & (np.abs(F[:, j-2] - 1) < fudge_thr))[0]
                if len(f) > 0:
                    F[f, j] = np.ones(len(f)) 

    return X, rho, eta, F
