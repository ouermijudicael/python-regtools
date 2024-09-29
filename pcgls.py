def pcgls(A, L, W, b, k, reorth, sm = None):
# %PCGLS "Precond." conjugate gradients appl. implicitly to normal equations.
# % [X,rho,eta,F] = pcgls(A,L,W,b,k,reorth,sm)
# %
# % Performs k steps of the `preconditioned' conjugate gradient
# % algorithm applied implicitly to the normal equations
# %    (A*L_p)'*(A*L_p)*x = (A*L_p)'*b ,
# % where L_p is the A-weighted generalized inverse of L.  Notice that the
# % matrix W holding a basis for the null space of L must also be specified.
# %
# % The routine returns all k solutions, stored as columns of the matrix X.
# % The solution seminorm and residual norm are returned in eta and rho,
# % respectively.
# %
# % If the generalized singular values sm of (A,L) are also provided,
# % pcgls computes the filter factors associated with each step and
# % stores them columnwise in the matrix F.
# %
# % Reorthogonalization of the normal equation residual vectors
# % A'*(A*X(:,i)-b) is controlled by means of reorth:
# %    reorth = 0 : no reorthogonalization (default),
# %    reorth = 1 : reorthogonalization by means of MGS.

# % References: A. Bjorck, "Numerical Methods for Least Squares Problems",
# % SIAM, Philadelphia, 1996.
# % P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems.
# % Numerical Aspects of Linear Inversion", SIAM, Philadelphia, 1997.

    import numpy as np
    from pinit import pinit
    from ltsolve import ltsolve
    from lsolve import lsolve

    # The fudge threshold is used to prevent filter factors from exploding.
    fudge_thr = 1e-4

    # Initialization.
    if k < 1:
        raise ValueError('Number of steps k must be positive')
    
    if sm is None:
        raise ValueError('Too few input arguments')

    if reorth < 0 or reorth > 1:
        raise ValueError('Illegal reorth')
    
    m, n = A.shape
    p = L.shape[0]
    X = np.zeros((n, k))

    eta = np.zeros(k)
    rho = np.zeros(k)

    F = np.zeros((p, k))
    Fd = np.zeros(p)
    gamma = (sm[:, 0] / sm[:, 1])**2

    # Prepare for computations with L_p.
    NAA, x_0 = pinit(W, A, b)

    # Prepare for CG iteration.
    x = x_0
    r = b - A @ x_0
    s = A.T @ r
    q1 = ltsolve(L, s)
    q = lsolve(L, q1, W, NAA)
    z = q
    dq = s @ q

    if reorth == 1:
        Q1n = np.zeros((p, k))
        Q1n[:, 0] = q1 / np.linalg.norm(q1)

    # Iterate.
    for j in range(k):  
        # Update x and r vectors; compute q1.
        Az = A @ z
        alpha = dq / (Az.T @ Az)
        x = x + alpha * z
        r = r - alpha * Az
        s = A.T @ r
        q1 = ltsolve(L, s)

        # Reorthogonalize q1 to previous q1-vectors, if required.
        if reorth == 1:
            for i in range(j):
                q1 = q1 - (Q1n[:, i].T @ q1) * Q1n[:, i]
            Q1n[:, j] = q1 / np.linalg.norm(q1)

        # Update z vector.
        q = lsolve(L, q1, W, NAA)
        dq2 = s @ q
        beta = dq2 / dq
        dq = dq2
        z = q + beta * z
        X[:, j] = x
        rho[j] = np.linalg.norm(r)
        eta[j] = np.linalg.norm(x)

        # Compute filter factors, if required.
        if sm is not None:
            if j == 1:
                F[:, 0] = alpha * gamma
                Fd = gamma - gamma * F[:, 0] + beta * gamma
            else:
                F[:, j] = F[:, j - 1] + alpha * Fd
                Fd = gamma - gamma * F[:, j] + beta * Fd
            if j > 2:
                f = np.where((np.abs(F[:, j-1] - 1) < fudge_thr) & (np.abs(F[:, j-2] - 1) < fudge_thr))[0]
                if len(f) > 0:
                    F[f, j] = np.ones(f.size)

    return X, rho, eta, F
