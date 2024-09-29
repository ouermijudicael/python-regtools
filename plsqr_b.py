def plsqr_b(A, L, W, b, k, reorth=0, sm=None):
# PLSQR_B "Precond." version of the LSQR Lanczos bidiagonalization algorithm.
# 
# [X,rho,eta,F] = plsqr_b(A,L,W,b,k,reorth,sm)
#
# Performs k steps of the `preconditioned' LSQR Lanczos
# bidiagonalization algorithm applied to the system
#    min || (A*L_p) x - b || ,
# where L_p is the A-weighted generalized inverse of L.  Notice
# that the matrix W holding a basis for the null space of L must
# also be specified.
#
# The routine returns all k solutions, stored as columns of
# the matrix X.  The solution seminorm and the residual norm are
# returned in eta and rho, respectively.
#
# If the generalized singular values sm of (A,L) are also provided,
#  then glsqr computes the filter factors associated with each step
# and stores them columnwise in the matrix F.
#
# Reorthogonalization is controlled by means of reorth:
#    reorth = 0 : no reorthogonalization (default),
#    reorth = 1 : reorthogonalization by means of MGS

# References: C. C. Paige & M. A. Saunders, "LSQR: an algorithm for
# sparse linear equations and sparse least squares", ACM Trans.
# Math. Software 8 (1982), 43-71.
# P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems.
# Numerical Aspects of Linear Inversion", SIAM, Philadelphia, 1997.

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
    X = np.zeros((n, k))
    pp, n1 = L.shape
    if n1 != n or m < n or n < pp:
        raise ValueError('Incorrect dimensions of A and L')
    
    if reorth == 0:
        UV = 0
    elif reorth == 1:
        if k >= n:
            raise ValueError('No. of iterations must satisfy k < n')
        U = np.zeros((m, k))
        V = np.zeros((pp, k))
        UV = 1
    else:
        raise ValueError('Illegal reorth')

    eta = np.zeros(k)
    rho = np.zeros(k)
    c2 = -1
    s2 = 0
    xnorm = 0
    z = 0

    if sm is not None:
        ls = len(sm)
        F = np.zeros((ls, k))
        Fv = np.zeros(ls)
        Fw = np.zeros(ls)
        s = (sm[:, 0] / sm[:, 1])**2

    # Prepare for computations with L_p.
    NAA, x_0 = pinit(W, A, b)

    # By subtracting the component A*x_0 from b we insure that
    # the corrent residual norms are computed.
    b = b - A @ x_0

    # Prepare for LSQR iteration.
    v = np.zeros(pp)
    x = v
    beta = np.linalg.norm(b)
    if beta == 0:
        raise ValueError('Right-hand side must be nonzero')
    u = b / beta
    if UV:
        U[:, 0] = u
    r = A.T @ u
    alpha = np.linalg.norm(r)
    v = r / alpha
    if UV:
        V[:, 0] = v
    phi_bar = beta
    rho_bar = alpha
    w = v
    if sm is not None:
        Fv = s / (alpha * beta)
        Fw = Fv

    # Perform Lanczos bidiagonalization with/without reorthogonalization.
    for i in range(1, k + 1):
        alpha_old = alpha
        beta_old = beta

        p = A @ v - alpha * u
        if reorth == 0:
            beta = np.linalg.norm(p)
            u = p / beta
        else:
            for j in range(i):
                p -= (U[:, j].T @ p) * U[:, j]
            beta = np.linalg.norm(p)
            u = p / beta

        # Compute L_p^T * A^T * u- beat * v.
        r = ltsolve(L, A.T@u, W, NAA) - beta * v
        if reorth == 0:
            alpha = np.linalg.norm(r)
            v = r / alpha
        else:
            for j in range(i):
                r -= (V[:, j].T @ r) * V[:, j]
            alpha = np.linalg.norm(r)
            v = r / alpha

        # Store U and V if necessary. 
        if UV:
            U[:, i] = u
            V[:, i] = v

        # construc and apply orthogonal transformation.
        rrho = np.linalg.norm([rho_bar, beta])
        c1 = rho_bar / rrho
        s1 = beta / rrho
        theta = s1 * alpha
        rho_bar = -c1 * alpha
        phi = c1 * phi_bar
        phi_bar = s1 * phi_bar

        delta = s2 * rrho
        gamma_bar = -c2 * rrho
        rhs = phi - delta * z
        z_bar = rhs / gamma_bar
        eta[i - 1] = np.linalg.norm([xnorm, z_bar])
        gamma = np.linalg.norm([gamma_bar, theta])
        c2 = gamma_bar / gamma
        s2 = theta / gamma
        z = rhs / gamma
        xnorm = np.linalg.norm([xnorm, z])
        rho[i - 1] = np.abs(phi_bar)

        # Compute filter factors, if required.
        if sm is not None:
            if i == 1:
                Fv_old = Fv
                Fv = Fv * (s - beta**2 - alpha_old**2) / (alpha * beta)
                F[:, i - 1] = (phi / rrho) * Fw
            else:
                tmp = Fv
                Fv = (Fv * (s - beta**2 - alpha_old**2) - Fv_old * alpha_old * beta_old) / (alpha * beta)
                Fv_old = tmp
                F[:, i - 1] = F[:, i - 2] + (phi / rrho) * Fw
            if i > 2:
                f = np.where((np.abs(F[:, i - 2] - 1) < fudge_thr) & (np.abs(F[:, i - 3] - 1) < fudge_thr))[0]
                if len(f) > 0:
                    F[f, i - 1] = np.ones(len(f))

        # Update solution.
        x = x + (phi / rrho) * w
        w = v - (theta / rrho) * w
        X[:, i - 1] = lsolve(L, x, W, NAA) + x_0

    return X, rho, eta, F

