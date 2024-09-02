import numpy as np

def lsqr_b(A, b, k, reorth=0, s=None):
    """
    Solution of least squares problems by Lanczos bidiagonalization.

    Parameters:
    A : matrix
    b : right-hand side vector
    k : number of steps
    reorth : reorthogonalization flag (default is 0)
    s : singular values (optional)

    Returns:
    X : matrix of solutions
    rho : residual norms
    eta : solution norms
    F : filter factors (if s is provided)
    """
    fudge_thr = 1e-4

    if k < 1:
        raise ValueError('Number of steps k must be positive')
    if reorth not in [0, 1]:
        raise ValueError('Illegal reorth')

    m, n = A.shape
    X = np.zeros((n, k))
    UV = 0
    if reorth == 1:
        U = np.zeros((m, k + 1))
        V = np.zeros((n, k + 1))
        UV = 1
        if k >= n:
            raise ValueError('No. of iterations must satisfy k < n')

    eta = np.zeros(k)
    rho = np.zeros(k)
    c2 = -1
    s2 = 0
    xnorm = 0
    z = 0

    if s is not None:
        ls = len(s)
        F = np.zeros((ls, k))
        Fv = np.zeros(ls)
        Fw = np.zeros(ls)
        s = s**2

    v = np.zeros(n)
    x = np.zeros(n)
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
    if s is not None:
        Fv = s / (alpha * beta)
        Fw = Fv

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

        r = A.T @ u - beta * v
        if reorth == 0:
            alpha = np.linalg.norm(r)
            v = r / alpha
        else:
            for j in range(i):
                r -= (V[:, j].T @ r) * V[:, j]
            alpha = np.linalg.norm(r)
            v = r / alpha

        if UV:
            U[:, i] = u
            V[:, i] = v

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
        rho[i - 1] = abs(phi_bar)

        if s is not None:
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
            Fw = Fv - (theta / rrho) * Fw

        x += (phi / rrho) * w
        w = v - (theta / rrho) * w
        X[:, i - 1] = x

    if s is not None:
        return X, rho, eta, F
    else:
        return X, rho, eta