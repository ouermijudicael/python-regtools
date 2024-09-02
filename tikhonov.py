import numpy as np

def tikhonov(U, s, V, b, lambdas, x_0=None):
    """
    Tikhonov regularization.

    Parameters:
    U, s, V : SVD components of matrix A
    b : right-hand side vector
    lambdas : regularization parameter(s)
    x_0 : initial estimate (optional)

    Returns:
    x_lambda : regularized solution(s)
    rho : residual norm(s)
    eta : solution norm(s)
    """
    if np.min(lambdas) < 0:
        raise ValueError('Illegal regularization parameter lambda')

    m, n = U.shape[0], V.shape[0]
    if(s.ndim == 1):
        p, ps = s.shape[0], 1
    else:
        p, ps = s.shape
    beta = U[:, :p].T @ b
    if( s.ndim == 1 ):
        zeta = s * beta
    else:
        zeta = s[:, 0] * beta
    ll = len(lambdas)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros(ll)
    eta = np.zeros(ll)

    if ps == 1:
        # Standard-form case
        if x_0 is not None:
            omega = V.T @ x_0
        for i in range(ll):
            if x_0 is None:
                x_lambda[:, i] = V[:, :p] @ (zeta / (s**2 + lambdas[i]**2))
                rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (s**2 + lambdas[i]**2))
            else:
                x_lambda[:, i] = V[:, :p] @ ((zeta + lambdas[i]**2 * omega) / (s**2 + lambdas[i]**2))
                rho[i] = lambdas[i]**2 * np.linalg.norm((beta - s * omega) / (s**2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(x_lambda[:, i])
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:n].T @ b]))**2)

    elif m >= n:
        # Overdetermined or square general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        if x_0 is not None:
            omega = np.linalg.solve(V, x_0)[:p]
        x0 = np.zeros(n) if p == n else V[:, p:n] @ U[:, p:n].T @ b
        for i in range(ll):
            if x_0 is None:
                xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 2]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            else:
                xi = (zeta + lambdas[i]**2 * (s[:, 2]**2) * omega) / (s[:, 0]**2 + lambdas[i]**2 * s[:, 2]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm((beta - s[:, 0] * omega) / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 2] * xi)
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:n].T @ b]))**2)

    else:
        # Underdetermined general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        if x_0 is not None:
            raise ValueError('x_0 not allowed')
        x0 = np.zeros(n) if p == m else V[:, p:m] @ U[:, p:m].T @ b
        for i in range(ll):
            xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 2]**2)
            x_lambda[:, i] = V[:, :p] @ xi + x0
            rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 2] * xi)

    return x_lambda, rho, eta