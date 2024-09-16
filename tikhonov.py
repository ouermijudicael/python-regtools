import numpy as np

def tikhonov(U, s, V, b, lambdas, x_0=None):
    # Tikhonov regularization.

    # Parameters:
    # U : numpy.ndarray
    #     Left singular vectors from the SVD.
    # s : numpy.ndarray or tuple of numpy.ndarray
    #     Singular values from the SVD or GSVD.
    # V : numpy.ndarray
    #     Right singular vectors from the SVD.
    # b : numpy.ndarray
    #     Right-hand side vector.
    # lambdas : numpy.ndarray or list
    #     Regularization parameter(s). Can be a scalar or a list.
    # x_0 : numpy.ndarray, optional
    #     Initial estimate (default is None, equivalent to zero).

    # Returns:
    # x_lambda : numpy.ndarray
    #     Tikhonov regularized solution.
    # rho : numpy.ndarray
    #     Residual norms.
    # eta : numpy.ndarray
    #     Solution norms or seminorms.
    
    lambdas = np.array(lambdas)
    if np.any(lambdas < 0):
        raise ValueError('Illegal regularization parameter lambda')

    m, n = U.shape[0], V.shape[0]
    p, ps = s.shape if isinstance(s, np.ndarray) and len(s.shape) > 1 else (len(s), 1)
    
    beta = U[:, :p].T @ b
    zeta = s[:, 0] * beta if ps > 1 else s * beta
    
    if len(lambdas.shape) == 0:
        lambdas = np.array([lambdas])
    ll = len(lambdas)
    
    x_lambda = np.zeros((n, ll))
    rho = np.zeros(ll)
    eta = np.zeros(ll)

    # Treat each lambda separately
    if ps == 1:
        # The standard-form case
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
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :p] @ beta + U[:, p:].T @ b)**2)
    
    elif m >= n:
        # The overdetermined or square general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        if x_0 is not None:
            omega = np.linalg.solve(V, x_0)[:p]
        if p == n:
            x0 = np.zeros(n)
        else:
            x0 = V[:, p:n] @ U[:, p:n].T @ b
        for i in range(ll):
            if x_0 is None:
                xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            else:
                xi = (zeta + lambdas[i]**2 * s[:, 1]**2 * omega) / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm((beta - s[:, 0] * omega) / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :p] @ beta + U[:, p:].T @ b)**2)
    
    else:
        # The underdetermined general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        if x_0 is not None:
            raise ValueError('x_0 not allowed in the underdetermined case')
        if p == m:
            x0 = np.zeros(n)
        else:
            x0 = V[:, p:m] @ U[:, p:m].T @ b
        for i in range(ll):
            xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
            x_lambda[:, i] = V[:, :p] @ xi + x0
            rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)

    return x_lambda, rho, eta
