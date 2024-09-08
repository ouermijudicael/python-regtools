import numpy as np

def lcfun(lambda_, s, beta, xi, fifth=None):
    """
    Auxiliary routine for l_corner; computes the NEGATIVE of the curvature.
    Note: lambda may be a vector. This is a translation of the MATLAB code provided.
    """

    # Initialization
    phi = np.zeros_like(lambda_)
    dphi = np.zeros_like(lambda_)
    psi = np.zeros_like(lambda_)
    dpsi = np.zeros_like(lambda_)
    eta = np.zeros_like(lambda_)
    rho = np.zeros_like(lambda_)
    
    # Handle least squares residual
    if len(beta) > len(s):
        LS = True
        rhoLS2 = beta[-1]**2
        beta = beta[:-1]
    else:
        LS = False

    # Compute intermediate quantities
    tmp = np.array(lambda_)
    if(len(tmp.shape) == 0):
        if(fifth is None):
            f = (s**2) / (s**2 + lambda_**2)
        else:
            f = s / (s + lambda_)
        cf = 1 - f
        eta = np.linalg.norm(f * xi)
        rho = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lambda_
        f2 = -f1 * (3 - 4 * f) / lambda_
        phi = np.sum(f * f1 * np.abs(xi)**2)
        psi = np.sum(cf * f1 * np.abs(beta)**2)
        dphi = np.sum((f1**2 + f * f2) * np.abs(xi)**2)
        dpsi = np.sum((-f1**2 + cf * f2) * np.abs(beta)**2)
    else:
        for i in range(len(lambda_)):
            if fifth is None:
                f = (s**2) / (s**2 + lambda_[i]**2)
            else:
                f = s / (s + lambda_[i])
            
            cf = 1 - f
            eta[i] = np.linalg.norm(f * xi)
            rho[i] = np.linalg.norm(cf * beta)
            f1 = -2 * f * cf / lambda_[i]
            f2 = -f1 * (3 - 4 * f) / lambda_[i]
            phi[i] = np.sum(f * f1 * np.abs(xi)**2)
            psi[i] = np.sum(cf * f1 * np.abs(beta)**2)
            dphi[i] = np.sum((f1**2 + f * f2) * np.abs(xi)**2)
            dpsi[i] = np.sum((-f1**2 + cf * f2) * np.abs(beta)**2)
    
    # Least squares residual handling
    if LS:
        rho = np.sqrt(rho**2 + rhoLS2)

    # Compute first and second derivatives of eta and rho with respect to lambda
    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)

    # Convert to derivatives of log(eta) and log(rho)
    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = ddeta / eta - dlogeta**2
    ddlogrho = ddrho / rho - dlogrho**2

    # Compute the curvature (g)
    g = -(dlogrho * ddlogeta - ddlogrho * dlogeta) / (dlogrho**2 + dlogeta**2)**1.5
    
    return g
