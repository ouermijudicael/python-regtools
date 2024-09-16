import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from ncpfun import ncpfun

def ncp(U, s, b, method='Tikh'):

# %NCP Plot the NCPs and find the one closest to a straight line.
# %
# % [reg_min,G,reg_param] = ncp(U,s,b,method)
# % [reg_min,G,reg_param] = ncp(U,sm,b,method)  ,  sm = [sigma,mu]
# %
# % Plots the normalized cumulative priodograms (NCPs) for the residual
# % vectors A*x - b.  The following methods are allowed:
# %    method = 'Tikh' : Tikhonov regularization
# %    method = 'tsvd' : truncated SVD or GSVD
# %    method = 'dsvd' : damped SVD or GSVD
# % If method is not specified, 'Tikh' is default.  U and s, or U and sm,
# % must be computed by the functions csvd and cgsvd, respectively.
# %
# % The NCP closest to a straight line is identified and the corresponding
# % regularization parameter reg_min is returned.  Moreover, dist holds the
# % distances to the straight line, and reg_param are the corresponding
# % regularization parameters.

# % Per Christian Hansen, DTU Compute, Jan. 4, 2008.

# % Reference: P. C. Hansen, M. Kilmer & R. H. Kjeldsen, "Exploiting
# % residual information in the parameter choice for discrete ill-posed
# % problems", BIT 46 (2006), 41-59.

    # Set defaults
    npoints = 200
    nNCPs = 20
    smin_ratio = 16 * np.finfo(float).eps

    # Initialization
    m = U.shape[0]
    p = s.shape[0]
    if len(s.shape) ==1:
        ps = 1
    else:
        ps = s.shape[1]

    beta = U.T @ b
    if ps == 2:
        s = s[::-1, 0] / s[::-1, 1]
        beta = beta[::-1]

    if method.lower().startswith('tikh'):
        # Vector of regularization parameters
        reg_param = np.zeros(npoints)
        reg_param[-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1]) ** (1 / (npoints - 1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]

        # Vector of distances to straight line
        dists = np.zeros(npoints)
        q = m // 2 if np.isreal(beta).all() else m - 1
        cp = np.zeros((q, npoints))
        for i in range(npoints):
            dists[i], cp[:, i] = ncpfun(reg_param[i], s, beta[:p], U[:, :p])

        # Plot selected NCPs
        stp = npoints // nNCPs
        plt.plot(cp[:, ::stp])
        # plt.hold(True)

        # Find minimum
        minG, minGi = min(dists), np.argmin(dists)
        reg_min = fminbound(lambda x: ncpfun(x, s, beta[:p], U[:, :p])[0],
                            reg_param[min(minGi + 1, npoints - 1)],
                            reg_param[max(minGi - 1, 0)],
                            disp=0)
        dist, cp = ncpfun(reg_min, s, beta[:p], U[:, :p])
        plt.plot(cp, '-r', linewidth=3)
        # plt.hold(False)
        plt.title(f'Selected NCPs. Most white for λ = {reg_min}')

    elif method.lower().startswith('tsvd'):
        # Matrix of residual vectors
        R = np.zeros((m, p-1))
        R[:, p-2] = beta[p-1] * U[:, p-1]
        for i in range(p-2, 0, -1):
            R[:, i-1] = R[:, i] + beta[i] * U[:, i]

        # Compute NCPs and distances
        q = m // 2 if np.isreal(R).all() else m - 1
        D = np.abs(np.fft.fft(R)) ** 2
        D = D[1:q+1, :]
        v = np.arange(1, q+1) / q
        cp = np.zeros((q, p-1))
        dist = np.zeros(p-1)
        for k in range(p-1):
            cp[:, k] = np.cumsum(D[:, k]) / np.sum(D[:, k])
            dist[k] = np.linalg.norm(cp[:, k] - v)

        # Locate minimum and plot
        dist_min, reg_min = min(dist), np.argmin(dist)
        plt.plot(cp)
        # plt.hold(True)
        plt.plot(np.arange(1, q+1), cp[:, reg_min], '-r', linewidth=3)
        # plt.hold(False)
        plt.title(f'Most white for k = {reg_min}')

        reg_param = np.arange(1, p)

    elif method.lower().startswith('dsvd'):
        # Vector of regularization parameters
        reg_param = np.zeros(npoints)
        reg_param[-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1]) ** (1 / (npoints - 1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]

        # Vector of distances to straight line
        dists = np.zeros(npoints)
        q = m // 2 if np.isreal(beta).all() else m - 1
        cp = np.zeros((q, npoints))
        for i in range(npoints):
            dists[i], cp[:, i] = ncpfun(reg_param[i], s, beta[:p], U[:, :p], 1)

        # Plot selected NCPs
        stp = npoints // nNCPs
        plt.plot(cp[:, ::stp])
        # plt.hold(True)

        # Find minimum, if requested
        minG, minGi = min(dists), np.argmin(dists)
        reg_min = fminbound(lambda x: ncpfun(x, s, beta[:p], U[:, :p], 1)[0],
                            reg_param[min(minGi + 1, npoints - 1)],
                            reg_param[max(minGi - 1, 0)],
                            disp=0)
        dist, cp = ncpfun(reg_min, s, beta[:p], U[:, :p])
        plt.plot(cp, '-r', linewidth=3)
        # plt.hold(False)
        plt.title(f'Selected NCPs. Most white for λ = {reg_min}')

    else:
        raise ValueError('Illegal method')

    return reg_min, dist, reg_param
