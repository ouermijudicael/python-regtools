import numpy as np
from scipy.optimize import fminbound
from lcfun import lcfun
from corner import corner
from scipy.interpolate import splrep, splev, splder

# function [reg_c,rho_c,eta_c] = l_corner(rho,eta,reg_param,U,s,b,method,M)
def l_corner(rho, eta, reg_param, U=None, s=None, b=None, method='Tikh', M=None):
#  L_CORNER Locate the "corner" of the L-curve.
# 
#  [reg_c,rho_c,eta_c] =
#         l_corner(rho,eta,reg_param)
#         l_corner(rho,eta,reg_param,U,s,b,method,M)
#         l_corner(rho,eta,reg_param,U,sm,b,method,M) ,  sm = [sigma,mu]
# 
#  Locates the "corner" of the L-curve in log-log scale.
# 
#  It is assumed that corresponding values of || A x - b ||, || L x ||,
#  and the regularization parameter are stored in the arrays rho, eta,
#  and reg_param, respectively (such as the output from routine l_curve).
# 
#  If nargin = 3, then no particular method is assumed, and if
#  nargin = 2 then it is issumed that reg_param = 1:length(rho).
# 
#  If nargin >= 6, then the following methods are allowed:
#     method = 'Tikh'  : Tikhonov regularization
#     method = 'tsvd'  : truncated SVD or GSVD
#     method = 'dsvd'  : damped SVD or GSVD
#     method = 'mtsvd' : modified TSVD,
#  and if no method is specified, 'Tikh' is default.  If the Spline Toolbox
#  is not available, then only 'Tikh' and 'dsvd' can be used.
# 
#  An eighth argument M specifies an upper bound for eta, below which
#  the corner should be found.

#  Per Christian Hansen, DTU Compute, January 31, 2015.

    # Ensure that rho and eta are column vectors.
    rho = np.array(rho).flatten()
    eta = np.array(eta).flatten()

    if U is None or s is None or b is None:
        method = 'none'
        if reg_param is None:
            reg_param = np.arange(1, len(rho) + 1)



    # Set this logical variable to 1 (true) if the corner algorithm
    # should always be used, even if the Spline Toolbox is available.
    alwayscorner = 0

    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps  # Neglect singular values less than s_thr.

    # Set default parameters for treatment of discrete L-curve.
    deg   = 2;  # Degree of local smooting polynomial.
    q     = 2;  # Half-width of local smoothing interval.
    order = 4;  # Order of fitting 2-D spline curve.

    # Initialization.
    if (len(rho) < order):
        raise ValueError('Too few data points for L-curve analysis')

    # method = 'none'
    # print('reg_param:', reg_param)
    if (U is not None and s is not None and b is not None):
        p, ps = s.shape if len(s.shape) > 1 else (len(s), 1)
        m,n = U.shape
        beta = U.T @ b
        b0 = b - U @ beta
        if (ps == 2):
            s = s[p-1::-1, 0] / s[p-1::-1, 1]
            beta = beta[p-1::-1]
        xi = beta / s
        
        if (m > n):
            beta = np.append(beta, np.linalg.norm(b0))



    # Restrict the analysis of the L-curve according to M (if specified).
    if (U is not None and s is not None and b is not None and M is not None):
        index = np.where(eta < M)
        rho = rho[index]
        eta = eta[index]
        reg_param = reg_param[index]

    if(method == 'Tikh' or method == 'tikh'):
    
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.
        g = lcfun(reg_param, s, beta, xi)
    
        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lambda rp: lcfun(rp, s, beta, xi), 
                          reg_param[min(gi + 1, len(g))], reg_param[max(gi - 1, 0)])
        
        kappa_max = -lcfun(reg_c, s, beta, xi)

        if (kappa_max < 0):
            lr = len(rho)
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            f = (s ** 2) / (s ** 2 + reg_c ** 2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if (m > n):
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    elif (method == 'tsvd' or method == 'tgsv' or method == 'mtsv' or method == 'none'):
        # Use the adaptive pruning algorithm to find the corner, if the
        # Spline Toolbox is not available.
        if (not 'splines' in globals() or alwayscorner):
            reg_c = corner(rho, eta)
            reg_c = reg_c[0] 
            rho_c = rho[reg_c]
            eta_c = eta[reg_c]
            return reg_c, rho_c, eta_c
        
        # Otherwise use local smoothing followed by fitting a 2-D spline curve
        # to the smoothed discrete L-curve. Restrict the analysis of the L-curve
        # according to s_thr.
        if (s is not None):
            if (U is not None and s is not None and b is not None and M is not None):
                s = s[index]

            index = np.where(s > s_thr)
            rho = rho[index]
            eta = eta[index]
            reg_param = reg_param[index]

        # Convert to logarithms.
        lr = len(rho)
        lrho = np.log(rho)
        leta = np.log(eta)
        slrho = lrho
        sleta = leta

        # For all interior points k = q+1:length(rho)-q-1 on the discrete
        # L-curve, perform local smoothing with a polynomial of degree deg
        # to the points k-q:k+q.
        v = np.arange(-q, q + 1)
        A = np.zeros((2 * q + 1, deg + 1))
        A[:, 0] = np.ones(len(v))
        for j in range(1, deg + 1):
            A[:, j] = A[:, j - 1] * v
        for k in range(q, lr - q - 1):
            cr = np.linalg.lstsq(A, lrho[k + v], rcond=None)[0]
            slrho[k] = cr[0]
            ce = np.linalg.lstsq(A, leta[k + v], rcond=None)[0]
            sleta[k] = ce[0]

        # # Fit a 2-D spline curve to the smoothed discrete L-curve.
        # sp = spmak(np.arange(1, lr + order), np.array([slrho, sleta]))
        # pp = ppbrk(sp2pp(sp), np.array([4, lr + 1]))

        # Assuming slrho and sleta are your input data arrays and order is the spline order
        t = np.arange(1, lr + order + 1)  # Equivalent to (1:lr+order) in MATLAB

        # Fit a spline to the data
        # splrep returns the tuple (tck) containing the knots, coefficients, and degree of the spline
        tck = splrep(t, np.vstack((slrho, sleta)), k=order-1)  # k=order-1 because in scipy k=degree of spline

        # Evaluate the spline at the desired points
        # To evaluate within the specific range [4, lr+1], use splev
        pp = splev(np.arange(4, lr + 2), tck)  # lr+1 in MATLAB is lr+2 in Python because of zero-indexing



        # Assuming pp is the piecewise polynomial from the previous code
        # Extract abscissa and ordinate splines (P) and differentiate them

        # Evaluate the spline at default points
        P = np.array(splev(np.linspace(4, lr+1, num=100), pp))  # Evaluate spline at 100 points in the range [4, lr+1]
        dpp = splder(pp)  # First derivative of the spline

        # Evaluate the first derivative
        D = np.array(splev(np.linspace(4, lr+1, num=100), dpp))

        # Compute the second derivative
        ddpp = splder(pp, n=2)  # Second derivative of the spline

        # Evaluate the second derivative
        DD = np.array(splev(np.linspace(4, lr+1, num=100), ddpp))

        # Extract x and y components of the original spline, first derivative, and second derivative
        ppx, ppy = P[0, :], P[1, :]
        dppx, dppy = D[0, :], D[1, :]
        ddppx, ddppy = DD[0, :], DD[1, :]
        
        # Compute the corner of the discretized .spline curve via max. curvature.
        # No need to refine this corner, since the final regularization
        # parameter is discrete anyway.
        # Define curvature = 0 where both dppx and dppy are zero.
        k1 = dppx * ddppy - ddppx * dppy
        k2 = (dppx ** 2 + dppy ** 2) ** 1.5
        I_nz = np.where(k2 != 0)
        kappa = np.zeros(len(dppx))
        kappa[I_nz] = -k1[I_nz] / k2[I_nz]
        kmax = np.max(kappa)
        ikmax = np.argmax(kappa)
        x_corner = ppx[ikmax]
        y_corner = ppy[ikmax]

        # Locate the point on the discrete L-curve which is closest to the
        # corner of the spline curve.  Prefer a point below and to the
        # left of the corner.  If the curvature is negative everywhere,

        if (kmax < 0):
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]

        else:
            index = np.where(lrho < x_corner and leta < y_corner)
            if (len(index) > 0):
                rpi = np.argmin((lrho[index] - x_corner) ** 2 + (leta[index] - y_corner) ** 2)
                rpi = index[rpi]
            else:
                rpi = np.argmin((lrho - x_corner) ** 2 + (leta - y_corner) ** 2)
            reg_c = reg_param[rpi]
            rho_c = rho[rpi]
            eta_c = eta[rpi]
        
    elif (method == 'dsvd' or method == 'dgsv'):
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.
        g = lcfun(reg_param, s, beta, xi)

        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lambda rp: lcfun(rp, s, beta, xi), 
                          reg_param[max(gi + 1, len(g))], reg_param[min(gi - 1, 0)])
        kappa_max = -lcfun(reg_c, s, beta, xi, 1)

        if (kappa_max < 0):
            lr = len(rho)
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            f = (s ** 2) / (s ** 2 + reg_c ** 2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if (m > n):
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    else:
        raise ValueError('Illegal method')
    
    return reg_c, rho_c, eta_c
