import numpy as np
from plot_lc import plot_lc
from l_corner import l_corner
import matplotlib.pyplot as plt
# function [reg_corner,rho,eta,reg_param] = l_curve(U,sm,b,method,L,V)
def l_curve(U, sm, b, method='Tikh', L=None, V=None):  
#  
# L_CURVE Plot the L-curve and find its "corner".
# 
# [reg_corner,rho,eta,reg_param] =
#                   l_curve(U,s,b,method)
#                   l_curve(U,sm,b,method)  ,  sm = [sigma,mu]
#                   l_curve(U,s,b,method,L,V)
# 
#  Plots the L-shaped curve of eta, the solution norm || x || or
#  semi-norm || L x ||, as a function of rho, the residual norm
#  || A x - b ||, for the following methods:
#     method = 'Tikh'  : Tikhonov regularization   (solid line )
#     method = 'tsvd'  : truncated SVD or GSVD     (o markers  )
#     method = 'dsvd'  : damped SVD or GSVD        (dotted line)
#     method = 'mtsvd' : modified TSVD             (x markers  )
#  The corresponding reg. parameters are returned in reg_param.  If no
#  method is specified then 'Tikh' is default.  For other methods use plot_lc.
# 
#  Note that 'Tikh', 'tsvd' and 'dsvd' require either U and s (standard-
#  form regularization) computed by the function csvd, or U and sm (general-
#  form regularization) computed by the function cgsvd, while 'mtvsd'
#  requires U and s as well as L and V computed by the function csvd.
# 
#  If any output arguments are specified, then the corner of the L-curve
#  is identified and the corresponding reg. parameter reg_corner is
#  returned.  Use routine l_corner if an upper bound on eta is required.

#  Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
#  the regularization of discrete ill-posed problems",  SIAM J. Sci.
#  Comput. 14 (1993), pp. 1487-1503.

#  Per Christian Hansen, DTU Compute, October 27, 2010.

    #  Set defaults.
    # if (nargin==3), method='Tikh'; end  % Tikhonov reg. is default.
    npoints = 200  # Number of points on the L-curve for Tikh and dsvd.
    smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.

    # Initialization.
    m,n = U.shape 
    p, ps = sm.shape if len(sm.shape) > 1 else (len(sm), 1)
    locate = True
    beta = U.T @ b
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    if (ps==1):
        s = sm; 
        beta = beta[:p]
    else:
        s = sm[p-1::-1, 0] / sm[p-1::-1, 1]
        beta = beta[p-1::-1]

    xi = beta[:p] / s
    xi[np.isinf(xi)] = 0

    if method.lower().startswith('tikh'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        s2 = s**2
        reg_param[-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1])**(1 / (npoints - 1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]
        for i in range(npoints):
            f = s2 / (s2 + reg_param[i]**2)
            eta[i] = np.linalg.norm(f * xi)
            rho[i] = np.linalg.norm((1 - f) * beta[:p])
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
        marker = '-'
        txt = 'Tikh.'
    
    elif method.lower().startswith('tsvd') or method.lower().startswith('tgsv'):
        eta = np.zeros(p)
        rho = np.zeros(p)
        eta[0] = abs(xi[0])**2
        for k in range(1, p):
            eta[k] = eta[k-1] + abs(xi[k])**2
        eta = np.sqrt(eta)
        if m > n:
            if beta2 > 0:
                rho[p-1] = beta2
            else:
                rho[p-1] = np.finfo(float).eps**2
        else:
            rho[p-1] = np.finfo(float).eps**2
        for k in range(p-2, -1, -1):
            rho[k] = rho[k+1] + abs(beta[k+1])**2
        rho = np.sqrt(rho)
        reg_param = np.arange(1, p+1)
        marker = 'o'
        if ps == 1:
            txt = 'TSVD'
        else:
            txt = 'TGSVD'

    elif method.lower().startswith('dsvd') or method.lower().startswith('dgsv'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        reg_param[-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1])**(1 / (npoints - 1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]
        for i in range(npoints):
            f = s / (s + reg_param[i])
            eta[i] = np.linalg.norm(f * xi)
            rho[i] = np.linalg.norm((1 - f) * beta[:p])
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
        marker = ':'
        if ps == 1:
            txt = 'DSVD'
        else:
            txt = 'DGSVD'

    elif method.lower().startswith('mtsv'):
        if L is None or V is None:
            raise ValueError('The matrices L and V must also be specified')
        p, n = L.shape
        rho = np.zeros(p)
        eta = np.zeros(p)
        Q, R = np.linalg.qr(L @ V[:, n-1:n-p-1:-1], mode='economic')
        for i in range(p):
            k = n - p + i
            Lxk = L @ V[:, :k] @ xi[:k]
            zk = np.linalg.solve(R[:n-k, :n-k], Q[:, :n-k].T @ Lxk)
            zk = zk[n-k-1::-1]
            eta[i] = np.linalg.norm(Q[:, n-k:p].T @ Lxk)
            if i < p:
                rho[i] = np.linalg.norm(beta[k:n] + s[k:n] * zk)
            else:
                rho[i] = np.finfo(float).eps
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
        reg_param = np.arange(n-p+1, n+1)
        txt = 'MTSVD'
        U = U[:, reg_param]
        sm = sm[reg_param]
        marker = 'x'
        ps = 2
        
    else:
        raise ValueError('Illegal method')

    # Locate the "corner" of the L-curve, if required.
    if (locate):
        [reg_corner,rho_c,eta_c] = l_corner(rho,eta,reg_param,U,sm,b,method);

    # Make plot.
    plot_lc(rho,eta,marker,ps,reg_param)
   
    if locate:
        ax = plt.axis()
        plt.loglog([min(rho)/100,rho_c],[eta_c,eta_c],':r', [rho_c,rho_c],[min(eta)/100,eta_c],':r')
        # ax.loglog([min(rho)/100,rho_c],[eta_c,eta_c],':r', [rho_c,rho_c],[min(eta)/100,eta_c],':r')
        plt.title(f'L-curve, {txt} corner at {reg_corner}')

    return reg_corner, rho, eta, reg_param