import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from gcvfun import gcvfun


def gcv(U,s,b,method=None):
# GCV Plot the GCV function and find its minimum.
#
# [reg_min,G,reg_param] = gcv(U,s,b,method)
# [reg_min,G,reg_param] = gcv(U,sm,b,method)  ,  sm = [sigma,mu]
#
# Plots the GCV-function
#          || A*x - b ||^2
#    G = -------------------
#        (trace(I - A*A_I)^2
# as a function of the regularization parameter reg_param. Here, A_I is a
# matrix which produces the regularized solution.
#
# The following methods are allowed:
#    method = 'Tikh' : Tikhonov regularization   (solid line )
#    method = 'tsvd' : truncated SVD or GSVD     (o markers  )
#    method = 'dsvd' : damped SVD or GSVD        (dotted line)
# If method is not specified, 'Tikh' is default.  U and s, or U and sm,
# must be computed by the functions csvd and cgsvd, respectively.
#
# If any output arguments are specified, then the minimum of G is
# identified and the corresponding reg. parameter reg_min is returned.


# Reference: G. Wahba, "Spline Models for Observational Data",
# SIAM, 1990.

    # Set defaults
    if method is None:
        method = 'Tikh' # Default method.
    npoints = 200 # Number of points on the curve.
    smin_ratio = 16 * np.finfo(float).eps # Smallest regularization parameter.

    # Initialization.
    m,n = U.shape
    p, ps = s.shape if len(s.shape) > 1 else (len(s), 1)
    beta = np.dot(U.T,b)
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    if ps == 2:
        s = s[::-1,0] / s[::-1,1]
        beta = beta[::-1]
    
    find_min = True

    if method.lower() == 'tikh':
        # Vector of regularization parameters.
        reg_param = np.zeros(npoints)
        G = np.zeros(npoints)
        s2 = s**2
        reg_param[-1] = max([s[p-1],s[0]*smin_ratio])
        ratio = (s[0]/reg_param[-1])**(1/(npoints-1))
        for i in range(npoints-2,-1,-1):
            reg_param[i] = ratio*reg_param[i+1]

        # Intrinsic residual.
        delta0 = 0
        if m > n and beta2 > 0:
            delta0 = beta2

        # Vector of GCV-function values.
        for i in range(npoints):
            G[i] = gcvfun(reg_param[i],s2,beta[:p],delta0,m-n)

        # Plot GCV function.
        plt.loglog(reg_param,G,'-')
        plt.xlabel('$\\lambda$')
        plt.ylabel('G($\\lambda$)')
        plt.title('GCV function')

        # Find minimum, if requested.
        if find_min:
            minG = np.min(G)
            minGi = np.argmin(G)
            reg_min = fminbound(lambda reg_min: gcvfun(reg_min, s2, beta[:p], delta0,m-n), reg_param[min(minGi+1,npoints)], reg_param[max(minGi-1,0)])
        
            minG = gcvfun(reg_min,s2,beta[:p],delta0,m-n)
            ax = plt.axis()
            # HoldState = plt.ishold()
            # plt.hold(True)
            plt.loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
            plt.title('GCV function, minimum at $\\lambda$ = {}'.format(reg_min))
            plt.axis(ax)
            # if not HoldState:
                # plt.hold(False)
    elif method.lower() == 'tsvd' or method.lower() == 'tgsv':

        #  Vector of GCV-function values.
        rho2 = np.zeros(p-1)
        rho2[p-2] = np.abs(beta[p-1])**2
        if m > n and beta2 > 0:
            rho2[p-2] = rho2[p-2] + beta2
        for k in range(p-3,-1,-1):
            rho2[k] = rho2[k+1] + np.abs(beta[k+1])**2
        G = np.zeros(p-1)
        for k in range(p-1):
            G[k] = rho2[k] / (m - k + (n - p))**2
        reg_param = np.arange(1,p)


        # Plot GCV function.
        plt.semilogy(reg_param,G,'o')
        plt.xlabel('k')
        plt.ylabel('G(k)')
        plt.title('GCV function')

        # Find minimum, if requested.
        if find_min:
            minG = np.min(G)
            reg_min = np.argmin(G)
            ax = plt.axis()
            # HoldState = plt.ishold()
            # plt.hold(True)
            plt.semilogy(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
            plt.title('GCV function, minimum at k = {}'.format(reg_min))
            plt.axis(ax)
            # if not HoldState:
                # plt.hold(False)

    elif method.lower() == 'dsvd' or method.lower() == 'dgsv':

        # Vector of regularization parameters.
        reg_param = np.zeros(npoints)
        G = np.zeros(npoints)
        reg_param[-1] = max([s[p-1],s[0]*smin_ratio])
        ratio = (s[0]/reg_param[-1])**(1/(npoints-1))
        for i in range(npoints-2,-1,-1):
            reg_param[i] = ratio*reg_param[i+1]

        # Intrinsic residual.
        delta0 = 0
        if m > n and beta2 > 0:
            delta0 = beta2

        # Vector of GCV-function values.
        for i in range(npoints):
            G[i] = gcvfun(reg_param[i],s,beta[:p],delta0,m-n)

        # Plot GCV function.
        plt.loglog(reg_param,G,':')
        plt.xlabel('$\\lambda$')
        plt.ylabel('G($\\lambda$)')
        plt.title('GCV function')

        # Find minimum, if requested.
        if find_min:
            minG = np.min(G)
            minGi = np.argmin(G)
            # reg_min = fminbnd('gcvfun',reg_param[min(minGi+1,npoints)],reg_param[max(minGi-1,0)],optimset('Display','off'),s,beta[:p],delta0,m-n)
            reg_min = fminbound(lambda reg_min: gcvfun(reg_min, s, beta[:p], delta0,m-n), reg_param[min(minGi+1,npoints)], reg_param[max(minGi-1,0)])
            minG = gcvfun(reg_min,s,beta[:p],delta0,m-n)
            ax = plt.axis()
            # HoldState = plt.ishold()
            # plt.hold(True)
            plt.loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
            plt.title('GCV function, minimum at $\\lambda$ = {}'.format(reg_min))
            plt.axis(ax)
            # if not HoldState:
                # plt.hold(False)

    else:
        raise ValueError('Illegal method')
    

    return reg_min, G, reg_param