import numpy as np
import time as time
def discrep(U, s, V, b, delta, x_0 = None):
    # function [x_delta,lambda] = discrep(U,s,V,b,delta,x_0)
    # %DISCREP Discrepancy principle criterion for choosing the reg. parameter.
    # %
    # % [x_delta,lambda] = discrep(U,s,V,b,delta,x_0)
    # % [x_delta,lambda] = discrep(U,sm,X,b,delta,x_0)  ,  sm = [sigma,mu]
    # %
    # % Least squares minimization with a quadratic inequality constraint:
    # %    min || x - x_0 ||       subject to   || A x - b || <= delta
    # %    min || L (x - x_0) ||   subject to   || A x - b || <= delta
    # % where x_0 is an initial guess of the solution, and delta is a
    # % positive constant.  Requires either the compact SVD of A saved as
    # % U, s, and V, or part of the GSVD of (A,L) saved as U, sm, and X.
    # % The regularization parameter lambda is also returned.
    # %
    # % If delta is a vector, then x_delta is a matrix such that
    # %    x_delta = [ x_delta(1), x_delta(2), ... ] .
    # %
    # % If x_0 is not specified, x_0 = 0 is used.

    # % Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
    # % Problems", Springer, 1984; Chapter 26.

    # % Per Christian Hansen, IMM, August 6, 2007.

    #  Initialization.
    m = U.shape[0]
    n = V.shape[0]
    p = s.shape[0]
    ps = 1 if len(s.shape) == 1 else s.shape[1]
    ld = len(delta)
    x_delta = np.zeros((n, ld))
    lambda_ = np.zeros(ld)
    rho = np.zeros(p)
    if np.min(delta) < 0:
        raise ValueError('Illegal inequality constraint delta')
    
    if x_0 is None:
        x_0 = np.zeros(n)
    omega = np.dot(V.T, x_0) if ps == 1 else np.linalg.solve(V, x_0)


    # Compute residual norms corresponding to TSVD/TGSVD.
    beta = np.dot(U.T, b)
    # print('n:', n, 'p:', p, 'ps:', ps, 'm:', m)
    # print('omega:', omega)
    # print('beta:', beta)
    if ps == 1:
        delta_0 = np.linalg.norm(b - np.dot(U, beta))
        rho[p-1] = delta_0**2
        for i in range(p-1, 0, -1):
            rho[i-1] = rho[i] + (beta[i] - s[i]*omega[i])**2

        # print('rho:', rho)
    else:
        delta_0 = np.linalg.norm(b - np.dot(U, beta))
        rho[0] = delta_0**2
        for i in range(p-1):
            rho[i+1] = rho[i] + (beta[i] - s[i, 0]*omega[i])**2


    # Check input.
    if np.min(delta) < delta_0:
        raise ValueError('Irrelevant delta < || (I - U*U'')*b ||')
    
    # Determine the initial guess via rho-vector, then solve the nonlinear
    # equation || b - A x ||^2 - delta_0^2 = 0 via Newton's method.
    if ps == 1:
        s2 = s**2
        for k in range(ld):
            if delta[k]**2 >= np.linalg.norm(beta - s*omega)**2 + delta_0**2:
                x_delta[:, k] = x_0
            else:
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = s[kmin]
                lambda_[k] = newton(lambda_0, delta[k], s, beta, omega, delta_0)
                e = s / (s2 + lambda_[k]**2)
                f = s * e
                # x_delta[:, k] = np.dot(np.transpose(V[:, :p]), e*beta + (1-f)*omega)
                x_delta[:, k] = np.transpose(V[:, :p]) @ (e*beta + (1-f)*omega) 
    # NOT TESTED
    elif m >= n:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:n], beta[p:n])
        for k in range(ld):
            if delta[k]**2 >= np.linalg.norm(beta[:p] - s[:, 0]*omega)**2 + delta_0**2:
                x_delta[:, k] = np.dot(V, [omega, np.dot(U[:, p:n].T, b)])
            else:
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = gamma[kmin]
                lambda_[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma**2 + lambda_[k]**2)
                f = gamma * e
                # x_delta[:, k] = np.dot(V[:, :p], e*beta[:p]/s[:, 1] + (1-f)*s[:, 1]*omega) + x_u 
                x_delta[:, k] = np.transpose(V[:, :p]) @ (e*beta[:p]/s[:, 1] + (1-f)*s[:, 1]*omega) + x_u 
    # NOT TESTED
    else:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:m], beta[p:m])
        for k in range(ld):
            if delta[k]**2 >= np.linalg.norm(beta[:p] - s[:, 0]*omega)**2 + delta_0**2:
                x_delta[:, k] = np.dot(V, [omega, np.dot(U[:, p:m].T, b)])
            else:
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = gamma[kmin]
                lambda_[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma**2 + lambda_[k]**2)
                f = gamma * e
                # x_delta[:, k] = np.dot(V[:, :p], e*beta[:p]/s[:, 1] + (1-f)*s[:, 1]*omega) + x_u
                x_delta[:, k] = np.transpose(V[:, :p]) @ (e*beta[:p]/s[:, 1] + (1-f)*s[:, 1]*omega) + x_u

    return x_delta, lambda_


def newton(lambda_0, delta, s, beta, omega, delta_0):
    # function lambda = newton(lambda_0,delta,s,beta,omega,delta_0)
    # %NEWTON Newton iteration (utility routine for DISCREP).
    # %
    # % lambda = newton(lambda_0,delta,s,beta,omega,delta_0)
    # %
    # % Uses Newton iteration to find the solution lambda to the equation
    # %    || A x_lambda - b || = delta ,
    # % where x_lambda is the solution defined by Tikhonov regularization.
    # %
    # % The initial guess is lambda_0.
    # %
    # % The norm || A x_lambda - b || is computed via s, beta, omega and
    # % delta_0.  Here, s holds either the singular values of A, if L = I,
    # % or the c,s-pairs of the GSVD of (A,L), if L ~= I.  Moreover,
    # % beta = U'*b and omega is either V'*x_0 or the first p elements of
    # % inv(X)*x_0.  Finally, delta_0 is the incompatibility measure.

    # % Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
    # % Problems", Springer, 1984; Chapter 26.

    # % Per Christian Hansen, IMM, 12/29/97.

    # % Set defaults.
    thr = np.sqrt(np.finfo(float).eps)  # Relative stopping criterion.
    it_max = 50      # Max number of iterations.

    # % Initialization.
    if lambda_0 < 0:
        raise ValueError('Initial guess lambda_0 must be nonnegative')
    p = s.shape[0]
    ps = 1 if len(s.shape) == 1 else s.shape[1]
    if ps == 2:
        sigma = s[:, 0]
        s = s[:, 0] / s[:, 1]
    s2 = s**2

    # % Use Newton's method to solve || b - A x ||^2 - delta^2 = 0.
    # % It was found experimentally, that this formulation is superior
    # % to the formulation || b - A x ||^(-2) - delta^(-2) = 0.
    lambda_ = lambda_0
    step = 1
    it = 0
    while abs(step) > thr*lambda_ and abs(step) > thr and it < it_max:
        it += 1
        f = s2 / (s2 + lambda_**2)
        if ps == 1:
            r = (1-f) * (beta - s * omega)
            z = f * r
        else:
            r = (1-f) * (beta - sigma * omega)
            z = f * r
        step = (lambda_ / 4) * (np.dot(r.T, r) + (delta_0 + delta) * (delta_0 - delta)) / np.dot(z.T, r)
        lambda_ -= step

        # % If lambda < 0 then restart with smaller initial guess.
        if lambda_ < 0:
            lambda_ = 0.5 * lambda_0
            lambda_0 = 0.5 * lambda_0

    # % Terminate with an error if too many iterations.
    if abs(step) > thr * lambda_ and abs(step) > thr:
        raise ValueError('Max. number of iterations ({}) reached'.format(it_max))
    
    return lambda_
    
