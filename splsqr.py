import numpy as np

def splsqr(A, b, lambda_, Vsp, maxit=None, tol=None, reorth= None):
#SPLSQR Subspace preconditioned LSQR for discrete ill-posed problems.
#
# x = splsqr(A,b,lambda,Vsp,maxit,tol,reorth)
#
# Subspace preconditioned LSQR (SP-LSQR) for solving the Tikhonov problem
#    min { || A x - b ||^2 + lambda^2 || x ||^2 }
#   with a preconditioner based on the subspace defined by the columns of
#   the matrix Vsp.  While not necessary, we recommend to use a matrix Vsp
#   with orthonormal columns.
#  
#   The output x holds all the solution iterates as columns, and the last
#   iterate x(:,end) is the best approximation to x_lambda.
#  
#   The parameter maxit is the maximum allowed number of iterations (default
#   value is maxit = 300).  The parameter tol is used a stopping criterion
#   for the norm of the least squares residual relative to the norm of the
#   right-hand side (default value is tol = 1e-12).
#  
#   A seventh input parameter reorth ~= 0 enforces MGS reorthogonalization
#   of the Lanczos vectors.

#   This is a model implementation of SP-LSQR.  In a real implementation the
#   Householder transformations should use LAPACK routines, only the final
#   iterate should be returned, and reorthogonalization is not used.  Also,
#   if Vsp represents a fast transformation (such as the DCT) then explicit
#   storage of Vsp should be avoided.  See the reference for details.

#   Reference: M. Jacobsen, P. C. Hansen and M. A. Saunders, "Subspace pre-
#   conditioned LSQR for discrete ill-posed problems", BIT 43 (2003), 975-989.


    # Input check.
    if maxit is None:
        maxit = 300
    if tol is None:
        tol = 1e-12
    if reorth is None:
        reorth = 0
    if maxit < 1:
        raise ValueError('Number of iterations must be positive')
    
    # Prepare for SP-LSQR algorithm.
    m, n = A.shape
    k = Vsp.shape[1]
    z = np.zeros(n)

    if reorth:
        UU = np.zeros((m + n, maxit))
        VV = np.zeros((n, maxit))

    # Initial QR factorization of [A;lamnda*eye(n)]*Vsp;
    QQ = np.linalg.qr(np.vstack((A @ Vsp, lambda_ * Vsp)))

    # Prepare for LSQR iterations.
    u = app_house_t(QQ, np.hstack((b, z)))
    u[:k] = 0
    beta = np.linalg.norm(u)
    u = u / beta

    v = app_house(QQ, u)
    v = A.T @ v[:m] + lambda_ * v[m:]
    alpha = np.linalg.norm(v)
    v = v / alpha
    
    w = v
    Wxw = np.zeros(n)

    phi_bar = beta
    rho_bar = alpha

    if reorth:
        UU[:, 0] = u
        VV[:, 0] = v


    # Iterate.
    for i in range(maxit):
        # beta*u = A*v - alpha*u;
        uu = np.hstack((A @ v, lambda_ * v))
        uu = app_house_t(QQ, uu)
        uu[:k] = 0
        u = uu - alpha * u
        if reorth:
            for j in range(i):
                u = u - (UU[:, j] @ u) * UU[:, j]
        beta = np.linalg.norm(u)
        u = u / beta

        # alpha * v = A'*u - beta*v;
        vv = app_house(QQ, u)
        v = A.T @ vv[:m] + lambda_ * vv[m:] - beta * v
        if reorth:
            for j in range(i):
                v = v - (VV[:, j] @ v) * VV[:, j]
        alpha = np.linalg.norm(v)
        v = v / alpha

        if reorth:
            UU[:, i] = u
            VV[:, i] = v

        # Update LSQR parameters.
        rho = np.linalg.norm(np.hstack((rho_bar, beta)))
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update the LSQR solution.
        Wxw = Wxw + (phi / rho) * w
        w = v - (theta / rho) * w

        # Compute residual and update the SP-LSQR iterate.
        r = np.hstack((b - A @ Wxw, -lambda_ * Wxw))
        r = app_house_t(QQ, r)
        r = r[:k]
        xv = np.linalg.solve(np.triu(QQ[:k, :]), r)
        x[:, i] = Vsp @ xv + Wxw

        # Stopping criterion.
        if phi_bar * alpha * abs(c) < tol * np.linalg.norm(b):
            break

    return x

# -----------------------------------------------------------------
def app_house(H,X):
# Y = app_house(H,X)
# Input:  H = matrix containing the necessary information of the
#             Householder vectors v in the lower triangle and R in
#             the upper triangle; e.g., computed as H = qr(A).
#         X = matrix to be multiplied with orthogonal matrix.
# Output: Y = Q*X
    
    n, p = H.shape
    Y = X
    for k in range(p):
        v = np.ones(n + 1 - k)
        v[1:n + 1 - k] = H[k + 1:n, k]
        beta = 2 / (v @ v)
        Y[k:n] = Y[k:n] - beta * v @ (v @ Y[k:n])

    return Y


# -----------------------------------------------------------------
def app_house_t(H,X):
# Y = app_house_t(H,X)
# Input:  H = matrix containing the necessary information of the
#              Householder vectors v in the lower triangle and R in
#              the upper triangle; e.g., computed as H = qr(A).
#         X = matrix to be multiplied with transposed orthogonal matrix.
# Output: Y = Q'*X

    n, p = H.shape
    Y = X
    for k in range(p - 1, -1, -1):
        v = np.ones(n + 1 - k)
        v[1:n + 1 - k] = H[k + 1:n, k]
        beta = 2 / (v @ v)
        Y[k:n] = Y[k:n] - beta * v @ (v @ Y[k:n])

    return Y
