import numpy as np

def tgsvd(U, sm, X, b, k):
    """
    Truncated GSVD regularization.

    Parameters:
    U -- m x m matrix from GSVD
    sm -- p x 2 matrix containing sigma and mu (generalized singular values)
    X -- n x n matrix from GSVD
    b -- m x 1 vector (right-hand side)
    k -- integer or array of truncation parameter(s)

    Returns:
    x_k -- solution vector(s) of size n x len(k)
    rho -- residual norms
    eta -- solution seminorms
    """
    
    # Initialization
    m = U.shape[0]
    n = X.shape[0]
    p = sm.shape[0]
    lk = len(k) if isinstance(k, (list, np.ndarray)) else 1
    k = np.atleast_1d(k)
    
    if np.min(k) < 0 or np.max(k) > p:
        raise ValueError("Illegal truncation parameter k")
    
    x_k = np.zeros((n, lk))
    eta = np.zeros(lk)
    rho = np.zeros(lk)
    
    beta = np.dot(U.T, b)
    xi = beta[:p] / sm[:, 0]  # sm[:, 0] corresponds to sigma
    
    if lk == 1:
        mxi = sm[:, 1] * xi  # sm[:, 1] corresponds to mu
    else:
        mxi = np.zeros_like(xi)
    
    # Overdetermined or square case
    if m >= n:
        if p == n:
            x_0 = np.zeros(n)
        else:
            x_0 = np.dot(X[:, p:n], np.dot(U[:, p:n].T, b))
        
        for j in range(lk):
            i = k[j]
            pi1 = p - i + 1
            if i == 0:
                x_k[:, j] = x_0
            else:
                x_k[:, j] = np.dot(X[:, pi1-1:p], xi[pi1-1:p]) + x_0

            if lk > 1:
                rho[j] = np.linalg.norm(beta[:p-i])
                eta[j] = np.linalg.norm(mxi[pi1-1:p])

        if lk > 1 and U.shape[0] > n:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - np.dot(U[:, :n], beta[:n]))**2)
    
    # Underdetermined case
    else:
        if p == m:
            x_0 = np.zeros(n)
        else:
            x_0 = np.dot(X[:, p:m], np.dot(U[:, p:m].T, b))
        
        for j in range(lk):
            i = k[j]
            pi1 = p - i + 1
            if i == 0:
                x_k[:, j] = x_0
            else:
                x_k[:, j] = np.dot(X[:, pi1-1:p], xi[pi1-1:p]) + x_0

            if lk > 1:
                rho[j] = np.linalg.norm(beta[:p-i])
                eta[j] = np.linalg.norm(mxi[pi1-1:p])

    return x_k, rho, eta

# # Example usage
# U = np.random.randn(5, 5)  # Example U matrix
# sm = np.random.randn(3, 2)  # Example sm matrix (sigma, mu)
# X = np.random.randn(5, 5)  # Example X matrix
# b = np.random.randn(5, 1)  # Example b vector
# k = [1, 2]  # Example truncation parameters

# x_k, rho, eta = tgsvd(U, sm, X, b, k)

# print("x_k =", x_k)
# print("rho =", rho)
# print("eta =", eta)
