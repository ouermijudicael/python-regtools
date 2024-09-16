import numpy as np
from scipy.linalg import svd, qr, inv

def gsvd(A, L):
    """
    Compute the Generalized Singular Value Decomposition (GSVD) of matrices A and L using standard SVD and QR.
    
    Parameters:
    A -- m x n matrix
    L -- p x n matrix
    
    Returns:
    U -- m x m orthogonal matrix
    V -- p x p orthogonal matrix
    X -- n x n nonsingular matrix
    alpha, beta -- generalized singular value vectors such that:
                   diag(alpha) * U.T * A * X = diag(beta) * V.T * L * X
    """
    
    # Ensure the input matrices are full matrices and correct shape
    A = np.atleast_2d(A)
    L = np.atleast_2d(L)
    
    # Get dimensions of A and L
    m, n = A.shape
    p, n1 = L.shape
    
    if n1 != n:
        raise ValueError("Number of columns in A and L must be the same.")
    if m + p < n:
        raise ValueError("Dimensions must satisfy m + p >= n.")
    
    # Step 1: Perform QR decompositions on A and L
    Q_A, R_A = qr(A, mode='economic')  # QR decomposition of A
    Q_L, R_L = qr(L, mode='economic')  # QR decomposition of L
    
    # Step 2: Perform SVD on the R_A and R_L blocks
    U_R, s_R, Vh_R = svd(R_A, full_matrices=False)  # SVD of R_A
    V_R, s_L, Wh_R = svd(R_L, full_matrices=False)  # SVD of R_L
    
    # Step 3: Combine the SVD results to obtain GSVD components
    U = np.dot(Q_A, U_R)  # m x n
    V = np.dot(Q_L, V_R)  # p x n
    X = inv(Vh_R.T)  # n x n
    
    # Step 4: Compute alpha and beta from singular values
    alpha = np.hstack([s_R, np.zeros(max(0, n - len(s_R)))])  # Generalized singular values for A
    beta = np.hstack([s_L, np.zeros(max(0, n - len(s_L)))])  # Generalized singular values for L
    
    return U, V, X, np.diag(alpha), np.diag(beta)

