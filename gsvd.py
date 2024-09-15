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

# # Example usage
# A = np.random.randn(5, 3)  # A is 5x3 matrix
# L = np.random.randn(4, 3)  # L is 4x3 matrix

# U, alpha, beta, X, V = gsvd_lapack(A, L)

# print("U =", U)
# print("alpha =", alpha)
# print("beta =", beta)
# print("X =", X)
# print("V =", V)



# import numpy as np
# from scipy.linalg import qr, svd

# def gsvd(A, L):
#     """
#     Generalized Singular Value Decomposition (GSVD) of matrix pair (A, L)

#     Parameters:
#     A - Matrix A of size (m x n)
#     L - Matrix L of size (p x n)

#     Returns:
#     U - Orthogonal matrix of size (m x m)
#     V - Orthogonal matrix of size (p x p)
#     X - Nonsingular matrix of size (n x n)
#     C - Generalized singular values for A (diagonal values)
#     S - Generalized singular values for L (diagonal values)
#     """
    
#     # Check dimensions
#     m, n = A.shape
#     p, n1 = L.shape
    
#     if n1 != n:
#         raise ValueError('The number of columns in A and L must be the same.')
    
#     if (m + p) < n:
#         raise ValueError('The combined rows of A and L must be at least n.')
    
#     # Step 1: Perform QR decomposition of A and L
#     QA, RA = qr(A, mode='economic')   # Economic QR decomposition of A
#     QL, RL = qr(L, mode='economic')   # Economic QR decomposition of L
    
#     # Step 2: Create a stacked matrix from RA and RL
#     stacked_matrix = np.vstack([RA, RL])  # Stack RA and RL vertically
    
#     # Step 3: Perform SVD on the stacked matrix
#     U_stacked, Sigma, V_stacked = svd(stacked_matrix, full_matrices=False)
    
#     # Step 4: Extract U, V, C, and S from the SVD results
#     rank_A = RA.shape[0]   # Number of rows of RA (rank of A)
#     rank_L = RL.shape[0]   # Number of rows of RL (rank of L)
    
#     # Extract U and V from the SVD
#     U = np.dot(QA, U_stacked[:rank_A, :])   # U corresponds to A (m x m)
#     V = np.dot(QL, U_stacked[rank_A:, :])   # V corresponds to L (p x p)
    
#     # Extract generalized singular values C and S
#     C = np.diag(Sigma[:rank_A])  # Generalized singular values for A
#     S = np.diag(Sigma[rank_A:])  # Generalized singular values for L
    
#     # Step 5: Extract X from V_stacked
#     X = V_stacked   # X corresponds to the nonsingular matrix
    
#     return U, V, X, C, S

# # # Example usage:
# # A = np.random.rand(5, 4)  # A is 5x4
# # L = np.random.rand(3, 4)  # L is 3x4

# # # Perform GSVD
# # U, V, X, C, S = custom_gsvd(A, L)

# # # Display results
# # print("U =\n", U)
# # print("V =\n", V)
# # print("X =\n", X)
# # print("C (Generalized singular values for A) =\n", C)
# # print("S (Generalized singular values for L) =\n", S)
