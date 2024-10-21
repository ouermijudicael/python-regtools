import numpy as np
import matplotlib.pyplot as plt

def picard(U, s, b, d=0):
    """
    Visual inspection of the Picard condition.

    Parameters:
    U (ndarray): Left singular vectors.
    s (ndarray): Singular values or generalized singular values.
    b (ndarray): Right-hand side vector.
    d (int): Smoothing parameter. Default is 0 (no smoothing).

    Returns:
    eta (ndarray): Solution coefficients.
    """
    n = len(s) if s.ndim == 1 else s.shape[0]
    ps = 1 if s.ndim == 1 else s.shape[1]
    beta = np.abs(U[:, :n].T @ b)
    eta = np.zeros(n)
    
    if ps == 2:
        s = s[:, 0] / s[:, 1]
    
    d21 = 2 * d + 1
    keta = np.arange(1 + d, n - d)
    
    if not np.all(s):
        print('Warning: Division by zero singular values')
    
    for i in keta:
        eta[i] = (np.prod(beta[i - d:i + d + 1]) ** (1 / d21)) / s[i]
    
    # Plot the data
    plt.semilogy(np.arange(1, n + 1), s, '.-', label='σ_i' if ps == 1 else 'σ_i/μ_i')
    plt.semilogy(np.arange(1, n + 1), beta, 'x', label='|u_i^Tb|')
    plt.semilogy(keta + 1, eta[keta], 'o', label='|u_i^Tb|/σ_i' if ps == 1 else '|u_i^Tb| / (σ_i/μ_i)')
    
    plt.xlabel('i')
    plt.title('Picard plot')
    plt.legend(loc='best' if ps == 1 else 'NorthWest')
    # plt.show()
    
    return eta

# # Example usage
# U = np.array([[0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
# s = np.array([3, 2])
# b = np.array([1, 2, 3])
# eta = picard(U, s, b)
# print("eta:", eta)
