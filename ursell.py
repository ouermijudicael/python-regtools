import numpy as np
from scipy.linalg import hankel

def ursell(n):
    # Ursell test problem: integral equation with no square integrable solution.
    
    # Discretization of a first kind Fredholm integral equation with
    # kernel K and right-hand side g given by:
    #     K(s,t) = 1/(s+t+1),  g(s) = 1,
    # where both integration intervals are [0,1].

    # Returns:
    # A -- matrix resulting from discretization
    # b -- right-hand side vector
    
    # Initialize r and c as zeros
    r = np.zeros(n)
    c = np.zeros(n)
    
    # Compute the matrix A
    for k in range(1, n+1):
        d1 = 1 + (1 + k) / n
        d2 = 1 + k / n
        d3 = 1 + (k - 1) / n
        c[k-1] = n * (d1 * np.log(d1) + d3 * np.log(d3) - 2 * d2 * np.log(d2))
        
        e1 = 1 + (n + k) / n
        e2 = 1 + (n + k - 1) / n
        e3 = 1 + (n + k - 2) / n
        r[k-1] = n * (e1 * np.log(e1) + e3 * np.log(e3) - 2 * e2 * np.log(e2))
    
    # Create the matrix A using the Hankel matrix
    A = hankel(c, r)
    
    # Compute the right-hand side b
    b = np.ones(n) / np.sqrt(n)
    
    return A, b

# # Example usage
# n = 5
# A, b = ursell(n)
# print("Matrix A:\n", A)
# print("Right-hand side b:\n", b)
