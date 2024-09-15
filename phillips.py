import numpy as np
from scipy.linalg import toeplitz

def phillips(n):
    """
    Discretization of the 'famous' first-kind Fredholm integral equation devised by D. L. Phillips.
    
    Parameters:
        n (int): The order of the problem, must be a multiple of 4.
    
    Returns:
        A (ndarray): Coefficient matrix.
        b (ndarray): Right-hand side.
        x (ndarray): Solution vector (optional).
    """
    
    # Check input
    if n % 4 != 0:
        raise ValueError('The order n must be a multiple of 4')

    # Compute the matrix A
    h = 12 / n
    n4 = n // 4
    r1 = np.zeros(n)
    c = np.cos(np.arange(-1, n4+1) * 4 * np.pi / n)
    r1[:n4] = h + 9 / (h * np.pi**2) * (2 * c[1:n4+1] - c[:n4] - c[2:n4+2])
    r1[n4] = h / 2 + 9 / (h * np.pi**2) * (np.cos(4 * np.pi / n) - 1)
    A = toeplitz(r1)

     # Compute the right-hand side b
    b = np.zeros(n)
    c_val = np.pi / 3
    for i in range(int(n/2), n):
        t1 = -6 + (i+1) * h
        t2 = t1 - h
        b[i] = (
            t1 * (6 - np.abs(t1) / 2)
            + ((3 - np.abs(t1) / 2) * np.sin(c_val * t1) - 2 / c_val * (np.cos(c_val * t1) - 1)) / c_val
            - t2 * (6 - np.abs(t2) / 2)
            - ((3 - np.abs(t2) / 2) * np.sin(c_val * t2) - 2 / c_val * (np.cos(c_val * t2) - 1)) / c_val
        )
        b[n - i - 1] = b[i]
    b = b / np.sqrt(h)
    
    # Compute the solution x
    x = None
    if n > 1:
        x = np.zeros(n)
        t_range = np.arange(0, 3 + 10 * np.finfo(float).eps, h)
        # t_range = np.linspace(0, 3 + 10 * np.finfo(float).eps, int(n4+1)) # n4+1 to cover full range
        x[2*n4:3*n4] = (h + np.diff(np.sin(t_range * c_val))/c_val ) / np.sqrt(h)
        x[n4:2*n4] = x[3*n4-1:2*n4-1:-1]
    
    return A, b, x




