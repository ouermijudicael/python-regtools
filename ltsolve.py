def ltsolve(L, y, W=None, T=None):
# LTSOLVE Utility routine for "preconditioned" iterative methods.
# 
# x = ltsolve(L,y,W,T)
#
# Computes the vector
#   x = (L_p)'*y
# where L_p is the A-weighted generalized inverse of L.
#
# Here, L is a p-by-n matrix, W holds a basis for the null space of L,
# and T is a utility matrix which should be computed by routine pinit.
#
# Alternatively, L is square, and W and T are not needed.
#
# If W and T are not specified, then instead the routine computes
#   x = inv(L(:,1:p))'*y(1:p) .
#
# Notice that x and y may be matrices, in which case x(:,i)
# corresponds to y(:,i).

# Reference: P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems.
# Numerical Aspects of Linear Inversion", SIAM, Philadelphia, 1997.
    import numpy as np

    p, n = L.shape
    nu = n - p
    # print ('nu:', nu, 'p:', p, 'n:', n)
    if nu == 0:
        return np.linalg.solve(L.T, y)
    
    if W is not None and T is not None:
        y = y[:p] - T[:, :p].T @ (W.T @ y)
    x = np.linalg.solve(L[:, :p].T, y[:p])

    
    return x

