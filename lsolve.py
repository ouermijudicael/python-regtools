def lsolve(L, y, W, T):
# %LSOLVE Utility routine for "preconditioned" iterative methods.
# %
# % x = lsolve(L,y,W,T)
# %
# % Computes the vector
# %    x = L_p*y
# % where L_p is the A-weighted generalized inverse of L.
# %
# % Here, L is a p-by-n matrix, W holds a basis for the null space of L,
# % and T is a utility matrix which should be computed by routine pinit.
# %
# % Alternatively, L is square, and W and T are not needed.
# %
# % Notice that x and y may be matrices, in which case
# %    x(:,i) = L_p*y(:,i) .

# % Reference: P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems.
# % Numerical Aspects of Linear Inversion", SIAM, Philadelphia, 1997.


    import numpy as np
    p, n = L.shape
    nu = n - p
    ly = y.shape[1]

    if nu == 0:
        return np.linalg.solve(L, y)

    x = np.linalg.solve(L[:, :p], y)
    x = np.vstack([x, np.zeros((nu, ly))]) - W @ (T[:, :p] @ x)

    return x
