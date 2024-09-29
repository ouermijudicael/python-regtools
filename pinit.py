def pinit(W,A,b=None):
    # Initialization for `preconditioning' of general-form problems.
    # Here, W holds a basis for the null space of L.
    #
    # Determines the matrix T needed in the iterative routines for
    # treating regularization problems in general form.
    #
    # If b is also specified then x_0, the component of the solution in
    # the null space of L, is also computed.
    #
    # Reference: P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems.
    # Numerical Aspects of Linear Inversion", SIAM, Philadelphia, 1997.
    #

    import numpy as np
    n,nu = W.shape

    # Special treatment of square L.
    if nu==0:
        T = np.array([])
        x_0 = np.zeros(n)
        return T,x_0

    # Compute T.
    S = np.linalg.pinv(A @ W)
    T = S @ A

    # If required, also compute x_0.
    if b is not None:
        x_0 = W @ (S @ b)
        return T,x_0
    else:
        return T, None

