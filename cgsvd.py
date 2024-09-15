import numpy as np
from scipy.linalg import svd
from gsvd import gsvd

def cgsvd(A,L):
# %CGSVD Compact generalized SVD (GSVD) of a matrix pair in regularization problems.
# %
# % sm = cgsvd(A,L)
# % [U,sm,X,V] = cgsvd(A,L) ,  sm = [sigma,mu]
# % [U,sm,X,V,W] = cgsvd(A,L) ,  sm = [sigma,mu]
# %
# % Computes the generalized SVD of the matrix pair (A,L). The dimensions of
# % A and L must be such that [A;L] does not have fewer rows than columns.
# %
# % If m >= n >= p then the GSVD has the form:
# %    [ A ] = [ U  0 ]*[ diag(sigma)      0    ]*inv(X)
# %    [ L ]   [ 0  V ] [      0       eye(n-p) ]
# %                     [  diag(mu)        0    ]
# % where
# %    U  is  m-by-n ,    sigma  is  p-by-1
# %    V  is  p-by-p ,    mu     is  p-by-1
# %    X  is  n-by-n .
# %
# % Otherwise the GSVD has a more complicated form (see manual for details).
# %
# % A possible fifth output argument returns W = inv(X).
 
# % Reference: C. F. Van Loan, "Computing the CS and the generalized 
# % singular value decomposition", Numer. Math. 46 (1985), 479-491. 
 
# % Per Christian Hansen, DTU Compute, August 22, 2009. 

    # Initialization. 
    m,n = A.shape
    p,n1 = L.shape

    if n1 != n:
        raise ValueError('The number of columns in A and L must be the same.')
    
    if (m + p) < n:
        raise ValueError('The combined rows of A and L must be at least n.')
    
    # Call Matlab's GSVD routine.

    # print('TAJO: A.shape:', A.shape, 'L.shape:', L.shape)
    U,V,W,C,S = gsvd(A,L)
    # print('C:', C)
    # print('S:', S)

    if m >= n:
        # The overdetermined or square case.
        q = min(p,n)
        sm = np.zeros((q,2))
        sm[:,0] = np.diag(C[:q,:q])
        sm[:,1] = np.diag(S[:q,:q])
        # sm = np.hvstack([np.diag(C[:q,:q]),np.diag(S[:q,:q])])
        X = np.linalg.inv(W.T)
    else:
        # The underdetermined case.
        sm = np.zeros((m+p-n,2))
        sm[:m+p-n,0] = np.diag(C[:m+p-n,n-m:])
        sm[:m+p-n,1] = np.diag(S[n-m:,n-m:])
        # sm = np.hstack([np.diag(C[:m+p-n,n-m:]),np.diag(S[n-m:,n-m:])])
        X = np.linalg.inv(W.T)
        X = X[:,n-m:]

    return U, sm, X, V, W
