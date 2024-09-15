import numpy as np
def i_laplace(n, example=1):
#    
# function [A,b,x,t] = i_laplace(n,example)
# I_LAPLACE Test problem: inverse Laplace transformation.
# %
# % [A,b,x,t] = i_laplace(n,example)
# %
# % Discretization of the inverse Laplace transformation by means of
# % Gauss-Laguerre quadrature.  The kernel K is given by
# %    K(s,t) = exp(-s*t) ,
# % and both integration intervals are [0,inf).
# %
# % The following examples are implemented, where f denotes
# % the solution, and g denotes the right-hand side:
# %    1: f(t) = exp(-t/2),        g(s) = 1/(s + 0.5)
# %    2: f(t) = 1 - exp(-t/2),    g(s) = 1/s - 1/(s + 0.5)
# %    3: f(t) = t^2*exp(-t/2),    g(s) = 2/(s + 0.5)^3
# %    4: f(t) = | 0 , t <= 2,     g(s) = exp(-2*s)/s.
# %              | 1 , t >  2
# %
# % The quadrature points are returned in the vector t.

# % Reference: J. M. Varah, "Pitfalls in the numerical solution of linear
# % ill-posed problems", SIAM J. Sci. Stat. Comput. 4 (1983), 164-176.

# % Per Christian Hansen, IMM, Oct. 21, 2006.

    # Initialization.
    if n <= 0:
        raise ValueError('The order n must be positive')

    # Compute equidistand collocation points s.
    s = 10/n * np.arange(1,n+1)

    # Compute abscissas t and weights v from the eigensystem of the
    # symmetric tridiagonal system derived from the recurrence
    # relation for the Laguerre polynomials.  Sorting of the
    # eigenvalues and -vectors is necessary.
    t = np.diag(2*(np.arange(1,n+1))-1) - np.diag(np.arange(1,n),1) - np.diag(np.arange(1,n),-1)
    t, Q = np.linalg.eig(t)
    indx = np.argsort(t)
    t = np.sort(t)
    v =  np.abs(Q[0, indx])
    nz = np.where(v!=0)


    # Set up the coefficient matrix A.  Due to limitations caused
    # by finite-precision arithmetic, A has zero columns if n > 195.
    A = np.zeros((n,n))
    for i in range(n):
        for j in nz:
            A[i,j] = (1-s[i])*t[j] + 2*np.log(v[j])
    A[:,nz] = np.exp(A[:,nz])

    # Compute the right-hand side b and the solution x by means of
    # simple collocation.
    if example == 1:
        b = 1/(s + .5)
        x = np.exp(-t/2)
    elif example == 2:
        b = 1/s - 1/(s + .5)
        x = 1 - np.exp(-t/2)
    elif example == 3:
        b = 2/((s + .5)**3)
        x = (t**2)*np.exp(-t/2)
    elif example == 4:
        b = np.exp(-2*s)/s
        f = np.where(t<=2)
        x = np.ones(n)
        x[f] = np.zeros(len(f))
    else:
        raise ValueError('Illegal example')
    
    return A,b,x,t
