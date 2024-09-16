import numpy as np

def gravity(n, example=1, a=0, b=1, d=0.25):
    """
    Discretization of a 1-D model problem in gravity surveying, in which
    a mass distribution f(t) is located at depth d, while the vertical
    component of the gravity field g(s) is measured at the surface.

    The problem is a first-kind Fredholm integral equation with kernel:
        K(s,t) = d * (d^2 + (s-t)^2)^(-3/2)

    Parameters:
    n       : Number of points for discretization.
    example : Defines the type of function for x (default: 1).
    a       : Left boundary of the s integration interval (default: 0).
    b       : Right boundary of the s integration interval (default: 1).
    d       : Depth at which the magnetic deposit is located (default: 0.25).

    Returns:
    A       : Discretized matrix for the Fredholm equation.
    b       : Right-hand side vector.
    x       : Solution vector.
    """

    # Set up abscissas and matrix
    dt = 1 / n
    ds = (b - a) / n
    t = dt * (np.arange(1, n + 1) - 0.5)
    s = a + ds * (np.arange(1, n + 1) - 0.5)
    T, S = np.meshgrid(t, s)
    A = dt * d * np.ones((n, n)) / (d**2 + (S - T)**2)**(3 / 2)

    # Set up solution vector and right-hand side
    nt = round(n / 3)
    nn = round(n * 7 / 8)
    x = np.ones(n)

    if example == 1:
        x = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)
    elif example == 2:
        x[:nt] = (2 / nt) * np.arange(1, nt + 1)
        x[nt:nn] = ((2 * nn - nt) - np.arange(nt + 1, nn + 1)) / (nn - nt)
        x[nn:] = (n - np.arange(nn + 1, n + 1)) / (n - nn)
    elif example == 3:
        x[:nt] = 2 * np.ones(nt)
    else:
        raise ValueError('Illegal value of example')

    b = A @ x
    return A, b, x
