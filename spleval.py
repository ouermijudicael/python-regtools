import numpy as np
from scipy.interpolate import PPoly

def spleval(f, npoints=300):
    """
    Evaluation of a spline or spline curve.

    Parameters:
    f : PPoly
        Spline or spline curve to evaluate.
    npoints : int, optional
        Number of points to evaluate. Default is 300.

    Returns:
    points : ndarray
        Points on the given spline or spline curve.
    """
    if isinstance(f, PPoly):
        breaks = f.x
        coefs = f.c
        l = len(breaks) - 1
        k = coefs.shape[0] - 1
        d = coefs.shape[1]
    else:
        raise ValueError('Input must be a PPoly object')

    x = np.linspace(breaks[0], breaks[-1], npoints)
    v = f(x)

    if d == 1:
        points = np.vstack((x, v))
    else:
        points = v

    return points