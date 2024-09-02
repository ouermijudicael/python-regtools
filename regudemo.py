import numpy as np
from csvd import csvd
from picard import picard
from shaw import shaw
from tikhonov import tikhonov
from fil_fac import fil_fac
from lsqr_b import lsqr_b
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Part 1. The discrete Picard condition
# --------------------------------------

# First generate a "pure" test problem where only rounding
# errors are present. Then generate another "noisy" test
# problem by adding white noise to the right-hand side.

# Next compute the SVD of the coefficient matrix A.

# Finally, check the Picard condition for both test problems
# graphically. Notice that for both problems the condition is
# indeed satisfied for the coefficients corresponding to the
# larger singular values, while the noise eventually starts to
# dominate.

A, b_bar, x = shaw(32)
# print('A.shape:', A.shape, 'b_bar.shape:', b_bar.shape, 'x.shape:', x.shape)
np.random.seed(41997)
e = 1e-3 * np.random.randn(len(b_bar))
b = b_bar + e
U, s, V = csvd(A)

plt.subplot(2, 1, 1)
picard(U, s, b_bar)
plt.subplot(2, 1, 2)
picard(U, s, b)
plt.pause(0.001)
plt.show()
plt.clf()


# Part 2.  Filter factors
# -----------------------
#
# Compute regularized solutions to the "noisy" problem from Part 1 
# by means of Tikhonov's method and LSQR without reorthogonalization.
# Also, compute the corresponding filter factors.
#
# A surface (or mesh) plot of the solutions clearly shows their dependence
# on the regularization parameter (lambda or the iteration number).
#
lambda_ = np.array([1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5])
X_tikh = tikhonov(U,s,V,b,lambda_)
F_tikh = fil_fac(s,lambda_)
iter = 30 
reorth = 0
[X_lsqr,rho,eta,F_lsqr] = lsqr_b(A,b,iter,reorth,s)

ny, nx = X_tikh[0].shape
X, Y = np.meshgrid(range(nx), range(ny))
fig = plt.figure()
plt.subplot(2, 2, 1)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(X, Y, X_tikh[0])
plt.title('Tikhonov solutions')
plt.gca().invert_yaxis()

plt.subplot(2, 2, 2)
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(X, Y, np.log10(F_tikh))
plt.title('Tikh filter factors, log scale')
plt.gca().invert_yaxis()

print('X_lsqr.shape:', X_lsqr.shape)
nx, ny = X_lsqr.shape
X, Y = np.meshgrid(range(ny), range(nx))
plt.subplot(2, 2, 3)
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(X, Y, X_lsqr)
plt.title('LSQR solutions')
plt.gca().invert_yaxis()

plt.subplot(2, 2, 4)
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(X, Y, np.log10(F_lsqr))
plt.title('LSQR filter factors, log scale')
plt.gca().invert_yaxis()


plt.show()
