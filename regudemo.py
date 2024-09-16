import numpy as np
from csvd import csvd
from picard import picard
from shaw import shaw
from tikhonov import tikhonov
from tsvd import tsvd
from fil_fac import fil_fac
from lsqr_b import lsqr_b
from l_curve import l_curve
from plot_lc import plot_lc
from gcv import gcv
from i_laplace import i_laplace
from get_l import get_l
from cgsvd import cgsvd
from tgsvd import tgsvd
from ursell import ursell
from phillips import phillips
from discrep import discrep
from ncp import ncp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math



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
plt.show()   # uncomment to show plot
# plt.clf()


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

# % Part 3.  The L-curve
# % --------------------
# %
# % Plot the L-curves for Tikhonov regularization and for
# % LSQR for the "noisy" test problem from Part 1.
# %
# % Notice the similarity between the two L-curves and thus,
# % in turn, by the two methods.

plt.figure()
plt.subplot(1,2,1); l_curve(U,s,b, method='tsvd'); #axis([1e-3,1,1,1e3])
plt.subplot(1,2,2); plot_lc(rho,eta,'o'); #axis([1e-3,1,1,1e3])
plt.show() # uncomment to show plot



# Part 4.  Regularization parameters
# ----------------------------------
#
# Use the L-curve criterion and GCV to determine the regularization
# parameters for Tikhonov regularization and truncated SVD.
#
# Then compute the relative errors for the four solutions.
plt.figure()
plt.subplot(1,2,1)
lambda_l, tmp0, tmp1, tmp2 = l_curve(U,s,b)   # axis([1e-3,1,1,1e3]),      pause, clf
plt.subplot(1,2,2)
k_l, dumm0, dumm1, dumm1 = l_curve(U,s,b,'tsvd') # axis([1e-3,1,1,1e3]),      pause, clf
plt.show() # uncomment to show plot
lambda_gcv, dumm0, dumm1 = gcv(U,s,b)     # axis([1e-6,1,1e-9,1e-1]),  pause, clf
plt.show() # uncomment to show plot
k_gcv, dumm0, dumm1 = gcv(U,s,b,'tsvd')   # axis([0,20,1e-9,1e-1]),    pause, clf
plt.show() # uncomment to show plot
x_tikh_l, dumm0, dumm1   = tikhonov(U,s,V,b,lambda_l)
x_tikh_gcv, dumm0, dumm1= tikhonov(U,s,V,b,lambda_gcv)

if math.isnan(k_l):
    x_tsvd_l = np.zeros((32,1)) # Spline Toolbox not available.
else:
    k_l = int(k_l) # Convert from float to int required for tsvd
    x_tsvd_l, dumm0, dumm1 = tsvd(U,s,V,b,k_l)

k_gcv = int(k_gcv) # Convert from float to int required for tsvd
x_tsvd_gcv, dumm0, dumm1 = tsvd(U,s,V,b,k_gcv)
print(np.array([np.linalg.norm(x-x_tikh_l),np.linalg.norm(x-x_tikh_gcv),np.linalg.norm(x-x_tsvd_l),np.linalg.norm(x-x_tsvd_gcv)])/np.linalg.norm(x))
plt.show()

# Part 5.  Standard form versus general form
# ------------------------------------------
#
# Generate a new test problem: inverse Laplace transformation
# with white noise in the right-hand side.
#
# For the general-form regularization, choose minimization of
# the first derivative.
#
# First display some left singular vectors of SVD and GSVD; then
# compare truncated SVD solutions with truncated GSVD solutions.
# Notice that TSVD cannot reproduce the asymptotic part of the
# solution in the right part of the figure.

n = 16; 
A,b,x, dumm0 = i_laplace(n,2)
b = b + 1e-4*np.random.randn(n)
L, dumm0 = get_l(n,1)
U,s,V = csvd(A) 
L = L.toarray() # Convert L to full matrix
UU,sm,XX, dumm0, dumm0 = cgsvd(A,L)
# print('UU.shape:', UU)
# print('sm.shape:', sm)
# print('XX.shape:', XX)
# plt.figure()
I = 1
for i in [3,6,9,12]:
    plt.subplot(2,2,I)
    plt.plot(range(1,n+1), V[:,i])
    plt.axis([1,n,-1,1])
    plt.xlabel('i = ' + str(i))
    I += 1
plt.subplot(2,2,1)
plt.text(12,1.2,'Right singular vectors V(:,i)')
plt.show() # uncomment to show plot

# print('XX.shape:', XX.shape)
# print('V.shape:', V.shape)
I = 1
for i in [n-2,n-5,n-8,n-11]:
    plt.subplot(2,2,I)
    plt.plot(range(1,n+1), XX[:,i])
    plt.axis([1,n,-1,1])
    plt.xlabel('i = ' + str(i))
    I += 1
plt.subplot(2,2,1)
plt.text(10,1.2,'Right generalized singular vectors XX(:,i)')
plt.show() # uncomment to show plot

k_tsvd = 7
k_tgsvd = 6
X_I, dumm0, dumm1 = tsvd(U,s,V,b, np.arange(1,k_tsvd+1))
X_L, dumm0, dumm1 = tgsvd(UU,sm,XX,b,np.arange(1,k_tgsvd+1))

# print('X_I.shape:', X_I.shape)
# print('X_L.shape:', X_L.shape)

# plt.figure()
plt.subplot(2,1,1)
plt.plot(range(1,n+1), X_I, range(1,n+1), x, 'x')
plt.axis([1,n,0,1.2])
plt.xlabel('L = I')
plt.subplot(2,1,2)
plt.plot(range(1,n+1), X_L, range(1,n+1), x, 'x')
plt.axis([1,n,0,1.2])
plt.xlabel('L != I')
# plt.show() # uncomment to show plot




# Part 6.  No square integrable solution
# --------------------------------------
#
# In the last example there is no square integrable solution to
# the underlying integral equation (NB: no noise is added).
#
# Notice that the discrete Picard condition does not seem to
# be satisfied, which indicates trouble!

A,b = ursell(32) 
U,s,V = csvd(A)
picard(U,s,b); 
plt.show() # uncomment to show plot
#  This concludes the demo.



plt.figure()
x_delta, lambda_ = discrep(U, s, V, b, [1e-3])
plt.subplot(1, 2, 1)
plt.plot(x_delta)
plt.title('x_delta')
plt.subplot(1, 2, 2)
plt.plot(lambda_)
plt.title('lambda')
plt.show() # uncomment to show plot

# example using  ncp
regmin, dist, regparam = ncp(U, s, b)
plt.show() # uncomment to show plot

# phillips example
A, b, x = phillips(100)
# print('b:', b)
plt.subplot(1, 3, 1)
plt.plot(x)
plt.title('x')
plt.subplot(1, 3, 2)
plt.plot(b)
plt.title('b')
plt.subplot(1, 3, 3)
plt.spy(A)
plt.title('A')
plt.show() # uncomment to show plot
