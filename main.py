import numpy as np
from csvd import csvd
from picard import picard
from shaw import shaw
import matplotlib.pyplot as plt

# plot shaw function
A, b, x = shaw(100)
plt.plot(x, label='x')
plt.show()
U, s, V = csvd(A)
eta = picard(U, s, b)
print("eta:", eta)


# Ex?ample usage
U = np.array([[0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
s = np.array([3, 2])
b = np.array([1, 2, 3])
eta = picard(U, s, b)
print("eta:", eta)
# print('U:\n', U)
# print('s:\n', s)
# print('V:\n', V)