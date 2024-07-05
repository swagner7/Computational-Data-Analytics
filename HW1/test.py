import numpy as np


a = np.array([[0,1,1],[1,2,3],[3,4,5]])

print(a.shape)

b = np.array([[9],[9],[9]])

print(b.shape)

x = np.append(a, b, 1)

print(x)