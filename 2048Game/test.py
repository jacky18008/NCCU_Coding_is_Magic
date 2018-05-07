import numpy as np 

A = np.ones(shape=(4, 4))
B = np.ones(shape=(4, 4))

A[3, 1] = 0
print(B)
print(np.where(A == B))
C = np.where(A == B, A+B, 0)
D = np.where(C == 0, B, C)
#B[:, 0:3] = A[:, 0:3]
print(D)
