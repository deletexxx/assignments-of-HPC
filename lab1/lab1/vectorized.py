import time
import numpy as np

A = np.random.rand(1024,1024)
B = np.random.rand(1024,1024)
C = np.zeros((1024,1024))

# print(A)
# print(B)
B1=np.transpose (B)
tic = time.time()
for m in range(1024):
    for n in range(1024):
        C[m][n]=np.dot(A[m],B1[n])
toc = time.time()
print ("Vectorized : time = " + str((toc-tic)) +" s")

# tic = time.time()
# C=np.dot(A,B)
# toc = time.time()
# print ("Vectorized : time = " + str((toc-tic)) +"s")
# print(C)
