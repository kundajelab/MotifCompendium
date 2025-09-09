import numpy as np
from numba import njit
import time

# Create large arrays
N, M = 2000, 2000
A_c = np.random.rand(N, M)           # C-contiguous
A_f = np.asfortranarray(A_c)         # F-contiguous (same values)
print(A_c.flags)
print(A_c.T.flags)
print(A_f.flags)
print(A_f.T.flags)

# JIT-compiled row-major loop (best for C-contiguous)
@njit
def sum_c_order(A):
    total = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            total += A[i, j]
    return total

# JIT-compiled col-major loop (best for F-contiguous)
@njit
def sum_f_order(A):
    total = 0.0
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            total += A[i, j]
    return total

# Warm-up JIT
sum_c_order(A_c)
sum_f_order(A_f)

# Benchmark C-order access on C-contiguous
start = time.time()
sum_c_order(A_c)
print("C-order loop on C-contig: {:.3f} sec".format(time.time() - start))

# Benchmark F-order access on C-contiguous (bad)
start = time.time()
sum_f_order(A_c)
print("F-order loop on C-contig: {:.3f} sec".format(time.time() - start))

# Benchmark F-order loop on F-contiguous (good)
start = time.time()
sum_f_order(A_f)
print("F-order loop on F-contig: {:.3f} sec".format(time.time() - start))

# Benchmark C-order loop on F-contiguous (bad)
start = time.time()
sum_c_order(A_f)
print("C-order loop on F-contig: {:.3f} sec".format(time.time() - start))
