import numpy as np
import time
from numba import njit, prange, set_num_threads, get_num_threads

# Set numba threads to 4
set_num_threads(4)
print("Numba using", get_num_threads(), "threads.\n")

# --------------------------
# Version 1: prange over i
# --------------------------
@njit(parallel=True)
def matmul_prange_i(X, Y):
    N, L, K = X.shape
    M = Y.shape[1]
    out = np.empty((N, L, M))
    for i in prange(N):
        for j in range(L):
            for m in range(M):
                val = 0.0
                for k in range(K):
                    val += X[i, j, k] * Y[k, m]
                out[i, j, m] = val
    return out

# --------------------------
# Version 2: prange over j
# --------------------------
@njit(parallel=True)
def matmul_prange_j(X, Y):
    N, L, K = X.shape
    M = Y.shape[1]
    out = np.empty((N, L, M))
    for i in range(N):
        x_i = X[i]
        for j in prange(L):
            for m in range(M):
                val = 0.0
                for k in range(K):
                    val += x_i[j, k] * Y[k, m]
                out[i, j, m] = val
    return out

# --------------------------
# Benchmark function
# --------------------------
def benchmark(name, func, X, Y):
    # Warm-up JIT
    print(X.flags)
    print(Y.flags)
    func(X, Y)
    start = time.perf_counter()
    result = func(X, Y)
    duration = time.perf_counter() - start
    print(f"{name}: {duration:.6f} seconds")
    return result

# --------------------------
# Test cases
# --------------------------
cases = [
    ("Case 1", (512, 90, 8), (8, 512)),
    ("Case 2", (64, 90, 8), (8, 64)),
]

for label, x_shape, y_shape in cases:
    print(f"--- {label} ---")
    X = np.random.rand(*x_shape)
    Y = np.random.rand(*y_shape)

    out1 = benchmark("prange over i", matmul_prange_i, X, Y)
    out2 = benchmark("prange over j", matmul_prange_j, X, Y)

    # Optional: verify correctness (light check)
    diff = np.max(np.abs(out1 - out2))
    print(f"Max difference between outputs: {diff:.6e}\n")

