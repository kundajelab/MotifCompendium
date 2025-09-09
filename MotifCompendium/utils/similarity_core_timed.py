# DEVELOPED BY SALIL DESHPANDE

import time

from numba import njit

import numpy as np

import MotifCompendium.utils.config as utils_config


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Get array module + prepare motifs
    start = time.time()
    motifsA_np = np.asarray(motifsA, dtype=np.float64)
    motifsB_np = np.asarray(motifsB, dtype=np.float64)
    # Normalize motifs
    motifsA_normalized = motifsA_np / np.linalg.norm(
        motifsA_np, axis=(1, 2), keepdims=True
    )
    motifsB_normalized = motifsB_np / np.linalg.norm(
        motifsB_np, axis=(1, 2), keepdims=True
    )
    del motifsA_np, motifsB_np  # Free up memory
    # Forward similarity
    sim_1, sim_1_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized
    )  # skew-symmetric alignment
    # Reverse complement
    motifsB_normalized_revcomp = _reverse_complement(motifsB_normalized)
    del motifsB_normalized  # Free up memory
    # Backward similarity
    sim_2, sim_2_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized_revcomp
    )  # symmetric alignment
    # Pick best similarity
    sim_12 = np.stack([sim_1, sim_2])
    sim = np.max(sim_12, axis=0)
    alignment_rc = np.argmax(sim_12, axis=0)
    alignment_h = np.where(alignment_rc == 0, sim_1_alignment, sim_2_alignment)
    # Guarantee similarity properties
    alignment_h[sim == 0] = (
        0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    )
    end = time.time()
    print(f"compute similarity and align: {end - start}")
    # assert(False)
    # Return
    if utils_config.get_use_gpu():
        sim = sim.get()
        alignment_rc = alignment_rc.get()
        alignment_h = alignment_h.get()
    return (
        sim.astype(np.single),
        alignment_rc.astype(np.bool_),
        alignment_h.astype(np.short),
    )


####################
# GLOBAL VARIABLES #
####################
_CUPY_IMPORT = None
_RIGHT_TENSOR = None
_LEFT_TENSOR = None


def _get_array_module():
    if utils_config.get_use_gpu():
        global _CUPY_IMPORT
        if _CUPY_IMPORT is None:
            import cupy as cp

            _CUPY_IMPORT = cp
        return _CUPY_IMPORT
    return np


#####################
# PRIVATE FUNCTIONS #
#####################
def _reverse_complement(motifs):
    """Computes the reverse complement of a (N, L, K) motif stack."""
    return motifs[:, ::-1, ::-1]


# @njit
def _tensor3_matmul_tensor2(x, y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    # start = time.time()
    # Old way
    N, L, K = x.shape
    M = y.shape[1]
    x_flat = np.reshape(x, (N * L, K))  # (NL, K)
    result = x_flat @ y  # (NL, M)
    return np.reshape(result, (N, L, M))  # (N, L, M)

    # Numba way 2
    # N, L, K = x.shape
    # M = y.shape[1]
    # out = np.zeros((N, L, M))
    # for i in range(N):
    #     out[i] = x[i] @ y
    # return out

    # Direct multiply
    # result = x@y
    # return result

    # Numba way
    # N, L, K = x.shape
    # M = y.shape[1]

    # out = np.zeros((N, L, M))
    # for n in range(N):
    #     for l in range(L):
    #         for m in range(M):
    #             for k in range(K):
    #                 out[n, l, m] += x[n, l, k] * y[k, m]
    # end = time.time()
    # print(f"_tensor3_matmul_tensor2: {end - start}")
    # return out


# @njit
def _compute_similarity(motif_set_1, motif_set_2):
    """Computes similarity and alignment for two sets of motifs."""
    # print("starting _compute_similarity")
    # start = time.time()
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape

    # result = np.zeros((N, 2*L-1, M))
    # for i in range(N):
    #     for h in range(2*L-1):
    #         if h < L:
    #             min_1 = 0
    #             max_1 = h+1
    #             min_2 = L-1-h
    #             max_2 = L
    #         else:
    #             min_1 = h - (L-1)
    #             max_1 = L
    #             min_2 = 0
    #             max_2 = 2*L-h-1
    #         motif_set_1_h = motif_set_1[i, min_1:max_1, :].copy()[np.newaxis, :, :]
    #         motif_set_2_h = motif_set_2[:, min_2:max_2, :].copy()
    #         for j in range(M):
    #             result[i, h, j] = np.sum(motif_set_1_h*motif_set_2_h[j, :, :])
    # # best_similarity = np.max(result, axis=1)
    # # best_alignments = np.argmax(result, axis=1) - (L - 1)
    # best_similarity, best_alignments = _max_and_argmax_along_axis1(result)
    # return best_similarity.T, best_alignments.T

    # assert L == L2
    # assert K == K2
    # Compute right side matrices
    # subtime = time.time()
    right_side_matrices = _compute_similarity_right_side(motif_set_1)  # (K, 3L-2, N)
    # subtime_end = time.time()
    # print(f"\t_compute_similarity_right_side: {subtime_end - subtime}")
    # Compute similarity per base
    # subtime = time.time()
    left_side_matrix = _compute_similarity_left_side(motif_set_2)
    # subtime_end = time.time()
    # print(f"\t_compute_similarity_left_side: {subtime_end - subtime}")

    # subtime = time.time()
    total_sum = _tensor3_matmul_tensor2(left_side_matrix[:, :, :, 0].copy(), right_side_matrices[0].copy())
    for i in range(1, K):
        total_sum += _tensor3_matmul_tensor2(left_side_matrix[:, :, :, i].copy(), right_side_matrices[i].copy())
    # subtime_end = time.time()
    # print(f"\tmatmul_time: {subtime_end - subtime}")
    # sims = []
    # left_side_matrix = _compute_similarity_left_side(motif_set_2)
    # for i in range(K):
    #     # left_side_matrix_i = _compute_similarity_left_side_i(
    #     #     motif_set_2[:, :, i]
        # )  # (M, 2L-1, 3L-2)
    #     # print(f"CHECK: {np.allclose(left_side_matrix[:, :, :, i], left_side_matrix_i)}")
    #     # print("mult1 method")
    #     # _tensor3_matmul_tensor2(left_side_matrix[:, :, :, i], right_side_matrices[i])
    #     # print("mult2 method")
    #     # _tensor3_matmul_tensor2(left_side_matrix_i, right_side_matrices[i])
    #     # print("mult3 method")
    #     # _tensor3_matmul_tensor2(left_side_matrix[:, :, :, i].copy(), right_side_matrices[i])
    #     sims.append(_tensor3_matmul_tensor2(left_side_matrix[:, :, :, i].copy(), right_side_matrices[i]))  # (M, 2L-1, N)
    # # for i in range(K):
    # #     left_side_matrix_i = _compute_similarity_left_side_i(
    # #         motif_set_2[:, :, i]
    # #     )  # (M, 2L-1, 3L-2)
    # #     sims.append(
    # #         _tensor3_matmul_tensor2(left_side_matrix_i, right_side_matrices[i])
    # #     )  # (M, 2L-1, N)
    # #     del left_side_matrix_i  # Free up memory

    # # Sum across ATCG
    # total_sum = np.zeros_like(sims[0], dtype=np.float64)  # (M, 2L-1, N)
    # for sim in sims:
    #     total_sum += sim
    # del sims  # Free up memory
    # Compute alignment
    # subtime = time.time()
    # best_similarity, best_alignments = _max_and_argmax_along_axis1(total_sum)
    best_similarity = np.max(total_sum, axis=1)
    best_alignments = np.argmax(total_sum, axis=1) - (L - 1)
    # subtime_end = time.time()
    # print(f"\talignment_time: {subtime_end - subtime}")
    # del total_sum  # Free up memory
    # assert best_similarity.shape == (M, N)
    # assert best_alignments.shape == (M, N)
    # end = time.time()
    # print(f"\t_compute_similarity: {end - start}")
    return best_similarity.T, best_alignments.T  # (N, M), (N, M)


# @njit
def _compute_similarity_left_side(motifs):
    # start = time.time()
    M, L, K = motifs.shape
    left_side_matrix = np.zeros((M, 2*L-1, 3*L-2, K))
    for i in range(M):
        for j in range(2*L-1):
            left_side_matrix[i, j, j:j+L, :] = motifs[i, :, :]
    # end = time.time()
    # print(f"\t_compute_similarity_left_side: {end - start}")
    return left_side_matrix


# @njit
def _compute_similarity_right_side(motifs):
    """Prepares the right side of the similarity calculation."""
    # start = time.time()

    N, L, K = motifs.shape  # (N, L, K)
    # right_side_matrices = np.zeros((N, 3*L-2, K))
    # right_side_matrices[:, L-1:2*L-1, :] = motifs

    right_side_matrices = np.zeros((K, 3*L-2, N))
    for i in range(N):
        right_side_matrices[:, L-1:2*L-1, i] = motifs[i].T


    # end = time.time()
    # print(f"\t_compute_similarity_right_side: {end - start}")
    return right_side_matrices  # (3L-2, N) K times


@njit(cache=True)
def _max_and_argmax_along_axis1(x):
    n, m, k = x.shape
    max_vals = np.empty((n, k), dtype=np.single)
    max_idxs = np.empty((n, k), dtype=np.short)
    for i in range(n):
        for j in range(k):
            max_val = x[i, 0, j]
            max_idx = 0
            for l in range(1, m):
                val = x[i, l, j]
                if val > max_val:
                    max_val = val
                    max_idx = l
            max_vals[i, j] = max_val
            max_idxs[i, j] = max_idx
    return max_vals, max_idxs