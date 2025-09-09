# DEVELOPED BY SALIL DESHPANDE

import time

from numba import njit, guvectorize
import numba

import numpy as np

import MotifCompendium.utils.config as utils_config


numba.set_num_threads(16)

####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Get array module + prepare motifs
    # print("starting")
    start = time.time()
    print(motifsA.shape, motifsB.shape)
    results = compute_similarity_and_align_inner(motifsA, motifsB)
    end = time.time()
    print(results[0].shape, results[1].shape, results[2].shape)
    print(f"compute similarity and align: {end - start}")
    return results


# @njit
def compute_similarity_and_align_inner(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Normalize motifs
    # print("normalizing A")
    motifsA_normalized = _normalize_mtx(motifsA)
    # print("normalizing B")
    motifsB_normalized = _normalize_mtx(motifsB)
    # Forward similarity
    # print("forward similarity")
    sim_1, sim_1_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized
    )  # skew-symmetric alignment
    # assert(False)
    # Reverse complement
    motifsB_normalized_revcomp = _reverse_complement(motifsB_normalized)
    # Backward similarity
    sim_2, sim_2_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized_revcomp
    )  # symmetric alignment
    # Pick best similarity
    sim, alignment_rc, alignment_h = _get_alignment_over_rc(sim_1, sim_1_alignment, sim_2, sim_2_alignment)
    # Return
    return sim, alignment_rc, alignment_h
    # return (
    #     sim.astype(np.single),
    #     alignment_rc.astype(np.bool_),
    #     alignment_h.astype(np.short),
    # )


#####################
# PRIVATE FUNCTIONS #
#####################
@njit(parallel=True)
def _normalize_mtx(X):
    N, L, K = X.shape
    X_normalized = np.empty_like(X)
    for i in numba.prange(N):
        norm = 0.0
        for j in range(L):
            for k in range(K):
                norm += X[i, j, k] ** 2
        invnorm = 1/np.sqrt(norm)
        for j in range(L):
            for k in range(K):
                X_normalized[i, j, k] = X[i, j, k] * invnorm
    return X_normalized


# @njit
def _compute_similarity(motif_set_1, motif_set_2):
    """Computes similarity and alignment for two sets of motifs."""
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape
    # Compute right side matrices
    # print("\tcomputing right side")
    right_side_matrices = _compute_similarity_right_side(motif_set_1)  # (K, 3L-2, N)
    # Compute left side matrices
    # print("\tcomputing left side")
    left_side_matrix = _compute_similarity_left_side(motif_set_2)# (K, M, 2L-1, 3L-2)
    # Compute similarity
    # print("\tmatmuls")
    # q = left_side_matrix[:, :, :, 0].copy()
    # print(q.shape)
    # print(q.flags)
    # q = right_side_matrices[0].T.copy()
    # print(q.shape)
    # print(q.flags)
    total_sum = _tensor3_matmul_tensor2(left_side_matrix[0], right_side_matrices[0])
    for i in range(1, K):
        total_sum += _tensor3_matmul_tensor2(left_side_matrix[i].copy(), right_side_matrices[i])
    # Compute best similarity and alignments
    # print("\tmax and argmax")
    total_sum = np.transpose(total_sum, (0, 2, 1))  # (N, M, 2L-1)
    best_similarity, best_alignments = _max_and_argmax_along_axis2(total_sum)
    # Return
    return best_similarity.T, best_alignments.T  # (N, M), (N, M)


@njit(parallel=True)
def _compute_similarity_left_side(motifs):
    """Prepares the left side of the similarity calculation."""
    M, L, K = motifs.shape
    left_side_matrix = np.zeros((K, M, 2*L-1, 3*L-2))
    for i in numba.prange(M):
        motifs_i = motifs[i]
        for k in range(K):
            motifs_i_k = motifs_i[:, k]
            for j in range(2*L-1):
                left_side_matrix[k, i, j, j:j+L] = motifs_i_k
    
    return left_side_matrix # (K, M, 2L-1, 3L-2)


@njit(parallel=True)
def _compute_similarity_right_side(motifs):
    """Prepares the right side of the similarity calculation."""
    N, L, K = motifs.shape  # (N, L, K)
    right_side_matrices = np.zeros((K, 3*L-2, N))
    for i in numba.prange(N):
        right_side_matrices[:, L-1:2*L-1, i] = motifs[i].T
    return right_side_matrices  # (K, 3L-2, N)


@njit(parallel=True)
def _tensor3_matmul_tensor2(X, Y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    # Old way
    N, L, K = X.shape
    M = Y.shape[1]
    out = np.zeros((N, L, M))  # (N, L, M)
    for n in numba.prange(N):
        for l in range(L):
            input_l = X[n, l]
            out_l = np.zeros(M)
            for k in range(K):
                for m in range(M):
                    out_l[m] += input_l[k] * Y[k, m]
            out[n, l] = out_l
    return out


'''
@njit(parallel=True)
def _max_and_argmax_along_axis2(X):
    N, M, L = X.shape
    max_vals = np.empty((N, M), dtype=np.single)
    max_idxs = np.empty((N, M), dtype=np.short)
    for i in range(N):
        for j in range(M):
            max_val = -1.0
            max_idx = 0
            for l in range(L):
                if X[i, j, l] > max_val:
                    max_val = X[i, j, l]
                    max_idx = l
            max_vals[i, j] = max_val
            max_idxs[i, j] = max_idx
    return max_vals, max_idxs
'''


@guvectorize(
    ["void(float64[:], float32[:], int16[:])"],
    "(l)->(),()",
    target="parallel"
)
def _max_and_argmax_along_axis2(vec, max_out, argmax_out):
    max_val = vec[0]
    argmax = 0
    for i in range(1, vec.shape[0]):
        if vec[i] > max_val:
            max_val = vec[i]
            argmax = i
    max_out[0] = max_val
    argmax_out[0] = argmax


@njit
def _reverse_complement(motifs):
    """Computes the reverse complement of a (N, L, K) motif stack."""
    return motifs[:, ::-1, ::-1]


'''
def _get_alignment_over_rc(sim_1, sim_1_alignment, sim_2, sim_2_alignment):
    N, M = sim_1.shape
    sim = np.empty((N, M), dtype=np.single)
    alignment_rc = np.empty((N, M), dtype=np.bool_)
    alignment_h = np.empty((N, M), dtype=np.short)

    for i in range(N):
        for j in range(M):
            if sim_1[i, j] > sim_2[i, j]:
                sim[i, j] = sim_1[i, j]
                alignment_rc[i, j] = False
                alignment_h[i, j] = sim_1_alignment[i, j]
            else:
                sim[i, j] = sim_2[i, j]
                alignment_rc[i, j] = True
                alignment_h[i, j] = sim_2_alignment[i, j]
    return sim, alignment_rc, alignment_h
'''

@guvectorize(
    ["void(float32, int16, float32, int16, float32[:], boolean[:], int16[:])"],
    "(),(),(),()->(),(),()",
    target="parallel"
)
def _get_alignment_over_rc(sim_1, sim_1_alignment, sim_2, sim_2_alignment, sim, alignment_rc, alignment_h):
    if sim_1 > sim_2:
        sim[0] = sim_1
        alignment_rc[0] = False
        alignment_h[0] = sim_1_alignment
    else:
        sim[0] = sim_2
        alignment_rc[0] = True
        alignment_h[0] = sim_2_alignment