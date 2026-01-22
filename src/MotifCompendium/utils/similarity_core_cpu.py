# DEVELOPED BY SALIL DESHPANDE

import os

os.environ["NUMBA_CACHE_DIR"] = (
    f"{os.path.dirname(os.path.abspath(__file__))}/.numba_cache"
)

from numba import njit
import numpy as np


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Normalize motifs
    # Normalize motifs
    motifsA_normalized = _normalize_mtx(motifsA)
    motifsB_normalized = _normalize_mtx(motifsB)
    # Forward similarity
    sim_1, sim_1_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized
    )  # skew-symmetric alignment
    # Reverse complement
    motifsB_normalized_revcomp = _reverse_complement(motifsB_normalized)
    # Backward similarity
    sim_2, sim_2_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized_revcomp
    )  # symmetric alignment
    # Pick best similarity
    sim, alignment_rc, alignment_h = _get_alignment_over_rc(
        sim_1, sim_1_alignment, sim_2, sim_2_alignment
    )
    # Guarantee similarity properties
    alignment_h[sim == 0] = (
        0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    )
    # Return
    return (
        sim.astype(np.single),
        alignment_rc.astype(np.bool_),
        alignment_h.astype(np.short),
    )


#####################
# PRIVATE FUNCTIONS #
#####################
@njit(cache=True)
def _normalize_mtx(X):
    N, L, K = X.shape
    X_normalized = np.empty_like(X)
    for i in range(N):
        norm = 0.0
        for j in range(L):
            for k in range(K):
                norm += X[i, j, k] ** 2
        invnorm = 1.0 / np.sqrt(norm)
        for j in range(L):
            for k in range(K):
                X_normalized[i, j, k] = X[i, j, k] * invnorm
    return X_normalized


@njit(cache=True)
def _compute_similarity(motif_set_1, motif_set_2):
    """Computes similarity and alignment for two sets of motifs."""
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape
    assert L == L2
    assert K == K2
    # Transpose for efficiency
    transpose = N < M
    if transpose:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
    # Compute right side matrices
    right_side_matrix = _compute_similarity_right_side(motif_set_1)  # (K, 3L-2, N)
    # Compute left side matrices
    left_side_matrix = _compute_similarity_left_side(motif_set_2)  # (K, M, 2L-1, 3L-2)
    # Compute similarity
    total_sum = _tensor3_matmul_tensor2(
        left_side_matrix[0], right_side_matrix[0]
    )  # (M, 2L-1, N)
    for i in range(1, K):
        total_sum += _tensor3_matmul_tensor2(
            left_side_matrix[i].copy(), right_side_matrix[i]
        )  # (M, 2L-1, N)
    # Compute best similarity and alignments
    total_sum = np.transpose(total_sum, (0, 2, 1))  # (N, M, 2L-1)
    best_similarity, best_alignments = _max_and_shifted_argmax_along_axis2(
        total_sum, shift=-(L - 1)
    )  # (N, M), (N, M)
    # Undo transpose if needed
    if transpose:
        best_similarity = best_similarity.T
        best_alignments = (
            -best_alignments.T
        )  # negative because transposing flips alignment
    assert best_similarity.shape == (M, N)
    assert best_alignments.shape == (M, N)
    return best_similarity.T, best_alignments.T  # (N, M), (N, M)


@njit(cache=True)
def _compute_similarity_left_side(motifs):
    """Prepares the left side of the similarity calculation."""
    M, L, K = motifs.shape
    left_side_matrix = np.zeros((K, M, 2 * L - 1, 3 * L - 2))
    for i in range(M):
        motifs_i = motifs[i]
        for k in range(K):
            motifs_i_k = motifs_i[:, k]
            for j in range(2 * L - 1):
                left_side_matrix[k, i, j, j : j + L] = motifs_i_k
    return left_side_matrix  # (K, M, 2L-1, 3L-2)


@njit(cache=True)
def _compute_similarity_right_side(motifs):
    """Prepares the right side of the similarity calculation."""
    N, L, K = motifs.shape  # (N, L, K)
    right_side_matrices = np.zeros((K, 3 * L - 2, N))
    for i in range(N):
        right_side_matrices[:, L - 1 : 2 * L - 1, i] = motifs[i].T
    return right_side_matrices  # (K, 3L-2, N)


@njit(cache=True)
def _tensor3_matmul_tensor2(X, Y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    # Old way
    N, L, K = X.shape
    M = Y.shape[1]
    out = np.zeros((N, L, M))  # (N, L, M)
    for n in range(N):
        for l in range(L):
            input_l = X[n, l]
            out_l = np.zeros(M)
            for k in range(K):
                for m in range(M):
                    out_l[m] += input_l[k] * Y[k, m]
            out[n, l] = out_l
    return out


@njit
def _max_and_shifted_argmax_along_axis2(X, shift=0):
    N, M, L = X.shape
    max_vals = np.empty((N, M), dtype=np.float64)
    max_idxs = np.empty((N, M), dtype=np.short)
    for i in range(N):
        for j in range(M):
            max_val = -1.0
            max_idx = 0
            for l in range(L):
                if X[i, j, l] > max_val:
                    max_val = X[i, j, l]
                    max_idx = l + shift
            max_vals[i, j] = max_val
            max_idxs[i, j] = max_idx
    return max_vals, max_idxs


@njit(cache=True)
def _reverse_complement(motifs):
    """Computes the reverse complement of a (N, L, K) motif stack."""
    return motifs[:, ::-1, ::-1]


@njit
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
