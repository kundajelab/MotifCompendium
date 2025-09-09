# DEVELOPED BY SALIL DESHPANDE

import time

from numba import njit
import numba

import numpy as np

import MotifCompendium.utils.config as utils_config


# numba.set_num_threads(4)

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
    results = compute_similarity_and_align_inner(motifsA, motifsB)
    end = time.time()
    print(f"compute similarity and align: {end - start}")
    return results


@njit
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
@njit
def _normalize_mtx(X):
    N, L, K = X.shape
    X_normalized = np.empty_like(X)
    for i in range(N):
        norm = 0.0
        for j in range(L):
            for k in range(K):
                norm += X[i, j, k] ** 2
        norm = np.sqrt(norm)
        for j in range(L):
            for k in range(K):
                X_normalized[i, j, k] = X[i, j, k] / norm
    return X_normalized


@njit
def _compute_similarity(motif_set_1, motif_set_2):
    """Computes similarity and alignment for two sets of motifs."""
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape

    out = np.zeros((N, M, 2*L-1))
    for i in range(L):
        for j in range(L):
            out[:, :, i+j] += _cross_product(motif_set_1[:, i].copy(), motif_set_2[:, j].copy())
    best_similarity, best_alignments = _max_and_argmax_along_axis2(out)
    return best_similarity, best_alignments


@njit
def _cross_product(X, Y):
    N, K = X.shape
    M = Y.shape[0]
    out = np.empty((N, M), dtype=X.dtype)
    
    for i in range(N):  # Parallel over rows of X
        for j in range(M):  # Rows of Y → cols of Y.T
            s = 0.0
            for k in range(K):
                s += X[i, k] * Y[j, k]
            out[i, j] = s
    return out


@njit
def _max_and_argmax_along_axis2(X):
    N, M, L = X.shape
    max_vals = np.empty((N, M), dtype=np.single)
    max_idxs = np.empty((N, M), dtype=np.short)
    for n in range(N):
        for m in range(M):
            max_val = -1.0
            max_idx = 0
            for l in range(L):
                if X[n, m, l] > max_val:
                    max_val = X[n, m, l]
                    max_idx = l
            max_vals[n, m] = max_val
            max_idxs[n, m] = max_idx
    return max_vals, max_idxs


@njit
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