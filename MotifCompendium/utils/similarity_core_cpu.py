# DEVELOPED BY SALIL DESHPANDE

from numba import njit
import numpy as np


####################
# PUBLIC FUNCTIONS #
####################
@njit
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Get array module + prepare motifs
    motifsA = np.asarray(motifsA, dtype=np.float64)
    motifsB = np.asarray(motifsB, dtype=np.float64)
    # Normalize motifs
    # motifsA_norm = np.sqrt(np.sum(motifsA**2, axis=(1, 2), keepdims=True))
    # motifsA_normalized = motifsA / motifsA_norm
    motifsA_normalized = _normalize_mtx(motifsA)
    # motifsB_norm = np.sqrt(np.sum(motifsB**2, axis=(1, 2), keepdims=True))
    # motifsB_normalized = motifsB / motifsB_norm
    motifsB_normalized = _normalize_mtx(motifsB)
    # del motifsA, motifsB  # Free up memory
    # Forward similarity
    sim_1, sim_1_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized
    )  # skew-symmetric alignment
    # Reverse complement
    motifsB_normalized_revcomp = _reverse_complement(motifsB_normalized)
    # del motifsB_normalized  # Free up memory
    # Backward similarity
    sim_2, sim_2_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized_revcomp
    )  # symmetric alignment
    # Pick best similarity
    # sim_12 = np.stack([sim_1, sim_2])
    # sim = np.max(sim_12, axis=0)
    # alignment_rc = np.argmax(sim_12, axis=0)
    sim, alignment_rc = _compare_two_matrices(sim_1, sim_2)
    alignment_h = np.where(alignment_rc == 0, sim_1_alignment, sim_2_alignment)
    # Guarantee similarity properties
    # alignment_h[sim == 0] = (
    #     0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    # )
    # Return
    return (
        sim.astype(np.single),
        alignment_rc.astype(np.bool_),
        alignment_h.astype(np.short),
    )


####################
# GLOBAL VARIABLES #
####################
# _CUPY_IMPORT = None
# _RIGHT_TENSOR = None
# _LEFT_TENSOR = None


#####################
# PRIVATE FUNCTIONS #
#####################
@njit
def _normalize_mtx(x):
    n, h, w = x.shape
    x_normalized = np.empty_like(x)
    for i in range(n):
        norm = 0.0
        for j in range(h):
            for k in range(w):
                norm += x[i, j, k] ** 2
        norm = np.sqrt(norm)
        if norm != 0:
            for j in range(h):
                for k in range(w):
                    x_normalized[i, j, k] = x[i, j, k] / norm
        else:
            for j in range(h):
                for k in range(w):
                    x_normalized[i, j, k] = 0.0
    return x_normalized


@njit
def _reverse_complement(motifs):
    """Computes the reverse complement of a (N, L, K) motif stack."""
    return motifs[:, ::-1, ::-1]


@njit
def _tensor3_matmul_tensor2(x, y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    N, M, K = x.shape
    K2, L = y.shape
    assert K == K2  # Ensure dimensions match

    out = np.zeros((N, M, L))
    for n in range(N):
        for m in range(M):
            for l in range(L):
                for k in range(K):
                    out[n, m, l] += x[n, m, k] * y[k, l]
    return out

@njit
def _compare_two_matrices(A, B):
    n, m = A.shape
    max_vals = np.empty((n, m))
    max_idxs = np.empty((n, m), dtype=np.int64)

    for i in range(n):
        for j in range(m):
            if A[i, j] > B[i, j]:
                max_vals[i, j] = A[i, j]
                max_idxs[i, j] = 0
            else:
                max_vals[i, j] = B[i, j]
                max_idxs[i, j] = 1

    return max_vals, max_idxs


@njit
def _max_and_argmax_along_axis1(x):
    n, m, k = x.shape
    max_vals = np.empty((n, k))
    max_idxs = np.empty((n, k), dtype=np.int64)

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


# @njit
def _compute_similarity(motif_set_1, motif_set_2):
    """Computes similarity and alignment for two sets of motifs."""
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape
    # assert L == L2
    # assert K == K2
    # Transpose for efficiency
    transpose = N < M
    if transpose:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
        # del temp  # Free up memory
    # Compute right side matrices
    right_side_matrices = _compute_similarity_right_side(
        motif_set_1
    )  # (3L-2, N) K times
    # Compute similarity per base
    sims = []
    for i in range(K):
        left_side_matrix_i = _compute_similarity_left_side_i(
            motif_set_2[:, :, i]
        )  # (M, 2L-1, 3L-2)
        sims.append(
            _tensor3_matmul_tensor2(left_side_matrix_i, right_side_matrices[i])
        )  # (M, 2L-1, N)
        # del left_side_matrix_i  # Free up memory
    # Sum across ATCG
    total_sum = np.zeros_like(sims[0], dtype=np.float64)  # (M, 2L-1, N)
    for sim in sims:
        total_sum += sim
    # del sims  # Free up memory
    # Compute alignment
    best_similarity = np.max(total_sum, axis=1)
    best_alignments = np.argmax(total_sum, axis=1) - (L - 1)
    # best_similarity, best_alignments = _max_and_argmax_along_axis1(total_sum)
    # best_alignments -= (L - 1)
    # del total_sum  # Free up memory
    # Undo transpose if needed
    if transpose:
        best_similarity = best_similarity.T
        best_alignments = (
            -best_alignments.T
        )  # negative because transposing flips alignment
    # assert best_similarity.shape == (M, N)
    # assert best_alignments.shape == (M, N)
    return best_similarity.T, best_alignments.T  # (N, M), (N, M)


# @njit
def _compute_similarity_left_side_i(motifs):
    """Prepares the left side of the similarity calculation."""
    M, L = motifs.shape
    left_side_matrix = _tensor3_matmul_tensor2(
        _LEFTTENSOR(L), motifs.T
    )  # (2L-1, 3L-2, M)
    left_side_matrix = np.transpose(left_side_matrix, axes=(2, 0, 1))  # (M, 2L-1, 3L-2)
    # assert left_side_matrix.shape == (M, 2 * L - 1, 3 * L - 2)
    return left_side_matrix  # (M, 2L-1, 3L-2)


# @njit
def _compute_similarity_right_side(motifs):
    """Prepares the right side of the similarity calculation."""
    N, L, K = motifs.shape  # (N, L, K)
    motifs_pivot = np.transpose(motifs, axes=(0, 2, 1))  # (N, K, L)
    right_side_prepivot = _tensor3_matmul_tensor2(
        motifs_pivot, _RIGHTTENSOR(L)
    )  # (N, K, 3L-2)
    # del motifs_pivot  # Free up memory
    right_side_matrix = np.transpose(
        right_side_prepivot, axes=(2, 0, 1)
    )  # (3L-2, N, K)
    # del right_side_prepivot  # Free up memory
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, N)
    # assert all(x.shape == (3 * L - 2, N) for x in right_side_matrices)
    return right_side_matrices  # (3L-2, N) K times


@njit
def _LEFTTENSOR(L):
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    _LEFT_TENSOR = np.zeros(
        (2 * L - 1, 3 * L - 2, L), dtype=np.float64
    )  # default (59, 88, 30)
    for i in range(2 * L - 1):  # default 59
        _LEFT_TENSOR[i, i : i + L, :] = np.eye(L, dtype=np.float64)  # default 30
    return _LEFT_TENSOR  # (2L-1, 3L-2, L)


@njit
def _RIGHTTENSOR(L):
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    _RIGHT_TENSOR = np.zeros((L, 3 * L - 2), dtype=np.float64)  # default (30, 88)
    _RIGHT_TENSOR[:, L - 1 : 2 * L - 1] = np.eye(
        L, dtype=np.float64
    )  # default 29:59
    return _RIGHT_TENSOR  # (L, 3L-2)
