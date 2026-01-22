# DEVELOPED BY SALIL DESHPANDE

import cupy as cp
import numpy as np


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Move to GPU
    motifsA_cp = cp.asarray(motifsA, dtype=cp.float64)
    motifsB_cp = cp.asarray(motifsB, dtype=cp.float64)
    # Normalize motifs
    motifsA_normalized = motifsA_cp / cp.linalg.norm(
        motifsA_cp, axis=(1, 2), keepdims=True
    )
    motifsB_normalized = motifsB_cp / cp.linalg.norm(
        motifsB_cp, axis=(1, 2), keepdims=True
    )
    del motifsA_cp, motifsB_cp  # Free up memory
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
    del motifsA_normalized, motifsB_normalized_revcomp  # Free up memory
    # Pick best similarity
    sim_12 = cp.stack([sim_1, sim_2])
    sim = cp.max(sim_12, axis=0)
    alignment_rc = cp.argmax(sim_12, axis=0)
    alignment_h = cp.where(alignment_rc == 0, sim_1_alignment, sim_2_alignment)
    # Guarantee similarity properties
    alignment_h[sim == 0] = (
        0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    )
    # Return
    return (
        sim.get().astype(np.single),
        alignment_rc.get().astype(np.bool_),
        alignment_h.get().astype(np.short),
    )


####################
# GLOBAL VARIABLES #
####################
_RIGHT_TENSOR_GPU = None
_LEFT_TENSOR_GPU = None


#####################
# PRIVATE FUNCTIONS #
#####################
def _reverse_complement(motifs):
    """Computes the reverse complement of a (N, L, K) motif stack."""
    return motifs[:, ::-1, ::-1]


def _tensor3_matmul_tensor2(x, y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    N, L, K = x.shape
    M = y.shape[1]
    x_flat = x.reshape(N * L, K)  # (NL, K)
    result = x_flat @ y  # (NL, M)
    return result.reshape(N, L, M)  # (N, L, M)


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
        del temp  # Free up memory
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
        del left_side_matrix_i  # Free up memory
    # Sum across ATCG
    total_sum = cp.zeros_like(sims[0], dtype=cp.float64)  # (M, 2L-1, N)
    for sim in sims:
        total_sum += sim
    del sims  # Free up memory
    # Compute alignment
    best_similarity = cp.max(total_sum, axis=1)
    best_alignments = cp.argmax(total_sum, axis=1) - (L - 1)
    del total_sum  # Free up memory
    # Undo transpose if needed
    if transpose:
        best_similarity = best_similarity.T
        best_alignments = (
            -best_alignments.T
        )  # negative because transposing flips alignment
    assert best_similarity.shape == (M, N)
    assert best_alignments.shape == (M, N)
    return best_similarity.T, best_alignments.T  # (N, M), (N, M)


def _compute_similarity_left_side_i(motifs):
    """Prepares the left side of the similarity calculation."""
    M, L = motifs.shape
    left_side_matrix = _tensor3_matmul_tensor2(
        _LEFTTENSOR(L), motifs.T
    )  # (2L-1, 3L-2, M)
    left_side_matrix = cp.transpose(left_side_matrix, axes=(2, 0, 1))  # (M, 2L-1, 3L-2)
    assert left_side_matrix.shape == (M, 2 * L - 1, 3 * L - 2)
    return left_side_matrix  # (M, 2L-1, 3L-2)


def _compute_similarity_right_side(motifs):
    """Prepares the right side of the similarity calculation."""
    N, L, K = motifs.shape  # (N, L, K)
    motifs_pivot = cp.transpose(motifs, axes=(0, 2, 1))  # (N, K, L)
    right_side_prepivot = _tensor3_matmul_tensor2(
        motifs_pivot, _RIGHTTENSOR(L)
    )  # (N, K, 3L-2)
    del motifs_pivot  # Free up memory
    right_side_matrix = cp.transpose(
        right_side_prepivot, axes=(2, 0, 1)
    )  # (3L-2, N, K)
    del right_side_prepivot  # Free up memory
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, N)
    assert all(x.shape == (3 * L - 2, N) for x in right_side_matrices)
    return right_side_matrices  # (3L-2, N) K times


def _LEFTTENSOR(L):
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    global _LEFT_TENSOR_GPU
    create_tensor = False
    if _LEFT_TENSOR_GPU is None:
        create_tensor = True
    elif _LEFT_TENSOR_GPU.shape[2] != L:
        del _LEFT_TENSOR_GPU  # Free up memory
        create_tensor = True
    if create_tensor:
        _LEFT_TENSOR_GPU = cp.zeros(
            (2 * L - 1, 3 * L - 2, L), dtype=cp.float64
        )  # default (59, 88, 30)
        for i in range(2 * L - 1):  # default 59
            _LEFT_TENSOR_GPU[i, i : i + L, :] = cp.eye(
                L, dtype=cp.float64
            )  # default 30
    return _LEFT_TENSOR_GPU  # (2L-1, 3L-2, L)


def _RIGHTTENSOR(L):
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    global _RIGHT_TENSOR_GPU
    create_tensor = False
    if _RIGHT_TENSOR_GPU is None:
        create_tensor = True
    elif _RIGHT_TENSOR_GPU.shape[0] != L:
        del _RIGHT_TENSOR_GPU  # Free up memory
        create_tensor = True
    if create_tensor:
        _RIGHT_TENSOR_GPU = cp.zeros(
            (L, 3 * L - 2), dtype=cp.float64
        )  # default (30, 88)
        _RIGHT_TENSOR_GPU[:, L - 1 : 2 * L - 1] = cp.eye(
            L, dtype=cp.float64
        )  # default 29:59
    return _RIGHT_TENSOR_GPU  # (L, 3L-2)
