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
    xp = _get_array_module()
    motifsA_xp = xp.asarray(motifsA, dtype=xp.float64)
    motifsB_xp = xp.asarray(motifsB, dtype=xp.float64)
    # Normalize motifs
    motifsA_normalized = motifsA_xp / xp.linalg.norm(
        motifsA_xp, axis=(1, 2), keepdims=True
    )
    motifsB_normalized = motifsB_xp / xp.linalg.norm(
        motifsB_xp, axis=(1, 2), keepdims=True
    )
    del motifsA_xp, motifsB_xp  # Free up memory
    # Forward similarity
    sim_1, sim_1_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized, xp
    )  # skew-symmetric alignment
    # Reverse complement
    motifsB_normalized_revcomp = _reverse_complement(motifsB_normalized)
    del motifsB_normalized  # Free up memory
    # Backward similarity
    sim_2, sim_2_alignment = _compute_similarity(
        motifsA_normalized, motifsB_normalized_revcomp, xp
    )  # symmetric alignment
    # Pick best similarity
    sim_12 = xp.stack([sim_1, sim_2])
    sim = xp.max(sim_12, axis=0)
    alignment_rc = xp.argmax(sim_12, axis=0)
    alignment_h = xp.where(alignment_rc == 0, sim_1_alignment, sim_2_alignment)
    # Guarantee similarity properties
    alignment_h[sim == 0] = (
        0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    )
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


def _tensor3_matmul_tensor2(x, y, xp):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    N, L, K = x.shape
    M = y.shape[1]
    x_flat = xp.reshape(x, (N * L, K))  # (NL, K)
    result = x_flat @ y  # (NL, M)
    return xp.reshape(result, (N, L, M))  # (N, L, M)


def _compute_similarity(motif_set_1, motif_set_2, xp):
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
        motif_set_1, xp
    )  # (3L-2, N) K times
    # Compute similarity per base
    sims = []
    for i in range(K):
        left_side_matrix_i = _compute_similarity_left_side_i(
            motif_set_2[:, :, i], xp
        )  # (M, 2L-1, 3L-2)
        sims.append(
            _tensor3_matmul_tensor2(left_side_matrix_i, right_side_matrices[i], xp)
        )  # (M, 2L-1, N)
        del left_side_matrix_i  # Free up memory
    # Sum across ATCG
    total_sum = xp.zeros_like(sims[0], dtype=xp.float64)  # (M, 2L-1, N)
    for sim in sims:
        total_sum += sim
    del sims  # Free up memory
    # Compute alignment
    best_similarity = xp.max(total_sum, axis=1)
    best_alignments = xp.argmax(total_sum, axis=1) - (L - 1)
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


def _compute_similarity_left_side_i(motifs, xp):
    """Prepares the left side of the similarity calculation."""
    M, L = motifs.shape
    left_side_matrix = _tensor3_matmul_tensor2(
        _LEFTTENSOR(L, xp), motifs.T, xp
    )  # (2L-1, 3L-2, M)
    left_side_matrix = xp.transpose(left_side_matrix, axes=(2, 0, 1))  # (M, 2L-1, 3L-2)
    assert left_side_matrix.shape == (M, 2 * L - 1, 3 * L - 2)
    return left_side_matrix  # (M, 2L-1, 3L-2)


def _compute_similarity_right_side(motifs, xp):
    """Prepares the right side of the similarity calculation."""
    N, L, K = motifs.shape  # (N, L, K)
    motifs_pivot = xp.transpose(motifs, axes=(0, 2, 1))  # (N, K, L)
    right_side_prepivot = _tensor3_matmul_tensor2(
        motifs_pivot, _RIGHTTENSOR(L, xp), xp
    )  # (N, K, 3L-2)
    del motifs_pivot  # Free up memory
    right_side_matrix = xp.transpose(
        right_side_prepivot, axes=(2, 0, 1)
    )  # (3L-2, N, K)
    del right_side_prepivot  # Free up memory
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, N)
    assert all(x.shape == (3 * L - 2, N) for x in right_side_matrices)
    return right_side_matrices  # (3L-2, N) K times


def _LEFTTENSOR(L, xp):
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    global _LEFT_TENSOR
    create_tensor = False
    if _LEFT_TENSOR is None:
        create_tensor = True
    if (not isinstance(_LEFT_TENSOR, xp.ndarray)) or (_LEFT_TENSOR.shape[2] != L):
        del _LEFT_TENSOR  # Free up memory
        create_tensor = True
    if create_tensor:
        _LEFT_TENSOR = xp.zeros(
            (2 * L - 1, 3 * L - 2, L), dtype=xp.float64
        )  # default (59, 88, 30)
        for i in range(2 * L - 1):  # default 59
            _LEFT_TENSOR[i, i : i + L, :] = xp.eye(L, dtype=xp.float64)  # default 30
    return _LEFT_TENSOR  # (2L-1, 3L-2, L)


def _RIGHTTENSOR(L, xp):
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    global _RIGHT_TENSOR
    create_tensor = False
    if _RIGHT_TENSOR is None:
        create_tensor = True
    if (not isinstance(_RIGHT_TENSOR, xp.ndarray)) or (_RIGHT_TENSOR.shape[0] != L):
        del _RIGHT_TENSOR  # Free up memory
        create_tensor = True
    if create_tensor:
        _RIGHT_TENSOR = xp.zeros((L, 3 * L - 2), dtype=xp.float64)  # default (30, 88)
        _RIGHT_TENSOR[:, L - 1 : 2 * L - 1] = xp.eye(
            L, dtype=xp.float64
        )  # default 29:59
    return _RIGHT_TENSOR  # (L, 3L-2)
