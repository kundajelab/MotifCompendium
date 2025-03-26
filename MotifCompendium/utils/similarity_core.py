import numpy as np

from MotifCompendium.utils.config import get_use_gpu


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    motifsA: np.ndarray, motifsB: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Get array module + prepare motifs
    xp = _get_array_module()
    motifsA_xp = xp.asarray(motifsA)
    motifsB_xp = xp.asarray(motifsB)
    # Normalize motifs
    motifsA_normalized = motifsA_xp / xp.linalg.norm(
        motifsA_xp, axis=(1, 2), keepdims=True
    )
    motifsB_normalized = motifsB_xp / xp.linalg.norm(
        motifsB_xp, axis=(1, 2), keepdims=True
    )
    del motifsA_xp, motifsB_xp  # Free up memory
    # Forward similarity
    sim_1, sim_1_aligns = _compute_similarity(
        motifsA_normalized, motifsB_normalized, xp
    )  # skew-symmetric alignment
    # Reverse complement
    motifsA_normalized_revcomp = _reverse_complement(motifsA_normalized)
    del motifsA_normalized  # Free up memory
    # Backward similarity
    sim_2, sim_2_aligns = _compute_similarity(
        motifsA_normalized_revcomp, motifsB_normalized, xp
    )  # symmetric alignment
    # Pick best similarity
    sim_12 = xp.stack([sim_1, sim_2])
    sim = xp.max(sim_12, axis=0)
    align_rc = xp.argmax(sim_12, axis=0)
    align_h = xp.where(align_rc == 0, sim_1_aligns, sim_2_aligns)
    # Guarantee similarity properties
    align_h[sim == 0] = (
        0  # When 0 similarity, set alignment to 0 for alignment symmetry properties
    )
    # Return
    if get_use_gpu():
        sim, align_rc, align_h = sim.get(), align_rc.get(), align_h.get()
    return sim.astype(np.single), align_rc.astype(np.bool_), align_h.astype(np.byte)


####################
# GLOBAL VARIABLES #
####################
_CUPY_IMPORT = None
_RIGHT_TENSOR = None
_LEFT_TENSOR = None


def _get_array_module():
    if get_use_gpu():
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


def _compute_similarity(motif_set_1, motif_set_2, xp):
    """Computes similarity and alignment for two sets of motifs."""
    # Get shapes
    N, L, K = motif_set_1.shape
    M, L2, K2 = motif_set_2.shape
    assert L == L2
    assert K == K2
    # Transpose for efficiency
    transpose = N > M
    if transpose:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
        del temp  # Free up memory
    # Compute right side matrices
    right_side_matrices = _compute_similarity_right_side(motif_set_2, xp)
    # Compute similarity per base
    sims = []
    for i in range(K):
        left_side_matrix_i = _compute_similarity_left_side_i(motif_set_1[:, :, i], xp)
        sims.append(left_side_matrix_i @ right_side_matrices[i])
        del left_side_matrix_i  # Free up memory
    # Sum across ATCG
    total_sum = xp.sum(xp.stack(sims), axis=0)
    del sims  # Free up memory
    # Compute alignment
    best_align_scores = xp.max(total_sum, axis=1)
    best_align_h = xp.argmax(total_sum, axis=1) - (L - 1)
    del total_sum  # Free up memory
    # Undo transpose if needed
    if transpose:
        best_align_scores = best_align_scores.T
        best_align_h = -best_align_h.T  # negative because transposing flips alignment
    assert best_align_scores.shape == (N, M)
    assert best_align_h.shape == (N, M)
    return best_align_scores, best_align_h


def _compute_similarity_right_side(motifs, xp):
    """Prepares the right side of the similarity calculation."""
    M, L, K = motifs.shape  # (M, L, K)
    motifs_pivot = xp.transpose(motifs, axes=(0, 2, 1))  # (M, K, L)
    right_side_prepivot = motifs_pivot @ _RIGHTTENSOR(L, xp)  # (M, K, 3L-2)
    del motifs_pivot  # Free up memory
    right_side_matrix = xp.transpose(
        right_side_prepivot, axes=(2, 0, 1)
    )  # (3L-2, M, K)
    del right_side_prepivot  # Free up memory
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, M)
    del right_side_matrix  # Free up memory
    for x in right_side_matrices:
        assert x.shape[1] == M
        assert x.shape[0] == 3 * L - 2
    return right_side_matrices


def _compute_similarity_left_side_i(motifs, xp):
    """Prepares the left side of the similarity calculation."""
    N, L = motifs.shape
    left_side_matrix = _LEFTTENSOR(L, xp) @ motifs.T  # (2L-1, 3L-2, N)
    left_side_matrix = xp.transpose(left_side_matrix, axes=(2, 0, 1))  # (N, 2L-1, 3L-2)
    assert left_side_matrix.shape[0] == N
    assert (left_side_matrix.shape[1] == 2 * L - 1) and (
        left_side_matrix.shape[2] == 3 * L - 2
    )
    return left_side_matrix


def _LEFTTENSOR(L, xp):
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    global _LEFT_TENSOR
    if (_LEFT_TENSOR is None) or (_LEFT_TENSOR.shape[2] != L):
        _LEFT_TENSOR = xp.zeros((2 * L - 1, 3 * L - 2, L))  # default (59, 88, 30)
        for i in range(2 * L - 1):  # default 59
            _LEFT_TENSOR[i, i : i + L, :] = xp.eye(L)  # default 30
    return _LEFT_TENSOR


def _RIGHTTENSOR(L, xp):
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    global _RIGHT_TENSOR
    if (_RIGHT_TENSOR is None) or (_RIGHT_TENSOR.shape[1] != 3 * L - 2):
        _RIGHT_TENSOR = xp.zeros((L, 3 * L - 2))  # default (30, 88)
        _RIGHT_TENSOR[:, L - 1 : 2 * L - 1] = xp.eye(L)  # default 29:59
    return _RIGHT_TENSOR
