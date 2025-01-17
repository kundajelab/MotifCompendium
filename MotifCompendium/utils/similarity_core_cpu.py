import numpy as np


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarity_and_align(
    simsA: np.ndarray, simsB: np.ndarray, sim_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # L1 normalize
    simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True)
    simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)

    match sim_type:
        case "l2" | "sqrt":
            if sim_type == "l2":
                # L2 normalization
                simsA = simsA / np.linalg.norm(simsA, axis=(1, 2), keepdims=True)
                simsB = simsB / np.linalg.norm(simsB, axis=(1, 2), keepdims=True)
            elif sim_type == "sqrt":
                # square root
                simsA = np.sqrt(simsA)
                simsB = np.sqrt(simsB)

            # forward similarity
            sim_1, sim_1_alignments = _compute_similarity_alignment(
                simsA, simsB
            )  # skew-symmetric alignment
            # backward similarity
            sim_2, sim_2_alignments = _compute_similarity_alignment(
                _reverse_complement(simsA), simsB
            )  # symmetric alignment
        
        case "jss":
            sim_1, sim_1_alignments = _compute_similarity_alignment_jss(
                simsA, simsB
            )
            sim_2, sim_2_alignments = _compute_similarity_alignment_jss(
                _reverse_complement(simsA), simsB
            )
    
    # aligning
    sim_12 = np.stack([sim_1, sim_2])
    sim = np.max(sim_12, axis=0)
    sim_fb = np.argmax(sim_12, axis=0)
    sim_alignments = np.where(sim_fb == 0, sim_1_alignments, sim_2_alignments)
    sim_alignments[sim == 0] = (
        0  # alignment erroneously always -29 when 0 similarity (argmax artifact)
    )
    return sim, sim_fb, sim_alignments
    

def compute_similarity(
    simsA: np.ndarray, simsB: np.ndarray, 
    alignment_fr: np.ndarray, alignment_h: np.ndarray,
    sim_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # L1 normalize
    simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True)
    simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)

    match sim_type:
        case "l2" | "sqrt":
            if sim_type == "l2":
                # L2 normalization
                simsA = simsA / np.linalg.norm(simsA, axis=(1, 2), keepdims=True)
                simsB = simsB / np.linalg.norm(simsB, axis=(1, 2), keepdims=True)
            elif sim_type == "sqrt":
                # square root
                simsA = np.sqrt(simsA)
                simsB = np.sqrt(simsB)

            # forward similarity
            similarity = _compute_similarity(
                simsA, simsB,
                alignment_fr, alignment_h
            ) 
        
        case "jss":
            similarity = _compute_similarity_jss(
                simsA, simsB,
                alignment_fr, alignment_h
            )

    return similarity.get()


#####################
# PRIVATE CONSTANTS #
#####################
_LEFTTENSOR = np.zeros((59, 88, 30))
for i in range(59):
    _LEFTTENSOR[i, i : i + 30, :] = np.eye(30)

_RIGHTTENSOR = np.zeros((30, 88))
_RIGHTTENSOR[:, 29:59] = np.eye(30)


#####################
# PRIVATE FUNCTIONS #
#####################
def _reverse_complement(motifs: np.ndarray) -> np.ndarray:
    """Computes the reverse complement of a (N, L, C) motif stack."""
    return motifs[:, ::-1, ::-1]


def _compute_similarity_alignment(
    motif_set_1: np.ndarray, motif_set_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes similarity and alignment for two sets of motifs."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motif_set_1.shape[0]
    M_original = motif_set_2.shape[0]
    if N_original > M_original:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
    transpose = N_original > M_original
    N = motif_set_1.shape[0]
    M = motif_set_2.shape[0]
    # COMPUTE LEFT SIDE TENSOR
    left_side_matrices = _compute_similarity_left_side(motif_set_1)
    # COMPUTE RIGHT SIDE TENSOR
    right_side_matrices = _compute_similarity_right_side(motif_set_2)
    # SUM ACROSS ATCG
    convs = [
        left_side_matrices[i] @ right_side_matrices[i]
        for i in range(len(left_side_matrices))
    ]
    total_sum = np.sum(convs, axis=0)
    # COMPUTE ALIGNMENT
    best_alignment_scores = np.max(total_sum, axis=1)
    best_alignments = np.argmax(total_sum, axis=1) - 29
    # REALIGN IF NEEDED
    if transpose:
        best_alignment_scores = best_alignment_scores.T
        best_alignments = (
            -best_alignments.T
        )  # negative because transposing flips alignment
    assert best_alignment_scores.shape == (N_original, M_original)
    assert best_alignments.shape == (N_original, M_original)
    return best_alignment_scores, best_alignments


def _compute_similarity_alignment_jss(
    motifsP: np.ndarray, motifsQ: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes Jensen-Shannon Similarity (JSS) and alignment for two sets of motifs,
    where JSS = 1 - sqrt(Jensen-Shannon Divergence)."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motifsP.shape[0]
    M_original = motifsQ.shape[0]
    if N_original > M_original:
        temp = motifsP
        motifsP = motifsQ
        motifsQ = temp
    transpose = N_original > M_original
    N = motifsP.shape[0] # (N, 30, K)
    M = motifsQ.shape[0] # (M, 30, K)
    K = motifsP.shape[2]

    ## COMPUTE JSD I: SELF-ENTROPY
    motifsP_flat = motifsP.reshape(N, -1) # (N, 30*K)
    motifsQ_flat = motifsQ.reshape(M, -1) # (M, 30*K)
    
    js_similarity = (np.sum(motifsP_flat * np.log2(motifsP_flat, where=(motifsP_flat != 0)), axis=1)[:, np.newaxis] \
        + np.sum(motifsQ_flat * np.log2(motifsQ_flat, where=(motifsQ_flat != 0)), axis=1)[np.newaxis, :])[:, np.newaxis, :] # (N, 1, M)
    
    del motifsP_flat, motifsQ_flat

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    left_side_tensorP = np.tensordot(motifsP, _LEFTTENSOR, axes=[[1],[2]])[:, :, :, :, np.newaxis] # (N, K, 59, 88, 1)
    right_side_tensorQ = np.transpose(np.tensordot(_RIGHTTENSOR, motifsQ, axes=[[0],[1]]), axes=(2,0,1))[np.newaxis,:, np.newaxis, :, :] # (1, K, 1, 88, M)

    mixed_tensorM_log = 0.5*(left_side_tensorP + right_side_tensorQ) # (N, K, 59, 88, M)
    mixed_tensorM_log = np.log2(mixed_tensorM_log, where=(mixed_tensorM_log != 0)) # (N, K, 59, 88, M)

    js_similarity = js_similarity - np.sum((left_side_tensorP * mixed_tensorM_log + right_side_tensorQ * mixed_tensorM_log), axis=(1, 3)) # (N, 59, M)

    del left_side_tensorP, right_side_tensorQ, mixed_tensorM_log

    ## JS SIMILARITY
    js_similarity = 0.5 * js_similarity
    js_similarity = np.clip(js_similarity, 0, 1)
    js_similarity = 1 - np.sqrt(js_similarity)

    # COMPUTE ALIGNMENT
    best_alignment_scores = np.max(js_similarity, axis=1)
    best_alignments = np.argmax(js_similarity, axis=1) - 29
    
    # REALIGN IF NEEDED
    if transpose:
        best_alignment_scores = best_alignment_scores.T
        best_alignments = (
            -best_alignments.T
        )  # negative because transposing flips alignment
    assert best_alignment_scores.shape == (N_original, M_original)
    assert best_alignments.shape == (N_original, M_original)
    return best_alignment_scores, best_alignments


def _compute_similarity(
    motif_set_1: np.ndarray, motif_set_2: np.ndarray,
    alignment_fr: np.ndarray, alignment_h: np.ndarray
) -> np.ndarray:
    """Computes similarity and alignment for two sets of motifs."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motif_set_1.shape[0]
    M_original = motif_set_2.shape[0]
    if N_original > M_original:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
    transpose = N_original > M_original
    N = motif_set_1.shape[0]
    M = motif_set_2.shape[0]
    # COMPUTE LEFT SIDE TENSOR ## TO BE UPDATED
    left_side_matrices = _compute_similarity_left_side(motif_set_1)
    # COMPUTE RIGHT SIDE TENSOR ## TO BE UPDATED
    right_side_matrices = _compute_similarity_right_side(motif_set_2)
    # SUM ACROSS ATCG
    convs = np.stack(
        [
            left_side_matrices[i] @ right_side_matrices[i]
            for i in range(len(left_side_matrices))
        ]
    )  # need to stack
    similarity_scores = np.sum(convs, axis=0)
    
    assert similarity_scores.shape == (N_original, M_original)
    return similarity_scores


def _compute_similarity_jss(
    motif_set_1: np.ndarray, motif_set_2: np.ndarray,
    alignment_fr: np.ndarray, alignment_h: np.ndarray
) -> np.ndarray:
    """Computes Jensen-Shannon Similarity (JSS) and alignment for two sets of motifs,
    where JSS = 1 - sqrt(Jensen-Shannon Divergence)."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motifsP.shape[0]
    M_original = motifsQ.shape[0]
    if N_original > M_original:
        temp = motifsP
        motifsP = motifsQ
        motifsQ = temp
    transpose = N_original > M_original
    N = motifsP.shape[0] # (N, 30, K)
    M = motifsQ.shape[0] # (M, 30, K)
    K = motifsP.shape[2]

    ## COMPUTE JSD I: SELF-ENTROPY
    motifsP_flat = motifsP.reshape(N, -1) # (N, 30*K)
    motifsQ_flat = motifsQ.reshape(M, -1) # (M, 30*K)
    
    js_similarity = (np.sum(motifsP_flat * np.log2(motifsP_flat, where=(motifsP_flat != 0)), axis=1)[:, np.newaxis] \
        + np.sum(motifsQ_flat * np.log2(motifsQ_flat, where=(motifsQ_flat != 0)), axis=1)[np.newaxis, :])[:, np.newaxis, :] # (N, 1, M)
    
    del motifsP_flat, motifsQ_flat

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    left_side_tensorP = np.tensordot(motifsP, _LEFTTENSOR, axes=[[1],[2]])[:, :, :, :, np.newaxis] # (N, K, 88, 1)
    right_side_tensorQ = np.transpose(np.tensordot(_RIGHTTENSOR, motifsQ, axes=[[0],[1]]), axes=(2,0,1))[np.newaxis,:, np.newaxis, :, :] # (1, K, 88, M)

    mixed_tensorM_log = 0.5*(left_side_tensorP + right_side_tensorQ) # (N, K, 88, M)
    mixed_tensorM_log = np.log2(mixed_tensorM_log, where=(mixed_tensorM_log != 0)) # (N, K, 88, M)

    js_similarity = js_similarity - np.sum((left_side_tensorP * mixed_tensorM_log + right_side_tensorQ * mixed_tensorM_log), axis=(1, 2)) # (N, M)

    del left_side_tensorP, right_side_tensorQ, mixed_tensorM_log


    ## JS SIMILARITY
    js_similarity = 0.5 * js_similarity
    js_similarity = np.clip(js_similarity, 0, 1)
    js_similarity = 1 - np.sqrt(js_similarity)

    assert js_similarity.shape == (N_original, M_original)
    return js_similarity


def _compute_similarity_left_side(motifs: np.ndarray) -> list[np.ndarray]:
    """Prepares the left side of the similarity calculation."""
    # motifs = (N, 30, K)
    K = motifs.shape[2]
    motif_slices = [motifs[:, :, i] for i in range(K)]  # (N, 30)
    motif_slice_pivots = [x.T for x in motif_slices]  # (30, N)
    left_side_prepivots = [_LEFTTENSOR @ x for x in motif_slice_pivots]  # (59, 88, N)
    left_side_matrices = [
        np.transpose(x, axes=(2, 0, 1)) for x in left_side_prepivots
    ]  # (N, 59, 88)
    for x in left_side_matrices:
        assert motifs.shape[0] == x.shape[0]
        assert (x.shape[1] == 59) and (x.shape[2] == 88)
    return left_side_matrices


def _compute_similarity_right_side(motifs: np.ndarray) -> list[np.ndarray]:
    """Prepares the right side of the similarity calculation."""
    # motifs = (M, 30, K)
    K = motifs.shape[2]
    motifs_pivot = np.transpose(motifs, axes=(0, 2, 1))  # (M, K, 30)
    right_side_prepivot = motifs_pivot @ _RIGHTTENSOR  # (M, K, 88)
    right_side_matrix = np.transpose(right_side_prepivot, axes=(2, 0, 1))  # (88, M, K)
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (88, M)
    for x in right_side_matrices:
        assert motifs.shape[0] == x.shape[1]
        assert x.shape[0] == 88
    return right_side_matrices