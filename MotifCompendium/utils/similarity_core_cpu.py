import numpy as np


####################
# PUBLIC FUNCTIONS #
####################
def cpu_compute_similarity_and_align(
    simsA: np.ndarray, simsB: np.ndarray, sim_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Normalize motifs
    match sim_type:
        case "sqrt":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)
            simsA = np.sqrt(simsA) # Sqrt normalize
            simsB = np.sqrt(simsB)
        case "l2":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True) 
            simsA = simsA / np.linalg.norm(simsA, axis=(1, 2), keepdims=True) # L2 normalize
            simsB = simsB / np.linalg.norm(simsB, axis=(1, 2), keepdims=True)
        case "jss":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)
        case "dot" | "log":
            pass
    
    # Caculate similarity and alignment
    match sim_type:
        # Cosine similarity
        case "l2" | "sqrt" | "dot":
            # forward similarity
            sim_1, sim_1_alignments = _compute_similarity_and_alignment(
                simsA, simsB
            )  # skew-symmetric alignment
            # backward similarity
            sim_2, sim_2_alignments = _compute_similarity_and_alignment(
                _reverse_complement(simsA), simsB
            )  # symmetric alignment
        
        # Jensen-Shannon Similarity
        case "jss" | "log":
            sim_1, sim_1_alignments = _compute_similarity_and_alignment_jss(
                simsA, simsB
            )
            sim_2, sim_2_alignments = _compute_similarity_and_alignment_jss(
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
    

def cpu_compute_similarity_known_alignment(
    simsA: np.ndarray, simsB: np.ndarray, 
    alignment_fr: np.ndarray, alignment_h: np.ndarray,
    sim_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # Normalize motifs
    match sim_type:
        case "sqrt":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)
            simsA = np.sqrt(simsA) # Sqrt normalize
            simsB = np.sqrt(simsB)
        case "l2":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True) 
            simsA = simsA / np.linalg.norm(simsA, axis=(1, 2), keepdims=True) # L2 normalize
            simsB = simsB / np.linalg.norm(simsB, axis=(1, 2), keepdims=True)
        case "jss":
            simsA = simsA / np.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / np.sum(simsB, axis=(1, 2), keepdims=True)
        case "dot" | "log":
            pass

    # Caculate similarity
    match sim_type:
        # Cosine similarity
        case "l2" | "sqrt" | "dot":
            # forward similarity
            sim_1, _ = _compute_similarity_and_alignment(
                simsA, simsB
            )  # skew-symmetric alignment
            # backward similarity
            sim_2, _ = _compute_similarity_and_alignment(
                _reverse_complement(simsA), simsB
            )  # symmetric alignment

            # aligning
            sim_12 = np.stack([sim_1, sim_2])
            sim = np.max(sim_12, axis=0)
            return sim.get()
        
        # Jensen-Shannon Similarity
        case "jss" | "log":
            similarity = _compute_similarity_known_alignment_jss(
                simsA, simsB,
                alignment_fr, alignment_h
            )

    return similarity.get()


#####################
# PRIVATE CONSTANTS #
#####################
def _LEFTTENSOR(L: int) -> np.ndarray:
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    x = np.zeros((2*L-1, 3*L-2, L))
    for i in range(2*L-1):
        x[i, i : i + L, :] = np.eye(L)
    return x

def _RIGHTTENSOR(L: int) -> np.ndarray:
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    x = np.zeros((L, 3*L-2))
    x[:, L-1:2*L-1] = np.eye(L)
    return x

#####################
# PRIVATE FUNCTIONS #
#####################
def _reverse_complement(motifs: np.ndarray) -> np.ndarray:
    """Computes the reverse complement of a (N, L, C) motif stack."""
    return motifs[:, ::-1, ::-1]


def _compute_similarity_and_alignment(
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


def _compute_similarity_and_alignment_jss(
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
    N = motifsP.shape[0] # (N, L, K)
    M = motifsQ.shape[0] # (M, L, K)
    L = motifsP.shape[1]
    K = motifsP.shape[2]

    ## COMPUTE JSD I: SELF-ENTROPY
    motifsP_flat = motifsP.reshape(N, -1) # (N, L*K)
    motifsQ_flat = motifsQ.reshape(M, -1) # (M, L*K)
    
    js_similarity = (np.sum(motifsP_flat * np.log2(motifsP_flat, where=(motifsP_flat != 0)), axis=1)[:, np.newaxis] \
        + np.sum(motifsQ_flat * np.log2(motifsQ_flat, where=(motifsQ_flat != 0)), axis=1)[np.newaxis, :])[:, np.newaxis, :] # (N, 1, M)
    
    del motifsP_flat, motifsQ_flat

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    left_side_tensorP = np.tensordot(motifsP, _LEFTTENSOR(L), axes=[[1],[2]])[:, :, :, :, np.newaxis] # (N, K, 2L-1, 3L-2, 1)
    right_side_tensorQ = np.transpose(np.tensordot(_RIGHTTENSOR(L), motifsQ, axes=[[0],[1]]), axes=(2,0,1))[np.newaxis,:, np.newaxis, :, :] # (1, K, 1, 3L-2, M)

    mixed_tensorM_log = 0.5*(left_side_tensorP + right_side_tensorQ) # (N, K, 2L-1, 3L-2, M)
    mixed_tensorM_log = np.log2(mixed_tensorM_log, where=(mixed_tensorM_log != 0)) # (N, K, 2L-1, 3L-2, M)

    js_similarity = js_similarity - np.sum((left_side_tensorP * mixed_tensorM_log + right_side_tensorQ * mixed_tensorM_log), axis=(1, 3)) # (N, 2L-1, M)

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


def _compute_similarity_known_alignment_jss(
    motifsP: np.ndarray, motifsQ: np.ndarray,
    alignment_fr: np.ndarray, alignment_h: np.ndarray
) -> np.ndarray:
    """Computes Jensen-Shannon Similarity (JSS) and alignment for two sets of motifs,
    where JSS = 1 - sqrt(Jensen-Shannon Divergence)."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motifsP.shape[0]
    M_original = motifsQ.shape[0]
    transpose = N_original > M_original
    if transpose:
        motifsP, motifsQ = motifsQ, motifsP
    
    N = motifsP.shape[0] # (N, L, K)
    M = motifsQ.shape[0] # (M, L, K)
    L = motifsP.shape[1] # (L)
    L_flank = L - 1
    H = motifsP.shape[1] * 3 - 2 # (3L-2)

    ## COMPUTE JSD I: SELF-ENTROPY
    motifsP_flat = motifsP.reshape(N, -1) # (N, L*K)
    motifsQ_flat = motifsQ.reshape(M, -1) # (M, L*K)
    
    js_similarity = (np.sum(motifsP_flat * np.log2(np.where(motifsP_flat != 0, motifsP_flat, 1)), axis=1)[:, np.newaxis] \
        + np.sum(motifsQ_flat * np.log2(np.where(motifsQ_flat != 0, motifsQ_flat, 1)), axis=1)[np.newaxis, :]) # (N, M)
    
    # Free memory
    del motifsP_flat, motifsQ_flat
    gc.collect()

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    motifsQ = np.pad(motifsQ, ((0, 0), (L_flank, L_flank), (0, 0))) # (M, 3L-2, K)

    for align_fr in [1, 0]:
        motifsP = _reverse_complement(motifsP) # Reverse complement, twice
        h_range = np.unique(alignment_h[alignment_fr == align_fr])

        for align_h in h_range: # Alignment_h
            idx = np.where((alignment_fr == align_fr) & (alignment_h == align_h)) # (n)

            motifsP_select = np.pad(motifsP[idx[0], :, :], ((0, 0), (L_flank + align_h, L_flank - align_h), (0, 0))) # (n, 3L-2, K)
            motifsQ_select = motifsQ[idx[1], :, :] # (n, 3L-2, K)
            mixed_tensorM_log = 0.5*(motifsP_select + motifsQ_select) # (n, 3L-2, K)
            mixed_tensorM_log = np.log2(np.where(mixed_tensorM_log != 0, mixed_tensorM_log, 1)) # (n, 3L-2, K)

            js_similarity[idx] = js_similarity[idx] - np.sum((motifsP_select * mixed_tensorM_log + motifsQ_select * mixed_tensorM_log), axis=(1, 2)) # (n, m)

            # Free memory
            del motifsQ_select, motifsP_select, mixed_tensorM_log
            gc.collect()

    motifsQ = motifsQ[:, L_flank:H - L_flank, :] # (M, L, K)

    ## JS SIMILARITY
    js_similarity = 1 - np.sqrt(np.clip((0.5*js_similarity), 0, 1))

    if transpose:
        js_similarity = js_similarity.T
    assert js_similarity.shape == (N_original, M_original)
    return js_similarity


def _compute_similarity_left_side(motifs: np.ndarray) -> list[np.ndarray]:
    """Prepares the left side of the similarity calculation."""
    # motifs = (N, L, K)
    L = motifs.shape[1]
    K = motifs.shape[2]
    motif_slices = [motifs[:, :, i] for i in range(K)]  # (N, L)
    motif_slice_pivots = [x.T for x in motif_slices]  # (L, N)
    left_side_prepivots = [_LEFTTENSOR(L) @ x for x in motif_slice_pivots]  # (2L-1, 3L-2, N)
    left_side_matrices = [
        np.transpose(x, axes=(2, 0, 1)) for x in left_side_prepivots
    ]  # (N, 2L-1, 3L-2)
    for x in left_side_matrices:
        assert motifs.shape[0] == x.shape[0]
        assert (x.shape[1] == (2*L-1)) and (x.shape[2] == (3*L-2))
    return left_side_matrices


def _compute_similarity_right_side(motifs: np.ndarray) -> list[np.ndarray]:
    """Prepares the right side of the similarity calculation."""
    # motifs = (M, L, K)
    L = motifs.shape[1]
    K = motifs.shape[2]
    motifs_pivot = np.transpose(motifs, axes=(0, 2, 1))  # (M, K, L)
    right_side_prepivot = motifs_pivot @ _RIGHTTENSOR(L)  # (M, K, 3L-2)
    right_side_matrix = np.transpose(right_side_prepivot, axes=(2, 0, 1))  # (3L-2, M, K)
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, M)
    for x in right_side_matrices:
        assert motifs.shape[0] == x.shape[1]
        assert x.shape[0] == (3*L-2)
    return right_side_matrices