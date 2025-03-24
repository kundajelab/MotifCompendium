import cupy as cp
import numpy as np
import gc

####################
# PUBLIC FUNCTIONS #
####################
def gpu_compute_similarity_and_align(
    simsA: np.ndarray, simsB: np.ndarray, sim_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes similarity and alignment taking into account reverse complements."""
    # convert to cupy
    simsA = cp.asarray(simsA)
    simsB = cp.asarray(simsB)

    # Normalize motifs
    match sim_type:
        case "sqrt":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True)
            simsA = cp.sqrt(simsA) # Sqrt normalize
            simsB = cp.sqrt(simsB)
        case "l2":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True) 
            simsA = simsA / cp.linalg.norm(simsA, axis=(1, 2), keepdims=True) # L2 normalize
            simsB = simsB / cp.linalg.norm(simsB, axis=(1, 2), keepdims=True)
        case "jss":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True)
        case "dot" | "log":
            pass

    # Calculate similarity and alignment
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
    sim_12 = cp.stack([sim_1, sim_2])
    sim = cp.max(sim_12, axis=0)
    sim_fr = cp.argmax(sim_12, axis=0)
    sim_alignments = cp.where(sim_fr == 0, sim_1_alignments, sim_2_alignments)
    sim_alignments[sim == 0] = 0  # alignment erroneously always -29
    return sim.get(), sim_fr.get(), sim_alignments.get()


def gpu_compute_similarity_known_alignment(
    simsA: np.ndarray, simsB: np.ndarray, 
    alignment_fr: np.ndarray, alignment_h: np.ndarray,
    sim_type: str,
) -> np.ndarray:
    """Computes similarity and alignment taking into account reverse complements."""
    # Convert to cupy
    simsA = cp.asarray(simsA)
    simsB = cp.asarray(simsB)
    alignment_h = cp.asarray(alignment_h)
    alignment_fr = cp.asarray(alignment_fr)

    # Normalize motifs
    match sim_type:
        case "sqrt":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True)
            simsA = cp.sqrt(simsA) # Sqrt normalize
            simsB = cp.sqrt(simsB)
        case "l2":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True) 
            simsA = simsA / cp.linalg.norm(simsA, axis=(1, 2), keepdims=True) # L2 normalize
            simsB = simsB / cp.linalg.norm(simsB, axis=(1, 2), keepdims=True)
        case "jss":
            simsA = simsA / cp.sum(simsA, axis=(1, 2), keepdims=True) # L1 normalize
            simsB = simsB / cp.sum(simsB, axis=(1, 2), keepdims=True)
        case "dot" | "log":
            pass

    # Calculate similarity
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
            sim_12 = cp.stack([sim_1, sim_2])
            sim = cp.max(sim_12, axis=0)
            return sim.get()
        
        # Jensen-Shannon Similarity
        case "jss" | "log":
            similarity = _compute_similarity_known_alignment_jss(
                simsA, simsB,
                alignment_fr, alignment_h
            )

    return similarity.get()


def gpu_similarity_worker(gpu_idx, inputs):
    """Worker function to perform many similarity calculations on one GPU."""
    outputs = []
    with cp.cuda.Device(gpu_idx):
        for calculation, simsA, simsB in inputs:
            result = gpu_compute_similarity_and_align(simsA, simsB)
            outputs.append((calculation, result))
    return outputs


#####################
# PRIVATE FUNCTIONS #
#####################
def _reverse_complement(motifs: cp.ndarray) -> cp.ndarray:
    """Computes the reverse complement of a (N, L, C) motif stack."""
    return motifs[:, ::-1, ::-1]


def _compute_similarity_and_alignment(
    motif_set_1: cp.ndarray, motif_set_2: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Computes similarity and alignment for two sets of motifs."""
    # TRANSPOSE FOR EFFICIENCY
    N_original = motif_set_1.shape[0]
    M_original = motif_set_2.shape[0]
    if N_original > M_original:
        temp = motif_set_1
        motif_set_1 = motif_set_2
        motif_set_2 = temp
        del temp
    transpose = N_original > M_original
    # COMPUTE LEFT SIDE TENSOR
    left_side_matrices = _compute_similarity_left_side(motif_set_1)
    # COMPUTE RIGHT SIDE TENSOR
    right_side_matrices = _compute_similarity_right_side(motif_set_2)
    # SUM ACROSS ATCG
    convs = cp.stack(
        [
            left_side_matrices[i] @ right_side_matrices[i]
            for i in range(len(left_side_matrices))
        ]
    )  # need to stack
    convs = cp.sum(convs, axis=0)
    # COMPUTE ALIGNMENT
    best_alignment_scores = cp.max(convs, axis=1)
    best_alignments = cp.argmax(convs, axis=1) - 29
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
    motifsP: cp.ndarray, motifsQ: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Computes Jensen-Shannon Similarity (JSS) and alignment for two sets of motifs,
    where JSS = 1 - sqrt(Jensen-Shannon Divergence)."""
    N = motifsP.shape[0] # (N, L, K)
    M = motifsQ.shape[0] # (M, L, K)
    L = motifsP.shape[1]
    K = motifsP.shape[2]

    ## COMPUTE JSD I: SELF-ENTROPY
    motifsP_flat = motifsP.reshape(N, -1) # (N, L*K)
    motifsQ_flat = motifsQ.reshape(M, -1) # (M, L*K)
    
    js_similarity = (cp.sum(motifsP_flat * cp.log2(cp.where(motifsP_flat != 0, motifsP_flat, 1)), axis=1)[:, cp.newaxis] \
        + cp.sum(motifsQ_flat * cp.log2(cp.where(motifsQ_flat != 0, motifsQ_flat, 1)), axis=1)[cp.newaxis, :])[:, cp.newaxis, :] # (N, 1, M)
    
    del motifsP_flat, motifsQ_flat
    cp._default_memory_pool.free_all_blocks()

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    for k in range(K):
        left_side_tensorP = cp.tensordot(motifsP[:,:,k], _LEFTTENSOR(L), axes=[[1],[2]])[:, :, :, cp.newaxis] # (N, 2L-1, 3L-2, 1)
        right_side_tensorQ = cp.tensordot(_RIGHTTENSOR(L), motifsQ[:,:,k], axes=[[0],[1]])[cp.newaxis, cp.newaxis, :, :] # (1, 1, 3L-2, M)

        mixed_tensorM_log = 0.5*(left_side_tensorP + right_side_tensorQ) # (N, 2L-1, 3L-2, M)
        mixed_tensorM_log = cp.log2(cp.where(mixed_tensorM_log != 0, mixed_tensorM_log, 1)) # (N, 2L-1, 3L-2, M)
        
        # Update JSD
        js_similarity = js_similarity - cp.sum((left_side_tensorP * mixed_tensorM_log + right_side_tensorQ * mixed_tensorM_log), axis=2) # (N, 2L-1, M)

    ## JS SIMILARITY
    js_similarity = 1 - cp.sqrt(cp.clip((0.5*js_similarity), 0, 1))

    # COMPUTE ALIGNMENT
    best_alignment_scores = cp.max(js_similarity, axis=1)
    best_alignments = cp.argmax(js_similarity, axis=1) - 29
    
    assert best_alignment_scores.shape == (N, M)
    assert best_alignments.shape == (N, M)
    return best_alignment_scores, best_alignments


def _compute_similarity_known_alignment_jss(
    motifsP: cp.ndarray, motifsQ: cp.ndarray,
    alignment_fr: cp.ndarray, alignment_h: cp.ndarray
) -> cp.ndarray:
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
    
    js_similarity = (cp.sum(motifsP_flat * cp.log2(cp.where(motifsP_flat != 0, motifsP_flat, 1)), axis=1)[:, cp.newaxis] \
        + cp.sum(motifsQ_flat * cp.log2(cp.where(motifsQ_flat != 0, motifsQ_flat, 1)), axis=1)[cp.newaxis, :]) # (N, M)
    
    # Free memory
    del motifsP_flat, motifsQ_flat
    cp._default_memory_pool.free_all_blocks()
    gc.collect()

    ## COMPUTE JSD II: ENTROPY DIFFERENCE
    motifsQ = cp.pad(motifsQ, ((0, 0), (L_flank, L_flank), (0, 0))) # (M, 3L-2, K)

    for align_fr in [1, 0]: # Alignment_fr
        motifsP = _reverse_complement(motifsP) # Reverse complement, twice
        h_range = cp.unique(alignment_h * (alignment_fr == align_fr)).astype(int).tolist() # (h)

        for align_h in h_range: # Alignment_h
            idx = cp.where((alignment_fr == align_fr) & (alignment_h == align_h)) # (n)
            
            motifsP_select = cp.pad(motifsP[idx[0], :, :], ((0, 0), (L_flank + align_h, L_flank - align_h), (0, 0))) # (n, 3L-2, K)
            motifsQ_select = motifsQ[idx[1], :, :] # (n, 3L-2, K)
            mixed_tensorM_log = 0.5*(motifsP_select + motifsQ_select) # (n, 3L-2, K)
            mixed_tensorM_log = cp.log2(cp.where(mixed_tensorM_log != 0, mixed_tensorM_log, 1)) # (n, 3L-2, K)

            js_similarity[idx] = js_similarity[idx] - cp.sum((motifsP_select * mixed_tensorM_log + motifsQ_select * mixed_tensorM_log), axis=(1, 2)) # (n, m)

            # Free memory
            del motifsQ_select, motifsP_select, mixed_tensorM_log
            cp._default_memory_pool.free_all_blocks()
            gc.collect()

    motifsQ = motifsQ[:, L_flank:H - L_flank, :] # (M, L, K)

    ## JS SIMILARITY
    js_similarity = 1 - cp.sqrt(cp.clip((0.5*js_similarity), 0, 1))

    if transpose:
        js_similarity = js_similarity.T
    assert js_similarity.shape == (N_original, M_original)
    return js_similarity


def _compute_similarity_left_side(motifs: cp.ndarray) -> list[cp.ndarray]:
    """Prepares the left side of the similarity calculation."""
    # motifs = (N, L, K)
    L = motifs.shape[1]
    K = motifs.shape[2]
    motif_slices = [motifs[:, :, i] for i in range(K)]  # (N, L)
    motif_slice_pivots = [x.T for x in motif_slices]  # (L, N)
    left_side_prepivots = [_LEFTTENSOR(L) @ x for x in motif_slice_pivots]  # (2L-1, 3L-2, N)
    left_side_matrices = [
        cp.transpose(x, axes=(2, 0, 1)) for x in left_side_prepivots
    ]  # (N, 2L-1, 3L-2)
    for x in left_side_matrices:
        assert motifs.shape[0] == x.shape[0]
        assert (x.shape[1] == (2*L-1)) and (x.shape[2] == (3*L-2))
    return left_side_matrices


def _compute_similarity_right_side(motifs: cp.ndarray) -> list[cp.ndarray]:
    """Prepares the right side of the similarity calculation."""
    # motifs = (M, L, K)
    L = motifs.shape[1]
    K = motifs.shape[2]
    motifs_pivot = cp.transpose(motifs, axes=(0, 2, 1))  # (M, K, L)
    right_side_prepivot = motifs_pivot @ _RIGHTTENSOR(L)  # (M, K, 3L-2)
    right_side_matrix = cp.transpose(right_side_prepivot, axes=(2, 0, 1))  # (3L-2, M, K)
    right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)]  # (3L-2, M)
    for x in right_side_matrices:
        assert motifs.shape[0] == x.shape[1]
        assert x.shape[0] == (3*L-2)
    return right_side_matrices


def _LEFTTENSOR(L: int) -> cp.ndarray:
    """Produces the LEFTTENSOR needed for the left side of the similarity calculation."""
    x = cp.zeros((2*L-1, 3*L-2, L))
    for i in range(2*L-1):
        x[i, i : i + L, :] = cp.eye(L)
    return x


def _RIGHTTENSOR(L: int) -> cp.ndarray:
    """Produces the RIGHTTENSOR needed for the right side of the similarity calculation."""
    x = cp.zeros((L, 3*L-2))
    x[:, L-1:2*L-1] = cp.eye(L)
    return x