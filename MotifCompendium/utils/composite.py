
import numpy as np
import cupy as cp
import pandas as pd

import matplotlib.pyplot as plt
import logomaker

import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config

###########################
# MODIFIED SMITH-WATERMAN #
###########################
def run_modified_smith_waterman_znf(
    cwm, b1h_ppm_ic,
    sim_threshold,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f, skip_penalty_l1, skip_penalty_l2,
    finger_length=3,
):
    '''
    Arguments:
        cwm: CWM motif matrix (as CWM) - np.array of shape (L, D)
        b1h_ppm_ic: Finger motif matrix (as IC-scaled PPM) - np.array of shape (F*M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for 1 bp overlap (finger) - float
        overlap_penalty_f2: Penalty for 2 bp overlap (finger) - float
        skip_penalty_f: Penalty for skipping a finger - float
        skip_penalty_l1: Penalty for skipping a CWM position (1 bp) - float
        skip_penalty_l2: Penalty for skipping a CWM position (2 bp or more) - float
        finger_length: Length of each finger motif - int (M)
    
    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+2, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: CWM alignment indices - list of int
        msw_scale_f: Finger scaling projection - np.array of shape (F,) 
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array of shape (F+1, L+1)
        A_final: Final arrangement matrix - list of list of str
        A_f_final: Final finger alignment matrix - list of list of str
        A_l_final: Final CWM alignment matrix - list of list of str
    '''
    # Setting: IC-scale off (store)
    if utils_config.get_ic_scale():
        ic_scale = True
        utils_config.set_ic_scale(False)
    else:
        ic_scale = False
    
    ## Preprocess motifs
    # CWMs
    cwm = np.abs(cwm)  # Absolute value (L, D)
    cwm_norm = cwm/np.linalg.norm(cwm, axis=(0, 1), keepdims=True)  # L2 normalize, per motif

    unrolled_cwm_norm = unroll_motif_np(cwm_norm, finger_length) # (L-M+1, finger_length M, D)
    unrolled_cwm_norm_cp = unroll_motif_cp(cp.asarray(cwm_norm), finger_length)
    unrolled_importance = (unrolled_cwm_norm * unrolled_cwm_norm).sum(axis=(1, 2)) # (L-M+1) (L2 norm, of each M-mer, before IC-scaling)

    # B1H
    b1h_ppm_ic = np.abs(b1h_ppm_ic)  # Absolute value (F*M, D)
    # b1h_ppm_ic = b1h_ppm_ic / np.sum(b1h_ppm_ic, axis=1, keepdims=True)
    # b1h_ppm_ic = utils_motif.ic_scale(b1h_ppm_ic)
    b1h_ppm_ic_n = unroll_znf_b1h(b1h_ppm_ic) # (num_fingers F, M, D)
    b1h_ppm_ic_n_cp = cp.asarray(b1h_ppm_ic_n)

    ## Calculate similarities
    sims_ff = compute_aligned_similarity(b1h_ppm_ic_n_cp, unrolled_cwm_norm_cp) # (F, L-M+1)
    sims_fr = compute_aligned_similarity(b1h_ppm_ic_n_cp, unrolled_cwm_norm_cp[:, ::-1, ::-1]) # (F, L-M+1)

    ## Dimensions
    F, M, D = b1h_ppm_ic_n.shape # (F, M, D)
    L, D = cwm.shape # (L, D)
    L_M_1 = L - M + 1 # (L-M+1)

    ## Run Modified Smith-Waterman: Local
    (msw_score, msw_orient, msw_actions, H_final, A_final) = calculate_modified_smith_waterman(
        matrix_f=b1h_ppm_ic_n,
        matrix_l=cwm_norm,
        sim_for=sims_ff,
        sim_rev=sims_fr,
        scale_factor=unrolled_importance,
        sim_threshold=sim_threshold,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f=skip_penalty_f,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        alignment_type="local",
        initial_score=0.0,
    )

    ## Add remaining global alignment
    remain_f = int(msw_actions[-1].split("@")[0][1:]) + 1  # Get last unused finger index
    remain_l = int(msw_actions[-1].split("@")[-1][1:]) + 1 # Get last unused CWM index
    if msw_orient == "rev":
        remain_sim = sims_fr
    else:
        remain_sim = sims_ff
    
    if remain_f < F and remain_l < L_M_1:
        (_, _, remain_actions, H_remain, A_remain) = calculate_modified_smith_waterman(
            matrix_f=b1h_ppm_ic_n[remain_f:],
            matrix_l=cwm_norm[remain_l:],
            sim_for=remain_sim[remain_f:, remain_l:],
            sim_rev=remain_sim[remain_f:, remain_l:],
            scale_factor=unrolled_importance[remain_l:],
            sim_threshold=sim_threshold,
            overlap_penalty_f1=overlap_penalty_f1,
            overlap_penalty_f2=overlap_penalty_f2,
            skip_penalty_f=skip_penalty_f,
            skip_penalty_l1=skip_penalty_l1,
            skip_penalty_l2=skip_penalty_l2,
            alignment_type="global",
            initial_score=msw_score,
            initial_f_idx=remain_f,
            initial_l_idx=remain_l,
            orientations=(msw_orient,),
        )

        # Update final alignments
        msw_actions = msw_actions + remain_actions
        H_final[remain_f + 1:, remain_l + 3:] = H_remain  # With local shifts
        for remain_fi, remain_row in enumerate(A_remain):  # With local shifts
            A_final[remain_f + 1 + remain_fi][remain_l + 3:] = remain_row

    ## Retrieve alignments
    (aligned_motif_fs, msw_scale_f, msw_align_f, msw_align_l) = get_alignment_modified_smith_waterman(
        matrix_f=b1h_ppm_ic_n,
        matrix_l=cwm_norm,
        scale_factor=unrolled_importance,
        msw_actions=msw_actions,
    )

    ## Calculate similarity score
    aligned_motif_fs_collapse = aligned_motif_fs.sum(axis=0)
    max_len = max(cwm.shape[0], aligned_motif_fs_collapse.shape[0])
    sim_score, _, _ = utils_similarity.compute_similarities(
        [utils_motif.pad_motif(aligned_motif_fs_collapse[np.newaxis, :, :], max_len), utils_motif.pad_motif(cwm[np.newaxis, :, :], max_len)],
        [(0, 1)],
    )[0]
    
    # Restore IC-scale setting
    if ic_scale:
        utils_config.set_ic_scale(True)

    return msw_score, sim_score, aligned_motif_fs, msw_orient, msw_align_f, msw_align_l, msw_scale_f, msw_actions, H_final, A_final

###########
# FINGERS #
###########
def unroll_znf_b1h(b1h_ppm):
    '''
    Argument:
        b1h_ppm: np.array of shape (L, D)
    Returns:
        b1h_ppm: np.array of shape (F, 3, D)
    '''
    assert(len(b1h_ppm.shape) == 2 and b1h_ppm.shape[-1] == 4)
    L = b1h_ppm.shape[0]
    assert(L%3 == 0)
    num_fingers = L // 3
    b1h_ppm_unroll = b1h_ppm.reshape(num_fingers, 3, 4)
    return b1h_ppm_unroll


#############
# UNROLLING #
#############
def unroll_motif_np(motif, U):
    L, K = motif.shape
    unrolled_motif = _tensor3_matmul_tensor2(_UNRAVELLER_NP(L, U), motif)
    unrolled_motif = unrolled_motif.transpose((1, 0, 2))
    assert(unrolled_motif.shape == (L-U+1, U, K))
    return unrolled_motif


def unroll_motif_cp(motif, U):
    L, K = motif.shape
    unrolled_motif = _tensor3_matmul_tensor2(_UNRAVELLER_CP(L, U), motif)
    unrolled_motif = unrolled_motif.transpose((1, 0, 2))
    assert(unrolled_motif.shape == (L-U+1, U, K))
    return unrolled_motif


_UNRAVELLER_TENSOR_NP = None
def _UNRAVELLER_NP(L, U):
    # L = length of input sequence
    # U = length to unravel to
    # Unravels (L, K) --> (L-U+1, U, K)
    global _UNRAVELLER_TENSOR_NP
    create_tensor = False
    if _UNRAVELLER_TENSOR_NP is None:
        create_tensor = True
    elif _UNRAVELLER_TENSOR_NP.shape != (U, L-U+1, L):
        create_tensor = True
    if create_tensor:
        _UNRAVELLER_TENSOR_NP = np.zeros((U, L-U+1, L))
        for i in range(U):
            _UNRAVELLER_TENSOR_NP[i, :, i:i+L-U+1] = np.eye(L-U+1)
    return _UNRAVELLER_TENSOR_NP

_UNRAVELLER_TENSOR_CP = None
def _UNRAVELLER_CP(L, U):
    # L = length of input sequence
    # U = length to unravel to
    # Unravels (L, K) --> (L-U+1, U, K)
    global _UNRAVELLER_TENSOR_CP
    create_tensor = False
    if _UNRAVELLER_TENSOR_CP is None:
        create_tensor = True
    elif _UNRAVELLER_TENSOR_CP.shape != (U, L-U+1, L):
        create_tensor = True
    if create_tensor:
        _UNRAVELLER_TENSOR_CP = cp.zeros((U, L-U+1, L))
        for i in range(U):
            _UNRAVELLER_TENSOR_CP[i, :, i:i+L-U+1] = cp.eye(L-U+1)
    return _UNRAVELLER_TENSOR_CP


def _tensor3_matmul_tensor2(x, y):
    """Multiplies a (N, L, K) tensor with a (K, M) tensor efficiently."""
    N, L, K = x.shape
    M = y.shape[1]
    x_flat = x.reshape(N * L, K)  # (NL, K)
    result = x_flat @ y  # (NL, M)
    return result.reshape(N, L, M)  # (N, L, M)


##############
# SIMILARITY #
##############
def compute_aligned_similarity(x, y):
    """Computes the similarity of a (N, L, K) tensor with a (M, L, K) tensor."""
    N, L, K = x.shape
    M = y.shape[0]
    x_normalized = x/cp.linalg.norm(x, axis=(1, 2), keepdims=True)
    y_normalized = y/cp.linalg.norm(y, axis=(1, 2), keepdims=True)
    x_2d = x_normalized.reshape(N, L*K)
    y_2d = y_normalized.reshape(M, L*K)
    result = (x_2d @ y_2d.T).get()  # (N, M) numpy array
    assert result.shape == (N, M), f"Expected ({N}, {M}), got {result.shape}"
    return result


############
# SHUFFLE #
############
def shuffle_motif_position(motif):
    '''
    Argument:
        Input, motif: np.array of shape (L, D)
    Returns:
        Shuffled, motif: np.array of shape (L, D)
    '''
    assert(len(motif.shape) == 2 and motif.shape[-1] == 4)
    # Shuffle positions
    L, D = motif.shape
    perm = np.random.permutation(L)
    shuffled_motif = motif[perm, :]
    return shuffled_motif


###################
# SMITH-WATERMAN  #
###################
def calculate_modified_smith_waterman(
    matrix_f, matrix_l,
    sim_for, sim_rev, 
    scale_factor, sim_threshold,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f, skip_penalty_l1, skip_penalty_l2,
    alignment_type="local",
    initial_score=0.0,
    initial_f_idx=0,
    initial_l_idx=0,
    orientations=("for", "rev"),
):
    '''
    Arguments:
        matrix_f: Finger motif matrices - np.array of shape (F, 3, D)
        matrix_l: CWM motif matrix - np.array of shape (L, D)
        sim_for: Similarity scores (forward) - np.array of shape (F, L-M+1)
        sim_rev: Similarity scores (reverse) - np.array of shape (F, L-M+1)
        scale_factor: Scaling factors for each CWM position - np.array of shape (L-M+1,)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for 1 bp overlap (finger) - float
        overlap_penalty_f2: Penalty for 2 bp overlap (finger) - float
        skip_penalty_f: Penalty for skipping a finger - float
        skip_penalty_l1: Penalty for skipping a CWM position by 1 - float
        skip_penalty_l2: Penalty for skipping a CWM position by 2 or more - float
        alignment_type: Type of alignment ("local" or "global") - str
        initial_score: Initial score for alignment - float
        initial_f_idx: Initial finger index - int
        initial_l_idx: Initial CWM position index - int
        orientations: Which orientation(s) to evaluate - tuple of "for"/"rev"
    Returns:
        msw_score: Optimal alignment score - float
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array of shape (F+1, L+1)
        A_final: Final arrangement matrix - list of list of str
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_threshold < 1
    assert overlap_penalty_f1 <= 0
    assert overlap_penalty_f2 <= 0
    assert skip_penalty_f <= 0
    assert skip_penalty_l1 <= 0
    assert skip_penalty_l2 <= 0
    assert alignment_type in ["local", "global"]

    F, L_M_1 = sim_for.shape # (F, L-M+1)
    L, D = matrix_l.shape # (L, D)

    # Shift: Global, local
    if alignment_type == 'local':
        shift_idx_f = 1
        shift_idx_l = 3
    elif alignment_type == 'global':
        shift_idx_f = 0
        shift_idx_l = 0

    # Record final
    H_final = np.zeros((F + shift_idx_f, L + shift_idx_l)) + initial_score # Optimal score (float) (F + shift_idx_f, L + shift_idx_l)
    A_final = [[[] for _ in range(L + shift_idx_l)] for _ in range(F + shift_idx_f)] # Optimal arrangement (str) (F + shift_idx_f, L + shift_idx_l)

    msw_score = -np.inf
    msw_actions = []
    msw_orient = None

    orient_sims = {"for": sim_for, "rev": sim_rev}
    for orient in orientations:
        sim = orient_sims[orient]
        # Initialize matrices
        H = np.zeros((F + shift_idx_f, L + shift_idx_l)) + initial_score # Local optimal score (float) (F + shift_idx_f, L + shift_idx_l)
        A = [[[] for _ in range(L + shift_idx_l)] for _ in range(F + shift_idx_f)] # Local optimal arrangement (str) (F + shift_idx_f, L + shift_idx_l)
        best_score = -np.inf
        best_actions = [] # Add in reverse order

        # F: Fingers
        for f in range(F):
            # L: CWM Lengths
            for l in range(L_M_1):
                idx_f = f + shift_idx_f # f index for matrices (shifted by shift_idx_f)
                idx_l = l + shift_idx_l # l index for matrices (shifted by shift_idx_l)

                # Diagonal 1,2,3 -----------------------------------------------------------------------
                sim_score = sim[f, l]
                match_score = scale_factor[l] * sim_score
                diag1_score, diag2_score, diag3_score = -np.inf, -np.inf, -np.inf
                diag1_actions, diag2_actions, diag3_actions = [], [], []

                if sim_score > sim_threshold:
                    ## Diagonal 1: [-1, -1] ----------
                    diag1_prev_score = H[idx_f-1, idx_l-1] # Previous states
                    diag1_prev_actions = A[idx_f-1][idx_l-1]

                    # F: Overlap: 2,1 bp; L: Skip penalty
                    diag1_penalty = match_score
                    diag1_action = [f"F{f + initial_f_idx}@match@L{l + initial_l_idx}"]
                    for action in diag1_prev_actions:
                        action_type = action.split("@")[1]
                        f_i = int(action.split("@")[0][1:])
                        l_j = int(action.split("@")[-1][1:])
                        if action_type == "match":
                            # Overlap: 2 bp (from previous finger)
                            if l - l_j == 1:
                                if l-2 >= 0:
                                    diag1_penalty = diag1_penalty + overlap_penalty_f1 + overlap_penalty_f2
                                    diag1_action = [f"F{f + initial_f_idx}@F-overlap@L{l-2 + initial_l_idx}"] + [f"F{f + initial_f_idx}@F-overlap@L{l-1 + initial_l_idx}"] + diag1_action
                                    break
                                elif l-1 >= 0:
                                    diag1_penalty = diag1_penalty + overlap_penalty_f1
                                    diag1_action = [f"F{f + initial_f_idx}@F-overlap@L{l-1 + initial_l_idx}"] + diag1_action
                                    break
                            # Overlap: 1 bp (from previous finger)
                            elif l - l_j == 2:
                                if l-1 >= 0:
                                    diag1_penalty = diag1_penalty + overlap_penalty_f1
                                    diag1_action = [f"F{f + initial_f_idx}@F-overlap@L{l-1 + initial_l_idx}"] + diag1_action
                                    break
                            elif l - l_j > 2:
                                break
                    
                    diag1_score = diag1_prev_score + diag1_penalty
                    diag1_actions = diag1_action + diag1_prev_actions

                    ## Diagonal 2: [-1, -2] ----------
                    diag2_prev_score = H[idx_f-1, idx_l-2]
                    diag2_prev_actions = A[idx_f-1][idx_l-2]

                    # Check for overlap (1bp)
                    diag2_penalty = match_score
                    diag2_action = [f"F{f + initial_f_idx}@match@L{l + initial_l_idx}"] + [f"F{f + initial_f_idx}@L-skip@L{l-1 + initial_l_idx}"]
                    for action in diag2_prev_actions:
                        action_type = action.split("@")[1]
                        f_i = int(action.split("@")[0][1:])
                        l_j = int(action.split("@")[-1][1:])
                        # F: Overlap: 1 bp; L: Skip penalty
                        if action_type == "match":
                            if l - l_j == 2:
                                if l-1 >= 0:
                                    diag2_penalty = diag2_penalty + overlap_penalty_f1
                                    diag2_action = [f"F{f + initial_f_idx}@F-overlap@L{l-1 + initial_l_idx}"] + diag2_action
                                    break
                            elif l - l_j == 3:
                                continue
                            elif l - l_j == 4:
                                diag2_penalty = diag2_penalty + skip_penalty_l1
                                break
                            elif l - l_j > 4:
                                diag2_penalty = diag2_penalty + skip_penalty_l2
                                break

                    diag2_score = diag2_prev_score + diag2_penalty
                    diag2_actions = diag2_action + diag2_prev_actions

                    ## Diagonal 3: [-1, -3] ----------
                    diag3_prev_score = H[idx_f-1, idx_l-3]
                    diag3_prev_actions = A[idx_f-1][idx_l-3]

                    diag3_penalty = match_score
                    diag3_action = [f"F{f + initial_f_idx}@match@L{l + initial_l_idx}"] + [f"F{f + initial_f_idx}@L-skip@L{l-1 + initial_l_idx}"] + [f"F{f + initial_f_idx}@L-skip@L{l-2 + initial_l_idx}"]
                    for action in diag3_prev_actions:
                        action_type = action.split("@")[1]
                        f_i = int(action.split("@")[0][1:])
                        l_j = int(action.split("@")[-1][1:])
                        # F: No overlap; L: Skip penalty
                        if action_type == "match":
                            if l - l_j == 4:
                                diag3_penalty = diag3_penalty + skip_penalty_l1
                                break
                            elif l - l_j > 4:
                                diag3_penalty = diag3_penalty + skip_penalty_l2
                                break

                    diag3_score = diag3_prev_score + diag3_penalty
                    diag3_actions = diag3_action + diag3_prev_actions

                ## Down: Skip finger -----------------------------------------------------------
                down_prev_score = H[idx_f-1, idx_l] # Previous states
                down_prev_actions = A[idx_f-1][idx_l]

                down_penalty = skip_penalty_f
                down_action = [f"F{f + initial_f_idx}@F-skip@L{l + initial_l_idx}"]

                down_score = down_prev_score + down_penalty
                down_actions = down_action + down_prev_actions

                ## Right: Skip CWM position -----------------------------------------------------------
                right_prev_score = H[idx_f, idx_l-1] # Previous states
                right_prev_actions = A[idx_f][idx_l-1]

                # Skip type: Inside finger, single, repeated
                right_penalty = skip_penalty_l1
                right_action = [f"F{f + initial_f_idx}@L-skip@L{l + initial_l_idx}"]
                for action in right_prev_actions:
                    action_type = action.split("@")[1]
                    f_i = int(action.split("@")[0][1:])
                    l_j = int(action.split("@")[-1][1:])
                    if action_type == "match":
                        if l - l_j < 3:
                            right_penalty = 0 # Inside match: No penalty
                            break
                    elif action_type == "L-skip":
                        if l - l_j > 1:
                            right_penalty = skip_penalty_l2 # Repeated: Extra penalty
                            break

                right_score = right_prev_score + right_penalty
                right_actions = right_action + right_prev_actions

                # Consider all options
                options_score = np.array([
                    diag1_score,  # Option 1-1: Bind finger f @ l (from state: [-1, -1])
                    diag2_score,  # Option 1-2: Bind finger f @ l (from state: [-1, -2])
                    diag3_score,  # Option 1-3: Bind finger f @ l (from state: [-1, -3])
                    down_score,  # Option 2: Keep finger f unused
                    right_score,  # Option 3: Bind finger f previous to l
                ])

                options_actions = [
                    diag1_actions,
                    diag2_actions,
                    diag3_actions,
                    down_actions,
                    right_actions,
                ]

                # Select best option
                bestoption_idx = np.argmax(options_score)
                bestoption_score = options_score[bestoption_idx]
                if alignment_type == "local":
                    bestoption_score = max(bestoption_score, initial_score) # Local alignment: No negative scores
                bestoption_actions = options_actions[bestoption_idx]

                # Record
                H[idx_f, idx_l] = bestoption_score
                A[idx_f][idx_l] = bestoption_actions

                # Local alignment: Record best score, so far
                if bestoption_score > best_score:
                    best_score = bestoption_score
                    best_actions = bestoption_actions

        # Global alignment: Take final cell
        if alignment_type == 'local':
            best_idx = H.argmax()
            best_i, best_j = np.unravel_index(best_idx, H.shape)
            best_score = float(H.flatten()[best_idx])
            best_actions = A[best_i][best_j]
        elif alignment_type == "global":
            last_H = np.concatenate([H[-1, :], H[:, -1]])
            last_A = A[-1] + [A[i][-1] for i in range(H.shape[0])]
            best_idx = last_H.argmax()
            best_score = last_H.flatten()[best_idx]
            best_actions = last_A[best_idx]

        # Record final optimal
        if best_score >= msw_score:
            H_final = H # Full matrix: Scores
            A_final = A # Full matrix: Actions
            msw_score = float(best_score) # Final score
            msw_actions = best_actions[::-1] # Reorder: Start to end
            msw_orient = orient # Orientation

    return (msw_score, msw_orient, msw_actions, H_final, A_final)


def get_alignment_modified_smith_waterman(
    matrix_f, matrix_l, 
    scale_factor,
    msw_actions,
):
    '''
    Arguments:
        matrix_f: B1H finger motif matrices - np.array of shape (F, 3, D)
        matrix_l: CWM motif matrix - np.array of shape (L, D)
        scale_factor: Scaling factors for each CWM position - np.array of shape (L-M+1,)
        msw_actions: Optimal alignment arrangement - list of str
    
    Returns:
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+2, D)
        msw_scale_f: Scaling of B1H finger motifs - np.array of shape (F)
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: CWM alignment indices - list of int
    '''
    F = matrix_f.shape[0]
    L, D = matrix_l.shape
    
    ## Traceback
    aligned_motif_fs = np.zeros((F, L + 2, D)) # (F, L+2, D)
    msw_scale_f = np.zeros(F)
    msw_align_f = [f_i for f_i in range(3*F)]
    msw_align_l = [l_j for l_j in range(L)]
    
    # First: Unused positions
    action = msw_actions[0]
    action_type = action.split("@")[1]
    f_i = int(action.split("@")[0][1:])
    l_j = int(action.split("@")[-1][1:])
    if f_i == 0 and l_j == 0:  # Start together
        pass
    elif f_i > 0 and l_j == 0:  # Skipped fingers
        align_l_j = msw_align_l.index(l_j)
        msw_align_l[align_l_j : align_l_j] = ['-'] * 3 * (f_i)
    elif l_j > 0 and f_i == 0:  # Skipped CWMs
        align_f_i = msw_align_f.index(3*f_i)
        msw_align_f[align_f_i : align_f_i] = ['-'] * (l_j)
    else:
        print(msw_actions)
        raise ValueError(f"Incomplete alignment: first action {action}")
    
    # End: Unused positions
    action = msw_actions[-1]
    action_type = action.split("@")[1]
    f_i = int(action.split("@")[0][1:])
    l_j = int(action.split("@")[-1][1:]) + 2  # 1 Finger = Matched position + 2 bp = 3bp
    if (f_i + 1) == F and (l_j + 1) == L:  # End together
        pass
    elif (f_i + 1) < F and (l_j + 1) == L: # Skipped fingers
        align_l_j = msw_align_l.index(l_j)
        msw_align_l[align_l_j + 1 : align_l_j + 1] = ['-'] * 3 * (F - f_i - 1)
    elif (l_j + 1) < L and (f_i + 1) == F:  # Skipped CWMs
        align_f_i = msw_align_f.index(3*(f_i))
        msw_align_f[align_f_i + 3 : align_f_i + 3] = ['-'] * (L - l_j - 1)
    else:
        print(msw_actions)
        raise ValueError(f"Incomplete alignment: final action {action}")

    # Middle actions
    last_match = (0, "match", 0)
    for action in msw_actions:
        action_type = action.split("@")[1]
        f_i = int(action.split("@")[0][1:])
        l_j = int(action.split("@")[-1][1:])
        # Match: Place scaled finger
        if action_type == "match":
            scale_ij = scale_factor[l_j]
            aligned_motif_fs[f_i, l_j:l_j+3, :] = matrix_f[f_i] * scale_ij
            msw_scale_f[f_i] = scale_ij
            last_match = (f_i, action_type, l_j)
        # F-overlap: Insert gap in CWM
        elif action_type == "F-overlap":
            align_l_j = msw_align_l.index(l_j)
            msw_align_l[align_l_j:align_l_j] = ['-']
        # F-skip: Insert 3 x gaps in CWM
        elif action_type == "F-skip":
            align_l_j = msw_align_l.index(l_j)
            msw_align_l[align_l_j:align_l_j] = ['-'] * 3
        # L-skip: Insert gap in finger
        elif action_type == "L-skip":
            if l_j > last_match[-1] + 2:
                align_f_i = msw_align_f.index(3*(f_i)) # Only add if not inside match
                msw_align_f[align_f_i:align_f_i] = ['-']

    return aligned_motif_fs, msw_scale_f, msw_align_f, msw_align_l

##############
# VISUALIZE  #
##############
def visualize_motif(motif, save_path):
    """
    motif: np.array of shape (L, 4)
    Draws the motif logo and saves to save_path.
    """
    L, D = motif.shape
    assert D == 4

    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 2),
        squeeze=False
    )

    ax = ax[0, 0]

    df = pd.DataFrame(motif, columns=["A", "C", "G", "T"])
    logomaker.Logo(df, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    ymin, ymax = ax.get_ylim()
    if ymin == ymax:
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)

def visualize_motifs_n(motifs, save_path):
    """
    motifs: np.array of shape (N, L, 4)
    Draws all motifs vertically stacked, and adds 
    vertical alignment lines where each motif's non-zero 
    region begins and ends (computed per motif, drawn on all axes).
    """

    N, L, D = motifs.shape
    assert D == 4

    # --------- Compute alignment boundaries ----------
    starts = []
    ends = []

    for motif in motifs:
        col_sum = motif.sum(axis=1)
        nz = np.nonzero(col_sum)[0]
        if len(nz) == 0:
            starts.append(None)
            ends.append(None)
        else:
            starts.append(int(nz[0]))
            ends.append(int(nz[-1]))

    # --------- Compute global max height ----------
    if len(motifs) > 1:
        max_height = max(
            motif.sum(axis=1).max()
            for motif in motifs[1:]
        )
    else:
        max_height = motifs[0].sum(axis=1).max()

    # --------- Create figure ----------
    fig, axes = plt.subplots(
        N, 1,
        figsize=(10, 2 * N),
        squeeze=False
    )

    for i, motif in enumerate(motifs):
        ax = axes[i, 0]

        # Background for first motif
        if i == 0:
            ax.set_facecolor('lightgreen')

        # Draw motif logo
        col_sum = motif.sum(axis=1)
        if np.all(col_sum == 0):
            ax.set_ylim(0, 1)
        else:
            df = pd.DataFrame(motif, columns=["A", "C", "G", "T"])
            logomaker.Logo(df, ax=ax)
        
        # Keep y-axis scale
        if i > 1:
            ax.set_ylim(0, max_height)

        # Draw vertical lines for *all* motif boundaries
        for s, e in zip(starts, ends):
            if s is not None:
                ax.axvline(s, color="red", linestyle="--", linewidth=1)
            if e is not None:
                ax.axvline(e, color="blue", linestyle="--", linewidth=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel(f"Motif {i+1}")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
