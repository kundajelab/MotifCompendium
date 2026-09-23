
import math

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
def run_modified_smith_waterman(
    target_motif, subunit_motif_stack,
    sim_threshold=0.0,
    overlap_penalty_l1=-0.0, overlap_penalty_l2=-0.0,
    overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
    skip_penalty_f1=-0.0, skip_penalty_f2=-0.0,
    skip_penalty_l1=-0.0, skip_penalty_l2=-0.0,
    return_matrices=False,
    label_unsigned=True,
):
    '''
    Aligns a chain of fixed-length finger motifs against a motif, for any finger length M > 1.

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs - np.array of shape (F, M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_l1: Penalty for 1 bp overlap - float, or 'gm' / 'geometric_mean' for minus the
            geometric mean of the overlapping fingers' placed heights at that position
        overlap_penalty_l2: Penalty for 2 bp or more overlap - float, or 'gm' / 'geometric_mean'
        overlap_penalty_f1: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when overlap occurs due to only 2 fingers overlapping - float
        overlap_penalty_f2: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when the overlapping position has 3 or more overlapping fingers - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for skipping 2 or more fingers - float
        skip_penalty_l1: Penalty for skipping a motif position (1 bp) - float
        skip_penalty_l2: Penalty for skipping a motif position (2 bp or more) - float
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+M-1, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: Motif alignment indices - list of int
        msw_scale_f: Finger scaling projection - np.array of shape (F,)
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array of shape (F+1, L+M), None unless requested
        A_final: Final arrangement matrix - list of list of str, None unless requested
    '''
    return _run_modified_smith_waterman(
        target_motif=target_motif,
        subunit_motif_stack=subunit_motif_stack,
        sim_threshold=sim_threshold,
        overlap_penalty_l1=overlap_penalty_l1,
        overlap_penalty_l2=overlap_penalty_l2,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        calculate_fn=calculate_modified_smith_waterman,
        return_matrices=return_matrices,
        label_unsigned=label_unsigned,
    )


def run_modified_smith_waterman_znf(
    cwm, b1h_ppm_ic,
    sim_threshold=0.0,
    overlap_penalty_l1=-0.02, overlap_penalty_l2=-0.1,
    overlap_penalty_f1=1.0, overlap_penalty_f2=1.0,
    skip_penalty_f1=-0.1, skip_penalty_f2=-0.1,
    skip_penalty_l1=-0.02, skip_penalty_l2=-0.1,
    return_matrices=False,
    label_unsigned=True,
    finger_length=3,
):
    '''
    Aligns a chain of B1H zinc-finger motifs against a CWM.

    With finger_length = 3 this uses the frozen 3 bp recurrence
    (calculate_modified_smith_waterman_znf), so that B1H zinc-finger results do not change;
    any other finger length uses the generalized recurrence (calculate_modified_smith_waterman).

    Arguments:
        cwm: CWM motif matrix (as CWM) - np.array of shape (L, D)
        b1h_ppm_ic: Finger motif matrix (as IC-scaled PPM) - np.array of shape (F*M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_l1: Penalty for 1 bp overlap - float, or 'gm' / 'geometric_mean' for minus the
            geometric mean of the overlapping fingers' placed heights at that position
        overlap_penalty_l2: Penalty for 2 bp or more overlap - float, or 'gm' / 'geometric_mean'
        overlap_penalty_f1: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when overlap occurs due to only 2 fingers overlapping - float
        overlap_penalty_f2: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when the overlapping position has 3 or more overlapping fingers - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for skipping 2 or more fingers - float
        skip_penalty_l1: Penalty for skipping a CWM position (1 bp) - float
        skip_penalty_l2: Penalty for skipping a CWM position (2 bp or more) - float
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool
        finger_length: Length of each finger motif - int (M)

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+M-1, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: CWM alignment indices - list of int
        msw_scale_f: Finger scaling projection - np.array of shape (F,)
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array of shape (F+1, L+M), None unless requested
        A_final: Final arrangement matrix - list of list of str, None unless requested
    '''
    # Recurrence: Frozen for 3 bp fingers, generalized otherwise
    if finger_length == 3:
        calculate_fn = calculate_modified_smith_waterman_znf
    else:
        calculate_fn = calculate_modified_smith_waterman

    # Unroll fingers
    subunit_motif_stack = unroll_fingers(b1h_ppm_ic, finger_length) # (F, M, D)

    return _run_modified_smith_waterman(
        target_motif=cwm,
        subunit_motif_stack=subunit_motif_stack,
        sim_threshold=sim_threshold,
        overlap_penalty_l1=overlap_penalty_l1,
        overlap_penalty_l2=overlap_penalty_l2,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        calculate_fn=calculate_fn,
        return_matrices=return_matrices,
        label_unsigned=label_unsigned,
    )


def _run_modified_smith_waterman(
    target_motif, subunit_motif_stack,
    sim_threshold,
    overlap_penalty_l1, overlap_penalty_l2,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    calculate_fn,
    return_matrices,
    label_unsigned,
):
    '''
    Aligns a chain of fixed-length finger motifs against a motif, for any finger length M > 1.

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs - np.array of shape (F, M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_l1: Penalty for 1 bp overlap - float, or 'gm' / 'geometric_mean' for minus the
            geometric mean of the overlapping fingers' placed heights at that position
        overlap_penalty_l2: Penalty for 2 bp or more overlap - float, or 'gm' / 'geometric_mean'
        overlap_penalty_f1: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when overlap occurs due to only 2 fingers overlapping - float
        overlap_penalty_f2: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when the overlapping position has 3 or more overlapping fingers - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for skipping 2 or more fingers - float
        skip_penalty_l1: Penalty for skipping a motif position (1 bp) - float
        skip_penalty_l2: Penalty for skipping a motif position (2 bp or more) - float
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+M-1, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: Motif alignment indices - list of int
        msw_scale_f: Finger scaling projection - np.array of shape (F,)
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array of shape (F+1, L+M), None unless requested
        A_final: Final arrangement matrix - list of list of str, None unless requested
    '''
    # Setting: IC-scale off (store)
    if utils_config.get_ic_scale():
        ic_scale = True
        utils_config.set_ic_scale(False)
    else:
        ic_scale = False

    ## Dimensions
    F, M, D = subunit_motif_stack.shape # (F, M, D)
    L, D = target_motif.shape # (L, D)
    L_M_1 = L - M + 1 # (L-M+1)
    assert L_M_1 > 0, f"Target motif length {L} is shorter than finger length {M}"

    ## Preprocess motifs
    # Unsigned: Absolute value
    if label_unsigned:
        target_motif = np.abs(target_motif)  # Absolute value (L, D)
        subunit_motif_stack = np.abs(subunit_motif_stack)  # Absolute value (F, M, D)

    # Normalize
    target_motif_norm = target_motif/np.linalg.norm(target_motif, axis=(0, 1), keepdims=True)  # L2 normalize, per motif
    subunit_motif_norms = np.linalg.norm(subunit_motif_stack, axis=(1, 2), keepdims=True)
    subunit_motif_stack_norm = subunit_motif_stack/np.where(subunit_motif_norms == 0, 1, subunit_motif_norms)  # L2 normalize, per motif (all-zero subunits stay zero)

    # Target motif: Unroll
    target_motif_norm_unroll = _unroll_motif(target_motif_norm, M, np) # (L-M+1, finger_length M, D)
    target_motif_norm_unroll_cp = _unroll_motif(cp.asarray(target_motif_norm), M, cp) # (L-M+1, finger_length M, D)
    target_motif_norm_unroll_importance = (target_motif_norm_unroll * target_motif_norm_unroll).sum(axis=(1, 2)) # (L - M + 1) (L2 norm, of each M-mer)

    # Subunit motifs
    subunit_motif_stack_norm_cp = cp.asarray(subunit_motif_stack_norm) # (F, M, D)

    ## Calculate similarities
    sims_ff = compute_aligned_similarity(subunit_motif_stack_norm_cp, target_motif_norm_unroll_cp) # (F, L-M+1)
    sims_fr = compute_aligned_similarity(subunit_motif_stack_norm_cp, target_motif_norm_unroll_cp[:, ::-1, ::-1]) # (F, L-M+1)

    ## Run Modified Smith-Waterman: Local
    (msw_score, msw_orient, msw_actions, H_final, A_final) = calculate_fn(
        matrix_l=target_motif_norm,
        matrix_f=subunit_motif_stack_norm,
        sim_for=sims_ff,
        sim_rev=sims_fr,
        scale_factor=target_motif_norm_unroll_importance,
        sim_threshold=sim_threshold,
        overlap_penalty_l1=overlap_penalty_l1,
        overlap_penalty_l2=overlap_penalty_l2,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        alignment_type="local",
        initial_score=0.0,
        return_matrices=return_matrices,
    )

    ## Add remaining global alignment
    remain_f, remain_l = _next_unused_indices(msw_actions, M)  # First unused finger / motif index
    if msw_orient == "rev":
        remain_sim = sims_fr
    else:
        remain_sim = sims_ff

    if len(msw_actions) > 0 and remain_f < F and remain_l < L_M_1:
        (_, _, remain_actions, H_remain, A_remain) = calculate_fn(
            matrix_l=target_motif_norm[remain_l:],
            matrix_f=subunit_motif_stack_norm[remain_f:],
            sim_for=remain_sim[remain_f:, remain_l:],
            sim_rev=remain_sim[remain_f:, remain_l:],
            scale_factor=target_motif_norm_unroll_importance[remain_l:],
            sim_threshold=sim_threshold,
            overlap_penalty_l1=overlap_penalty_l1,
            overlap_penalty_l2=overlap_penalty_l2,
            overlap_penalty_f1=overlap_penalty_f1,
            overlap_penalty_f2=overlap_penalty_f2,
            skip_penalty_f1=skip_penalty_f1,
            skip_penalty_f2=skip_penalty_f2,
            skip_penalty_l1=skip_penalty_l1,
            skip_penalty_l2=skip_penalty_l2,
            alignment_type="global",
            initial_score=msw_score,
            initial_f_idx=remain_f,
            initial_l_idx=remain_l,
            orientations=(msw_orient,),
            return_matrices=return_matrices,
        )

        # Update final alignments
        msw_actions = msw_actions + remain_actions
        if return_matrices:
            H_final[remain_f + 1:, remain_l + M:] = H_remain  # With local shifts
            for remain_fi, remain_row in enumerate(A_remain):  # With local shifts
                A_final[remain_f + 1 + remain_fi][remain_l + M:] = remain_row

    ## Retrieve alignments
    if len(msw_actions) == 0:  # No alignment found: Return empty alignment
        aligned_motif_fs = np.zeros((F, L + M - 1, D))
        msw_scale_f = np.zeros(F)
        msw_align_f = [f_i for f_i in range(M*F)]
        msw_align_l = [l_j for l_j in range(L)]
        sim_score = 0.0
    else:
        (aligned_motif_fs, msw_scale_f, msw_align_f, msw_align_l) = get_alignment_modified_smith_waterman(
            matrix_l=target_motif_norm,
            matrix_f=subunit_motif_stack_norm,
            scale_factor=target_motif_norm_unroll_importance,
            msw_actions=msw_actions,
        )

        ## Calculate similarity score
        aligned_motif_fs_collapse = aligned_motif_fs.sum(axis=0)
        max_len = max(L, aligned_motif_fs_collapse.shape[0])
        sim_score, _, _ = utils_similarity.compute_similarities(
            [utils_motif.pad_motif(aligned_motif_fs_collapse[np.newaxis, :, :], max_len), utils_motif.pad_motif(target_motif[np.newaxis, :, :], max_len)],
            [(0, 1)],
        )[0]

    # Restore IC-scale setting
    if ic_scale:
        utils_config.set_ic_scale(True)

    return msw_score, sim_score, aligned_motif_fs, msw_orient, msw_align_f, msw_align_l, msw_scale_f, msw_actions, H_final, A_final


def _next_unused_indices(msw_actions, finger_length):
    '''
    First unused finger index and first unused motif position, after an alignment.

    A matched finger occupies finger_length positions, so the motif index advances past the
    whole footprint of the last match (the final action itself may be an F-overlap or a skip).

    Arguments:
        msw_actions: Alignment arrangement - list of str
        finger_length: Length of each finger motif - int (M)
    Returns:
        remain_f: First unused finger index - int
        remain_l: First unused motif position - int
    '''
    if len(msw_actions) == 0:
        return 0, 0

    last_action = msw_actions[-1]
    remain_f = int(last_action.split("@")[0][1:]) + 1  # Last used finger index
    remain_l = int(last_action.split("@")[-1][1:]) + 1  # Last used motif index

    # Match: The finger occupies finger_length positions
    for action in reversed(msw_actions):
        if action.split("@")[1] == "match":
            remain_l = max(remain_l, int(action.split("@")[-1][1:]) + finger_length)
            break

    return remain_f, remain_l


###########
# FINGERS #
###########
def unroll_fingers(ppm, finger_length):
    '''
    Argument:
        ppm: Concatenated finger motif matrices - np.array of shape (F*M, D)
        finger_length: Length of each finger motif - int (M)
    Returns:
        ppm_unroll: np.array of shape (F, M, D)
    '''
    assert(len(ppm.shape) == 2)
    assert(finger_length > 1)
    L, D = ppm.shape
    assert(L % finger_length == 0)
    num_fingers = L // finger_length
    ppm_unroll = ppm.reshape(num_fingers, finger_length, D)
    return ppm_unroll


#############
# UNROLLING #
#############
def _unroll_motif(motif, M, xp):
    '''
    Unravels (L, K) --> (L-M+1, M, K), by gathering each window directly.
    Arguments:
        motif: np.array of shape (L, K)
        M: Window size - int
        xp: np or cp
    '''
    L, K = motif.shape
    assert L >= M
    # Gather as (M, L-M+1, K) then transpose, not (L-M+1, M, K) directly: the values are equal,
    # but downstream sums reduce in memory order, and a C-contiguous result rounds differently
    # in the last bit -- enough to flip argmax ties in the DP (changed 140/300 alignments).
    windows = xp.arange(M)[:, None] + xp.arange(L - M + 1)[None, :]  # (M, L-M+1)
    unrolled_motif = motif[windows].transpose((1, 0, 2))  # (L-M+1, M, K)
    assert(unrolled_motif.shape == (L-M+1, M, K))
    return unrolled_motif


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
def _previous_state(H, A, idx_f, idx_l, initial_score):
    '''
    Score and arrangement of a previous DP cell.

    Out-of-range cells (reachable in global alignment, where the matrices are not padded)
    are unvisited, not wrap-around neighbours.
    '''
    if idx_f < 0 or idx_l < 0:
        return initial_score, []
    return H[idx_f, idx_l], A[idx_f][idx_l]


_GEOMETRIC_MEAN = "gm"


def _parse_overlap_penalty(penalty, name):
    '''
    An overlap penalty: a fixed penalty (a number <= 0), or 'gm' / 'geometric_mean'
    (any case) for the geometric mean of the overlapping fingers, returned as "gm".
    '''
    if isinstance(penalty, str):
        if penalty.lower() in ("gm", "geometric_mean"):
            return _GEOMETRIC_MEAN
        raise ValueError(f"{name} must be a number <= 0, or 'gm' / 'geometric_mean', got {penalty!r}")
    assert penalty <= 0, f"{name} must be <= 0, got {penalty}"
    return penalty


def _num_overlap(l, l_j, finger_length):
    '''Positions shared by fingers at l and l_j, truncated at the start of the matrix.'''
    return min(finger_length - (l - l_j), l)


def _calculate_overlap_penalties(
    f, l, l_j, prev_actions, finger_length,
    overlap_penalty_l1, overlap_penalty_l2,
    overlap_penalty_f1, overlap_penalty_f2,
    finger_row_norms, scale_factor,
    initial_f_idx, initial_l_idx,
):
    '''
    Penalty for placing a finger at motif position l, when the nearest previously placed
    finger sits at motif position l_j (both in local coordinates).

    The two fingers overlap over positions [l, l_j + M - 1]. The leftmost overlapping
    position is charged overlap_penalty_l1 and the rest overlap_penalty_l2; each is
    multiplied by overlap_penalty_f1 when exactly 2 fingers cover that position, or by
    overlap_penalty_f2 when 3 or more do. The overlap is truncated at the start of the
    matrix, so that it is always representable as gap positions before l.

    A "gm" rule charges, instead of a fixed penalty, minus the geometric mean of the placed
    heights of the n fingers covering that position: -(h_1 * ... * h_n)^(1/n), where a
    finger matched at l_c has height ||its row at p - l_c||_2 * scale_factor[l_c] there --
    what is drawn in the aligned motif. It vanishes wherever any covering finger is empty.

    Arguments:
        f: Finger being placed - int (local)
        l: Motif position of the finger being placed - int
        l_j: Motif position of the nearest previously placed finger - int
        prev_actions: Arrangement of the previous DP cell - list of str
        finger_length: Length of each finger motif - int (M)
        finger_row_norms: L2 norm of each finger row, needed for "gm" - np.array of shape (F, M), or None
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        initial_f_idx: Initial finger index - int
        initial_l_idx: Initial motif position index - int
    Returns:
        penalties: Penalty for each overlapping position, left to right - list of float
    '''
    M = finger_length
    num_overlap = _num_overlap(l, l_j, M)
    if num_overlap <= 0:
        return []

    # Fixed penalties, neutral multipliers: No need to find the overlapping fingers
    fixed = not isinstance(overlap_penalty_l1, str) and not isinstance(overlap_penalty_l2, str)
    if fixed and overlap_penalty_f1 == 1.0 and overlap_penalty_f2 == 1.0:
        return [overlap_penalty_l1] + [overlap_penalty_l2] * (num_overlap - 1)

    # Previously placed fingers that can reach the overlapping positions
    prev_matches = []  # (finger, motif position), local
    for action in prev_actions:
        parts = action.split("@")
        if parts[1] != "match":
            continue
        l_i = int(parts[-1][1:]) - initial_l_idx
        if l_i < l - M + 1:
            break
        prev_matches.append((int(parts[0][1:]) - initial_f_idx, l_i))

    penalties = []
    for i, p in enumerate(range(l, l + num_overlap)):
        rule = overlap_penalty_l1 if i == 0 else overlap_penalty_l2
        covering = [(f, l)] + [(f_i, l_i) for f_i, l_i in prev_matches if l_i <= p <= l_i + M - 1]
        if isinstance(rule, str):  # Geometric mean of the placed heights
            heights = [finger_row_norms[f_c, p - l_c] * scale_factor[l_c] for f_c, l_c in covering]
            penalty_l = -math.prod(heights) ** (1.0 / len(heights))
        else:
            penalty_l = rule
        penalties.append(penalty_l * (overlap_penalty_f1 if len(covering) <= 2 else overlap_penalty_f2))
    return penalties


def _overlap_actions(f, l, num_overlap, initial_f_idx, initial_l_idx):
    '''Gap positions inserted into the motif, for an overlap of num_overlap positions.'''
    return [
        f"F{f + initial_f_idx}@F-overlap@L{l - num_overlap + i + initial_l_idx}"
        for i in range(num_overlap)
    ]


def _diagonal_actions(f, l, k, num_overlap, initial_f_idx, initial_l_idx, guard_negative):
    '''
    Arrangement added by diagonal k: the overlap gaps, the match, and the k-1 skipped
    motif positions the diagonal steps over.

    guard_negative drops skipped positions that fall before the start of the matrix. The
    frozen 3 bp recurrence emits them unguarded, so it passes False to stay bit-identical.
    '''
    actions = (
        _overlap_actions(f, l, num_overlap, initial_f_idx, initial_l_idx)
        + [f"F{f + initial_f_idx}@match@L{l + initial_l_idx}"]
    )
    for i in range(1, k):
        if guard_negative and l - i < 0:
            continue
        actions.append(f"F{f + initial_f_idx}@L-skip@L{l - i + initial_l_idx}")
    return actions


def _calculate_finger_skip_penalty(prev_actions, skip_penalty_f1, skip_penalty_f2):
    '''
    Penalty for leaving a finger unused. A run of g consecutively skipped fingers costs
    skip_penalty_f1 + (g-1) * skip_penalty_f2.

    Consecutive finger skips all sit at the same motif position, so the run is detected from
    the most recent action of the previous cell rather than from a position distance.
    '''
    if skip_penalty_f1 == skip_penalty_f2:
        return skip_penalty_f1
    if len(prev_actions) > 0 and prev_actions[0].split("@")[1] == "F-skip":
        return skip_penalty_f2  # Repeated: Extra penalty
    return skip_penalty_f1


def _extract_best_alignment(H, A, alignment_type, F, L_M_1, shift_idx_f, shift_idx_l):
    '''
    Best score and arrangement of a filled DP matrix.

    Only rows 0..F-1 and columns 0..L_M_1-1 (plus the shifts) are ever visited, so the global
    alignment reads its final row and column from there rather than from the matrix edges.
    '''
    if alignment_type == 'local':
        best_idx = H.argmax()
        best_i, best_j = np.unravel_index(best_idx, H.shape)
        best_score = float(H.flatten()[best_idx])
        best_actions = A[best_i][best_j]
    elif alignment_type == "global":
        last_f = F - 1 + shift_idx_f
        last_l = L_M_1 - 1 + shift_idx_l
        last_H = np.concatenate([H[last_f, :last_l + 1], H[:last_f + 1, last_l]])
        last_A = [A[last_f][j] for j in range(last_l + 1)] + [A[i][last_l] for i in range(last_f + 1)]
        best_idx = last_H.argmax()
        best_score = float(last_H.flatten()[best_idx])
        best_actions = last_A[best_idx]
    return best_score, best_actions


def calculate_modified_smith_waterman(
    matrix_l, matrix_f,
    sim_for, sim_rev,
    scale_factor, sim_threshold,
    overlap_penalty_l1, overlap_penalty_l2,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    alignment_type="local",
    initial_score=0.0,
    initial_f_idx=0,
    initial_l_idx=0,
    orientations=("for", "rev"),
    return_matrices=False,
):
    '''
    Modified Smith-Waterman, for any finger length M > 1.

    Each finger covers M motif positions, so a finger placed at motif position l can be
    reached from M diagonals: diagonal k comes from the cell M - k positions earlier and
    inserts k-1 skipped motif positions. Given the distance d = l - l_j to the nearest
    previously placed finger:
        d <  M: the fingers overlap over M - d positions
        d == M: the fingers are adjacent
        d >  M: the fingers are separated by a gap of d - M positions

    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        sim_for: Similarity scores (forward) - np.array of shape (F, L-M+1)
        sim_rev: Similarity scores (reverse) - np.array of shape (F, L-M+1)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_l1: Penalty for 1 bp overlap - float, or 'gm' / 'geometric_mean' for minus the
            geometric mean of the overlapping fingers' placed heights at that position
        overlap_penalty_l2: Penalty for 2 bp or more overlap - float, or 'gm' / 'geometric_mean'
        overlap_penalty_f1: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when overlap occurs due to only 2 fingers overlapping - float
        overlap_penalty_f2: Penalty multiplier x overlap_penalty_l1 or overlap_penalty_l2 at each
            overlapping position, when the overlapping position has 3 or more overlapping fingers - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for skipping 2 or more fingers - float
        skip_penalty_l1: Penalty for skipping a motif position by 1 - float
        skip_penalty_l2: Penalty for skipping a motif position by 2 or more - float
        alignment_type: Type of alignment ("local" or "global") - str
        initial_score: Initial score for alignment - float
        initial_f_idx: Initial finger index - int
        initial_l_idx: Initial motif position index - int
        orientations: Which orientation(s) to evaluate - tuple of "for"/"rev"
        return_matrices: Also return H_final/A_final; None unless requested - bool
    Returns:
        msw_score: Optimal alignment score - float
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Final score matrix - np.array (F+shift_f, L+shift_l), None unless requested
        A_final: Final arrangement matrix - list of list of str, None unless requested
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    overlap_penalty_l1 = _parse_overlap_penalty(overlap_penalty_l1, "overlap_penalty_l1")
    overlap_penalty_l2 = _parse_overlap_penalty(overlap_penalty_l2, "overlap_penalty_l2")
    assert overlap_penalty_f1 >= 0
    assert overlap_penalty_f2 >= 0
    assert skip_penalty_f1 <= 0
    assert skip_penalty_f2 <= 0
    assert skip_penalty_l1 <= 0
    assert skip_penalty_l2 <= 0
    assert alignment_type in ["local", "global"]

    F, L_M_1 = sim_for.shape # (F, L-M+1)
    L, D = matrix_l.shape # (L, D)
    M = matrix_f.shape[1] # Finger length
    assert M > 1
    assert L_M_1 == L - M + 1

    # Geometric-mean overlap penalty: Norm of each finger row
    if isinstance(overlap_penalty_l1, str) or isinstance(overlap_penalty_l2, str):
        finger_row_norms = np.linalg.norm(matrix_f, axis=2) # (F, M)
    else:
        finger_row_norms = None

    # Zero penalties: Skip their logic entirely. Overlaps are still detected, because the
    # F-overlap actions (gaps in the motif alignment) depend on them, not on the penalty.
    use_overlap = not (overlap_penalty_l1 == 0 and overlap_penalty_l2 == 0) and not (overlap_penalty_f1 == 0 and overlap_penalty_f2 == 0)
    use_skip_l = not (skip_penalty_l1 == 0 and skip_penalty_l2 == 0)
    use_skip_f = not (skip_penalty_f1 == 0 and skip_penalty_f2 == 0)

    # Shift: Global, local
    if alignment_type == 'local':
        shift_idx_f = 1
        shift_idx_l = M
    elif alignment_type == 'global':
        shift_idx_f = 0
        shift_idx_l = 0

    # Record final
    H_final = None # Optimal score (float) (F + shift_idx_f, L + shift_idx_l), if requested
    A_final = None # Optimal arrangement (str) (F + shift_idx_f, L + shift_idx_l), if requested

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
            # L: Motif Lengths
            for l in range(L_M_1):
                idx_f = f + shift_idx_f # f index for matrices (shifted by shift_idx_f)
                idx_l = l + shift_idx_l # l index for matrices (shifted by shift_idx_l)

                # Diagonals 1..M -----------------------------------------------------------------------
                sim_score = sim[f, l]
                match_score = scale_factor[l] * sim_score
                diag_scores = [-np.inf] * M
                diag_overlaps = [0] * M

                if sim_score > sim_threshold:
                    for k in range(1, M + 1):
                        ## Diagonal k: [-1, -k] ----------
                        diag_prev_score, diag_prev_actions = _previous_state(H, A, idx_f-1, idx_l-k, initial_score) # Previous states

                        # F: Overlap; L: Skip penalty
                        diag_penalty = match_score
                        for action in diag_prev_actions:
                            parts = action.split("@")
                            if parts[1] != "match":
                                continue
                            l_j = int(parts[-1][1:]) - initial_l_idx  # Nearest previous finger
                            assert l - l_j >= k, f"Previous finger at {l_j} is ahead of diagonal {k} at {l}"
                            # Overlap: (M - d) bp (from previous finger)
                            if l - l_j < M:
                                if use_overlap:
                                    overlap_penalties = _calculate_overlap_penalties(
                                        f, l, l_j, diag_prev_actions, M,
                                        overlap_penalty_l1, overlap_penalty_l2,
                                        overlap_penalty_f1, overlap_penalty_f2,
                                        finger_row_norms, scale_factor,
                                        initial_f_idx, initial_l_idx,
                                    )
                                    diag_overlaps[k-1] = len(overlap_penalties)
                                    for overlap_penalty in overlap_penalties:
                                        diag_penalty = diag_penalty + overlap_penalty
                                else:
                                    diag_overlaps[k-1] = max(0, _num_overlap(l, l_j, M))
                            # Gap: (d - M) bp (from previous finger)
                            elif use_skip_l:
                                if l - l_j == M + 1:
                                    diag_penalty = diag_penalty + skip_penalty_l1
                                elif l - l_j > M + 1:
                                    diag_penalty = diag_penalty + skip_penalty_l2
                            break

                        diag_scores[k-1] = diag_prev_score + diag_penalty

                ## Down: Skip finger -----------------------------------------------------------
                down_prev_score, down_prev_actions = _previous_state(H, A, idx_f-1, idx_l, initial_score) # Previous states

                # Skip type: Single, repeated
                down_penalty = _calculate_finger_skip_penalty(down_prev_actions, skip_penalty_f1, skip_penalty_f2) if use_skip_f else 0.0
                down_action = [f"F{f + initial_f_idx}@F-skip@L{l + initial_l_idx}"]

                down_score = down_prev_score + down_penalty

                ## Right: Skip motif position -----------------------------------------------------------
                right_prev_score, right_prev_actions = _previous_state(H, A, idx_f, idx_l-1, initial_score) # Previous states

                # Skip type: Inside finger, single, repeated
                right_action = [f"F{f + initial_f_idx}@L-skip@L{l + initial_l_idx}"]
                right_penalty = skip_penalty_l1 if use_skip_l else 0.0
                for action in (right_prev_actions if use_skip_l else ()):
                    parts = action.split("@")
                    action_type = parts[1]
                    l_j = int(parts[-1][1:]) - initial_l_idx
                    if action_type == "match":
                        if l - l_j < M:
                            right_penalty = 0 # Inside match: No penalty
                            break
                    elif action_type == "L-skip":
                        if l - l_j > 1:
                            right_penalty = skip_penalty_l2 # Repeated: Extra penalty
                            break

                right_score = right_prev_score + right_penalty

                # Consider all options
                options_score = diag_scores + [
                    down_score,  # Option 2: Keep finger f unused
                    right_score,  # Option 3: Bind finger f previous to l
                ]  # Options 1-k: Bind finger f @ l (from state: [-1, -k])

                # Select best option (first maximum, as np.argmax)
                bestoption_idx = max(range(len(options_score)), key=options_score.__getitem__)
                bestoption_score = options_score[bestoption_idx]
                if alignment_type == "local":
                    bestoption_score = max(bestoption_score, initial_score) # Local alignment: No negative scores

                # Build the arrangement: Only for the best option
                if bestoption_idx < M:
                    k = bestoption_idx + 1
                    _, diag_prev_actions = _previous_state(H, A, idx_f-1, idx_l-k, initial_score)
                    diag_action = _diagonal_actions(
                        f, l, k, diag_overlaps[k-1], initial_f_idx, initial_l_idx, guard_negative=True)
                    bestoption_actions = diag_action + diag_prev_actions
                elif bestoption_idx == M:
                    bestoption_actions = down_action + down_prev_actions
                else:
                    bestoption_actions = right_action + right_prev_actions

                # Record
                H[idx_f, idx_l] = bestoption_score
                A[idx_f][idx_l] = bestoption_actions

                # Local alignment: Record best score, so far
                if bestoption_score > best_score:
                    best_score = bestoption_score
                    best_actions = bestoption_actions

        # Global alignment: Take final cell
        best_score, best_actions = _extract_best_alignment(H, A, alignment_type, F, L_M_1, shift_idx_f, shift_idx_l)

        # Record final optimal
        if best_score >= msw_score:
            if return_matrices:  # Otherwise never read: skip, and let the matrices be freed
                H_final = H # Full matrix: Scores
                A_final = A # Full matrix: Actions
            msw_score = float(best_score) # Final score
            msw_actions = best_actions[::-1] # Reorder: Start to end
            msw_orient = orient # Orientation

    return (msw_score, msw_orient, msw_actions, H_final, A_final)


def calculate_modified_smith_waterman_znf(
    matrix_l, matrix_f,
    sim_for, sim_rev,
    scale_factor, sim_threshold,
    overlap_penalty_l1, overlap_penalty_l2,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    alignment_type="local",
    initial_score=0.0,
    initial_f_idx=0,
    initial_l_idx=0,
    orientations=("for", "rev"),
    return_matrices=False,
):
    '''
    Modified Smith-Waterman, for the 3 bp zinc-finger case only.

    This is the original three-diagonal recurrence, kept frozen so that B1H zinc-finger
    results do not change. calculate_modified_smith_waterman() is the general-M version;
    the two do not agree exactly at M=3, because the general version applies the
    overlap/adjacent/gap rule uniformly to every diagonal.

    Arguments and returns: see calculate_modified_smith_waterman().
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    overlap_penalty_l1 = _parse_overlap_penalty(overlap_penalty_l1, "overlap_penalty_l1")
    overlap_penalty_l2 = _parse_overlap_penalty(overlap_penalty_l2, "overlap_penalty_l2")
    assert overlap_penalty_f1 >= 0
    assert overlap_penalty_f2 >= 0
    assert skip_penalty_f1 <= 0
    assert skip_penalty_f2 <= 0
    assert skip_penalty_l1 <= 0
    assert skip_penalty_l2 <= 0
    assert alignment_type in ["local", "global"]

    F, L_M_1 = sim_for.shape # (F, L-M+1)
    L, D = matrix_l.shape # (L, D)
    M = matrix_f.shape[1] # Finger length
    assert M == 3, f"calculate_modified_smith_waterman_znf() requires a 3 bp finger, got {M}"
    assert L_M_1 == L - M + 1

    # Geometric-mean overlap penalty: Norm of each finger row
    if isinstance(overlap_penalty_l1, str) or isinstance(overlap_penalty_l2, str):
        finger_row_norms = np.linalg.norm(matrix_f, axis=2) # (F, M)
    else:
        finger_row_norms = None

    # Zero penalties: Skip their logic entirely. Overlaps are still detected, because the
    # F-overlap actions (gaps in the motif alignment) depend on them, not on the penalty.
    use_overlap = not (overlap_penalty_l1 == 0 and overlap_penalty_l2 == 0) and not (overlap_penalty_f1 == 0 and overlap_penalty_f2 == 0)
    use_skip_l = not (skip_penalty_l1 == 0 and skip_penalty_l2 == 0)
    use_skip_f = not (skip_penalty_f1 == 0 and skip_penalty_f2 == 0)

    # Shift: Global, local
    if alignment_type == 'local':
        shift_idx_f = 1
        shift_idx_l = 3
    elif alignment_type == 'global':
        shift_idx_f = 0
        shift_idx_l = 0

    # Record final
    H_final = None # Optimal score (float) (F + shift_idx_f, L + shift_idx_l), if requested
    A_final = None # Optimal arrangement (str) (F + shift_idx_f, L + shift_idx_l), if requested

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
            # L: Motif Lengths
            for l in range(L_M_1):
                idx_f = f + shift_idx_f # f index for matrices (shifted by shift_idx_f)
                idx_l = l + shift_idx_l # l index for matrices (shifted by shift_idx_l)

                # Diagonal 1,2,3 -----------------------------------------------------------------------
                sim_score = sim[f, l]
                match_score = scale_factor[l] * sim_score
                diag1_score, diag2_score, diag3_score = -np.inf, -np.inf, -np.inf
                diag_overlaps = [0, 0, 0]

                if sim_score > sim_threshold:
                    ## Diagonal 1: [-1, -1] ----------
                    diag1_prev_score, diag1_prev_actions = _previous_state(H, A, idx_f-1, idx_l-1, initial_score) # Previous states

                    # F: Overlap: 2,1 bp; L: Skip penalty
                    diag1_penalty = match_score
                    for action in diag1_prev_actions:
                        parts = action.split("@")
                        l_j = int(parts[-1][1:]) - initial_l_idx
                        if parts[1] == "match":
                            # Overlap: 2,1 bp (from previous finger)
                            if l - l_j < 3:
                                if use_overlap:
                                    overlap_penalties = _calculate_overlap_penalties(
                                        f, l, l_j, diag1_prev_actions, 3,
                                        overlap_penalty_l1, overlap_penalty_l2,
                                        overlap_penalty_f1, overlap_penalty_f2,
                                        finger_row_norms, scale_factor,
                                        initial_f_idx, initial_l_idx,
                                    )
                                    num_overlap = len(overlap_penalties)
                                else:
                                    overlap_penalties = ()
                                    num_overlap = max(0, _num_overlap(l, l_j, 3))
                                if num_overlap > 0:
                                    for overlap_penalty in overlap_penalties:
                                        diag1_penalty = diag1_penalty + overlap_penalty
                                    diag_overlaps[0] = num_overlap
                                    break
                            elif l - l_j > 2:
                                break

                    diag1_score = diag1_prev_score + diag1_penalty

                    ## Diagonal 2: [-1, -2] ----------
                    diag2_prev_score, diag2_prev_actions = _previous_state(H, A, idx_f-1, idx_l-2, initial_score)

                    # Check for overlap (1bp)
                    diag2_penalty = match_score
                    for action in diag2_prev_actions:
                        parts = action.split("@")
                        l_j = int(parts[-1][1:]) - initial_l_idx
                        # F: Overlap: 1 bp; L: Skip penalty
                        if parts[1] == "match":
                            if l - l_j == 2:
                                if use_overlap:
                                    overlap_penalties = _calculate_overlap_penalties(
                                        f, l, l_j, diag2_prev_actions, 3,
                                        overlap_penalty_l1, overlap_penalty_l2,
                                        overlap_penalty_f1, overlap_penalty_f2,
                                        finger_row_norms, scale_factor,
                                        initial_f_idx, initial_l_idx,
                                    )
                                    num_overlap = len(overlap_penalties)
                                else:
                                    overlap_penalties = ()
                                    num_overlap = max(0, _num_overlap(l, l_j, 3))
                                if num_overlap > 0:
                                    for overlap_penalty in overlap_penalties:
                                        diag2_penalty = diag2_penalty + overlap_penalty
                                    diag_overlaps[1] = num_overlap
                                    break
                            elif l - l_j == 3:
                                continue
                            elif l - l_j == 4:
                                if use_skip_l:
                                    diag2_penalty = diag2_penalty + skip_penalty_l1
                                break
                            elif l - l_j > 4:
                                if use_skip_l:
                                    diag2_penalty = diag2_penalty + skip_penalty_l2
                                break

                    diag2_score = diag2_prev_score + diag2_penalty

                    ## Diagonal 3: [-1, -3] ----------
                    diag3_prev_score, diag3_prev_actions = _previous_state(H, A, idx_f-1, idx_l-3, initial_score)

                    diag3_penalty = match_score
                    for action in (diag3_prev_actions if use_skip_l else ()):  # Gap penalties only
                        parts = action.split("@")
                        l_j = int(parts[-1][1:]) - initial_l_idx
                        # F: No overlap; L: Skip penalty
                        if parts[1] == "match":
                            if l - l_j == 4:
                                diag3_penalty = diag3_penalty + skip_penalty_l1
                                break
                            elif l - l_j > 4:
                                diag3_penalty = diag3_penalty + skip_penalty_l2
                                break

                    diag3_score = diag3_prev_score + diag3_penalty

                ## Down: Skip finger -----------------------------------------------------------
                down_prev_score, down_prev_actions = _previous_state(H, A, idx_f-1, idx_l, initial_score) # Previous states

                # Skip type: Single, repeated
                down_penalty = _calculate_finger_skip_penalty(down_prev_actions, skip_penalty_f1, skip_penalty_f2) if use_skip_f else 0.0
                down_action = [f"F{f + initial_f_idx}@F-skip@L{l + initial_l_idx}"]

                down_score = down_prev_score + down_penalty

                ## Right: Skip motif position -----------------------------------------------------------
                right_prev_score, right_prev_actions = _previous_state(H, A, idx_f, idx_l-1, initial_score) # Previous states

                # Skip type: Inside finger, single, repeated
                right_action = [f"F{f + initial_f_idx}@L-skip@L{l + initial_l_idx}"]
                right_penalty = skip_penalty_l1 if use_skip_l else 0.0
                for action in (right_prev_actions if use_skip_l else ()):
                    parts = action.split("@")
                    action_type = parts[1]
                    l_j = int(parts[-1][1:]) - initial_l_idx
                    if action_type == "match":
                        if l - l_j < 3:
                            right_penalty = 0 # Inside match: No penalty
                            break
                    elif action_type == "L-skip":
                        if l - l_j > 1:
                            right_penalty = skip_penalty_l2 # Repeated: Extra penalty
                            break

                right_score = right_prev_score + right_penalty

                # Consider all options
                options_score = [
                    diag1_score,  # Option 1-1: Bind finger f @ l (from state: [-1, -1])
                    diag2_score,  # Option 1-2: Bind finger f @ l (from state: [-1, -2])
                    diag3_score,  # Option 1-3: Bind finger f @ l (from state: [-1, -3])
                    down_score,  # Option 2: Keep finger f unused
                    right_score,  # Option 3: Bind finger f previous to l
                ]

                # Select best option (first maximum, as np.argmax)
                bestoption_idx = max(range(5), key=options_score.__getitem__)
                bestoption_score = options_score[bestoption_idx]
                if alignment_type == "local":
                    bestoption_score = max(bestoption_score, initial_score) # Local alignment: No negative scores

                # Build the arrangement: Only for the best option
                if bestoption_idx < 3:
                    k = bestoption_idx + 1
                    _, diag_prev_actions = _previous_state(H, A, idx_f-1, idx_l-k, initial_score)
                    diag_action = _diagonal_actions(
                        f, l, k, diag_overlaps[k-1], initial_f_idx, initial_l_idx, guard_negative=False)
                    bestoption_actions = diag_action + diag_prev_actions
                elif bestoption_idx == 3:
                    bestoption_actions = down_action + down_prev_actions
                else:
                    bestoption_actions = right_action + right_prev_actions

                # Record
                H[idx_f, idx_l] = bestoption_score
                A[idx_f][idx_l] = bestoption_actions

                # Local alignment: Record best score, so far
                if bestoption_score > best_score:
                    best_score = bestoption_score
                    best_actions = bestoption_actions

        # Global alignment: Take final cell
        best_score, best_actions = _extract_best_alignment(H, A, alignment_type, F, L_M_1, shift_idx_f, shift_idx_l)

        # Record final optimal
        if best_score >= msw_score:
            if return_matrices:  # Otherwise never read: skip, and let the matrices be freed
                H_final = H # Full matrix: Scores
                A_final = A # Full matrix: Actions
            msw_score = float(best_score) # Final score
            msw_actions = best_actions[::-1] # Reorder: Start to end
            msw_orient = orient # Orientation

    return (msw_score, msw_orient, msw_actions, H_final, A_final)


def get_alignment_modified_smith_waterman(
    matrix_l, matrix_f,
    scale_factor,
    msw_actions,
):
    '''
    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        msw_actions: Optimal alignment arrangement - list of str

    Returns:
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+M-1, D)
        msw_scale_f: Scaling of finger motifs - np.array of shape (F)
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: Motif alignment indices - list of int
    '''
    F, M, _ = matrix_f.shape
    L, D = matrix_l.shape

    ## Traceback
    aligned_motif_fs = np.zeros((F, L + M - 1, D)) # (F, L+M-1, D)
    msw_scale_f = np.zeros(F)
    msw_align_f = [f_i for f_i in range(M*F)]
    msw_align_l = [l_j for l_j in range(L)]

    # First: Unused positions
    action = msw_actions[0]
    action_type = action.split("@")[1]
    f_i = int(action.split("@")[0][1:])
    l_j = int(action.split("@")[-1][1:])
    if f_i > 0:  # Skipped fingers
        align_l_j = msw_align_l.index(l_j)
        msw_align_l[align_l_j : align_l_j] = ['-'] * M * (f_i)
    if l_j > 0:  # Skipped motif positions
        align_f_i = msw_align_f.index(M*f_i)
        msw_align_f[align_f_i : align_f_i] = ['-'] * (l_j)

    # End: Unused positions
    action = msw_actions[-1]
    action_type = action.split("@")[1]
    f_i = int(action.split("@")[0][1:])
    l_j = int(action.split("@")[-1][1:]) + M - 1  # 1 Finger = Matched position + (M-1) bp
    for last_action in reversed(msw_actions):  # The final action may sit before the last match
        if last_action.split("@")[1] == "match":
            l_j = max(l_j, int(last_action.split("@")[-1][1:]) + M - 1)
            break
    if (F - f_i - 1) > 0:  # Skipped fingers
        align_l_j = msw_align_l.index(l_j)
        msw_align_l[align_l_j + 1 : align_l_j + 1] = ['-'] * M * (F - f_i - 1)
    if (L - l_j - 1) > 0:  # Skipped motif positions
        align_f_i = msw_align_f.index(M*(f_i))
        msw_align_f[align_f_i + M : align_f_i + M] = ['-'] * (L - l_j - 1)

    # Middle actions
    last_match = (0, "match", 0)
    for action in msw_actions:
        action_type = action.split("@")[1]
        f_i = int(action.split("@")[0][1:])
        l_j = int(action.split("@")[-1][1:])
        # Match: Place scaled finger
        if action_type == "match":
            scale_ij = scale_factor[l_j]
            aligned_motif_fs[f_i, l_j:l_j+M, :] = matrix_f[f_i] * scale_ij
            msw_scale_f[f_i] = scale_ij
            last_match = (f_i, action_type, l_j)
        # F-overlap: Insert gap in motif
        elif action_type == "F-overlap":
            align_l_j = msw_align_l.index(l_j)
            msw_align_l[align_l_j:align_l_j] = ['-']
        # F-skip: Insert M x gaps in motif
        elif action_type == "F-skip":
            align_l_j = msw_align_l.index(l_j)
            msw_align_l[align_l_j:align_l_j] = ['-'] * M
        # L-skip: Insert gap in finger
        elif action_type == "L-skip":
            if l_j > last_match[-1] + M - 1:
                align_f_i = msw_align_f.index(M*(f_i)) # Only add if not inside match
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
