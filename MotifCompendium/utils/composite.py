import math
from types import SimpleNamespace

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
def run_modified_smith_waterman_composite(
    target_motif, subunit_motif_stack,
    sim_threshold=0.0,
    overlap_penalty_f1=-0.0, overlap_penalty_f2=-0.0,
    skip_penalty_f1=-0.0, skip_penalty_f2=-0.0,
    skip_penalty_l1=-0.0, skip_penalty_l2=-0.0,
    match_penalty_f=-0.0,
    general_penalty_f=True,
    general_penalty_root=2,
    global_exact=False,
    match_count_penalty=-0.0,
    match_count_fraction=1.0,
    return_match_profile=False,
    label_unsigned=True,
    allow_reverse_order=True,
    allow_reverse_comp=True,
    max_homo_subunits=2,
):
    '''
    Aligns a composite of subunit motifs against a motif, for any subunit length M > 1.

    Matched subunits are chained left to right in the order of an expanded stack, which
    therefore decides the arrangements that can be matched. The stack is made of blocks of all
    F subunits, in the order given or reversed (allow_reverse_order), as given or reverse
    complemented (allow_reverse_comp), repeated max_homo_subunits times. With
    allow_reverse_comp, the blocks are also tried with the two orientations swapped (so that,
    e.g., one copy allows both forward-then-reverse and reverse-then-forward).

    A subunit may overhang either end of the target, overlapping it by as little as 1 bp.

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs - np.array of shape (F, M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two subunits - float
        overlap_penalty_f2: Penalty for each additional overlapping bp - float
        skip_penalty_f1: Penalty for skipping 1 subunit (of the expanded stack) - float
        skip_penalty_f2: Penalty for each additional subunit skipped in a row - float
        skip_penalty_l1: Penalty for a gap of 1 motif position between two subunits - float
        skip_penalty_l2: Penalty for each motif position of a longer gap, but one - float
        general_penalty_f: Rescale the summed match score by (number of subunits matched)^(-1/p),
            p = general_penalty_root, so that every added subunit renormalizes all match scores
            (default: True) - bool
        general_penalty_root: p: 2 re-L2-normalizes (1/sqrt), 1 averages; a new subunit raises the
            score only if it exceeds about 1/p of the average match (default: 2) - float > 0
        match_penalty_f: Penalty for every subunit matched, in the score: a match must add more
            than it (match scores are normalized, so that a tiling of the target sums to about 1)
            - float <= 0
        global_exact: Find the best alignment exactly, keeping the best alignment of every state
            (motif positions and subunits skipped since the last match, subunits matched) in each
            DP cell; otherwise keep one per cell, which is faster but, with a penalty or
            general_penalty_f, can miss the best alignment (default: False) - bool
        match_count_penalty: Penalty per subunit matched, for choosing the number of matches: keeps
            the alignment with the N that maximizes its best score with N matches +
            match_count_penalty * N (see calculate_modified_smith_waterman()) - float <= 0
        match_count_fraction: Keep the fewest matches whose penalized best score reaches this
            fraction of the maximum; 1 keeps the maximum (default: 1) - float in (0, 1]
        return_match_profile: Also return msw_profile - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool
        allow_reverse_order: Also match the subunits in reverse order - bool
        allow_reverse_comp: Also match the subunits reverse complemented - bool
        max_homo_subunits: Copies of each block, i.e. matches of each subunit per subunit
            order and orientation; must be a positive integer (default: 2)

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_subunit_stack: Each matched subunit, scaled, in the orientation it matched in,
            left to right. The target spans columns M-1 to L+M-2, leaving M-1 columns past
            either end - np.array of shape (N, 2*M + L - 2, D)
        aligned_subunit_idx: Subunit of each match, as an index into subunit_motif_stack
            - np.array of shape (N,) of int
        aligned_subunit_alignment_rc: Orientation of each match, with respect to its subunit
            (0: forward, 1: reverse complement) - np.array of shape (N,) of int
        aligned_subunit_alignment_h: Column of aligned_subunit_stack where each match starts
            (as oriented) - np.array of shape (N,) of int
        aligned_subunit_scale: Scaling of each match, including 1/sqrt(N) with
            general_penalty_f - np.array of shape (N,)
        msw_profile: Best score of an alignment with exactly N matches, by N (0: empty; -inf:
            impossible) - np.array, only if return_match_profile
    '''
    if (isinstance(max_homo_subunits, (bool, np.bool_))
            or not isinstance(max_homo_subunits, (int, np.integer))
            or max_homo_subunits < 1):
        raise ValueError("max_homo_subunits must be a positive integer")

    # Repeat subunit stacks: Blocks of all F subunits
    # - Reverse motif order (F) x 2 (allow_reverse_order)
    # - Reverse complement motifs (M, D) x 2 (allow_reverse_comp)
    # - Max homo multimers (max_homo_subunits)
    F, M, D = subunit_motif_stack.shape
    L = target_motif.shape[0]
    orders = [np.arange(F)] + ([np.arange(F)[::-1]] if allow_reverse_order else [])
    orients = [0] + ([1] if allow_reverse_comp else [])
    blocks = [(order, rc) for order in orders for rc in orients] * max_homo_subunits
    stack_subunit_idx = np.concatenate([order for order, _ in blocks])  # Subunit of each stack entry
    stack_rc = np.concatenate([np.full(F, rc) for _, rc in blocks])  # Orientation of each stack entry
    expanded_motif_stack = subunit_motif_stack[stack_subunit_idx]
    expanded_motif_stack[stack_rc == 1] = expanded_motif_stack[stack_rc == 1][:, ::-1, ::-1]  # (2 * 2 * max_homo_subunits * F, M, D)

    outputs = _run_modified_smith_waterman(
        target_motif=target_motif,
        subunit_motif_stack=expanded_motif_stack,
        sim_threshold=sim_threshold,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        return_matrices=False,
        label_unsigned=label_unsigned,
        general_penalty_f=general_penalty_f,
        general_penalty_root=general_penalty_root,
        match_penalty_f=match_penalty_f,
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        return_match_profile=return_match_profile,
        place_as_matched=True,
        orientations=("for", "rev") if allow_reverse_comp else ("for",),
        align_remaining=False,
    )
    (msw_score, sim_score, aligned_motif_fs, msw_orient, _, _, msw_scale_f, msw_actions, _, _) = outputs[:10]

    ## Map the matches back to the input subunits
    # A "rev" pass matched every stack entry against reverse-complemented windows, which flips
    # its orientation once more (the same search, with the two orientations swapped)
    matches = [(int(parts[0][1:]), int(parts[-1][1:]))
               for parts in (action.split("@") for action in msw_actions) if parts[1] == "match"]
    stack_idx = np.array([i for i, _ in matches], dtype=int)
    aligned_subunit_idx = stack_subunit_idx[stack_idx]
    aligned_subunit_alignment_rc = stack_rc[stack_idx] ^ int(msw_orient == "rev")
    aligned_subunit_alignment_h = np.array([l for _, l in matches], dtype=int)  # Padded: The target starts at column M-1
    aligned_subunit_scale = msw_scale_f[stack_idx]

    # Aligned subunits: One per match (already placed as matched)
    aligned_subunit_stack = np.zeros((len(matches), 2*M + L - 2, D))
    for n, (i, l) in enumerate(matches):
        h = aligned_subunit_alignment_h[n]
        aligned_subunit_stack[n, h:h + M] = aligned_motif_fs[i, l:l + M]

    return (msw_score, sim_score, aligned_subunit_stack, aligned_subunit_idx,
            aligned_subunit_alignment_rc, aligned_subunit_alignment_h, aligned_subunit_scale) + outputs[10:]


def run_modified_smith_waterman_alignment(
    target_motif, reference_motif,
    sim_threshold=0.0,
    overlap_penalty_f1=-0.02, overlap_penalty_f2=-0.1,
    skip_penalty_f1=-0.1, skip_penalty_f2=-0.1,
    skip_penalty_l1=-0.02, skip_penalty_l2=-0.1,
    match_penalty_f=-0.0,
    general_penalty_f=False,
    general_penalty_root=2,
    global_exact=True,
    match_count_penalty=-0.0,
    match_count_fraction=1.0,
    return_match_profile=False,
    return_matrices=False,
    label_unsigned=True,
    finger_length=3,
):
    '''
    Aligns a chain of fixed-length finger motifs against a CWM, in the order and orientation
    given: unlike run_modified_smith_waterman_composite(), fingers are not reordered, reverse
    complemented individually, or repeated. Each alignment pass matches every finger in one
    orientation (msw_orient).

    Every finger length, including 3 bp B1H zinc fingers, uses the generalized recurrence
    (calculate_modified_smith_waterman).

    A finger may overhang either end of the CWM, overlapping it by as little as 1 bp. Every CWM
    position in the results (actions, alignment tracks, columns of aligned_motif_fs) is padded
    by M-1: CWM position t is position t + M - 1.

    Arguments:
        target_motif: CWM motif matrix (as CWM) - np.array of shape (L, D)
        reference_motif: Finger motif matrix (as IC-scaled PPM) - np.array of shape (F*M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float
        overlap_penalty_f2: Penalty for each additional overlapping bp - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float
        skip_penalty_l1: Penalty for a gap of 1 CWM position between two fingers - float
        skip_penalty_l2: Penalty for each CWM position of a longer gap, but one - float
        general_penalty_f: Rescale the summed match score by (number of fingers matched)^(-1/p),
            p = general_penalty_root, so that every added finger renormalizes all match scores
            (default: False) - bool
        general_penalty_root: p: 2 re-L2-normalizes (1/sqrt), 1 averages; a new finger raises the
            score only if it exceeds about 1/p of the average match (default: 2) - float > 0
        match_penalty_f: Penalty for every finger matched, in the score: a match must add more
            than it (match scores are normalized, so that a tiling of the target sums to about 1)
            - float <= 0
        global_exact: Find the best alignment exactly, keeping the best alignment of every state
            (CWM positions and fingers skipped since the last match, fingers matched) in each DP
            cell; otherwise keep one per cell, which is faster but, with a penalty or
            general_penalty_f, can miss the best alignment (default: True) - bool
        match_count_penalty: Penalty per finger matched, for choosing the number of matches: keeps
            the alignment with the N that maximizes its best score with N matches +
            match_count_penalty * N (see calculate_modified_smith_waterman()) - float <= 0
        match_count_fraction: Keep the fewest matches whose penalized best score reaches this
            fraction of the maximum; 1 keeps the maximum (default: 1) - float in (0, 1]
        return_match_profile: Also return msw_profile - bool
            Only the local alignment is chosen: the remaining alignment still adds the fingers
            left after it (see _run_modified_smith_waterman()).
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool
        finger_length: Length of each finger motif - int (M)

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+2M-2, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int
        msw_align_l: CWM alignment indices (padded) - list of int
        msw_scale_f: Finger scaling projection - np.array of shape (F,)
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Best score of an alignment ending at each DP cell - np.array of shape
            (F+1, L+3M-2), None unless requested
        state_final: H_match and the penalty matrices by DP cell and state - dict, None unless
            requested
        msw_profile: Best score of an alignment with exactly N matches, by N (0: empty; -inf:
            impossible) - np.array, only if return_match_profile
    '''
    subunit_motif_stack = reference_motif.reshape(
        -1, finger_length, reference_motif.shape[-1]
    )  # (F, M, D)
    return _run_modified_smith_waterman(
        target_motif=target_motif,
        subunit_motif_stack=subunit_motif_stack,
        sim_threshold=sim_threshold,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        return_matrices=return_matrices,
        label_unsigned=label_unsigned,
        general_penalty_f=general_penalty_f,
        general_penalty_root=general_penalty_root,
        match_penalty_f=match_penalty_f,
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        return_match_profile=return_match_profile,
    )


def run_modified_smith_waterman_znf(
    target_motif, reference_motif,
    sim_threshold=0.0,
    overlap_penalty_f1=-0.02, overlap_penalty_f2=-0.1,
    skip_penalty_f1=-0.1, skip_penalty_f2=-0.1,
    skip_penalty_l1=-0.02, skip_penalty_l2=-0.1,
    match_penalty_f=-0.0,
    general_penalty_f=False,
    general_penalty_root=2,
    global_exact=True,
    match_count_penalty=-0.0,
    match_count_fraction=1.0,
    return_match_profile=False,
    return_matrices=False,
    label_unsigned=True,
    finger_length=3,
):
    '''
    Exactly run_modified_smith_waterman_alignment(), kept for backward compatibility.

    Deprecated: this will be removed in the future; use run_modified_smith_waterman_alignment().
    It no longer uses the frozen 3 bp recurrence (calculate_modified_smith_waterman_znf).

    Arguments and returns: see run_modified_smith_waterman_alignment().
    '''
    return run_modified_smith_waterman_alignment(
        target_motif=target_motif,
        reference_motif=reference_motif,
        sim_threshold=sim_threshold,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        general_penalty_f=general_penalty_f,
        general_penalty_root=general_penalty_root,
        match_penalty_f=match_penalty_f,
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        return_match_profile=return_match_profile,
        return_matrices=return_matrices,
        label_unsigned=label_unsigned,
        finger_length=finger_length,
    )


def _run_modified_smith_waterman(
    target_motif, subunit_motif_stack,
    sim_threshold,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    return_matrices,
    label_unsigned,
    general_penalty_f=False,
    general_penalty_root=2,
    match_penalty_f=-0.0,
    global_exact=True,
    match_count_penalty=-0.0,
    match_count_fraction=1.0,
    return_match_profile=False,
    place_as_matched=False,
    orientations=("for", "rev"),
    align_remaining=True,
):
    '''
    Aligns a chain of fixed-length finger motifs against a motif, for any finger length M > 1.

    The target is padded with M-1 zeros on either side, so that a finger may overhang either
    end, overlapping the target by as little as 1 bp. Every motif position in the results is in
    these padded coordinates: target position t is position t + M - 1.

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs - np.array of shape (F, M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float
        overlap_penalty_f2: Penalty for each additional overlapping bp - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float
        skip_penalty_l1: Penalty for a gap of 1 motif position - float
        skip_penalty_l2: Penalty for each motif position of a longer gap, but one - float
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Use absolute value of motifs (unsigned) - bool
        general_penalty_f: Rescale the match score by (number of fingers matched)^(-1/p), so that
            every added finger renormalizes all match scores - bool
        general_penalty_root: p, the root of the rescaling (2: 1/sqrt) - float > 0
        match_penalty_f: Penalty for every finger matched - float <= 0
        global_exact: Keep the best alignment of every state in each DP cell (exact), rather
            than one alignment per cell - bool
        match_count_penalty, match_count_fraction: Choose the number of matches of the local
            alignment (see calculate_modified_smith_waterman()) - float <= 0, float in (0, 1]
        return_match_profile: Also return msw_profile - bool
        place_as_matched: Place each finger in the orientation it matched in (reverse
            complemented, when msw_orient is "rev"), rather than as given. The alignment and
            ZNF wrappers place them as given, so that their results do not change - bool
        orientations: Which orientation(s) to evaluate - tuple of "for"/"rev"
        align_remaining: After the local alignment, also align the fingers left after it to the
            rest of the motif (globally), so that every finger is accounted for. Their matches are
            added to the alignment, but not to msw_score. The composite wrapper turns it off: its
            leftover stack entries are unused copies, not fingers to account for - bool

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L+2M-2, D)
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_align_f: Finger alignment indices - list of int (F*M + L+2M-2 - C)
        msw_align_l: Motif alignment indices - list of int (F*M + L+2M-2 - C)
        msw_scale_f: Finger scaling projection - np.array of shape (F,)
        msw_actions: Optimal alignment arrangement - list of str
        H_final: Best score of an alignment ending at each DP cell - np.array of shape
            (F+1, L+3M-2), None unless requested
        state_final: H_match and the penalty matrices by DP cell and state (see
            calculate_modified_smith_waterman()) - dict, None unless requested
        msw_profile: Best score of a local alignment with exactly N matches, by N - np.array,
            only if return_match_profile
    '''
    # Setting: IC-scale off (store)
    if utils_config.get_ic_scale():
        ic_scale = True
        utils_config.set_ic_scale(False)
    else:
        ic_scale = False

    ## Preprocess motifs
    F, M, D = subunit_motif_stack.shape # (F, M, D)

    # Unsigned: Absolute value
    if label_unsigned:
        target_motif = np.abs(target_motif)  # Absolute value (L, D)
        subunit_motif_stack = np.abs(subunit_motif_stack)  # Absolute value (F, M, D)

    # Target motif: Pad M-1 zeros on either side, so that every window overlaps it by 1 bp or more
    target_motif = np.pad(target_motif, ((M - 1, M - 1), (0, 0))) # (L+2M-2, D)

    ## Dimensions: Padded
    L, D = target_motif.shape # (L, D)
    L_M_1 = L - M + 1 # (L-M+1)

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
    (msw_score, msw_orient, msw_actions, msw_state, H_final, state_final, msw_profile) = calculate_modified_smith_waterman(
        matrix_l=target_motif_norm,
        matrix_f=subunit_motif_stack_norm,
        sim_for=sims_ff,
        sim_rev=sims_fr,
        scale_factor=target_motif_norm_unroll_importance,
        sim_threshold=sim_threshold,
        overlap_penalty_f1=overlap_penalty_f1,
        overlap_penalty_f2=overlap_penalty_f2,
        skip_penalty_f1=skip_penalty_f1,
        skip_penalty_f2=skip_penalty_f2,
        skip_penalty_l1=skip_penalty_l1,
        skip_penalty_l2=skip_penalty_l2,
        general_penalty_f=general_penalty_f,
        general_penalty_root=general_penalty_root,
        match_penalty_f=match_penalty_f,
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        return_match_profile=return_match_profile,
        alignment_type="local",
        orientations=orientations,
        return_matrices=return_matrices,
    )

    ## Add remaining global alignment
    remain_f, remain_l = _next_unused_indices(msw_actions, M)  # First unused finger / motif index
    if msw_orient == "rev":
        remain_sim = sims_fr
    else:
        remain_sim = sims_ff

    if align_remaining and len(msw_actions) > 0 and remain_f < F and remain_l < L_M_1:
        (_, _, remain_actions, msw_state, H_remain, state_remain, _) = calculate_modified_smith_waterman(
            matrix_l=target_motif_norm[remain_l:],
            matrix_f=subunit_motif_stack_norm[remain_f:],
            sim_for=remain_sim[remain_f:, remain_l:],
            sim_rev=remain_sim[remain_f:, remain_l:],
            scale_factor=target_motif_norm_unroll_importance[remain_l:],
            sim_threshold=sim_threshold,
            overlap_penalty_f1=overlap_penalty_f1,
            overlap_penalty_f2=overlap_penalty_f2,
            skip_penalty_f1=skip_penalty_f1,
            skip_penalty_f2=skip_penalty_f2,
            skip_penalty_l1=skip_penalty_l1,
            skip_penalty_l2=skip_penalty_l2,
            general_penalty_f=general_penalty_f,
            general_penalty_root=general_penalty_root,
            match_penalty_f=match_penalty_f,
            global_exact=global_exact,
            alignment_type="global",
            initial_state=msw_state,  # Continues the local alignment
            initial_f_idx=remain_f,
            initial_l_idx=remain_l,
            orientations=(msw_orient,),
            return_matrices=return_matrices,
        )

        # Update final alignments
        msw_actions = msw_actions + remain_actions
        if return_matrices:
            H_final[remain_f + 1:, remain_l + M:] = H_remain[1:, M:]  # With local shifts
            state_final["remain"] = state_remain

    ## Retrieve alignments
    if len(msw_actions) == 0:  # No alignment found: Return empty alignment
        aligned_motif_fs = np.zeros((F, L, D))
        msw_scale_f = np.zeros(F)
        msw_align_f, msw_align_l = _alignment_tracks([], F, M, L)
        sim_score = 0.0
    else:
        # "rev": Each finger matched a reverse-complemented window
        if place_as_matched and msw_orient == "rev":
            subunit_motif_stack_placed = subunit_motif_stack_norm[:, ::-1, ::-1]
        else:
            subunit_motif_stack_placed = subunit_motif_stack_norm
        # General finger penalty: Every finger is scaled by (number of fingers matched)^(-1/root)
        if general_penalty_f and msw_state["F_total"] > 0:
            rescale_factor = _match_rescale(msw_state["F_total"], general_penalty_root)
        else:
            rescale_factor = 1.0
        (aligned_motif_fs, msw_scale_f, msw_align_f, msw_align_l) = get_alignment_modified_smith_waterman(
            matrix_l=target_motif_norm,
            matrix_f=subunit_motif_stack_placed,
            scale_factor=target_motif_norm_unroll_importance,
            msw_actions=msw_actions,
            rescale_factor=rescale_factor,
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

    outputs = (msw_score, sim_score, aligned_motif_fs, msw_orient, msw_align_f, msw_align_l, msw_scale_f, msw_actions, H_final, state_final)
    return outputs + (msw_profile,) if return_match_profile else outputs


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
    windows = xp.arange(M)[:, None] + xp.arange(L - M + 1)[None, :]  # (M, L-M+1)
    unrolled_motif = motif[windows].transpose((1, 0, 2))  # (L-M+1, M, K)
    assert(unrolled_motif.shape == (L-M+1, M, K))
    return unrolled_motif


##############
# SIMILARITY #
##############
def compute_aligned_similarity(x, y):
    """Computes the similarity of a (N, L, K) tensor with a (M, L, K) tensor (0 for an all-zero motif)."""
    N, L, K = x.shape
    M = y.shape[0]
    x_norms = cp.linalg.norm(x, axis=(1, 2), keepdims=True)
    y_norms = cp.linalg.norm(y, axis=(1, 2), keepdims=True)
    x_normalized = x/cp.where(x_norms == 0, 1, x_norms)
    y_normalized = y/cp.where(y_norms == 0, 1, y_norms)
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
    Score and arrangement of a previous DP cell (frozen 3 bp recurrence only).

    Out-of-range cells (reachable in global alignment, where the matrices are not padded)
    are unvisited, not wrap-around neighbours.
    '''
    if idx_f < 0 or idx_l < 0:
        return initial_score, []
    return H[idx_f, idx_l], A[idx_f][idx_l]


def _parse_overlap_penalty(penalty, name):
    '''
    An overlap penalty: a number <= 0. ('l2_norm' was removed: general_penalty_f rescales the
    match score by the number of fingers instead.)
    '''
    if isinstance(penalty, str):
        raise ValueError(f"{name} must be a number <= 0, got {penalty!r} "
                         f"('l2_norm' was removed; see general_penalty_f)")
    assert penalty <= 0, f"{name} must be <= 0, got {penalty}"
    return penalty


def _num_overlap(l, l_j, finger_length):
    '''Positions shared by fingers at l and l_j, truncated at the start of the matrix.'''
    return min(finger_length - (l - l_j), l)


def _calculate_overlap_penalties(l, l_j, finger_length, overlap_penalty_f1, overlap_penalty_f2):
    '''
    Penalty for placing a finger at motif position l, when the nearest previously placed
    finger sits at motif position l_j (frozen 3 bp recurrence only).

    The leftmost overlapping position is charged overlap_penalty_f1 and the rest
    overlap_penalty_f2, over the overlap truncated at the start of the matrix.

    Returns:
        penalties: Penalty for each overlapping position, left to right - list of float
    '''
    num_overlap = max(0, _num_overlap(l, l_j, finger_length))
    if num_overlap == 0:
        return []
    return [overlap_penalty_f1] + [overlap_penalty_f2] * (num_overlap - 1)


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
    Penalty for leaving a finger unused (frozen 3 bp recurrence only). A run of g
    consecutively skipped fingers costs skip_penalty_f1 + (g-1) * skip_penalty_f2.

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
    Best score and arrangement of a filled DP matrix (frozen 3 bp recurrence only).

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


#############
# DP STATES #
#############
_STATE_PARTS = ("H_match", "H_penalty_overlap_f", "H_penalty_skip_f", "H_penalty_skip_l", "H_penalty_match_f")
_EMPTY_STATE = {"H_match": 0.0, "H_penalty_overlap_f": 0.0, "H_penalty_skip_f": 0.0,
                "H_penalty_skip_l": 0.0, "H_penalty_match_f": 0.0, "F_total": 0}


def _state_space(
    finger_length,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    match_penalty_f,
    num_f_total, match_weight,
    split=False,
):
    '''
    The states of a DP cell: everything that decides how an alignment can continue from it.

    Keeping only the best alignment of a cell would lose a better one whenever the next
    penalty depends on how the cell was reached, so a cell can keep the best alignment of every
    state (a triple of indices):
        L_skip: Motif positions since the last match: -1 (no match yet), 0 .. M, or M+1 (M+1
            or more). The next finger's overlap and gap penalties depend on it. Past M+1, the
            gap before any later finger is 2 bp or more, so each further position adds exactly
            skip_penalty_l2 to it: that is charged as the position is passed.
        F_skip: Fingers skipped since the last match: 0, or 1 (1 or more). The next finger skip
            costs skip_penalty_f1 or skip_penalty_f2.
        F_total: Matches so far (0 .. num_f_total-1), for the 1/F_total^(1/root) rescaling.
    An axis has a single value when no penalty depends on it, so zero penalties cost nothing.

    Of an alignment, a cell keeps only what its value needs (its parts): H, its score before
    rescaling, or with split (general_penalty_f with penalties), H_match and H_penalty, the sum
    of its penalties, since only H_match is rescaled. (The four penalties of the alignment found
    are recovered from its moves: see _path_state().)

    Both engines use the same states. The exact engine (_row_candidates()) indexes its matrices
    by them; one state per cell (_one_state_row_candidates()) stores the state of its alignment
    with it, as a flat index, and moves it with the tables by state (diag_to, down_to, ...).

    Arguments:
        finger_length: Length of each finger motif - int (M)
        overlap_penalty_f1, overlap_penalty_f2, skip_penalty_f1, skip_penalty_f2,
            skip_penalty_l1, skip_penalty_l2, match_penalty_f: Penalties - float
        num_f_total: Values of F_total (1: not tracked) - int
        match_weight: Weight of H_match in a state's value - float
        split: Keep H_match and H_penalty apart - bool
    Returns:
        space - SimpleNamespace:
            M, shape (of the states), states (flat index of each state), d0 (index of L_skip = 0)
            weight, num_parts, pen (the part that penalties go to), part_names
            overlap_f, skip_l: Overlap and gap penalties of diagonal k, by L_skip index -
                np.array (M+1, nD), and as lists (overlap_list, skip_list)
            right_from, right_to: The right move's source and target L_skip indices
            skip_f1, skip_f2, skip_l2, match_f: Penalties
            By state (flat index), as lists: d_of (L_skip index), count (F_total), diag_to (the
                state after a match; None past the last F_total), down_to and down_pen (after a
                finger skip, and its penalty), right_state and right_pen (after a motif skip, and
                its penalty; None: none); and start_state
    '''
    M = finger_length
    track_l = not (overlap_penalty_f1 == 0 and overlap_penalty_f2 == 0
                   and skip_penalty_l1 == 0 and skip_penalty_l2 == 0)
    track_f = skip_penalty_f1 != skip_penalty_f2
    nD = M + 3 if track_l else 1  # L_skip: -1, 0, ..., M+1
    nF = 2 if track_f else 1
    nN = num_f_total
    shape = (nD, nF, nN)
    states = np.arange(nD * nF * nN).reshape(shape)
    d0 = 1 if track_l else 0

    # Diagonal k, from L_skip: The distance to the previous match is d = L_skip + k
    overlap_f = np.zeros((M + 1, nD))
    skip_l = np.zeros((M + 1, nD))
    if track_l:
        for k in range(1, M + 1):
            for L_skip in range(M + 2):
                d = L_skip + k
                if d < M:  # Overlap: (M - d) bp
                    overlap_f[k, L_skip + 1] = overlap_penalty_f1 + (M - d - 1) * overlap_penalty_f2
                elif d == M + 1:  # Gap: 1 bp
                    skip_l[k, L_skip + 1] = skip_penalty_l1
                elif d > M + 1:  # Gap: (d - M) bp
                    skip_l[k, L_skip + 1] = (d - M - 1) * skip_penalty_l2

    # The moves, by state
    d_of, count, diag_to, down_to, down_pen, right_state, right_pen = [], [], [], [], [], [], []
    for d, f_skip, n in np.ndindex(*shape):
        d_of.append(d)
        count.append(n)
        diag_to.append(int(states[d0, 0, n + 1]) if n + 1 < nN else (int(states[d0, 0, 0]) if nN == 1 else None))
        down_to.append(int(states[d, nF - 1, n]))
        down_pen.append(skip_penalty_f2 if f_skip == 1 else skip_penalty_f1)
        if nD == 1 or d == 0:  # Untracked, or no match yet: Stays
            right_state.append(int(states[d, f_skip, n]))
            right_pen.append(None)
        elif d < M + 2:  # L_skip + 1
            right_state.append(int(states[d + 1, f_skip, n]))
            right_pen.append(None)
        else:  # M+1 or more: The next gap grows by 1 bp
            right_state.append(int(states[d, f_skip, n]))
            right_pen.append(skip_penalty_l2)

    return SimpleNamespace(
        M=M,
        shape=shape,
        states=states,
        d0=d0,
        weight=match_weight,
        num_parts=2 if split else 1,
        pen=1 if split else 0,
        part_names=("H_match", "H_penalty") if split else ("H",),
        overlap_f=overlap_f,
        skip_l=skip_l,
        overlap_list=overlap_f.tolist(),
        skip_list=skip_l.tolist(),
        right_from=np.array([0] + list(range(1, M + 2))),  # L_skip -1 stays -1; D -> D + 1
        right_to=np.array([0] + list(range(2, M + 3))),
        skip_f1=skip_penalty_f1,
        skip_f2=skip_penalty_f2,
        skip_l2=skip_penalty_l2,
        match_f=match_penalty_f,
        d_of=d_of,
        count=count,
        diag_to=diag_to,
        down_to=down_to,
        down_pen=down_pen,
        right_state=right_state,
        right_pen=right_pen,
        start_state=int(states[d0, 0, 1 if nN > 1 else 0]),
    )


def _state_value(parts, weight):
    '''
    Value of states, from their parts (last axis): H * weight, or with split parts,
    H_match * weight + H_penalty.
    '''
    value = parts[..., 0] * weight
    if parts.shape[-1] == 2:
        value = value + parts[..., 1]
    return value


def _match_rescale(num_matches, general_penalty_root):
    '''
    Weight of the match score of an alignment with num_matches matches, with general_penalty_f:
    num_matches^(-1/general_penalty_root) (1/sqrt(num_matches) for the default root, 2).
    '''
    if general_penalty_root == 2:
        return 1 / math.sqrt(num_matches)
    return num_matches ** (-1 / general_penalty_root)


def _initial_parts(space, initial_state):
    '''Parts (see _state_space()) of the alignment that a DP starts from.'''
    H_match = initial_state["H_match"]
    if space.num_parts == 2:
        return [H_match, initial_state["H_penalty_overlap_f"] + initial_state["H_penalty_skip_f"]
                + initial_state["H_penalty_skip_l"] + initial_state["H_penalty_match_f"]]
    return [H_match + initial_state["H_penalty_overlap_f"] + initial_state["H_penalty_skip_f"]
            + initial_state["H_penalty_skip_l"] + initial_state["H_penalty_match_f"]]


def _offer(best, target, candidate, candidate_move, candidate_source, weight):
    '''Takes a candidate for the target states where it is strictly better (earlier candidates win ties).'''
    value, parts, move, source = best
    candidate_value = _state_value(candidate, weight)
    better = candidate_value > value[target]
    value[target] = np.where(better, candidate_value, value[target])
    parts[target] = np.where(better[..., None], candidate, parts[target])
    move[target] = np.where(better, candidate_move, move[target])
    source[target] = np.where(better, candidate_source, source[target])


def _row_candidates(space, parts, i, j0, j1, match_scores, matchable):
    '''
    The diagonal and down candidates (see _cell_candidates()) of DP cells (i, j0 .. j1-1), at
    once: they only depend on row i-1.

    Returns:
        best: value, parts, move and source of the best candidate of each state, by cell -
            list of np.array (j1-j0, *space.shape), parts with a last axis of space.num_parts
    '''
    M, pen = space.M, space.pen
    nD, nF, nN = space.shape
    n = j1 - j0
    parts_best = np.zeros((n,) + space.shape + (space.num_parts,))
    parts_best[..., 0] = -np.inf  # Unreachable
    best = [np.full((n,) + space.shape, -np.inf), parts_best,
            np.full((n,) + space.shape, -1), np.zeros((n,) + space.shape, dtype=int)]
    cells = (slice(None),)

    # Diagonal k: Match, to L_skip = 0, F_skip = 0 and F_total + 1, from the best source state
    if matchable.any():
        n_from, n_to = (slice(0, 1), slice(0, 1)) if nN == 1 else (slice(0, nN - 1), slice(1, nN))
        for k in range(1, M + 1):
            candidate = parts[i - 1, j0 - k:j1 - k].copy()
            candidate[..., 0] += match_scores[:, None, None, None]
            candidate[..., pen] += space.overlap_f[k][None, :, None, None]
            candidate[..., pen] += space.skip_l[k][None, :, None, None]
            candidate[..., pen] += space.match_f
            candidate[~matchable, ..., 0] = -np.inf
            candidate = candidate.reshape(n, nD * nF, nN, space.num_parts)
            first = _state_value(candidate, space.weight).argmax(axis=1)  # By cell and F_total
            picked = np.take_along_axis(candidate, first[:, None, :, None], axis=1)[:, 0]
            picked_source = space.states.reshape(nD * nF, nN)[first, np.arange(nN)]
            _offer(best, cells + (space.d0, 0, n_to), picked[:, n_from], k - 1,
                   picked_source[:, n_from], space.weight)

    # Down: Skip the finger, to F_skip = 1 or more
    below = parts[i - 1, j0:j1]
    if nF == 1:
        candidate = below.copy()
        candidate[..., pen] += space.skip_f1
        _offer(best, cells, candidate, M, space.states, space.weight)
    else:
        for f_skip, skip_penalty in ((0, space.skip_f1), (1, space.skip_f2)):
            candidate = below[:, :, f_skip].copy()
            candidate[..., pen] += skip_penalty
            _offer(best, cells + (slice(None), 1), candidate, M, space.states[:, f_skip], space.weight)
    return best


def _cell_candidates(space, parts, i, j, match_score, matchable, allow_start, best):
    '''
    Best alignment of every state of DP cell (i, j), from its source cells.

    The fill and the traceback both call this (after _row_candidates()): the traceback finds
    each move by recomputing it from the same stored inputs, with the same operations, rather
    than from a stored matrix of moves.

    Candidates, in tie-break order (the first maximum wins, as np.argmax), and their move:
        0 .. M-1: Diagonal k = 1..M: Match, from (i-1, j-k)
        M: Down: Skip the finger, from (i-1, j)
        M+1: Right: Skip a motif position, from (i, j-1)
        M+2: Start: A new alignment, with this match first (local alignment only)
    Within a move, source states are taken in index order.

    Arguments:
        space: DP states - see _state_space()
        parts: DP matrices: The parts of the best alignment of each state (see _state_space())
            - np.array (rows, cols, *space.shape, space.num_parts)
        i, j: DP cell - int
        match_score: scale_factor * similarity at the cell - float
        matchable: Similarity above sim_threshold - bool
        allow_start: Local alignment - bool
        best: The cell's diagonal and down candidates (see _row_candidates()), completed in place
            - list of np.array (space.shape)
    Returns:
        best: value, parts, move (-1: unreachable) and source (the state of the source cell, as
            a flat index) of each state - list of np.array (space.shape)
    '''
    M, pen = space.M, space.pen
    nD, nF, nN = space.shape

    # Right: Skip a motif position, to L_skip + 1 (no match yet: stays -1)
    left = parts[i, j - 1]
    if nD == 1:
        _offer(best, (slice(None),), left, M + 1, space.states, space.weight)
    else:
        _offer(best, (space.right_to,), left[space.right_from], M + 1,
               space.states[space.right_from], space.weight)
        # L_skip M+1 or more: The next gap grows by 1 bp, which costs skip_penalty_l2
        capped = left[M + 2].copy()
        capped[..., pen] += space.skip_l2
        _offer(best, (M + 2,), capped, M + 1, space.states[M + 2], space.weight)

    # Start: A new local alignment, at this match
    if matchable and allow_start:
        n_1 = 1 if nN > 1 else 0
        candidate = np.zeros((1, space.num_parts))
        candidate[0, 0] = match_score
        candidate[0, pen] += space.match_f
        _offer(best, (space.d0, 0, slice(n_1, n_1 + 1)), candidate, M + 2, np.zeros(1, dtype=int),
               space.weight)
    return best


def _fill_modified_smith_waterman(space, match_scores, matchable, initial_state, allow_start, num_cols):
    '''
    Fills the DP matrices of one orientation.

    Row 0 and the first M columns stand for the cells before the matrix: they hold the
    initial state (an empty alignment, or the alignment that a global alignment continues).

    Arguments:
        match_scores: scale_factor * similarity - np.array (F, L-M+1)
        matchable: Similarity above sim_threshold - np.array of bool (F, L-M+1)
    Returns:
        parts: The parts of the best alignment of each cell and state (see _state_space()) -
            np.array (F+1, num_cols, *space.shape, space.num_parts)
    '''
    F, L_M_1 = match_scores.shape
    M = space.M
    parts = np.zeros((F + 1, num_cols) + space.shape + (space.num_parts,))
    parts[..., 0] = -np.inf  # Unreachable
    initial = (slice(None), slice(None), 0, 0, initial_state["F_total"] if space.shape[2] > 1 else 0)
    parts[initial] = _initial_parts(space, initial_state)

    for f in range(F):
        row = _row_candidates(space, parts, f + 1, M, M + L_M_1, match_scores[f], matchable[f])
        for l in range(L_M_1):
            best = _cell_candidates(space, parts, f + 1, l + M, match_scores[f, l], matchable[f, l],
                                    allow_start, [x[l] for x in row])
            parts[f + 1, l + M] = best[1]
    return parts


def _best_cell(score, local, finger_length, F, L_M_1):
    '''
    Best (cell, state) of filled matrices: the first maximum anywhere for a local alignment
    (as np.argmax), else in the last row or column (as _extract_best_alignment()).

    Returns:
        best_score: Its score - float
        best_cell: (i, j, L_skip, F_skip, F_total indices) - tuple of int
    '''
    if local:
        best = int(score.argmax())
        return float(score.flat[best]), tuple(int(x) for x in np.unravel_index(best, score.shape))

    M = finger_length
    last_i, last_j = F, M + L_M_1 - 1
    cells = ([(last_i, j) for j in range(M, last_j + 1)]
             + [(i, last_j) for i in range(1, last_i + 1)])
    candidates = np.stack([score[cell].reshape(-1) for cell in cells])
    best = int(candidates.argmax())
    c, s = divmod(best, candidates.shape[1])
    return (float(candidates.flat[best]),
            cells[c] + tuple(int(x) for x in np.unravel_index(s, score.shape[2:])))


def _traceback_step(option, f, l, i, j, finger_length):
    '''
    A move of the traceback (see _cell_candidates()), from DP cell (i, j) of finger f and motif
    position l back to its source cell.

    Returns:
        step: (action, finger, motif position, diagonal k: 0 for a start or a skip) - tuple
        i, j: Source cell - int
        started: The move starts the alignment - bool
    '''
    M = finger_length
    if option < M:  # Diagonal k: Match
        return ("match", f, l, option + 1), i - 1, j - (option + 1), False
    if option == M:  # Down: Skip finger
        return ("F-skip", f, l, 0), i - 1, j, False
    if option == M + 1:  # Right: Skip motif position
        return ("L-skip", f, l, 0), i, j - 1, False
    return ("match", f, l, 0), i, j, True  # Start: The first match


def _traceback(moves_at, i, j, row, finger_length, F, L_M_1):
    '''
    Moves of the best alignment ending at row (a state, or one alignment) of DP cell (i, j),
    back to the first cell of the matrix it leaves, or to its start. Either engine gives each
    move by recomputing it (moves_at(i, j): the move and source row of every row of the cell).

    Returns:
        steps: (action, finger, motif position, diagonal k: 0 for a start or a skip), in
            alignment order - list of tuple
    '''
    M = finger_length
    steps = []
    while 1 <= i <= F and M <= j < M + L_M_1:  # Visited cells: The rest hold the initial state
        move, source = moves_at(i, j)
        option = int(move[row])
        assert option >= 0, f"Unreachable row {row} at cell {(i, j)}"
        row = int(source[row])
        step, i, j, started = _traceback_step(option, i - 1, j - M, i, j, M)
        steps.append(step)
        if started:
            break
    return steps[::-1]


def _path_state(steps, space, match_scores, initial_state, general_penalty_f, general_penalty_root):
    '''
    H_match, the four penalty sums and F_total of an alignment, and its score, from its moves:
    each move is charged as the DP charges it (see _cell_candidates()), in order, from
    initial_state.

    Returns:
        state: H_match, H_penalty_overlap_f, H_penalty_skip_f, H_penalty_skip_l,
            H_penalty_match_f and F_total - dict
        score: H_match * r + the four penalties, r = F_total^(-1/general_penalty_root) with
            general_penalty_f (else 1) - float
    '''
    M = space.M
    track_l = space.shape[0] > 1
    state = dict(initial_state)
    d, f_skip = 0, 0  # L_skip index (0: no match yet), fingers skipped since the last match
    for action, f, l, k in steps:
        if action == "match":
            if k == 0:  # Start: A new alignment
                state, d, k = dict(_EMPTY_STATE), 0, 1
            state["H_match"] += match_scores[f, l]
            state["H_penalty_overlap_f"] += space.overlap_list[k][d]
            state["H_penalty_skip_l"] += space.skip_list[k][d]
            state["H_penalty_match_f"] += space.match_f
            state["F_total"] += 1
            d, f_skip = space.d0, 0
        elif action == "F-skip":
            state["H_penalty_skip_f"] += space.skip_f2 if f_skip > 0 and space.shape[1] > 1 else space.skip_f1
            f_skip += 1
        elif track_l and d == M + 2:  # L-skip, past M+1: The next gap grows by 1 bp
            state["H_penalty_skip_l"] += space.skip_l2
        elif track_l and d > 0:  # L-skip
            d += 1
    rescale = 1.0
    if general_penalty_f and state["F_total"] > 0:
        rescale = _match_rescale(state["F_total"], general_penalty_root)
    score = (state["H_match"] * rescale + state["H_penalty_overlap_f"] + state["H_penalty_skip_f"]
             + state["H_penalty_skip_l"] + state["H_penalty_match_f"])
    return state, score


def _path_actions(steps, finger_length, initial_f_idx, initial_l_idx):
    '''
    Arrangement of an alignment's moves, in alignment order: each match brings the gaps of its
    overlap with the previous match, and the motif positions its diagonal steps over (see
    _diagonal_actions()).
    '''
    actions = []
    last_l = None  # Motif position of the previous match
    for action, f, l, k in steps:
        if action != "match":
            actions.append(f"F{f + initial_f_idx}@{action}@L{l + initial_l_idx}")
            continue
        num_overlap = 0
        if last_l is not None and l - last_l < finger_length:
            num_overlap = max(0, _num_overlap(l, last_l, finger_length))
        actions.extend(reversed(_diagonal_actions(
            f, l, max(k, 1), num_overlap, initial_f_idx, initial_l_idx, guard_negative=True)))
        last_l = l
    return actions


def _state_matrices(space, parts, score):
    '''
    DP matrices of an exact fill, with only the state axes that a penalty depends on.

    Returns:
        H_final: Best score of an alignment ending at each cell - np.array (rows, cols)
        state_final: The parts (see _state_space()) of the best alignment of each cell and
            state, and the value of each state axis (L_skip M+1 is "M+1 or more"; F_skip 1 is
            "1 or more") - dict
    '''
    names = ("L_skip", "F_skip", "F_total")
    values = (np.arange(-1, space.M + 2), np.arange(2), np.arange(space.shape[2]))
    keep = [n > 1 for n in space.shape]
    shape = score.shape[:2] + tuple(n for n, k in zip(space.shape, keep) if k)
    state_final = {key: parts[..., p].reshape(shape) for p, key in enumerate(space.part_names)}
    state_final["axes"] = {name: value for name, value, k in zip(names, values, keep) if k}
    H_final = score.reshape(score.shape[:2] + (-1,)).max(axis=-1)
    return H_final, state_final


def _trace_all_states(space, parts, score, match_scores, matchable, allow_start, cell, return_matrices):
    '''
    The best alignment ending at a (cell, state) of an exact fill.

    Returns:
        steps: Its moves (see _traceback()) - list of tuple
        matrices: H_final and state_final (see _state_matrices()), None unless return_matrices - tuple
    '''
    F, L_M_1 = match_scores.shape
    M = space.M

    def moves_at(i, j):
        f, l = i - 1, j - M
        row = _row_candidates(space, parts, i, j, j + 1, match_scores[f, l:l + 1], matchable[f, l:l + 1])
        _, _, move, source = _cell_candidates(space, parts, i, j, match_scores[f, l], matchable[f, l],
                                              allow_start, [x[0] for x in row])
        return move.reshape(-1), source.reshape(-1)

    row = int(np.ravel_multi_index(cell[2:], space.shape))
    steps = _traceback(moves_at, cell[0], cell[1], row, M, F, L_M_1)
    return steps, _state_matrices(space, parts, score) if return_matrices else None


def _best_by_count(score):
    '''
    Best (cell, state) of a local alignment for each F_total (the last axis of score): the first
    maximum, as np.argmax.

    Returns:
        best_score: Best score by F_total - np.array (num_n,)
        best_cell: Its (cell, state), by F_total - list of tuple of int
    '''
    num_n = score.shape[-1]
    flat = score.reshape(-1, num_n)
    rows = flat.argmax(axis=0)
    best_cell = [tuple(int(x) for x in np.unravel_index(r * num_n + n, score.shape)) for n, r in enumerate(rows)]
    return flat[rows, np.arange(num_n)], best_cell


def _select_match_count(msw_profile, match_count_penalty, match_count_fraction):
    '''
    Number of matches N to align with, from the best score with exactly N matches (msw_profile):
    the smallest N whose penalized score, msw_profile[N] + match_count_penalty * N, reaches
    match_count_fraction of its maximum. With a fraction of 1, that is the N that maximizes it
    (the fewest matches, on ties). N = 0 (the empty alignment, score 0) is always a candidate.
    '''
    penalized = msw_profile + match_count_penalty * np.arange(len(msw_profile))
    return int(np.flatnonzero(penalized >= match_count_fraction * penalized.max())[0])


def _align_all_states(finger_length, penalties, general_penalty_f, general_penalty_root, match_scores,
                      matchable, initial_state, allow_start, num_cols, return_matrices, profile=False):
    '''
    Best alignment of one orientation, exactly: each DP cell keeps the best alignment of every
    state (see _state_space()).

    With general_penalty_f, the rescaling depends on the final F_total: without penalties, one
    pass keeps the best H_match per F_total, rescaled at the end; with penalties, one pass per
    final F_total weighs every match by it.

    With profile (local alignment only), F_total is always a state, so that the best alignment
    with exactly N matches is known for every N.

    Returns:
        Without profile:
            best_score: Score of the best alignment - float
            best_steps: Its moves (see _traceback()) - list of tuple
            best_matrices: H_final and state_final (see _state_matrices()), None unless
                return_matrices - tuple
        With profile:
            msw_profile: Best score with exactly N matches, by N (-inf: impossible) - np.array
            trace: The steps and matrices of the best alignment with N matches, for N - function
    '''
    M = finger_length
    F, L_M_1 = match_scores.shape
    split = general_penalty_f and any(penalties)

    # Passes: (F_total values, match weight, final weight by F_total, final F_total)
    num_f_total = initial_state["F_total"] + min(F, L_M_1) + 1
    if not general_penalty_f:
        passes = [(num_f_total if profile else 1, 1.0, 1.0, None)]
    elif not split:
        # No penalties: A larger H_match is better for the same F_total, so rescale at the end
        rescale = np.array([1.0] + [_match_rescale(n, general_penalty_root) for n in range(1, num_f_total)])
        passes = [(num_f_total, 1.0, rescale, None)]
    else:
        # Penalties: Which alignment is better depends on the final F_total, so weigh every
        # match by it, one final F_total per pass
        passes = [(n + 1, _match_rescale(n, general_penalty_root), _match_rescale(n, general_penalty_root), n)
                  for n in range(max(initial_state["F_total"], 1), num_f_total)]

    best_score = -np.inf
    msw_profile = np.full(num_f_total, -np.inf)
    found = [None] * num_f_total  # By F_total: traced back ("done") or to trace back ("fill")
    for num_n, match_weight, final_weight, final_f_total in passes:
        space = _state_space(M, *penalties, num_n, match_weight, split=split)
        parts = _fill_modified_smith_waterman(space, match_scores, matchable, initial_state, allow_start, num_cols)

        # Score of the alignment ending at each cell and state
        score = _state_value(parts, final_weight)
        if final_f_total is not None:  # Only alignments of final_f_total fingers were weighted by it
            visited = score[1:F + 1, M:M + L_M_1]
            visited[..., np.arange(num_n) != final_f_total] = -np.inf

        if profile:
            pass_best, pass_cells = _best_by_count(score)
            # One pass per final F_total: Its own, and the empty alignment (row 0, kept unweighted)
            counts = range(num_n) if final_f_total is None else (final_f_total, 0)
            for n in counts:
                if pass_best[n] > msw_profile[n]:
                    msw_profile[n] = pass_best[n]
                    if final_f_total is None:  # All F_total share the fill: Trace back on request
                        found[n] = ("fill", space, parts, score, pass_cells[n])
                    else:
                        found[n] = ("done",) + _trace_all_states(
                            space, parts, score, match_scores, matchable, allow_start, pass_cells[n],
                            return_matrices)
            continue

        pass_score, cell = _best_cell(score, allow_start, M, F, L_M_1)
        if pass_score > best_score:  # Ties: Fewer fingers
            best_score = pass_score
            best_steps, best_matrices = _trace_all_states(
                space, parts, score, match_scores, matchable, allow_start, cell, return_matrices)

    if not profile:
        return best_score, best_steps, best_matrices

    def trace(n):
        if found[n][0] == "done":
            return found[n][1:]
        _, space, parts, score, cell = found[n]
        return _trace_all_states(space, parts, score, match_scores, matchable, allow_start, cell, return_matrices)
    return msw_profile, trace


##########################
# DP: ONE STATE PER CELL #
##########################
def _one_state_row_candidates(space, parts, states, i, j0, j1, match_scores, matchable, rescale, by_count):
    '''
    The diagonal and down candidates of DP cells (i, j0 .. j1-1), as _row_candidates(), when
    each cell keeps one alignment rather than the best of every state: one row or, by_count,
    one row per F_total. A row holds the alignment's parts (as the exact engine's, see
    _state_space()) and, rather than being indexed by its state, its state (a flat index into
    space.shape, which covers only the states a penalty depends on).

    A row's value is H * rescale[F_total], or with split parts, H_match * rescale[F_total] +
    H_penalty: with general_penalty_f, each alignment is rescaled by its own F_total.

    Arguments:
        space: DP states - see _state_space()
        parts, states: Rows of each DP cell: parts (tuple, or None: no alignment) and state -
            list of list of list
        i, j0, j1: DP cells - int
        match_scores, matchable: Of cells j0 .. j1-1 - list of float, list of bool
        rescale: Weight of H_match by F_total - list of float
        by_count: One row per F_total - bool
    Returns:
        best: By cell, and by row: [value, parts, state, move, source row] - list of list of list
    '''
    M = space.M
    split = space.num_parts == 2
    overlap_f, skip_l, match_f = space.overlap_list, space.skip_list, space.match_f
    d_of, count, diag_to, down_to, down_pen = space.d_of, space.count, space.diag_to, space.down_to, space.down_pen
    num_rows = len(parts[i - 1][j0])
    best = []
    for j, match_score, ok in zip(range(j0, j1), match_scores, matchable):
        cell_best = []
        for row in range(num_rows):
            row_best = [-math.inf, None, 0, -1, row]

            # Diagonal k: Match, from the row with one match less (by_count) or the row
            source = row - 1 if by_count else row
            if ok and source >= 0:
                for k in range(1, M + 1):
                    p = parts[i - 1][j - k][source]
                    if p is None:
                        continue
                    s = states[i - 1][j - k][source]
                    s_to = diag_to[s]
                    if s_to is None:
                        continue
                    d = d_of[s]
                    if split:
                        candidate = (p[0] + match_score, p[1] + overlap_f[k][d] + skip_l[k][d] + match_f)
                        value = candidate[0] * rescale[count[s_to]] + candidate[1]
                    else:
                        candidate = (p[0] + match_score + overlap_f[k][d] + skip_l[k][d] + match_f,)
                        value = candidate[0] * rescale[count[s_to]]
                    if value > row_best[0]:  # Strictly: Earlier candidates win ties
                        row_best = [value, candidate, s_to, k - 1, source]

            # Down: Skip the finger
            p = parts[i - 1][j][row]
            if p is not None:
                s = states[i - 1][j][row]
                s_to = down_to[s]
                if split:
                    candidate = (p[0], p[1] + down_pen[s])
                    value = candidate[0] * rescale[count[s_to]] + candidate[1]
                else:
                    candidate = (p[0] + down_pen[s],)
                    value = candidate[0] * rescale[count[s_to]]
                if value > row_best[0]:
                    row_best = [value, candidate, s_to, M, row]
            cell_best.append(row_best)
        best.append(cell_best)
    return best


def _one_state_cell_candidates(space, parts, states, i, j, match_score, matchable, allow_start, rescale,
                               by_count, best):
    '''
    The right and start candidates of DP cell (i, j), as _cell_candidates(), when each cell
    keeps one alignment (see _one_state_row_candidates()).

    Arguments:
        best: The cell's diagonal and down candidates, by row (see _one_state_row_candidates()),
            completed in place - list of list
    Returns:
        best: By row: [value, parts, state, move, source row] - list of list
    '''
    M = space.M
    split = space.num_parts == 2
    count, right_state, right_pen = space.count, space.right_state, space.right_pen

    # Right: Skip a motif position (past M+1: the next gap grows by 1 bp, which costs skip_penalty_l2)
    for row, row_best in enumerate(best):
        p = parts[i][j - 1][row]
        if p is None:
            continue
        s = states[i][j - 1][row]
        s_to, charge = right_state[s], right_pen[s]
        if charge is not None:
            p = (p[0], p[1] + charge) if split else (p[0] + charge,)
        value = p[0] * rescale[count[s_to]] + p[1] if split else p[0] * rescale[count[s_to]]
        if value > row_best[0]:
            row_best[:] = [value, p, s_to, M + 1, row]

    # Start: A new local alignment, at this match
    row = 1 if by_count else 0
    if matchable and allow_start and row < len(best):
        s_to = space.start_state
        if split:
            candidate = (match_score, 0.0 + space.match_f)
            value = candidate[0] * rescale[count[s_to]] + candidate[1]
        else:
            candidate = (match_score + space.match_f,)
            value = candidate[0] * rescale[count[s_to]]
        if value > best[row][0]:
            best[row][:] = [value, candidate, s_to, M + 2, row]
    return best


def _fill_one_state(space, match_scores, matchable, initial_state, allow_start, rescale, num_cols, by_count):
    '''
    Fills the DP matrices of one orientation, keeping one alignment per cell or, by_count, per
    cell and F_total (see _one_state_row_candidates()), as _fill_modified_smith_waterman().

    Returns:
        parts, states: Rows of each DP cell (see _one_state_row_candidates()) - list of list of
            list (F+1, num_cols, rows)
        score: Value of each row - np.array (F+1, num_cols, rows)
    '''
    F, L_M_1 = match_scores.shape
    M = space.M
    num_rows = len(rescale) if by_count else 1
    initial_row = initial_state["F_total"] if by_count else 0
    initial_parts = tuple(_initial_parts(space, initial_state))
    initial_index = int(space.states[0, 0, initial_state["F_total"] if space.shape[2] > 1 else 0])
    initial_value = initial_parts[0] * rescale[space.count[initial_index]]
    if space.num_parts == 2:
        initial_value = initial_value + initial_parts[1]
    empty_parts, empty_states, empty_values = [None] * num_rows, [0] * num_rows, [-math.inf] * num_rows
    empty_parts[initial_row], empty_states[initial_row], empty_values[initial_row] = (
        initial_parts, initial_index, initial_value)
    parts = [[empty_parts] * num_cols for _ in range(F + 1)]
    states = [[empty_states] * num_cols for _ in range(F + 1)]
    score = [[empty_values] * num_cols for _ in range(F + 1)]

    match_scores, matchable = match_scores.tolist(), matchable.tolist()
    for f in range(F):
        row = _one_state_row_candidates(space, parts, states, f + 1, M, M + L_M_1, match_scores[f],
                                        matchable[f], rescale, by_count)
        for l in range(L_M_1):
            best = _one_state_cell_candidates(space, parts, states, f + 1, l + M, match_scores[f][l],
                                              matchable[f][l], allow_start, rescale, by_count, row[l])
            score[f + 1][l + M] = [b[0] for b in best]
            parts[f + 1][l + M] = [b[1] for b in best]
            states[f + 1][l + M] = [b[2] for b in best]
    return parts, states, np.array(score)


def _one_state_matrices(space, parts, states, score):
    '''
    DP matrices of a one-state fill, as _state_matrices().

    Returns:
        H_final: Score of the best alignment kept at each cell - np.array (rows, cols)
        state_final: The parts (see _state_space()) of the alignment kept at each cell (or
            by_count, at each cell and F_total: "axes"), and its state, by the fields a penalty
            depends on (L_skip M+1 is "M+1 or more"; F_skip 1 is "1 or more") - dict
    '''
    num_rows = score.shape[-1]
    unreachable = (-np.inf,) + (0.0,) * (space.num_parts - 1)
    values = np.array([[[p if p is not None else unreachable for p in cell] for cell in row] for row in parts],
                      dtype=float)  # (rows, cols, rows of a cell, parts)
    fields = np.unravel_index(np.array(states), space.shape)  # L_skip index, F_skip, F_total
    if num_rows == 1:
        values, fields = values[:, :, 0], tuple(field[:, :, 0] for field in fields)
    state_final = {name: values[..., p] for p, name in enumerate(space.part_names)}
    for name, field, size, offset in zip(("L_skip", "F_skip", "F_total"), fields, space.shape, (-1, 0, 0)):
        if size > 1:
            state_final[name] = field + offset
    state_final["axes"] = {} if num_rows == 1 else {"F_total": np.arange(num_rows)}
    return score.max(axis=-1), state_final


def _align_one_state(finger_length, penalties, general_penalty_f, general_penalty_root, match_scores,
                     matchable, initial_state, allow_start, num_cols, return_matrices, profile=False):
    '''
    Best alignment of one orientation, keeping one alignment per DP cell (see
    _one_state_row_candidates()). With general_penalty_f, each cell's alignment is rescaled by
    its own F_total.

    Faster than _align_all_states(), and the same without penalties or general_penalty_f. With
    them, the alignment a cell keeps is not always the one that continues best, so the best
    alignment can be missed.

    With profile (local alignment only), a cell keeps one alignment per F_total, so that an
    alignment with exactly N matches is known for every N. When no penalty depends on L_skip or
    F_skip, and the rescaling does not trade off against penalties, that is exact, and done by
    _align_all_states(), which is faster.

    Returns: as _align_all_states(), with matrices from _one_state_matrices()
    '''
    M = finger_length
    F, L_M_1 = match_scores.shape
    split = general_penalty_f and any(penalties)
    if profile and _state_space(M, *penalties, 1, 1.0).shape[:2] == (1, 1) and not split:
        return _align_all_states(M, penalties, general_penalty_f, general_penalty_root, match_scores,
                                 matchable, initial_state, allow_start, num_cols, return_matrices, profile=True)

    num_f_total = initial_state["F_total"] + min(F, L_M_1) + 1
    space = _state_space(M, *penalties, num_f_total if general_penalty_f or profile else 1, 1.0, split=split)
    if general_penalty_f:
        rescale = [1.0] + [_match_rescale(n, general_penalty_root) for n in range(1, num_f_total)]
    else:
        rescale = [1.0] * num_f_total
    parts, states, score = _fill_one_state(space, match_scores, matchable, initial_state, allow_start,
                                           rescale, num_cols, by_count=profile)
    scores_list, matchable_list = match_scores.tolist(), matchable.tolist()

    def moves_at(i, j):
        f, l = i - 1, j - M
        row = _one_state_row_candidates(space, parts, states, i, j, j + 1, [scores_list[f][l]],
                                        [matchable_list[f][l]], rescale, profile)
        best = _one_state_cell_candidates(space, parts, states, i, j, scores_list[f][l], matchable_list[f][l],
                                          allow_start, rescale, profile, row[0])
        return [b[3] for b in best], [b[4] for b in best]

    def trace(i, j, row):
        steps = _traceback(moves_at, i, j, row, M, F, L_M_1)
        return steps, _one_state_matrices(space, parts, states, score) if return_matrices else None

    if profile:
        msw_profile, best_cells = _best_by_count(score)
        return msw_profile, lambda n: trace(*best_cells[n])

    best_score, cell = _best_cell(score[:, :, None, None, :], allow_start, M, F, L_M_1)
    best_steps, best_matrices = trace(cell[0], cell[1], 0)
    return best_score, best_steps, best_matrices


def calculate_modified_smith_waterman(
    matrix_l, matrix_f,
    sim_for, sim_rev,
    scale_factor, sim_threshold,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    general_penalty_f=False,
    general_penalty_root=2,
    match_penalty_f=-0.0,
    global_exact=True,
    match_count_penalty=-0.0,
    match_count_fraction=1.0,
    return_match_profile=False,
    alignment_type="local",
    initial_state=None,
    initial_f_idx=0,
    initial_l_idx=0,
    orientations=("for", "rev"),
    return_matrices=False,
):
    '''
    Modified Smith-Waterman, for any finger length M > 1.

    Each finger covers M motif positions, so a finger placed at motif position l can be
    reached from M diagonals: diagonal k comes from the cell k positions earlier. Given the
    distance d = l - l_j to the previously placed finger:
        d <  M: the fingers overlap over M - d positions, which costs
                overlap_penalty_f1 + (M - d - 1) * overlap_penalty_f2
        d == M: the fingers are adjacent
        d >  M: the fingers are separated by a gap of G = d - M positions, which costs
                skip_penalty_l1 if G == 1, else (G - 1) * skip_penalty_l2
    A run of g fingers skipped between two matches costs skip_penalty_f1 + (g-1) * skip_penalty_f2.
    Every match costs match_penalty_f. Nothing before the first match or after the last one is
    charged.

    The score of an alignment is
        H = H_match * r + H_penalty_overlap_f + H_penalty_skip_f + H_penalty_skip_l + H_penalty_match_f,
    with H_match the sum of scale_factor * similarity over its matches, the four penalty sums,
    and r = F_total^(-1/general_penalty_root) for F_total matches with general_penalty_f (else 1):
    each added finger renormalizes all match scores (the root 2 re-L2-normalizes them; 1 averages
    them). A new match then raises the score only if it exceeds about 1/general_penalty_root of
    the alignment's average match (before penalties), and n equal matches score as n^(1 - 1/root).

    With global_exact, the alignment found is the best one: each cell keeps the best alignment
    of every state that decides how it can continue (see _state_space()), and with
    general_penalty_f, r is that of the final F_total (see _align_all_states()). Without it, each
    cell keeps only its best alignment, with its state, and r is that of the cell's own F_total
    (see _align_one_state()): faster, and the same without penalties or general_penalty_f, but
    with them the best alignment can be missed. Either way, the alignment is traced back by
    recomputing each move (see _cell_candidates()), so no matrix of moves is kept.

    The number of matches can be chosen, rather than left to the best score (local alignment
    only). msw_profile[N] is the best score of an alignment with exactly N matches (0 for the
    empty alignment, -inf where impossible); it is exact with global_exact, and without it, by
    keeping one alignment per cell and F_total. The alignment kept has the smallest N whose
    penalized score, msw_profile[N] + match_count_penalty * N, reaches match_count_fraction of
    its maximum: with the default fraction, 1, the N that maximizes it (the fewest matches, on
    ties). msw_score is its own score, unpenalized. Without global_exact, keeping one alignment
    per cell and F_total can find a better alignment than one per cell, so for a sweep of
    match_count_penalty, set return_match_profile at 0 too. Match scores are normalized (the matches
    of a tiling of the motif sum to about 1; with general_penalty_f, their rescaled sum), so
    match_count_penalty = -0.02 keeps a match only if it raises the score by 0.02.

    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        sim_for: Similarity scores (forward) - np.array of shape (F, L-M+1)
        sim_rev: Similarity scores (reverse) - np.array of shape (F, L-M+1)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float
        overlap_penalty_f2: Penalty for each additional overlapping bp - float
        skip_penalty_f1: Penalty for skipping 1 finger - float
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float
        skip_penalty_l1: Penalty for a gap of 1 motif position - float
        skip_penalty_l2: Penalty for each motif position of a longer gap, but one - float
        general_penalty_f: Rescale the match score by F_total^(-1/general_penalty_root) - bool
        general_penalty_root: Root of the rescaling (2: 1/sqrt(F_total); 1: the average) - float > 0
        match_penalty_f: Penalty for every match - float <= 0
        global_exact: Keep the best alignment of every state in each cell (exact), rather than
            one alignment per cell - bool
        match_count_penalty: Penalty per match, for choosing the number of matches - float <= 0
        match_count_fraction: Choose the fewest matches whose penalized score reaches this
            fraction of the best (1: the best) - float in (0, 1]
        return_match_profile: Compute msw_profile even without a match count penalty or a
            fraction below 1 - bool
        alignment_type: Type of alignment ("local" or "global") - str
        initial_state: State of the alignment that a global alignment continues: H_match, the
            four penalty sums and F_total (as msw_state); None for an empty alignment - dict
        initial_f_idx: Initial finger index - int
        initial_l_idx: Initial motif position index - int
        orientations: Which orientation(s) to evaluate - tuple of "for"/"rev"
        return_matrices: Also return H_final/state_final; None unless requested - bool
    Returns:
        msw_score: Optimal alignment score - float
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_actions: Optimal alignment arrangement - list of str
        msw_state: H_match, the four penalty sums and F_total of the optimal alignment - dict
        H_final: Best score of an alignment ending at each DP cell - np.array (F+1, L+M), None
            unless requested
        state_final: The parts that the DP keeps (H, or H_match and H_penalty; see
            _state_space()) by DP cell and state (see _state_matrices()), or without
            global_exact, by DP cell with the state fields a penalty depends on (see
            _one_state_matrices()), None unless requested - dict
        msw_profile: Best score with exactly N matches, by N - np.array (min(F, L-M+1) + 1,),
            None unless a match count penalty, fraction or return_match_profile
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    overlap_penalty_f1 = _parse_overlap_penalty(overlap_penalty_f1, "overlap_penalty_f1")
    overlap_penalty_f2 = _parse_overlap_penalty(overlap_penalty_f2, "overlap_penalty_f2")
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
    assert match_penalty_f <= 0
    assert general_penalty_root > 0, f"general_penalty_root must be > 0, got {general_penalty_root}"
    local = alignment_type == "local"
    penalties = (overlap_penalty_f1, overlap_penalty_f2, skip_penalty_f1, skip_penalty_f2,
                 skip_penalty_l1, skip_penalty_l2, match_penalty_f)
    initial_state = {key: (initial_state or _EMPTY_STATE)[key] for key in _EMPTY_STATE}
    align = _align_all_states if global_exact else _align_one_state
    assert match_count_penalty <= 0, f"match_count_penalty must be <= 0, got {match_count_penalty}"
    assert match_count_fraction is not None and 0 < match_count_fraction <= 1, \
        f"match_count_fraction must be in (0, 1], got {match_count_fraction}"
    profile = return_match_profile or match_count_penalty != 0 or match_count_fraction < 1
    assert local or not profile, "The number of matches is chosen for local alignment only"

    ## Align: Each orientation
    orient_sims = {"for": sim_for, "rev": sim_rev}
    results = []
    for orient in orientations:
        sim = orient_sims[orient]
        match_scores = scale_factor[None, :] * sim # (F, L-M+1)
        matchable = sim > sim_threshold # (F, L-M+1)
        results.append((orient, match_scores, align(
            M, penalties, general_penalty_f, general_penalty_root, match_scores, matchable, initial_state,
            local, L + M, return_matrices, profile=profile)))

    ## Record final optimal
    msw_profile = None
    if profile:
        # The best alignment with N matches, for every N (ties: the later orientation); then N
        msw_profile = np.full(len(results[0][2][0]), -np.inf)
        owners = [None] * len(msw_profile)
        for orient, orient_scores, (orient_profile, trace) in results:
            for n, score_n in enumerate(orient_profile):
                if score_n > -np.inf and score_n >= msw_profile[n]:
                    msw_profile[n], owners[n] = score_n, (orient, orient_scores, trace)
        n = _select_match_count(msw_profile, match_count_penalty, match_count_fraction)
        msw_orient, best_scores, trace = owners[n]
        best_steps, best_matrices = trace(n)
    else:
        best_score = -np.inf
        for orient, orient_scores, (orient_score, steps, matrices) in results:
            if orient_score >= best_score:
                best_score, msw_orient, best_scores = orient_score, orient, orient_scores # Orientation
                best_steps, best_matrices = steps, matrices

    # The alignment's parts and score, from its moves (the DP keeps only what its choices need)
    rules = _state_space(M, *penalties, 1, 1.0)
    msw_state, msw_score = _path_state(best_steps, rules, best_scores, initial_state, general_penalty_f,
                                       general_penalty_root)
    msw_score = float(msw_score) # Final score
    msw_actions = _path_actions(best_steps, M, initial_f_idx, initial_l_idx) # Start to end
    H_final, state_final = best_matrices if return_matrices else (None, None)

    return (msw_score, msw_orient, msw_actions, msw_state, H_final, state_final, msw_profile)


def calculate_modified_smith_waterman_znf(
    matrix_l, matrix_f,
    sim_for, sim_rev,
    scale_factor, sim_threshold,
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
    Modified Smith-Waterman, for the 3 bp zinc-finger case only: the original three-diagonal
    recurrence, kept frozen only to reproduce old results. Nothing calls it; use
    calculate_modified_smith_waterman(), which is correct for any finger length (3 bp included).

    Bug 1: On diagonals 2 and 3, a finger adjacent to the previous one (no gap) is charged
    skip_penalty_l1 / skip_penalty_l2 for its distance to the finger before that.
    Bug 2: Diagonal 1 also never charges a gap penalty, but has not been seen to
    change a result.
    It keeps one arrangement per cell (A), which calculate_modified_smith_waterman() replaced by
    exact states and a recomputed traceback.

    Arguments and returns: see calculate_modified_smith_waterman().
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    overlap_penalty_f1 = _parse_overlap_penalty(overlap_penalty_f1, "overlap_penalty_f1")
    overlap_penalty_f2 = _parse_overlap_penalty(overlap_penalty_f2, "overlap_penalty_f2")
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

    # Zero penalties: Skip their logic entirely. Overlaps are still detected, because the
    # F-overlap actions (gaps in the motif alignment) depend on them, not on the penalty.
    use_overlap = not (overlap_penalty_f1 == 0 and overlap_penalty_f2 == 0)
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
                                        l, l_j, 3, overlap_penalty_f1, overlap_penalty_f2)
                                    num_overlap = max(0, _num_overlap(l, l_j, 3))
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
                                        l, l_j, 3, overlap_penalty_f1, overlap_penalty_f2)
                                    num_overlap = max(0, _num_overlap(l, l_j, 3))
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
    rescale_factor=1.0,
):
    '''
    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        msw_actions: Optimal alignment arrangement - list of str
        rescale_factor: Scaling of every finger, on top of scale_factor: 1/sqrt(number of
            fingers matched) with general_penalty_f, else 1 - float

    Returns:
        aligned_motif_fs: Aligned finger motifs - np.array of shape (F, L, D)
        msw_scale_f: Scaling of finger motifs - np.array of shape (F)
        msw_align_f: Finger alignment track - list of int and '-'
        msw_align_l: Motif alignment track - list of int and '-'
    '''
    F, M, _ = matrix_f.shape
    L, D = matrix_l.shape

    ## Place scaled fingers
    aligned_motif_fs = np.zeros((F, L, D)) # (F, L, D)
    msw_scale_f = np.zeros(F)
    for action in msw_actions:
        parts = action.split("@")
        if parts[1] == "match":
            f_i = int(parts[0][1:])
            l_j = int(parts[-1][1:])
            scale_ij = scale_factor[l_j] * rescale_factor
            aligned_motif_fs[f_i, l_j:l_j+M, :] = matrix_f[f_i] * scale_ij
            msw_scale_f[f_i] = scale_ij

    ## Alignment tracks
    msw_align_f, msw_align_l = _alignment_tracks(msw_actions, F, M, L)

    return aligned_motif_fs, msw_scale_f, msw_align_f, msw_align_l


def _alignment_tracks(msw_actions, F, M, L):
    '''
    The alignment as two equal-length tracks, one entry per column: msw_align_f holds
    finger columns (column k of finger f is f*M + k) and msw_align_l motif positions,
    with '-' where a track has nothing in that column.

    Built from where each finger was placed, so a matched finger's columns sit over exactly
    the motif positions it covers. Every finger column and every motif position appears
    once, in order. Where fingers overlap, the shared motif position is shown under the
    earlier finger, and the later finger's overlapping columns face '-'. An unmatched finger
    faces '-' where it was skipped, or beside the nearest matched finger if the alignment
    never reached it. Motif positions under no finger face '-'.

    Arguments:
        msw_actions: Optimal alignment arrangement - list of str
        F: Number of fingers - int
        M: Finger length - int
        L: Motif length - int
    Returns:
        msw_align_f: Finger alignment track - list of int and '-'
        msw_align_l: Motif alignment track - list of int and '-'
    '''
    placed = {}   # Finger -> motif position of its match
    skipped = {}  # Finger -> motif position where it was skipped
    for action in msw_actions:
        parts = action.split("@")
        if parts[1] == "match":
            placed[int(parts[0][1:])] = int(parts[-1][1:])
        elif parts[1] == "F-skip":
            skipped.setdefault(int(parts[0][1:]), int(parts[-1][1:]))

    msw_align_f, msw_align_l = [], []
    next_l = 0  # Next motif position to show

    def show_motif_until(l):
        nonlocal next_l
        while next_l < l:  # Motif positions under no finger
            msw_align_f.append('-')
            msw_align_l.append(next_l)
            next_l += 1

    for f in range(F):
        if f in placed:
            l = placed[f]
            show_motif_until(l)
            num_overlap = min(next_l - l, M)  # Already shown under earlier fingers
            for k in range(M):
                msw_align_f.append(f*M + k)
                if k < num_overlap:
                    msw_align_l.append('-')
                else:
                    msw_align_l.append(next_l)  # == l + k
                    next_l += 1
        else:
            if f in skipped:
                show_motif_until(skipped[f])
            elif placed and f < min(placed):  # Before the alignment: Beside the first match
                show_motif_until(placed[min(placed)])
            for k in range(M):
                msw_align_f.append(f*M + k)
                msw_align_l.append('-')
    show_motif_until(L)

    return msw_align_f, msw_align_l

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
