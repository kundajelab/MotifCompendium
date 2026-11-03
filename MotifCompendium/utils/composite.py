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
    overlap_penalty_fraction=0.0,
    match_count_max=2,
    match_count_penalty=0.05,
    match_count_fraction=1.0,
    subunit_motif_trim=True,
    return_match_profile=False,
    label_unsigned=True,
    allow_reverse_comp=True,
    ic_scale=True,
    target_similarity=0.9,
    collapse="sum_norm",
    collapse_p=None,
    overlap_penalty_mode="masked",
):
    '''
    Aligns a composite of subunit motifs of any lengths against a motif: each subunit is trimmed
    to its own length (subunit_motif_trim), and the matches are ordered by position on the motif alone.

    A candidate is a subunit, as given or reverse complemented (allow_reverse_comp), placed at a
    start on the motif; it may overhang either end, overlapping the motif by as little as 1 bp. Its
    window must have a similarity (cosine) above sim_threshold. An alignment is up to
    match_count_max (1 or 2) candidates with strictly increasing starts: any subunits, in any
    orientation, repeated or not.

    Each candidate is scaled by its least-squares weight, b = <motif, candidate>, the cosine of the
    candidate (placed at its start, zero elsewhere) to the whole L2-normalized motif. The scaled
    candidates are collapsed into one motif r, and the alignment's score is the cosine similarity
    of the motif with r, L2-normalized, cos(motif, r) = <motif, r> / |r|. One candidate scores b,
    and candidates that do not overlap score sqrt(sum_k b_k^2), whatever the collapse. Where two
    candidates overlap, collapse sets r there:
        "sum_norm": the sum of the two (the collapsed alignment returned): a subunit placed over
            another adds only what it explains in addition. Exact (see _best_collapsed_pair())
        "lsq_norm": the sum of the two with both scales refit together by non-negative least
            squares (r is then the best fit of the two, already at its least-squares scale), in
            place of each subunit's own b. Exact (see _best_collapsed_pair())
        "max_norm": the larger of the two, entry by entry: a subunit placed over another changes
            r only where it exceeds it, so overlapping costs nothing. Exact (see _best_pooled_pair())
        "lp_norm": the Lp norm of the two, entry by entry, (x_i^p + x_j^p)^(1/p) with p =
            collapse_p: from the sum (p = 1, sum_norm) to the maximum (p = inf, max_norm); two
            equal entries pool to 2^(1/p) times either. Exact (see _best_pooled_pair())
        "mean_norm": their mean. The best pair that overlaps is searched among the 3,000
            placements with the highest b^2 (see _best_mean_pair())
        "none": no collapse: the candidates' squared scores b^2 add, as if they did not overlap
            (only overlap_penalty_fraction charges overlaps). Exact, for any number of candidates
            (see _chain_composite())

    overlap_penalty_fraction charges an overlap of two consecutive candidates, subtracted from the
    squared score (0, the default: none), as overlap_penalty_mode sets:
        "masked" (the default): through the incoming (later-starting) candidate's own match score.
            Its window similarity, over its whole window, still decides whether it is matchable
            (sim_threshold), but its match score b^2 = <motif, b u>, a sum over the positions of
            its window, loses overlap_penalty_fraction * b * <motif, u> summed over the positions
            the previous candidate covers; b and u are not rescaled. With 1, the overlap is masked
            from the incoming candidate's dot product: with collapse="none" it counts once, for the
            earlier candidate, and a candidate inside the previous one adds nothing
        "energy": that fraction of the motif energy of the positions both cover (the overlap
            penalty until 2026-11-03)
    Exact with every collapse (see _chain_composite() and the pair searches).

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs, padded to a common length M - np.array of
            shape (F, M, D)
        sim_threshold: Minimum similarity of a subunit to its window (cosine) to consider a
            match - float
        overlap_penalty_fraction: Fraction of what overlap_penalty_mode charges for an overlap,
            subtracted from the squared score; the collapsed score scores overlaps by itself, so 0
            (the default) adds nothing - float >= 0
        match_count_max: Most subunits matched (default: 2) - 1 or 2; with collapse="none", any
            positive int, or None for no limit
        match_count_penalty: Penalty per subunit matched, in cosine units: keeps the alignment with
            the N that maximizes its best score with N matches - match_count_penalty * N, and
            subtracts match_count_penalty * N from msw_score; a second subunit is kept only if it
            raises the cosine by more than this (default: 0.05) - float >= 0
        match_count_fraction: Keep the fewest matches whose penalized best score reaches this
            fraction of the maximum; 1 keeps the maximum (default: 1) - float in (0, 1]
        target_similarity: Stop adding subunits once the best alignment reaches this cosine: the
            number of matches is the fewer of the penalized choice and the fewest whose best
            score reaches it (None: no stop) (default: 0.9) - float or None
        return_match_profile: Also return msw_profile - bool
        subunit_motif_trim: How to trim each subunit, as logo_trimming in
            assign_label_from_other_compendium(): False keeps all M positions; True trims the
            flanks at the standard threshold of 1/M; a number in [0, 1] is the threshold of
            utils_motif.trim_motif() (0 trims only all-zero positions) (default: True)
            - bool, float or int
        label_unsigned: Compare the absolute values of the motifs (unsigned); if False, compare
            the signed motifs on 8 channels, so that a negative region only matches a negative
            subunit (default: True) - bool
        allow_reverse_comp: Also match the subunits reverse complemented - bool
        ic_scale: IC-scale the target and the subunits (once, before trimming and any
            calculation), so that every similarity, match score and scaling uses the IC-scaled
            motifs; pass motifs that are not IC-scaled already. The global IC-scale setting is
            ignored (default: True) - bool
        collapse: How two overlapping subunits are collapsed: "sum_norm", "lsq_norm", "max_norm",
            "lp_norm", "mean_norm" or "none" (see above) (default: "sum_norm") - str
        collapse_p: p of collapse="lp_norm", at least 1 (1: the same as "sum_norm"; inf: the same
            as "max_norm"); only with "lp_norm" (default: None) - float
        overlap_penalty_mode: What overlap_penalty_fraction charges for an overlap (see above):
            "masked", the incoming candidate's own match score inside it; "energy", the motif
            energy of the overlapping positions (default: "masked") - str

    Returns:
        msw_score: Score of the optimal alignment (cosine), minus match_count_penalty * N - float
        sim_score: MotifCompendium similarity score of the optimal alignment (collapse "none" with
            the masked overlap penalty: with each incoming subunit, times overlap_penalty_fraction,
            taken out where it covers the previous one, as scored) - float
        aligned_subunit_stack: Each matched subunit (IC-scaled, if ic_scale, and trimmed),
            scaled, in the orientation it matched in, left to right. The target spans columns M-1 to L+M-2, leaving M-1
            columns past either end - np.array of shape (N, 2*M + L - 2, D)
        aligned_subunit_idx: Subunit of each match, as an index into subunit_motif_stack
            - np.array of shape (N,) of int
        aligned_subunit_alignment_rc: Orientation of each match (0: forward, 1: reverse
            complement) - np.array of shape (N,) of int
        aligned_subunit_alignment_h: Column of aligned_subunit_stack where each match starts
            (as oriented, trimmed) - np.array of shape (N,) of int
        aligned_subunit_scale: Scaling of each match: its least-squares weight b, times the
            least-squares weight of the collapsed alignment, <motif, r> / |r|^2 (1 without overlaps),
            so that the summed aligned subunits are the best fit of the collapsed alignment to the
            target - np.array of shape (N,)
        msw_profile: Best score (cosine) of an alignment with exactly N matches, by N (0: empty;
            -inf: impossible), before match_count_penalty - np.array, only if return_match_profile
    '''
    ## Check arguments
    F, M, D = subunit_motif_stack.shape
    L = target_motif.shape[0]
    overlap_penalty_fraction = _check_penalty(overlap_penalty_fraction, "overlap_penalty_fraction")
    match_count_penalty = _check_penalty(match_count_penalty, "match_count_penalty")
    assert match_count_fraction is not None and 0 < match_count_fraction <= 1, \
        f"match_count_fraction must be in (0, 1], got {match_count_fraction}"
    assert sim_threshold < 1
    if collapse not in ("sum_norm", "lsq_norm", "max_norm", "lp_norm", "mean_norm", "none"):
        raise ValueError(f"collapse must be 'sum_norm', 'lsq_norm', 'max_norm', 'lp_norm', 'mean_norm' or 'none', got {collapse!r}")
    if collapse == "lp_norm":
        if (isinstance(collapse_p, (bool, np.bool_)) or not isinstance(collapse_p, (int, float, np.integer, np.floating))
                or not collapse_p >= 1):
            raise ValueError(f"collapse_p must be a number >= 1 (or inf) with collapse='lp_norm', got {collapse_p!r}")
    elif collapse_p is not None:
        raise ValueError(f"collapse_p applies only to collapse='lp_norm', got collapse={collapse!r}")
    pool_p = {"sum_norm": 1.0, "max_norm": np.inf}.get(collapse, None if collapse_p is None else float(collapse_p))
    if overlap_penalty_mode not in ("energy", "masked"):
        raise ValueError(f"overlap_penalty_mode must be 'energy' or 'masked', got {overlap_penalty_mode!r}")
    masked = overlap_penalty_mode == "masked" and overlap_penalty_fraction > 0
    if collapse == "none":
        _check_match_count_max(match_count_max)
    elif isinstance(match_count_max, (bool, np.bool_)) or match_count_max not in (1, 2):
        raise ValueError(f"match_count_max must be 1 or 2 with collapse={collapse!r}, got {match_count_max!r}")
    if target_similarity is not None and not (isinstance(target_similarity, (int, float, np.integer, np.floating))
                                              and not isinstance(target_similarity, (bool, np.bool_))):
        raise ValueError(f"target_similarity must be a number or None, got {target_similarity!r}")

    # Setting: IC-scale off (store), so that compute_similarities() does not IC-scale again
    ic_scale_config = utils_config.get_ic_scale()
    if ic_scale_config:
        utils_config.set_ic_scale(False)

    # IC-scale: Only once, here
    if ic_scale:
        target_motif = utils_motif.ic_scale(target_motif)
        subunit_motif_stack = utils_motif.ic_scale(subunit_motif_stack)

    # Unsigned: Absolute value
    if label_unsigned:
        target_motif = np.abs(target_motif)
        subunit_motif_stack = np.abs(subunit_motif_stack)

    ## Candidates: Each subunit, trimmed, as given and reverse complemented
    subunits = _trim_subunits(subunit_motif_stack, subunit_motif_trim)
    orients = (0, 1) if allow_reverse_comp else (0,)
    cand_subunit = np.array([f for f in range(F) for _ in orients], dtype=int)
    cand_rc = np.array([rc for _ in range(F) for rc in orients], dtype=int)
    cand_motif = []
    for f, rc in zip(cand_subunit, cand_rc):
        motif = subunits[f][::-1, ::-1] if rc else subunits[f]
        norm = np.linalg.norm(motif)
        cand_motif.append(motif / norm if norm > 0 else motif)  # L2 normalize (all-zero subunits stay zero)
    cand_len = np.array([len(motif) for motif in cand_motif], dtype=int)

    ## Target: Pad M-1 zeros on either side, so that a subunit may overhang either end
    target_padded = np.pad(target_motif, ((M - 1, M - 1), (0, 0)))  # (L+2M-2, D)
    target_norm = target_padded / np.linalg.norm(target_padded)
    num_starts = len(target_norm)

    ## Similarities and projections, by candidate and start (padded positions)
    sim = np.zeros((len(cand_motif), num_starts))
    energy = np.zeros((len(cand_motif), num_starts))
    valid = np.zeros((len(cand_motif), num_starts), dtype=bool)
    # masked: prefix[c, l, j], the summed <motif, candidate> of the first j positions of candidate c at l (sums to b)
    prefix = np.zeros((len(cand_motif), num_starts, int(cand_len.max()) + 1)) if masked else None
    for m in np.unique(cand_len):
        cands = np.flatnonzero(cand_len == m)
        windows = _unroll_motif(target_norm, m, np)  # (L+2M-1-m, m, D)
        stack_m = np.stack([cand_motif[c] for c in cands])
        if not label_unsigned:  # 8 channels: Each base's positive and negative part apart
            stack_m, windows_sim = utils_motif.motif_4_to_8(stack_m), utils_motif.motif_4_to_8(windows)
        else:
            windows_sim = windows
        sim[cands, :len(windows)] = compute_aligned_similarity(cp.asarray(stack_m), cp.asarray(windows_sim))
        energy[cands, :len(windows)] = (windows * windows).sum(axis=(1, 2))[None, :]
        valid[cands, :len(windows)] = True
        if masked:  # Per position, in the similarity space: (cands, windows, m)
            per_position = np.einsum("cxd,lxd->clx", stack_m, windows_sim)
            prefix[cands[:, None], np.arange(len(windows))[None, :], 1:m + 1] = np.cumsum(per_position, axis=2)
            prefix[cands[:, None], np.arange(len(windows))[None, :], m + 1:] = prefix[cands, :len(windows), m:m + 1]
    matchable = valid & (sim > sim_threshold)
    pairs_c, pairs_l = np.nonzero(matchable)  # (K,), (K,): The matchable placements
    proj = np.sqrt(energy[pairs_c, pairs_l]) * np.maximum(sim[pairs_c, pairs_l], 0.0)  # (K,): b = |w| * cosine
    ends = pairs_l + cand_len[pairs_c]  # (K,)

    ## Align: The best alignment with each number of matches, as squared scores (explained energy)
    position_energy = (target_norm * target_norm).sum(axis=1)  # (L+2M-2,), sums to 1
    if collapse == "none":
        # masked: The cost of an overlap of j positions is p * b * prefix[..., j], b the candidate's projection
        costs = (overlap_penalty_fraction * (np.sqrt(energy) * np.maximum(sim, 0.0))[:, :, None] * prefix
                 if masked else None)  # (C, S, max(cand_len)+1)
        explained, trace = _chain_composite(energy * sim ** 2, matchable, cand_len, overlap_penalty_fraction,
                                            position_energy, match_count_max, overlap_costs=costs)
        index = {(int(c), int(l)): k for k, (c, l) in enumerate(zip(pairs_c, pairs_l))}
        best = {N: ([index[m] for m in trace(N)], 1.0) for N in range(len(explained)) if np.isfinite(explained[N])}
    else:
        explained = np.full(match_count_max + 1, -np.inf)
        explained[0] = 0.0
        best = {0: ([], 1.0)}  # N: (placements, least-squares weight of the collapsed alignment)
        if len(proj):
            k1 = int(np.argmax(proj))
            explained[1], best[1] = proj[k1] ** 2, ([k1], 1.0)
        if len(proj) and match_count_max >= 2:
            sim_space = (lambda x: x) if label_unsigned else utils_motif.motif_4_to_8
            unit = [sim_space(c) for c in cand_motif]  # Candidates in the similarity space (unit norm)
            # masked: cost[k, j], the cost of an overlap of the first j positions of placement k, p * b_k * prefix
            cost = (overlap_penalty_fraction * proj[:, None] * prefix[pairs_c, pairs_l]) if masked else None  # (K, max(cand_len)+1)
            if pool_p == 1 or collapse == "lsq_norm":  # sum_norm, lp_norm with p = 1, or lsq_norm
                value, pair, scale = _best_collapsed_pair(proj, pairs_l, ends, pairs_c, unit, position_energy,
                                                          overlap_penalty_fraction, score="lsq_norm" if collapse == "lsq_norm" else "sum_norm",
                                                          overlap_cost_prefix=cost)
            elif collapse == "mean_norm":
                value, pair, scale = _best_mean_pair(proj, pairs_l, ends, pairs_c, unit, sim_space(target_norm),
                                                     position_energy, overlap_penalty_fraction, overlap_cost_prefix=cost)
            else:
                value, pair, scale = _best_pooled_pair(proj, pairs_l, ends, pairs_c, unit, sim_space(target_norm),
                                                       position_energy, overlap_penalty_fraction, pool_p, overlap_cost_prefix=cost)
            if pair is not None:
                explained[2], best[2] = value, (pair, scale)
    msw_profile = np.where(np.isfinite(explained), np.sqrt(np.maximum(explained, 0.0)), -np.inf)  # Cosines

    ## Number of matches: The penalized best, then the fewest reaching target_similarity
    n = _select_match_count(msw_profile, match_count_penalty, match_count_fraction)
    if target_similarity is not None:
        reached = [N for N in range(1, n) if msw_profile[N] >= target_similarity]
        n = reached[0] if reached else n
    matches, collapsed_scale = best[n]
    msw_score = float(msw_profile[n] - match_count_penalty * n)

    ## Aligned subunits: One per match, scaled by its projection and the collapsed alignment's weight
    aligned_subunit_stack = np.zeros((n, num_starts, D))
    aligned_subunit_scale = np.zeros(n)
    for i, k in enumerate(matches):
        c, l = pairs_c[k], pairs_l[k]
        aligned_subunit_scale[i] = proj[k] * (collapsed_scale[i] if np.ndim(collapsed_scale) else collapsed_scale)
        aligned_subunit_stack[i, l:l + cand_len[c]] = cand_motif[c] * aligned_subunit_scale[i]
    aligned_subunit_idx = np.array([cand_subunit[pairs_c[k]] for k in matches], dtype=int)
    aligned_subunit_alignment_rc = np.array([cand_rc[pairs_c[k]] for k in matches], dtype=int)
    aligned_subunit_alignment_h = np.array([pairs_l[k] for k in matches], dtype=int)

    ## Similarity score: The collapsed alignment against the target (mean_norm: the mean where two overlap;
    ## max_norm, lp_norm: their pool, in the similarity space)
    if n == 0:
        sim_score = 0.0
    else:
        collapsed = aligned_subunit_stack.sum(axis=0)
        if masked and collapse == "none":  # The incoming subunit loses, by the fraction, what it covers of the previous one
            for i in range(1, n):
                lo, hi = pairs_l[matches[i]], min(ends[matches[i - 1]], ends[matches[i]])
                if hi > lo:
                    collapsed[lo:hi] -= overlap_penalty_fraction * aligned_subunit_stack[i, lo:hi]
        if collapse == "mean_norm" and n == 2:
            k_i, k_j = matches
            lo, hi = max(pairs_l[k_i], pairs_l[k_j]), min(ends[k_i], ends[k_j])
            collapsed[lo:hi] /= 2
        elif pool_p is not None and pool_p > 1 and n == 2:
            if label_unsigned:
                collapsed = _pool(aligned_subunit_stack[0], aligned_subunit_stack[1], pool_p, np)
            else:  # 8 channels: Each base's positive and negative part pooled apart, then signed again
                eight = utils_motif.motif_4_to_8(aligned_subunit_stack)
                pooled = _pool(eight[0], eight[1], pool_p, np)  # (S, 8)
                collapsed = pooled @ utils_motif._MOTIF_4_TO_8_POS.T - pooled @ utils_motif._MOTIF_4_TO_8_NEG.T
        sim_score, _, _ = utils_similarity.compute_similarities(
            [collapsed[np.newaxis], target_padded[np.newaxis]], [(0, 1)])[0]

    # Restore IC-scale setting
    if ic_scale_config:
        utils_config.set_ic_scale(True)

    outputs = (msw_score, sim_score, aligned_subunit_stack, aligned_subunit_idx,
               aligned_subunit_alignment_rc, aligned_subunit_alignment_h, aligned_subunit_scale)
    return outputs + (msw_profile,) if return_match_profile else outputs


def _trim_subunits(subunit_motif_stack, subunit_motif_trim):
    '''
    Each subunit, trimmed as logo_trimming (see run_modified_smith_waterman_composite()): False
    keeps it whole, True trims at 1/M, and a number in [0, 1] is the threshold of trim_motif().

    Returns:
        subunits: Trimmed subunits - list of np.array of shape (M_f, D)
    '''
    F, M, _ = subunit_motif_stack.shape
    if isinstance(subunit_motif_trim, (bool, np.bool_)):
        threshold = 1 / M if subunit_motif_trim else None
    elif isinstance(subunit_motif_trim, (int, float, np.integer, np.floating)) and 0 <= subunit_motif_trim <= 1:
        threshold = float(subunit_motif_trim)
    else:
        raise ValueError(f"subunit_motif_trim must be a bool or a number in [0, 1], got {subunit_motif_trim!r}")
    if threshold is None:
        return [subunit_motif_stack[f] for f in range(F)]
    subunits = []
    for f in range(F):
        subunit = utils_motif.trim_motif(subunit_motif_stack[f], threshold)
        if subunit is None:
            raise ValueError(f"Subunit {f} is trimmed away entirely (subunit_motif_trim={subunit_motif_trim!r})")
        subunits.append(subunit)
    return subunits


def _check_match_count_max(match_count_max):
    '''match_count_max: None, or a positive integer.'''
    if match_count_max is not None and (isinstance(match_count_max, (bool, np.bool_))
                                        or not isinstance(match_count_max, (int, np.integer))
                                        or match_count_max < 1):
        raise ValueError(f"match_count_max must be a positive integer or None, got {match_count_max!r}")


def _first_running_max(values):
    '''
    Running maximum of values, and the index of its first occurrence (np.argmax of each prefix).
    '''
    running = np.maximum.accumulate(values)
    new = np.empty(len(values), dtype=bool)
    new[0] = True
    new[1:] = values[1:] > running[:-1]
    return running, np.maximum.accumulate(np.where(new, np.arange(len(values)), 0))


def _chain_composite(match_scores, matchable, cand_len, overlap_penalty_fraction, overlap_weight, max_count,
                     overlap_costs=None):
    '''
    Best chain of candidates (variable-length subunits at starts) with exactly N matches, for
    every N, exactly: starts strictly increase, and an overlap with the previous candidate costs
    overlap_penalty_fraction times the summed overlap_weight of the overlapping positions (the
    motif energy, for run_modified_smith_waterman_composite(); 1 per position counts the bp). The
    score of a chain is the sum of its match scores, minus its overlap costs.

    With W[x] the summed weight of positions before x, an overlap of a candidate at [l, t) with
    a previous one ending at e costs p * (W[min(e, t)] - W[l]). Candidates are taken by start l.
    best[e, n] is the best alignment so far with n matches whose last candidate ends at e (starts
    before l). A candidate c at l, ending at t = l + M_c, follows the best alignment with n-1
    matches that ends:
        at e <= l: no overlap - a running maximum of best over e
        at l < e <= t: a cost of p * (W[e] - W[l]) - a running maximum of best[e] - p*W[e],
            plus p*W[l]
        at e > t: the candidate is covered, a cost of p * (W[t] - W[l]) - a running maximum of
            best, from the right
    and is added to best once every candidate at l is scored.

    overlap_costs, if given, replaces p * (W[min(e, t)] - W[l]) by a cost of the candidate's own:
    overlap_costs[c, l, j] for an overlap of its first j = min(e, t) - l positions (the masked
    overlap penalty). Each candidate then takes the best of best[e] - overlap_costs[c, l, e - l]
    over l < e <= t itself (at most M_c ends), and overlap_costs[c, l, M_c] when covered.

    Arguments:
        match_scores: Match score of each candidate at each start - np.array (C, S)
        matchable: Similarity above sim_threshold - np.array of bool (C, S)
        cand_len: Length of each candidate - np.array (C,) of int
        overlap_penalty_fraction: Penalty on the weight of the overlapping positions - float >= 0
        overlap_weight: Weight of each start position (the motif energy) - np.array (S,)
        max_count: Most matches (None: no limit) - int or None
        overlap_costs: Cost of an overlap of the first j positions of each candidate at each
            start, in place of the weight's (None: the weight's) - np.array (C, S, >= max(cand_len)+1)
    Returns:
        msw_profile: Best score with exactly N matches, by N (0: the empty alignment; -inf:
            impossible) - np.array
        trace: The (candidate, start) of the best alignment with N matches, left to right, for
            N - function
    '''
    C, S = match_scores.shape
    pairs_c, pairs_l = np.nonzero(matchable.T)[::-1]  # (K,), (K,): Matchable (candidate, start), by start
    starts = np.unique(pairs_l)  # (<= S,): Distinct starts with a pair
    num_n = len(starts) if max_count is None else min(len(starts), max_count)  # Starts strictly increase
    ends = pairs_l + cand_len[pairs_c]  # (K,)
    penalty = overlap_penalty_fraction

    ## Fill: The best value of each pair by n, and the pair before it
    K = len(pairs_c)
    value = np.full((K, num_n + 1), -np.inf)  # (K, num_n+1)
    previous = np.full((K, num_n + 1), -1)  # (K, num_n+1): Pair before, in the best alignment with n matches
    best = np.full((S + 1 + int(cand_len.max()), num_n + 1), -np.inf)  # (P, num_n+1), P = S+1+max(cand_len): By end and n
    best_pair = np.full(best.shape, -1)  # (P, num_n+1)
    weight = np.zeros(best.shape[0])  # (P,)
    weight[:min(S, len(overlap_weight))] = overlap_weight[:S]
    W = np.concatenate([[0.0], np.cumsum(weight)])  # (P+1,): W[x]: Summed weight of positions before x
    order = np.argsort(pairs_l, kind="stable")  # (K,)
    bounds = np.searchsorted(pairs_l[order], starts)  # (<= S,)
    for s_i, l in enumerate(starts):
        ks = order[bounds[s_i]:bounds[s_i + 1] if s_i + 1 < len(starts) else len(order)]  # (K_l,): The pairs at l
        t = ends[ks]  # (K_l,)
        scores = match_scores[pairs_c[ks], l]  # (K_l,)
        value[ks, 1] = scores
        for n in range(2, num_n + 1):
            prior = best[:, n - 1]  # (P,)
            # e <= l: No overlap
            pre_max, pre_arg = _first_running_max(prior)  # (P,), (P,)
            options = [np.full(len(ks), pre_max[l]), np.full(len(ks), pre_arg[l])]  # [(K_l,), (K_l,)]
            rev_max, rev_arg = _first_running_max(prior[::-1])  # (P,), (P,): rev index j: e = len-1-j
            j = len(prior) - 1 - (t + 1)  # (K_l,)
            out_e = len(prior) - 1 - rev_arg[np.maximum(j, 0)]  # (K_l,)
            if overlap_costs is None:
                # l < e <= t: Overlap of e - l bp
                inner = prior[l + 1:] - penalty * W[l + 1:len(prior)]  # (P-l-1,)
                in_max, in_arg = _first_running_max(inner)  # (P-l-1,), (P-l-1,)
                in_val, in_e = in_max[t - l - 1] + penalty * W[l], in_arg[t - l - 1] + l + 1  # (K_l,), (K_l,)
                # e > t: The candidate is covered (overlap of M_c bp)
                out_val = np.where(j >= 0, rev_max[np.maximum(j, 0)] - penalty * (W[t] - W[l]), -np.inf)  # (K_l,)
            else:
                # l < e <= t: Overlap of j_e = e - l bp, at the candidate's own cost (first maximum: less overlap)
                lens = t - l  # (K_l,)
                J = int(lens.max())
                cost = overlap_costs[pairs_c[ks], l]  # (K_l, >= J+1)
                inner = np.where(np.arange(1, J + 1)[None, :] <= lens[:, None],
                                 prior[l + 1:l + 1 + J][None, :] - cost[:, 1:J + 1], -np.inf)  # (K_l, J)
                in_arg = inner.argmax(axis=1)  # (K_l,)
                in_val, in_e = inner[np.arange(len(ks)), in_arg], in_arg + l + 1  # (K_l,), (K_l,)
                # e > t: The candidate is covered (overlap of M_c bp)
                out_val = np.where(j >= 0, rev_max[np.maximum(j, 0)] - cost[np.arange(len(ks)), lens], -np.inf)  # (K_l,)
            # Ties: Less overlap first
            vals = np.stack([options[0], in_val, out_val])  # (3, K_l)
            es = np.stack([options[1], in_e, out_e])  # (3, K_l)
            pick = vals.argmax(axis=0)  # (K_l,)
            prev_val = vals[pick, np.arange(len(ks))]  # (K_l,)
            prev_e = es[pick, np.arange(len(ks))]  # (K_l,)
            ok = np.isfinite(prev_val)  # (K_l,)
            value[ks, n] = np.where(ok, prev_val + scores, -np.inf)
            previous[ks, n] = np.where(ok, best_pair[prev_e, n - 1], -1)
        # Add the candidates at l, by end (ties: the first pair)
        for n in range(1, num_n + 1):
            v = value[ks, n]  # (K_l,)
            for k, e, vk in zip(ks, t, v):
                if vk > best[e, n]:
                    best[e, n], best_pair[e, n] = vk, k

    ## The best chain with n matches, for every n
    msw_profile = np.full(num_n + 1, -np.inf)  # (num_n+1,)
    msw_profile[0] = 0.0
    found = [None] * (num_n + 1)  # [num_n+1]
    for n in range(1, num_n + 1):
        if K == 0 or not np.isfinite(value[:, n]).any():
            continue
        found[n] = int(value[:, n].argmax())
        msw_profile[n] = value[found[n], n]

    def trace(n):
        if n == 0:
            return []
        k = found[n]
        chain = []
        while n > 0:
            chain.append((int(pairs_c[k]), int(pairs_l[k])))
            k, n = previous[k, n], n - 1
        return chain[::-1]
    return msw_profile, trace


def _best_disjoint_pair(m, starts, ends):
    '''
    Best pair of placements that do not overlap (the first ends at or before the second starts), by
    m_i + m_j, exactly: a running maximum of m over ends.

    Returns:
        value: m_i + m_j (-inf: no pair) - float
        pair: The two placements, by start (None: no pair) - list of int
    '''
    K = len(m)
    P = int(ends.max()) + 1
    o = np.lexsort((np.arange(K), -m, ends))  # By end, the best first (ties: the first placement)
    first = np.ones(K, dtype=bool)
    first[1:] = ends[o][1:] != ends[o][:-1]
    by_end, arg_end = np.full(P, -np.inf), np.full(P, -1)
    by_end[ends[o][first]], arg_end[ends[o][first]] = m[o][first], o[first]
    run, run_arg = _first_running_max(by_end)  # (P,), (P,): The best ending at or before each position
    totals = np.where(np.isfinite(run[starts]), run[starts] + m, -np.inf)  # (K,)
    j = int(np.argmax(totals))
    if not np.isfinite(totals[j]):
        return -np.inf, None
    return float(totals[j]), [int(arg_end[run_arg[starts[j]]]), j]


def _best_mean_pair(proj, starts, ends, cands, unit, target, position_energy, overlap_penalty_fraction,
                    top=3000, row_block=512, overlap_cost_prefix=None):
    '''
    Best pair of placements with strictly increasing starts, by the energy their collapsed
    alignment explains once L2-normalized, with the mean of the two where they overlap:

        <t, r>^2 / |r|^2 - p * E(overlap),
        <t, r> = B - (A_i + A_j) / 2,    |r|^2 = B - 3/4 (Q_i + Q_j) + X / 2

    with B = b_i^2 + b_j^2, A_k and Q_k the sums of <t_p, v_kp> and |v_kp|^2 over the overlap
    (v = b u), X = <v_i, v_j>, and E(overlap) the motif energy there. A pair that does not overlap
    scores B: the best exactly. The mean can score an overlapping pair above B (a weak partner
    halves a subunit's columns that fit poorly), so those pairs have no bound to stop on: they are
    searched among the top placements by b^2 (all of them if there are fewer).

    Arguments:
        proj, starts, ends, cands, unit, position_energy, overlap_penalty_fraction,
            overlap_cost_prefix: As _best_collapsed_pair()
        target: The L2-normalized target, in the similarity space - np.array (S, D')
        top: Placements searched for overlapping pairs - int
        row_block: Block size of the search - int
    Returns:
        value: Best squared score (-inf: no pair) - float
        pair: The two placements, by start (None: no pair) - list of int
        scale: Least-squares weight of the pair's collapsed alignment, <t, r> / |r|^2 - float
    '''
    m = proj * proj
    value, pair = _best_disjoint_pair(m, starts, ends)
    scale = 1.0
    I = np.argsort(-m, kind="stable")[:top]
    T = len(I)
    if T < 2:
        return value, pair, scale
    P = int(ends.max()) + 1
    U = np.zeros((T, P, unit[0].shape[1]))  # Each placed candidate
    for r, k in enumerate(I):
        U[r, starts[k]:ends[k]] = unit[cands[k]]
    t = np.zeros(U.shape[1:])
    t[:min(P, len(target))] = target[:P]
    U = cp.asarray(U)
    Ct = cp.concatenate([cp.zeros((T, 1)), cp.cumsum((U * cp.asarray(t)[None]).sum(axis=2), axis=1)], axis=1)  # (T, P+1)
    Cu = cp.concatenate([cp.zeros((T, 1)), cp.cumsum((U * U).sum(axis=2), axis=1)], axis=1)  # (T, P+1)
    U = U.reshape(T, -1)
    W = cp.asarray(np.concatenate([[0.0], np.cumsum(position_energy)]))
    bI, mI, lI, eI = (cp.asarray(x[I]) for x in (proj, m, starts, ends))
    CI = None if overlap_cost_prefix is None else cp.asarray(overlap_cost_prefix[I])  # (T, J+1)
    cols = cp.arange(T)[None, :]
    for s in range(0, T, row_block):
        rows = slice(s, min(T, s + row_block))
        ri = cp.arange(rows.start, rows.stop)[:, None]
        lo = cp.maximum(lI[rows, None], lI[None, :])
        hi = cp.minimum(eI[rows, None], eI[None, :])
        ok = (lo < hi) & (lI[rows, None] < lI[None, :])  # Overlapping, the row first
        lo, hi = cp.where(ok, lo, 0), cp.where(ok, hi, 0)
        Ai = bI[rows, None] * (Ct[ri, hi] - Ct[ri, lo])
        Aj = bI[None, :] * (Ct[cols, hi] - Ct[cols, lo])
        Qi = mI[rows, None] * (Cu[ri, hi] - Cu[ri, lo])
        Qj = mI[None, :] * (Cu[cols, hi] - Cu[cols, lo])
        X = (U[rows] @ U.T) * bI[rows, None] * bI[None, :]
        B = mI[rows, None] + mI[None, :]
        num, den = B - (Ai + Aj) / 2, B - 0.75 * (Qi + Qj) + X / 2
        pen = (overlap_penalty_fraction * (W[hi] - W[lo]) if CI is None
               else _masked_pair_cost(CI[rows], CI, lI[rows, None], lI[None, :], hi))
        val = cp.where(ok & (num > 0) & (den > 0), num * num / cp.maximum(den, 1e-300) - pen, -cp.inf)
        flat = int(cp.argmax(val))
        a, c = divmod(flat, T)
        if float(val[a, c]) > value:
            value = float(val[a, c])
            pair = [int(I[s + a]), int(I[c])]  # The row starts first
            scale = float(num[a, c] / den[a, c])
    return value, pair, scale


def _masked_pair_cost(cost_rows, cost_cols, l_rows, l_cols, hi):
    '''
    Masked overlap cost of each pair of a block: the cost of the later-starting placement at the
    number of its positions the other covers, cost[k, hi - l_k] (0 where they do not overlap).

    Arguments:
        cost_rows, cost_cols: Cost of an overlap of the first j positions, of the block's rows and
            of its columns - cp.array (R, J+1), (C, J+1)
        l_rows, l_cols: Starts - cp.array (R, 1), (1, C)
        hi: End of the overlap, the earlier end - cp.array (R, C)
    Returns:
        cost - cp.array (R, C)
    '''
    J = cost_rows.shape[1] - 1
    from_rows = cp.take_along_axis(cost_rows, cp.clip(hi - l_rows, 0, J), axis=1)  # (R, C)
    from_cols = cp.take_along_axis(cost_cols, cp.clip(hi - l_cols, 0, J).T, axis=1).T  # (R, C)
    return cp.where(l_rows > l_cols, from_rows, from_cols)


def _pool(x_i, x_j, p, xp):
    '''
    Lp pool of two nonnegative arrays, entry by entry: (x_i^p + x_j^p)^(1/p), the maximum for p = inf.
    Where one is 0 it is the other, so arrays that do not overlap pool to their sum.
    '''
    hi, lo = xp.maximum(x_i, x_j), xp.minimum(x_i, x_j)
    if np.isinf(p):
        return hi
    ratio = xp.where(hi > 0, lo / xp.where(hi > 0, hi, 1.0), 0.0)
    return hi * (1.0 + ratio ** p) ** (1.0 / p)


def _best_pooled_pair(proj, starts, ends, cands, unit, target, position_energy, overlap_penalty_fraction, p,
                      row_block=256, col_block=8192, pair_block=4096, overlap_cost_prefix=None):
    '''
    Best pair of placements with strictly increasing starts, by the energy their collapsed
    alignment explains once L2-normalized, with the Lp pool of the two where they overlap,
    r = (v_i^p + v_j^p)^(1/p) entry by entry (v = b u; p = inf: the maximum), exactly:

        <t, r>^2 / |r|^2 - q * E(overlap)    (overlap_cost_prefix: minus the masked cost instead)

    A pair that does not overlap scores B = b_i^2 + b_j^2: the best exactly. An overlapping pair is
    bounded, as t and v are nonnegative and max(v_i, v_j) <= r <= v_i + v_j, r >= 2^(1/p-1) (v_i + v_j):
        on the overlap O, y = <t, r_O> <= min(A, |t_O| z), with z = |r_O| between
            z_lo = sqrt(max(Q_i, Q_j, 4^(1/p-1) (Q_i + Q_j + 2X))) and z_hi = sqrt(Q_i + Q_j + 2X)
        and outside it r = v_i + v_j, so cos(t, r) <= max over z of (B - A + min(A, |t_O| z)) / sqrt(B - Q_i - Q_j + z^2)
    with A, Q_k the sums of <t, v_i + v_j> and |v_k|^2 over O and X = <v_i, v_j>. The maximum over z
    is at an end, at the kink z = A / |t_O|, or at z = |t_O| (B - Q_i - Q_j) / (B - A). With
    b_i >= b_j it is at most B / b_i = b_i + b_j^2 / b_i: placements are taken by decreasing b, as
    the first of a pair while 2 b_i beats the best so far, and paired with those with b_j^2 above
    b_i (best - b_i). Pairs whose bound beats the best so far are scored exactly, by bound first.

    Arguments:
        proj, starts, ends, cands, unit, position_energy: As _best_collapsed_pair()
        target: The L2-normalized target, in the similarity space - np.array (S, D')
        overlap_penalty_fraction: q - float >= 0
        p: Order of the pool, > 1 (inf: the maximum) - float
        row_block, col_block, pair_block: Block sizes of the search - int
        overlap_cost_prefix: As _best_collapsed_pair() (E(overlap) still bounds <t, r> on the overlap)
    Returns:
        value: Best squared score (-inf: no pair) - float
        pair: The two placements, by start (None: no pair) - list of int
        scale: Least-squares weight of the pair's collapsed alignment, <t, r> / |r|^2 - float
    '''
    m = proj * proj  # (K,): Energy each placement explains alone
    value, pair = _best_disjoint_pair(m, starts, ends)
    scale = 1.0
    order = np.argsort(-m, kind="stable")
    b_sorted, m_sorted = proj[order], m[order]
    best = lambda: np.sqrt(value) if np.isfinite(value) else -np.inf  # The best cosine so far

    ## Placements that can be in a pair that beats the best so far
    S = best()
    n_cand = len(order) if not np.isfinite(S) else int(np.searchsorted(-m_sorted, -(b_sorted[0] * (S - b_sorted[0])), side="left"))
    if n_cand < 2:
        return value, pair, scale
    I = order[:n_cand]  # (n_cand,)
    P = int(ends.max()) + 1
    U = np.zeros((n_cand, P, unit[0].shape[1]))  # Each placed candidate
    for r, k in enumerate(I):
        U[r, starts[k]:ends[k]] = unit[cands[k]]
    t = np.zeros(U.shape[1:])
    t[:min(P, len(target))] = target[:P]
    U = cp.asarray(U)
    Ct = cp.concatenate([cp.zeros((n_cand, 1)), cp.cumsum((U * cp.asarray(t)[None]).sum(axis=2), axis=1)], axis=1)
    Cu = cp.concatenate([cp.zeros((n_cand, 1)), cp.cumsum((U * U).sum(axis=2), axis=1)], axis=1)
    U, t = U.reshape(n_cand, -1), cp.asarray(t.reshape(-1))
    W = cp.asarray(np.concatenate([[0.0], np.cumsum(position_energy)]))  # W[x]: Energy of positions before x
    bI, mI, lI, eI = (cp.asarray(x[I]) for x in (proj, m, starts, ends))
    CI = None if overlap_cost_prefix is None else cp.asarray(overlap_cost_prefix[I])  # (n_cand, J+1)
    c_p = 4.0 ** (1.0 / p - 1.0)  # |r_O|^2 >= c_p |v_i + v_j|^2 on O

    def cos_bound(z, base_t, base_q, A, tau):
        return (base_t + cp.minimum(A, tau * z)) / cp.sqrt(cp.maximum(base_q + z * z, 1e-300))

    for s in range(0, n_cand, row_block):
        S = best()
        if np.isfinite(S) and 2 * b_sorted[s] <= S:  # No pair left can beat the best
            break
        rows = slice(s, min(n_cand, s + row_block))
        n_col = n_cand if not np.isfinite(S) else int(np.searchsorted(-m_sorted[:n_cand], -(b_sorted[s] * (S - b_sorted[s])), side="left"))
        if n_col <= s + 1:
            break
        ri = cp.arange(rows.start, rows.stop)[:, None]
        row_ok = (2 * bI[rows] > S)[:, None] if np.isfinite(S) else cp.ones((rows.stop - rows.start, 1), dtype=bool)
        for c0 in range(s + 1, n_col, col_block):
            cols = slice(c0, min(n_col, c0 + col_block))
            ci = cp.arange(cols.start, cols.stop)[None, :]
            lo = cp.maximum(lI[rows, None], lI[None, cols])
            hi = cp.minimum(eI[rows, None], eI[None, cols])
            ok = row_ok & (ci > ri) & (lo < hi) & (lI[rows, None] != lI[None, cols])  # Overlapping, distinct starts
            lo, hi = cp.where(ok, lo, 0), cp.where(ok, hi, 0)
            A = bI[rows, None] * (Ct[ri, hi] - Ct[ri, lo]) + bI[None, cols] * (Ct[ci, hi] - Ct[ci, lo])
            Qi = mI[rows, None] * (Cu[ri, hi] - Cu[ri, lo])
            Qj = mI[None, cols] * (Cu[ci, hi] - Cu[ci, lo])
            X = (U[rows] @ U[cols].T) * bI[rows, None] * bI[None, cols]
            B = mI[rows, None] + mI[None, cols]
            E = W[hi] - W[lo]
            base_t, base_q, tau = B - A, B - Qi - Qj, cp.sqrt(cp.maximum(E, 0.0))
            z_hi = cp.sqrt(cp.maximum(Qi + Qj + 2 * X, 0.0))
            z_lo = cp.minimum(cp.sqrt(cp.maximum(cp.maximum(Qi, Qj), c_p * (Qi + Qj + 2 * X))), z_hi)
            z_star = cp.where(base_t > 0, tau * base_q / cp.where(base_t > 0, base_t, 1.0), z_hi)
            z_kink = cp.where(tau > 0, A / cp.where(tau > 0, tau, 1.0), z_hi)
            bound = cp.maximum(cos_bound(z_lo, base_t, base_q, A, tau), cos_bound(z_hi, base_t, base_q, A, tau))
            for z in (z_star, z_kink):
                bound = cp.maximum(bound, cos_bound(cp.clip(z, z_lo, z_hi), base_t, base_q, A, tau))
            bound = cp.minimum(bound, 1.0)
            pen = (overlap_penalty_fraction * E if CI is None
                   else _masked_pair_cost(CI[rows], CI[cols], lI[rows, None], lI[None, cols], hi))
            bound = cp.where(ok, bound * bound - pen, -cp.inf)
            a, c = cp.nonzero(bound > value)
            if len(a) == 0:
                continue
            by_bound = cp.argsort(-bound[a, c])
            a, c = a[by_bound], c[by_bound]
            bounds = bound[a, c]
            for q0 in range(0, len(a), pair_block):  # Exactly, by bound first
                if float(bounds[q0]) <= value:
                    break
                qa, qc = a[q0:q0 + pair_block], c[q0:q0 + pair_block]
                i, j = rows.start + qa, cols.start + qc
                r = _pool(U[i] * bI[i, None], U[j] * bI[j, None], p, cp)
                y, z2 = r @ t, (r * r).sum(axis=1)
                val = cp.where(y > 0, y * y / cp.maximum(z2, 1e-300), -cp.inf) - pen[qa, qc]
                k = int(cp.argmax(val))
                if float(val[k]) > value:
                    value = float(val[k])
                    pair = sorted([int(I[int(i[k])]), int(I[int(j[k])])], key=lambda x: starts[x])
                    scale = float(y[k] / z2[k])
    return value, pair, scale


def _best_collapsed_pair(proj, starts, ends, cands, unit, position_energy, overlap_penalty_fraction,
                         row_block=512, col_block=16384, score="sum_norm", overlap_cost_prefix=None):
    '''
    Best pair of placements with strictly increasing starts, by the energy their collapsed
    alignment (the sum of the two) explains once L2-normalized (its squared cosine to the motif),
    exactly:

        "sum_norm": B^2 / (B + 2X) - p * E(overlap),    B = b_i^2 + b_j^2,    X = b_i b_j g
        "lsq_norm": with the two scales x_i, x_j refit by non-negative least squares, the energy of
            that fit (its squared cosine, as it is at its least-squares scale): x_i b_i + x_j b_j =
            (B - 2 g b_i b_j) / (1 - g^2) if both x are non-negative, else the larger b^2 alone

    with g = <u_i, u_j> >= 0 and E(overlap) the motif energy of the positions both windows cover
    (overlap_cost_prefix: the masked cost of the later-starting placement instead, also >= 0).
    A pair that does not overlap scores B: the best is a running maximum of b^2 over ends. Overlap
    only lowers either score below B, so an overlapping pair can win only if its B beats the best
    so far: those pairs are searched in decreasing b^2 (on the GPU, in blocks), and the search
    stops once none left can.

    Arguments:
        proj: Projection b of each placement - np.array (K,)
        starts, ends: Window [start, end) of each placement - np.array (K,) of int
        cands: Candidate of each placement - np.array (K,) of int
        unit: Each candidate, unit norm, in the similarity space - list of np.array (M_c, D')
        position_energy: Motif energy of each position - np.array (S,)
        overlap_penalty_fraction: p - float >= 0
        row_block, col_block: Block sizes of the search - int
        score: "sum_norm" or "lsq_norm" - str
        overlap_cost_prefix: The masked overlap penalty, in place of p * E(overlap): the cost of an
            overlap of the first j positions of each placement, charged to the later-starting one
            of a pair (None: p * E(overlap)) - np.array (K, J+1)
    Returns:
        value: Best squared score (-inf: no pair) - float
        pair: The two placements, by start (None: no pair) - list of int
        scale: Scaling of the pair's subunits, times b: "sum_norm", the least-squares weight of
            the collapsed alignment, B / (B + 2X); "lsq_norm", each refit scale over its b, by start
            - float, or list of float ("lsq_norm")
    '''
    K = len(proj)
    m = proj * proj  # (K,): Energy each placement explains alone
    P = int(ends.max()) + 1

    ## Pairs that do not overlap: Exactly
    value, pair = _best_disjoint_pair(m, starts, ends)
    scale = 1.0

    ## Pairs that overlap: Both among the placements with m > value - max(m), by decreasing m
    order = np.argsort(-m, kind="stable")
    m_max = m[order[0]]
    n_cand = K if not np.isfinite(value) else int(np.searchsorted(-m[order], -(value - m_max), side="left"))
    if n_cand < 2:
        return value, pair, scale
    I = order[:n_cand]  # (n_cand,)
    U = np.zeros((n_cand, P, unit[0].shape[1]))  # Each placed candidate
    for r, k in enumerate(I):
        U[r, starts[k]:ends[k]] = unit[cands[k]]
    U = cp.asarray(U.reshape(n_cand, -1))
    W = cp.asarray(np.concatenate([[0.0], np.cumsum(position_energy)]))  # W[x]: Energy of positions before x
    mI, bI, lI, eI = (cp.asarray(x[I]) for x in (m, proj, starts, ends))
    CI = None if overlap_cost_prefix is None else cp.asarray(overlap_cost_prefix[I])  # (n_cand, J+1)
    m_sorted = m[I]  # Decreasing
    for s in range(0, n_cand, row_block):
        if m_sorted[s] + m_max <= value:  # No pair left can beat value
            break
        rows = slice(s, min(n_cand, s + row_block))
        n_col = n_cand if not np.isfinite(value) else int(np.searchsorted(-m_sorted, -(value - m_sorted[s]), side="left"))
        for c0 in range(0, n_col, col_block):
            cols = slice(c0, min(n_col, c0 + col_block))
            X = (U[rows] @ U[cols].T) * bI[rows, None] * bI[None, cols]  # (R, C)
            lo = cp.maximum(lI[rows, None], lI[None, cols])
            hi = cp.minimum(eI[rows, None], eI[None, cols])
            B = mI[rows, None] + mI[None, cols]
            ok = (lo < hi) & (lI[rows, None] != lI[None, cols]) & (B > value)  # Overlapping, distinct starts
            pen = (overlap_penalty_fraction * (W[hi] - W[lo]) if CI is None
                   else _masked_pair_cost(CI[rows], CI[cols], lI[rows, None], lI[None, cols], hi))
            if score == "lsq_norm":  # The refit scales x_i, x_j of the two (non-negative), or the better one alone
                b_i, b_j = bI[rows, None], bI[None, cols]
                g = cp.where(b_i * b_j > 0, X / cp.where(b_i * b_j > 0, b_i * b_j, 1.0), 0.0)
                det = 1 - g * g
                x_i = (b_i - g * b_j) / cp.where(det > 1e-12, det, 1.0)
                x_j = (b_j - g * b_i) / cp.where(det > 1e-12, det, 1.0)
                both = (x_i >= 0) & (x_j >= 0) & (det > 1e-12)
                val = cp.where(both, x_i * b_i + x_j * b_j, B - cp.minimum(mI[rows, None], mI[None, cols]))
                val = val - pen
            else:
                val = B * B / cp.maximum(B + 2 * X, 1e-300) - pen
            val = cp.where(ok, val, -cp.inf)
            flat = int(cp.argmax(val))
            a, c = divmod(flat, val.shape[1])
            if float(val[a, c]) > value:
                i, jj = int(I[s + a]), int(I[c0 + c])
                value = float(val[a, c])
                pair = sorted([i, jj], key=lambda k: starts[k])
                if score == "lsq_norm":
                    if bool(both[a, c]):
                        ratio = lambda x, b: float(x / b) if float(b) > 0 else 0.0
                        by_row = {i: ratio(x_i[a, c], b_i[a, 0]), jj: ratio(x_j[a, c], b_j[0, c])}
                    else:  # The better one alone
                        by_row = {i: float(mI[s + a] >= mI[c0 + c]), jj: float(mI[s + a] < mI[c0 + c])}
                    scale = [by_row[k] for k in pair]
                else:
                    scale = float(B[a, c] / (B[a, c] + 2 * X[a, c]))
    return value, pair, scale


def run_modified_smith_waterman_alignment(
    target_motif, reference_motif,
    sim_threshold=0.0,
    overlap_penalty_f1=0.02, overlap_penalty_f2=0.1,
    skip_penalty_f1=0.1, skip_penalty_f2=0.1,
    skip_penalty_l1=0.02, skip_penalty_l2=0.1,
    global_exact=True,
    match_count_penalty=0.0,
    match_count_fraction=1.0,
    match_count_max=None,
    return_match_profile=False,
    return_matrices=False,
    label_unsigned=True,
    finger_length=3,
    ic_scale=False,
):
    '''
    Aligns a chain of fixed-length finger motifs against a CWM, in the order and orientation
    given: unlike run_modified_smith_waterman_composite(), fingers keep their length and are not
    reordered, reverse complemented individually, or repeated. Each alignment pass matches every
    finger in one orientation (msw_orient).

    Every finger length, including 3 bp B1H zinc fingers, uses the generalized recurrence
    (calculate_modified_smith_waterman).

    A finger may overhang either end of the CWM, overlapping it by as little as 1 bp. Every CWM
    position in the results (actions, alignment tracks, columns of aligned_motif_fs) is padded
    by M-1: CWM position t is position t + M - 1.

    Arguments:
        target_motif: CWM motif matrix (as CWM) - np.array of shape (L, D)
        reference_motif: Finger motif matrix (as IC-scaled PPM) - np.array of shape (F*M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float >= 0
        overlap_penalty_f2: Penalty for each additional overlapping bp - float >= 0
        skip_penalty_f1: Penalty for skipping 1 finger - float >= 0
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float >= 0
        skip_penalty_l1: Penalty for the first CWM position of a gap between two fingers
            - float >= 0
        skip_penalty_l2: Penalty for each additional CWM position of a gap - float >= 0
            (Every penalty is subtracted from the score.)
        global_exact: Find the best alignment exactly, keeping the best alignment of every state
            (CWM positions and fingers skipped since the last match, fingers matched) in each DP
            cell; otherwise keep one per cell, which is faster but, with a penalty, can miss the
            best alignment (default: True) - bool
        match_count_penalty: Penalty per finger matched: keeps the alignment with the N that
            maximizes its best score with N matches - match_count_penalty * N, and subtracts
            match_count_penalty * N from msw_score (see calculate_modified_smith_waterman())
            - float >= 0
        match_count_fraction: Keep the fewest matches whose penalized best score reaches this
            fraction of the maximum; 1 keeps the maximum (default: 1) - float in (0, 1]
        match_count_max: Most fingers matched, whatever the penalty and fraction (None: no
            limit) (default: None) - int >= 1, or None
        return_match_profile: Also return msw_profile - bool
            Only the local alignment is chosen (and limited): the remaining alignment still adds the fingers
            left after it (see _run_modified_smith_waterman()).
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Compare the absolute values of the motifs (unsigned); if False, compare the
            signed motifs on 8 channels, so that a negative region only matches a negative finger
            (see run_modified_smith_waterman_composite()) (default: True) - bool
        finger_length: Length of each finger motif - int (M)
        ic_scale: IC-scale the CWM and the fingers (once, before any calculation), so that
            every similarity, match score and scaling uses the IC-scaled motifs; pass motifs
            that are not IC-scaled already (the B1H PPM fingers usually are). The global
            IC-scale setting is ignored (default: False) - bool

    Returns:
        msw_score: Modified Smith Waterman score of optimal alignment - float
        sim_score: MotifCompendium similarity score of optimal alignment - float
        aligned_motif_fs: Aligned finger motifs (IC-scaled, if ic_scale) - np.array of shape
            (F, L+2M-2, D)
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
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        match_count_max=match_count_max,
        return_match_profile=return_match_profile,
        ic_scale=ic_scale,
    )


def run_modified_smith_waterman_znf(
    target_motif, reference_motif,
    sim_threshold=0.0,
    overlap_penalty_f1=0.02, overlap_penalty_f2=0.1,
    skip_penalty_f1=0.1, skip_penalty_f2=0.1,
    skip_penalty_l1=0.02, skip_penalty_l2=0.1,
    global_exact=True,
    match_count_penalty=0.0,
    match_count_fraction=1.0,
    match_count_max=None,
    return_match_profile=False,
    return_matrices=False,
    label_unsigned=True,
    finger_length=3,
    ic_scale=False,
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
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        match_count_max=match_count_max,
        return_match_profile=return_match_profile,
        return_matrices=return_matrices,
        label_unsigned=label_unsigned,
        finger_length=finger_length,
        ic_scale=ic_scale,
    )


def _run_modified_smith_waterman(
    target_motif, subunit_motif_stack,
    sim_threshold,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    return_matrices,
    label_unsigned,
    global_exact=True,
    match_count_penalty=0.0,
    match_count_fraction=1.0,
    match_count_max=None,
    return_match_profile=False,
    ic_scale=False,
):
    '''
    Aligns a chain of fixed-length finger motifs against a motif, for any finger length M > 1.
    The fingers are matched in both orientations (msw_orient), placed as given, and, after the
    local alignment, the fingers left after it are aligned to the rest of the motif (globally),
    so that every finger is accounted for: their matches are added to the alignment, but not to
    msw_score.

    The target is padded with M-1 zeros on either side, so that a finger may overhang either
    end, overlapping the target by as little as 1 bp. Every motif position in the results is in
    these padded coordinates: target position t is position t + M - 1.

    Arguments:
        target_motif: Target motif - np.array of shape (L, D)
        subunit_motif_stack: Stack of subunit motifs - np.array of shape (F, M, D)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float >= 0
        overlap_penalty_f2: Penalty for each additional overlapping bp - float >= 0
        skip_penalty_f1: Penalty for skipping 1 finger - float >= 0
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float >= 0
        skip_penalty_l1: Penalty for the first motif position of a gap - float >= 0
        skip_penalty_l2: Penalty for each additional motif position of a gap - float >= 0
        return_matrices: Also return the full DP matrices. They are large and no known caller
            reads them, so they default to None - bool
        label_unsigned: Compare absolute values (unsigned); if False, signed motifs on 8 channels
            - bool
        global_exact: Keep the best alignment of every state in each DP cell (exact), rather
            than one alignment per cell - bool
        match_count_penalty, match_count_fraction, match_count_max: Choose (and limit) the number
            of matches of the local alignment (see calculate_modified_smith_waterman()) - float
            >= 0, float in (0, 1], int >= 1 or None
        return_match_profile: Also return msw_profile - bool
        ic_scale: IC-scale the target and the fingers, once, before any calculation (the
            global IC-scale setting is ignored) - bool

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
    # Setting: IC-scale off (store), so that compute_similarities() does not IC-scale again
    ic_scale_config = utils_config.get_ic_scale()
    if ic_scale_config:
        utils_config.set_ic_scale(False)

    ## Preprocess motifs
    F, M, D = subunit_motif_stack.shape # (F, M, D)

    # IC-scale: Only once, here
    if ic_scale:
        target_motif = utils_motif.ic_scale(target_motif)
        subunit_motif_stack = utils_motif.ic_scale(subunit_motif_stack)

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

    ## Calculate similarities: On 8 channels (the positive and negative part of each base apart), as
    # MotifCompendium's similarity, so that a signed window only matches a subunit of the same sign.
    # Unsigned motifs have no negative part, so 4 channels give the same similarities. The 8
    # channels are laid out so that [::-1, ::-1] still reverse complements
    if label_unsigned:
        sim_stack_cp, sim_windows_cp = subunit_motif_stack_norm_cp, target_motif_norm_unroll_cp
    else:
        sim_stack_cp = cp.asarray(utils_motif.motif_4_to_8(subunit_motif_stack_norm)) # (F, M, 8)
        sim_windows_cp = cp.asarray(utils_motif.motif_4_to_8(target_motif_norm_unroll)) # (L-M+1, M, 8)
    sims_ff = compute_aligned_similarity(sim_stack_cp, sim_windows_cp) # (F, L-M+1)
    sims_fr = compute_aligned_similarity(sim_stack_cp, sim_windows_cp[:, ::-1, ::-1]) # (F, L-M+1)

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
        global_exact=global_exact,
        match_count_penalty=match_count_penalty,
        match_count_fraction=match_count_fraction,
        match_count_max=match_count_max,
        return_match_profile=return_match_profile,
        alignment_type="local",
        return_matrices=return_matrices,
    )

    ## Add remaining global alignment
    remain_f, remain_l = _next_unused_indices(msw_actions, M)  # First unused finger / motif index
    if msw_orient == "rev":
        remain_sim = sims_fr
    else:
        remain_sim = sims_ff

    if len(msw_actions) > 0 and remain_f < F and remain_l < L_M_1:
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
    if ic_scale_config:
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
    """Computes the cosine similarity of a (N, L, K) tensor with a (M, L, K) tensor, for any number of
    channels K (4, or 8 for signed motifs) (0 for an all-zero motif)."""
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


def _check_penalty(penalty, name):
    '''
    A penalty: a number >= 0, which is subtracted from the score. (Penalties used to be given
    as numbers <= 0, and added; the string 'l2_norm' was removed.)
    '''
    if isinstance(penalty, str):
        raise ValueError(f"{name} must be a number >= 0, got {penalty!r} ('l2_norm' was removed)")
    if not penalty >= 0:
        raise ValueError(f"{name} must be >= 0 (penalties are subtracted from the score), got {penalty}")
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
_EMPTY_STATE = {"H_match": 0.0, "H_penalty_overlap_f": 0.0, "H_penalty_skip_f": 0.0,
                "H_penalty_skip_l": 0.0, "F_total": 0}


def _state_space(
    finger_length,
    overlap_penalty_f1, overlap_penalty_f2,
    skip_penalty_f1, skip_penalty_f2,
    skip_penalty_l1, skip_penalty_l2,
    num_f_total,
):
    '''
    The states of a DP cell: everything that decides how an alignment can continue from it.

    Keeping only the best alignment of a cell would lose a better one whenever the next
    penalty depends on how the cell was reached, so a cell can keep the best alignment of every
    state (a triple of indices):
        L_skip: Motif positions since the last match: -1 (no match yet), 0 .. M, or M+1 (M+1
            or more). The next finger's overlap and gap penalties depend on it. Past M+1, the
            gap before any later finger is 2 bp or more, so each further position adds exactly
            skip_penalty_l2 to its penalty: that is charged as the position is passed.
        F_skip: Fingers skipped since the last match: 0, or 1 (1 or more). The next finger skip
            costs skip_penalty_f1 or skip_penalty_f2.
        F_total: Matches so far (0 .. num_f_total-1), to choose and limit the number of matches.
    An axis has a single value when no penalty depends on it, so zero penalties cost nothing.

    Of an alignment, a cell keeps only its score H (H_match minus its penalties), in a last axis
    of length 1 ("parts"). (The three penalty sums of the alignment found are recovered from its
    moves: see _path_state().)

    Penalties are given as numbers >= 0, and the tables hold what each move adds to the score:
    minus the penalty. A gap of G bp costs skip_penalty_l1 + (G - 1) * skip_penalty_l2, as an
    overlap of o bp costs overlap_penalty_f1 + (o - 1) * overlap_penalty_f2.

    Both engines use the same states. The exact engine (_row_candidates()) indexes its matrices
    by them; one state per cell (_one_state_row_candidates()) stores the state of its alignment
    with it, as a flat index, and moves it with the tables by state (diag_to, down_to, ...).

    Arguments:
        finger_length: Length of each finger motif - int (M)
        overlap_penalty_f1, overlap_penalty_f2, skip_penalty_f1, skip_penalty_f2,
            skip_penalty_l1, skip_penalty_l2: Penalties - float >= 0
        num_f_total: Values of F_total (1: not tracked) - int
    Returns:
        space - SimpleNamespace:
            M, shape (of the states), states (flat index of each state), d0 (index of L_skip = 0)
            overlap_f, skip_l: Overlap and gap charges (minus the penalties) of diagonal k, by
                L_skip index - np.array (M+1, nD), and as lists (overlap_list, skip_list)
            right_from, right_to: The right move's source and target L_skip indices
            skip_f1, skip_f2, skip_l2: Charges (minus the penalties)
            By state (flat index), as lists: d_of (L_skip index), count (F_total), diag_to (the
                state after a match; None past the last F_total), down_to and down_pen (after a
                finger skip, and its charge), right_state and right_pen (after a motif skip, and
                its charge; None: none); and start_state
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
                    overlap_f[k, L_skip + 1] = -(overlap_penalty_f1 + (M - d - 1) * overlap_penalty_f2)
                elif d > M:  # Gap: (d - M) bp
                    skip_l[k, L_skip + 1] = -(skip_penalty_l1 + (d - M - 1) * skip_penalty_l2)

    # The moves, by state
    d_of, count, diag_to, down_to, down_pen, right_state, right_pen = [], [], [], [], [], [], []
    for d, f_skip, n in np.ndindex(*shape):
        d_of.append(d)
        count.append(n)
        diag_to.append(int(states[d0, 0, n + 1]) if n + 1 < nN else (int(states[d0, 0, 0]) if nN == 1 else None))
        down_to.append(int(states[d, nF - 1, n]))
        down_pen.append(-skip_penalty_f2 if f_skip == 1 else -skip_penalty_f1)
        if nD == 1 or d == 0:  # Untracked, or no match yet: Stays
            right_state.append(int(states[d, f_skip, n]))
            right_pen.append(None)
        elif d < M + 2:  # L_skip + 1
            right_state.append(int(states[d + 1, f_skip, n]))
            right_pen.append(None)
        else:  # M+1 or more: The next gap grows by 1 bp
            right_state.append(int(states[d, f_skip, n]))
            right_pen.append(-skip_penalty_l2)

    return SimpleNamespace(
        M=M,
        shape=shape,
        states=states,
        d0=d0,
        overlap_f=overlap_f,
        skip_l=skip_l,
        overlap_list=overlap_f.tolist(),
        skip_list=skip_l.tolist(),
        right_from=np.array([0] + list(range(1, M + 2))),  # L_skip -1 stays -1; D -> D + 1
        right_to=np.array([0] + list(range(2, M + 3))),
        skip_f1=-skip_penalty_f1,
        skip_f2=-skip_penalty_f2,
        skip_l2=-skip_penalty_l2,
        d_of=d_of,
        count=count,
        diag_to=diag_to,
        down_to=down_to,
        down_pen=down_pen,
        right_state=right_state,
        right_pen=right_pen,
        start_state=int(states[d0, 0, 1 if nN > 1 else 0]),
    )


def _state_value(parts):
    '''Value of states, from their parts (last axis): their score H, as a new array.'''
    return parts[..., 0].copy()


def _initial_parts(space, initial_state):
    '''Parts (see _state_space()) of the alignment that a DP starts from.'''
    H_match = initial_state["H_match"]
    return [H_match - (initial_state["H_penalty_overlap_f"] + initial_state["H_penalty_skip_f"]
                       + initial_state["H_penalty_skip_l"])]


def _offer(best, target, candidate, candidate_move, candidate_source):
    '''Takes a candidate for the target states where it is strictly better (earlier candidates win ties).'''
    value, parts, move, source = best
    candidate_value = _state_value(candidate)
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
            list of np.array (j1-j0, *space.shape), parts with a last axis of 1 (H)
    '''
    M = space.M
    nD, nF, nN = space.shape
    n = j1 - j0
    parts_best = np.zeros((n,) + space.shape + (1,))
    parts_best[..., 0] = -np.inf  # Unreachable
    best = [np.full((n,) + space.shape, -np.inf), parts_best,
            np.full((n,) + space.shape, -1), np.zeros((n,) + space.shape, dtype=int)]
    cells = (slice(None),)

    # Diagonal k: Match, to L_skip = 0, F_skip = 0 and F_total + 1, from the best source state
    if matchable.any():
        n_from, n_to = (slice(0, 1), slice(0, 1)) if nN == 1 else (slice(0, nN - 1), slice(1, nN))
        # Without L_skip states, the right move is free and keeps the state, so row i-1 never
        # decreases along j: diagonal 1 is never beaten (and wins ties), so it is the only one tried
        for k in range(1, M + 1 if nD > 1 else 2):
            candidate = parts[i - 1, j0 - k:j1 - k].copy()
            candidate[..., 0] += match_scores[:, None, None, None]
            candidate[..., 0] += space.overlap_f[k][None, :, None, None]
            candidate[..., 0] += space.skip_l[k][None, :, None, None]
            candidate[~matchable, ..., 0] = -np.inf
            candidate = candidate.reshape(n, nD * nF, nN, 1)
            if nD * nF == 1:  # One source state per F_total: Nothing to choose
                picked = candidate[:, 0]
                picked_source = np.broadcast_to(space.states.reshape(1, nN), (n, nN))
            else:
                first = _state_value(candidate).argmax(axis=1)  # By cell and F_total
                picked = np.take_along_axis(candidate, first[:, None, :, None], axis=1)[:, 0]
                picked_source = space.states.reshape(nD * nF, nN)[first, np.arange(nN)]
            _offer(best, cells + (space.d0, 0, n_to), picked[:, n_from], k - 1,
                   picked_source[:, n_from])

    # Down: Skip the finger, to F_skip = 1 or more
    below = parts[i - 1, j0:j1]
    if nF == 1:
        candidate = below.copy()
        candidate[..., 0] += space.skip_f1
        _offer(best, cells, candidate, M, space.states)
    else:
        for f_skip, skip_penalty in ((0, space.skip_f1), (1, space.skip_f2)):
            candidate = below[:, :, f_skip].copy()
            candidate[..., 0] += skip_penalty
            _offer(best, cells + (slice(None), 1), candidate, M, space.states[:, f_skip])
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
            - np.array (rows, cols, *space.shape, 1)
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
    M = space.M
    nD, nF, nN = space.shape

    # Right: Skip a motif position, to L_skip + 1 (no match yet: stays -1)
    left = parts[i, j - 1]
    if nD == 1:
        _offer(best, (slice(None),), left, M + 1, space.states)
    else:
        _offer(best, (space.right_to,), left[space.right_from], M + 1,
               space.states[space.right_from])
        # L_skip M+1 or more: The next gap grows by 1 bp, which costs skip_penalty_l2
        capped = left[M + 2].copy()
        capped[..., 0] += space.skip_l2
        _offer(best, (M + 2,), capped, M + 1, space.states[M + 2])

    # Start: A new local alignment, at this match
    if matchable and allow_start:
        n_1 = 1 if nN > 1 else 0
        candidate = np.zeros((1, 1))
        candidate[0, 0] = match_score
        _offer(best, (space.d0, 0, slice(n_1, n_1 + 1)), candidate, M + 2, np.zeros(1, dtype=int))
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
            np.array (F+1, num_cols, *space.shape, 1)
    '''
    F, L_M_1 = match_scores.shape
    M = space.M
    parts = np.zeros((F + 1, num_cols) + space.shape + (1,))
    parts[..., 0] = -np.inf  # Unreachable
    initial = (slice(None), slice(None), 0, 0, initial_state["F_total"] if space.shape[2] > 1 else 0)
    parts[initial] = _initial_parts(space, initial_state)

    # A row that cannot match, under a free finger skip, is the row above: its cells take the
    # cell above (down, first) or the cell to the left, which the row above already took
    free_down = space.shape[1] == 1 and space.skip_f1 == 0
    for f in range(F):
        if free_down and not matchable[f].any():
            parts[f + 1] = parts[f]
            continue
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


def _path_state(steps, space, match_scores, initial_state):
    '''
    H_match, the three penalty sums and F_total of an alignment, and its score, from its moves:
    each move is charged as the DP charges it (see _cell_candidates()), in order, from
    initial_state.

    Returns:
        state: H_match, H_penalty_overlap_f, H_penalty_skip_f, H_penalty_skip_l (each >= 0) and
            F_total - dict
        score: H_match - the three penalty sums - float
    '''
    M = space.M
    track_l = space.shape[0] > 1
    state = dict(initial_state)
    d, f_skip = 0, 0  # L_skip index (0: no match yet), fingers skipped since the last match
    # The tables hold charges (minus the penalties): Penalty sums subtract them
    for action, f, l, k in steps:
        if action == "match":
            if k == 0:  # Start: A new alignment
                state, d, k = dict(_EMPTY_STATE), 0, 1
            state["H_match"] += match_scores[f, l]
            state["H_penalty_overlap_f"] -= space.overlap_list[k][d]
            state["H_penalty_skip_l"] -= space.skip_list[k][d]
            state["F_total"] += 1
            d, f_skip = space.d0, 0
        elif action == "F-skip":
            state["H_penalty_skip_f"] -= space.skip_f2 if f_skip > 0 and space.shape[1] > 1 else space.skip_f1
            f_skip += 1
        elif track_l and d == M + 2:  # L-skip, past M+1: The next gap grows by 1 bp
            state["H_penalty_skip_l"] -= space.skip_l2
        elif track_l and d > 0:  # L-skip
            d += 1
    score = (state["H_match"] - state["H_penalty_overlap_f"] - state["H_penalty_skip_f"]
             - state["H_penalty_skip_l"])
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
        state_final: H, the score of the best alignment of each cell and state, and the value of each state axis (L_skip M+1 is "M+1 or more"; F_skip 1 is
            "1 or more") - dict
    '''
    names = ("L_skip", "F_skip", "F_total")
    values = (np.arange(-1, space.M + 2), np.arange(2), np.arange(space.shape[2]))
    keep = [n > 1 for n in space.shape]
    shape = score.shape[:2] + tuple(n for n, k in zip(space.shape, keep) if k)
    state_final = {"H": parts[..., 0].reshape(shape)}
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
    by_row = score.reshape(score.shape[0], -1, num_n)  # DP row by DP row: argmax(axis=0) is strided
    best = np.full(num_n, -np.inf)
    rows = np.zeros(num_n, dtype=np.int64)
    for i in range(by_row.shape[0]):
        block = by_row[i]
        row_max = block.max(axis=0)
        better = np.flatnonzero(row_max > best)  # Strictly: The first maximum wins, as np.argmax
        if better.size:
            best[better] = row_max[better]
            rows[better] = i * block.shape[0] + block[:, better].argmax(axis=0)
    best_cell = [tuple(int(x) for x in np.unravel_index(r * num_n + n, score.shape)) for n, r in enumerate(rows)]
    return best, best_cell


def _select_match_count(msw_profile, match_count_penalty, match_count_fraction):
    '''
    Number of matches N to align with, from the best score with exactly N matches (msw_profile):
    the smallest N whose penalized score, msw_profile[N] - match_count_penalty * N, reaches
    match_count_fraction of its maximum. With a fraction of 1, that is the N that maximizes it
    (the fewest matches, on ties). N = 0 (the empty alignment, score 0) is always a candidate.
    '''
    penalized = msw_profile - match_count_penalty * np.arange(len(msw_profile))
    return int(np.flatnonzero(penalized >= match_count_fraction * penalized.max())[0])


def _max_matches(F, L_M_1, max_count):
    '''
    Most matches an alignment can have: one per finger and per motif position, and at most
    max_count (None: no limit). The F_total axis stops there, so no alignment with more is kept.
    '''
    return min(F, L_M_1) if max_count is None else min(F, L_M_1, max_count)


def _align_all_states(finger_length, penalties, match_scores, matchable, initial_state, allow_start, num_cols,
                      return_matrices, profile=False, max_count=None):
    '''
    Best alignment of one orientation, exactly: each DP cell keeps the best alignment of every
    state (see _state_space()).

    With profile (local alignment only), F_total is a state, so that the best alignment with
    exactly N matches is known for every N (at most max_count: see _max_matches()).

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
    num_f_total = initial_state["F_total"] + _max_matches(F, L_M_1, max_count) + 1
    space = _state_space(M, *penalties, num_f_total if profile else 1)
    parts = _fill_modified_smith_waterman(space, match_scores, matchable, initial_state, allow_start, num_cols)
    score = _state_value(parts)  # Score of the alignment ending at each cell and state

    if not profile:
        best_score, cell = _best_cell(score, allow_start, M, F, L_M_1)
        best_steps, best_matrices = _trace_all_states(
            space, parts, score, match_scores, matchable, allow_start, cell, return_matrices)
        return best_score, best_steps, best_matrices

    msw_profile, best_cells = _best_by_count(score)

    def trace(n):
        return _trace_all_states(space, parts, score, match_scores, matchable, allow_start, best_cells[n],
                                 return_matrices)
    return msw_profile, trace


##########################
# DP: ONE STATE PER CELL #
##########################
def _one_state_row_candidates(space, parts, states, i, j0, j1, match_scores, matchable, by_count):
    '''
    The diagonal and down candidates of DP cells (i, j0 .. j1-1), as _row_candidates(), when
    each cell keeps one alignment rather than the best of every state: one row or, by_count,
    one row per F_total. A row holds the alignment's parts (as the exact engine's, see
    _state_space()) and, rather than being indexed by its state, its state (a flat index into
    space.shape, which covers only the states a penalty depends on). A row's value is its H.

    Arguments:
        space: DP states - see _state_space()
        parts, states: Rows of each DP cell: parts (tuple, or None: no alignment) and state -
            list of list of list
        i, j0, j1: DP cells - int
        match_scores, matchable: Of cells j0 .. j1-1 - list of float, list of bool
        by_count: One row per F_total - bool
    Returns:
        best: By cell, and by row: [value, parts, state, move, source row] - list of list of list
    '''
    M = space.M
    overlap_f, skip_l = space.overlap_list, space.skip_list
    d_of, diag_to, down_to, down_pen = space.d_of, space.diag_to, space.down_to, space.down_pen
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
                    candidate = (p[0] + match_score + overlap_f[k][d] + skip_l[k][d],)
                    value = candidate[0]
                    if value > row_best[0]:  # Strictly: Earlier candidates win ties
                        row_best = [value, candidate, s_to, k - 1, source]

            # Down: Skip the finger
            p = parts[i - 1][j][row]
            if p is not None:
                s = states[i - 1][j][row]
                s_to = down_to[s]
                candidate = (p[0] + down_pen[s],)
                value = candidate[0]
                if value > row_best[0]:
                    row_best = [value, candidate, s_to, M, row]
            cell_best.append(row_best)
        best.append(cell_best)
    return best


def _one_state_cell_candidates(space, parts, states, i, j, match_score, matchable, allow_start, by_count,
                               best):
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
    right_state, right_pen = space.right_state, space.right_pen

    # Right: Skip a motif position (past M+1: the next gap grows by 1 bp, which costs skip_penalty_l2)
    for row, row_best in enumerate(best):
        p = parts[i][j - 1][row]
        if p is None:
            continue
        s = states[i][j - 1][row]
        s_to, charge = right_state[s], right_pen[s]
        if charge is not None:
            p = (p[0] + charge,)
        value = p[0]
        if value > row_best[0]:
            row_best[:] = [value, p, s_to, M + 1, row]

    # Start: A new local alignment, at this match
    row = 1 if by_count else 0
    if matchable and allow_start and row < len(best):
        s_to = space.start_state
        candidate = (match_score,)
        value = candidate[0]
        if value > best[row][0]:
            best[row][:] = [value, candidate, s_to, M + 2, row]
    return best


def _fill_one_state(space, match_scores, matchable, initial_state, allow_start, num_cols, by_count):
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
    num_rows = space.shape[2] if by_count else 1
    initial_row = initial_state["F_total"] if by_count else 0
    initial_parts = tuple(_initial_parts(space, initial_state))
    initial_index = int(space.states[0, 0, initial_state["F_total"] if space.shape[2] > 1 else 0])
    initial_value = initial_parts[0]
    empty_parts, empty_states, empty_values = [None] * num_rows, [0] * num_rows, [-math.inf] * num_rows
    empty_parts[initial_row], empty_states[initial_row], empty_values[initial_row] = (
        initial_parts, initial_index, initial_value)
    parts = [[empty_parts] * num_cols for _ in range(F + 1)]
    states = [[empty_states] * num_cols for _ in range(F + 1)]
    score = [[empty_values] * num_cols for _ in range(F + 1)]

    match_scores, matchable = match_scores.tolist(), matchable.tolist()
    for f in range(F):
        row = _one_state_row_candidates(space, parts, states, f + 1, M, M + L_M_1, match_scores[f],
                                        matchable[f], by_count)
        for l in range(L_M_1):
            best = _one_state_cell_candidates(space, parts, states, f + 1, l + M, match_scores[f][l],
                                              matchable[f][l], allow_start, by_count, row[l])
            score[f + 1][l + M] = [b[0] for b in best]
            parts[f + 1][l + M] = [b[1] for b in best]
            states[f + 1][l + M] = [b[2] for b in best]
    return parts, states, np.array(score)


def _one_state_matrices(space, parts, states, score):
    '''
    DP matrices of a one-state fill, as _state_matrices().

    Returns:
        H_final: Score of the best alignment kept at each cell - np.array (rows, cols)
        state_final: H, the score of the alignment kept at each cell (or
            by_count, at each cell and F_total: "axes"), and its state, by the fields a penalty
            depends on (L_skip M+1 is "M+1 or more"; F_skip 1 is "1 or more") - dict
    '''
    num_rows = score.shape[-1]
    unreachable = (-np.inf,)
    values = np.array([[[p if p is not None else unreachable for p in cell] for cell in row] for row in parts],
                      dtype=float)  # (rows, cols, rows of a cell, parts)
    fields = np.unravel_index(np.array(states), space.shape)  # L_skip index, F_skip, F_total
    if num_rows == 1:
        values, fields = values[:, :, 0], tuple(field[:, :, 0] for field in fields)
    state_final = {"H": values[..., 0]}
    for name, field, size, offset in zip(("L_skip", "F_skip", "F_total"), fields, space.shape, (-1, 0, 0)):
        if size > 1:
            state_final[name] = field + offset
    state_final["axes"] = {} if num_rows == 1 else {"F_total": np.arange(num_rows)}
    return score.max(axis=-1), state_final


def _align_one_state(finger_length, penalties, match_scores, matchable, initial_state, allow_start, num_cols,
                     return_matrices, profile=False, max_count=None):
    '''
    Best alignment of one orientation, keeping one alignment per DP cell (see
    _one_state_row_candidates()).

    Faster than _align_all_states(), and the same without penalties. With them, the alignment a
    cell keeps is not always the one that continues best, so the best alignment can be missed.

    With profile (local alignment only), a cell keeps one alignment per F_total, so that an
    alignment with exactly N matches is known for every N. When no penalty depends on L_skip or
    F_skip, that is exact, and done by _align_all_states(), which is faster.

    Returns: as _align_all_states(), with matrices from _one_state_matrices()
    '''
    M = finger_length
    F, L_M_1 = match_scores.shape
    if profile and _state_space(M, *penalties, 1).shape[:2] == (1, 1):
        return _align_all_states(M, penalties, match_scores, matchable, initial_state, allow_start, num_cols,
                                 return_matrices, profile=True, max_count=max_count)

    num_f_total = initial_state["F_total"] + _max_matches(F, L_M_1, max_count) + 1
    space = _state_space(M, *penalties, num_f_total if profile else 1)
    parts, states, score = _fill_one_state(space, match_scores, matchable, initial_state, allow_start,
                                           num_cols, by_count=profile)
    scores_list, matchable_list = match_scores.tolist(), matchable.tolist()

    def moves_at(i, j):
        f, l = i - 1, j - M
        row = _one_state_row_candidates(space, parts, states, i, j, j + 1, [scores_list[f][l]],
                                        [matchable_list[f][l]], profile)
        best = _one_state_cell_candidates(space, parts, states, i, j, scores_list[f][l], matchable_list[f][l],
                                          allow_start, profile, row[0])
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
    global_exact=True,
    match_count_penalty=0.0,
    match_count_fraction=1.0,
    match_count_max=None,
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
        d <  M: the fingers overlap over o = M - d positions, which costs
                overlap_penalty_f1 + (o - 1) * overlap_penalty_f2
        d == M: the fingers are adjacent
        d >  M: the fingers are separated by a gap of G = d - M positions, which costs
                skip_penalty_l1 + (G - 1) * skip_penalty_l2
    A run of g fingers skipped between two matches costs skip_penalty_f1 + (g-1) * skip_penalty_f2.
    Every penalty is affine (a first unit, then each further one), given as a number >= 0, and
    subtracted. Nothing before the first match or after the last one is charged.

    The score of an alignment is
        H = H_match - H_penalty_overlap_f - H_penalty_skip_f - H_penalty_skip_l,
    with H_match the sum of scale_factor * similarity over its matches, and the three penalty
    sums.

    With global_exact, the alignment found is the best one: each cell keeps the best alignment
    of every state that decides how it can continue (see _state_space(), _align_all_states()).
    Without it, each cell keeps only its best alignment, with its state (see _align_one_state()):
    faster, and the same without penalties, but with them the best alignment can be missed.
    Either way, the alignment is traced back by recomputing each move (see _cell_candidates()),
    so no matrix of moves is kept.

    The number of matches can be chosen, rather than left to the best score (local alignment
    only). msw_profile[N] is the best score of an alignment with exactly N matches (0 for the
    empty alignment, -inf where impossible); it is exact with global_exact, and without it, by
    keeping one alignment per cell and F_total. The alignment kept has the smallest N whose
    penalized score, msw_profile[N] - match_count_penalty * N, reaches match_count_fraction of
    its maximum: with the default fraction, 1, the N that maximizes it (the fewest matches, on
    ties). msw_score is that penalized score (msw_profile itself is not). Without global_exact, keeping one alignment
    per cell and F_total can find a better alignment than one per cell, so for a sweep of
    match_count_penalty, set return_match_profile at 0 too. Match scores are normalized (the matches
    of a tiling of the motif sum to about 1), so match_count_penalty = 0.02 keeps a match only if
    it raises the score by more than 0.02. With global_exact, this is a per-match penalty in the
    score. match_count_max limits N whatever the penalty and fraction: the DP keeps
    no alignment with more matches, so msw_profile stops at it, and the alignment found is the
    best (exactly, with global_exact) of at most match_count_max matches.

    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        sim_for: Similarity scores (forward) - np.array of shape (F, L-M+1)
        sim_rev: Similarity scores (reverse) - np.array of shape (F, L-M+1)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        sim_threshold: Minimum similarity to consider a match - float
        overlap_penalty_f1: Penalty for the first overlapping bp of two fingers - float >= 0
        overlap_penalty_f2: Penalty for each additional overlapping bp - float >= 0
        skip_penalty_f1: Penalty for skipping 1 finger - float >= 0
        skip_penalty_f2: Penalty for each additional finger skipped in a row - float >= 0
        skip_penalty_l1: Penalty for the first motif position of a gap - float >= 0
        skip_penalty_l2: Penalty for each additional motif position of a gap - float >= 0
        global_exact: Keep the best alignment of every state in each cell (exact), rather than
            one alignment per cell - bool
        match_count_penalty: Penalty per match, for choosing the number of matches, subtracted
            from msw_score - float >= 0
        match_count_fraction: Choose the fewest matches whose penalized score reaches this
            fraction of the best (1: the best) - float in (0, 1]
        match_count_max: Most matches, whatever the penalty and fraction (None: no limit)
            - int >= 1, or None
        return_match_profile: Compute msw_profile even without a match count penalty, a
            fraction below 1 or a match_count_max - bool
        alignment_type: Type of alignment ("local" or "global") - str
        initial_state: State of the alignment that a global alignment continues: H_match, the
            three penalty sums and F_total (as msw_state); None for an empty alignment - dict
        initial_f_idx: Initial finger index - int
        initial_l_idx: Initial motif position index - int
        orientations: Which orientation(s) to evaluate - tuple of "for"/"rev"
        return_matrices: Also return H_final/state_final; None unless requested - bool
    Returns:
        msw_score: Optimal alignment score, minus match_count_penalty * N when the number of
            matches N is chosen - float
        msw_orient: Optimal orientation ("for" or "rev") - str
        msw_actions: Optimal alignment arrangement - list of str
        msw_state: H_match, the three penalty sums (each >= 0) and F_total of the optimal
            alignment, before match_count_penalty - dict
        H_final: Best score of an alignment ending at each DP cell - np.array (F+1, L+M), None
            unless requested
        state_final: The parts that the DP keeps (H, or H_match and H_penalty; see
            _state_space()) by DP cell and state (see _state_matrices()), or without
            global_exact, by DP cell with the state fields a penalty depends on (see
            _one_state_matrices()), None unless requested - dict
        msw_profile: Best score with exactly N matches, by N - np.array (min(F, L-M+1,
            match_count_max) + 1,), None unless a match count penalty, fraction, maximum or
            return_match_profile
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    penalties = tuple(_check_penalty(penalty, name) for penalty, name in (
        (overlap_penalty_f1, "overlap_penalty_f1"), (overlap_penalty_f2, "overlap_penalty_f2"),
        (skip_penalty_f1, "skip_penalty_f1"), (skip_penalty_f2, "skip_penalty_f2"),
        (skip_penalty_l1, "skip_penalty_l1"), (skip_penalty_l2, "skip_penalty_l2")))
    match_count_penalty = _check_penalty(match_count_penalty, "match_count_penalty")
    assert alignment_type in ["local", "global"]

    F, L_M_1 = sim_for.shape # (F, L-M+1)
    L, D = matrix_l.shape # (L, D)
    M = matrix_f.shape[1] # Finger length
    assert M > 1
    assert L_M_1 == L - M + 1
    local = alignment_type == "local"
    initial_state = {key: (initial_state or _EMPTY_STATE)[key] for key in _EMPTY_STATE}
    align = _align_all_states if global_exact else _align_one_state
    assert match_count_fraction is not None and 0 < match_count_fraction <= 1, \
        f"match_count_fraction must be in (0, 1], got {match_count_fraction}"
    _check_match_count_max(match_count_max)
    profile = (return_match_profile or match_count_penalty != 0 or match_count_fraction < 1
               or match_count_max is not None)
    assert local or not profile, "The number of matches is chosen for local alignment only"

    ## Align: Each orientation
    orient_sims = {"for": sim_for, "rev": sim_rev}
    results = []
    for orient in orientations:
        sim = orient_sims[orient]
        match_scores = scale_factor[None, :] * sim # (F, L-M+1)
        matchable = sim > sim_threshold # (F, L-M+1)
        results.append((orient, match_scores, align(
            M, penalties, match_scores, matchable, initial_state, local, L + M,
            return_matrices, profile=profile, max_count=match_count_max)))

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
    rules = _state_space(M, *penalties, 1)
    msw_state, msw_score = _path_state(best_steps, rules, best_scores, initial_state)
    if profile:  # The number of matches was chosen: Its penalty is part of the score
        msw_score -= match_count_penalty * n
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

    Penalties are numbers >= 0, subtracted, as in calculate_modified_smith_waterman(), but the
    gap rule is the old one: skip_penalty_l1 for a 1 bp gap, else (G - 1) * skip_penalty_l2.

    Arguments and returns: see calculate_modified_smith_waterman().
    '''
    assert matrix_f.shape[-1] == matrix_l.shape[-1]
    assert sim_for.shape == sim_rev.shape
    assert sim_for.shape[0] == matrix_f.shape[0]
    assert sim_threshold < 1
    # Penalties are given as numbers >= 0; the frozen recurrence adds them, so negate them once
    (overlap_penalty_f1, overlap_penalty_f2, skip_penalty_f1, skip_penalty_f2,
     skip_penalty_l1, skip_penalty_l2) = (-_check_penalty(penalty, name) for penalty, name in (
        (overlap_penalty_f1, "overlap_penalty_f1"), (overlap_penalty_f2, "overlap_penalty_f2"),
        (skip_penalty_f1, "skip_penalty_f1"), (skip_penalty_f2, "skip_penalty_f2"),
        (skip_penalty_l1, "skip_penalty_l1"), (skip_penalty_l2, "skip_penalty_l2")))
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
):
    '''
    Arguments:
        matrix_l: Motif matrix - np.array of shape (L, D)
        matrix_f: Finger motif matrices - np.array of shape (F, M, D)
        scale_factor: Scaling factors for each motif position - np.array of shape (L-M+1,)
        msw_actions: Optimal alignment arrangement - list of str

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
            scale_ij = scale_factor[l_j]
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
