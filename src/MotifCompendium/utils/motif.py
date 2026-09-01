import functools

import numpy as np
import pandas as pd


##################
# MOTIF CHECKING #
##################
def single_or_many_motifs(func):
    """Decorator to handle single or many motifs.

    Functions using this decorator should have their first argument be a motif stack.
    They can assume that the input will pass validate_motif_stack(). However, when
    calling the function, the first argument can be supplied either as a single motif or
    a motif stack. Functions using this decorator should always return a single list or
    np.ndarray such that output[i] corresponds to input_motif_stack[i]. If a single
    motif is passed in, the decorator will return the first element of the list.
    """

    @functools.wraps(func)
    def wrapper(motifs, *args, **kwargs):
        try:
            validate_motif(motifs)
        except Exception as e:
            raise e
        if len(motifs.shape) == 2:
            result = func(motifs[np.newaxis, :, :], *args, **kwargs)
            return result[0]
        return func(motifs, *args, **kwargs)

    return wrapper


def validate_motif(motifs: np.ndarray) -> None:
    """Validate that a variable represents a motif/stack."""
    if not isinstance(motifs, np.ndarray):
        raise TypeError("motifs must be a np.ndarray.")
    if not (len(motifs.shape) in [2, 3]):
        raise ValueError("Must be a single motif (L, 4) or a motif stack (N, L, 4).")
    if not (motifs.shape[-1] == 4):
        raise ValueError("Motifs must have 4 channels (ACGT).")


def validate_single_motif(motif: np.ndarray) -> None:
    """Validate that a motif is a (L, 4) np.ndarray."""
    validate_motif(motif)
    if not len(motif.shape) == 2:
        raise ValueError("Motifs must be of shape (L, 4).")


def validate_motif_stack(motifs: np.ndarray) -> None:
    """Validate that a motif stack is a (N, L, 4) np.ndarray."""
    validate_motif(motifs)
    if not len(motifs.shape) == 3:
        raise ValueError(f"Motifs must be of shape (N, L, 4).")


#######################
# MOTIF MANIPULATIONS #
#######################
@single_or_many_motifs
def reverse_complement(x: np.ndarray) -> np.ndarray:
    """Reverse complements a motif/stack."""
    return x[:, ::-1, ::-1]


@single_or_many_motifs
def pad_motif(motifs: np.ndarray, pad_to: int) -> np.ndarray:
    """Pad (by adding 0s) a motif/stack to a specified length.

    If the given motifs are shorter than pad_to, pad with 0s until it is large enough.
    If the given motifs are larger than pad_to, raises an error.

    Args:
        motifs: A (L, 4) motif or (N, L, 4) motif stack.
        pad_to: The length to pad the motifs to.

    Returns:
        A (pad_to, 4) motif or (N, pad_to, 4) motif stack.
    """
    if not (isinstance(pad_to, int) and pad_to > 0):
        raise ValueError("pad_to must be a positive integer.")
    N, L, K = motifs.shape
    if L > pad_to:
        raise ValueError(f"Cannot pad motif of length {L} to {pad_to}.")
    padded_motif = np.zeros((N, pad_to, K))
    padded_motif[:, 0:L, :] = motifs
    return padded_motif


@single_or_many_motifs
def resize_motif(motifs: np.ndarray, resize_to: int) -> np.ndarray:
    """Resize a motif or motif stacks (by squashing or padding) to a specified length.

    If the given motifs are shorter than resize_to, pad with 0s until it is large
    enough. If the given motifs are larger than resize_to, squash them down to a smaller
    motif. Selects the base pairs to keep that have the highest absolute weights.

    Args:
        motifs: A (L, 4) motif or (N, L, 4) motif stack.
        resize_to: The length to squash or pad the motifs to.

    Returns:
        A (resize_to, 4) motif or (N, resize_to, 4) motif stack.
    """
    if not (isinstance(resize_to, int) and resize_to > 0):
        raise ValueError("resize_to must be a positive integer.")
    N, L, K = motifs.shape
    if L < resize_to:
        return pad_motif(motifs, pad_to=resize_to)
    elif L > resize_to:
        # Squash to the desired length
        num_squashes = L - resize_to + 1
        squash_sums = np.empty((N, num_squashes))
        for i in range(num_squashes):
            squash_sums[:, i] = np.sum(
                np.abs(motifs[:, i : i + resize_to, :]), axis=(1, 2)
            )
        top_i = np.argmax(squash_sums, axis=1)
        return motifs[
            np.arange(N)[:, np.newaxis], top_i[:, np.newaxis] + np.arange(resize_to)
        ]
    else:
        return motifs


def trim_motif(motif: np.ndarray, importance: float = 1 / 30):
    """Trim a motif by removing flanking low-importance positions.

    Find the leftmost and rightmost positions in the motif that have a percentage
    importance greater than the importance threshold. Return a trimmed motif that only
    includes the positions between those two positions, inclusive. If the importance
    threshold is too high and the entire motif is trimmed, None is returned. If the
    importance threshold is 0, only positions with 0 importance are trimmed.

    Args:
        motif: A (L, 4) motif or (N, L, 4) motif stack.
        importance: The minimum level of importance a position must have to be included
          in the trimmed motif.

    Returns:
        A (L_trimmed, 4) motif or (N, L_trimmed, 4) motif stack, where L_trimmed is the
          number of positions in the trimmed motif. If the importance threshold is too
          high and the entire motif is trimmed, None is returned.
    """
    validate_single_motif(motif)
    if not (isinstance(importance, (int, float)) and 0 <= importance <= 1):
        raise ValueError("importance must be a number in [0, 1].")
    motif_abs = np.abs(motif)
    per_position_totals = np.sum(motif_abs, axis=1)
    included_positions = per_position_totals > (
        importance * np.sum(per_position_totals)
    )
    if np.sum(included_positions) == 0:
        return None
    min_index = np.argmax(included_positions)
    max_index = motif.shape[0] - np.argmax(included_positions[::-1])
    return motif[min_index:max_index]


################################
# OPERATIONS ON ALIGNED MOTIFS #
################################
@single_or_many_motifs
def view_motif_from_position_range(
    motifs: np.ndarray,
    current_min_pos: int,
    current_max_pos: int,
    new_min_pos: int,
    new_max_pos: int,
) -> np.ndarray:
    """Gets the view of a motif/stack at a specified position range.

    Given a motif or motif stack and current positional bounds, get the motifs as viewed
    from a new positional bound. If the new bounds are outside the current bounds, the
    view will be padded with zeros. If the new bounds are inside the current bounds, the
    view will be cropped.

    Args:
        motifs: A (L, 4) motif or (N, L, 4) motif stack.
        current_min_pos: The position of the 0th index in the length axis.
        current_max_pos: The position of the (L-1)st index in the length axis.
        new_min_pos: The new minimum position from which to view the motif.
        new_max_pos: The new maximum position from which to view the motif.

    Returns:
        A (L_view, 4) motif or (N, L_view, 4) motif stack, where L_view is the length of
          the viewing window. L_view = new_max_pos - new_min_pos + 1.
    """
    if not (current_max_pos - current_min_pos) == (motifs.shape[1] - 1):
        raise ValueError("Current positional range must match motif length.")
    if not (new_min_pos < new_max_pos):
        raise ValueError("New positional range must have a positive length.")
    # Pad if needed
    if new_min_pos < current_min_pos:
        pad_left = current_min_pos - new_min_pos
        motifs = np.pad(motifs, ((0, 0), (pad_left, 0), (0, 0)))
        current_min_pos = new_min_pos
    if new_max_pos > current_max_pos:
        pad_right = new_max_pos - current_max_pos
        motifs = np.pad(motifs, ((0, 0), (0, pad_right), (0, 0)))
        current_max_pos = new_max_pos
    # Crop out new view
    new_min_idx = new_min_pos - current_min_pos
    new_max_idx = new_max_pos - current_min_pos
    return motifs[:, new_min_idx : new_max_idx + 1, :]


def align_motifs(
    motif_stack: np.ndarray,
    alignment_rc: np.ndarray,
    alignment_h: np.ndarray,
    *,
    match_original_range: bool = False,
) -> np.ndarray:
    """Aligned a motif stack based on alignment vectors.

    Uses the alignment information to place the motifs in the motif stack in the correct
    orientation and position.

    Args:
        motif_stack: A (N, L, 4) motif stack.
        alignment_rc: A (N, ) forward/reverse complement alignment vector.
        alignment_h: A (N, ) horizontal alignment vector.
        match_original_range: Whether to match the original positional range of the
          motifs. If False, the returned motifs will have a different length and the
          original positional range will be shifted and expanded to fit all the aligned
          motifs. If True, the returned motifs will only include the portion of the
          aligned motifs that exist in the original positional range.

    Returns:
        An aligned motif stack of shape (N, L', 4). If match_original_range is False,
          L' = L + max(alignment_h) - min(alignment_h). If match_original_range is True,
          L' = L. Before the alignment, the positional bounds are 0 and L-1. After the
          alignment, if match_original_range is False, the positional bounds are
          min(alignment_h) and max(alignment_h) + L - 1 and if match_original_range is
          True, the positional bounds are still 0 and L-1 but truncation may have
          occurred.
    """
    # Check inputs
    validate_motif_stack(motif_stack)
    N, L, K = motif_stack.shape
    if not (isinstance(alignment_rc, np.ndarray) and alignment_rc.shape == (N,)):
        raise ValueError("alignment_rc must be a vector whose length matches N.")
    if not (isinstance(alignment_h, np.ndarray) and alignment_h.shape == (N,)):
        raise ValueError("alignment_h must be a vector whose length matches N.")
    # Create correctly complemented motif stack
    alignment_rc_mtx = np.expand_dims(alignment_rc, axis=(1, 2))
    complemented_motifs = (
        motif_stack * (1 - alignment_rc_mtx)
        + reverse_complement(motif_stack) * alignment_rc_mtx
    )
    # Align motifs
    h_max = np.max(alignment_h)
    h_min = np.min(alignment_h)
    L_new = L + h_max - h_min
    aligned_motifs = np.zeros((N, L_new, K))
    for i in range(N):
        h_i = alignment_h[i] - h_min
        aligned_motifs[i, h_i : h_i + L, :] = complemented_motifs[i, :, :]
    # Match original positional range if needed
    if match_original_range:
        aligned_motifs = view_motif_from_position_range(
            aligned_motifs,
            current_min_pos=h_min,
            current_max_pos=h_max + L - 1,
            new_min_pos=0,
            new_max_pos=L - 1,
        )
    return aligned_motifs


def average_motifs(
    motif_stack: np.ndarray,
    alignment_rc: np.ndarray,
    alignment_h: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    match_original_length: bool = True,
) -> np.ndarray:
    """Compute the average motif of a stack of motifs.

    Calls align_motifs() to compute an aligned motif stack, then averages the aligned
    motifs. If weights are provided, a weighted average is computed. Then, if
    match_original_length is True, the average motif is resized to match the original
    length of the motifs, selecting the most important positions in the average. If
    match_original_length is False, the average motif is not resized and may be longer
    than the original motifs.

    Args:
        motif_stack: A (N, L, 4) motif stack.
        alignment_rc: A (N, ) forward/reverse complement alignment vector.
        alignment_h: A (N, ) horizontal alignment vector.
        weights: A (N, ) vector of relative weights for each motif. If None, all motifs
          are weighed equally.
        match_original_length: Whether to match the original length of the motifs. If
          True, the average motif will be made to be the same length as the original
          motifs. If False, the average motif will have the length of the original motif
          stack.

    Returns:
        The average motif.
    """
    # Check inputs
    validate_motif_stack(motif_stack)
    N, L, K = motif_stack.shape
    if weights is None:
        weights = np.ones(N)
    if not (
        isinstance(weights, np.ndarray)
        and weights.shape == (N,)
        and (weights >= 0).all()
    ):
        raise ValueError(
            "Weights must be a non-negative vector whose length matches that of the motif stack."
        )
    # Average
    aligned_motifs = align_motifs(motif_stack, alignment_rc, alignment_h)
    average_motif = np.average(aligned_motifs, axis=0, weights=weights)
    if match_original_length:
        average_motif = resize_motif(average_motif, resize_to=L)
    return average_motif


def compute_motif_scalar_projection(
    project_motifs: np.ndarray, onto_motifs: np.ndarray, *, keepdims: bool = True
) -> np.ndarray:
    """Compute the scalar projection of one set of motifs onto another set of motifs.

    Compute the scalar projection of project_motifs onto onto_motifs. project_motifs and
    onto_motifs are expected to be motif stacks with the same number of motifs and with
    the same lengths. The scalar projection of each pair is returned. Scalar projections
    are computed by treating each motif as a vector.

    Args:
        project_motifs: A motif stack of shape (N, L, 4) to project onto onto_motifs.
        onto_motifs: A motif stack of shape (N, L, 4) that project_motifs will be
          projected onto.
        keepdims: Whether or not to keep the dimensions of the scalar projection.

    Returns:
        A np.ndarray of scalar projections. If keepdims is True, the shape will be
          (N, L, 4). If keepdims is False, the shape will be (N,).

    Note:
        The scalar projection of u onto v is uTv/vTv. Motifs are treated as vectors for
          the purposes of this calculation, so the dot product between motif1 and motif2
          would be computed as np.sum(motif1*motif2).
    """
    # Check inputs
    validate_motif_stack(project_motifs)
    validate_motif_stack(onto_motifs)
    if project_motifs.shape != onto_motifs.shape:
        raise ValueError(
            f"project_motifs and onto_motifs must have the same shape."
            f"  project_motifs.shape: {project_motifs.shape}, "
            f"  onto_motifs.shape: {onto_motifs.shape}"
        )
    uTv = np.sum(project_motifs * onto_motifs, axis=(1, 2), keepdims=keepdims)
    vTv = np.sum(onto_motifs**2, axis=(1, 2), keepdims=keepdims)
    return np.divide(
        uTv, vTv, where=(vTv != 0), out=np.zeros_like(uTv)
    )  # Avoid divide by zero


def remove_motif_component(
    main_motifs: np.ndarray,
    remove_motifs: np.ndarray,
    alignment_rc: np.ndarray,
    alignment_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the component of one set of motifs from another set of motifs.

    Given a set of motifs of interest, main_motifs, and another set of motifs,
    remove_motifs, whose component you want to remove from main_motifs, remove the
    component of remove_motifs from main_motifs. Removal is done by subtracting the
    projection of main_motifs onto remove_motifs from main_motifs. Alignment
    information of how to reverse complement and shift the main_motifs to align with
    remove_motifs is also required.

    Args:
        main_motifs: A (N, L, 4) motif stack representing the motifs from which
          remove_motifs will be removed.
        remove_motifs: A (N, L, 4) motif stack representing the motifs whose components
          will be removed from main_motifs.
        alignment_rc: A (N, ) forward/reverse complement alignment vector that specifies
          how remove_motifs align with main_motifs. alignment_rc[i] represents whether
          or not remove_motifs[i] should be reverse complemented to align with
          main_motifs[i].
        alignment_h: A (N, ) horizontal alignment vector that specifies how
          remove_motifs align with main_motifs. alignment_h[i] represents how many
          positions to the right remove_motifs[i] must be shifted to align with
          main_motifs[i].

    Returns:
        A (N, L, 4) motif stack representing the subtracted motifs after main_motifs has
          been removed from remove_motifs.
    """
    # Check inputs
    validate_motif_stack(remove_motifs)
    validate_motif_stack(main_motifs)
    if not remove_motifs.shape == main_motifs.shape:
        raise ValueError("main_motifs and remove_motifs must have the same shape.")
    # Align remove_motifs to main_motifs
    remove_motifs_aligned = align_motifs(
        remove_motifs, alignment_rc, alignment_h, match_original_range=True
    )
    # Find the scalar projection of main_motifs onto remove_motifs_aligned
    scalar_projection = compute_motif_scalar_projection(
        main_motifs,
        remove_motifs_aligned,
        keepdims=True,
    )  # Project main_motif onto remove_motifs_aligned
    scaled_removed_motifs = (
        scalar_projection * remove_motifs_aligned
    )  # Component in main_motifs
    main_motifs_updated = main_motifs - scaled_removed_motifs
    # NOTE: no longer clipping: main_motifs_updated = np.clip(main_motifs_updated, a_min=0, a_max=None)
    # Return
    return main_motifs_updated, scaled_removed_motifs


##############################
# MOTIF ENTROPY/MEASUREMENTS #
##############################
def minusxlogx(x: np.ndarray, base: int) -> np.ndarray:
    """Compute -x*log_base(x) with support in x >= 0."""
    return (
        x * np.log2(x, where=(x > 0), out=np.zeros_like(x, dtype=x.dtype))
    ) / -np.log2(
        base
    )  # Minus at end for efficiency


def normalized_last_axis_entropy(x: np.ndarray) -> np.ndarray:
    """Computes the entropy on the last axis assuming that x >= 0."""
    x_sum = np.sum(x, axis=-1, keepdims=True)
    x_normalized = np.divide(
        x, x_sum, out=np.zeros_like(x, dtype=x.dtype), where=(x_sum != 0)
    )
    return np.sum(minusxlogx(x_normalized, base=x.shape[-1]), axis=-1, keepdims=True)


@single_or_many_motifs
def ic_scale(motifs: np.ndarray) -> np.ndarray:
    """Rescale a motif/stack by per position information content.

    Each position in the motif is scaled by the information content at that position.
    The information content at a position is computed as 1 - base4entropy of the per
    base importance at that position.

    Args:
        motifs: A (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        An information content scaled (L, 4) motif or (N, L, 4) motif stack.

    Note:
        If a position only has one base at a position, it will not change. If only two
          bases are present but are represented equally, their weights will be halved.
          And if all bases are present and represented equally, the weights will for all
          bases at that position will be set to 0.
    """
    x_abs = np.abs(motifs)
    entropy = normalized_last_axis_entropy(x_abs)  # (N, L, 1)
    ic = 1 - entropy
    return motifs * ic


@single_or_many_motifs
def abs_ic_scale_normalize(x: np.ndarray) -> np.ndarray:
    """Takes the absolute value then IC scales and L1 normalizes."""
    x = np.abs(x)
    x = ic_scale(x)
    x /= np.sum(x, axis=(1, 2), keepdims=True)
    return x


@single_or_many_motifs
def motif_posneg_sum(x: np.ndarray) -> str | list[str]:
    """Classifies each motif as being positive or negative based on sum."""
    return ["pos" if np.sum(m) > 0 else "neg" for m in np.sum(x, axis=(1, 2)) > 0]


@single_or_many_motifs
def motif_posneg_max(x: np.ndarray) -> str | list[str]:
    """Classifies each motif as being positive or negative based on max value."""
    return ["pos" if np.max(m) > 0 else "neg" for m in np.sum(x, axis=(1, 2)) > 0]


################################
# MOTIF FORMAT TRANSFORMATIONS #
################################
def motif_to_df(motifs: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker."""
    validate_single_motif(motifs)
    return pd.DataFrame(motifs, columns=["A", "C", "G", "T"])


@single_or_many_motifs
def motif_to_string(
    motifs: np.ndarray, *, specificity: float = 0.7, importance: float = 1 / 30
) -> str | list[str]:
    """Transforms motifs into ATCG strings.

    Each motif is turned into an ATCG string. Not all positions are included in the
    string. The first and last positions that are position that are included are the
    first/last position that have a total importance greater than the importance
    attribute. For all included positions, if a single base has greater than a
    specificity% importance, that base is included in the string. If no base meets the
    requirement then a hyphen (-) is included in the string.

    Args:
        motifs: A (L, 4) motif or (N, L, 4) motif stack.
        specificity: The percentage of importance a base must have at a position to be
          included in the string.
        importance: The minimum level of importance a position must have to be included
          in the string.

    Returns:
        A string or list of strings representing the motifs.
    """
    # Check inputs
    if not (isinstance(specificity, (int, float)) and 0.5 < specificity <= 1):
        raise ValueError("specificity must be a number in (0.5, 1].")
    if not (isinstance(importance, (int, float)) and 0 <= importance <= 1):
        raise ValueError("importance must be a number in [0, 1].")
    # Non-negative + L1-normalized + IC-scaled
    motifs = abs_ic_scale_normalize(motifs)
    # Turns to 1s and 0s
    per_position_totals = np.sum(motifs, axis=2, keepdims=True)
    meets_specificity = (
        np.divide(
            motifs,
            per_position_totals,
            out=np.zeros_like(motifs),
            where=(per_position_totals != 0),
        )
        >= specificity
    )
    meets_importance = per_position_totals >= importance
    motif_to_str = meets_specificity * meets_importance
    assert (
        np.sum(motif_to_str, axis=2) <= 1
    ).all()  # Ensure only one base per position
    # Make strings
    str_revstr = []
    base_map = np.array(["A", "C", "G", "T"])
    for m in motif_to_str:
        m_valid = np.sum(m, axis=1) > 0
        min_index = np.argmax(m_valid)
        max_index = m_valid.shape[0] - np.argmax(m_valid[::-1])
        motif_str_list, motif_revstr_list = [], []
        for i in range(min_index, max_index):
            pos = m[i]
            if np.sum(pos) == 0:
                motif_str_list.append("-")
                motif_revstr_list.insert(0, "-")
            else:
                base_idx = np.argmax(pos)
                base = base_map[base_idx]
                motif_str_list.append(base)
                revbase = base_map[-base_idx - 1]
                motif_revstr_list.insert(0, revbase)
        motif_str = "".join(motif_str_list)
        motif_revstr = "".join(motif_revstr_list)
        str_revstr.append((motif_str, motif_revstr))
    return str_revstr
