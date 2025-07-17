import functools

import numpy as np
import pandas as pd


##################
# MOTIF CHECKING #
##################
def single_or_many_motifs(func):
    """Decorator to handle single or many motifs.

    Functions using this decorator will always have their first argument be a motif
      stack. However, for the user, the first argument can be either a single motif or
      a motif stack. The return value will either be a single motif or a motif stack,
      matching the input. Also, this decorator assumes that any function that uses this
      decorator will return a single output that can be indexed so that output[i]
      corresponds to the output of the ith motif that was input.
    """

    @functools.wraps(func)
    def wrapper(motifs, *args, **kwargs):
        validate_motif_basic(motifs)
        if not (len(motifs.shape) in [2, 3]):
            raise ValueError(
                "Must input a single motif or motif stack as a np.ndarray."
            )
        if len(motifs.shape) == 2:
            result = func(np.expand_dims(motifs, axis=0), *args, **kwargs)
            return result[0]
        return func(motifs, *args, **kwargs)

    return wrapper


def validate_motif_basic(motifs: np.ndarray) -> None:
    """Validate that motifs are np.ndarrays."""
    if not (isinstance(motifs, np.ndarray) and (motifs.shape[-1] in [4, 8])):
        raise TypeError("Motifs must be a np.ndarray of 4 or 8 channels.")


def validate_motif_stack(motifs: np.ndarray) -> None:
    """Validate that motifs are a motif stack."""
    validate_motif_basic(motifs)
    if not len(motifs.shape) == 3:
        raise ValueError("Motif stack must be of shape (N, L, 4/8).")


def validate_motif_stack_standard(motifs: np.ndarray) -> None:
    """Validate that motifs are a standard (N, L, 4) shape."""
    validate_motif_stack(motifs)
    if not motifs.shape[2] == 4:
        raise ValueError("Motif stack must be of shape (N, L, 4).")


def validate_motif_stack_similarity(motifs: np.ndarray) -> None:
    """Validate that motifs are fit for similarity calculations."""
    validate_motif_stack(motifs)
    if not (motifs >= 0).all():
        raise ValueError("Motifs must be non-negative.")


def validate_motif_stack_compendium(motifs: np.ndarray) -> None:
    """Validate that motifs belong in a MotifCompendium."""
    validate_motif_stack_similarity(motifs)
    if not np.allclose(motifs.sum(axis=(1, 2)), 1):
        raise ValueError("Motifs must sum to 1.")


def validate_motif_stack_entropy(motifs: np.ndarray) -> None:
    """Validate that motifs are fit for entropy calculations."""
    validate_motif_stack_compendium(motifs)
    if not motifs.shape[2] == 4:
        raise ValueError("Motif stack must be of shape (N, L, 4).")


#######################
# MOTIF MANIPULATIONS #
#######################
@single_or_many_motifs
def reverse_complement(x: np.ndarray) -> np.ndarray:
    """Reverse complements motifs."""
    return x[:, ::-1, ::-1]


# 4 CHANNEL = (A, C, G, T)
# 8 CHANNEL = (A+, A-, C+, C-, G-, G+, T-, T+)


_MOTIF_4_TO_8_POS = np.zeros((4, 8))
_MOTIF_4_TO_8_POS[0, 0] = 1
_MOTIF_4_TO_8_POS[1, 2] = 1
_MOTIF_4_TO_8_POS[2, 5] = 1
_MOTIF_4_TO_8_POS[3, 7] = 1


_MOTIF_4_TO_8_NEG = np.zeros((4, 8))
_MOTIF_4_TO_8_NEG[0, 1] = 1
_MOTIF_4_TO_8_NEG[1, 3] = 1
_MOTIF_4_TO_8_NEG[2, 4] = 1
_MOTIF_4_TO_8_NEG[3, 6] = 1


@single_or_many_motifs
def motif_4_to_8(x: np.ndarray) -> np.ndarray:
    """Converts a 4 channel motif(s) into an 8 channel motif(s)."""
    if not x.shape[2] == 4:
        raise ValueError("Input motif(s) must have 4 channels.")
    x_pos = np.maximum(x, 0)
    x_neg = np.maximum(-x, 0)
    x_pos_8 = x_pos @ _MOTIF_4_TO_8_POS
    x_neg_8 = x_neg @ _MOTIF_4_TO_8_NEG
    x_8 = x_pos_8 + x_neg_8
    return x_8


@single_or_many_motifs
def motif_8_to_4_signed(x: np.ndarray) -> np.ndarray:
    """Converts an 8 channel motif(s) into a signed 4 channel motif(s)."""
    if not x.shape[2] == 8:
        raise ValueError("Input motif(s) must have 8 channels.")
    x_pos_4 = x @ _MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ _MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 - x_neg_4
    return x_4


@single_or_many_motifs
def motif_8_to_4_unsigned(x: np.ndarray) -> np.ndarray:
    """Converts an 8 channel motif(s) into an unsigned 4 channel motif(s)."""
    if not x.shape[2] == 8:
        raise ValueError("Input motif(s) must have 8 channels.")
    x_pos_4 = x @ _MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ _MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 + x_neg_4
    return x_4


_MOTIF_4_TO_COPAIR6_MASK = np.triu(np.ones(4), k=1).astype(np.bool_)


@single_or_many_motifs
def motif4_to_copair6(x: np.ndarray) -> np.ndarray:
    """Converts a 4 channel motif(s) into a 6 channel co-pair motif(s)."""
    if not (x.shape[2] == 4):
        raise ValueError("Input motif(s) must have 4 channels.")
    x_cross = x[:, :, :, np.newaxis] @ x[:, :, np.newaxis, :]  # (N, L, 4, 4)
    x_copair = x_cross[:, :, _MOTIF_4_TO_COPAIR6_MASK]  # (N, L, 6)
    return 4*x_copair # Renormalize because max value is 0.25


def align_motifs(
    motif_stack: np.ndarray, alignment_rc: np.ndarray, alignment_h: np.ndarray
) -> np.ndarray:
    """Create an aligned motif stack based on the alignment matrices.

    Uses the alignment information to place the motifs in the motif stack in the correct
      orientation and position.

    Args:
        motif_stack: A (N, L, K) motif stack to be aligned.
        alignment_rc: A (N, ) forward/reverse complement alignment vector.
        alignment_h: A (N, ) horizontal alignment vector.

    Returns:
        An aligned motif stack of shape (N, L', K), where
          L' = L + max(alignment_h, 0) - min(alignment_h, 0).
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
    return aligned_motifs


@single_or_many_motifs
def pad_motif(motif: np.ndarray, pad_to: int) -> np.ndarray:
    """Pad a motif or motif stack (by adding 0s) to a specified length.

    If the given motif is shorter than pad_to, pad with 0s until it is large enough.
      If the given motif is larger than pad_to, raise an error.

    Args:
        motif: A (L, K) motif or (N, L, K) motif stack.
        pad_to: The length to pad the motif to.

    Returns:
        A (pad_to, K) motif or (N, pad_to, K) motif stack.
    """
    validate_motif_stack(motif)
    if not (isinstance(pad_to, int) and pad_to > 0):
        raise ValueError("pad_to must be a positive integer.")
    N, L, K = motif.shape
    if L > pad_to:
        raise ValueError(
            f"Cannot pad motif of length {L} to {pad_to}."
        )
    padded_motif = np.zeros((N, pad_to, K))
    padded_motif[:, 0:L, :] = motif
    return padded_motif


def resize_motif(motif: np.ndarray, resize_to: int) -> np.ndarray:
    """Resize a motif (by squashing or padding) to a specified length.

    If the given motif is shorter than resize_to, pad with 0s until it is large enough.
      If the given motif is larger than resize_to, squash it down to a smaller motif.
      Selects the base pairs to keep that have the highest weights.

    Args:
        motif: A (L, K) motif.
        resize_to: The length to squash or pad the motif to.

    Returns:
        A (resize_to, K) motif.
    """
    validate_motif_basic(motif)
    if not len(motif.shape) == 2:
        raise ValueError("resize_motif() only resizes 2D motifs.")
    if not (isinstance(resize_to, int) and resize_to > 0):
        raise ValueError("resize_to must be a positive integer.")
    L, K = motif.shape
    if L < resize_to:
        return pad_motif(motif, pad_to=resize_to)
    elif L > resize_to:
        # Squash to the desired length
        i_sums = [
            (np.sum(np.abs(motif[i : i + resize_to, :])), i)
            for i in range(L - resize_to + 1)
        ]
        top_i = max(i_sums)[1]
        return motif[top_i : top_i + resize_to, :]
    else:
        return motif


def trim_motif(motif: np.ndarray, importance: float = 1 / 30):
    """Trim a motif by removing flanking low-importance positions.
    
    Find the leftmost and rightmost positions in the motif that have a percentage
      importance greater than the importance threshold. Return a trimmed motif that only
      includes the positions between those two positions, inclusive. The returned motif
      is not normalized. If the importance threshold is too high and the entire motif
      is trimmed, None is returned. If the importance threshold is 0, only positions
      with 0 importance are trimmed.

    Args:
        motif: A (L, K) motif.
        importance: The minimum level of importance a position must have to be included
          in the trimmed motif.
    """
    validate_motif_basic(motif)
    if not (isinstance(importance, (int, float)) and 0 <= importance <= 1):
        raise ValueError("importance must be a number in [0, 1].")
    motif_abs = np.abs(motif)
    per_position_totals = np.sum(motif_abs, axis=1)
    included_positions = per_position_totals > importance*np.sum(per_position_totals)
    if np.sum(included_positions) == 0:
        return None
    min_index = np.argmax(included_positions)
    max_index = motif.shape[0] - np.argmax(included_positions[::-1])
    return motif[min_index:max_index]


@single_or_many_motifs
def view_motif_from_position_range(
    motif: np.ndarray,
    current_min_pos: int,
    current_max_pos: int,
    new_min_pos: int,
    new_max_pos: int,
) -> np.ndarray:
    """Gets the view of the motif at a specified position range.

    Given a motif or motif stack and current position bounds, get the motif as viewed
      from a new position bound. If the new bounds are outside the current bounds, the
      view will be padded with zeros. If the new bounds are inside the current bounds,
      the view will be cropped.

    Args:
        motif: A motif or motif stack of length L.
        current_min_pos: The position of the 0th index in the length axis.
        current_max_pos: The position of the (L-1)st index in the length axis.
        new_min_pos: The new minimum position from which to view the motif.
        new_max_pos: The new maximum position from which to view the motif.

    Returns:
        The motif as viewed from a new position range.
    """
    validate_motif_stack(motif)
    if not (current_max_pos - current_min_pos) == (motif.shape[1] - 1):
        raise ValueError("Current position range must match motif length.")
    if not (new_min_pos < new_max_pos):
        raise ValueError("New position range must have a positive length.")
    # Pad if needed
    if new_min_pos < current_min_pos:
        pad_left = current_min_pos - new_min_pos
        motif = np.pad(motif, ((0, 0), (pad_left, 0), (0, 0)))
        current_min_pos = new_min_pos
    if new_max_pos > current_max_pos:
        pad_right = new_max_pos - current_max_pos
        motif = np.pad(motif, ((0, 0), (0, pad_right), (0, 0)))
        current_max_pos = new_max_pos
    # Crop out new view
    new_min_idx = new_min_pos - current_min_pos
    new_max_idx = new_max_pos - current_min_pos
    return motif[:, new_min_idx : new_max_idx + 1, :]


def average_motifs(
    motif_stack: np.ndarray,
    alignment_rc: np.ndarray,
    alignment_h: np.ndarray,
    match_original_length: bool = True,
    l1_normalize: bool = True,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the average motif of a stack of motifs.

    Calls align_motifs() to compute an aligned motif stack, then averages the aligned
      motifs. If weights are provided, a weighted average is computed. Then,

    Args:
        motif_stack: A (N, L, K) motif stack to be averaged.
        alignment_rc: A (N, ) forward/reverse complement alignment vector.
        alignment_h: A (N, ) horizontal alignment vector.
        match_original_length: Whether to match the original length of the motifs. If
          True, the average motif will be made to be the same length as the original
          motifs. If False, the average motif will have the length of the aligned motif
          stack.
        l1_normalize: Whether or not to L1 normalize the average motif before returning.
        weights: A (N, ) vector of weights for each motif. If None, all motifs are
          weighed equally.

    Returns:
        The average motif.
    """
    aligned_motifs = align_motifs(motif_stack, alignment_rc, alignment_h)
    if weights is None:
        weights = np.ones(aligned_motifs.shape[0])
    if not (
        isinstance(weights, np.ndarray)
        and weights.shape == (aligned_motifs.shape[0],)
        and (weights >= 0).all()
    ):
        raise ValueError(
            "Weights must be a non-negative vector whose length matches that of the motif stack."
        )
    average_motif = np.average(aligned_motifs, axis=0, weights=weights)
    if match_original_length:
        average_motif = resize_motif(average_motif, resize_to=motif_stack.shape[1])
    if l1_normalize:
        average_motif = average_motif / np.sum(np.abs(average_motif))
    return average_motif


####################
# PUBLIC FUNCTIONS #
####################
def motif_to_df(motif: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker."""
    validate_motif_basic(motif)
    if not (len(motif.shape) == 2 and motif.shape[1] == 4):
        raise ValueError("motif must be of shape (L, 4).")
    return pd.DataFrame(motif, columns=["A", "C", "G", "T"])


@single_or_many_motifs
def motif_to_string(
    x: np.ndarray, specificity: float = 0.7, importance: float = 1 / 30
) -> str | list[str]:
    """Transforms motifs into ATCG strings.

    Each motif is turned into an ATCG string. Not all positions are included in the
      string. The first and last positions that are position that are included are the
      first/last position that have a total importance greater than the importance
      attribute. For all included positions, if a single base has greater than a
      specificity% importance, that base is included in the string. If no base meets the
      requirement then a hyphen (-) is included in the string.

    Args:
        x: A (L, 4) motif or (N, L, 4) motif stack representing motifs to compute
          strings from.
        specificity: The percentage of importance a base must have at a position to be
          included in the string.
        importance: The minimum level of importance a position must have to be included
          in the string.

    Returns:
        A string or list of strings representing the motifs.
    """
    # Check inputs
    validate_motif_stack_standard(x)
    if not (x >= 0).all():
        raise ValueError("Motif strings can only be generated for non-negative motifs.")
    if not (isinstance(specificity, (int, float)) and 0.5 < specificity <= 1):
        raise ValueError("specificity must be a number in (0.5, 1].")
    if not (isinstance(importance, (int, float)) and 0 <= importance <= 1):
        raise ValueError("importance must be a number in [0, 1].")
    # Turns to 1s and 0s
    per_position_totals = np.sum(x, axis=2, keepdims=True)
    meets_specificity = np.divide(x, per_position_totals, out=np.zeros_like(x), where=(per_position_totals != 0)) >= specificity
    meets_importance = per_position_totals >= importance
    motif_to_str = meets_specificity * meets_importance
    assert (np.sum(motif_to_str, axis=2) <= 1).all() # Ensure only one base per position
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


@single_or_many_motifs
def motif_posneg_sum(x: np.ndarray) -> str | list[str]:
    """Classifies each motif as being positive or negative based on sum."""
    validate_motif_stack_standard(x)
    return ["pos" if np.sum(m) > 0 else "neg" for m in np.sum(x, axis=(1, 2)) > 0]


@single_or_many_motifs
def motif_posneg_max(x: np.ndarray) -> str | list[str]:
    """Classifies each motif as being positive or negative based on max value."""
    validate_motif_stack_standard(x)
    return ["pos" if np.max(m) > 0 else "neg" for m in np.sum(x, axis=(1, 2)) > 0]


def compute_motif_scalar_projection(
    project_motifs: np.ndarray, onto_motifs: np.ndarray, keepdims: bool = True
) -> np.ndarray:
    """Compute the scalar projection of one set of motifs onto another set of motifs.

    Compute the scalar projection of project_motifs onto onto_motifs. project_motifs and
      onto_motifs are expected to be motif stacks with the same number of motifs. The
      scalar projection of each pair is returned. Scalar projections are computed by
      treating each motif as a vector.

    Args:
        project_motifs: A motif stack of shape (N, L, K) to project onto onto_motifs.
        onto_motifs: A motif stack of shape (N, L, K) that project_motifs will be
          projected onto.
        keepdims: Whether or not to keep the dimensions of the scalar projection.

    Returns:
        A np.ndarray of scalar projections. If keepdims is True, the shape will be
          (N, L, K). If keepdims is False, the shape will be (N,).

    Notes:
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
) -> np.ndarray:
    """Remove the component of one set of motifs from another set of motifs.

    Given a set of motifs of interest, remove_motifs, and another set of motifs,
      main_motifs, whose component you want to remove from remove_motifs, remove the
      component of main_motifs from remove_motifs. Removal is done by subtracting the
      projection of main_motifs onto remove_motifs from remove_motifs. Alignment
      information of how to shift and reverse complement the main_motifs to align
      with remove_motifs is also required.

    Args:
        main_motifs: A (N, L, K) motif stack representing the motifs to remove from
          remove_motifs.
        remove_motifs: A (N, L, K) motif stack representing the motifs from which to
          remove the effects of main_motifs.
        alignment_rc: A (N, ) forward/reverse complement alignment vector for how the
          main_motifs align with remove_motifs. If alignment_rc[i] = 0/1, then
          main_motifs[i] must not/must be reverse complemented to align with
          remove_motifs[i].
        alignment_h: A (N, ) horizontal alignment vector for how the main_motifs
          align with remove_motifs. alignment_h[i] represents how many positions to the
          right main_motifs[i] must be shifted to align with remove_motifs[i].

    Returns:
        A (N, L, K) motif stack representing the subtracted motifs after
          main_motifs has been removed from remove_motifs.
    """
    # Check inputs
    validate_motif_stack(remove_motifs)
    validate_motif_stack(main_motifs)
    if not remove_motifs.shape == main_motifs.shape:
        raise ValueError("main_motifs and remove_motifs must have the same shape.")
    # Align remove_motifs to main_motifs
    remove_motifs_aligned = align_motifs(remove_motifs, alignment_rc, alignment_h)
    min_h = np.min(alignment_h)
    max_h = np.max(alignment_h)
    # View the aligned remove_motifs from the perspective of main_motifs
    remove_motifs_aligned = view_motif_from_position_range(
        remove_motifs_aligned,
        min_h,
        max_h+remove_motifs.shape[1]-1,
        0,
        remove_motifs.shape[1] - 1,
    )
    # Scale and subtract aligned remove_motifs
    scalar_projection = compute_motif_scalar_projection(
        main_motifs,
        remove_motifs_aligned,
        keepdims=True,
    )  # Project vector onto the main motif to match scale
    main_motifs_updated = main_motifs - scalar_projection * remove_motifs_aligned
    main_motifs_updated = np.clip(
        main_motifs_updated, a_min=0, a_max=None
    )  # Clip negative values
    return main_motifs_updated


########################
# ENTROPY CALCULATIONS #
########################
import inspect

def minusxlogx(x: np.ndarray, base: int) -> np.ndarray:
    """Compute -x*logb(x) with support in x >= 0."""
    print(base)
    return (x * np.log2(x, where=(x > 0), out=np.zeros_like(x, dtype=x.dtype)))/-np.log2(base) # Minus at end for efficiency


def normalized_last_axis_entropy(x: np.ndarray) -> np.ndarray:
    """Computes the entropy on the last axis."""
    x_normalized = x / np.sum(x, axis=-1, keepdims=True)
    return np.sum(minusxlogx(x_normalized, base=x.shape[-1]), axis=-1, keepdims=True)


@single_or_many_motifs
def ic_scale(x: np.ndarray, invert: bool = False) -> float | np.ndarray:
    """Rescale a 4 channel motif by per position information content.

    Each position in the motif is scaled by the information content at that position.
      The information content is computed as 1 - base4entropy of the per base importance
      at that position. If invert, the motif will be scaled by the inverse of the
      information content.

    Args:
        x: A (L, 4) motif or (N, L, 4) motif stack.
        invert: Whether or not to invert the information content scaling.

    Returns:
        An information content scaled (L, 4) motif or (N, L, 4) motif stack.

    Notes:
        If a position only has one base at a position, it will not change. If only two
          bases are present but are represented equally, their weights will be halved.
          And if all bases are present and represented equally, the weights will for all
          bases at that position will be set to 0.
    """
    validate_motif_stack_standard(x) # (N, L, 4)
    x_abs = np.abs(x)
    entropy = normalized_last_axis_entropy(x_abs) # (N, L, 1)
    ic = 1 - entropy
    if invert:
        scaled = np.divide(x, ic, out=np.zeros_like(x, dtype=x.dtype), where=(ic != 0))
    else:
        scaled = x * ic
    return scaled


@single_or_many_motifs
def calculate_full_motif_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the full motif entropy of a motif or motif stack.

    Computes the full motif entropy of a motif or motif stack. The full motif entropy
      is computed as an entropy across all L*K dimensions of (L, K) motifs.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The full motif entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too low, the motif is likely a single nucleotide motif and
          you may want to filter it out. You will need to tune the threshold for
          identifying these low entropy motifs for your particular setting, but you may
          want to start with a threshold of < 0.35.
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.75.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    x_fullmotif = np.reshape(x, (x.shape[0], -1))
    return normalized_last_axis_entropy(x_fullmotif)


@single_or_many_motifs
def calculate_weighted_base_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the position-weighted across-base entropy of a motif or motif stack.

    Computes the across-base entropy at each position, and takes of weighted average
      of those entropy. The entropies at each position are weighted by the total
      importance at that position.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The weighted base entropy as a float for a single motif or an array of floats
          for a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.5.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    across_base_entropy = normalized_last_axis_entropy(x)  # (N, L, 1)
    position_importance = np.sum(x, axis=2, keepdims=True) # (N, L, 1)
    return np.sum(across_base_entropy * position_importance, axis=(1, 2))  # (N, )


@single_or_many_motifs
def calculate_weighted_position_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the base-weighted across-position entropy of a motif or motif stack.

    Computes the across-position entropy for each base, and takes the weighted average
      of those entropies. The entropies of each base are weighted by the total
      importance of that position.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The weighted position entropy as a float for a single motif or an array of
          floats for a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too high, the motif is a broad, noisy motif, and you may
          want to filter it out. You will need to tune the threshold for identifying
          these high entropy motifs for your particular setting, but you may want to
          start with a threshold of > 0.71.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    across_position_entropy = np.stack([normalized_last_axis_entropy(x[:, :, i]).squeeze() for i in range(x.shape[2])], axis=1)  # (N, 4)
    base_importance = np.sum(x, axis=1) # (N, 4)
    return np.sum(across_position_entropy * base_importance, axis=1)  # (N, )


@single_or_many_motifs
def calculate_position_versus_base_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate across-position * (1 - across-base) entropy for a motif or motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      combined score of across-position entropy * (1 - across-base entropy).
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position versus base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too high, the motif is a broad, noisy motif, and you may
          want to filter it out. You will need to tune the threshold for identifying
          these high entropy motifs for your particular setting, but you may want to
          start with a threshold of > 0.45.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    across_position_entropy = normalized_last_axis_entropy(np.sum(x, axis=2)).squeeze()  # (N, )
    across_base_entropy = normalized_last_axis_entropy(np.sum(x, axis=1)).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@single_or_many_motifs
def calculate_copair_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair entropy of a motif or motif stack.

    Transforms a motif into a co-pair format where each channel represents the
      co-occurrence of two bases at the same position. Then, the copair values are
      normalized and the across-position and across-base entropies are computed. Then,
      the combined score of across-position entropy * (1 - across-base entropy) is
      computed just like in calculate_position_versus_base_entropy().

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.35.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    # Calculate copair
    x_cross = x[:, :, :, np.newaxis] @ x[:, :, np.newaxis, :]  # (N, L, 4, 4)
    copair_mask = np.triu(np.ones(x.shape[2]), k=1).astype(np.bool_)
    copair = x_cross[:, :, copair_mask] # (N, L, 6)
    copair /= np.sum(copair, axis=(1, 2), keepdims=True)  # Normalize copair
    # across-position entropy * (1 - across-base entropy) for copair
    across_position_entropy = normalized_last_axis_entropy(np.sum(copair, axis=2)).squeeze()  # (N, )
    across_base_entropy = normalized_last_axis_entropy(np.sum(copair, axis=1)).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@single_or_many_motifs
def calculate_copair_composition(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair composition of a motif or motif stack.

    Computes a measure of how much of a motif can be represented by co-occurring pairs
      of bases. This is done by computing the copair score at each position for a given
      copair, which is 2 * min(base1, base2) at that position. Then, the max copair
      score at each position is computed and summed across the length of the motif.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the composition is too high, the motif is likely noise, and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          composition motifs for your particular setting, but you may want to start with a
          threshold of > 0.41.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    # Calculate copair
    N, L, K = x.shape
    num_copairs = K*(K - 1) // 2  # Number of unique copairs for K bases
    copair_scores = np.zeros((N, L, num_copairs), dtype=x.dtype)  # (N, L, 6)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            copair_scores[:, :, idx] = 2*np.minimum(x[:, :, i], x[:, :, j])
            idx += 1
    # Overall copair composition
    return np.sum(np.max(copair_scores, axis=2), axis=1) # (N, )


@single_or_many_motifs
def calculate_dinucleotide_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide entropy of a motif or motif stack.

    Transforms a motif into a dinucleotide format where each channel represents the
      occurrence of one base at a position and another base at the next position. Then,
      the dinucleotide values are normalized and the across-position and across-base
      entropies are computed. Then, the combined score of
      across-position entropy * (1 - across-base entropy) is computed just like in
      calculate_position_versus_base_entropy().

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The dinucleotide entropy as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the entropy is too high, the motif is likely a repeat or GC content, and
          you may want to filter it out. You will need to tune the threshold for
          identifying these high entropy motifs for your particular setting, but you may
          want to start with a threshold of > 0.42.
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L/2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L/2, 4)
    x_even = x_even[:, :x_odd.shape[1], :]  # Ensure even and odd lengths match
    dinucleotide = x_even[:, :, :, np.newaxis] @ x_odd[:, :, np.newaxis, :]  # (N, L/2, 4, 4)
    dinucleotide = np.reshape(dinucleotide, (dinucleotide.shape[0], dinucleotide.shape[1], -1))  # (N, L/2, 16)
    dinucleotide /= np.sum(dinucleotide, axis=(1, 2), keepdims=True)  # Normalize dinucleotide
    # across-position entropy * (1 - across-base entropy) for dinucleotide
    across_position_entropy = normalized_last_axis_entropy(np.sum(dinucleotide, axis=2)).squeeze()  # (N, )
    across_base_entropy = normalized_last_axis_entropy(np.sum(dinucleotide, axis=1)).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@single_or_many_motifs
def calculate_dinucleotide_alternating_composition(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide composition of a motif or motif stack.

    Computes a measure of how much of a motif can be represented by an alternating
      dinucleotide sequence. This is done by computing the dinucleotide mass at each
      pair of positions, which is the sum of contributions by both bases. Then, the
      highest total importance across all possible non-repeating dinucleotides is
      returned.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The dinucleotide composition as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the composition is too high, the motif is likely a repeat or GC content,
          and you may want to filter it out. You will need to tune the threshold for
          identifying these high composition motifs for your particular setting, but you
          may want to start with a threshold of > 0.88.
        This metric does not respond to repeats that do not have a regular, even spacing
          between them like CG---CG. It is recommended to pair this metric with
          calculate_dinucleotide_score() to capture all dinucleotide patterns. You can
          be a little looser with this filter and a little stricter with
          calculate_dinucleotide_score().
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L/2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L/2, 4)
    x_even = x_even[:, :x_odd.shape[1], :]  # Ensure even and odd lengths match
    x_even_bases = np.sum(x_even, axis=1)  # (N, 4)
    x_odd_bases = np.sum(x_odd, axis=1)  # (N, 4)
    # Compute dinucleotide composition
    dinucleotide_composition = x_even_bases[:, :, np.newaxis] + x_odd_bases[:, np.newaxis, :]  # (N, 4, 4)
    dinucleotide_composition *= 1 - np.eye(x.shape[2])[np.newaxis, :, :]  # Remove diagonal (self-pairs)
    return np.max(dinucleotide_composition, axis=(1, 2))


@single_or_many_motifs
def calculate_dinucleotide_score(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide occurrence of a motif or motif stack.

    Computes a score of how much of the motif contains the same dinucleotide pair
      repeatedly. This is done by computing a dinucleotide score at each position
      as the geometric mean of the importance of one base at that position and the
      importance of the other base at the subsequent position. Then, this score is
      summed across the length of the motif. The highest dinucleotide score across all
      possible non-repeating dinucleotides is returned.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].
    
    Notes:
        When the composition is too high, the motif is likely a repeat or GC content,
          and you may want to filter it out. You will need to tune the threshold for
          identifying these high composition motifs for your particular setting, but you
          may want to start with a threshold of > 0.44.
        This metric can be overly sensitive to repeating patterns even if the entire
          motif is not repeating. It is recommended to pair this metric with
          calculate_dinucleotide_alternating_composition() to capture all dinucleotide
          patterns. You should be a little stricter with this filter and can be a
          little looser with calculate_dinucleotide_alternating_composition().
    """
    print(inspect.currentframe().f_code.co_name, x.shape)
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    x_0 = x[:, :-1, :]  # (N, L-1, 4)
    x_1 = x[:, 1:, :]  # (N, L-1, 4)
    dinucleotide_scores = np.sqrt(x_0[:, :, :, np.newaxis] @ x_1[:, :, np.newaxis, :])  # (N, L-1, 4, 4)
    dinucleotide_scores *= 1 - np.eye(x.shape[2])[np.newaxis, np.newaxis, :, :]  # Remove diagonal (self-pairs)
    return np.max(np.sum(dinucleotide_scores, axis=1), axis=(1, 2))  # (N, )



















#####################
# ALL ENTROPY TESTS #
#####################
@single_or_many_motifs
def TEST_calculate_position_over_base_entropy(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy = normalized_last_axis_entropy(np.sum(x, axis=2)).squeeze()  # (N, )
    across_base_entropy = normalized_last_axis_entropy(np.sum(x, axis=1)).squeeze()  # (N, )
    return across_position_entropy / across_base_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_2(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy = normalized_last_axis_entropy(np.sum(x, axis=2)).squeeze()  # (N, )
    across_base_entropy = normalized_last_axis_entropy(np.sum(x, axis=1)).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_3(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy_per_base = np.stack([normalized_last_axis_entropy(x[:, :, i]).squeeze() for i in range(4)], axis=1)  # (N, 4)
    across_position_entropy = np.max(across_position_entropy_per_base, axis=1)  # (N, )
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_4(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy_per_base = np.stack([normalized_last_axis_entropy(x[:, :, i]).squeeze() for i in range(4)], axis=1)  # (N, 4)
    across_position_entropy_per_base *= np.sum(x, axis=1)
    across_position_entropy = np.max(across_position_entropy_per_base, axis=1)  # (N, )
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_5(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy_per_base = np.stack([normalized_last_axis_entropy(x[:, :, i]).squeeze() for i in range(4)], axis=1)  # (N, 4)
    across_position_entropy_per_base *= np.sum(x, axis=1)
    across_position_entropy = np.sum(across_position_entropy_per_base, axis=1)  # (N, )
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_6(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_position_entropy_per_base = np.stack([normalized_last_axis_entropy(x[:, :, i]).squeeze() for i in range(4)], axis=1)  # (N, 4)
    across_position_entropy = np.sum(across_position_entropy_per_base, axis=1)  # (N, ) # JUST SUM
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_7(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_bases = []
    for i in range(4):
        x_i = x[:, :, i].copy()
        x_i /= np.sum(x_i, axis=1, keepdims=True)  # Normalize across bases
        across_bases.append(normalized_last_axis_entropy(x_i).squeeze())
    across_position_entropy = np.stack(across_bases, axis=1)  # (N, 4)
    across_position_entropy *= np.sum(x, axis=1)
    across_position_entropy = np.sum(across_position_entropy, axis=1)  # (N, )
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_position_over_base_entropy_8(x: np.array) -> float | np.ndarray:
    """Calculate the across-position/across-base entropy ratio of a motif/motif stack.

    Computes the across-position entropy of each motif by summing the importance across
      all bases at each position. Then, computes the across-base entropy of each motif
      by summing the importance across all positions for each base. Then, computes the
      ratio of the two entropies.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The position over base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, inf).
    """
    validate_motif_stack_entropy(x)
    across_bases = []
    for i in range(4):
        x_i = x[:, :, i].copy()
        x_i /= np.sum(x_i, axis=1, keepdims=True)  # Normalize across bases
        across_bases.append(normalized_last_axis_entropy(x_i).squeeze())
    across_position_entropy = np.stack(across_bases, axis=1)  # (N, 4)
    across_position_entropy = np.sum(across_position_entropy, axis=1)  # (N, ) JUST SUM
    return across_position_entropy


@single_or_many_motifs
def TEST_calculate_copair_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair entropy of a motif/motif stack.

    Transforms a motif into a co-pair format where each channel represents the
      co-occurrence of two bases at the same position. The max copair value for a given
      position and base is 1. Then, the max copair value per position is computed. Then,
      a weighted average of these copair values and per-position importance is computed
      to give a bounded [0, 1] measure of copair occurrence.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    x_normalized = x / np.sum(x, axis=2, keepdims=True)  # Normalize across bases
    x_copair = motif4_to_copair6(x_normalized)
    x_copair_score = np.max(x_copair, axis=2) # (N, L)
    position_importance = np.sum(x, axis=2) # (N, L)
    return np.sum(x_copair_score * position_importance, axis=1)


@single_or_many_motifs
def TEST_calculate_copair_entropy_2(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair entropy of a motif/motif stack.

    Transforms a motif into a co-pair format where each channel represents the
      co-occurrence of two bases at the same position. The max copair value for a given
      position and base is 1. Then, the max copair value per position is computed. Then,
      a weighted average of these copair values and per-position importance is computed
      to give a bounded [0, 1] measure of copair occurrence.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    N, L, _ = x.shape
    copair_scores = np.zeros((N, L, 6), dtype=x.dtype)  # (N, L, 6)
    idx = 0
    for i in range(4):
        for j in range(i + 1, 4):
            copair_scores[:, :, idx] = 2*np.minimum(x[:, :, i], x[:, :, j])
            idx += 1
    copair_scores_max = np.max(np.sum(copair_scores, axis=1), axis=1) # (N, ) find worst copair channel
    return copair_scores_max


@single_or_many_motifs
def TEST_calculate_copair_entropy_3(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair entropy of a motif/motif stack.

    Transforms a motif into a co-pair format where each channel represents the
      co-occurrence of two bases at the same position. The max copair value for a given
      position and base is 1. Then, the max copair value per position is computed. Then,
      a weighted average of these copair values and per-position importance is computed
      to give a bounded [0, 1] measure of copair occurrence.
    
    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
    
    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    N, L, _ = x.shape
    copair_scores = np.zeros((N, L, 6), dtype=x.dtype)  # (N, L, 6)
    idx = 0
    for i in range(4):
        for j in range(i + 1, 4):
            copair_scores[:, :, idx] = 2*np.minimum(x[:, :, i], x[:, :, j])
            idx += 1
    copair_scores_sum = np.sum(np.max(copair_scores, axis=2), axis=1) # (N, )
    return copair_scores_sum


@single_or_many_motifs
def TEST_calculate_dinuc_score(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L//2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L//2, 4)
    x_even_bases = np.sum(x_even, axis=1)  # (N, 4)
    x_odd_bases = np.sum(x_odd, axis=1)  # (N, 4)
    dinuc_scores = x_even_bases[:, :, np.newaxis] + x_odd_bases[:, np.newaxis, :]  # (N, 4, 4)
    dinuc_scores *= 1 - np.eye(4)[np.newaxis, :, :]  # Remove diagonal (self-pairs)
    return np.max(dinuc_scores, axis=(1, 2))


@single_or_many_motifs
def TEST_calculate_dinuc_score_2(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L//2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L//2, 4)
    x_even_bases = np.sum(x_even, axis=1)  # (N, 4)
    x_odd_bases = np.sum(x_odd, axis=1)  # (N, 4)
    dinuc_scores = x_even_bases[:, :, np.newaxis] + x_odd_bases[:, np.newaxis, :]  # (N, 4, 4)
    dinuc_scores *= 1 - np.eye(4)[np.newaxis, :, :]  # Remove diagonal (self-pairs)
    dinuc_scores = np.max(dinuc_scores, axis=(1, 2))  # (N, )
    positional_entropy = normalized_last_axis_entropy(np.sum(x, axis=2)).squeeze()  # (N, )
    return dinuc_scores * positional_entropy # high dinuc score + high positional entropy


@single_or_many_motifs
def TEST_calculate_dinuc_score_3(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    x_0 = x[:, 1:, :]  # (N, L-1, 4)
    x_1 = x[:, :-1, :]  # (N, L-1, 4)
    x_sum = (x_0[:, :, :, np.newaxis] + x_1[:, :, np.newaxis, :])/2  # (N, L-1, 4, 4)
    dinuc_scores = np.sum(x_sum, axis=1)  # (N, 4, 4)
    dinuc_scores *= 1 - np.eye(4)[np.newaxis, :, :]  # Remove diagonal (self-pairs)
    return np.max(dinuc_scores, axis=(1, 2))  # (N, )


@single_or_many_motifs
def TEST_calculate_dinuc_score_4(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    x_0 = x[:, 1:, :]  # (N, L-1, 4)
    x_1 = x[:, :-1, :]  # (N, L-1, 4)
    dinuc_scores = np.zeros((x_0.shape[0], x_0.shape[1], 4, 4), dtype=x_0.dtype)  # (N, L-1, 4, 4)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            dinuc_scores[:, :, i, j] = (x_0[:, :, i] + x_1[:, :, j]) * (x_0[:, :, i] > 0) * (x_1[:, :, j] > 0)  # (N, L-1)
    dinuc_scores = np.sum(dinuc_scores, axis=1)  # (N, 4, 4)
    return np.max(dinuc_scores, axis=(1, 2))  # (N, )


@single_or_many_motifs
def TEST_calculate_dinuc_score_5(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    x_0 = x[:, 1:, :]  # (N, L-1, 4)
    x_1 = x[:, :-1, :]  # (N, L-1, 4)
    dinuc_scores = np.zeros((x_0.shape[0], x_0.shape[1], 4, 4), dtype=x_0.dtype)  # (N, L-1, 4, 4)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            dinuc_scores[:, :, i, j] = np.sqrt(x_0[:, :, i] * x_1[:, :, j])
    dinuc_scores = np.sum(dinuc_scores, axis=1)  # (N, 4, 4)
    return np.max(dinuc_scores, axis=(1, 2))  # (N, )


@single_or_many_motifs
def TEST_calculate_trinuc_score(x: np.ndarray) -> float | np.ndarray:
    """Calculate the dinucleotide score of a motif/motif stack.

    The dinucleotide score is a measure of how well the motif can be represented by a
      repeating dinucleotide pattern.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].
    """
    validate_motif_stack_entropy(x)
    L = x.shape[1]
    idxs_0 = np.arange(0, L, 3)
    idxs_1 = np.arange(1, L, 3)
    idxs_2 = np.arange(2, L, 3)
    x_0 = x[:, idxs_0, :]  # (N, L//3, 4)
    x_1 = x[:, idxs_1, :]  # (N, L//3, 4)
    x_2 = x[:, idxs_2, :]  # (N, L//3, 4)
    x_0_bases = np.sum(x_0, axis=1)  # (N, 4)
    x_1_bases = np.sum(x_1, axis=1)  # (N, 4)
    x_2_bases = np.sum(x_2, axis=1)  # (N, 4)
    trinuc_scores = x_0_bases[:, :, np.newaxis, np.newaxis] + x_1_bases[:, np.newaxis, :, np.newaxis] + x_2_bases[:, np.newaxis, np.newaxis, :]  # (N, 4, 4, 4)
    for i in range(4):
        trinuc_scores[:, i, i, i] = 0  # Remove diagonal (self-trios)
    return np.max(trinuc_scores, axis=(1, 2, 3))


@single_or_many_motifs
def TEST_motif_nonuniformity_score(x: np.ndarray) -> float | np.ndarray:
    """Calculate the non-uniformity score of a motif/motif stack.

    The non-uniformity score is a measure of how uniformly distributed the bases are at
      each position in the motif. A higher score indicates a more non-uniform distribution.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The non-uniformity score as a float for a single motif or an array of floats for a
          motif stack.
    """
    validate_motif_stack_entropy(x)
    per_position_totals = np.sum(x, axis=2) # (N, L)
    L = per_position_totals.shape[1]
    return np.sum(np.abs(per_position_totals - 1/L), axis=1) / (2*(L-1)/L) # Normalize by max non-uniformity score of 1


@single_or_many_motifs
def TEST_motif_spikiness(x: np.ndarray) -> float | np.ndarray:
    """Calculate the spikiness score of a motif/motif stack.

    The spikiness score is a measure of how much the base distribution at each position
    deviates from a uniform distribution. A higher score indicates a more "spiky"
    distribution.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The spikiness score as a float for a single motif or an array of floats for a
          motif stack.
    """
    validate_motif_stack_entropy(x)
    per_position_totals = np.sum(x, axis=2) # (N, L)
    deltas = per_position_totals[:, 1:] - per_position_totals[:, :-1]  # (N, L-1)
    return np.sum(np.abs(deltas), axis=1)


@single_or_many_motifs
def TEST_motif_spikiness2(x: np.ndarray) -> float | np.ndarray:
    """Calculate the spikiness score of a motif/motif stack.

    The spikiness score is a measure of how much the base distribution at each position
    deviates from a uniform distribution. A higher score indicates a more "spiky"
    distribution.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The spikiness score as a float for a single motif or an array of floats for a
          motif stack.
    """
    validate_motif_stack_entropy(x)
    per_position_totals = np.sum(x, axis=2) # (N, L)
    normalized = per_position_totals/np.max(per_position_totals, axis=1, keepdims=True)  # Normalize to [0, 1]
    scores = normalized*(1 - normalized)
    return np.sum(scores, axis=1)



###############
# OLD ENTROPY #
###############
import math


def valid_2d_l1norm_motif(motif: np.ndarray) -> np.ndarray:
    """Check that motif is a valid, positive, 2D, L1-normalized motif."""
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 8:
        motif = motif_8_to_4_unsigned(motif)  # Convert motif to 8-channel
    if np.any(motif < 0):
        motif = np.abs(motif) # Motif must be positive
    
    # L1-normalize, into a probability
    motif = motif / np.sum(motif)
    return motif


def norm_shannon_entropy(prob_array: np.array, axis: int | None = None) -> float:
    """Normalized Shannon entropy, normalized by the number of discrete states 
    to range between [0,1], along a given axis.
    
    Args:
        prob_array: (L, 4) motif (np.ndarray)
        axis: Axis along which to calculate the entropy. Default is None, which
            calculates the entropy across all axes.
    
    Returns:
        norm_entropy: float
    """
    # Normalize, along axis
    prob_array = prob_array / np.sum(prob_array, axis=axis, keepdims=True)
    axis_size = prob_array.size if axis is None else prob_array.shape[axis]

    # Calculate normalized Shannon entropy, normalized by number of possible states [0,1]
    norm_entropy = (-np.sum(prob_array * np.log2(np.where(prob_array == 0, 1, prob_array)), axis=axis)
        / np.log2(axis_size))
    return norm_entropy


def motif8_to_copair28(motif8: np.array) -> np.array:
    """Expand base channel (e.g., 8-channel: A+,C+,G+,T+,A-,C-,G-,T-)
    to co-occurrrence of all, non-repeating dinucleotide pairs per position
    (e.g., 28-channel combinations)."""
    # Current dimensions: motif8
    rows, cols = motif8.shape
    nuc = 2  # Co-occurring pair

    # New dimensions
    new_cols = int(
        math.factorial(cols)
        / (math.factorial(cols - nuc) * math.factorial(nuc))
    )  # C(n,r)
    copair28 = np.zeros((rows, new_cols))

    ## Matrix multiplication + mask
    mask = np.tril(np.ones(cols), k=-1).astype(
        bool
    )  # Mask for bottom left off-diagonal half, to exclude self-repeats

    for i in range(rows):
        copair_perm = np.outer(motif8[i, :], motif8[i, :])  # Permutations: With order
        copair28[i, :] = copair_perm[
            mask
        ]  # Combinations: Without order; Exclude self (e.g., AA,CC,GG,TT)
    return copair28


def motif8_to_dinuc64(motif8: np.array) -> np.array:
    """Transform (L-position channel, 8-base channel) motif
    to (L/2-position channel, 64-base channel), evaluating two positions at a time,
    for all A+,C+,G+,T+,A-,C-,G-,T- dinucleotide pair permutations: 64 permutations."""
    # Current dimensions: motif8
    rows, cols = motif8.shape

    # Dimensions: Dinuc64
    new_rows = rows // 2  # Drop final position if odd length
    new_cols = cols**2  # Include self-repeats
    dinuc64 = np.zeros((new_rows, new_cols))

    # Calculate dinucleotide pairs, as product of distributions
    # mask = ~np.eye(cols, dtype=bool) # Include self-repeats, if commented out

    for i in range(new_rows):
        dinuc_pair = np.outer(motif8[2 * i, :], motif8[2 * i + 1, :])
        dinuc64[i, :] = dinuc_pair.flatten()
    return dinuc64


def calculate_motif_entropy_old(MOTIFS: np.array) -> float:
    """Calculate normalized Shannon entropy of motif, across all positions and bases.

    Calculation: Shannon entropy on (L, 4)
    Purpose:    (High) Archetype: Noise/chaos
                (Low) Archetype: Sharp nucleotide peak (e.g., G)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        motif_entropy: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate Shannon entropy
        motif_entropy = norm_shannon_entropy(motif)
        result.append(motif_entropy)
    return result


def calculate_weighted_base_entropy_old(MOTIFS: np.array) -> float:
    """Calculate information-weighted base-wise entropy.
    
    Calculation: Weighted average of per-base entropy, across position
    For motif_i,j (L, 4)
      = Sum across position_i [ Relative information of position_i * Entropy of position_i ]
      = Sum across position_i [ Sum across base_j (motif_i,j) /  Sum_i,j (motif_i,j)
        * Normalized Shannon entropy (L1-normalized motif_i) ]
      = Sum across position_i [ Sum across base_j (motif_i,j) /  Sum_i,j (motif_i,j)
        * Normalized Shannon entropy ( motif_i / sum_i (motif_i) )
    
    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)
    
    Returns:
        weighted_base_entropy: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate weigthed base entropy: weight * norm entropy
        weighted_base_entropy = np.sum(np.sum(motif, axis=1) * norm_shannon_entropy(motif, axis=1)) / np.sum(motif)
        result.append(weighted_base_entropy)
    return result


def calculate_posbase_entropy_ratio_old(MOTIFS: np.array) -> float:
    """Calculate ratio of position-wise entropy / base_wise_entropy.

    Calculation: Entropy across position (L,) / Entropy across base (8,)
    Purpose: (High) Archetype: Single nucleotide repeats (e.g., AAAAA, GGGGG)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        posbase_entropy_ratio: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Positional, base-wise probability
        pos_prob = np.sum(motif, axis=1) # Sum across bases (L,)
        base_prob = np.sum(motif, axis=0)  # Sum across positions (,8)

        # Calculate position-wise, base-wise entropy
        pos_entropy = norm_shannon_entropy(pos_prob)
        base_entropy = norm_shannon_entropy(base_prob)

        posbase_entropy_ratio = pos_entropy / base_entropy
        result.append(posbase_entropy_ratio)
    return result


def calculate_posbase_entropy_ratio_old_2(MOTIFS: np.array) -> float:
    """Calculate ratio of position-wise entropy / base_wise_entropy.

    Calculation: Entropy across position (L,) / Entropy across base (8,)
    Purpose: (High) Archetype: Single nucleotide repeats (e.g., AAAAA, GGGGG)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        posbase_entropy_ratio: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Positional, base-wise probability
        pos_prob = np.sum(motif, axis=1) # Sum across bases (L,)
        base_prob = np.sum(motif, axis=0)  # Sum across positions (,8)

        # Calculate position-wise, base-wise entropy
        pos_entropy = norm_shannon_entropy(pos_prob)
        base_entropy = norm_shannon_entropy(base_prob)

        posbase_entropy_ratio = pos_entropy * (1 - base_entropy)
        result.append(posbase_entropy_ratio)
    return result


def calculate_copair_entropy_ratio_old(MOTIFS: np.array) -> float:
    """Calculate ratio of position-wise entropy / base co-occurrence pair entropy.

    Calculation: Entropy across position (L,) /
        Entropy across all pairs of co-occurring, non-repeating bases (28,)
    Purpose: (High) Archetype: High GC, AT bias

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        copair_entropy_ratio: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate joint distribution of all non-repeating, non-ordered pairs of bases, normalized
        copair28 = motif8_to_copair28(motif)
        copair28_prob = copair28 / np.sum(copair28)

        # Positional, base-wise probability
        copair_pos_prob = np.sum(copair28_prob, axis=1)  # Sum across bases (L,)
        copair_base_prob = np.sum(copair28_prob, axis=0)  # Sum across positions (,8)

        # Calculate position-wise, base-wise entropy
        copair_pos_entropy = norm_shannon_entropy(copair_pos_prob)
        copair_base_entropy = norm_shannon_entropy(copair_base_prob)

        copair_entropy_ratio = copair_pos_entropy / copair_base_entropy
        result.append(copair_entropy_ratio)
    return result


def calculate_copair_entropy_ratio_old2(MOTIFS: np.array) -> float:
    """Calculate ratio of position-wise entropy / base co-occurrence pair entropy.

    Calculation: Entropy across position (L,) /
        Entropy across all pairs of co-occurring, non-repeating bases (28,)
    Purpose: (High) Archetype: High GC, AT bias

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        copair_entropy_ratio: float
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate joint distribution of all non-repeating, non-ordered pairs of bases, normalized
        copair28 = motif8_to_copair28(motif)
        copair28_prob = copair28 / np.sum(copair28)

        # Positional, base-wise probability
        copair_pos_prob = np.sum(copair28_prob, axis=1)  # Sum across bases (L,)
        copair_base_prob = np.sum(copair28_prob, axis=0)  # Sum across positions (,8)

        # Calculate position-wise, base-wise entropy
        copair_pos_entropy = norm_shannon_entropy(copair_pos_prob)
        copair_base_entropy = norm_shannon_entropy(copair_base_prob)

        copair_entropy_ratio = copair_pos_entropy * (1 - copair_base_entropy)
        result.append(copair_entropy_ratio)
    return result


def calculate_dinuc_entropy_ratio_old(MOTIFS: np.array) -> float:
    """Calculate ratio of two-position entropy / two-base entropy.
    Calculation: Entropy across pairs of positions (L/2,) /
        Entropy across all dinucleotide pairs (64,)
    Purpose: (High) Archetype: Dinucleotide repeats (e.g., GCGCGC, ATATAT)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        dinuc_entropy_ratio: Calculated Shannon entropy (float)
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate distribution of all dinucleotide pairs of bases, per 2 positions, normalized
        dinuc64 = motif8_to_dinuc64(motif)
        dinuc64_prob = dinuc64 / np.sum(dinuc64)

        # Positional, base-wise probability
        dinuc_pos_prob = np.sum(dinuc64_prob, axis=1) # Sum across bases (L/2,)
        dinuc_base_prob = np.sum(dinuc64_prob, axis=0) # Sum across positions (,64)

        # Calculate position-wise, base-wise entropy
        dinuc_pos_entropy = norm_shannon_entropy(dinuc_pos_prob)
        dinuc_base_entropy = norm_shannon_entropy(dinuc_base_prob)

        dinuc_entropy_ratio = dinuc_pos_entropy / dinuc_base_entropy
        result.append(dinuc_entropy_ratio)
    return result


def calculate_dinuc_entropy_ratio_old2(MOTIFS: np.array) -> float:
    """Calculate ratio of two-position entropy / two-base entropy.
    Calculation: Entropy across pairs of positions (L/2,) /
        Entropy across all dinucleotide pairs (64,)
    Purpose: (High) Archetype: Dinucleotide repeats (e.g., GCGCGC, ATATAT)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        dinuc_entropy_ratio: Calculated Shannon entropy (float)
    """
    result = []
    for motif in MOTIFS:
        # Check and normalize motif
        motif = valid_2d_l1norm_motif(motif)

        # Calculate distribution of all dinucleotide pairs of bases, per 2 positions, normalized
        dinuc64 = motif8_to_dinuc64(motif)
        dinuc64_prob = dinuc64 / np.sum(dinuc64)

        # Positional, base-wise probability
        dinuc_pos_prob = np.sum(dinuc64_prob, axis=1) # Sum across bases (L/2,)
        dinuc_base_prob = np.sum(dinuc64_prob, axis=0) # Sum across positions (,64)

        # Calculate position-wise, base-wise entropy
        dinuc_pos_entropy = norm_shannon_entropy(dinuc_pos_prob)
        dinuc_base_entropy = norm_shannon_entropy(dinuc_base_prob)

        dinuc_entropy_ratio = dinuc_pos_entropy * (1 - dinuc_base_entropy)
        result.append(dinuc_entropy_ratio)
    return result