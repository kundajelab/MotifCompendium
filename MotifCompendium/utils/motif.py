import functools

import numpy as np
import pandas as pd

import MotifCompendium.utils.config as utils_config

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


def calculate_metrics(func):
    """Decorator to handle trimming and serial execution of motif metrics.

    Wraps functions decorated with @single_or_many_motifs. Adds trim and serial
    control: trims motif flanks before passing to func, and optionally serializes
    execution over a motif stack.
    """
    decorated = single_or_many_motifs(func)

    @functools.wraps(func)
    def wrapper(motifs: np.ndarray, 
            trim_importance: float | int | None = None,
            trim_length: int | None = None,
            *args, **kwargs
        ):
        validate_motif_basic(motifs)
        if motifs.ndim not in (2, 3):
            raise ValueError("Must input a single motif or motif stack as a np.ndarray.")

        # No trim
        if trim_importance is None and trim_length is None:
            return decorated(motifs, *args, **kwargs)

        # Single: 2D motif
        single = motifs.ndim == 2
        if single:
            motifs = np.expand_dims(motifs, axis=0)
        # Trim, normalize, and serialize
        results = np.stack([
            decorated(
                l1_norm_motif(
                    trim_motif(
                        motif=motifs[i],
                        importance=trim_importance,
                        length=trim_length,
                    )
                ), *args, **kwargs)
            for i in range(motifs.shape[0])
        ], axis=0)
        return results[0] if single else results

    return wrapper


def validate_motif_basic(motifs: np.ndarray) -> None:
    """Validate that motifs are np.ndarrays with with a last channel size of 4."""
    if not (isinstance(motifs, np.ndarray) and (motifs.shape[-1] == 4)):
        raise TypeError("Motifs must be a np.ndarray of 4 channels.")

def validate_motif_single(motifs: np.ndarray) -> None:
    """Validate that motif is a single 2D np.ndarray with a last channel size of 4."""
    validate_motif_basic(motifs)
    if not len(motifs.shape) == 2:
        raise ValueError("Motif must be a single 2D np.ndarray of shape (L, 4).")

def validate_motif_stack(motifs: np.ndarray) -> None:
    """Validate that motifs are a motif stack."""
    validate_motif_basic(motifs)
    if not len(motifs.shape) == 3:
        raise ValueError("Motif stack must be of shape (N, L, 4).")


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


def validate_motif_stack_l1(motifs: np.ndarray) -> None:
    """Validate that motifs are L1 normalized (per motif), summing to 1."""
    validate_motif_stack_similarity(motifs)
    if not np.allclose(motifs.sum(axis=(1, 2)), 1):
        raise ValueError("Motifs must sum to 1.")


def validate_motif_stack_entropy(motifs: np.ndarray) -> None:
    """Validate that motifs are fit for entropy calculations."""
    validate_motif_stack_l1(motifs)
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
        raise ValueError(f"Cannot pad motif of length {L} to {pad_to}.")
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


def trim_motif(
        motif: np.ndarray,
        importance: float | None = 0,
        length: int | None = None
    ) -> np.ndarray | None:
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
        length: The desired length of the trimmed motif. If None, the length is determined
          by the importance threshold. The window of this length with the highest total
          absolute contribution is returned.
    """
    validate_motif_single(motif)
    if (importance is not None) and (length is not None):
        raise ValueError("Cannot specify both importance and length.")
    if importance is not None and not (isinstance(importance, (int, float)) and 0 <= importance <= 1):
        raise ValueError("importance must be a number in [0, 1].")
    if length is not None and not (isinstance(length, int) and length > 0):
        raise ValueError("length must be a positive integer.")
    if importance is None and length is None:
        raise ValueError("Must specify either importance or length.")

    motif_abs = np.abs(motif)
    per_position_totals = np.sum(motif_abs, axis=1)

    if length is not None:
        if length > motif.shape[0]:
            raise ValueError(f"length ({length}) exceeds motif length ({motif.shape[0]}).")
        window_totals = np.convolve(per_position_totals, np.ones(length, dtype=float), mode='valid')
        best_start = int(np.argmax(window_totals))
        return motif[best_start:best_start + length]

    if importance is not None:
        included_positions = per_position_totals > importance * np.sum(per_position_totals)
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

    Returns:
        The average motif.
    """
    validate_motif_stack(motif_stack)
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
    # Convert to 4-channel, to ensure +/- channels align
    if aligned_motifs.shape[2] == 8:
        aligned_motifs = motif_8_to_4_signed(aligned_motifs)
    average_motif = np.average(aligned_motifs, axis=0, weights=weights)
    if motif_stack.shape[2] == 8:
        average_motif = motif_4_to_8(average_motif)
    if match_original_length:
        average_motif = resize_motif(average_motif, resize_to=motif_stack.shape[1])
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
    meets_specificity = (
        np.divide(
            x,
            per_position_totals,
            out=np.zeros_like(x),
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
        raise ValueError(f"main_motifs and remove_motifs must have the same shape: {main_motifs.shape} vs {remove_motifs.shape}.")
    # Align remove_motifs to main_motifs
    remove_motifs_aligned = align_motifs(remove_motifs, alignment_rc, alignment_h)
    min_h = np.min(alignment_h)
    max_h = np.max(alignment_h)
    # View the aligned remove_motifs from the perspective of main_motifs
    remove_motifs_aligned = view_motif_from_position_range(
        remove_motifs_aligned,
        min_h,
        max_h + remove_motifs.shape[1] - 1,
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


@single_or_many_motifs
def l1_norm_motif(motifs: np.ndarray) -> np.ndarray:
    """L1 normalize by motif.

    Each motif is normalized such that the sum of the absolute values of all elements
      in the motif is equal to 1.

    Args:
        x: A (L, K) motif or (N, L, 4) motif stack.

    Returns:
        The L1 normalized motif or motif stack.
    """
    # L1 normalize by motif
    norm = np.sum(motifs, axis=(1, 2), keepdims=True)
    motifs = np.divide(
        motifs, norm, out=np.zeros_like(motifs, dtype=motifs.dtype), where=norm!=0
    )
    return motifs


@single_or_many_motifs
def l1_norm_position(motifs: np.ndarray) -> np.ndarray:
    """L1 normalize by position.

    Each motif is normalized such that the sum of the absolute values of all elements
      in the motif is equal to 1.

    Args:
        x: A (L, K) motif or (N, L, 4) motif stack.

    Returns:
        The L1 normalized motif or motif stack.
    """
    # L1 normalize by position
    norm = np.sum(motifs, axis=(-1), keepdims=True)
    motifs = np.divide(
        motifs, norm, out=np.zeros_like(motifs, dtype=motifs.dtype), where=norm!=0
    )
    return motifs


def minusxlogx(x: np.ndarray, base: int) -> np.ndarray:
    """Compute -x*logb(x) with support in x >= 0.
    Args:
        x: A np.ndarray of values to compute -x*logb(x) for. Must be non-negative.
        base: The base of the logarithm.

    Returns:
        A np.ndarray of the same shape as x containing the values of -x*logb(x).
    """
    return (
        x * np.log2(x, where=(x > 0), out=np.zeros_like(x, dtype=x.dtype))
    ) / -np.log2(
        base
    )  # Minus at end for efficiency


def normalized_last_axis_entropy(x: np.ndarray) -> np.ndarray:
    """Computes the entropy on the last axis assuming that x >= 0.
    Args:
        x: A (N, L, K) motif stack.

    Returns:
        The entropy of the last axis, normalized to be in [0, 1] by dividing by log2(K).
        (N, L, 1) array of entropies.
    """
    x_sum = np.sum(x, axis=-1, keepdims=True)
    x_normalized = np.divide(
        x, x_sum, out=np.zeros_like(x, dtype=x.dtype), where=(x_sum != 0)
    )
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
    validate_motif_stack_standard(x)  # (N, L, 4)
    x_abs = np.abs(x)
    entropy = normalized_last_axis_entropy(x_abs)  # (N, L, 1)
    ic = 1 - entropy
    if invert:
        scaled = np.divide(x, ic, out=np.zeros_like(x, dtype=x.dtype), where=(ic != 0))
    else:
        scaled = x * ic
    return scaled


###################
# ENTROPY METRICS #
###################
@calculate_metrics
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
    validate_motif_stack_entropy(x)
    x_fullmotif = np.reshape(x, (x.shape[0], -1))
    return normalized_last_axis_entropy(x_fullmotif).squeeze(axis=-1)


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    across_base_entropy = normalized_last_axis_entropy(x)  # (N, L, 1)
    position_importance = np.sum(x, axis=2, keepdims=True)  # (N, L, 1)
    return np.sum(across_base_entropy * position_importance, axis=(1, 2))  # (N, )


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    across_position_entropy = np.stack(
        [normalized_last_axis_entropy(x[:, :, i]).squeeze(axis=-1) for i in range(x.shape[2])],
        axis=1,
    )  # (N, 4)
    base_importance = np.sum(x, axis=1)  # (N, 4)
    return np.sum(across_position_entropy * base_importance, axis=1)  # (N, )


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    across_position_entropy = normalized_last_axis_entropy(
        np.sum(x, axis=2)
    ).squeeze(axis=-1)  # (N, )
    across_base_entropy = normalized_last_axis_entropy(
        np.sum(x, axis=1)
    ).squeeze(axis=-1)  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    # Calculate copair
    x_cross = x[:, :, :, np.newaxis] @ x[:, :, np.newaxis, :]  # (N, L, 4, 4)
    copair_mask = np.triu(np.ones(x.shape[2]), k=1).astype(np.bool_)
    copair = x_cross[:, :, copair_mask]  # (N, L, 6)
    copair /= np.sum(copair, axis=(1, 2), keepdims=True)  # Normalize copair
    # across-position entropy * (1 - across-base entropy) for copair
    across_position_entropy = normalized_last_axis_entropy(
        np.sum(copair, axis=2)
    ).squeeze(axis=-1)  # (N, )
    across_base_entropy = normalized_last_axis_entropy(
        np.sum(copair, axis=1)
    ).squeeze(axis=-1)  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    # Calculate copair
    N, L, K = x.shape
    num_copairs = K * (K - 1) // 2  # Number of unique copairs for K bases
    copair_scores = np.zeros((N, L, num_copairs), dtype=x.dtype)  # (N, L, 6)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            copair_scores[:, :, idx] = 2 * np.minimum(x[:, :, i], x[:, :, j])
            idx += 1
    # Overall copair composition
    return np.sum(np.max(copair_scores, axis=2), axis=1)  # (N, )


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L/2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L/2, 4)
    x_even = x_even[:, : x_odd.shape[1], :]  # Ensure even and odd lengths match
    dinucleotide = (
        x_even[:, :, :, np.newaxis] @ x_odd[:, :, np.newaxis, :]
    )  # (N, L/2, 4, 4)
    dinucleotide = np.reshape(
        dinucleotide, (dinucleotide.shape[0], dinucleotide.shape[1], -1)
    )  # (N, L/2, 16)
    dinucleotide /= np.sum(
        dinucleotide, axis=(1, 2), keepdims=True
    )  # Normalize dinucleotide
    # across-position entropy * (1 - across-base entropy) for dinucleotide
    across_position_entropy = normalized_last_axis_entropy(
        np.sum(dinucleotide, axis=2)
    ).squeeze(axis=-1)  # (N, )
    across_base_entropy = normalized_last_axis_entropy(
        np.sum(dinucleotide, axis=1)
    ).squeeze(axis=-1)  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    L = x.shape[1]
    even_idxs = np.arange(0, L, 2)
    odd_idxs = np.arange(1, L, 2)
    x_even = x[:, even_idxs, :]  # (N, L/2, 4)
    x_odd = x[:, odd_idxs, :]  # (N, L/2, 4)
    x_even = x_even[:, : x_odd.shape[1], :]  # Ensure even and odd lengths match
    x_even_bases = np.sum(x_even, axis=1)  # (N, 4)
    x_odd_bases = np.sum(x_odd, axis=1)  # (N, 4)
    # Compute dinucleotide composition
    dinucleotide_composition = (
        x_even_bases[:, :, np.newaxis] + x_odd_bases[:, np.newaxis, :]
    )  # (N, 4, 4)
    dinucleotide_composition *= (
        1 - np.eye(x.shape[2])[np.newaxis, :, :]
    )  # Remove diagonal (self-pairs)
    return np.max(dinucleotide_composition, axis=(1, 2))


@calculate_metrics
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
    validate_motif_stack_entropy(x)
    # Calculate dinucleotide
    x_0 = x[:, :-1, :]  # (N, L-1, 4)
    x_1 = x[:, 1:, :]  # (N, L-1, 4)
    dinucleotide_scores = np.sqrt(
        x_0[:, :, :, np.newaxis] @ x_1[:, :, np.newaxis, :]
    )  # (N, L-1, 4, 4)
    dinucleotide_scores *= (
        1 - np.eye(x.shape[2])[np.newaxis, np.newaxis, :, :]
    )  # Remove diagonal (self-pairs)
    return np.max(np.sum(dinucleotide_scores, axis=1), axis=(1, 2))  # (N, )

@calculate_metrics
def calculate_truncated(x: np.ndarray, threshold: float = 0.1) -> bool | np.ndarray:
    """Calculate whether a motif or motif stack is truncated.

    A motif is classified as truncated if the max peak position is in the first 10%
      or last 10% of the motif. The max peak position is the position with the highest
      absolute importance, summed across all bases.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.
        threshold: Fraction of the motif length to consider as the start and end of the
          motif for classifying truncation.
    Returns:
        A boolean for a single motif or an array of booleans for a motif stack indicating
          whether each motif is truncated.
    """
    max_pos = np.argmax(np.sum(np.abs(x), axis=2), axis=1)  # (N, )
    motif_length = x.shape[1]
    return (max_pos < threshold * motif_length) | (max_pos > (1 - threshold) * motif_length)