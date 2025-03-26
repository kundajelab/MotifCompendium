import functools

import numpy as np
import pandas as pd


####################
# MOTIF MANAGEMENT #
####################
def single_or_many_motifs(func):
    """Decorator to handle single or many motifs.

    Functions using this decorator will always have their first argument be a motif
      stack. However, for the user, the first argument can be either a single motif or
      a motif stack. The return value will be be output accordingly.
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
    if not isinstance(motifs, np.ndarray):
        raise TypeError("Motifs must be a np.ndarray.")


def validate_motif_stack(motifs: np.ndarray) -> None:
    """Validate that motifs are a motif stack."""
    validate_motif_basic(motifs)
    if not (len(motifs.shape) == 3 and motifs.shape[2] in [4, 8]):
        raise ValueError("Motif stack must be of shape (N, L, 4/8).")


def validate_motif_stack_standard(motifs: np.ndarray) -> None:
    """Validate that motifs are a standard (N, L, 4) shape."""
    validate_motif_stack(motifs)
    if not motifs.shape[2] == 4:
        raise ValueError("Motif stack must be of shape (N, L, 4).")


def validate_motif_stack_similarity(motifs: np.ndarray) -> None:
    """Validate that motifs are fit for similarity calculations."""
    validate_motif_stack(motifs)
    if not ((motifs >= 0).all() and np.allclose(motifs.sum(axis=(1, 2)), 1)):
        raise ValueError("Motifs must be non-negative and sum to 1.")


#######################
# MOTIF MANIPULATIONS #
#######################
@single_or_many_motifs
def motif_4_to_8(x: np.ndarray) -> np.ndarray:
    """Converts a 4 channel motif(s) into an 8 channel motif(s)."""
    if not x.shape[-1] == 4:
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
    if not x.shape[-1] == 8:
        raise ValueError("Input motif(s) must have 8 channels.")
    x_pos_4 = x @ _MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ _MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 - x_neg_4
    return x_4


@single_or_many_motifs
def motif_8_to_4_unsigned(x: np.ndarray) -> np.ndarray:
    """Converts an 8 channel motif(s) into an unsigned 4 channel motif(s)."""
    if not x.shape[-1] == 8:
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
        motif_stack: A motif stack to be aligned. (N, L, K)
        alignment_rc: A forward/reverse complement alignment vector. (N, )
        alignment_h: A horizontal alignment vector. (N, )

    Returns:
        An aligned motif stack. (N, L_new, K), 
        where L_new = L + max(alignment_h, 0) - min(alignment_h, 0).
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
        + motif_stack[:, ::-1, ::-1] * alignment_rc_mtx
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
        # Pad with zeros
        motif_resized = np.zeros((resize_to, K))
        motif_resized[0:L, :] = motif
        return motif_resized
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


def average_motifs(
    motif_stack: np.ndarray,
    alignment_rc: np.ndarray,
    alignment_h: np.ndarray,
    match_original_length: bool = True,
    l1_norm: bool = True,
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
        l1_norm: Whether or not to L1 normalize the average motif before returning.
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
    if l1_norm:
        average_motif = average_motif / np.sum(np.abs(average_motif))
    return average_motif


@single_or_many_motifs
def ic_scale(x: np.ndarray, invert: bool = False) -> np.ndarray:
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
    # Check input
    if not x.shape[2] == 4:
        raise ValueError("IC scaling only allowed for 4 channel motif.")
    # x = (N, L, 4)
    x_abs = np.abs(x)
    x_avg = x_abs / np.sum(x_abs, axis=2, keepdims=True)
    xlogx = x_avg * np.log2(x_avg, where=(x_avg != 0))
    entropy = np.sum(-xlogx, axis=2, keepdims=True) / 2
    ic = 1 - entropy
    if invert:
        scaled = x / ic
    else:
        scaled = x * ic
    return scaled


@single_or_many_motifs
def calculate_ls_scale(var_motif: np.ndarray, ref_motif: np.ndarray) -> np.ndarray:
    """Calculate least squares scaling factor.
    
    Find the best scaling factor, a, to apply to the variable motif to match the scale
      of the reference motif, by minimizing the least squares between the two motifs
      (i.e., min ||a * variable - ref||^2).
    
    Args:
        var_motif: A np.ndarray representing the variable motif. (N, L, K)
        ref_motif: A np.ndarray representing the reference motif. (N, L, K)
        
    Returns:
        A np.ndarray representing the best scaling factor. (N, L, K)
    """
    if var_motif.shape != ref_motif.shape:
        raise ValueError("var_motif and ref_motif must have the same shape.")

    epsilon = 1e-8
    return np.sum((var_motif * ref_motif), axis=(1, 2), keepdims=True) / (np.sum((var_motif ** 2), axis=(1, 2), keepdims=True) + epsilon)


@single_or_many_motifs
def subtract_motifs(
    motifs_core: np.ndarray, 
    motifs_subtract: np.ndarray,
    align_rc: np.ndarray,
    align_h: np.ndarray
) -> np.ndarray:
    """Subtract two motif stacks, based on idx, alignment, and forward/reverse complement.

    Given two motif stacks, motifs_core and motifs_subtract, align motifs_subtract based on 
    align_rc and align_h, scale motifs_subtract by least squares to match scale of motifs_core, 
    subtract motifs_core by motifs_subtract with clipping, to return the remaining components
    of motifs_core.

    Args:
        motifs_core: A np.ndarray representing a stack of K channel motifs,
          to subtract from. (N, L, K)
        motifs_subtract: A np.ndarray representing a stack of K channel motifs,
          to be subtracted by, following order of motifs_core. (N, L, K)
        align_rc: A np.ndarray containing the forward/reverse complement
          relationship between any two motifs. (N,)
        align_h: A np.ndarray containing the horizontal shift information between
          any two motifs. (N,)

    Returns:
        A np.ndarray representing the subtracted motifs. (N, L, K)
    """
    # Check inputs
    validate_motif_stack(motifs_core)
    validate_motif_stack(motifs_subtract)
    if not (align_rc.shape == align_h.shape == (motifs_core.shape[0],)):
        raise ValueError("align_rc, align_h must have the same length as motifs_core.")

    # Subtract motifs
    motifs_subtract = align_motifs(motifs_subtract, align_rc, align_h) # Align motifs
    start = max(-np.min(align_h), 0) + min(np.max(align_h), 0) # Start index
    motifs_subtract = motifs_subtract[:, start:start + motifs_core.shape[1], :] # Clip motifs to core length, L
    motifs_subtract = motifs_subtract * calculate_ls_scale(motifs_subtract, motifs_core) # Least squares scale to best match core
    updated_motifs = np.clip(motifs_core - motifs_subtract, 0, None) # Clip negative values
    return updated_motifs


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
        x: A (L, 4) motif or (N, L, 4) motif stack representing motifs to compute strings from.
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
    meets_specificity = x / per_position_totals >= specificity
    meets_importance = per_position_totals >= importance
    motif_to_str = meets_specificity * meets_importance
    assert (np.sum(motif_to_str, axis=2) <= 1).all()
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
def motif_posneg(x: np.ndarray) -> str | list[str]:
    """Classifies each motif as being positive or negative."""
    validate_motif_stack_standard(x)
    return ["pos" if np.sum(m) > 0 else "neg" for m in np.sum(x, axis=(1, 2)) > 0]


###########
# ENTROPY #
###########
def motif8_to_copair28(motif8: np.array) -> np.array:
    """Expand base channel (e.g., 8-channel: A+,C+,G+,T+,A-,C-,G-,T-)
    to co-occurrrence of all, non-repeating dinucleotide pairs per position
    (e.g., 28-channel combinations)."""
    # Current dimensions: motif8
    rows, cols = motif8.shape
    nuc = 2  # Co-occurring pair

    # New dimensions
    new_cols = int(
        np.math.factorial(cols)
        / (np.math.factorial(cols - nuc) * np.math.factorial(nuc))
    )  # C(n,r)
    new_cols_2 = int(new_cols / 2)
    copair28 = np.zeros((rows, new_cols))

    ## Matrix multiplication + mask
    mask = np.tril(np.ones(cols), k=-1).astype(
        bool
    )  # Mask for bottom left off-diagonal half

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
    # Calculate dinucleotide pairs, as product of distributions
    mask = ~np.eye(cols, dtype=bool)

    # Calculate dinucleotide pairs, as product of distributions
    mask = ~np.eye(cols, dtype=bool)

    for i in range(new_rows):
        dinuc_pair = np.outer(motif8[2 * i, :], motif8[2 * i + 1, :])
        dinuc64[i, :] = dinuc_pair.flatten()

    return dinuc64


def shannon_entropy(prob_array: np.array, epsilon: float = 1e-10) -> float:
    # Normalize, flatten array
    prob_array = prob_array / np.sum(prob_array)
    prob_array = prob_array.flatten()

    # Replace zeroes with epsilon
    prob_array[prob_array == 0] = epsilon
    length = prob_array.shape[0]

    # Calculate Shannon entropy
    entropy = -np.sum(prob_array * np.log2(prob_array)) / np.log2(length)

    return entropy


def calculate_motif_entropy(motif: np.array) -> float:
    """Calculate Shannon entropy of motif.

    Calculation: Shannon entropy on (L,8)
    Purpose:    (High) Archetype: Noise/chaos
                (Low) Archetype: Sharp nucleotide peak (e.g., G)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        motif_entropy: float
    """
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 4:
        motif = motif_4_to_8(motif)  # Convert motif to 8-channel

    # Standardize motif: Normalized, as probability
    rows, cols = motif.shape
    motif8_prob = motif / np.sum(motif)

    # Calculate Shannon entropy
    motif_entropy = shannon_entropy(motif8_prob)

    return motif_entropy


def calculate_posbase_entropy_ratio(motif: np.array) -> float:
    """Calculate ratio of position-wise entropy / base_wise_entropy.

    Calculation: Entropy across position (L,) / Entropy across base (8,)
    Purpose: (High) Archetype: Single nucleotide repeats (e.g., AAAAA, GGGGG)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        posbase_entropy_ratio: float
    """
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 4:
        motif = motif_4_to_8(motif)  # Convert motif to 8-channel

    # Standardize motif: Normalized, as probability
    rows, cols = motif.shape
    motif8_prob = motif / np.sum(motif)

    # Sum across position-wise, base-wise
    pos_prob = np.sum(motif8_prob, axis=1) / np.sum(
        motif8_prob
    )  # Sum across bases (L,), normalize
    base_prob = np.sum(motif8_prob, axis=0) / np.sum(
        motif8_prob
    )  # Sum across positions (,8), normalize

    # Calculate position-wise, base-wise entropy
    pos_entropy = shannon_entropy(pos_prob)
    base_entropy = shannon_entropy(base_prob)

    posbase_entropy_ratio = pos_entropy / base_entropy

    return posbase_entropy_ratio


def calculate_copair_entropy_ratio(motif: np.array) -> float:
    """Calculate ratio of position-wise entropy / base co-occurrence pair entropy.

    Calculation: Entropy across position (L,) /
        Entropy across all pairs of co-occurring, non-repeating bases (28,)
    Purpose: (High) Archetype: High GC, AT bias

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        copair_entropy_ratio: float
    """
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 4:
        motif = motif_4_to_8(motif)  # Convert motif to 8-channel

    # Standardize motif: Normalized, as probability
    rows, cols = motif.shape
    motif8_prob = motif / np.sum(motif)

    # Calculate joint distribution of all non-repeating, non-ordered pairs of bases, normalized
    copair28 = motif8_to_copair28(motif)
    copair28_prob = copair28 / np.sum(copair28)

    # Sum across position-wise, base-wise
    copair_pos_prob = np.sum(copair28_prob, axis=1) / np.sum(
        copair28_prob
    )  # Sum across bases (L,), normalize
    copair_base_prob = np.sum(copair28_prob, axis=0) / np.sum(
        copair28_prob
    )  # Sum across positions (,8), normalize

    # Calculate position-wise, base-wise entropy
    copair_pos_entropy = shannon_entropy(copair_pos_prob)
    copair_base_entropy = shannon_entropy(copair_base_prob)

    copair_entropy_ratio = copair_pos_entropy / copair_base_entropy

    return copair_entropy_ratio


def calculate_dinuc_entropy_ratio(motif: np.array) -> float:
    """Calculate ratio of two-position entropy / two-base entropy.
    Calculation: Entropy across pairs of positions (L/2,) /
        Entropy across all dinucleotide pairs (64,)
    Purpose: (High) Archetype: Dinucleotide repeats (e.g., GCGCGC, ATATAT)

    Args:
        motif: (L, 4) or (L, 8) motif (np.ndarray)

    Returns:
        dinuc_entropy_ratio: Calculated Shannon entropy (float)
    """
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 4:
        motif = motif_4_to_8(motif)  # Convert motif to 8-channel

    # Standardize motif: Normalized, as probability
    rows, cols = motif.shape
    motif8_prob = motif / np.sum(motif)

    # Calculate distribution of all dinucleotide pairs of bases, per 2 positions, normalized
    dinuc64 = motif8_to_dinuc64(motif)
    dinuc64_prob = dinuc64 / np.sum(dinuc64)

    # Sum across position-wise, base-wise
    dinuc_pos_prob = np.sum(dinuc64_prob, axis=1) / np.sum(
        dinuc64_prob
    )  # Sum across bases (L/2,), normalize
    dinuc_base_prob = np.sum(dinuc64_prob, axis=0) / np.sum(
        dinuc64_prob
    )  # Sum across positions (,64), normalize

    # Calculate position-wise, base-wise entropy
    dinuc_pos_entropy = shannon_entropy(dinuc_pos_prob)
    dinuc_base_entropy = shannon_entropy(dinuc_base_prob)

    dinuc_entropy_ratio = dinuc_pos_entropy / dinuc_base_entropy

    return dinuc_entropy_ratio


def check_negpattern_pospeak(motif: np.array) -> bool:
    """Check negative pattern motifs with positive peaks.
    
    Note: Assumes input motif is a negative pattern motif."""
    # Check if motif is valid
    if not isinstance(motif, np.ndarray):
        raise TypeError("Motif must be a NumPy array.")
    if len(motif.shape) != 2:
        raise ValueError("Motif must be a 2D array.")
    if motif.shape[1] not in [4, 8]:
        raise ValueError("Motif second dimension must be 4 or 8.")
    if motif.shape[1] == 8:
        motif = motif_8_to_4(motif)  # Convert motif to 4-channel

    return np.max(motif) > 0