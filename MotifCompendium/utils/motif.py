import numpy as np
import pandas as pd


####################
# PUBLIC FUNCTIONS #
####################
def ic_scale(x: np.ndarray) -> np.ndarray:
    """Rescale a 4 channel motif by per position information content.

    Each position in the motif is scaled by the information content at that position.
      The information content is computed as 1 - base4entropy of the per base importance
      at that position.

    Args:
        x: A (L, 4) motif.

    Returns:
        An information content scaled (L, 4) motif.

    Notes:
        If a position only has one base at a position, it will not change. If only two
          bases are present but are represented equally, their weights will be halved.
          And if all bases are present and represented equally, the weights will for all
          bases at that position will be set to 0.
    """
    # INPUT = (L, 4)
    x_abs = np.abs(x)
    x_norm = x_abs / np.sum(x_abs, axis=1, keepdims=True)
    xlogx = x_norm * np.log2(x_norm, where=(x_norm != 0))
    entropy = np.sum(-xlogx, axis=1, keepdims=True) / 2
    ic = 1 - entropy
    return x * ic


def ic_invert(x: np.ndarray) -> np.ndarray:
    """Invert a motif by information content scaling.
    
    Args:
        x: A (L, 4) motif, in log_4 space.
        
    Returns:
        An information content inverted (L, 4) motif, in linear space.
    
    Notes:
        Motifs must be in log_4 space. (e.g., After ic_scale)
    """
    x_abs = np.abs(x)
    x_norm = x_abs / np.sum(x_abs, axis=1, keepdims=True)
    xlogx = x_norm * np.log2(x_norm, where=(x_norm != 0))
    entropy = np.sum(-xlogx, axis=1, keepdims=True) / 2
    ic = 1 - entropy
    return x / ic


def motif_4_to_8(x: np.ndarray) -> np.ndarray:
    """Converts a 4 channel motif into an 8 channel motif."""
    x_pos = np.maximum(x, 0)
    x_neg = np.maximum(-x, 0)
    x_pos_8 = x_pos @ _MOTIF_4_TO_8_POS
    x_neg_8 = x_neg @ _MOTIF_4_TO_8_NEG
    x_8 = x_pos_8 + x_neg_8
    return x_8


def motif_8_to_4(x: np.ndarray) -> np.ndarray:
    """Converts in 8 channel motif into a 4 channel motif."""
    x_pos_4 = x @ _MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ _MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 - x_neg_4
    return x_4


def motif_8_to_4_abs(x: np.ndarray) -> np.ndarray:
    """Converts in 8 channel motif into a 4 channel motif."""
    x_pos_4 = x @ _MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ _MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 + x_neg_4
    return x_4


def validate_motif_stack(motifs: np.ndarray) -> None:
    """Validate motifs."""
    if not isinstance(motifs, np.ndarray):
        raise TypeError("Motifs must be a np.ndarray.")
    if not (
        (len(motifs.shape) == 3)
        and (motifs.shape[2] in [4, 8])
    ):
        raise ValueError("Motif stack must be of shape (N, L, 4/8).")
    if not (((motifs >= 0).all()) and (np.allclose(motifs.sum(axis=(1, 2)), 1))):
        raise ValueError("Motifs must be non-negative and sum to 1.")


def motif_to_df(motif: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker."""
    return pd.DataFrame(motif, columns=["A", "C", "G", "T"])


def resize_motif(motif: np.ndarray, resize_to: int) -> np.ndarray:
    """Resize a motif (by squashing or padding) to a specified length.

    If the given motif is shorter than resize_to, pad with 0s until it is large enough.
      If the given motif is larger than resize_to, squash it down to a smaller motif.
      Selects the base pairs to keep that have the highest weights.

    Args:
        motif: A (L, C) motif.
        resize_to: The length to squash or pad the motif to.

    Returns:
        A (resize_to, C) motif.
    """
    L, C = motif.shape
    if L < resize_to:
        # Pad with zeros
        motif_resized = np.zeros((resize_to, C))
        motif_resized[0:L, :] = motif
        return motif_resized
    elif L > resize_to:
        # Squash to the desired length
        i_sums = [
            (np.sum(np.abs(motif[i : i + resize_to, :])), i)
            for i in range(L - resize_to + 1)
        ]
        top_i = max(i_sums, key=lambda x: x[0])[1]
        return motif[top_i : top_i + resize_to, :]
    else:
        return motif


def average_motifs(
    motifs_8: np.ndarray, 
    alignment_fb: np.ndarray, 
    alignment_h: np.ndarray,
    weights: list[float | int] | None = None
) -> np.ndarray:
    """Compute the average of many motifs.

    Takes in a stack of motifs and aligns them based on alignment matrices. Then,
      averages them and crops as necessary.

    Args:
        motifs_8: A np.ndarray reprsenting a stack of 8 channel motifs to average.
        alignment_fb: A np.ndarray containing the forward/reverse complement
          relationship between any two motifs.
        alignment_h: A np.ndarray containing the horizontal shift information between
          any two motifs.
        weights: A list of weights to apply to each motif. (Optional)

    Returns:
        An 8 channel motif that is an average of the provided motif stack.

    Notes:
        Assumes that the input is an 8 channel motif of shape (N, L, 8).
    """
    motifs_4 = (
        motif_8_to_4(motifs_8)
        if motifs_8.shape[2] == 8
        else motifs_8
    )
    N = motifs_4.shape[0]
    L = motifs_4.shape[1]
    if N == 1:
        return motifs_8[0, :, :]
    max_shift = np.max(alignment_h[:, 0])
    min_shift = np.min(alignment_h[:, 0])
    width = L + max_shift - min_shift
    if weights is None:
        weights = np.ones(N)

    motif_sum = np.zeros((width, 4))
    for i in range(N):
        s = alignment_h[i, 0]
        start = s - min_shift
        end = start + L
        motif_sum[start:end, :] += (
            motifs_4[i, :, :] if alignment_fb[i, 0] == 0 else motifs_4[i, ::-1, ::-1]
        ) * weights[i]
    motif_avg = motif_sum / np.sum(weights)

    squashed_motif = resize_motif(
        motif=motif_avg, 
        resize_to=L)
    squashed_motif_8 = (
        motif_4_to_8(squashed_motif)
        if motifs_8.shape[2] == 8
        else squashed_motif
    )
    squashed_motif_8 /= np.sum(squashed_motif_8)
    return squashed_motif_8


def normalize_motifs(motifs: np.ndarray, sim_type: str) -> np.ndarray:
    """Normalize a stack of motifs, based on the similarity type.

    Args:
        motifs: A np.ndarray representing a stack of motifs. (N, L, 4)
        sim_type: The similarity type to use for normalization.
    
    Returns:
        A np.ndarray representing the normalized motifs. (N, L, 4)
    """
    match sim_type:
        case "sqrt":
            return np.sqrt(motifs / np.sum(motifs, axis=(1, 2), keepdims=True)) # L1 sqrt normalize
        case "l2":
            return motifs / np.linalg.norm(motifs, axis=(1, 2), keepdims=True) # L2 normalize
        case "jss":
            return motifs / np.sum(motifs, axis=(1, 2), keepdims=True) # L1 normalize
        case "dot" | "log":
            return motifs


def calculate_least_squares_scale(var_motif: np.ndarray, ref_motif: np.ndarray) -> float:
    """Find the best scaling factor, a,  to apply to the variable motif to match the scale
      of the reference motif, by minimizing the least squares between the two motifs
      (i.e., min ||a * target - ref||^2).
    
    Args:
        var_motif: A np.ndarray representing the target motif. (L, 4)
        ref_motif: A np.ndarray representing the reference motif. (L, 4)
        
    Returns:
        A float representing the best scaling factor.
    """
    return np.sum(var_motif * ref_motif) / np.sum(var_motif ** 2)


def subtract_motifs(
    motifs_core: np.ndarray, 
    motifs_subtract: np.ndarray, 
    align_idx: np.ndarray,
    align_fr: np.ndarray,
    align_h: np.ndarray
) -> np.ndarray:
    """Subtract two motif stacks, based on idx, alignment, and forward/reverse complement.

    Args:
        motifs_core: A np.ndarray representing a stack of 4 channel motifs,
          to subtract from. (N, L, 4)
        motifs_subtract: A np.ndarray representing a stack of 4 channel motifs,
          to be subtracted. (M, L, 4)
        align_idx: A np.ndarray containing the index of the motif in motifs_sub
          to subtract per motif in motifs_core. (N,)
        align_fr: A np.ndarray containing the forward/reverse complement
          relationship between any two motifs. (N,)
        align_h: A np.ndarray containing the horizontal shift information between
          any two motifs. (N,)

    Returns:
        A np.ndarray representing the subtracted motifs. (N, L, 4)
    """
    updated_motifs = np.zeros_like(motifs_core)
    for i in range(motifs_core.shape[0]):
        motif_core = motifs_core[i, :, :] # (L, 4)
        motif_subtract = motifs_subtract[align_idx[i], :, :] # (L, 4)
        if align_fr[i] == 1:
            motif_subtract = motif_subtract[::-1, ::-1] # Align fr: Reverse complement
            motif_subtract = shift_and_fill_zero(motif_subtract, align_h[i])
        else:
            motif_subtract = shift_and_fill_zero(motif_subtract, -align_h[i])
        motif_subtract = motif_subtract * calculate_least_squares_scale(motif_subtract, motif_core) # Scale to best match core
        updated_motifs[i, :, :] = np.clip(motif_core - motif_subtract, 0, None) # Clip negative values
    return updated_motifs


def shift_and_fill_zero(motif: np.ndarray, shift: int) -> np.ndarray:
    """Shift a motif and replace with zeros.

    Args:
        motif: A np.ndarray representing a 4 channel motif. (L, 4)
        shift: The amount to shift the motif by.
    """
    if abs(shift) >= motif.shape[0]:
        raise ValueError("Shift value exceeds the length of the motif.")
    shifted = np.zeros_like(motif)
    if shift > 0:
        shifted[shift:, :] = motif[:-shift, :]
    elif shift < 0:
        shifted[:shift, :] = motif[-shift:, :]
    else:
        shifted = motif
    return shifted


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


def shannon_entropy(prob_array: np.array) -> float:
    # Normalize, flatten array
    prob_array = prob_array / np.sum(prob_array)
    prob_array = prob_array.flatten()
    length = prob_array.shape[0]

    # Filter out zero probabilities
    prob_array = prob_array[prob_array > 0]
    
    # Calculate normalized Shannon entropy
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


#####################
# PRIVATE CONSTANTS #
#####################
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


#####################
# PRIVATE FUNCTIONS #
#####################
