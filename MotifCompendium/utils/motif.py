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
    # INPUT = (30, 4)
    x_abs = np.abs(x)
    x_avg = x_abs / np.sum(x_abs, axis=1, keepdims=True)
    xlogx = x_avg * np.log2(x_avg, where=(x_avg != 0))
    entropy = np.sum(-xlogx, axis=1, keepdims=True) / 2
    ic = 1 - entropy
    return x * ic


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
        and (motifs.shape[1] == 30)
        and (motifs.shape[2] in [4, 8])
    ):
        raise ValueError("Motif stack must be of shape (N, 30, 4/8).")
    if not (((motifs >= 0).all()) and (np.allclose(motifs.sum(axis=(1, 2)), 1))):
        raise ValueError("Motifs must be non-negative and sum to 1.")


def motif_to_df(motif: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker."""
    return pd.DataFrame(motif, columns=["A", "C", "G", "T"])


def cwm4_to_cwm8(cwm4: np.array) -> np.array:
    """Expand base pair to A+,C+,G+,T+,A-,C-,G-,T-"""
    rows, cols = cwm4.shape
    cwm8 = np.zeros((rows, 2 * cols))

    cwm8[:, :cols] = np.where(cwm4 > 0, cwm4, 0)
    cwm8[:, cols:] = np.where(cwm4 < 0, -cwm4, 0)

    return cwm8


def cwm8_to_pair28(cwm8: np.array) -> np.array:
    """Expand A+,C+,G+,T+,A-,C-,G-,T- to non-repeating, without order dinucleotide pair combinations"""
    # Current dimensions: cwm8
    rows, cols = cwm8.shape
    nuc = 2  # Dinucleotide pair

    # New dimensions
    new_cols = int(
        np.math.factorial(cols)
        / (np.math.factorial(cols - nuc) * np.math.factorial(nuc))
    )  # C(n,r)
    new_cols_2 = int(new_cols / 2)
    pair28 = np.zeros((rows, new_cols))

    ## Matrix multiplication + mask
    mask = np.tril(np.ones(cols), k=-1).astype(
        bool
    )  # Mask for bottom left off-diagonal half

    for i in range(rows):
        pair_perm = np.outer(cwm8[i, :], cwm8[i, :])  # Permutations: With order
        pair28[i, :] = pair_perm[
            mask
        ]  # Combinations: Without order; Exclude self (e.g., AA,CC,GG,TT)

    return pair28


def cwm8_to_dinuc64(cwm8: np.array) -> np.array:
    """Evaluate two positions at a time, as non-self A+,C+,G+,T+,A-,C-,G-,T- dinucleotide pair permutations"""
    # Current dimensions: cwm8
    rows, cols = cwm8.shape

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
        dinuc_pair = np.outer(cwm8[2 * i, :], cwm8[2 * i + 1, :])
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


def calculate_entropy(cwm4: np.array, length: int = 30) -> tuple:
    """Calculate entropy metrics, to quantify motif information complexity.

    List of Entropy metrics:
        (1) Base entropy:
            Calculation: Shannon entropy on (30,8)
            Purpose:    (High) Archetype #1: Noise/chaos
                        (Low) Archetype #2: Sharp nucleotide peak (e.g., G)
        (2) Pos-base entropy ratio:
            Calculation: Position-wise entropy on (30,) / Base-wise entropy on (4,)
            Purpose:    (High) Archetype #3: Single nucleotide repeats (e.g., AAAAA, GGGGG)
        (3) Pair nucleotide ratio:
            Purpose:    (High) Archetype #4: High GC, AT bias
        (4) Dinucleotide ratio:
            Purpose:    (High) Dinucleotide repeats (e.g., GCGCGC, ATATAT)

    Args:
        cwm4: (L, 4) motif.

    Returns:
        A tuple of (Base entropy, Pos-base ratio, Pair ratio, Dinuc ratio)

    Notes:
        If a position only has one base at a position, it will not change. If only two
          bases are present but are represented equally, their weights will be halved.
          And if all bases are present and represented equally, the weights will for all
          bases at that position will be set to 0.
    """
    if not isinstance(cwm4, np.ndarray):
        raise TypeError("cwm4 must be a NumPy array")
    if len(cwm4.shape) != 2:
        raise ValueError("cwm4 must be a 2D array")

    # Find normalized, standard length CWM
    cwm4_prob = cwm4 / np.sum(cwm4)
    rows, cols = cwm4.shape

    # Create cwm8, pair28, dinuc64, as probability
    cwm8 = cwm4_to_cwm8(cwm4)
    cwm8_prob = cwm8 / np.sum(cwm8)

    pair28 = cwm8_to_pair28(cwm8)
    pair28_prob = pair28 / np.sum(pair28)

    dinuc64 = cwm8_to_dinuc64(cwm8)
    dinuc64_prob = dinuc64 / np.sum(dinuc64)

    # Sum across bases (w,), normalize
    pos_prob = np.sum(cwm8_prob, axis=1) / np.sum(cwm8_prob)
    pair_pos_prob = np.sum(pair28_prob, axis=1) / np.sum(pair28_prob)
    dinuc_pos_prob = np.sum(dinuc64_prob, axis=1) / np.sum(dinuc64_prob)

    # Sum across positions (8 or 64,), normalize
    base_prob = np.sum(cwm8_prob, axis=0) / np.sum(cwm8_prob)
    pair_base_prob = np.sum(pair28_prob, axis=0) / np.sum(pair28_prob)
    dinuc_base_prob = np.sum(dinuc64_prob, axis=0) / np.sum(dinuc64_prob)

    # Calulcate entropy metrics
    # (1) Base entropy
    cwm_entropy = shannon_entropy(cwm8_prob)

    # (2) Pos-base Entropy ratio:
    pos_entropy = shannon_entropy(pos_prob)
    base_entropy = shannon_entropy(base_prob)
    entropy_ratio = (
        pos_entropy / base_entropy
    )  # High entropy when: High positional = Broad profile, Low base: Single base

    # (3) Pair nucleotide ratio:
    pair_pos_entropy = shannon_entropy(pair_pos_prob)
    pair_base_entropy = shannon_entropy(pair_base_prob)
    pair_entropy_ratio = pair_pos_entropy / pair_base_entropy

    # (4) Dinucleotide ratio
    dinuc_pos_entropy = shannon_entropy(dinuc_pos_prob)
    dinuc_base_entropy = shannon_entropy(dinuc_base_prob)
    dinuc_entropy_ratio = dinuc_pos_entropy / dinuc_base_entropy

    return (cwm_entropy, entropy_ratio, pair_entropy_ratio, dinuc_entropy_ratio)


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

# Suggested entropy metric thresholds for chrombpnet-based TF-Modisco motifs
ENTROPY_THRESHOLD_DICT = {
    "1_singlepeak": ("cwm_entropy", "lt", 0.4),
    "2_noisemix": ("cwm_entropy", "gt", 0.75),
    "3_broadsingle": ("entropy_ratio", "gt", 3.0),
    "4_broadbias": ("pair_entropy_ratio", "gt", 3.0),
    "5_broadCpG": ("dinuc_entropy_ratio", "gt", 4.0),
}


#####################
# PRIVATE FUNCTIONS #
#####################
def average_motifs(
    motifs_8: np.ndarray, alignment_fb: np.ndarray, alignment_h: np.ndarray
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

    Returns:
        An 8 channel motif that is an average of the provided motif stack.

    Notes:
        Assumes that the input is an 8 channel motif of shape (N, 30, 8).
    """
    motifs_4 = _8_to_4(motifs_8)
    N = motifs_4.shape[0]
    if N == 1:
        return motifs_8[0, :, :]
    max_shift = np.max(alignment_h[:, 0])
    min_shift = np.min(alignment_h[:, 0])
    width = 30 + max_shift - min_shift
    motif_sum = np.zeros((width, 4))
    for i in range(N):
        s = alignment_h[i, 0]
        motif_sum[np.abs(min_shift) + s : np.abs(min_shift) + s + 30, :] += (
            motifs_4[i, :, :] if alignment_fb[i, 0] == 0 else motifs_4[i, ::-1, ::-1]
        )
    motif_avg = motif_sum / N
    squashed_motif = squash_motif(motif_avg)
    squashed_motif_8 = _4_to_8(squashed_motif)
    squashed_motif_8 /= np.sum(squashed_motif_8)
    return squashed_motif_8


def squash_motif(motif: np.ndarray, squash_to: int = 30) -> np.ndarray:
    """Squashes a larger motif into a smaller size.

    Takes a large motif and squashes/trims it down to a smaller motif. Selects the base
      pairs to keep that have the highest weights.

    Args:
        motif: A (L, C) motif.
        squash_to: The size that the motif should be sqashed to.

    Returns:
        A (squash_to, C) motif.
    """
    N, c = motif.shape
    i_sums = []
    for i in range(N - squash_to + 1):
        i_sums.append((np.sum(np.abs(motif[i : i + squash_to, :])), i))
    i_sums = sorted(i_sums, reverse=True)
    top_i = i_sums[0][1]
    return motif[top_i : top_i + squash_to, :]
