import numpy as np

from MotifCompendium import MotifCompendium as MotifCompendiumClass
import MotifCompendium.utils.motif as utils_motif


####################
# ENTROPY ANALYSES #
####################
def calculate_filters(
    mc: MotifCompendiumClass,
    metric_list: list[str] = [
        "motif_entropy",
        "weighted_base_entropy",
        "weighted_position_entropy",
        "posbase_entropy_score",
        "copair_entropy_score",
        "copair_composition",
        "dinuc_entropy_score",
        "dinuc_composition",
        "dinuc_score",
        "posneg_inverted",
        "truncated",
    ],
) -> None:
    """Calculates filter metrics and stores them in the MotifCompendium metadata.

    Calculates the filter metrics for each motif in the provided MotifCompendium and
    stores the values in the metadata table of the MotifCompendium. The filters are
    intended to be used for filtering out low quality motifs. The filters can only be
    chosen from a predefined list of metrics.

    Args:
        mc: The MotifCompendium to compute motif filters for.
        metric_list: A list of filter metrics to calculated. Metrics must be one of:
          - "motif_entropy": Computes the Shannon entropy of the motif treated as a
              Lx4 vector.
              When Low: Sharp nucleotide peak (e.g., G).
              When High: Noise/chaos.
          - "weighted_base_entropy": Computes the position-weighted base entropy of the
              motif.
              When High: Noisy motif core (e.g., motif is not a single base).
          - "weighted_position_entropy": Computes the base-weighted position entropy of
              the motif.
              When High: Wide repeats (e.g., AAAAA, GGGGG).
          - "posbase_entropy_score": Computes the position entropy * (1 - base entropy)
              entropy score for the motif.
              When High: Wide repeats (e.g., AAAAA, GGGGG).
          - "copair_entropy_score": Computes the frequency of co-occurring bases, and
              uses that copair representation to compute an entropy score for the motif.
              When High: Noisy motif with base pair ambiguity (e.g. C/G share the same
              position).
          - "copair_composition": Computes a measure of how much of the motif can be
              represented by pairs of co-occurring bases.
              When High: Noisy motif with base pair ambiguity (e.g. C/G share the same
              position).
          - "dinuc_entropy_score": Computes the frequency of repeating dinucleotide
              pairs, and uses that dinucleotide representation to compute an entropy
              score for the motif.
              When High: Dinucleotide repeats (e.g. GCGCGC, ATATAT).
          - "dinuc_composition": Computes a measure of how much of the motif can be
              represented by an alternating dinucleotide pair.
              When High: Dinucleotide repeats (e.g. GCGCGC, ATATAT).
          - "dinuc_score": Computes a score of how much dinucleotide repeating occurs
              within the motif. This filter can identify a prominent dinucleotide pair
              that does not appear in a strictly alternating manner.
              When High: Dinucleotide repeats (e.g. GCGCGC, ATATAT).
          - "posneg_inverted": Checks if a positive motif exists in an otherwise
              negative pattern or if a negative motifs in an otherwise positive pattern.
              When True: Positive motif in a negative pattern or visa versa.
          - "possum_negsum_ratio": Computes the ratio of the sum of positive values to
              the sum of negative values in a motif. (Or vice versa if the sum of
              negative values is greater than the sum of positive values). When close to
              1: A motif with a similar maximum positive and negative value, which may
              indicate a low information content motif.
          - "posmax_negmax_ratio": Computes the ratio of the max positive value to the
              max negative value in a motif. (Or vice versa if max negative value is
              greater than max positive value). When close to 1: A motif with a similar
              maximum positive and negative value, which may indicate a low information
              content motif.
          - "maxmean_ratio": Computes the ratio of the max value in the motif to the
              mean value in the motif. When High: A motif with a very strong peak
              compared to the average value in the motif.
          - "truncated": Checks if the motif is truncated and likely has more mass
              extending beyond the edge of the motif length.
              When True: A truncated motif that has been cut off by the window size.

    Note:
        After these filters are calculated, they can be thresholded to identify and
          filter out low quality or low information content motifs. For guidance on the
          value of thresholds to use, see MotifCompendium Tutorial 6 - Motif Filtering.
    """
    # Calculate filter metrics
    for filter_metric in metric_list:
        match filter_metric:
            case "motif_entropy":
                mc["motif_entropy"] = calculate_full_motif_entropy(mc.motifs)
            case "weighted_base_entropy":
                mc["weighted_base_entropy"] = calculate_weighted_base_entropy(mc.motifs)
            case "weighted_position_entropy":
                mc["weighted_position_entropy"] = calculate_weighted_position_entropy(
                    mc.motifs
                )
            case "posbase_entropy_score":
                mc["posbase_entropy_score"] = calculate_position_versus_base_entropy(
                    mc.motifs
                )
            case "copair_entropy_score":
                mc["copair_entropy_score"] = calculate_copair_entropy(mc.motifs)
            case "copair_composition":
                mc["copair_composition"] = calculate_copair_composition(mc.motifs)
            case "dinuc_entropy_score":
                mc["dinuc_entropy_score"] = calculate_dinucleotide_entropy(mc.motifs)
            case "dinuc_composition":
                mc["dinuc_composition"] = (
                    calculate_dinucleotide_alternating_composition(mc.motifs)
                )
            case "dinuc_score":
                mc["dinuc_score"] = calculate_dinucleotide_score(mc.motifs)
            case "posneg_inverted":
                mc["posneg_inverted"] = (
                    utils_motif.motif_posneg_max(mc.motifs) != mc["posneg"]
                )
            case "possum_negsum_ratio":
                mc["possum_negsum_ratio"] = calculate_possum_vs_negsum(mc.motifs)
            case "posmax_negmax_ratio":
                mc["posmax_negmax_ratio"] = calculate_posmax_vs_negmax(mc.motifs)
            case "maxmean_ratio":
                mc["maxmean_ratio"] = calculate_max_vs_mean(mc.motifs)
            case "truncated":
                max_pos = np.argmax(np.sum(np.abs(mc.motifs), axis=-1), axis=-1)  # (N,
                mc["truncated"] = (max_pos < 2) | (max_pos > mc.motifs.shape[1] - 3)
            case _:
                raise ValueError(f"Filter metric {filter_metric} is not implemented.")


@utils_motif.single_or_many_motifs
def calculate_full_motif_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the full motif entropy of a motif/stack.

    Computes the full motif entropy of a motif or motif stack. The full motif entropy is
    computed as an entropy across all L*K dimensions of (L, K) motifs.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The full motif entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too low, the motif is likely a single nucleotide motif and
          you may want to filter it out. You will need to tune the threshold for
          identifying these low entropy motifs for your particular setting, but you may
          want to start with a threshold of < 0.35.
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.75.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
    x_fullmotif = np.reshape(x, (x.shape[0], -1))
    return utils_motif.normalized_last_axis_entropy(x_fullmotif)


@utils_motif.single_or_many_motifs
def calculate_weighted_base_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the position-weighted across-base entropy of a motif/stack.

    Computes the across-base entropy at each position, and takes of weighted average of
    those entropy. The entropies at each position are weighted by the total importance at
    that position.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The weighted base entropy as a float for a single motif or an array of floats
          for a motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.5.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
    across_base_entropy = utils_motif.normalized_last_axis_entropy(x)  # (N, L, 1)
    position_importance = np.sum(x, axis=2, keepdims=True)  # (N, L, 1)
    return np.sum(across_base_entropy * position_importance, axis=(1, 2))  # (N, )


@utils_motif.single_or_many_motifs
def calculate_weighted_position_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate the base-weighted across-position entropy of a motif/stack.

    Computes the across-position entropy for each base, and takes the weighted average of
    those entropies. The entropies of each base are weighted by the total importance of
    that position.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The weighted position entropy as a float for a single motif or an array of
          floats for a motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too high, the motif is a broad, noisy motif, and you may
          want to filter it out. You will need to tune the threshold for identifying
          these high entropy motifs for your particular setting, but you may want to
          start with a threshold of > 0.71.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
    across_position_entropy = np.stack(
        [
            utils_motif.normalized_last_axis_entropy(x[:, :, i]).squeeze()
            for i in range(x.shape[2])
        ],
        axis=1,
    )  # (N, 4)
    base_importance = np.sum(x, axis=1)  # (N, 4)
    return np.sum(across_position_entropy * base_importance, axis=1)  # (N, )


@utils_motif.single_or_many_motifs
def calculate_position_versus_base_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate across-position * (1 - across-base) entropy for a motif/stack.

    Computes the across-position entropy of each motif by summing the importance across
    all bases at each position. Then, computes the across-base entropy of each motif by
    summing the importance across all positions for each base. Then, computes the
    combined score of across-position entropy * (1 - across-base entropy).

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The position versus base entropy ratio as a float for a single motif or an array
          of floats for a motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too high, the motif is a broad, noisy motif, and you may
          want to filter it out. You will need to tune the threshold for identifying
          these high entropy motifs for your particular setting, but you may want to
          start with a threshold of > 0.45.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
    across_position_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(x, axis=2)
    ).squeeze()  # (N, )
    across_base_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(x, axis=1)
    ).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@utils_motif.single_or_many_motifs
def calculate_copair_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair entropy of a motif/stack.

    Transforms a motif into a co-pair format where each channel represents the
    co-occurrence of two bases at the same position. Then, the copair values are
    normalized and the across-position and across-base entropies are computed. Then, the
    combined score of across-position entropy * (1 - across-base entropy) is computed just
    like in calculate_position_versus_base_entropy().

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too high, the motif is likely noise and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          entropy motifs for your particular setting, but you may want to start with a
          threshold of > 0.35.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
    # Calculate copair
    x_cross = x[:, :, :, np.newaxis] @ x[:, :, np.newaxis, :]  # (N, L, 4, 4)
    copair_mask = np.triu(np.ones(x.shape[2]), k=1).astype(np.bool_)
    copair = x_cross[:, :, copair_mask]  # (N, L, 6)
    copair /= np.sum(copair, axis=(1, 2), keepdims=True)  # Normalize copair
    # across-position entropy * (1 - across-base entropy) for copair
    across_position_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(copair, axis=2)
    ).squeeze()  # (N, )
    across_base_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(copair, axis=1)
    ).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@utils_motif.single_or_many_motifs
def calculate_copair_composition(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of copair composition of a motif/stack.

    Computes a measure of how much of a motif can be represented by co-occurring pairs of
    bases. This is done by computing the copair score at each position for a given copair,
    which is 2 * min(base1, base2) at that position. Then, the max copair score at each
    position is computed and summed across the length of the motif.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The copair entropy as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, 1].

    Note:
        When the composition is too high, the motif is likely noise, and you may want to
          filter it out. You will need to tune the threshold for identifying these high
          composition motifs for your particular setting, but you may want to start with a
          threshold of > 0.41.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
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


@utils_motif.single_or_many_motifs
def calculate_dinucleotide_entropy(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide entropy of a motif/stack.

    Transforms a motif into a dinucleotide format where each channel represents the
    occurrence of one base at a position and another base at the next position. Then, the
    dinucleotide values are normalized and the across-position and across-base entropies
    are computed. Then, the combined score of
    across-position entropy * (1 - across-base entropy) is computed just like in
    calculate_position_versus_base_entropy().

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide entropy as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].

    Note:
        When the entropy is too high, the motif is likely a repeat or GC content, and
          you may want to filter it out. You will need to tune the threshold for
          identifying these high entropy motifs for your particular setting, but you may
          want to start with a threshold of > 0.42.
    """
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
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
    across_position_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(dinucleotide, axis=2)
    ).squeeze()  # (N, )
    across_base_entropy = utils_motif.normalized_last_axis_entropy(
        np.sum(dinucleotide, axis=1)
    ).squeeze()  # (N, )
    return across_position_entropy * (1 - across_base_entropy)


@utils_motif.single_or_many_motifs
def calculate_dinucleotide_alternating_composition(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide composition of a motif/stack.

    Computes a measure of how much of a motif can be represented by an alternating
    dinucleotide sequence. This is done by computing the dinucleotide mass at each pair of
    positions, which is the sum of contributions by both bases. Then, the highest total
    importance across all possible non-repeating dinucleotides is returned.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide composition as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].

    Note:
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
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
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


@utils_motif.single_or_many_motifs
def calculate_dinucleotide_score(x: np.ndarray) -> float | np.ndarray:
    """Calculate a measure of dinucleotide occurrence of a motif/stack.

    Computes a score of how much of the motif contains the same dinucleotide pair
    repeatedly. This is done by computing a dinucleotide score at each position as the
    geometric mean of the importance of one base at that position and the importance of
    the other base at the subsequent position. Then, this score is summed across the
    length of the motif. The highest dinucleotide score across all possible non-repeating
    dinucleotides is returned.

    Args:
        x: A non-negative, normalized (L, 4) motif or (N, L, 4) motif stack.

    Returns:
        The dinucleotide score as a float for a single motif or an array of floats for
          a motif stack. The values are bounded in [0, 1].

    Note:
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
    x = utils_motif.abs_ic_scale_normalize(x)  # Prerequisite for filter calculations
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
    return (max_pos < threshold * motif_length) | (
        max_pos > (1 - threshold) * motif_length
    )


@calculate_metrics
def calculate_possum_vs_negsum(x: np.ndarray) -> float | np.ndarray:
    """Calculate the ratio of the sum of positive values to the absolute value of the sum of negative values in a motif or motif stack.

    This metric is calculated by taking the sum of all positive values and the absolute value
      of the sum of all negative values across all positions and bases in the motif, then
      dividing the larger by the smaller. This metric captures how much stronger the
      overall positive signal is compared to the overall negative signal in a motif.

    Args:
        x: A (L, K) motif or (N, L, K) motif stack.

    Returns:
        The ratio of the sum of positive values vs. the absolute value of the sum of
          negative values as a float for a single motif or an array of floats for a
          motif stack. The values are bounded in [0, inf).

    Notes:
        - When this ratio is close to 1, the motif has equal positive and negative signal
          that is close in magnitude. This suggests that the motif may be noisy.
        - When either positive or negative signal is 0, the ratio will be +inf.
        - When both positive and negative signal are 0, the ratio will be 0.
    """
    validate_motif_stack_standard(x)

    sum_positive = np.sum(x * (x > 0), axis=(1, 2))  # (N,)
    abs_sum_negative = np.abs(np.sum(x * (x < 0), axis=(1, 2)))  # (N,)

    pos_dominant = sum_positive >= abs_sum_negative

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            pos_dominant,
            sum_positive / abs_sum_negative,
            abs_sum_negative / sum_positive,
        )

    # Handle 0/0 explicitly
    both_zero = (sum_positive == 0) & (abs_sum_negative == 0)
    ratio = np.where(both_zero, 0.0, ratio)

    return ratio


@calculate_metrics
def calculate_posmax_vs_negmax(x: np.ndarray) -> float | np.ndarray:
    """Calculate the ratio of the dominant signal to the weaker signal in a motif or motif stack.

    This metric is calculated by taking the maximum positive value and the absolute value
      of the minimum negative value across all positions and bases in the motif, then
      dividing the larger by the smaller. This metric captures how much stronger the
      dominant signal is compared to the weaker signal in a motif.

    Args:
        x: A (L, K) motif or (N, L, K) motif stack.

    Returns:
        The ratio of the dominant signal vs. the weaker signal as a float for a
          single motif or an array of floats for a motif stack. The values are bounded in
          [1, inf).

    Notes:
        - When this ratio is close to 1, the motif has equal positive and negative signal
          that is close in magnitude. This suggests that the motif may be noisy.
        - When either positive or negative signal is 0, the ratio will be +inf.
        - When both positive and negative signal are 0, the ratio will be 0.
    """
    validate_motif_stack_standard(x)

    max_positive = np.max(x, axis=(1, 2))  # (N,)
    abs_min_negative = np.abs(np.min(x, axis=(1, 2)))  # (N,)

    pos_dominant = max_positive >= abs_min_negative

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            pos_dominant,
            max_positive / abs_min_negative,
            abs_min_negative / max_positive,
        )

    # Handle 0/0 explicitly
    both_zero = (max_positive == 0) & (abs_min_negative == 0)
    ratio = np.where(both_zero, 0.0, ratio)

    return ratio


@calculate_metrics
def calculate_max_vs_mean(x: np.ndarray) -> float | np.ndarray:
    """Calculate the ratio of the max value to the mean value in a motif or motif stack.

    This metric is calculated by taking the maximum absolute value across all positions and
      bases in the motif, and dividing it by the mean absolute value across all positions
      and bases in the motif. This metric captures how much stronger the strongest signal
      is compared to the average signal in a motif.

    Args:
        x: A (L, K) motif or (N, L, K) motif stack.

    Returns:
        The ratio of the max value to the mean value as a float for a single motif or an
          array of floats for a motif stack. The values are bounded in [0, inf).

    Notes:
        When this ratio is very high, it means that the motif has a very strong peak
        compared to the rest of the signal. This suggests that the motif has a
        very sharp peak.
    """
    validate_motif_stack_standard(x)
    max_abs = np.max(np.abs(x), axis=(1, 2))  # (N, )
    mean_abs = np.mean(np.abs(x), axis=(1, 2))  # (N, )
    return np.divide(
        max_abs,
        mean_abs,
        out=np.zeros_like(max_abs, dtype=x.dtype),
        where=(mean_abs != 0),
    )
