import numpy as np
import pandas as pd


def average_motifs(motifs_8: np.ndarray, alignment_fb: np.ndarray, alignment_h: np.ndarray) -> np.ndarray:
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


MOTIF_4_TO_8_POS = np.zeros((4, 8))
MOTIF_4_TO_8_POS[0, 0] = 1
MOTIF_4_TO_8_POS[1, 2] = 1
MOTIF_4_TO_8_POS[2, 5] = 1
MOTIF_4_TO_8_POS[3, 7] = 1

MOTIF_4_TO_8_NEG = np.zeros((4, 8))
MOTIF_4_TO_8_NEG[0, 1] = 1
MOTIF_4_TO_8_NEG[1, 3] = 1
MOTIF_4_TO_8_NEG[2, 4] = 1
MOTIF_4_TO_8_NEG[3, 6] = 1


def _4_to_8(x: np.ndarray) -> np.ndarray:
    """Converts a 4 channel motif into an 8 channel motif.
    """
    x_pos = np.maximum(x, 0)
    x_neg = np.maximum(-x, 0)
    x_pos_8 = x_pos @ MOTIF_4_TO_8_POS
    x_neg_8 = x_neg @ MOTIF_4_TO_8_NEG
    x_8 = x_pos_8 + x_neg_8
    return x_8


def _8_to_4(x: np.ndarray) -> np.ndarray:
    """Converts in 8 channel motif into a 4 channel motif.
    """
    x_pos_4 = x @ MOTIF_4_TO_8_POS.T
    x_neg_4 = x @ MOTIF_4_TO_8_NEG.T
    x_4 = x_pos_4 - x_neg_4
    return x_4


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


def squash_motif(motif: np.ndarray, squash_to=30: int) -> np.ndarray:
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

