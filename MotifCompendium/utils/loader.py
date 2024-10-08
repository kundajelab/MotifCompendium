import multiprocessing

import h5py
import numpy as np
import pandas as pd

from .motif import ic_scale, motif_4_to_8


####################
# PUBLIC FUNCTIONS #
####################
def load_modiscos(modisco_dict: dict[str, str], max_cpus: int | None = None, ic: bool = True) -> tuple[np.ndarray, list[str], list[int]]:
    """Load motifs, names, and seqlet counts from multiple Modisco file.

    Motifs from each Modisco file are extracted by calling load_modisco(). The results
      are then concatenated. Loading can be parallelized across Modisco files.

    Args:
        modisco_dict: A dictionary from model name to Modisco file path.
        max_cpus: The maximum number of processes to use for loading motifs from
          Modisco files. If None, Modisco files will be loaded serially.
        ic: Whether or not to apply information content scaling to Modisco motifs.

    Returns:
        A tuple of motifs, motif names, and number of seqlets per motifs.

    Notes:
        For parallel loading, the number of processes used will be the minimum of
          max_cpus and multiprocessing.cpu_count().
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
        Motifs are returned as an (N, 30, 8) motif stack.
        Using ic scaling is highly recommended.
    """
    if max_cpus is None:
        # Load serially
        sims, motif_names, seqlet_counts = [], [], []
        for m_name, m_loc in modisco_dict.items():
            m_sims, m_motif_names, m_seqlet_counts = load_modisco(m_loc, ic=ic)
            m_motif_names = [f"{m_name}-{x}" for x in m_motif_names]
            sims.append(m_sims)
            motif_names += m_motif_names
            seqlet_counts += m_seqlet_counts
        sims = np.concatenate(sims, axis=0)
    else:
        # Load in parallel
        num_processes = min(max_cpus, multiprocessing.cpu_count())  # don't use more CPUs than available
        m_names, m_locs = [], []
        for m_name, m_loc in modisco_dict.items():
            m_names.append(m_name)
            m_locs.append(m_loc)
        payloads = [(m_loc, ic) for m_loc in m_locs]
        with multiprocessing.Pool(processes=num_processes) as p:
            results = p.starmap(load_modisco, payloads)
        sims, motif_names, seqlet_counts = [], [], []
        for i, r in enumerate(results):
            m_sims, m_motif_names, m_seqlet_counts = r
            m_motif_names = [f"{m_names[i]}-{x}" for x in m_motif_names]
            sims.append(m_sims)
            motif_names.append(m_motif_names)
            seqlet_counts.append(m_seqlet_counts)
        sims = np.concatenate(sims, axis=0)
    return sims, motif_names, seqlet_counts


def load_modisco(modisco_file: str, ic: bool = True) -> tuple[np.ndarray, list[str], list[int]]:
    """Load motifs, names, and seqlet counts from a Modisco file.

    Each motif from a Modisco results file is extracted, normalized, and optionally has
      ic scaling applied. The name and number of seqlets for each motif are also
      extracted.

    Args:
        modisco_file: A Modisco file path.
        ic: Whether or not to apply information content scaling to Modisco motifs.

    Returns:
        A tuple of motifs, motif names, and number of seqlets per motifs.

    Notes:
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
        Motifs are returned as an (N, 30, 8) 8 channel motif stack.
        Using ic scaling is highly recommended.
    """
    sims, motif_names, seqlet_counts = [], [], []
    with h5py.File(modisco_file, "r") as f:
        if "pos_patterns" in f:
            for pattern in list(f["pos_patterns"]):
                seqlets = f["pos_patterns"][pattern]["seqlets"]["contrib_scores"][()]
                motif_sim = _sequence_importance_from_seqlets(seqlets, ic)
                sims.append(motif_sim)
                motif_names.append(f"pos.{pattern}")
                seqlet_counts.append(seqlets.shape[0])
        if "neg_patterns" in f:
            for pattern in list(f["neg_patterns"]):
                seqlets = f["neg_patterns"][pattern]["seqlets"]["contrib_scores"][()]
                motif_sim = _sequence_importance_from_seqlets(seqlets, ic)
                sims.append(motif_sim)
                motif_names.append(f"neg.{pattern}")
                seqlet_counts.append(seqlets.shape[0])
    sims = np.stack(sims, axis=0)
    return sims, motif_names, seqlet_counts


def load_pfm(pfm_file: str) -> tuple[np.ndarray, list[str]]:
    """Load motifs and names from a PFM file.

    Each PFM from a PFM file is extracted. Then, the PFM is transformed into a PWM using
      per position information content scaling.

    Args:
        pfm_file: The PFM file path.

    Returns:
        A tuple of motifs and motif names.

    Notes:
        Assumes a standard PFM file format.
        Motifs are returned as (N, 30, 4) 4 channel motif stack.
    """
    names = []
    pwms = []
    active_pwm = False
    with open(pfm_file, "r") as f:
        for line in f:
            x = line.strip()
            if active_pwm:
                if x.startswith(">"):
                    # submit
                    pwms.append(squash_motif(pd.DataFrame(current_pwm)))
                    names.append(current_pwm_name)
                    # restart
                    current_pwm_name = x[1:]
                    current_pwm = {"A": [], "C": [], "G": [], "T": []}
                else:
                    a, c, g, t = x.split()
                    a, c, g, t = float(a), float(c), float(g), float(t)
                    acgt = np.asarray([a, c, g, t])
                    xlogx = acgt * np.log2(acgt, where=(acgt != 0))
                    entropy = np.sum(-xlogx) / 2
                    ic = 1 - entropy
                    acgt_ic = acgt * ic
                    current_pwm["A"].append(acgt_ic[0])
                    current_pwm["C"].append(acgt_ic[1])
                    current_pwm["G"].append(acgt_ic[2])
                    current_pwm["T"].append(acgt_ic[3])
            else:
                assert x.startswith(">")
                active_pwm = True
                current_pwm_name = x[1:]
                current_pwm = {"A": [], "C": [], "G": [], "T": []}
    pwms_mtx = np.stack([x.to_numpy() for x in pwms], axis=0)
    pwms_mtx /= np.sum(pwms_mtx, axis=(1, 2), keepdims=True)
    return pwms_mtx, names


#####################
# PRIVATE FUNCTIONS #
#####################
def _sequence_importance_from_seqlets(seqlets: np.ndarray, ic: bool) -> np.ndarray:
    """Compute a sequence importance matrix representation of a motif from seqlets.

    Seqlets are normalized, averaged, optionally information content scaled, transformed
      to 8 channel motifs, and then renormalized.

    Args:
        x: An (N, L, 4) stack of N seqlets.

    Returns:
        An (L, 8) sequence importance matrix.

    Note:
        The returned sim will be non-negative and have a sum equal to 1.
    """
    # INPUT = (N, L, 4)
    # Normalize
    seqlets_normalized = seqlets / np.sum(np.abs(seqlets), axis=(1, 2), keepdims=True)
    # Average and normalize
    seqlets_avg = np.mean(seqlets_normalized, axis=0)
    seqlets_avg = seqlets_avg / np.sum(np.abs(seqlets_avg))
    # Information content scaling
    if ic:
        seqlets_avg = ic_scale(seqlets_avg)
    # Create sim
    sim = motif_4_to_8(seqlets_avg)
    sim /= np.abs(sim)
    return sim
