import functools
import multiprocessing
import os

import h5py
import numpy as np
import pandas as pd

import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.motif as utils_motif


####################
# PUBLIC FUNCTIONS #
####################
def which_file_load_failed(func):
    """Decorator to say which file failed to load.

    Helps identify which file load failed. Helps in multiprocessing settings.
    """

    @functools.wraps(func)
    def wrapper(file_loc, *args, **kwargs):
        if not os.path.exists(file_loc):
            raise FileNotFoundError(f"File {file_loc} not found.")
        file_size = os.path.getsize(file_loc)
        if file_loc.endswith(".h5") and file_size < 2000:
            raise ValueError(
                f"File {file_loc} is likely empty (size: {file_size} bytes)."
            )
        elif file_size <= 0:
            raise ValueError(f"File {file_loc} is empty (size: {file_size} bytes).")
        try:
            result = func(file_loc, *args, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Failed to load file {file_loc}.") from e

    return wrapper


def load_modiscos(
    modisco_dict: dict[str, str],
    load_subpatterns: bool = False,
    modisco_region_width: int = 400,
    ic: bool = True,
) -> tuple[
    np.ndarray, list[str], list[int], list[str], list[str], list[float], list[float]
]:
    """Load motifs, names, and other per-motif information from multiple Modisco files.

    Motifs from each Modisco file are extracted by calling load_modisco(). The results
      are then concatenated. Parallelizes the loading if config.get_max_cpus() > 1.

    Args:
        modisco_dict: A dictionary from model name to Modisco file path.
        load_subpatterns: Whether or not to load subpatterns from the Modisco file. If
          True, motifs will be loaded at the subpattern level. If False, motifs will be
          loaded at the pattern level.
        modisco_region_width: The region width used during Modisco. This argument only
          needs to be specified if using a non-standard region width.
        ic: Whether or not to apply information content scaling to Modisco motifs.

    Returns:
        A tuple of motifs, motif names, number of seqlets per motifs, model names,
          whether the motifs were positive or negative patterns, the mean seqlet
          distance with respect to the Modisco region start position, and the mean
          seqlet contribution.

    Notes:
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
        Motifs are returned as a normalized (N, 30, 4) 4 channel motif stack.
        Using ic scaling is highly recommended.
    """
    # Determine the number of processes to use
    num_processes = min(
        utils_config.get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    if num_processes == 1 or len(modisco_dict) == 1:
        # Load serially
        (
            motifs,
            motif_names,
            seqlet_counts,
            model_names,
            posnegs,
            avgdist_summits,
            avg_contribs,
        ) = ([], [], [], [], [], [], [])
        for m_name, m_loc in modisco_dict.items():
            (
                m_motifs,
                m_motif_names,
                m_seqlet_counts,
                m_posnegs,
                m_avgdist_summits,
                m_avg_contribs,
            ) = load_modisco(
                m_loc,
                load_subpatterns=load_subpatterns,
                modisco_region_width=modisco_region_width,
                ic=ic,
            )
            m_motif_names = [f"{m_name}-{x}" for x in m_motif_names]
            motifs.append(m_motifs)
            motif_names += m_motif_names
            seqlet_counts += m_seqlet_counts
            model_names += [m_name] * len(m_motif_names)
            posnegs += m_posnegs
            avgdist_summits += m_avgdist_summits
            avg_contribs += m_avg_contribs
        # Pad motifs to max length
        max_length = max(x.shape[1] for x in motifs)
        motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
        # Concatenate motifs
        motifs = np.concatenate(motifs, axis=0)
    else:
        m_names, m_locs = [], []
        for m_name, m_loc in modisco_dict.items():
            m_names.append(m_name)
            m_locs.append(m_loc)
        payloads = [
            (m_loc, load_subpatterns, modisco_region_width, ic) for m_loc in m_locs
        ]
        with multiprocessing.Pool(processes=num_processes) as p:
            results = p.starmap(load_modisco, payloads)
        (
            motifs,
            motif_names,
            seqlet_counts,
            model_names,
            posnegs,
            avgdist_summits,
            avg_contribs,
        ) = ([], [], [], [], [], [], [])
        for i, r in enumerate(results):
            (
                m_motifs,
                m_motif_names,
                m_seqlet_counts,
                m_posnegs,
                m_avgdist_summits,
                m_avg_contribs,
            ) = r
            m_motif_names = [f"{m_names[i]}-{x}" for x in m_motif_names]
            motifs.append(m_motifs)
            motif_names += m_motif_names
            seqlet_counts += m_seqlet_counts
            model_names += [m_names[i]] * len(m_motif_names)
            posnegs += m_posnegs
            avgdist_summits += m_avgdist_summits
            avg_contribs += m_avg_contribs
        # Pad motifs to max length
        max_length = max(x.shape[1] for x in motifs)
        motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
        # Concatenate motifs
        motifs = np.concatenate(motifs, axis=0)
    return (
        motifs,
        motif_names,
        seqlet_counts,
        model_names,
        posnegs,
        avgdist_summits,
        avg_contribs,
    )


@which_file_load_failed
def load_modisco(
    modisco_file: str,
    load_subpatterns: bool = False,
    modisco_region_width: int = 400,
    ic: bool = True,
) -> tuple[np.ndarray, list[str], list[int], list[str], list[float], list[float]]:
    """Load motifs, names, and other per-motif information from a single Modisco file.

    Each motif from a Modisco results file is extracted, normalized, and optionally has
      ic scaling applied. The motif names, number of seqlets per motif, whether the
      motifs are positive or negative patterns, the mean seqlet distance with respect to
      the Modisco region start position, and the mean seqlet contribution are also
      returned.

    Args:
        modisco_file: A Modisco file path.
        load_subpatterns: Whether or not to load subpatterns from the Modisco file. If
          True, motifs will be loaded at the subpattern level. If False, motifs will be
          loaded at the pattern level.
        modisco_region_width: The region width used during Modisco. This argument only
          needs to be specified if using a non-standard region width.
        ic: Whether or not to apply information content scaling to Modisco motifs.

    Returns:
        A tuple of motifs, motif names, number of seqlets per motifs, model names,
          whether the motifs were positive or negative patterns, the mean seqlet
          distance with respect to the Modisco region start position, and the mean
          seqlet contribution.

    Notes:
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
        Motifs are returned as a normalized (N, 30, 4) 4 channel motif stack.
        Using ic scaling is highly recommended.
    """
    motifs, motif_names, seqlet_counts, posnegs, avgdist_summits, avg_contribs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with h5py.File(modisco_file, "r") as f:
        for pattern_type in ["pos_patterns", "neg_patterns"]:
            pattern_posneg = pattern_type.split("_")[0]
            if not pattern_type in f:
                continue
            for pattern in list(f[pattern_type]):
                # Subpatterns
                if load_subpatterns:
                    for subpattern in [
                        key
                        for key in list(f[pattern_type][pattern])
                        if "subpattern" in key
                    ]:
                        seqlets = f[pattern_type][pattern][subpattern]["seqlets"][
                            "contrib_scores"
                        ][()]
                        motif_sim = _sequence_importance_from_seqlets(seqlets, ic)
                        motifs.append(motif_sim)
                        motif_names.append(f"{pattern_posneg}.{pattern}-{subpattern}")
                        seqlet_counts.append(seqlets.shape[0])
                        posnegs.append(pattern_posneg)
                        avgdist_summits.append(
                            np.mean(
                                np.abs(
                                    f[pattern_type][pattern][subpattern]["seqlets"][
                                        "start"
                                    ][
                                        :
                                    ]  # Start position
                                    - (
                                        modisco_region_width // 2
                                    )  # Modisco region half-width
                                    + np.unravel_index(
                                        np.abs(motif_sim).argmax(), motif_sim.shape
                                    )[
                                        0
                                    ]  # Motif peak
                                )
                            )
                        )
                        avg_contribs.append(
                            np.mean(np.sum(seqlets, axis=(1, 2)), axis=0)
                        )
                # Main patterns
                else:
                    seqlets = f[pattern_type][pattern]["seqlets"]["contrib_scores"][()]
                    motif_sim = _sequence_importance_from_seqlets(seqlets, ic)
                    motifs.append(motif_sim)
                    motif_names.append(f"{pattern_posneg}.{pattern}")
                    seqlet_counts.append(seqlets.shape[0])
                    posnegs.append(pattern_posneg)
                    avgdist_summits.append(
                        np.mean(
                            np.abs(
                                f[pattern_type][pattern]["seqlets"]["start"][
                                    :
                                ]  # Start position
                                - (
                                    modisco_region_width // 2
                                )  # Modisco region half-width
                                + np.unravel_index(
                                    np.abs(motif_sim).argmax(), motif_sim.shape
                                )[
                                    0
                                ]  # Motif peak
                            )
                        )
                    )
                    avg_contribs.append(np.mean(np.sum(seqlets, axis=(1, 2)), axis=0))
    motifs = np.stack(motifs, axis=0)
    return motifs, motif_names, seqlet_counts, posnegs, avgdist_summits, avg_contribs


def load_pfms(
    pfm_dict: dict[str, str],
    ic: bool = True,
    motif_length: int | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load motifs and names from multiple files containing PFMs.

    Motifs from each file containing Position Frequency Matrices (PFMs) are extracted
      by calling load_pfm(). The results are then concatenated. Files in PFM or MEME
      format are supported. Parallelizes the loading if config.get_max_cpus() > 1.

    Args:
        pfm_dict: A dictionary from model name to path of file specyfing PFMs in PFM or
          MEME format.
        ic: Whether or not to apply information content scaling to the PFMs, effectively
          making them PWMs.
        motif_length: If specified, all motifs will be set to this length using
          utils_motif.resize_motif(). If None, the motifs will be resized to match the
          length of the longest motif.

    Returns:
        A tuple of motifs and motif names.

    Notes:
        Only accepts files in the PFM or MEME file formats.
        Motifs are returned as a normalized (N, 30, 4) 4 channel motif stack.
    """
    # Determine the number of processes to use
    num_processes = min(
        utils_config.get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    if num_processes == 1 or len(pfm_dict) == 1:
        # Load serially
        motifs, motif_names, file_names = [], [], []
        for p_name, p_loc in pfm_dict.items():
            p_motifs, p_motif_names = load_pfm(p_loc, ic=ic, motif_length=motif_length)
            p_motif_names = [f"{p_name}-{x}" for x in p_motif_names]
            motifs.append(p_motifs)
            motif_names += p_motif_names
            file_names += [p_name] * len(p_motif_names)
        # Pad motifs to max length
        max_length = max(x.shape[1] for x in motifs)
        motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
        # Concatenate motifs
        motifs = np.concatenate(motifs, axis=0)
    else:
        p_names, p_locs = [], []
        for p_name, p_loc in pfm_dict.items():
            p_names.append(p_name)
            p_locs.append(p_loc)
        payloads = [(p_loc, ic, motif_length) for p_loc in p_locs]
        with multiprocessing.Pool(processes=num_processes) as p:
            results = p.starmap(load_pfm, payloads)
        motifs, motif_names, file_names = [], [], []
        for i, r in enumerate(results):
            p_motifs, p_motif_names = r
            p_motif_names = [f"{p_names[i]}-{x}" for x in p_motif_names]
            motifs.append(p_motifs)
            motif_names += p_motif_names
            file_names += [p_names[i]] * len(p_motif_names)
        # Pad motifs to max length
        max_length = max(x.shape[1] for x in motifs)
        motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
        # Concatenate motifs
        motifs = np.concatenate(motifs, axis=0)
    return motifs, motif_names, file_names


@which_file_load_failed
def load_pfm(
    pfm_file: str, ic: bool = True, motif_length: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load motifs and names from a file containing PFMs.

    Each Position Frequency Matrix (PFM) from the PFM file is extracted, normalized, and
      optionally has ic scaling applied. The motif names are also returned. Files in PFM
      or MEME format are supported.

    Args:
        pfm_file: The PFM file path. The file must be in PFM or MEME format.
        ic: Whether or not to apply information content scaling to the PFMs, effectively
          making them PWMs.
        motif_length: If specified, all motifs will be set to this length using
          utils_motif.resize_motif(). If None, the motifs will be resized to match the
          length of the longest motif.

    Returns:
        A tuple of motifs and motif names.

    Notes:
        Only accepts files in the PFM or MEME file formats.
        Motifs are returned as a normalized (N, 30, 4) 4 channel motif stack.
    """
    file_basename = os.path.basename(pfm_file)
    if "pfm" in file_basename:
        try:
            return _load_pfm_file_pfm_format(pfm_file, ic, motif_length)
        except Exception as e:
            raise ValueError(
                f"Attempted to load {pfm_file} as a file in PFM format (due to 'pfm' in the file name), but failed."
            ) from e
    elif "meme" in pfm_file:
        try:
            return _load_meme_file_meme_format(pfm_file, ic, motif_length)
        except Exception as e:
            raise ValueError(
                f"Attempted to load {pfm_file} as a file in MEME format (due to 'meme' in the file name), but failed."
            ) from e
    else:
        raise ValueError(
            f"Could not determine file format for {pfm_file}. Please have the file name include 'pfm' or 'meme'."
        )


#####################
# PRIVATE FUNCTIONS #
#####################
def _sequence_importance_from_seqlets(seqlets: np.ndarray, ic: bool) -> np.ndarray:
    """Compute a sequence importance matrix representation of a motif from seqlets.

    Seqlets are normalized, averaged, optionally information content scaled, and then renormalized.

    Args:
        x: An (N, L, 4) stack of N seqlets.

    Returns:
        An (L, 4) sequence importance matrix.

    Note:
        The returned motif will be non-negative and have a sum equal to 1.
    """
    # INPUT = (N, L, 4)
    # Normalize
    seqlet_sums = np.sum(np.abs(seqlets), axis=(1, 2), keepdims=True)
    if np.any(seqlet_sums == 0) or np.isnan(seqlet_sums).any():
        raise ValueError("Seqlets contain zero or NaN values.")
    seqlets_normalized = seqlets / seqlet_sums
    # Average and normalize
    seqlets_avg = np.mean(seqlets_normalized, axis=0)
    seqlets_avg = seqlets_avg / np.sum(np.abs(seqlets_avg))
    # Information content scaling
    if ic:
        seqlets_avg = utils_motif.ic_scale(seqlets_avg)
        seqlets_avg = seqlets_avg / np.sum(np.abs(seqlets_avg))
    return seqlets_avg


@which_file_load_failed
def _load_pfm_file_pfm_format(
    pfm_file: str, ic: bool, motif_length: int | None
) -> tuple[np.ndarray, list[str]]:
    """Load motifs and names from a file in PFM format."""
    # Check file
    names = []
    pwms = []
    active_pwm = False
    longest_motif_length = -1
    with open(pfm_file, "r") as f:
        for line in f:
            x = line.strip()
            if active_pwm:
                if x.startswith(">"):
                    # submit
                    current_pwm_np = pd.DataFrame(current_pwm).to_numpy()
                    if motif_length is not None:
                        current_pwm_np = utils_motif.resize_motif(
                            current_pwm_np, motif_length
                        )
                    longest_motif_length = max(
                        longest_motif_length, current_pwm_np.shape[0]
                    )
                    pwms.append(current_pwm_np)
                    names.append(current_pwm_name)
                    # restart
                    current_pwm_name = x[1:]
                    current_pwm = {"A": [], "C": [], "G": [], "T": []}
                else:
                    a, c, g, t = x.split()
                    a, c, g, t = float(a), float(c), float(g), float(t)
                    acgt = np.asarray([[a, c, g, t]])  # (1, 4)
                    current_pwm["A"].append(acgt[0, 0])
                    current_pwm["C"].append(acgt[0, 1])
                    current_pwm["G"].append(acgt[0, 2])
                    current_pwm["T"].append(acgt[0, 3])
            else:
                assert x.startswith(">")
                active_pwm = True
                current_pwm_name = x[1:]
                current_pwm = {"A": [], "C": [], "G": [], "T": []}
    # Resize
    pwms = [utils_motif.pad_motif(x, longest_motif_length) for x in pwms]
    pwms = np.stack(pwms, axis=0)
    # Normalize
    pwms /= np.sum(pwms, axis=(1, 2), keepdims=True)
    # IC Scale
    if ic:
        pwms = utils_motif.ic_scale(pwms)
        pwms /= np.sum(pwms, axis=(1, 2), keepdims=True)
    return pwms, names


@which_file_load_failed
def _load_meme_file_meme_format(
    meme_file: str, ic: bool, motif_length: int | None
) -> tuple[np.ndarray, list[str]]:
    """Load motifs and names from a file in MEME format."""
    # Check file
    names = []
    pwms = []
    active_pwm = False
    longest_motif_length = -1
    with open(meme_file, "r") as f:
        for line in f:
            x = line.strip()
            if not active_pwm:
                if x.startswith("MOTIF"):
                    active_pwm = True
                    current_pwm_name = x.split(" ")[
                        -1
                    ]  # MEME ALLOWS FOR ALTERNATE NAMES IN [2]
                    looking_for_motif_info = True
            else:
                if looking_for_motif_info:
                    if not x.startswith("letter-probability matrix"):
                        continue
                    motif_info = x.split(": ")[1]
                    motif_info_list = motif_info.split(" ")
                    assert len(motif_info_list) % 2 == 0
                    motif_info_dict = dict()
                    for i in range(int(len(motif_info_list) / 2)):
                        motif_info_dict[motif_info_list[2 * i]] = (
                            int(float(motif_info_list[2 * i + 1]))
                            if float.is_integer(float(motif_info_list[2 * i + 1]))
                            else float(motif_info_list[2 * i + 1])
                        )
                    assert motif_info_dict["alength="] == 4
                    num_bases_remaining = motif_info_dict["w="]
                    looking_for_motif_info = False
                    current_pwm = {"A": [], "C": [], "G": [], "T": []}
                else:
                    # read line
                    a, c, g, t = x.split()
                    a, c, g, t = float(a), float(c), float(g), float(t)
                    acgt = np.asarray([[a, c, g, t]])  # (1, 4)
                    current_pwm["A"].append(acgt[0, 0])
                    current_pwm["C"].append(acgt[0, 1])
                    current_pwm["G"].append(acgt[0, 2])
                    current_pwm["T"].append(acgt[0, 3])
                    num_bases_remaining -= 1
                    # if motif over --> submit and restart
                    if num_bases_remaining == 0:
                        # submit
                        current_pwm_np = pd.DataFrame(current_pwm).to_numpy()
                        if motif_length is not None:
                            current_pwm_np = utils_motif.resize_motif(
                                current_pwm_np, motif_length
                            )
                        longest_motif_length = max(
                            longest_motif_length, current_pwm_np.shape[0]
                        )
                        pwms.append(current_pwm_np)
                        names.append(current_pwm_name)
                        # restart
                        active_pwm = False
    # Resize
    pwms = [utils_motif.pad_motif(x, longest_motif_length) for x in pwms]
    pwms = np.stack(pwms, axis=0)
    # Normalize
    pwms /= np.sum(pwms, axis=(1, 2), keepdims=True)
    # IC Scale
    if ic:
        pwms = utils_motif.ic_scale(pwms)
        pwms /= np.sum(pwms, axis=(1, 2), keepdims=True)
    return pwms, names
