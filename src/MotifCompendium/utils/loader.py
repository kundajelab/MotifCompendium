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
    normalize_over_seqlets: bool = False,
    modisco_region_width: int = 400,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motifs, names, and other per-motif metadata from multiple Modisco files.

    Motifs and per-motif metadata from each Modisco file are extracted by calling
      load_modisco(). The results are then concatenated. Parallelizes the loading if
      config.get_max_cpus() > 1.

    Args:
        modisco_dict: A dictionary from model name to Modisco file path.
        load_subpatterns: Whether or not to load subpatterns from the Modisco file. If
          True, motifs will be loaded at the subpattern level. If False, motifs will be
          loaded at the pattern level.
        normalize_over_seqlets: Whether or not to return motifs as normalized, seqlet-
          averaged motifs. If False, the motifs will be returned as is. If True, the
          seqlets will be normalized, averaged, and then re-normalized in order to
          construct the motifs. This option treats all seqlets as equally important
          regardless of their contribution magnitude.
        modisco_region_width: The region width used during Modisco. This argument only
          needs to be specified if using a non-standard region width.

    Returns:
        Motifs and a metadata DataFrame. The motifs are returned as a (N, L, 4) motif
          stack. The metadata columns are: the motif name, the number of seqlets used to
          construct the motif, whether the motif was called positive or negative, the
          average contribution score of the motif, and the average distance of the motif
          from the center of Modisco regions.

    Note:
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
    """
    # Set up return lists
    motifs, metadatas = [], []
    # Determine the number of processes to use
    num_processes = min(
        utils_config.get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    # Load modisco files
    if num_processes == 1 or len(modisco_dict) == 1:
        # Load serially
        for m_name, m_loc in modisco_dict.items():
            m_motifs, m_metadata = load_modisco(
                m_loc, load_subpatterns, normalize_over_seqlets, modisco_region_width
            )
            motifs.append(m_motifs)
            m_metadata["name"] = [f"{m_name}-{x}" for x in m_metadata["name"]]
            m_metadata["model"] = m_name
            metadatas.append(m_metadata)
    else:
        # Load in parallel
        m_names, m_locs = [], []
        for m_name, m_loc in modisco_dict.items():
            m_names.append(m_name)
            m_locs.append(m_loc)
        payloads = [
            (m_loc, load_subpatterns, normalize_over_seqlets, modisco_region_width)
            for m_loc in m_locs
        ]
        with multiprocessing.Pool(processes=num_processes) as p:
            results = p.starmap(load_modisco, payloads)
        for i, r in enumerate(results):
            m_motifs, m_metadata = r
            motifs.append(m_motifs)
            m_metadata["name"] = [f"{m_names[i]}-{x}" for x in m_metadata["name"]]
            m_metadata["model"] = m_names[i]
            metadatas.append(m_metadata)
    # Pad motifs to max length
    max_length = max(x.shape[1] for x in motifs)
    motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
    # Concatenate motifs
    motifs = np.concatenate(motifs, axis=0)
    metadata = pd.concat(metadatas, axis=0, ignore_index=True)
    # Return
    return motifs, metadata


@which_file_load_failed
def load_modisco(
    modisco_file: str,
    load_subpatterns: bool = False,
    normalize_over_seqlets: bool = False,
    modisco_region_width: int = 400,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motifs, names, and other per-motif metadata from a single Modisco file.

    Each motif from the specified Modisco file is extracted. The motifs alongside per-
      motif metadata are returned. By default, the motifs are returned as is, but they
      can optionally be returned as normalized, seqlet-averaged motifs.

    Args:
        modisco_file: A Modisco file path.
        load_subpatterns: Whether or not to load subpatterns from the Modisco file. If
          True, motifs will be loaded at the subpattern level. If False, motifs will be
          loaded at the pattern level.
        normalize_over_seqlets: Whether or not to return motifs as normalized, seqlet-
          averaged motifs. If False, the motifs will be returned as is. If True, the
          seqlets will be normalized, averaged, and then renormalized in order to
          construct the motifs. This option treats all seqlets as equally important
          regardless of their contribution magnitude.
        modisco_region_width: The region width used during Modisco. This argument only
          needs to be specified if using a non-standard region width.

    Returns:
        Motifs and a metadata pd.DataFrame. The motifs are returned as a (N, L, 4) motif
          stack. The metadata columns are: the motif name, the number of seqlets used to
          construct the motif, whether the motif was called positive or negative, the
          average contribution score of the motif, and the average distance of the motif
          from the center of Modisco regions.

    Note:
        Assumes that all motifs are stored within "pos_patterns" or "neg_patterns".
    """
    # Set up return lists
    motifs, motif_names, seqlet_counts, posnegs, avg_contribs, avgdist_summits = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    # Load modisco file
    with h5py.File(modisco_file, "r") as f:
        for pattern_type in ["pos_patterns", "neg_patterns"]:
            pattern_posneg = pattern_type.split("_")[0]
            if pattern_type not in f:
                continue
            for pattern in list(f[pattern_type]):
                # Subpatterns
                if load_subpatterns:
                    for subpattern in [
                        key
                        for key in list(f[pattern_type][pattern])
                        if "subpattern" in key
                    ]:
                        subpattern_group = f[pattern_type][pattern][subpattern]
                        modisco_motif = subpattern_group["contrib_scores"][()]
                        seqlets = subpattern_group["seqlets"]["contrib_scores"][()]
                        motif = (
                            _normalized_motif_from_seqlets(seqlets)
                            if normalize_over_seqlets
                            else modisco_motif
                        )
                        motifs.append(motif)
                        motif_names.append(f"{pattern_posneg}.{pattern}-{subpattern}")
                        seqlet_counts.append(seqlets.shape[0])
                        posnegs.append(pattern_posneg)
                        avg_contribs.append(np.sum(modisco_motif))
                        avgdist_summits.append(
                            np.mean(
                                np.abs(
                                    subpattern_group["seqlets"]["start"][
                                        :
                                    ]  # Start position
                                    - (
                                        modisco_region_width // 2
                                    )  # Modisco region half-width
                                    + np.unravel_index(
                                        np.abs(motif).argmax(), motif.shape
                                    )[
                                        0
                                    ]  # Motif peak
                                )
                            )
                        )
                # Main patterns
                else:
                    pattern_group = f[pattern_type][pattern]
                    modisco_motif = pattern_group["contrib_scores"][()]
                    seqlets = pattern_group["seqlets"]["contrib_scores"][()]
                    motif = (
                        _normalized_motif_from_seqlets(seqlets)
                        if normalize_over_seqlets
                        else modisco_motif
                    )
                    motifs.append(motif)
                    motif_names.append(f"{pattern_posneg}.{pattern}")
                    seqlet_counts.append(seqlets.shape[0])
                    posnegs.append(pattern_posneg)
                    avg_contribs.append(np.sum(modisco_motif))
                    avgdist_summits.append(
                        np.mean(
                            np.abs(
                                pattern_group["seqlets"]["start"][:]  # Start position
                                - (
                                    modisco_region_width // 2
                                )  # Modisco region half-width
                                + np.unravel_index(np.abs(motif).argmax(), motif.shape)[
                                    0
                                ]  # Motif peak
                            )
                        )
                    )
    # Concatenate motifs and create metadata
    motifs = np.stack(motifs, axis=0) if len(motifs) > 1 else motifs[0][np.newaxis, :, :]
    metadata = pd.DataFrame(
        {
            "name": motif_names,
            "seqlet_count": seqlet_counts,
            "posneg": posnegs,
            "avg_contrib": avg_contribs,
            "avgdist_summit": avgdist_summits,
        }
    )
    # Return
    return motifs, metadata


def load_pfms(
    pfm_dict: dict[str, str],
    motif_length: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motifs and names from multiple files containing PFMs.

    Motifs from each file containing Position Frequency Matrices (PFMs) are extracted
      by calling load_pfm(). The results are then concatenated. Files in PFM or MEME
      format are supported. Parallelizes the loading if config.get_max_cpus() > 1.

    Args:
        pfm_dict: A dictionary from model name to path of file specyfing PFMs in PFM or
          MEME format.
        motif_length: If specified, all motifs will be set to this length using
          utils_motif.resize_motif(). If None, the motifs will be resized to match the
          length of the longest motif.

    Returns:
        A tuple of motifs and motif names. The motifs are returned as a (N, L, 4) motif
          stack where L is motif_length if specified or the length of the longest motif
          otherwise.

    Note:
        Only accepts files in the PFM or MEME file formats.
    """
    # Setup return lists
    motifs, metadatas = [], []
    # Determine the number of processes to use
    num_processes = min(
        utils_config.get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    # Load
    if num_processes == 1 or len(pfm_dict) == 1:
        # Load serially
        for p_name, p_loc in pfm_dict.items():
            p_motifs, p_metadata = load_pfm(p_loc, motif_length=motif_length)
            motifs.append(p_motifs)
            p_metadata["file"] = p_name
            metadatas.append(p_metadata)
    else:
        p_names, p_locs = [], []
        for p_name, p_loc in pfm_dict.items():
            p_names.append(p_name)
            p_locs.append(p_loc)
        payloads = [(p_loc, motif_length) for p_loc in p_locs]
        with multiprocessing.Pool(processes=num_processes) as p:
            results = p.starmap(load_pfm, payloads)
        for i, r in enumerate(results):
            p_motifs, p_metadata = r
            motifs.append(p_motifs)
            p_metadata["file"] = p_names[i]
            metadatas.append(p_metadata)
    # Pad motifs to max length
    max_length = max(x.shape[1] for x in motifs)
    motifs = [utils_motif.pad_motif(x, pad_to=max_length) for x in motifs]
    # Concatenate motifs
    motifs = np.concatenate(motifs, axis=0)
    metadata = pd.concat(metadatas, axis=0, ignore_index=True)
    # Return
    return motifs, metadata


@which_file_load_failed
def load_pfm(
    pfm_file: str, motif_length: int | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motifs and per-motif metadata from a single file containing PFMs.

    Each Position Frequency Matrix (PFM) from the PFM file is extracted. The motifs are
      returned as is, whether or not they actually adhere to the PFM definition. The
      length of each motif is set to the length of the longest motif or to motif_length
      if specified. Files in PFM or MEME format are supported.

    Args:
        pfm_file: The PFM file path. The file must be in PFM or MEME format.
        motif_length: If specified, all motifs will be set to this length using
          utils_motif.resize_motif(). If None, the motifs will be resized to match the
          length of the longest motif.

    Returns:
        Motifs and a metadata pd.DataFrame. The motifs are returned as a (N, L, 4) motif
          stack.

    Note:
        Only accepts files in the PFM or MEME file formats.
    """
    file_basename = os.path.basename(pfm_file)
    if "pfm" in file_basename:
        try:
            return _load_pfm_file_pfm_format(pfm_file, motif_length=motif_length)
        except Exception as e:
            raise ValueError(
                f"Attempted to load {pfm_file} as a file in PFM format (due to 'pfm' in the file name), but failed."
            ) from e
    elif "meme" in pfm_file:
        try:
            return _load_meme_file_meme_format(pfm_file, motif_length=motif_length)
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
def _normalized_motif_from_seqlets(seqlets: np.ndarray) -> np.ndarray:
    """Compute a normalized motif from seqlets.

    Seqlets are normalized, averaged, and then renormalized.

    Args:
        x: An (N, L, 4) stack of N seqlets.

    Returns:
        An (L, 4) sequence importance matrix.

    Note:
        The returned motif will be non-negative and have an absolute sum of 1.
    """
    normalized_seqlets = seqlets / np.sum(np.abs(seqlets), axis=(1, 2), keepdims=True)
    averaged_normalized_seqlets = np.mean(normalized_seqlets, axis=0)
    motif = averaged_normalized_seqlets / np.sum(np.abs(averaged_normalized_seqlets))
    return motif


@which_file_load_failed
def _load_pfm_file_pfm_format(
    pfm_file: str, motif_length: int | None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motifs and names from a file in PFM format."""
    # Set up return lists
    motifs = []
    names = []
    # Prepare file parsing
    active_motif = False
    longest_motif_length = -1
    # Parse file
    with open(pfm_file, "r") as f:
        for line in f:
            x = line.strip()
            if active_motif:
                if x.startswith(">"):
                    # Submit
                    current_motif_np = pd.DataFrame(current_motif).to_numpy()
                    if motif_length is not None:
                        current_motif_np = utils_motif.resize_motif(
                            current_motif_np, motif_length
                        )
                    longest_motif_length = max(
                        longest_motif_length, current_motif_np.shape[0]
                    )
                    motifs.append(current_motif_np)
                    names.append(current_motif_name)
                    # Restart
                    current_motif_name = x[1:]
                    current_motif = {"A": [], "C": [], "G": [], "T": []}
                else:
                    a, c, g, t = x.split()
                    a, c, g, t = float(a), float(c), float(g), float(t)
                    acgt = np.asarray([[a, c, g, t]])  # (1, 4)
                    current_motif["A"].append(acgt[0, 0])
                    current_motif["C"].append(acgt[0, 1])
                    current_motif["G"].append(acgt[0, 2])
                    current_motif["T"].append(acgt[0, 3])
            else:
                assert x.startswith(">")
                active_motif = True
                current_motif_name = x[1:]
                current_motif = {"A": [], "C": [], "G": [], "T": []}
    # Concatenate motifs and create metadata
    motifs = [utils_motif.pad_motif(x, longest_motif_length) for x in motifs]
    motifs = np.stack(motifs, axis=0) if len(motifs) > 1 else motifs[0][np.newaxis, :, :]
    metadata = pd.DataFrame({"name": names})
    # Return
    return motifs, metadata


@which_file_load_failed
def _load_meme_file_meme_format(
    meme_file: str, motif_length: int | None
) -> tuple[np.ndarray, list[str]]:
    """Load motifs and names from a file in MEME format."""
    # Set up return lists
    motifs = []
    names = []
    # Prepare file parsing
    active_motif = False
    longest_motif_length = -1
    # Parse file
    with open(meme_file, "r") as f:
        for line in f:
            x = line.strip()
            if not active_motif:
                if x.startswith("MOTIF"):
                    active_motif = True
                    current_motif_name = x.split(" ")[
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
                    current_motif = {"A": [], "C": [], "G": [], "T": []}
                else:
                    # read line
                    a, c, g, t = x.split()
                    a, c, g, t = float(a), float(c), float(g), float(t)
                    acgt = np.asarray([[a, c, g, t]])  # (1, 4)
                    current_motif["A"].append(acgt[0, 0])
                    current_motif["C"].append(acgt[0, 1])
                    current_motif["G"].append(acgt[0, 2])
                    current_motif["T"].append(acgt[0, 3])
                    num_bases_remaining -= 1
                    # if motif over --> submit and restart
                    if num_bases_remaining == 0:
                        # submit
                        current_motif_np = pd.DataFrame(current_motif).to_numpy()
                        if motif_length is not None:
                            current_motif_np = utils_motif.resize_motif(
                                current_motif_np, motif_length
                            )
                        longest_motif_length = max(
                            longest_motif_length, current_motif_np.shape[0]
                        )
                        motifs.append(current_motif_np)
                        names.append(current_motif_name)
                        # restart
                        active_motif = False
    # Concatenate motifs and create metadata
    motifs = [utils_motif.pad_motif(x, longest_motif_length) for x in motifs]
    motifs = np.stack(motifs, axis=0) if len(motifs) > 1 else motifs[0][np.newaxis, :, :]
    metadata = pd.DataFrame({"name": names})
    # Return
    return motifs, metadata
