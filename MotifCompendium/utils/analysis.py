import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot

from MotifCompendium import MotifCompendium as MotifCompendiumClass
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif


#######################
# SIMILARITY ANALYSES #
#######################
def plot_similarity_distribution(
    mc: MotifCompendiumClass,
    save_loc: str,
    vals: list[float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    tolerance: float = 0.001,
    n_per: int = 3,
) -> None:
    """Plots examples of various similarities in a MotifCompendium.

    For a set of similarities, create an html file displaying multiple examples of pairs
      of motifs that have a similarity within [val, val+tolerance) for each similarity
      value.

    Args:
        mc: The MotifCompendium to analyze.
        save_loc: The location where to save the output html file.
        vals: The list of similarity scores to display examples of.
        tolerance: The tolerance of error with respect to target similarity values to display.
        n_per: The number of examples of each similarity score to display.
    """
    label = [False for _ in range(len(mc))]
    for val in vals:
        # Find all locations where similarity is val <= similarity < val+tolerance
        indices = np.where((mc.similarity >= val) & (mc.similarity < val + tolerance))
        indices = list(zip(indices[0], indices[1]))
        p = 1
        for i, j in indices:
            if not (label[i] or label[j]):
                label[i] = f"Similarity {val} example {p}"
                label[j] = f"Similarity {val} example {p}"
                p += 1
            if p > n_per:
                break
    # Create MotifCompendium of only displayed motifs
    label_series = pd.Series(label)
    mc_distribution = mc[label_series != False]
    distribution_clusters = label_series[label_series != False].tolist()
    mc_distribution.motif_collection_html(
        save_loc, distribution_clusters, average_motif=False
    )


def plot_clustering_similarity_mismatch(
    mc: MotifCompendiumClass,
    clustering: str,
    similarity_threshold: float,
    save_loc: str,
    max_examples: int = 100,
) -> None:
    """Plots examples of when similarity did not match a motif clustering.

    Given a MotifCompendium, a clustering/grouping of motifs, and a similarity
      threshold, plot examples of when motif similarities did not match with the
      clustering. Plot examples where two motifs are clustered together in the
      clustering but have a lower similarity than the threshold. Also, plot examples
      where two motifs are not clustered together in the ground truth but have a higher
      similarity than the threshold.

    Args:
        mc: The MotifCompendium to analyze.
        clustering: The column in the MotifCompendium metadata to use compare against.
        similarity_threshold: The similarity value to threshold at.
        save_loc: The location where to save the output html file.
        max_examples: The maximum number of mismatch examples to plot.
    """
    quality = mc.clustering_quality(clustering)
    label = [False for _ in range(len(mc))]
    # Low internal similarity
    n_examples = 0
    for c in quality.columns:
        if quality.loc[c, c] >= similarity_threshold:
            continue
        c_select = mc[clustering] == c
        similarity_slice_ii_df = mc.get_similarity_slice(c_select, c_select)
        similarity_slice_ii_df_stacked = similarity_slice_ii_df.stack()
        row_label, col_label = similarity_slice_ii_df_stacked.idxmin()
        label[row_label] = f"Low internal similarity {c} ({quality.loc[c, c]:.3})"
        label[col_label] = f"Low internal similarity {c} ({quality.loc[c, c]:.3})"
        n_examples += 1
        if n_examples >= (max_examples) // 2:
            break
    # High external similarity
    n_examples = 0
    for i, ci in enumerate(quality.columns):
        for j, cj in enumerate(quality.columns):
            if j <= i:
                continue
            if quality.loc[ci, cj] < similarity_threshold:
                continue
            similarity_slice_ij_df = mc.get_similarity_slice(
                mc[clustering] == ci, mc[clustering] == cj
            )
            similarity_slice_ij_df_stacked = similarity_slice_ij_df.stack()
            row_label, col_label = similarity_slice_ij_df_stacked.idxmax()
            label[row_label] = (
                f"High external similarity {ci} & {cj} ({quality.loc[ci, cj]:.3})"
            )
            label[col_label] = (
                f"High external similarity {ci} & {cj} ({quality.loc[ci, cj]:.3})"
            )
            n_examples += 1
            if n_examples >= (max_examples) // 2:
                break
        if n_examples >= (max_examples) // 2:
            break
    # Create MotifCompendium of only displayed motifs
    label_series = pd.Series(label)
    mc_mismatch = mc[label_series != False]
    mismatch_clusters = label_series[label_series != False].tolist()
    mc_mismatch.motif_collection_html(save_loc, mismatch_clusters, average_motif=False)


def judge_clustering(
    mc: MotifCompendiumClass,
    cluster_col: str,
    show: bool = False,
    save_loc: str | None = None,
) -> None:
    """Plots histograms of inter-cluster and intra-cluster similarities.

    Judges a motif clustering by computing the quality of the clustering and then
      plotting the distribution of minimum intercluster similarities as well as plotting
      the distribution of the maximum intracluster similarity.

    Args:
        mc: The MotifCompendium to analyze.
        clustering: The motif clustering to judge.
        show: Whether or not to show the plot with plt.show().
        save_loc: The file prefix to save the clustering quality and the clustering
          quality plot to.
    """
    # Get clustering quality
    clustering_quality = mc.clustering_quality(cluster_col).to_numpy()
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    bins = np.linspace(0, 1, 20)
    # Plot intra-cluster similarities
    diag = np.diag(clustering_quality)
    diag = np.sort(diag)
    sns.histplot(diag, ax=axs[0], stat="proportion", kde=True, bins=bins)
    axs[0].set_title("lowest intra-cluster similarities")
    axs[0].set_xlim(0, 1)  # Shared
    # Plot inter-cluster similarities
    triu = np.triu(clustering_quality, k=1)
    triu = triu[triu != 0]
    triu = np.sort(triu)
    sns.histplot(triu, ax=axs[1], stat="proportion", kde=True, bins=bins)
    axs[1].set_title("highest inter-cluster similarities")
    axs[1].set_xlabel("similarity")
    # Title
    plt.suptitle(f"{cluster_col} ({clustering_quality.shape[0]} clusters)")
    # Save/show/close
    if save_loc is not None:
        plt.savefig(save_loc)
    if show:
        plt.show()
    plt.close(fig)


#######################
# DOWNSTREAM ANALYSES #
#######################
def plot_unique_per_cluster(
    mc: MotifCompendiumClass, group_by: str, save_loc: str
) -> None:
    """Identifies and plots the most unique in each cluster.

    For each cluster, identifies the most unique motif (motif with the minimum maximal
      similarity with all motifs not in that cluster). Then displays them all.

    Args:
        mc: The MotifCompendium to analyze.
        group_by: The grouping to find unique clusters within.
        save_loc: The path to save the unique clusters html to.

    Notes:
        The most unique motif is defined as the motif within a cluster whose
          maximal similarity with all motifs not in that cluster is the lowest.
    """
    clustering = [False for _ in range(len(mc))]
    for c in set(mc[group_by]):
        similarity_contrast_c_df = mc.get_similarity_slice(
            mc[group_by] == c, mc[group_by] != c
        )
        c_best_similarities = similarity_contrast_c_df.max(axis=1)
        most_unique = c_best_similarities.idxmin()
        most_unique_similarity = c_best_similarities.min()
        clustering[most_unique] = f"{c} ({most_unique_similarity:.3})"
    # Create MotifCompendium of only displayed motifs
    clustering_series = pd.Series(clustering)
    mc_unique = mc[clustering_series != False]
    unique_clusters = clustering_series[clustering_series != False].tolist()
    mc_unique.motif_collection_html(save_loc, unique_clusters, average_motif=False)


def cluster_grouping_upset_plot(
    mc: MotifCompendiumClass, clustering: str, grouping: str, save_loc: str, **kwargs
) -> None:
    """Creates an upset plot of how many motif clusters span across different groups.

    Given a grouping that each motif belongs (ex: source celltype) to and a clustering
      of motifs, create an Upset Plot that displays which clusters belong to which
      groups.

    Args:
        mc: The MotifCompendium to analyze.
        clustering: The motif clustering to consider.
        grouping: The grouping to compute cluster source intersections with respect to.
        save_loc: The file to save the Upset Plot to.
        **kwargs: Additional named arguments that usetplot.UpSet() takes.

    Notes:
        Requires package upsetplot to run.
        Consider running with argument min_subset_size.
    """
    membership_lists = [
        list(set(mc[mc[clustering] == c][grouping])) for c in set(mc[clustering])
    ]

    clusters_by_grouping = upsetplot.from_memberships(membership_lists)
    fig = plt.figure()
    upsetplot.UpSet(clusters_by_grouping, subset_size="count", **kwargs).plot(fig=fig)
    plt.savefig(save_loc, bbox_inches="tight")
    plt.close(fig=fig)


def export_full_compendium_modisco(
    mc: MotifCompendiumClass,
    name_col: str,
    save_loc: str,
    inverse_ic: bool = False,
) -> None:
    """Exports MotifCompendium in the Modisco file format.

    Assumes that the MotifCompendium is already clustered and just exports it to an h5py
      structure that matches Modisco outputs.

    Args:
        mc: The MotifCompendium to analyze.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the Modisco h5py to.
    Notes:
        The resultant h5py file can be fed directly into FiNeMo (hitcaller).
        Motif names cannot have slashes (/) in them!
    """
    utils_motif.validate_motif_stack(mc.motifs)
    size_4 = mc.motifs.shape[2] == 4
    motifs = mc.motifs if size_4 else utils_motif.motif_8_to_4_signed(mc.motifs)
    pos_neg = utils_motif.motif_posneg_sum(motifs)
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = motifs.shape[1]
        # Positive
        if "pos" in pos_neg:
            pos_group = f.create_group("pos_patterns")
            mc_pos = mc[pd.Series(pos_neg) == "pos"]
            motifs_pos = (
                mc_pos.motifs
                if size_4
                else utils_motif.motif_8_to_4_signed(mc_pos.motifs)
            )
            for i in range(len(mc_pos)):
                name = f"{mc_pos.metadata.loc[i, name_col]}_{i}"
                if "/" in name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                motif = motifs_pos[i, :, :]
                if inverse_ic:
                    motif = utils_motif.ic_scale(motif, invert=True)
                pos_cluster = pos_group.create_group(name)
                pos_cluster.create_dataset("contrib_scores", data=motif)
        # Negative
        if "neg" in pos_neg:
            neg_group = f.create_group("neg_patterns")
            mc_neg = mc[pd.Series(pos_neg) == "neg"]
            motifs_neg = (
                mc_neg.motifs
                if size_4
                else utils_motif.motif_8_to_4_signed(mc_neg.motifs)
            )
            for i in range(len(mc_neg)):
                name = f"{mc_neg.metadata.loc[i, name_col]}_{i}"
                if "/" in name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                motif = motifs_neg[i, :, :]
                if inverse_ic:
                    motif = utils_motif.ic_scale(motif, invert=True)
                neg_cluster = neg_group.create_group(name)
                neg_cluster.create_dataset("contrib_scores", data=motif)


def export_clusters_modisco(
    mc: MotifCompendiumClass,
    cluster_name: str,
    save_loc: str,
    inverse_ic: bool = False,
    weight_col: str | None = None,
) -> None:
    """Exports cluster average motifs in the Modisco file format.

    Given a clustering, compute the average MotifCompendium and call
      export_full_compendium_modisco() on it to export them to an h5py structure that
      matches Modisco outputs.

    Args:
        mc: The MotifCompendium to analyze.
        cluster_name: The motif clustering to compute average motifs on.
        save_loc: The location to save the Modisco h5py to.
        inverse_ic: Whether or not to revert IC scaling on motifs. Consider using if
          motifs were ingested from Modisco with IC scaling.
        safe: Whether or not to construct the MotifCompendium safely.

    Notes:
        The resultant h5py file can be fed directly into FiNeMo (hitcaller).
        Cluster names cannot have slashes (/) in them!
    """
    # Validate motifs
    utils_motif.validate_motif_stack(mc.motifs)
    size_4 = mc.motifs.shape[2] == 4
    # Average motifs in the cluster
    mc_avg = mc.cluster_averages(
        clustering=cluster_name,
        aggregations=[],
        weight_col=weight_col,
    )
    mc_avg[cluster_name] = (mc_avg["name"].str.split("#").str[1]).astype(
        mc[cluster_name].dtype
    )
    mc_avg.sort("name", inplace=True)
    # Create Modisco export
    avg_motifs = mc_avg.motifs if size_4 else utils_motif.motif_8_to_4_signed(mc_avg.motifs)
    pos_neg = utils_motif.motif_posneg_sum(avg_motifs)
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = avg_motifs.shape[1]
        # Positive
        if "pos" in pos_neg:
            pos_group = f.create_group("pos_patterns")
            mc_avg_pos = mc_avg[pd.Series(pos_neg) == "pos"]
            avg_motifs_pos = (
                mc_avg_pos.motifs
                if size_4
                else utils_motif.motif_8_to_4_signed(mc_avg_pos.motifs)
            )
            # Pattern = Cluster average
            for i in range(len(mc_avg_pos)):
                pattern_name = f"{mc_avg_pos.metadata.loc[i, cluster_name]}_{i}"
                if "/" in pattern_name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                pos_pattern = pos_group.create_group(pattern_name)
                avg_motif = avg_motifs_pos[i, :, :]
                if inverse_ic:
                    avg_motif = utils_motif.ic_scale(avg_motif, invert=True)
                pos_pattern.create_dataset("contrib_scores", data=avg_motif)
                # Sub pattern = Motif
                mc_i = mc[mc[cluster_name] == mc_avg_pos.metadata.loc[i, cluster_name]]
                motifs_i = (mc_i.motifs if size_4 else utils_motif.motif_8_to_4_signed(mc_i.motifs))
                for j in range(len(mc_i)):
                    subpattern_name = f"{mc_i.metadata.loc[j, 'name']}_{j}"
                    if "/" in subpattern_name:
                        raise ValueError("Motif names cannot have slashes (/) in them!")
                    pos_pattern_subpattern = pos_pattern.create_group(subpattern_name)
                    motif = motifs_i[j, :, :]
                    if inverse_ic:
                        motif = utils_motif.ic_scale(motif, invert=True)
                    pos_pattern_subpattern.create_dataset("contrib_scores", data=motif)

        # Negative
        if "neg" in pos_neg:
            neg_group = f.create_group("neg_patterns")
            mc_avg_neg = mc_avg[pd.Series(pos_neg) == "neg"]
            avg_motifs_neg = (
                mc_avg_neg.motifs
                if size_4
                else utils_motif.motif_8_to_4_signed(mc_avg_neg.motifs)
            )
            # Pattern = Cluster average
            for i in range(len(mc_avg_neg)):
                pattern_name = f"{mc_avg_neg.metadata.loc[i, cluster_name]}_{i}"
                if "/" in pattern_name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                neg_pattern = neg_group.create_group(pattern_name)
                avg_motif = avg_motifs_neg[i, :, :]
                if inverse_ic:
                    motif = utils_motif.ic_scale(avg_motif, invert=True)
                neg_pattern.create_dataset("contrib_scores", data=avg_motif)
                # Sub pattern = Motif
                mc_i = mc[mc[cluster_name] == mc_avg_neg.metadata.loc[i, cluster_name]]
                motifs_i = (mc_i.motifs if size_4 else utils_motif.motif_8_to_4_signed(mc_i.motifs))
                for j in range(len(mc_i)):
                    subpattern_name = f"{mc_i.metadata.loc[j, 'name']}_{j}"
                    if "/" in subpattern_name:
                        raise ValueError("Motif names cannot have slashes (/) in them!")
                    neg_pattern_subpattern = neg_pattern.create_group(subpattern_name)
                    motif = motifs_i[j, :, :]
                    if inverse_ic:
                        motif = utils_motif.ic_scale(motif, invert=True)
                    neg_pattern_subpattern.create_dataset("contrib_scores", data=motif)


def export_full_compendium_meme(
    mc: MotifCompendiumClass,
    name_col: str,
    save_loc: str,
    inverse_ic: bool = False,
) -> None:
    """Exports MotifCompendium in the MEME file format.

    Assumes that the MotifCompendium is already clustered and just exports it to a MEME
      file format.

    Args:
        mc: The MotifCompendium to analyze.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the MEME file to.

    Notes:
        The resultant MEME file can be fed directly into FiNeMo (hitcaller).
        Motif names cannot have slashes (/) in them!
    """
    # Validate motifs
    utils_motif.validate_motif_stack(mc.motifs)
    size_4 = mc.motifs.shape[2] == 4
    motifs = mc.motifs if size_4 else utils_motif.motif_8_to_4_signed(mc.motifs)
    pos_neg = utils_motif.motif_posneg_sum(motifs)
    # Write MEME file
    with open(save_loc, "w") as f:
        f.write("MEME version 4\n")
        f.write(f"ALPHABET= {'ACGT' if size_4 else 'ACGTN'}\n")
        f.write(f"strands: {'+' if 'pos' in pos_neg else ''}{'-' if 'neg' in pos_neg else ''}\n")
        f.write(f"Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")
        for i in range(len(mc)):
            name = f"{mc.metadata.loc[i, name_col]}"
            if "/" in name:
                raise ValueError("Motif names cannot have slashes (/) in them!")
            motif = motifs[i, :, :]
            # Remove empty flanks
            motif = utils_motif.remove_zero_flanks(motif)
            # Inverse IC scaling
            if inverse_ic:
                motif = utils_motif.ic_scale(motif, invert=True)
            # Write motif
            f.write(f"\nMOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= {motif.shape[1]} w= {motif.shape[0]} nsites= {mc.metadata.loc[i, 'num_seqlets']} E= 0\n")
            for j in range(motif.shape[0]):
                f.write(" ".join([f"{x:.6f}" for x in motif[j, :]]) + "\n")


####################
# ENTROPY ANALYSES #
####################
def calculate_filters(
    mc: MotifCompendiumClass,
    metric_list: list[str] = [
        "motif_entropy",
        "weighted_base_entropy",
        "weighted_position_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "copair_composition",
        "dinuc_entropy_ratio",
        "dinuc_composition",
        "dinuc_score"
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
          - "truncated": Checks if the motif is truncated and likely has more mass
              extending beyond the edge of the motif length.
              When True: A truncated motif that has been cut off by the window size.

    Notes:
        After these filters are calculated, they can be thresholded to identify and
          filter out low quality or low information content motifs. For guidance on the
          value of thresholds to use, see MotifCompendium Tutorial 6 - Motif Filtering.
    """
    # Calculate filter metrics
    mc_motifs = mc.get_standard_motif_stack()
    mc_motifs_abs = np.abs(mc_motifs)
    for filter_metric in metric_list:
        match filter_metric:
            case "motif_entropy":
                mc["motif_entropy"] = utils_motif.calculate_full_motif_entropy(mc_motifs_abs)
            case "weighted_base_entropy":
                mc["weighted_base_entropy"] = utils_motif.calculate_weighted_base_entropy(mc_motifs_abs)
            case "weighted_position_entropy":
                mc["weighted_position_entropy"] = utils_motif.calculate_weighted_position_entropy(mc_motifs_abs)
            case "posbase_entropy_ratio":
                mc["posbase_entropy_ratio"] = utils_motif.calculate_position_versus_base_entropy(mc_motifs_abs)
            case "copair_entropy_ratio":
                mc["copair_entropy"] = utils_motif.calculate_copair_entropy(mc_motifs_abs)
            case "copair_composition":
                mc["copair_composition"] = utils_motif.calculate_copair_composition(mc_motifs_abs)
            case "dinuc_entropy_ratio":
                mc["dinuc_entropy_ratio"] = utils_motif.calculate_dinucleotide_entropy(mc_motifs_abs)
            case "dinuc_composition":
                mc["dinuc_composition"] = utils_motif.calculate_dinucleotide_alternating_composition(mc_motifs_abs)
            case "dinuc_score":
                mc["dinucleotide_score"] = utils_motif.calculate_dinucleotide_score(mc_motifs_abs)
            case "posneg_inverted":
                mc["posneg_inverted"] = (
                    utils_motif.motif_posneg_max(mc_motifs)
                    != mc["posneg"]
                )
            case "truncated":
                max_pos = mc_motifs.sum(axis=-1).argmax(axis=-1) # (N,)
                mc["truncated"] = (max_pos < 2) | (max_pos > mc_motifs.shape[1] - 3)
            case _:
                raise ValueError(f"Filter metric {filter_metric} is not implemented.")


###########################
# EXISTING MOTIF DATABASE #
###########################
def assign_label_from_pfms(
    mc: MotifCompendiumClass,
    pfm_file: str,
    save_col_prefix: str = "match",
    max_submotifs: int = 1,
    min_score: float = 0.5,
    save_images: bool = True,
) -> None:
    """Automatic labeling of motifs from a pfm file.

    For each motif in the MotifCompendium, computes the similarity between that motif
      and all motifs in the PFM file, for max_submotif iterations. The highest similarity
      and closest motif match for each iteration will be saved as columns
      {save_col_prefix}_score and {save_col_prefix}_name in the MotifCompendium
      metadata.

    Args:
        mc: The MotifCompendium to analyze.
        pfm_file: The PFM file path.
        save_col_prefix: The prefix to use for the saved columns.
        max_submotifs: The maximum number of submotifs to consider in a match.
        min_score: The minimum similarity score to consider as a match.
    """
    # Load PFM database, with same length as motifs
    L = mc.motifs.shape[1]
    if pfm_file.endswith("pfms.txt"):
        pfm_motifs, pfm_names = utils_loader.load_pfm(pfm_file, L)
    elif pfm_file.endswith(".meme.txt") or pfm_file.endswith(".meme"):
        pfm_motifs, pfm_names = utils_loader.load_meme(pfm_file, L)
    else:
        raise ValueError("pfm_file must be a _pfm.txt, .meme, or .meme.txt file.")

    # Assign labels
    mc.assign_label_from_motifs(
        pfm_motifs,
        pfm_names,
        save_col_prefix=save_col_prefix,
        max_submotifs=max_submotifs,
        min_score=min_score,
        save_images=save_images,
    )
