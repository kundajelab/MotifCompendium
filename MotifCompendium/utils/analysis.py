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
        tolerance: The tolerance of error with respect to target similarity values to
          display.
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
            if label[row_label] or label[col_label]:
                continue
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


def export_compendium_modisco(
    mc: MotifCompendiumClass,
    name_col: str,
    save_loc: str,
    inverse_ic: bool = False,
) -> None:
    """Exports MotifCompendium in the Modisco file format.

    Exports a MotifCompendium into an h5py file that matches the structure of Modisco
      outputs. Each motif in the MotifCompendium becomes a pattern in the Modisco
      output.

    Args:
        mc: The MotifCompendium to export.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the Modisco h5py to.
        inverse_ic: Whether or not to perform inverse information content scaling on
          motifs. This should only be set to True if you want to revert previous IC
          scaling. If you built your MotifCompendium from Modisco motifs and did not
          explicitly turn off IC scaling, then your motifs were IC scaled and you may
          want to perform inverse IC scaling before exporting them.

    Notes:
        Motif names cannot have slashes (/) in them!
        The resultant h5py file can be fed directly into FiNeMo.
        If you are exporting your motifs to FiNeMo and your motifs were previously IC
          scaled but you don't want to revert the IC scaling, you may need to adjust the
          default trimming threshold during finemo call-hits with option -t 0.15.
    """
    pos_neg = pd.Series(utils_motif.motif_posneg_sum(mc.get_standard_motif_stack()))
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = mc.motifs.shape[1]
        # Positive
        if "pos" in pos_neg:
            pos_group = f.create_group("pos_patterns")
            mc_pos = mc[pos_neg == "pos"]
            motifs_pos = mc_pos.get_standard_motif_stack()
            pos_names = mc_pos[name_col].tolist()
            for i in range(len(mc_pos)):
                name = f"{pos_names[i]}_{i}"
                if "/" in name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                motif = motifs_pos[i]
                if inverse_ic:
                    motif = utils_motif.ic_scale(motif, invert=True)
                pos_cluster = pos_group.create_group(name)
                pos_cluster.create_dataset("contrib_scores", data=motif)
        # Negative
        if "neg" in pos_neg:
            neg_group = f.create_group("neg_patterns")
            mc_neg = mc[pos_neg == "neg"]
            motifs_neg = mc_neg.get_standard_motif_stack()
            neg_names = mc_neg[name_col].tolist()
            for i in range(len(mc_neg)):
                name = f"{neg_names[i]}_{i}"
                if "/" in name:
                    raise ValueError("Motif names cannot have slashes (/) in them!")
                motif = motifs_neg[i]
                if inverse_ic:
                    motif = utils_motif.ic_scale(motif, invert=True)
                neg_cluster = neg_group.create_group(name)
                neg_cluster.create_dataset("contrib_scores", data=motif)


def export_compendium_clustered_modisco(
    mc: MotifCompendiumClass,
    cluster_name: str,
    save_loc: str,
    inverse_ic: bool = False,
    weight_col: str | None = None,
    export_subpatterns: bool = False,
) -> None:
    """Exports cluster average motifs in the Modisco file format.

    Exports a MotifCompendium into an h5py file that matches the structure of Modisco
      outputs. A clustering is specified, and the cluster averages each become a pattern
      in the Modisco output. Optionally, each motif in the MotifCompendium can become a
      subpattern of the cluster it is a part of.

    Args:
        mc: The MotifCompendium to export.
        cluster_name: The motif clustering to group motifs by.
        save_loc: The location to save the Modisco h5py to.
        inverse_ic: Whether or not to perform inverse information content scaling on
          motifs. This should only be set to True if you want to revert previous IC
          scaling. If you built your MotifCompendium from Modisco motifs and did not
          explicitly turn off IC scaling, then your motifs were IC scaled and you may
          want to perform inverse IC scaling before exporting them.
        weight_col: The name of the metadata column to be used to weight motifs when
          computing motif averages. The data in the weight_col should be numeric.
        export_subpatterns: Whether or not to export the individual motifs as
          subpatterns under the cluster average patterns.

    Notes:
        Cluster names cannot have slashes (/) in them!
        If export_subpatterns is True, then motif names cannot have slashes (/) in them!
          Also, motif names will be taken from the "name" column in the MotifCompendium.
        The resultant h5py file can be fed directly into FiNeMo.
        If you are exporting your motifs to FiNeMo and your motifs were previously IC
          scaled but you don't want to revert the IC scaling, you may need to adjust the
          default trimming threshold during finemo call-hits with option -t 0.15.
    """
    if export_subpatterns and "name" not in mc.columns():
        raise KeyError(
            "If export_subpatterns is True, then the MotifCompendium must have a 'name' column."
        )
    mc_avg = mc.cluster_averages(
        clustering=cluster_name,
        aggregations=[],
        weight_col=weight_col,
    )
    mc_avg.sort("source_cluster", inplace=True)
    pos_neg = pd.Series(utils_motif.motif_posneg_sum(mc_avg.get_standard_motif_stack()))
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = mc_avg.motifs.shape[1]
        # Positive
        if "pos" in pos_neg:
            pos_group = f.create_group("pos_patterns")
            mc_avg_pos = mc_avg[pos_neg == "pos"]
            avg_motifs_pos = mc_avg_pos.get_standard_motif_stack()
            pos_pattern_names = mc_avg_pos["source_cluster"].tolist()
            for i in range(len(mc_avg_pos)):
                pattern_name = f"{pos_pattern_names[i]}_{i}"
                if "/" in pattern_name:
                    raise ValueError("Cluster names cannot have slashes (/) in them!")
                avg_motif = avg_motifs_pos[i]
                if inverse_ic:
                    avg_motif = utils_motif.ic_scale(avg_motif, invert=True)
                pos_pattern = pos_group.create_group(pattern_name)
                pos_pattern.create_dataset("contrib_scores", data=avg_motif)
                # Subpatterns
                if export_subpatterns:
                    mc_i = mc[mc[cluster_name] == pattern_name]
                    motifs_i = mc_i.get_standard_motif_stack()
                    subpattern_names_i = mc_i["name"].tolist()
                    for j in range(len(mc_i)):
                        subpattern_name = f"{subpattern_names_i[j]}_{j}"
                        if "/" in subpattern_name:
                            raise ValueError(
                                "Motif names cannot have slashes (/) in them!"
                            )
                        pos_pattern_subpattern = pos_pattern.create_group(
                            subpattern_name
                        )
                        motif = motifs_i[j]
                        if inverse_ic:
                            motif = utils_motif.ic_scale(motif, invert=True)
                        pos_pattern_subpattern.create_dataset(
                            "contrib_scores", data=motif
                        )
        # Negative
        if "neg" in pos_neg:
            neg_group = f.create_group("neg_patterns")
            mc_avg_neg = mc_avg[pos_neg == "neg"]
            avg_motifs_neg = mc_avg_neg.get_standard_motif_stack()
            neg_pattern_names = mc_avg_neg["source_cluster"].tolist()
            for i in range(len(mc_avg_neg)):
                pattern_name = f"{neg_pattern_names[i]}_{i}"
                if "/" in pattern_name:
                    raise ValueError("Cluster names cannot have slashes (/) in them!")
                avg_motif = avg_motifs_neg[i]
                if inverse_ic:
                    avg_motif = utils_motif.ic_scale(avg_motif, invert=True)
                neg_pattern = neg_group.create_group(pattern_name)
                neg_pattern.create_dataset("contrib_scores", data=avg_motif)
                # Subpatterns
                if export_subpatterns:
                    mc_i = mc[mc[cluster_name] == pattern_name]
                    motifs_i = mc_i.get_standard_motif_stack()
                    subpattern_names_i = mc_i["name"].tolist()
                    for j in range(len(mc_i)):
                        subpattern_name = f"{subpattern_names_i[j]}_{j}"
                        if "/" in subpattern_name:
                            raise ValueError(
                                "Motif names cannot have slashes (/) in them!"
                            )
                        neg_pattern_subpattern = neg_pattern.create_group(
                            subpattern_name
                        )
                        motif = motifs_i[j]
                        if inverse_ic:
                            motif = utils_motif.ic_scale(motif, invert=True)
                        neg_pattern_subpattern.create_dataset(
                            "contrib_scores", data=motif
                        )


def export_compendium_meme(
    mc: MotifCompendiumClass,
    name_col: str,
    save_loc: str,
    inverse_ic: bool = False,
) -> None:
    """Exports MotifCompendium in the MEME file format.

    Exports a MotifCompendium into a MEME file format with each motif in the
      MotifCompendium becoming a motif in the MEME output.

    Args:
        mc: The MotifCompendium to export.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the MEME file to.
        inverse_ic: Whether or not to perform inverse information content scaling on
          motifs. This should only be set to True if you want to revert previous IC
          scaling. If you built your MotifCompendium from Modisco motifs and did not
          explicitly turn off IC scaling, then your motifs were IC scaled and you may
          want to perform inverse IC scaling before exporting them.

    Notes:
        Assumes that there is a "num_seqlets" column in the MotifCompendium.
    """
    # Validate motifs
    motifs = mc.get_standard_motif_stack()
    motif_names = mc[name_col].tolist()
    num_seqlets = mc.metadata["num_seqlets"].tolist()
    # Write MEME file
    with open(save_loc, "w") as f:
        f.write("MEME version 4\n")
        f.write(f"ALPHABET= ACGT\n")
        f.write(f"strands: +\n")
        f.write(f"Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")
        for i in range(len(mc)):
            name = motif_names[i]
            motif = motifs[i]
            # Remove empty flanks
            motif = utils_motif.trim_motif(motif, 0)  # Remove zero flanks
            # Inverse IC scaling
            if inverse_ic:
                motif = utils_motif.ic_scale(motif, invert=True)
            # Write motif
            f.write(f"\nMOTIF {name}\n")
            f.write(
                f"letter-probability matrix: alength= {motif.shape[1]} w= {motif.shape[0]} nsites= {num_seqlets[i]} E= 0\n"
            )
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
                mc["motif_entropy"] = utils_motif.calculate_full_motif_entropy(
                    mc_motifs_abs
                )
            case "weighted_base_entropy":
                mc["weighted_base_entropy"] = (
                    utils_motif.calculate_weighted_base_entropy(mc_motifs_abs)
                )
            case "weighted_position_entropy":
                mc["weighted_position_entropy"] = (
                    utils_motif.calculate_weighted_position_entropy(mc_motifs_abs)
                )
            case "posbase_entropy_score":
                mc["posbase_entropy_score"] = (
                    utils_motif.calculate_position_versus_base_entropy(mc_motifs_abs)
                )
            case "copair_entropy_score":
                mc["copair_entropy_score"] = utils_motif.calculate_copair_entropy(
                    mc_motifs_abs
                )
            case "copair_composition":
                mc["copair_composition"] = utils_motif.calculate_copair_composition(
                    mc_motifs_abs
                )
            case "dinuc_entropy_score":
                mc["dinuc_entropy_score"] = utils_motif.calculate_dinucleotide_entropy(
                    mc_motifs_abs
                )
            case "dinuc_composition":
                mc["dinuc_composition"] = (
                    utils_motif.calculate_dinucleotide_alternating_composition(
                        mc_motifs_abs
                    )
                )
            case "dinuc_score":
                mc["dinuc_score"] = utils_motif.calculate_dinucleotide_score(
                    mc_motifs_abs
                )
            case "posneg_inverted":
                mc["posneg_inverted"] = (
                    utils_motif.motif_posneg_max(mc_motifs) != mc["posneg"]
                )
            case "truncated":
                max_pos = mc_motifs_abs.sum(axis=-1).argmax(axis=-1)  # (N,)
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
    ic: bool = False,
    min_score: float = 0.5,
    max_submotifs: int = 1,
    save_images: bool = True,
    logo_trim: bool | float | int = False,
) -> None:
    """Automatic labeling of motifs from a file containing PFMs.

    Given a reference file containing labeled PFMs, for each motif in the provided
      MotifCompendium, find the closest match that appears in the PFM file. The match
      score and the label of the best match are saved in the provided MotifCompendium's
      metadata. Optionally, the logos of the matched motifs can be saved as images.
      Composite matching can be enabled by setting max_submotifs > 1.

    Args:
        mc: The MotifCompendium whose motifs you want to assign labels to.
        pfm_file: The PFM file path.
        save_col_prefix: The prefix to use for the saved columns. All saved columns and
          saved images generated from the labeling process will begin with
          save_col_prefix. These columns will have the structure
          f"{save_col_prefix}_{score/name/logo}{i}".
        ic: Whether or not to apply information content scaling to the PFMs, effectively
          making them PWMs.
        min_score: The minimum similarity score to consider a match.
        max_submotifs: The maximum number of submotifs to consider in a match. If
          max_submotifs = 1, only a single match is given to each motif. If
          max_submotifs > 1, the best match for each motif can be from a combination of
          multiple reference motifs.
        save_images: Whether or not to save the logos of the matched motifs. If True,
          the logos will appear as a saved image. If False, logos will not be saved as
          saved images.
        logo_trim: This argument is only relevant if save_images is True. A bool or
          float/int indicating how the motif should be trimmed when plotting. If False,
          the motif will not be trimmed at all. If True, the motif will be trimmed at
          the flanks with a standard threshold of 1/L. If a number is provided, that
          number must be in [0, 1], and will define the trimming threshold. At a value
          of 0, only zero positions are trimmed and at a value of 1, all positions would
          be trimmed.
    """
    # Load PFM database, with same length as motifs
    L = mc.motifs.shape[1]
    pfm_motifs, pfm_names = utils_loader.load_pfm(pfm_file, ic=ic)
    # Assign labels
    mc.assign_label_from_motifs(
        pfm_motifs,
        pfm_names,
        min_score,
        max_submotifs=max_submotifs,
        save_images=save_images,
        logo_trim=logo_trim,
        save_col_prefix=save_col_prefix,
    )


def assign_label_from_other_compendium(
    assign_to_mc: MotifCompendiumClass,
    assign_from_mc: MotifCompendiumClass,
    from_label_col: str = "name",
    save_col_prefix: str = "match",
    min_score: float = 0.5,
    max_submotifs: int = 1,
    save_images: bool = True,
    logo_trim: bool | float | int = True,
) -> None:
    """Automatic labeling of motifs from another MotifCompendium.

    Given a reference MotifCompendium containing labeled motifs, for each motif in the
      unlabeled MotifCompendium, find the closest match that appears in the labeled
      MotifCompendium. The match score and the label of the best match are saved in the
      unlabeled MotifCompendium's metadata. Optionally, the logos of the matched motifs
      can be saved as images. Composite matching can be enabled by setting max_submotifs
      > 1.

    Args:
        assign_to_mc: The target MotifCompendium to assign labels to.
        assign_from_mc: The reference MotifCompendium to use labels from.
        from_label_col: The column in assign_from_mc to use as labels.
        save_col_prefix: The prefix to use for the saved columns. All saved columns and
          saved images generated from the labeling process will begin with
          save_col_prefix. These columns will have the structure
          f"{save_col_prefix}_{score/name/logo}{i}".
        min_score: The minimum similarity score to consider a match.
        max_submotifs: The maximum number of submotifs to consider in a match. If
          max_submotifs = 1, only a single match is given to each motif. If
          max_submotifs > 1, the best match for each motif can be from a combination of
          multiple reference motifs.
        save_images: Whether or not to save the logos of the matched motifs. If True,
          the logos will appear as a saved image. If False, logos will not be saved as
          saved images. The logos will come from
          assign_from_mc.get_saved_images("logo (fwd)"), if available. If not, they will
          be generated on the fly.
        logo_trim: This argument is only relevant if save_images is True. A bool or
          float/int indicating how the motif should be trimmed when plotting. If False,
          the motif will not be trimmed at all. If True, the motif will be trimmed at
          the flanks with a standard threshold of 1/L. If a number is provided, that
          number must be in [0, 1], and will define the trimming threshold. At a value
          of 0, only zero positions are trimmed and at a value of 1, all positions would
          be trimmed.
    """
    if not (
        isinstance(assign_to_mc, MotifCompendiumClass)
        and isinstance(assign_from_mc, MotifCompendiumClass)
    ):
        raise TypeError(
            "Both assign_to_mc and assign_from_mc must be MotifCompendium instances."
        )
    # Check if other_col_match exists in other MotifCompendium
    if from_label_col in assign_from_mc.metadata.columns:
        labels = assign_from_mc.metadata[from_label_col].tolist()
    else:
        raise KeyError(f"{from_label_col} not in other metadata.")
    # Check if forward logos in other MotifCompendium
    if "logo (fwd)" in assign_from_mc.images():
        other_logos = assign_from_mc.get_images("logo (fwd)")
    else:
        other_logos = None
    # Assign labels
    assign_to_mc.assign_label_from_motifs(
        assign_from_mc.motifs,
        labels,
        min_score,
        max_submotifs=max_submotifs,
        save_images=save_images,
        logo_trim=logo_trim,
        utf8_images=other_logos,
        save_col_prefix=save_col_prefix,
    )
