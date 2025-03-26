import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot

import MotifCompendium
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.plotting as utils_plotting
import MotifCompendium.utils.similarity as utils_similarity


#######################
# SIMILARITY ANALYSES #
#######################
def plot_similarity_distribution(
    mc: MotifCompendium,
    save_loc: str,
    vals: list[float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    tolerance: float = 0.001,
    n_per: int = 3,
) -> None:
    """Plots examples of various similarities in a MotifCompendium.

    For a set of similarities, create an html file displaying multiple examples of pairs
      of motifs that have a similarity within +-tolerance of the specified similarity
      values.

    Args:
        mc: The MotifCompendium to analyze.
        save_loc: The location where to save the output html file.
        vals: The list of similarity scores to display examples of.
        tolerance: The tolerance of error with respect to target similarity values to display.
        n_per: The number of examples of each similarity score to display.
    """
    clustering = [False for _ in range(len(mc))]
    similarity = mc.similarity
    for val in vals:
        # Find all locations where similarity is val-tolerance < similarity < val+tolerance
        indices = np.where(np.abs(similarity - val) < tolerance)
        indices = list(zip(indices[0], indices[1]))
        p = 1
        for i, j in indices:
            if not (clustering[i] or clustering[j]):
                clustering[i] = f"Similarity {val} example {p}"
                clustering[j] = f"Similarity {val} example {p}"
                p += 1
            if p > n_per:
                break
    # Create MotifCompendium of only displayed motifs
    clustering_series = pd.Series(clustering)
    mc_distribution = mc[clustering_series != False]
    distribution_clusters = clustering_series[clustering_series != False].tolist()
    mc_distribution.motif_collection_html(
        save_loc, distribution_clusters, average_motif=False
    )


def plot_ground_truth_mismatch(
    mc: MotifCompendium,
    ground_truth: str,
    save_loc: str,
    similarity_threshold: float = 0.95,
    max_examples: int = 100,
) -> None:
    """Plots examples of when similarity did not match a ground truth motif grouping.

    Given a MotifCompendium, a ground truth grouping of motifs, and a similarity
      threshold, plot examples of when motif similarities did not match with the ground
      truth. Plot examples where two motifs are grouped together in the ground truth but
      have a lower similarity than the threshold. Also plot examples where two motifs
      are not grouped together in the ground turth but havea  higher similarity than the
      threshold.

    Args:
        mc: The MotifCompendium to analyze.
        ground_truth: The column in the MotifCompendium metadata to use as the ground
          truth label.
        save_loc: The location where to save the output html file.
        similarity_threshold: The similarity value to threshold at.
        max_examples: The maximum number of mismatch examples to plot.
    """
    quality = mc.clustering_quality(ground_truth, with_names=True)
    clustering = [False for _ in range(len(mc))]
    # Low internal similarity
    n_examples = 0
    for i, c in enumerate(quality.columns):
        if quality[i, i] >= similarity_threshold:
            continue
        c_select = mc[ground_truth] == c
        similarity_slice_ii_df = mc.get_similarity_slice(c_select, c_select)
        similarity_slice_ii_df_stacked = similarity_slice_ii_df.stack()
        row_label, col_label = similarity_slice_ii_df_stacked.idxmin()
        clustering[row_label] = f"Low internal similarity {c} ({quality[i, i]:.3})"
        clustering[col_label] = f"Low internal similarity {c} ({quality[i, i]:.3})"
        n_examples += 1
        if n_examples >= (max_examples) // 2:
            break
    # High external similarity
    n_examples = 0
    for i, ci in enumerate(quality.columns):
        for j, cj in enumerate(quality.columns):
            if j <= i:
                continue
            if quality[i, j] < similarity_threshold:
                continue
            similarity_slice_ij_df = mc.get_similarity_slice(
                mc[ground_truth] == ci, mc[ground_truth] == cj
            )
            similarity_slice_ij_df_stacked = similarity_slice_ij_df.stack()
            row_label, col_label = similarity_slice_ij_df_stacked.idxmax()
            clustering[row_label] = (
                f"High external similarity {ci} & {cj} ({quality[i, j]:.3})"
            )
            clustering[col_label] = (
                f"High external similarity {ci} & {cj} ({quality[i, j]:.3})"
            )
            n_examples += 1
            if n_examples >= (max_examples) // 2:
                break
        if n_examples >= (max_examples) // 2:
            break
    # Create MotifCompendium of only displayed motifs
    clustering_series = pd.Series(clustering)
    mc_mismatch = mc[clustering_series != False]
    mismatch_clusters = clustering_series[clustering_series != False].tolist()
    mc_mismatch.motif_collection_html(save_loc, mismatch_clusters, average_motif=False)


def judge_clustering(mc: MotifCompendium, cluster_col: str, save_loc: str) -> None:
    """Plots histograms of inter-cluster and intra-cluster similarities.

    Judges a motif clustering by computing the quality of the clustering and then
      plotting the distribution of minimum intercluster similarities as well as plotting
      the distribution of the maximum intracluster similarity.

    Args:
        mc: The MotifCompendium to analyze.
        cluster_col: The motif clustering to judge.
        save_loc: The file prefix to save the clustering quality and the clustering
          quality plot to.
    """
    # Get clustering quality
    clustering_quality = mc.clustering_quality(cluster_col)
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    sns.histplot(np.diag(clustering_quality), ax=axs[0], stat="proportion", kde=True)
    axs[0].set_title("worst intra-cluster similarities")
    n_clusters = clustering_quality.shape[0]
    triu = [
        clustering_quality[i, j]
        for i in range(n_clusters)
        for j in range(i + 1, n_clusters)
    ]
    sns.histplot(triu, ax=axs[1], stat="proportion", kde=True)
    axs[1].set_title("best inter-cluster similarities")
    axs[1].set_xlabel("similarity")
    plt.suptitle(f"{cluster_col} ({n_clusters} clusters)")
    plt.savefig(save_loc)
    plt.close(fig)


#######################
# DOWNSTREAM ANALYSES #
#######################
def plot_unique_per_cluster(mc: MotifCompendium, group_by: str, save_loc: str) -> None:
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
    mc: MotifCompendium, clustering: str, grouping: str, save_loc: str, **kwargs
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
    mc: MotifCompendium,
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
    pos_neg = utils_motif.motif_posneg(motifs)
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
    mc: MotifCompendium,
    cluster_name: str,
    save_loc: str,
    inverse_ic: bool = False,
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
    mc_cluster_avg = mc.cluster_averages(
        cluster_name,
        aggregations=[],
        safe=False,
    )
    mc_cluster_avg["name"] = (mc_cluster_avg["name"].str.split("#").str[1]).astype(
        mc[cluster_name].dtype
    )
    mc_cluster_avg.sort_values("name", inplace=True)
    export_full_compendium_modisco(
        mc_cluster_avg, "name", save_loc, inverse_ic=inverse_ic
    )


####################
# ENTROPY ANALYSES #
####################
def calculate_filters(
    mc: MotifCompendium,
    metric_list: list[str] = [
        "motif_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
    ],
) -> None:
    """Calculate filter metrics, to be used for filtering low quality motifs.
    Update metadata table with filter metric values.

    List of filter metrics:
        (1) Motif entropy:
            Calculation: Shannon entropy on (L,8)
            Purpose:    (Low) Archetype #1: Sharp nucleotide peak (e.g., G)
                        (High) Archetype #2: Noise/chaos
        (2) Pos-base entropy ratio:
            Calculation: Position-wise entropy on (L,) / Base-wise entropy on (8,)
            Purpose:    (High) Archetype #3: Single nucleotide repeats (e.g., AAAAA, GGGGG)
        (3) Co-pair entropy ratio:
            Calculation: Entropy across position (L,) /
                Entropy across all pairs of co-occurring, non-repeating bases (28,)
            Purpose:    (High) Archetype #4: High GC, AT bias
        (4) Dinucleotide entropy ratio:
            Calculation: Entropy across pairs of positions (L/2,) /
                Entropy across all dinucleotide pairs (64,)
            Purpose:    (High) Dinucleotide repeats (e.g., GCGCGC, ATATAT)

    Args:
        metric_list: List of filter metrics to calculate.
          Possible values: ['motif_entropy', 'posbase_entropy_ratio',
          'copair_entropy_ratio', 'dinuc_entropy_ratio']
    """
    # Check if filter metrics are valid
    metric_list = list(set(metric_list))  # Convert metric_list into a unique list
    valid_filter_metrics = [
        "motif_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
    ]
    for filter_metric in metric_list:
        if filter_metric not in valid_filter_metrics:
            raise ValueError(
                f"Filter metric {filter_metric} is not valid. Must be one of: {valid_filter_metrics}"
            )

    # Calculate filter metrics
    for filter_metric in metric_list:
        metrics_list = []
        match filter_metric:
            case "motif_entropy":
                for motif in mc.motifs:
                    metric = utils_motif.calculate_motif_entropy(motif)
                    metrics_list.append(metric)
                mc["motif_entropy"] = metrics_list

            case "posbase_entropy_ratio":
                for motif in mc.motifs:
                    metric = utils_motif.calculate_posbase_entropy_ratio(motif)
                    metrics_list.append(metric)
                mc["posbase_entropy_ratio"] = metrics_list

            case "copair_entropy_ratio":
                for motif in mc.motifs:
                    metric = utils_motif.calculate_copair_entropy_ratio(motif)
                    metrics_list.append(metric)
                mc["copair_entropy_ratio"] = metrics_list

            case "dinuc_entropy_ratio":
                for motif in mc.motifs:
                    metric = utils_motif.calculate_dinuc_entropy_ratio(motif)
                    metrics_list.append(metric)
                mc["dinuc_entropy_ratio"] = metrics_list

            case "negpattern_pospeak":
                for i in range(len(mc)):
                    if mc.metadata["posneg"][i] == "neg":
                        metric = utils_motif.check_negpattern_pospeak(mc.motifs[i])
                        metrics_list.append(metric)
                mc["negpattern_pospeak"] = metrics_list

            case _:
                raise ValueError(
                    f"filter metric {filter_metric} is not valid. Must be one of: {valid_filter_metrics}"
                )


###########################
# EXISTING MOTIF DATABASE #
###########################
def label_from_pfms(
    mc: MotifCompendium,
    pfm_file: str,
    save_col_sim: str = "pfm_match_similarity",
    save_col_name: str = "pfm_match_name",
    save_col_logo: str = "pfm_match_logo",
    max_submotifs: int = 1,
    min_score: float = 0.5,
) -> None:
    """Automatic labeling of motifs from a pfm file.

    For each motif in the MotifCompendium, computes the similarity between that motif
      and all motifs in the PFM file. The highest similarity and closest motif match
      will be saved as columns save_col_sim and save_col_name in the MotifCompendium
      metadata.

    Args:
        mc: The MotifCompendium to analyze.
        pfm_file: The PFM file path.
        save_col_sim: The column under which the highest similarity will be stored.
        save_col_name: The column under which the closest match will be stored.
        save_col_logo: The column under which the closest match logo will be stored.
        max_submotifs: The maximum number of submotifs to consider in a match.
        min_score: The minimum similarity score to consider as a match.
    """
    # Load PFM database, with same length as motifs
    L = mc.motifs.shape[1]
    if pfm_file.endswith("_pfms.txt"):
        pfm_motifs, names = utils_loader.load_pfm(pfm_file, L)
    elif pfm_file.endswith(".meme.txt") or pfm_file.endswith(".meme"):
        pfm_motifs, names = utils_loader.load_meme(pfm_file, L)
    else:
        raise ValueError("pfm_file must be a _pfm.txt, .meme, or .meme.txt file.")

    mc_motifs = mc.motifs.copy()
    if mc_motifs.shape[2] == 8:
        mc_motifs = utils_motif.motif_8_to_4_unsigned(mc_motifs)

    # Find best match, per iteration
    match_scores = []
    match_names = []
    match_motifs = []
    match_idxs = []
    for i in range(max_submotifs):
        # Calcualte similarity
        pfm_sim, pfm_align_rc, pfm_align_h = utils_similarity.compute_similarities(
            [mc_motifs, pfm_motifs],
            [(0, 1)],
        )[0]
        # Unscale L2 similarity, for dot product only
        pfm_sim = pfm_sim * (
            np.linalg.norm(mc_motifs, axis=(1, 2))[:, np.newaxis]
            * np.linalg.norm(pfm_motifs, axis=(1, 2))[np.newaxis, :]
        )  # (N, N)
        # Scale score by i
        match_score = np.max(pfm_sim, axis=1) * np.sqrt(i)  # (N,)
        pfm_match_idx = np.argmax(pfm_sim, axis=1)  # (N,)
        pfm_align_rc = pfm_align_rc[
            np.arange(pfm_align_rc.shape[0]), pfm_match_idx
        ]  # (N,)
        pfm_align_h = pfm_align_h[
            np.arange(pfm_align_h.shape[0]), pfm_match_idx
        ]  # (N,)
        match_motif = pfm_motifs[pfm_match_idx, :, :]

        # Subtract best match
        mc_motifs = utils_motif.subtract_motifs(
            mc_motifs, match_motif, pfm_align_rc, pfm_align_h
        )

        # Save match information
        match_name = [names[x] for x in pfm_match_idx]
        match_motifs.append(match_motif)
        match_scores.append(match_score)
        match_names.append(match_name)
        match_idxs.append(pfm_match_idx)

    # Save match information
    for i in range(max_submotifs):
        if max_submotifs == 1:
            i = ""
        mc[f"{save_col_sim}{i}"] = match_scores[i]
        mc[f"{save_col_name}{i}"] = match_names[i]
        match_motif = match_motifs[i]
        if match_motif.shape[2] == 8:
            match_motif = utils_motif.motif_8_to_4_signed(match_motif)
        motif_plotting_inputs = [
            utils_plotting.LogoPlottingInput(motif) for motif in match_motif
        ]
        mc.__images[f"{save_col_logo}{i}"] = [
            motif_input.utf8_plot
            for motif_input in utils_plotting.plot_many_motif_logos(
                motif_plotting_inputs
            )
        ]

        # Remove matches below match_score < min_score
        mc[f"{save_col_name}{i}"][mc[f"{save_col_sim}{i}"] < min_score] = None
        mc.__images[f"{save_col_logo}{i}"][mc[f"{save_col_sim}{i}"] < min_score] = ""
