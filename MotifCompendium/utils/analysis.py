import os
import multiprocessing

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import MotifCompendium
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.plotting as utils_plotting


#######################
# SIMILARITY ANALYSES #
#######################
def plot_similarity_distribution(
    mc: MotifCompendium,
    save_loc: str,
    vals: list[float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    n_per: int = 5,
) -> None:
    """Plots examples of various similarities in a MotifCompendium.

    For a set of similarities, create an html file displaying multiple examples of pairs
      of motifs that have a similarity within +-0.01 of the specified similarity values.

    Args:
        mc: The MotifCompendium to analyze.
        save_loc: The location where to save the output html file.
        vals: The list of similarity scores to display examples of.
        n_per: The number of examples of each similarity score to display.
    """
    clustering = [False for _ in range(len(mc))]
    similarity = mc.similarity
    for val in vals:
        # Find all locations where similarity is val-0.005 < similarity < val+0.005
        indices = np.where(np.abs(similarity - val) < 0.005)
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


def judge_clustering(mc: MotifCompendium, clustering: str, save_loc: str) -> None:
    """Plots histograms of inter-cluster and intra-cluster similarities.

    Judges a motif clustering by computing the quality of the clustering and then
      plotting the distribution of minimum intercluster similarities as well as plotting
      the distribution of the maximum intracluster similarity.

    Args:
        mc: The MotifCompendium to analyze.
        clustering: The motif clustering to judge.
        save_loc: The file prefix to save the clustering quality and the clustering
          quality plot to.
    """
    # Get clustering quality
    clustering_quality = mc.clustering_quality(clustering)
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
    plt.suptitle(f"{clustering} ({n_clusters} clusters)")
    plt.savefig(save_loc)
    plt.close()


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
    import upsetplot

    clusters_by_grouping = upsetplot.from_memberships(membership_lists)
    upsetplot.UpSet(clusters_by_grouping, subset_size="count", **kwargs).plot()
    plt.savefig(save_loc, bbox_inches="tight")
    plt.close()


def export_clusters_modisco(
    mc: MotifCompendium,
    cluster_name: str,
    save_loc: str,
    ic: bool = False,
    max_chunk: int | None = None,
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
    sim_type: str = "l2",
) -> None:
    """Exports cluster averages in the Modisco file format.

    Given a clustering, compute cluster averages and export them in an h5py structure
      that matches Modisco outputs.

    Args:
        mc: The MotifCompendium to analyze.
        cluster_name: The motif clustering to compute average motifs on.
        save_loc: The location to save the Modisco h5py to.
        ic: Whether or not to revert IC-scaled motifs back to linear space, by applying
          inverse IC-scaling.
        max_chunk: The maximum number of motifs to compute similarity on at a time.
        max_cpus: The maximum number of CPUs to use for computing similarity (only used
          if use_gpu is False).
        use_gpu: Whether or not to use GPUs to accelerate computing similarity.
        sim_type: The type of similarity metric to compute: 'l2', 'sqrt', 'jss'
        safe: Whether or not to construct the MotifCompendium safely.

    Notes:
        The resultant h5py file can be fed directly into FiNeMo (hitcaller).
    """
    mc_cluster_avg = mc.cluster_averages(
        cluster_name,
        aggregations=[],
        max_chunk=max_chunk,
        max_cpus=max_cpus,
        use_gpu=use_gpu,
        sim_type=sim_type,
        safe=False,
    )  # kwargs for new avg similarity calculations
    pos_neg = np.sum(mc_cluster_avg.motifs, axis=(1, 2)) > 0
    pos_neg = ["pos" if x > 0 else "neg" for x in pos_neg]
    mc_cluster_avg.metadata["pos_neg"] = pos_neg
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = 30
        # Positive
        if "pos" in set(mc_cluster_avg["pos_neg"]):
            pos_group = f.create_group("pos_patterns")
            mc_cluster_avg_pos = mc_cluster_avg[mc_cluster_avg["pos_neg"] == "pos"]
            for i in range(len(mc_cluster_avg_pos)):
                name = str(mc_cluster_avg_pos.metadata.loc[i, "name"])
                cwm = mc_cluster_avg_pos.motifs[i, :, :]
                if ic:
                    cwm = utils_motif.ic_invert(cwm)
                pos_cluster = pos_group.create_group(name)
                pos_cluster.create_dataset("contrib_scores", data=cwm)
        # Negative
        if "neg" in set(mc_cluster_avg["pos_neg"]):
            neg_group = f.create_group("neg_patterns")
            mc_cluster_avg_neg = mc_cluster_avg[mc_cluster_avg["pos_neg"] == "neg"]
            for i in range(len(mc_cluster_avg_neg)):
                name = str(mc_cluster_avg_neg.metadata.loc[i, "name"])
                cwm = mc_cluster_avg_neg.motifs[i, :, :]
                if ic:
                    cwm = utils_motif.ic_invert(cwm)
                neg_cluster = neg_group.create_group(name)
                neg_cluster.create_dataset("contrib_scores", data=cwm)


def export_modisco(
    mc: MotifCompendium,
    save_loc: str,
    name_col: str = "name",
    ic: bool = False,
) -> None:
    """Exports MotifCompendium in the Modisco file format.

    Assumes that the MotifCompendium is already clustered and just exports it to an h5py
      structure that matches Modisco outputs.

    Args:
        mc: The MotifCompendium to analyze.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the Modisco h5py to.
        ic: Whether or not to revert IC-scaled motifs back to linear space.
    Notes:
        The motif names cannot have slashes (/) in them!
        The resultant h5py file can be fed directly into FiNeMo (hitcaller).
    """
    size_4 = mc.motifs.shape[2] == 4
    motifs = mc.motifs if size_4 else utils_motif.motif_8_to_4(mc.motifs)
    pos_neg = np.sum(motifs, axis=(1, 2)) > 0
    pos_neg = ["pos" if x > 0 else "neg" for x in pos_neg]
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = 30
        # Positive
        if "pos" in pos_neg:
            pos_group = f.create_group("pos_patterns")
            mc_pos = mc[pd.Series(pos_neg) == "pos"]
            for i in range(len(mc_pos)):
                name = str(mc_pos.metadata.loc[i, name_col])
                assert "/" not in name
                cwm = (
                    mc_pos.motifs[i, :, :]
                    if size_4
                    else utils_motif.motif_8_to_4(mc_pos.motifs[i, :, :])
                )
                if ic:
                    cwm = utils_motif.ic_invert(cwm)

                pos_cluster = pos_group.create_group(name)
                pos_cluster.create_dataset("contrib_scores", data=cwm)
        # Negative
        if "neg" in pos_neg:
            neg_group = f.create_group("neg_patterns")
            mc_neg = mc[pd.Series(pos_neg) == "neg"]
            for i in range(len(mc_neg)):
                name = str(mc_neg.metadata.loc[i, name_col])
                assert "/" not in name
                cwm = (
                    mc_neg.motifs[i, :, :]
                    if size_4
                    else utils_motif.motif_8_to_4(mc_neg.motifs[i, :, :])
                )
                if ic:
                    cwm = utils_motif.ic_invert(cwm)

                neg_cluster = neg_group.create_group(name)
                neg_cluster.create_dataset("contrib_scores", data=cwm)


####################
# ENTROPY ANALYSES #
####################
def calculate_entropy(
    mc: MotifCompendium,
    entropy_list: list[str] = [
        "motif_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
    ],
) -> None:
    """Calculate entropy metrics, to quantify motif information complexity.
    Update metadata table with entropy metric values.

    List of Entropy metrics:
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
        entropy_list: List of entropy metrics to calculate.
          Possible values: ['motif_entropy', 'posbase_entropy_ratio',
          'copair_entropy_ratio', 'dinuc_entropy_ratio']
    """
    # Check if entropy metrics are valid
    entropy_list = list(set(entropy_list))  # Convert entropy_list into a unique list
    valid_entropy_metrics = [
        "motif_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
    ]
    for entropy_metric in entropy_list:
        if entropy_metric not in valid_entropy_metrics:
            raise ValueError(
                f"Entropy metric {entropy_metric} is not valid. Must be one of: {valid_entropy_metrics}"
            )

    # Calculate entropy metrics
    for entropy_metric in entropy_list:
        metrics_list = []
        match entropy_metric:
            case "motif_entropy":
                for i in range(mc.motifs.shape[0]):
                    metric = utils_motif.calculate_motif_entropy(mc.motifs[i])
                    metrics_list.append(metric)
                mc["motif_entropy"] = metrics_list

            case "posbase_entropy_ratio":
                for i in range(mc.motifs.shape[0]):
                    metric = utils_motif.calculate_posbase_entropy_ratio(mc.motifs[i])
                    metrics_list.append(metric)
                mc["posbase_entropy_ratio"] = metrics_list

            case "copair_entropy_ratio":
                for i in range(mc.motifs.shape[0]):
                    metric = utils_motif.calculate_copair_entropy_ratio(mc.motifs[i])
                    metrics_list.append(metric)
                mc["copair_entropy_ratio"] = metrics_list

            case "dinuc_entropy_ratio":
                for i in range(mc.motifs.shape[0]):
                    metric = utils_motif.calculate_dinuc_entropy_ratio(mc.motifs[i])
                    metrics_list.append(metric)
                mc["dinuc_entropy_ratio"] = metrics_list

            case _:
                raise ValueError(
                    f"Entropy metric {entropy_metric} is not valid. Must be one of: {valid_entropy_metrics}"
                )


###########################
# EXISTING MOTIF DATABASE #
###########################
def label_from_pfms(
    mc: MotifCompendium,
    pfm_file: str,
    save_col_sim: str = "pfm_match_similarity",
    save_col_match: str = "pfm_match",
    save_col_logo: str | None = None,
    max_chunk: int | None = None,
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
    sim_type: str = "l2",
) -> None:
    """Automatic labeling of motifs from a pfm file.

    For each motif in the MotifCompendium, computes the similarity between that motif
      and all motifs in the PFM file. The highest similarity and closest motif match
      will be saved as columns save_col_sim and save_col_match in the MotifCompendium
      metadata.

    Args:
        mc: The MotifCompendium to analyze.
        pfm_file: The PFM file path.
        save_col_sim: The column under which the highest similarity will be stored.
        save_col_match: The column under which the closest match will be stored.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_cpus: The maximum number of CPUs to use for computing similarity. If None,
          it will only use a single CPU.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        sim_type: The type of similarity metric to compute: 'l2', 'sqrt', 'jss'
    """
    if pfm_file.endswith("_pfms.txt"):
        pfm_motifs, names = utils_loader.load_pfm(pfm_file)
    elif pfm_file.endswith(".meme.txt") or pfm_file.endswith(".meme"):
        pfm_motifs, names = utils_loader.load_meme(pfm_file)
    else:
        raise ValueError("pfm_file must be a _pfm.txt, .meme, or .meme.txt file.")
    mc_motifs = mc.motifs
    if mc_motifs.shape[2] == 8:
        mc_motifs = utils_motif.motif_8_to_4_abs(mc_motifs)
    pfm_similarity, _, _ = utils_similarity.compute_similarities_and_alignments(
        [mc_motifs, pfm_motifs], [(0, 1)], max_chunk, max_cpus, use_gpu, sim_type=sim_type
    )[0]
    mc[save_col_sim] = np.max(pfm_similarity, axis=1)
    mc[save_col_match] = [names[x] for x in np.argmax(pfm_similarity, axis=1)]

    # Create matched pfm logos
    if save_col_logo:
        save_col_logo = "LOGOIMAGEDATA__" + save_col_logo
        match_motif_dicts = [
            {"motif": utils_motif.motif_to_df(pfm_motifs[x])} for x in np.argmax(pfm_similarity, axis=1)
        ]
        # Create motif plots
        if max_cpus is None:
            match_motif_strings = []
            for i in range(len(match_motif_dicts)):
                match_motif_strings.append(utils_plotting._motifdict_to_utf8_plot(match_motif_dicts[i]))
        else:
            # Plot in parallel
            num_processes = min(
                max_cpus, multiprocessing.cpu_count()
            )  # don't use more CPUs than available
            with multiprocessing.Pool(processes=num_processes) as p:
                match_motif_strings = p.map(utils_plotting._motifdict_to_utf8_plot, match_motif_dicts)
        mc[save_col_logo] = match_motif_strings


def label_composites_from_pfms(
    mc: MotifCompendium,
    pfm_file: str,
    min_score: float = 0.5,
    max_constituents: int = 3,
    save_col_comp_sim: str = "pfm_composite_similarity",
    save_col_comp_match: str = "pfm_composite_match",
    save_col_logo: str | None = None,
    max_chunk: int | None = None,
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
    sim_type: str = "l2",
) -> None:
    """Label composite motifs from a pfm file.
    
    For each motif in the MotifCompendium, computes the best similarity and alignment
      between that motif and all motifs in the PFM file. The best matching PFM motif
      is subtracted from the motif. With the remaining motif, another best similarity
      is calculated against all motifs in the PFM file. The process continue for 
      the specified number of times. The composite score is the similarity score during. 
      the final iteration. Composite score and best matching PFM motifs are saved as columns
      in the MotifCompendium metadata.
    
    Args:
        mc: The MotifCompendium to analyze.
        pfm_file: The PFM file path.
        min_score: The minimum similarity score to consider as a composite match.
        max_constituents: The maximum number of constituent motifs to conside in a composite.
        save_col_comp_sim: The columns under which the final composite similarity scores 
          will be stored, as save_col_comp_sim0, save_col_comp_sim1, etc.
        save_col_comp_match: The columns under which the composite matches will be stored,
          as save_col_comp_match0, save_col_comp_match1, etc.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_cpus: The maximum number of CPUs to use for computing similarity. If None,
          it will only use a single CPU.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        sim_type: The type of similarity metric to compute: 'l2', 'sqrt', 'jss'"""
    # Load pfm
    if pfm_file.endswith("_pfms.txt"):
        pfm_motifs, pfm_names = utils_loader.load_pfm(pfm_file)
    elif pfm_file.endswith(".meme.txt") or pfm_file.endswith(".meme"):
        pfm_motifs, pfm_names = utils_loader.load_meme(pfm_file)
    else:
        raise ValueError("pfm_file must be a _pfm.txt, .meme, or .meme.txt file.")
    mc_motifs = mc.motifs
    if mc_motifs.shape[2] == 8:
        mc_motifs = utils_motif.motif_8_to_4_abs(mc_motifs)

    # Pre-normalize motifs
    mc_motifs = utils_motif.normalize_motifs(mc_motifs, sim_type=sim_type)
    pfm_motifs = utils_motif.normalize_motifs(pfm_motifs, sim_type=sim_type)

    # Perform un-normalized similarity calculations
    match sim_type:
        case "l2" | "sqrt" | "dot":
            calc_type = "dot"
        case "jss" | "log":
            calc_type = "log"

    # Calculate composite similarity
    composite_scores = []
    composite_names = []
    composite_idxs = []
    iter = 0
    max_composite_score = 1
    while iter < max_constituents and max_composite_score > min_score:
        pfm_similarity, pfm_alignfr, pfm_alignh = utils_similarity.compute_similarities_and_alignments(
            [mc_motifs, pfm_motifs], [(0, 1)], 
            max_chunk, max_cpus, use_gpu, sim_type=calc_type, safe=False,)[0]

        composite_score = np.max(pfm_similarity, axis=1) / _calculate_composite_scale(iter+1, sim_type)
        pfm_match_idx = np.argmax(pfm_similarity, axis=1)
        pfm_alignfr = pfm_alignfr[np.arange(pfm_alignfr.shape[0]), pfm_match_idx]
        pfm_alignh = pfm_alignh[np.arange(pfm_alignh.shape[0]), pfm_match_idx]
        match_names = []
        for i, x in enumerate(pfm_match_idx):
            if composite_score[i] > min_score:
                match_names.append(pfm_names[x])
            else:
                match_names.append(None)
                composite_score[i] = 0
        composite_scores.append(composite_score)
        composite_names.append(match_names)
        composite_idxs.append(pfm_match_idx)

        # Subtract best match
        mc_motifs = utils_motif.subtract_motifs(mc_motifs, pfm_motifs, pfm_match_idx, pfm_alignfr, pfm_alignh)
        
        # Iterate
        max_composite_score = np.max(composite_score) # To determine whether to continue        
        iter += 1

    # Save in metadata table
    for i in range(iter):
        mc[f"{save_col_comp_sim}{i}"] = composite_scores[i]
        mc[f"{save_col_comp_match}{i}"] = composite_names[i]
    
    # Create matched pfm logos
    if save_col_logo:
        for i in range(iter):
            match_motif_dicts = []
            for j, x in enumerate(composite_idxs[i]):
                if composite_scores[i][j] > min_score:
                    match_motif_dicts.append({"motif": utils_motif.motif_to_df(pfm_motifs[x])})
                else:
                    match_motif_dicts.append(None)
            # Create motif plots
            if max_cpus is None:
                match_motif_strings = []
                for j in range(len(match_motif_dicts)):
                    match_motif_strings.append(utils_plotting._motifdict_to_utf8_plot(match_motif_dicts[j]))
            else:
                # Plot in parallel
                num_processes = min(
                    max_cpus, multiprocessing.cpu_count()
                )  # don't use more CPUs than available
                with multiprocessing.Pool(processes=num_processes) as p:
                    match_motif_strings = p.map(utils_plotting._motifdict_to_utf8_plot, match_motif_dicts)
        
            # Save in metadata table
            mc[f"{save_col_logo}{i}"] = match_motif_strings
    

def _calculate_composite_scale(iter: int, sim_type: str) -> float:
    """Calculate the scale factor for composite similarity calculation,
      based on the number of iterations and similarity metric type."""
    if iter > 1:
        match sim_type:
            case "l2":
                return 1 / np.sqrt(iter)
            case "jss":
                return 1 - np.sqrt(0.5 * (-np.log(iter) 
                    - (iter + 1) / iter * np.log((iter + 1) / (2 * iter)) 
                    + (iter - 1) / iter * np.log(2 * iter)))
            case _:
                return 1.0
    else:
        return 1.0