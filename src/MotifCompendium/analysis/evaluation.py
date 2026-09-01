import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot

from MotifCompendium import MotifCompendium as MotifCompendiumClass


#######################
# SIMILARITY ANALYSES #
#######################
def plot_similarity_distribution(
    mc: MotifCompendiumClass,
    save_loc: str,
    vals: list[float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    *,
    tolerance: float = 0.001,
    n_per: int = 3,
) -> None:
    """Plots examples of various similarities in a MotifCompendium.

    For a set of similarities, create an html file displaying multiple examples of pairs
    of motifs that have a similarity within [val, val+tolerance) for each similarity
    value.

    Args:
        mc: The MotifCompendium to plot the similarity distribution for.
        save_loc: The location where to save the output HTML file.
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
    *,
    max_examples: int = 100,
) -> None:
    """Plots examples of when similarity did not match a motif clustering.

    Given a MotifCompendium, a clustering/grouping of motifs, and a similarity threshold,
    plot examples of when motif similarities did not match with the clustering. Plot
    examples where two motifs are clustered together in the clustering but have a lower
    similarity than the threshold. Also, plot examples where two motifs are not clustered
    together in the ground truth but have a higher similarity than the threshold.

    Args:
        mc: The MotifCompendium to plot clustering mismatch examples for.
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
    *,
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

    Note:
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
    mc: MotifCompendiumClass,
    clustering: str,
    grouping: str,
    *,
    show: bool = False,
    save_loc: str | None = None,
    **kwargs,
) -> None:
    """Creates an upset plot of how many motif clusters span across different groups.

    Given a grouping that each motif belongs (ex: source celltype) to and a clustering of
    motifs, create an Upset Plot that displays which clusters belong to which groups.

    Args:
        mc: The MotifCompendium to analyze.
        clustering: The motif clustering to consider.
        grouping: The grouping to compute cluster source intersections with respect to.
        show: Whether or not to show the Upset Plot with plt.show().
        save_loc: The file to save the Upset Plot to. If None, the heatmap is not saved.
        **kwargs: Additional named arguments that usetplot.UpSet() takes.

    Note:
        Requires package upsetplot to run.
        Consider running with argument min_subset_size.
    """
    membership_lists = [
        list(set(mc[mc[clustering] == c][grouping])) for c in set(mc[clustering])
    ]
    clusters_by_grouping = upsetplot.from_memberships(membership_lists)
    fig = plt.figure()
    upsetplot.UpSet(clusters_by_grouping, subset_size="count", **kwargs).plot(fig=fig)
    # Save/show/close
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig=fig)
