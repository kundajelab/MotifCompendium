import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict


def motif_to_df(motif: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker.
    """
    return pd.DataFrame(motif, columns=["A", "C", "G", "T"])


def _prep_plotting_from_motifs(cwms, clusters, sim_fb, sim_alignments, names, average):
    """Function that prepares the creation of an HTML output.
    """
    cluster_indices = []
    plotter_inputs = []
    plot_names = []
    for c_name, c in clusters.items():
        cluster_start_idx = len(plotter_inputs)
        plotter_inputs_c = []
        plot_names_c = []

        c_dfs = []
        c_df_index = set()
        first_idx = c[0]
        for idx in c:
            # Prepare df
            idx_cwm = (
                cwms[idx, :, :]
                if sim_fb[idx, first_idx] == 0
                else cwms[idx, ::-1, ::-1]
            )
            idx_shift = sim_alignments[idx, first_idx]
            idx_df = pd.DataFrame(idx_cwm, columns=["A", "C", "G", "T"])
            idx_df.index += idx_shift
            idx_name = names[idx]
            # Update lists
            c_dfs.append(idx_df)
            c_df_index.update(idx_df.index)
            plot_names_c.append(idx_name)
        # Update/align index
        c_df_index = sorted(c_df_index)
        c_dfs = [x.reindex(c_df_index, fill_value=0) for x in c_dfs]
        # Prep plotting inputs
        plotter_inputs_c = [(x, "white") for x in c_dfs]
        assert len(plotter_inputs_c) == len(plot_names_c)
        # Create average plot if needed
        if average:
            motifs_concat = pd.concat(c_dfs)
            average_motif = motifs_concat.groupby(motifs_concat.index).mean()
            plotter_inputs_c.insert(0, (average_motif, "palegreen"))
            plot_names_c.insert(0, "AVERAGE")
        # Add to total
        plotter_inputs += plotter_inputs_c
        plot_names += plot_names_c
        cluster_end_idx = len(plotter_inputs)
        cluster_indices.append((c_name, cluster_start_idx, cluster_end_idx))
    assert len(cluster_indices) == len(clusters)
    return cluster_indices, plotter_inputs, plot_names


def create_html(
    cwms,
    clusters,
    sim_fb,
    sim_alignments,
    names,
    html_out_loc,
    average=True,
    max_parallel=16,
):
    """Creates an html output of motifs grouped into clusters.
    """
    from .make_report import generate_report

    clustering = defaultdict(list)
    for i, c in enumerate(clusters):
        clustering[c].append(i)
    cluster_indices, plotter_inputs, plot_names = _prep_plotting_from_motifs(
        cwms, clustering, sim_fb, sim_alignments, names, average
    )
    generate_report(
        cluster_indices, plotter_inputs, plot_names, html_out_loc, max_parallel
    )


def plot_motif_on_ax(motif, ax, motif_name=None):
    """Plots a motif on an axis.
    """
    motif_df = motif_to_df(motif)
    logomaker.Logo(motif_df, ax=ax)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if motif_name is not None:
        ax.set_title(motif_name)


def plot_motifs(motifs, motif_names=None):
    """Creates a plot of multiple motifs.
    """
    if type(motifs) == np.ndarray:
        verify_motif_stack(motifs)
        motifs = [motifs[i, :, :] for i in range(motifs.shape[0])]
    # check_error(type(motifs) == list, "error: motifs must be provided as a list or an np.ndarray.")
    if motif_names is None:
        motif_names = [None for x in range(len(motifs))]
    # check_error(len(motif_names) == len(motifs), "error: number of motifs and names not matching.")
    fig, axs = plt.subplots(len(motifs), 1)
    for i in range(len(motifs)):
        plot_motif_on_ax(motifs[i], axs[i], motif_names[i])
    plt.show()


def plot_heatmap(data, annot=False, labels=None, show=False, save_loc=None):
    """Plots heatmaps.
    """
    if labels is None:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data, index=labels, columns=labels)
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(df, annot=annot)
    if save_loc is not None:
        heatmap.get_figure().savefig(save_loc)
    if show:
        plt.show()

