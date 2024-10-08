import base64
from collections import defaultdict
import io
import multiprocessing
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader
import logomaker
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


####################
# PUBLIC FUNCTIONS #
####################
def motif_collection_html(
    motif_groups: dict[str, list[dict[str, Any]]],
    html_out: str,
    max_cpus: int | None = None,
) -> None:
    """Creates an html output displaying groups of motifs.

    Creates a two panel HTML file displaying motifs grouped into sections. Motifs will
      be displayed in the order that they are listed. The HTML file is produced using
      jinja2 and motif_collection_template.html.

    Args:
        motif_groups: A dictionary from group name to groups, where each group is
          itself a list of motif dictionaries. Each motif dictionary maps strings to
          attributes about the motif. Motif dictionaries must contain a key 'motif'
          which is a pd.DataFrame that is plotted using logomaker and a key 'name' which
          is the name of the motif.
        html_out: The path to save he html file.
        max_cpus: The maximum number of CPUs to use for parallelizing plotting.
    """
    # Use Agg backend
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Use Agg backend
    # Create motif plots
    if max_cpus is None:
        for group in motif_groups.values():
            for motif_dict in group:
                _utf8_motif_plot(motif_dict)
    else:
        # Keep track of which plots come from which groups to reorder them later
        all_motif_dicts = []
        group_to_motif_dict_idx = dict()
        start = 0
        for group_name, group in motif_groups.items():
            end = start + len(group)
            group_to_motif_dict_idx[group_name] = (start, end)
            start = end
            all_motif_dicts += group
        # Plot in parallel
        num_processes = min(
            max_cpus, multiprocessing.cpu_count()
        )  # don't use more CPUs than available
        with multiprocessing.Pool(processes=num_processes) as p:
            all_motif_dicts = p.map(
                _utf8_motif_plot, all_motif_dicts
            )  # Not pass by ref
        # Redefine motif_groups using updated motif dicts
        for group_name in group_to_motif_dict_idx:
            start, end = group_to_motif_dict_idx[group_name]
            motif_groups[group_name] = all_motif_dicts[start:end]
    # Create Jinja2 environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    # Load HTML template
    template = env.get_template("motif_collection_template.html")
    # Render HTML with data
    rendered_html = template.render(data=motif_groups, sorted=sorted)
    # Write HTML to file
    with open(html_out, "w") as f:
        f.write(rendered_html)
    # Switch to original backend
    matplotlib.use(current_backend)


def summary_table_html():
    print("not yet implemented")
    assert False


def plot_heatmap(
    data: np.ndarray,
    annot: bool = False,
    labels: list | None = None,
    show: bool = False,
    save_loc: str | None = None,
):
    """Plot a heatmap.

    Creates a heatmap. Includes additional options to annotate/label the heatmap and
      optionally show and/or save the resulting plot.

    Args:
        data: A np.ndarray to plot a heatmap of.
        annot: Whether or not to annotate each cell of the heatmap with its value.
        labels: Labels for the rows/columns of the heatmap. If None, rows and columns
          are not annotated. If labels are provided, data is assumed to be square.
        show: Whether or not to show the heatmap with plt.show().
        save_loc: Where to save the heatmap to. If None, the heatmap is not saved.
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


#####################
# PRIVATE FUNCTIONS #
#####################
def _utf8_motif_plot(motif_dict: dict[str, Any]) -> None:
    """Generates UTF-8 encoded plot of a motif and saves it into the input dictionary.

    Generates UTF-8 encoded plots using logomaker. Can make use of additional formatting
      arguments in motif_dict if they are present. Saves the encoded plot into the input
      dictionary under key 'utf8_plot'.

    Args:
        motif_dict: A dictionary containing key 'motif' which can be passed to
          logomaker.
    """
    # Parse through arguments in motif_dict
    motif = motif_dict["motif"]
    face_color = motif_dict["bgcolor"] if "bgcolor" in motif_dict else "white"
    # Plot
    fig, ax = plt.subplots(figsize=(6, 2), facecolor=face_color)
    logomaker.Logo(motif, ax=ax)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_axis_off()
    # Encode image in UTF-8
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    utf8_plot = base64.b64encode(buf.read()).decode("utf-8")
    # Update dictionary
    motif_dict["utf8_plot"] = utf8_plot
    return motif_dict
