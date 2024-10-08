from collections import defaultdict
from typing import Any

from jinja2 import Environment, FileSystemLoader
import logomaker
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
    max_cpus: int | None = None
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
    # Create motif plots
    if max_cpus is None:
        for group in motif_groups.values():
            _utf8_motif_plot_many(group)
    else:
        num_processes = min(max_cpus, multiprocessing.cpu_count())  # don't use more CPUs than available
        with multiprocessing.Pool(processes=num_processes) as p:
            p.map(_utf8_motif_plot_many, motif_groups.values())
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


def summary_table_html():
    print("not yet implemented")
    assert False


def plot_heatmap(
    data: np.ndarray,
    annot: bool = False,
    labels: list | None = None,
    show: bool = False,
    save_loc: str | None = None
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
def _utf8_motif_plot_many(motif_dicts: list[dict[str, Any]]) -> None:
    """Generates UTF-8 encoded plots of many motifs.

    Generates UTF-8 encoded plots by invoking _utf8_motif_plot(). Enters them into
      each motif dict in the input under the key 'utf8_plot'.

    Args:
        motif_dicts: A list of motif inputs to be passed to _utf8_motif_plot().

    Notes:
        Each motif_dict must have key 'motif' which is a pd.DataFrame that can be read
          by logomaker. It may additionally have arguments that change plot formatting.
          Please refer to _utf8_motif_plot() implementation for further details.
    """
    for motif_dict in motif_dicts:
        _utf8_motif_plot(motif_dict)

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