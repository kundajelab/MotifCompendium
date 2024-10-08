from jinja2 import Environment, FileSystemLoader
import logomaker
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use("fast")

import pandas as pd

import base64
import io
import os
import multiprocessing
from typing import List, Tuple

import MotifCompendium


def motif_to_df(motif: np.ndarray) -> pd.DataFrame:
    """Transforms a motif into a pd.DataFrame ready for plotting with logomaker."""
    return pd.DataFrame(motif, columns=["A", "C", "G", "T"])


def plot64_from_motif(plotter_input: Tuple[pd.DataFrame, str]) -> str:
    """Generates a UTF-8 encoded plot of a motif.

    Plots a motif logo, saves it as bytes, and decodes that as a UTF-8 string.

    Args:
        plotter_input: A tuple of a DataFrame representing a motif and a string of the
          background color of the motif plot.

    Returns:
        A string representing a UTF-8 encoded image of a motif logo.
    """
    motif_df, face_color = plotter_input
    fig, ax = plt.subplots(figsize=(6, 2), facecolor=face_color)
    logomaker.Logo(motif_df, ax=ax)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot64_from_motifs_parallel(motifs: List, max_parallel: int) -> List[str]:
    """Generates UTF-8 encoded plots of many motifs.

    Generates UTF-8 encoded plots by invoking plot64_from_motif() in parallel.

    Args:
        motifs: A list of motif inputs to be passed to plot64_from_motif().
        max_parallel: The

    Returns:
        A list of strings representing UTF-8 encoded images of motif logos.

    Notes:
        hello.
    """
    max_cpus = min(max_parallel, multiprocessing.cpu_count())
    if max_cpus > 1:
        with multiprocessing.Pool(processes=max_cpus) as p:
            return p.map(plot64_from_motif, motifs)
    else:
        return [plot64_from_motif(x) for x in motifs]


def generate_table(motifs, metadata, output_file, max_parallel):
    # Use Agg backend
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    # Make plots
    fwd_motif_payloads, rev_motif_payloads = [], []
    for i in range(motifs.shape[0]):
        fwd_motif_payloads.append((motif_to_df(motifs[i, :, :]), "white"))
        rev_motif_payloads.append((motif_to_df(motifs[i, ::-1, ::-1]), "white"))
    fwd_motif_strings = plot64_from_motifs_parallel(fwd_motif_payloads, max_parallel)
    rev_motif_strings = plot64_from_motifs_parallel(rev_motif_payloads, max_parallel)

    # Rearrange data
    metadata.insert(0, "logo (fwd)", fwd_motif_strings)
    metadata.insert(1, "logo (rev)", rev_motif_strings)

    # Create Jinja2 environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))

    # Load HTML template
    template = env.get_template("table_template.html")

    # Prepare data
    columns = metadata.columns.tolist()
    rows = metadata.to_dict(orient="records")

    # Render HTML with data
    rendered_html = template.render(columns=columns, rows=rows)

    # Write HTML to file
    with open(output_file, "w") as f:
        f.write(rendered_html)

    # Switch back to original backend
    matplotlib.use(current_backend)


###
# Usage
###
"""
data = {
    'Column1': ['Value1', 'Value2', 'Value3', 'Value1', 'Value1'],
    'Column2': ['Value4', 'Value5', 'Value6', 'Value5', 'Value5'],
    'Column3': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],

    'Column4': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],
    'Column5': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],
    'Column6': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],
    'Column7': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],
    'Column8': ['Value7', 'Value8', 'Value9', 'Value3', 'Value9'],
}
df = pd.DataFrame(data)

motifs = np.random.rand(5, 30, 4)
motifs /= np.sum(motifs, axis=(1, 2), keepdims=True)
generate_table(motifs, df, "/oak/stanford/groups/akundaje/salil512/web/temp/table_test.html", 4)

assert(False)
"""
mc = MotifCompendium.load(
    "/users/salil512/ENCODE_atlas_motif_clustering/chromatin/chromatin_atlas_counts_95_annotated.mc"
)
print(mc)
generate_table(
    mc.logos,
    mc.metadata,
    "/oak/stanford/groups/akundaje/salil512/web/temp/table_test.html",
    32,
)
