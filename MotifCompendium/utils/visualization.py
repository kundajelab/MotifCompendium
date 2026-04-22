import os
import json

from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd

import MotifCompendium.utils.plotting as utils_plotting


####################
# PUBLIC FUNCTIONS #
####################
def motif_collection_html(
    motif_groups: dict[str, list[utils_plotting.LogoPlottingInput]], html_out: str
) -> None:
    """Creates an html file displaying groups of motifs.

    Creates a two panel HTML file displaying motifs grouped into sections. Motifs will
      be displayed in the order that they are listed. The HTML file is produced using
      jinja2 and motif_collection_template.html.

    Args:
        motif_groups: A dictionary from group names to groups, where each group is
          specified as a list of LogoPlottingInput objects.
        html_out: The path to save he html file.
    """
    if not html_out.endswith(".html"):
        html_out += ".html"
    # Keep track of which plots come from which groups to reorder them later
    all_motifs = []
    group_to_motif_idx = dict()
    start = 0
    for group_name, group in motif_groups.items():
        end = start + len(group)
        group_to_motif_idx[group_name] = (start, end)
        start = end
        all_motifs += group
    # Plot
    all_motifs = utils_plotting.plot_many_motif_logos(all_motifs)
    # Redefine motif_groups using updated LogoPlottingInput objects
    for group_name, (start, end) in group_to_motif_idx.items():
        motif_groups[group_name] = all_motifs[start:end]
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


def table_html(
    table: pd.DataFrame, image_column: list[bool], html_out: str, editable: bool
) -> None:
    CHUNK_SIZE = 1_000   # tune to taste; ~1 k rows ≈ smooth 30-50 ms steps

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pd.DataFrame")
    if not html_out.endswith(".html"):
        html_out += ".html"

    image_column = list(image_column)
    table = table.copy()
    table.insert(0, "index", table.index)
    image_column.insert(0, False)
    table = table.astype(object).where(pd.notna(table), None)

    columns  = table.columns.tolist()
    table = table.applymap(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    rows_all = table.to_dict(orient="records")

    def _json(obj) -> str:
        return json.dumps(obj, ensure_ascii=True).replace("</", "<\\/")

    meta_payload = _json({
        "columns":      columns,
        "image_column": image_column,
        "editable":     editable,
    })

    row_chunks = [
        _json(rows_all[i : i + CHUNK_SIZE])
        for i in range(0, len(rows_all), CHUNK_SIZE)
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env      = Environment(loader=FileSystemLoader(current_dir))
    template = env.get_template("table_template.html")
    rendered = template.render(meta_payload=meta_payload, row_chunks=row_chunks)

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(rendered)