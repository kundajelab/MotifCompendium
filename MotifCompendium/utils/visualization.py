import os

from jinja2 import Environment, FileSystemLoader
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
    for (start, end), group_name in group_to_motif_idx.items():
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


def table_html(table: pd.DataFrame, image_column: list[bool], html_out: str) -> None:
    """Creates an html file displaying the values in a pd.DataFrame.

    Creates a table HTML file of the values in a pd.DataFrame. Some of the columns can
      be UTF-8 encoded images. The HTML file is produced using jinja2 and
      table_template.html.

    Args:
        table: A table to display table.
        image_column: A list of booleans indicating which columns contain UTF-8 encoded
          images.
        html_out: The path to save he html file.
    """
    columns = table.columns.tolist()
    rows = table.to_dict(orient="records")
    # Create Jinja2 environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    # Load HTML template
    template = env.get_template("table_template.html")
    # Render HTML with data
    rendered_html = template.render(
        columns=columns, rows=rows, image_column=image_column
    )
    # Write HTML to file
    with open(html_out, "w") as f:
        f.write(rendered_html)
