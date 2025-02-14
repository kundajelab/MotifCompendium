import base64
import io
import multiprocessing

import logomaker
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use("fast")
import numpy as np
import pandas as pd
import seaborn as sns

from MotifCompendium.utils.config import get_max_cpus
from MotifCompendium.utils.motif import motif_to_df


#######################
# LOGO PLOTTING INPUT #
#######################
class LogoPlottingInput:
    """An object for storing inputs and outputs for plotting the logo of a motif.

    The LogoPlottingInput object is intended to the input to logo plotting functions.
      The get_motif_df() function produces a pd.DataFrame that can be passed to
      logomaker.

    Attributes:
        motif: A np.ndarray of shape (30, 4) representing the motif that needs to be
          plotted.
        revcomp: A boolean indicating whether the motif needs to be reverse
          complemented.
        pos: An int representing the position of the motif (shifted post-reverse
          complement).
        xmin: An int representing the minimum value of the x-axis in the logo plot.
        xmax: An int representing the maximum value of the x-axis in the logo plot.
        name: A str representing the name of the motif. Not plotted.
        encode: A bool representing whether the plot should be encoded as a UTF-8
          string.
        bgcolor: A str representing the background color of the logo.
        ax: A matplotlib.axes.Axes object representing the axes of the plot.
        save_loc: A str representing the location where the plot should be saved.
        utf8_plot: A str representing the UTF-8 encoded plot of the motif.
    """

    def __init__(
        self,
        motif: np.ndarray,
        revcomp: bool = False,
        pos: int = 0,
        name: str = "motif",
        bgcolor: str = "white",
        encode: bool = True,
        ax: matplotlib.axes.Axes | None = None,
        save_loc: str | None = None,
    ) -> None:
        """LogoPlottingInput constructor.

        This constructor takes in a motif and additional formatting arguments.

        Args:
            motif: A np.ndarray that is assigned to self.motifs.
            revcomp: A bool that is assigned to self.revcomp.
            pos: An int that is assigned to self.pos.
            name: A str that is assigned to self.name.
            bgcolor: A str that is assigned to self.bgcolor.
            encode: A bool that is assigned to self.encode.
            ax: A matplotlib.axes.Axes object that is assigned to self.ax.
            save_loc: A str that is assigned to self.save_loc.
        """
        # Motif
        self.motif = motif
        self.revcomp = revcomp
        self.pos = pos
        self.xmin = 0
        self.xmax = motif.shape[0]
        self.name = name
        # Plot options
        self.bgcolor = bgcolor
        self.encode = encode
        # Outputs
        self.ax = ax
        self.save_loc = save_loc
        self.utf8_plot = ""

    def set_bounds(self, xmin: int, xmax: int) -> None:
        """Sets the bounds of the x-axis for the motif."""
        self.xmin = xmin
        self.xmax = xmax

    def get_motif_df(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of the motif that can be passed to logomaker."""
        # Reverse complement
        if self.revcomp:
            motif_df = motif_to_df(self.motif[::-1, ::-1])
        else:
            motif_df = motif_to_df(self.motif)
        # Then shift
        motif_df.index += self.pos
        # Then set bounds
        motif_df.reindex(range(self.xmin, self.xmax + 1), fill_value=0)
        return motif_df


###########################
# LOGO PLOTTING FUNCTIONS #
###########################
def plot_motif_logo(motif_info: LogoPlottingInput) -> LogoPlottingInput:
    """Plots the logo of a motif.

    Plots the logo of a motif as specified by a LogoPlottingInput object. If
      motif_info.ax is not None, the logo is plotted to that Axes object. If
      motif_info.save_loc, the plot figure is saved to that location. If
      motif_info.encode is True, the plot is encoded as a UTF-8 string and stored in
      motif_info.utf8_plot. Returns the updated LogoPlottingInput object.

    Args:
        motif_info: A LogoPlottingInput object specifying how the motif should be plotted.

    Returns:
        A LogoPlottingInput object containing the plot of the motif.
    """
    # Get Axes
    if motif_info.ax is None:
        fig, ax = plt.subplots(figsize=(6, 2), facecolor=motif_info.bgcolor)
        plot_ax = ax
    else:
        plot_ax = motif_info.ax
    # Plot
    logomaker.Logo(motif_info.get_motif_df(), ax=plot_ax)
    plot_ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    plot_ax.set_axis_off()
    # Save
    if motif_info.save_loc is not None:
        plot_ax.figure.savefig(motif_info.save_loc, bbox_inches="tight")
    # Encode image in UTF-8
    if motif_info.encode:
        motif_info.utf8_plot = encode_figure_as_utf8(fig)
    # Return updated LogoPlottingInput
    return motif_info


def plot_many_motif_logos(
    motif_info_list: list[LogoPlottingInput],
) -> list[LogoPlottingInput]:
    """Plot the logos of multiple motifs.

    Calls plot_motif_logo() on each motif in motif_info_list and returns the updated
      list. Parallelizes the plotting if allowed by config.get_max_cpus() > 1.

    Args:
        motif_info_list: A list of LogoPlottingInput object specifying how each motif
          should be plotted.

    Returns:
        A LogoPlottingInput object containing the plot of the motif.
    """
    # Use Agg backend
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Use Agg backend
    # Determine the number of processes to use
    num_processes = min(
        get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    # Plot
    if num_processes == 1:
        for motif_info in motif_info_list:
            plot_motif_logo(motif_info)
    else:
        with multiprocessing.Pool(processes=num_processes) as p:
            motif_info_list = p.map(
                plot_motif_logo, motif_info_list
            )  # Overwrite because map not pass by ref
    # Reset backend
    matplotlib.use(current_backend)
    # Return updated LogoPlottingInputs
    return motif_info_list


def encode_figure_as_utf8(fig: plt.figure.Figure) -> str:
    """Encodes a figure as a UTF-8 string.

    Encodes a figure as a UTF-8 string and returns the string.

    Args:
        fig: A plt.figure.Figure object.

    Returns:
        A UTF-8 encoded string of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


############################
# OTHER PLOTTING FUNCTIONS #
############################
def plot_heatmap(
    data: np.ndarray,
    annot: bool = False,
    labels: list[str] | None = None,
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
