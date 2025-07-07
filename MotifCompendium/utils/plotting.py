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

import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.motif as utils_motif


#######################
# LOGO PLOTTING INPUT #
#######################
class LogoPlottingInput:
    """An object for storing inputs and outputs for plotting the logo of a motif.

    The LogoPlottingInput object is intended to the input to logo plotting functions.
      The get_motif_df() function produces a pd.DataFrame that can be passed to
      logomaker.

    Attributes:
        motif: A np.ndarray of shape (L, 4) representing the motif that needs to be
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
        fast_plot: A bool representing whether or not the plot should be generated
          quickly using custom plotting code. If True, use fast plotting code. If False,
          plot using logomaker.
        bgcolor: A str representing the background color of the logo.
        ax: A matplotlib.axes.Axes object representing the axes of the plot.
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
        self.utf8_plot = ""

    def set_bounds(self, xmin: int, xmax: int) -> None:
        """Sets the bounds of the x-axis for the motif."""
        self.xmin = xmin
        self.xmax = xmax

    def get_motif_df(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of the motif that can be passed to logomaker."""
        # Reverse complement
        if self.revcomp:
            motif_df = utils_motif.motif_to_df(
                utils_motif.reverse_complement(self.motif)
            )
        else:
            motif_df = utils_motif.motif_to_df(self.motif)
        # Then shift
        motif_df.index += self.pos
        # Then set bounds
        return motif_df.reindex(range(self.xmin, self.xmax + 1), fill_value=0)


###########################
# LOGO PLOTTING FUNCTIONS #
###########################
def plot_motif(
    motif: np.ndarray,
    show: bool = True,
    save_loc: str | None = None,
) -> None:
    """Plots a single motif.

    Plots a single motif. If show is True, the figure is shown. If save_loc is not None,
      the motif plot is saved to that location.

    Args:
        motif: A np.ndarray of shape (L, 4) representing the motif to be plotted.
        show: Whether or not to show the heatmap with plt.show().
        save_loc: Where to save the motif plot to. If None, the motif plot is not saved.
    """
    # Check inputs
    utils_motif.validate_motif_basic(motif)
    if not len(motif.shape) == 2 and motif.shape[1] == 4:
        raise TypeError("motif must be a 2D array with shape (L, 4).")
    # Plot
    fig, ax = plt.subplots(figsize=(6, 2))
    motif_logo = LogoPlottingInput(motif, ax=ax, encode=False)
    _plot_motif_logo(motif_logo)
    # Output
    if save_loc is not None:
        fig.savefig(save_loc, bbox_inches="tight")
    if show:
        plt.show()


def plot_motif_stack(
    motif_stack: np.ndarray,
    alignment_rc: np.ndarray | None = None,
    alignment_h: np.ndarray | None = None,
    show: bool = False,
    save_loc: str | None = None,
) -> None:
    """Plots a stack of motifs.

    Plots a stack of motifs by calling plot_many_motif_logos(). If alignment information
      is provided, the motifs are aligned accordingly. All motifs are plotted on the same
      figure. If show is True, the figure is shown.

    Args:
        motif_stack: A stack of motifs to be plotted.
        alignment_rc: A (N, ) forward/reverse complement alignment vector. If None, no
          alignment is performed.
        alignment_h: A (N, ) horizontal alignment vector. If None, no alignment is
          performed.
        show: Whether or not to show the heatmap with plt.show().
        save_loc: Where to save the motif plot to. If None, the motif plot is not saved.
    """
    # Check inputs
    utils_motif.validate_motif_stack_standard(motif_stack)
    N = motif_stack.shape[0]
    if alignment_rc is not None and not (
        isinstance(alignment_rc, np.ndarray) and alignment_rc.shape == (N,)
    ):
        raise TypeError(
            "alignment_rc must be a vector whose length matches that of the motif stack."
        )
    if alignment_h is not None and not (
        isinstance(alignment_h, np.ndarray) and alignment_h.shape == (N,)
    ):
        raise TypeError(
            "alignment_h must be a vector whose length matches that of the motif stack."
        )
    # Defaults for alignment_rc and alignment_h
    if alignment_rc is None:
        alignment_rc = np.zeros(N).astype(bool)
    if alignment_h is None:
        alignment_h = np.zeros(N)
    # Create figure
    fig, axs = plt.subplots(N, figsize=(6, 2 * N))
    # Prepare inputs
    motif_info_list = []
    for i, ax in enumerate(axs):
        motif_info_list.append(
            LogoPlottingInput(
                motif_stack[i],
                revcomp=alignment_rc[i],
                pos=alignment_h[i],
                encode=False,
                ax=ax,
            )
        )
    x_min = int(np.min(alignment_h))
    x_max = int(np.max(alignment_h)) + motif_stack.shape[1]
    for motif_info in motif_info_list:
        motif_info.set_bounds(x_min, x_max)
    # Plot
    motif_info_list = plot_many_motif_logos(motif_info_list)
    # Determine if plots need to be transferred (ocurrs when parallel plotting)
    if motif_info_list[0].ax.figure != fig:
        # Create new figure
        new_fig = plt.figure(figsize=(6, 2 * N))
        for motif_info in motif_info_list:
            # Get variables
            motif_info_ax = motif_info.ax
            motif_info_figure = motif_info_ax.figure
            # Assign axis to new figure
            motif_info_ax.remove()
            motif_info_ax.figure = new_fig
            new_fig.add_axes(motif_info_ax)
            # Remove axis's figure
            plt.close(motif_info_figure)
        # Remove old figure
        plt.close(fig)
        fig = new_fig
        fig.tight_layout()
    # Output
    if save_loc is not None:
        fig.savefig(save_loc, bbox_inches="tight")
    if show:
        plt.show()


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
        utils_config.get_max_cpus(), multiprocessing.cpu_count()
    )  # don't use more CPUs than available
    # Plot
    if num_processes == 1 or len(motif_info_list) == 1:
        for motif_info in motif_info_list:
            _plot_motif_logo(motif_info)
    else:
        with multiprocessing.Pool(processes=num_processes) as p:
            motif_info_list = p.map(
                _plot_motif_logo, motif_info_list
            )  # Overwrite because map not pass by ref
    # Reset backend
    matplotlib.use(current_backend)
    # Return updated LogoPlottingInputs
    return motif_info_list


def encode_figure_as_utf8(fig: matplotlib.figure.Figure) -> str:
    """Encodes a figure as a UTF-8 string.

    Encodes a figure as a UTF-8 string and returns the string.

    Args:
        fig: A matplotlib.figure.Figure object.

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
    # Save/show/close
    if save_loc is not None:
        heatmap.get_figure().savefig(save_loc)
    if show:
        plt.show()
    plt.close()


###################################
# PRIVATE LOGO PLOTTING FUNCTIONS #
###################################
def _plot_motif_logo(motif_info: LogoPlottingInput) -> LogoPlottingInput:
    """Plots the logo of a motif.

    Plots the logo of a motif as specified by a LogoPlottingInput object. If
      motif_info.ax is not None, the logo is plotted to that Axes object. If
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
        fig = plot_ax.figure
    # Plot
    if not (motif_info.motif == 0).all():  # Only plot if motif is not all zeros
        if utils_config.get_fast_plotting():
            _plot_logo_on_axis_fast(motif_info.get_motif_df(), plot_ax)
        else:
            logo = logomaker.Logo(motif_info.get_motif_df(), ax=plot_ax)
            # logo.style_spines(visible=False)
    plot_ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    plot_ax.set_axis_off()
    # Encode image in UTF-8
    if motif_info.encode:
        motif_info.utf8_plot = encode_figure_as_utf8(fig)
    # Return updated LogoPlottingInput
    return motif_info


def _plot_logo_on_axis_fast(motif_df: pd.DataFrame, ax: matplotlib.axes.Axes) -> None:
    """Plots a motif with rectangles.

    Fast custom code for plotting a motif with rectangles. This is faster than using
        logomaker.Logo().

    Args:
        motif_df: A pd.DataFrame of the motif.
        ax: A matplotlib.axes.Axes object to plot the motif on.
    """
    cols = motif_df.columns
    max_y = 0
    min_y = 0
    for index, row in motif_df.iterrows():
        # Get order of nucleotides
        pos_nucleotides = []
        neg_nucleotides = []
        for c in cols:
            if row[c] > 0:
                pos_nucleotides.append((row[c], c))
            if row[c] < 0:
                neg_nucleotides.append((row[c], c))
        pos_nucleotides = sorted(pos_nucleotides)
        neg_nucleotides = sorted(neg_nucleotides, reverse=True)
        # Plot
        x = index - 0.5  # Between (index - 0.5, index + 0.5)
        # Plot pos nucleotides
        y = 0
        for n_height, n in pos_nucleotides:
            match (n):
                case "A":
                    _plot_a(x + 0.05, y, 0.9, n_height, ax)
                case "T":
                    _plot_t(x + 0.05, y, 0.9, n_height, ax)
                case "C":
                    _plot_c(x + 0.05, y, 0.9, n_height, ax)
                case "G":
                    _plot_g(x + 0.05, y, 0.9, n_height, ax)
                case _:
                    raise KeyError("Illegal nucleotide")
            y += n_height
        max_y = max(max_y, y)
        # Plot neg nucleotides
        y = 0
        for n_height, n in neg_nucleotides:
            match (n):
                case "A":
                    _plot_a(x + 0.05, y, 0.9, n_height, ax)
                case "T":
                    _plot_t(x + 0.05, y, 0.9, n_height, ax)
                case "C":
                    _plot_c(x + 0.05, y, 0.9, n_height, ax)
                case "G":
                    _plot_g(x + 0.05, y, 0.9, n_height, ax)
                case _:
                    raise KeyError("Illegal nucleotide")
            y += n_height
        min_y = min(min_y, y)
    # Set xlim, ylim
    ax.set_xlim([min(motif_df.index) - 0.5, max(motif_df.index) + 0.5])
    ax.set_ylim([1.1 * min_y, 1.1 * max_y])


def _plot_a(x, y, width, height, ax) -> None:
    """Plots adenine."""
    dx = width / 4
    dy = height / 6
    # left
    ax.add_patch(
        matplotlib.patches.Rectangle((x, y), dx, 5 * dy, facecolor="green", aa=False)
    )
    # right
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y), dx, 5 * dy, facecolor="green", aa=False
        )
    )
    # top
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + dx, y + 5 * dy), 2 * dx, dy, facecolor="green", aa=False
        )
    )
    ax.add_patch(
        matplotlib.patches.Polygon(
            [[x, y + 5 * dy], [x + dx, y + 5 * dy], [x + dx, y + 6 * dy]],
            closed=False,
            facecolor="green",
            aa=False,
        )
    )
    ax.add_patch(
        matplotlib.patches.Polygon(
            [
                [x + 4 * dx, y + 5 * dy],
                [x + 3 * dx, y + 5 * dy],
                [x + 3 * dx, y + 6 * dy],
            ],
            closed=False,
            facecolor="green",
            aa=False,
        )
    )
    # middle
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + dx, y + 3 * dy), 2 * dx, dy, facecolor="green", aa=False
        )
    )


def _plot_t(x, y, width, height, ax) -> None:
    """Plots thyamine."""
    dx = width / 8
    dy = height / 6
    # top
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x, y + 5 * dy), 8 * dx, dy, facecolor="red", aa=False
        )
    )
    # middle
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y), 2 * dx, 5 * dy, facecolor="red", aa=False
        )
    )


def _plot_c(x, y, width, height, ax) -> None:
    """Plots cytosine."""
    dx = width / 4
    dy = height / 6
    # right bottom lip
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y + dy), dx, dy, facecolor="mediumblue", aa=False
        )
    )
    # right top lip
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y + 4 * dy), dx, dy, facecolor="mediumblue", aa=False
        )
    )
    # left
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x, y + dy), dx, 4 * dy, facecolor="mediumblue", aa=False
        )
    )
    # bottom
    ax.add_patch(
        matplotlib.patches.Polygon(
            [[x, y + dy], [x + 4 * dx, y + dy], [x + 3 * dx, y], [x + dx, y]],
            closed=False,
            facecolor="mediumblue",
            aa=False,
        )
    )
    # ax.add_patch(matplotlib.patches.Rectangle((x+dx, y), 2*dx, dy, facecolor="mediumblue", aa=False)) # bottom
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x, y + dy], [x+dx, y + dy], [x+dx, y]],
    #     closed=False,
    #     facecolor="mediumblue",
    #     aa=False
    # ))
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x+4*dx, y+dy], [x+3*dx, y+dy], [x+3*dx, y]],
    #     closed=False,
    #     facecolor="mediumblue",
    #     aa=False
    # ))
    # top
    ax.add_patch(
        matplotlib.patches.Polygon(
            [
                [x, y + 5 * dy],
                [x + 4 * dx, y + 5 * dy],
                [x + 3 * dx, y + 6 * dy],
                [x + dx, y + 6 * dy],
            ],
            closed=False,
            facecolor="mediumblue",
            aa=False,
        )
    )
    # ax.add_patch(matplotlib.patches.Rectangle((x+dx, y+5*dy), 2*dx, dy, facecolor="mediumblue", aa=False)) # top
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x, y + 5*dy], [x + dx, y + 5*dy], [x + dx, y + 6*dy]],
    #     closed=False,
    #     facecolor="mediumblue",
    #     aa=False
    # ))
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x+4*dx, y + 5*dy], [x + 3*dx, y + 5*dy], [x + 3*dx, y + 6*dy]],
    #     closed=False,
    #     facecolor="mediumblue",
    #     aa=False
    # ))
    return


def _plot_g(x, y, width, height, ax) -> None:
    """Plots guanine."""
    dx = width / 4
    dy = height / 6
    # right bottom lip
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y + dy), dx, dy, facecolor="orange", aa=False
        )
    )
    # right top lip
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 3 * dx, y + 4 * dy), dx, dy, facecolor="orange", aa=False
        )
    )
    # left
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x, y + dy), dx, 4 * dy, facecolor="orange", aa=False
        )
    )
    # G line
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x + 2 * dx, y + 2 * dy), 2 * dx, dy, facecolor="orange", aa=False
        )
    )
    # bottom
    ax.add_patch(
        matplotlib.patches.Polygon(
            [[x, y + dy], [x + 4 * dx, y + dy], [x + 3 * dx, y], [x + dx, y]],
            closed=False,
            facecolor="orange",
            aa=False,
        )
    )
    # ax.add_patch(matplotlib.patches.Rectangle((x+dx, y), 2*dx, dy, facecolor="orange", aa=False)) # bottom
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x, y + dy], [x+dx, y + dy], [x+dx, y]],
    #     closed=False,
    #     facecolor="orange",
    #     aa=False
    # ))
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x+4*dx, y+dy], [x+3*dx, y+dy], [x+3*dx, y]],
    #     closed=False,
    #     facecolor="orange",
    #     aa=False
    # ))
    # top
    ax.add_patch(
        matplotlib.patches.Polygon(
            [
                [x, y + 5 * dy],
                [x + 4 * dx, y + 5 * dy],
                [x + 3 * dx, y + 6 * dy],
                [x + dx, y + 6 * dy],
            ],
            closed=False,
            facecolor="orange",
            aa=False,
        )
    )
    # ax.add_patch(matplotlib.patches.Rectangle((x+dx, y+5*dy), 2*dx, dy, facecolor="orange", aa=False)) # top
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x, y + 5*dy], [x + dx, y + 5*dy], [x + dx, y + 6*dy]],
    #     closed=False,
    #     facecolor="orange",
    #     aa=False
    # ))
    # ax.add_patch(matplotlib.patches.Polygon(
    #     [[x+4*dx, y + 5*dy], [x + 3*dx, y + 5*dy], [x + 3*dx, y + 6*dy]],
    #     closed=False,
    #     facecolor="orange",
    #     aa=False
    # ))
    return


def _transfer_axis_content(source_ax, target_ax):
    """
    Transfer all content from source_ax to target_ax
    """
    # Transfer patches (rectangles, polygons, etc)
    for patch in source_ax.patches[
        :
    ]:  # Use [:] to make a copy since we'll modify the list
        patch_type = type(patch)

        # Get general properties that apply to most patches
        props = {
            "alpha": patch.get_alpha(),
            "facecolor": patch.get_facecolor(),
            "edgecolor": patch.get_edgecolor(),
            "linewidth": patch.get_linewidth(),
            "linestyle": patch.get_linestyle(),
            "zorder": patch.get_zorder(),
            "aa": getattr(patch, "_antialiased", False),  # Your patches use aa=False
        }

        # Filter out None properties
        props = {k: v for k, v in props.items() if v is not None}

        # Handle specific patch types
        if isinstance(patch, matplotlib.patches.Rectangle):
            xy = patch.get_xy()
            width = patch.get_width()
            height = patch.get_height()
            new_patch = matplotlib.patches.Rectangle(xy, width, height, **props)

        elif isinstance(patch, matplotlib.patches.Polygon):
            vertices = patch.get_xy()
            closed = patch.get_closed()
            new_patch = matplotlib.patches.Polygon(vertices, closed=closed, **props)

        # Add other patch types as needed
        else:
            print(f"Warning: Patch type {patch_type} not handled")
            continue

        target_ax.add_patch(new_patch)

    # Transfer lines
    for line in source_ax.lines[:]:
        target_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            alpha=line.get_alpha(),
            zorder=line.get_zorder(),
        )

    # Transfer text elements
    for text in source_ax.texts[:]:
        target_ax.text(
            text.get_position()[0],
            text.get_position()[1],
            text.get_text(),
            fontsize=text.get_fontsize(),
            color=text.get_color(),
            ha=text.get_ha(),
            va=text.get_va(),
            alpha=text.get_alpha(),
            rotation=text.get_rotation(),
            zorder=text.get_zorder(),
        )

    # Transfer collections (like scatter plots)
    for collection in source_ax.collections[:]:
        # This is a bit complex as collections can vary
        # Possible to add more specific handling based on your needs
        target_ax.add_collection(collection.copy())

    # Transfer images
    for image in source_ax.images[:]:
        target_ax.imshow(
            image.get_array(),
            extent=image.get_extent(),
            alpha=image.get_alpha(),
            zorder=image.get_zorder(),
        )

    # Copy axis limits and settings
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())
    target_ax.set_xscale(source_ax.get_xscale())
    target_ax.set_yscale(source_ax.get_yscale())

    # Copy visibility settings for spines
    for spine in source_ax.spines:
        target_ax.spines[spine].set_visible(source_ax.spines[spine].get_visible())

    # Copy axis visibility
    if not source_ax.axison:
        target_ax.set_axis_off()

    # Make sure changes are reflected
    target_ax.figure.canvas.draw_idle()

    return target_ax
