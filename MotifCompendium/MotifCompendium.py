from __future__ import annotations
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

import MotifCompendium.utils.clustering as utils_clustering
import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.plotting as utils_plotting
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.visualization as utils_visualization


############
# SETTINGS #
############
def set_compute_options(
    max_chunk: int | None = None,
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
):
    """Set default values for max_chunk, max_cpus, and use_gpu.

    Args:
        max_chunk: The maximum number of motifs to compute similarity on at a time. SET
          TO -1 TO USE NO CHUNKING.
        max_cpus: The maximum number of CPUs to use. Used while loading multiple Modisco
          files, generating plots, and, when use_gpu is False, computing similarity.
        use_gpu: Whether or not to GPU accelerate similarity computations.

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_cpus to the number of available
          CPUs.
        Currently, multi-GPU calculation is not supported.
        If memory constraints are not an issue, set as None for faster
          performance. Otherwise, decrease max_chunk until calculations fit in memory.
          For a GPU with ~12GB of memory, use max_chunk=1000.
    """
    if max_chunk is not None:
        utils_config.set_max_chunk(max_chunk)
    if max_cpus is not None:
        utils_config.set_max_cpus(max_cpus)
    if use_gpu is not None:
        utils_config.set_use_gpu(use_gpu)


######################################
# MOTIF COMPENDIUM FACTORY FUNCTIONS #
######################################
def load(file_loc: str, safe: bool = True) -> MotifCompendium:
    """Loads a MotifCompendium object from file.

    Args:
        file_loc: The MotifCompendium file path.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        The corresponding MotifCompendium object.

    Notes:
        Assumes the file is an h5py file with datasets 'motifs', 'similarity',
          'alignment_fr', and 'alignment_h', as well as a DataFrame called 'metadata'.
        Old objects may be incompatable with the newest version of the load function.
        Safe loading validates object integrity but may take significantly longer for
          large objects.
    """
    try:
        with h5py.File(file_loc, "r") as f:
            motifs = f["motifs"][:]
            similarity = f["similarity"][:]
            alignment_fr = f["alignment_fr"][:]
            alignment_h = f["alignment_h"][:]
        metadata = pd.read_hdf(file_loc, key="metadata")
        __images = pd.read_hdf(file_loc, key="__images")
    except:
        raise ValueError(
            "File does not contain the necessary datasets to load a MotifCompendium."
        )
    return MotifCompendium(
        motifs, similarity, alignment_fr, alignment_h, metadata, __images, safe
    )


def inspect(file_loc: str) -> pd.DataFrame:
    """Inspects and returns the metadata of a MotifCompendium object from file.

    Args:
        file_loc: The MotifCompendium file path.

    Returns:
        The corresponding MotifCompendium object.

    Notes:
        Does not ever explicitly load the object, just the metadata from the object.
    """
    try:
        metadata = pd.read_hdf(file_loc, key="metadata")
    except:
        raise KeyError("File does not contain metadata.")
    try:
        __images = pd.read_hdf(file_loc, key="__images")
    except:
        raise KeyError("File does not contain __images.")
    print(
        f"Motif Compendium with {len(metadata)} motifs.\n--- Metadata ---\n{metadata}\n--- Images ---\n{list(__images.columns)}"
    )
    return metadata


def build(
    motifs: np.ndarray,
    metadata: pd.DataFrame | None = None,
    l2: bool | None = None,
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of motifs.

    Computes pairwise similarities on a set of motifs. Creates metadata if needed. Then,
      passes everything to the MotifCompendium constructor.

    Args:
        motifs: A stack of motifs of shape (N, 30, 8/4).
        metadata: The metadata for all motifs. If None, it will be set to a DataFrame
          with generic motif names.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization) when
          computing motif similarity.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in motifs.

    Notes:
        Motifs are assumed to be in ACGT order if 4-channel or A+A-C+C-G-G+T-T+ order if
          8-channel.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    # Check motifs
    utils_motif.validate_motif_stack(motifs)
    # Compute similarity
    similarity, alignment_fr, alignment_h = utils_similarity.compute_similarities(
        [motifs], [(0, 0)], l2
    )[0]
    # Guarantee similarity properties
    np.fill_diagonal(similarity, 1)  # Sometimes diagonal is 0.999... but should be 1
    similarity = (similarity + similarity.T) / 2  # Ensure symmetric
    # Metadata
    if metadata is None:
        metadata = pd.DataFrame()
        metadata["name"] = [f"motif_{i}" for i in range(motifs.shape[0])]
    # Images
    __images = pd.DataFrame()
    # Construct object
    return MotifCompendium(
        motifs, similarity, alignment_fr, alignment_h, metadata, __images, safe
    )


def build_from_modisco(
    modisco_dict: dict[str, str],
    ic: bool = True,
    l2: bool | None = None,
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of Modisco outputs.

    Loads motifs and metadata from all Modisco outputs then passes them to build().

    Args:
        modisco_dict: A dictionary from model name to modisco file path.
        ic: Whether or not to apply information content scaling to Modisco motifs.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization) when
          computing motif similarity.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in all Modisco objects.

    Notes:
        Assumes the model names have no '-'' or '.' in them.
        Using information content scaling is highly recommended.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    # Load from Modisco
    motifs, motif_names, seqlet_counts, model_names = utils_loader.load_modiscos(
        modisco_dict, ic
    )
    # Build metadata
    metadata = pd.DataFrame()
    metadata["name"] = motif_names
    metadata["num_seqlets"] = seqlet_counts
    metadata["model"] = model_names
    metadata["posneg"] = metadata["name"].str.split(".").str[0].str.split("-").str[1]
    # Construct object
    return build(
        motifs,
        metadata=metadata,
        l2=l2,
        safe=safe,
    )


def combine(
    compendiums: list[MotifCompendium],
    l2: bool | None = None,
    safe: bool = True,
) -> MotifCompendium:
    """Combines multiple MotifCompendium into one MotifCompendium.

    Computes similarity between each MotifCompendium's motifs to construct a single
      large similarity matrix. Then, passes the concatenated set of motifs and overall
      similarity matrix to the MotifCompendium constructor.

    Args:
        compendiums: A list of MotifCompendium objects.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization) when
          computing motif similarity.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs from all individual
          MotifCompendium.

    Notes:
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    n = len(compendiums)
    # SIMILARITIES
    motifs_list = [mc.motifs for mc in compendiums]
    calculations = []
    for i in range(n):
        for j in range(i + 1, n):
            calculations.append((i, j))
    similarity_results = utils_similarity.compute_similarities(
        motifs_list, calculations, l2
    )
    similarity_block = [[None for i in range(n)] for i in range(n)]
    alignment_fr_block = [[None for i in range(n)] for i in range(n)]
    alignment_h_block = [[None for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_block[i][j] = compendiums[i].similarity
                alignment_fr_block[i][j] = compendiums[i].alignment_fr
                alignment_h_block[i][j] = compendiums[i].alignment_h
            elif i < j:
                (
                    similarity_block[i][j],
                    alignment_fr_block[i][j],
                    alignment_h_block[i][j],
                ) = similarity_results[calculations.index((i, j))]
            elif i > j:
                similarity_block[i][j] = similarity_block[j][i].T
                alignment_fr_block[i][j] = alignment_fr_block[j][i].T
                alignment_h_block[i][j] = alignment_h_block[j][i].T
    similarity = np.block(similarity_block)
    # Guarantee diagonal symmetry
    np.fill_diagonal(similarity, 1)
    similarity = (similarity + similarity.T) / 2

    alignment_fr = np.block(alignment_fr_block)
    alignment_h = np.block(alignment_h_block)
    # motifs
    motifs = np.concatenate(motifs_list, axis=0)
    # METADATA
    metadata = pd.concat([mc.metadata for mc in compendiums], ignore_index=True)
    return MotifCompendium(
        motifs, similarity, alignment_fr, alignment_h, metadata, safe
    )


##########################
# MOTIF COMPENDIUM CLASS #
##########################
class MotifCompendium:
    """An object for storing information about motifs and their mutual similarities.

    The MotifCompendium object is intended to be the best way to store, cluster,
      visualize, and analyze patterns in a massive number of motifs. MotifCompendium is
      designed for flexible interfacing and can be indexed like a Pandas DataFrame.
      EX: mc_neural = mc[mc["organ"] == "brain"].

    Attributes:
        motifs: A np.ndarray representing the motifs. Of shape (N, 30, 8/4). motifs[i, :, :]
          represents motif i.
        similarity: A np.ndarray containing the pairwise similarity scores between each
          motif. Of shape (N, N). similarity[i, j] is the similarity between motif i and
          motif j.
        alignment_fr: A np.ndarray containing the forward/reverse complement
          relationship between two motifs. Of shape (N, N). alignment_fr[i, j] is 0/1 if
          motif i should/shouldn't be reverse complemented to best align with motif j.
        alignment_h: A np.ndarray containing the horizontal shift information between
          two motifs. Of shape (N, N). alignment_h[i, j] represents how many bases to
          the right motif i should be shifted to best align with motif j.
        metadata: A pd.DataFrame containing metadata for each motif. Of length N.
          metadata.iloc[i, :] contains metadata about motif i.
        __images: A pd.DataFrame containing UTF-8 embedded images related to each motif.
          Private. Not meant to be modified by the user.
    """

    ##################
    # CORE FUNCTIONS #
    ##################
    def __init__(
        self,
        motifs: np.ndarray,
        similarity: np.ndarray,
        alignment_fr: np.ndarray,
        alignment_h: np.ndarray,
        metadata: pd.DataFrame,
        __images: pd.DataFrame,
        safe: bool,
    ) -> None:
        """MotifCompendium constructor.

        This constructor takes in already defined versions of each class attribute. If
          safe is True, validate() is run; otherwise, it is not run.

        Args:
            motifs: A np.ndarray that is assigned to self.motifs.
            similarity: A np.ndarray that is assigned to self.similarity.
            alignment_fr: A np.ndarray that is assigned to self.alignment_fr.
            alignment_h: A np.ndarray that is assigned to self.alignment_h.
            metadata: A pd.DataFrame that is assigned to self.metadata.
            __images: A pd.DataFrame that is assigned to self.__images.
            safe: Whether or not to construct the MotifCompendium safely.

        Notes:
            In general, users should use factory functions and not access this
              constructor directly.
            Safe construction validates object integrity but may take significantly
              longer for large objects.
        """
        self.motifs = motifs
        self.similarity = similarity
        self.alignment_fr = alignment_fr
        self.alignment_h = alignment_h
        self.metadata = metadata
        self.__images = __images
        if safe:
            self.validate()

    def save(self, save_loc: str) -> None:
        """Saves the MotifCompendium to file.

        Saves the current MotifCompendium to a compressed h5py file.

        Args:
            save_loc: Where to save the MotifCompendium to.
        """
        if not save_loc.endswith(".mc"):
            save_loc += ".mc"
        with h5py.File(save_loc, "w") as f:
            f.create_dataset("motifs", data=self.motifs)
            f.create_dataset("similarity", data=self.similarity)
            f.create_dataset("alignment_fr", data=self.alignment_fr)
            f.create_dataset("alignment_h", data=self.alignment_h)
        self.metadata.to_hdf(save_loc, key="metadata", mode="a")
        self.__images.to_hdf(save_loc, key="__images", mode="a")

    def validate(self) -> None:
        """Verifies the integrity of the MotifCompendium.

        Checks the validity of each attribute (motifs, similarity, similarity_fb, similarity_h).

        Notes:
            This function can take a long time to run, especially for very large MotifCompendium.
        """
        # motifs
        utils_motif.validate_motif_stack(self.motifs)
        # similarity
        if not isinstance(self.similarity, np.ndarray):
            raise TypeError("self.similarity must be a np.ndarray.")
        if not (
            (len(self.similarity.shape) == 2)
            and (np.allclose(self.similarity, self.similarity.T))
        ):
            raise ValueError("self.similarity must be a square transpose matrix.")
        if not ((np.max(self.similarity) == 1) and ((self.similarity >= 0).all())):
            raise ValueError("self.similarity must have similarities between (0, 1].")
        # alignment_fr
        if not isinstance(self.alignment_fr, np.ndarray):
            raise TypeError("self.alignment_fr must be a np.ndarray.")
        if not (
            (len(self.alignment_fr.shape) == 2)
            and (np.allclose(self.alignment_fr, self.alignment_fr.T))
        ):
            raise ValueError("self.alignment_fr must be a square transpose matrix.")
        if not ((self.alignment_fr == 0) | (self.alignment_fr == 1)).all():
            raise ValueError("self.alignment_fr must have values being either 0 or 1.")
        # alignment_h
        if not isinstance(self.alignment_h, np.ndarray):
            raise TypeError("self.alignment_h must be a np.ndarray.")
        if not (len(self.alignment_h.shape) == 2):
            raise ValueError("self.alignment_h must be a square matrix.")
        if not (
            np.allclose(
                self.alignment_h,
                np.where(
                    self.alignment_fr == 0, -self.alignment_h.T, self.alignment_h.T
                ),
            )
        ):
            raise ValueError(
                "self.alignment_h is symmetric for reverse complement motifs and skew-symmetric for motifs that are already aligned."
            )
        # metadata
        if not isinstance(self.metadata, pd.DataFrame):
            raise TypeError("self.metadata must be a pd.DataFrame.")
        # __images
        if not isinstance(self.__images, pd.DataFrame):
            raise TypeError("self.__images must be a pd.DataFrame.")
        # shape matches
        if not (
            (self.motifs.shape[0] == self.similarity.shape[0])
            and (self.motifs.shape[0] == self.alignment_fr.shape[0])
            and (self.motifs.shape[0] == self.alignment_h.shape[0])
            and (self.motifs.shape[0] == len(self.metadata))
        ):
            raise TypeError("Attribute shapes do not align.")

    def get_saved_images(self) -> list[str]:
        """Returns a list of saved images in the MotifCompendium."""
        return list(self.__images.columns)

    def __str__(self) -> str:
        """String representation of the MotifCompendium."""
        return f"Motif Compendium with {len(self.metadata)} motifs.\n--- Metadata ---\n{self.metadata}\n--- Images ---\n{list(self.__images.columns)}"

    def __len__(self) -> int:
        """Length of the MotifCompendium."""
        return len(self.metadata)

    def __getitem__(self, key: str | pd.Series) -> pd.Series | MotifCompendium:
        """Get columns or subsets of the MotifCompendium.

        Allows indexing into the MotifCompendium with the same syntax as a Pandas
          DataFrame.

        Args:
            key: What is being used to subset the object.

        Returns:
            If a string is input, the corresponding column from the metadata is
              returned. If a pd.Series of booleans is given, a new MotifCompendium
              representing just the subset of motifs that were True in the key
              is returned.

        Note:
            When a subset of the MotifCompendium is returned, it is returned as a new
              MotifCompendium object and is not built safely.
        """
        if isinstance(key, str):
            return self.metadata[key]
        elif isinstance(key, pd.Series) and key.dtype == bool:
            # metadata
            metadata_slice = self.metadata[key]
            keep_idxs = list(metadata_slice.index)
            metadata_slice = metadata_slice.reset_index(drop=True)
            # __images
            __images_slice = self.__images[
                key.reindex(self.__images.index)
            ].reset_index(drop=True)
            # matrices
            motifs_slice = self.motifs[keep_idxs, :, :]
            similarity_slice = self.similarity[keep_idxs, :][:, keep_idxs]
            alignment_fr_slice = self.alignment_fr[keep_idxs, :][:, keep_idxs]
            alignment_h_slice = self.alignment_h[keep_idxs, :][:, keep_idxs]
            return MotifCompendium(
                motifs_slice,
                similarity_slice,
                alignment_fr_slice,
                alignment_h_slice,
                metadata_slice,
                __images_slice,
                safe=False,
            )
        else:
            raise TypeError("MotifCompendium cannot be indexed by this.")

    def __setitem__(self, key: str, value) -> None:
        """Set the value of a column in the metadata.

        Allows adding or setting columns of the MotifCompendium with the same syntax as
          a Pandas DataFrame.

        Args:
            key: The name of the column to set.
            value: The value to set that column assignment to.

        Note:
            Works exactly like setting a Pandas DataFrame column does.
        """
        if isinstance(key, str):
            self.metadata[key] = value
        else:
            raise TypeError("MotifCompendium column names must be strings.")

    def __eq__(self, other: MotifCompendium) -> bool:
        """Checks object equality between MotifCompendium."""
        if isinstance(other, MotifCompendium):
            return (
                np.allclose(self.motifs, other.motifs)
                and np.allclose(self.similarity, other.similarity)
                and np.allclose(self.alignment_fr, other.alignment_fr)
                and np.allclose(self.alignment_h, other.alignment_h)
                and self.metadata.equals(other.metadata)
                and self.__images.equals(other.__images)
            )
        return False

    ###########################
    # VIZUALIZATION FUNCTIONS #
    ###########################
    def motif_collection_html(
        self, html_out: str, group_by: str, average_motif: bool = True
    ) -> None:
        """Creates an html file displaying all motifs in the current MotifCompendium.

        Produces an html file at the specified location with all motifs from the current
          MotifCompendium. Motifs are grouped by the group_by field in the metadata.
          Group averages are plotted with a green background at the top of each group.

        Args:
            html_out: The path to save the html file.
            group_by: The column in the metadata to group motifs by for visualization.
              Or, an externally provided pd.Series that fulfills the same role.
            average_motif: Whether or not to show the average motif per cluster.

        Notes:
            Assumes self.metadata has a "name" column.
            If you just want to plot cluster averages, consider doing
              mc.cluster_averages(group_by).motif_collection_html(html_out, "name") with
              the appropriate CPU/GPU/chunking options.
        """
        # Prepare motifs/names/groups
        motifs = (
            utils_motif.motif_8_to_4(self.motifs)
            if self.motifs.shape[2] == 8
            else self.motifs
        )
        names = list(self.metadata["name"])
        if isinstance(group_by, str):
            if group_by in self.metadata.columns:
                groups = list(self.metadata[group_by])
            else:
                raise KeyError(f"{group_by} not in metadata.")
        elif isinstance(group_by, list):
            groups = group_by
        else:
            raise TypeError(
                "group_by must either be a string (column name in metadata) or a list."
            )
        # Group motifs
        motif_groups = dict()  # group name --> {motif name --> motif dict}
        group_seeds = dict()  # group name --> index of seed motif in group
        group_xmin_xmax = dict()  # group name --> group name --> (xmin, xmax) for group
        for i, x in enumerate(groups):
            if x in motif_groups:
                cluster_x_seed = group_seeds[x]
                motif_info_i = utils_plotting.LogoPlottingInput(
                    motifs[i],
                    revcomp=(self.alignment_fr[i, cluster_x_seed] == 1),
                    pos=self.alignment_h[i, cluster_x_seed],
                    name=names[i],
                )
                group_xmin_xmax[x] = (
                    min(group_xmin_xmax[x][0], self.alignment_h[i, cluster_x_seed]),
                    max(
                        group_xmin_xmax[x][1],
                        self.alignment_h[i, cluster_x_seed] + motifs.shape[1],
                    ),
                )
                motif_groups[x].append(motif_info_i)
            else:
                motif_info_i = utils_plotting.LogoPlottingInput(
                    motifs[i], name=names[i]
                )
                motif_groups[x] = [motif_info_i]
                group_seeds[x] = i
                group_xmin_xmax[x] = (0, motifs.shape[1])
        # Reindex all motifs + average
        for group_name, group in motif_groups.items():
            # Reindex all motifs in group
            for motif_info in group:
                motif_info.set_bounds(*group_xmin_xmax[group_name])
            # Average
            if average_motif:
                group_motifs = [motif_info.get_motif_df() for motif_info in group]
                motifs_concat = pd.concat(group_motifs)
                average_motif_df = motifs_concat.groupby(motifs_concat.index).mean()
                min_index_val = min(average_motif_df.index)
                max_index_val = max(average_motif_df.index)
                average_motif_info = utils_plotting.LogoPlottingInput(
                    average_motif_df.to_numpy(),
                    pos=min_index_val,
                    name="AVERAGE",
                    bgcolor="palegreen",
                )
                average_motif_info.set_bounds(min_index_val, max_index_val)
                group.insert(0, average_motif_info)
        # Submit to plotting function
        utils_visualization.motif_collection_html(motif_groups, html_out)

    def summary_table_html(self, html_out: str, columns: list[str] = ["name"]) -> None:
        """Creates an html file summarizing all motifs and metadata about them.

        Produces an html file at the specified location with all motifs from the current
          MotifCompendium. Each motif has one row in the summary table. Logos for the
          forward and reverse complement of each motif will be displayed in the first
          two columns. Columns from the current metadata as well as other images saved
          in the object (which can be viewed with MotifCompendium.get_saved_images())
          can be displayed as columns in the summary table.

        Args:
            html_out: The path to save the html file.
            columns: The list of column names in the metadata or saved images to display
              as columns in the summary table.

        Notes:
            It is highly suggested that this function just be run on a MotifCompendium
              of motif clusters. Consider doing
              mc.cluster_averages(cluster).summary_table_html(html_out, summary_cols).
        """
        # If forward and reverse logos aren't in __images, create and add them
        if "logo (fwd)" not in self.__images.columns:
            motifs = (
                utils_motif.motif_8_to_4(self.motifs)
                if self.motifs.shape[2] == 8
                else self.motifs
            )
            motif_plotting_inputs = [
                utils_plotting.LogoPlottingInput(motif) for motif in motifs
            ]
            self.__images["logo (fwd)"] = [
                motif_input.utf8_plot
                for motif_input in utils_plotting.plot_many_motif_logos(
                    motif_plotting_inputs
                )
            ]
        if "logo (rev)" not in self.__images.columns:
            motifs = (
                utils_motif.motif_8_to_4(self.motifs)
                if self.motifs.shape[2] == 8
                else self.motifs
            )
            motif_plotting_inputs = [
                utils_plotting.LogoPlottingInput(motif, revcomp=True)
                for motif in motifs
            ]
            self.__images["logo (rev)"] = [
                motif_input.utf8_plot
                for motif_input in utils_plotting.plot_many_motif_logos(
                    motif_plotting_inputs
                )
            ]
        # Build table
        columns = ["logo (fwd)", "logo (rev)"] + columns
        table_columns = []
        image_column = []
        for c in columns:
            if c in self.metadata.columns:
                if c in self.__images.columns:
                    raise KeyError(
                        f"Column {c} is a column in metadata and __images. Object invalid."
                    )
                else:
                    table_columns.append(self.metadata[c])
                    image_column.append(False)
            elif c in self.__images.columns:
                table_columns.append(self.__images[c])
                image_column.append(True)
            else:
                raise KeyError(
                    f"{c} must be a column in metadata or a generated image."
                )
        table_df = pd.concat(table_columns, axis=1)

        utils_visualization.table_html(table_df, image_column, html_out)

    def heatmap(
        self,
        annot: bool = False,
        label: bool = False,
        show: bool = False,
        save_loc: str | None = None,
    ) -> None:
        """Creates a heatmap of the similarity matrix.

        Produces a heatmap of the similarity matrix of this MotifCompendium with various
          formatting, display, and save options.

        Args:
            annot: Whether or not to display the similarity score value in each cell in
              the heatmap.
            label: Whether or not to label rows and columns with motif names.
            show: Whether or not to show the heatmap with plt.show().
            save_loc: Where to save the heatmap to. If None, the heatmap is not saved.

        Notes:
            Assumes self.metadata has a "name" column.
            Consider visualizing just one cluster at a time with
              mc[mc["cluster"] == "cluster_1"].heatmap().
        """
        if label:
            utils_plotting.plot_heatmap(
                self.similarity,
                annot=annot,
                labels=list(self.metadata["name"]),
                show=show,
                save_loc=save_loc,
            )
        else:
            utils_plotting.plot_heatmap(
                self.similarity, annot=annot, show=show, save_loc=save_loc
            )

    ######################
    # ANALYSIS FUNCTIONS #
    ######################
    def cluster(
        self,
        algorithm: str = "leiden",
        similarity_threshold: float = 0.9,
        save_name: str = "cluster",
        **kwargs,
    ) -> None:
        """Cluster motifs.

        Cluster motifs using the algorithm of choice and the similarity threshold. The
            cluster assignment will be saved into the metadata in column save_name.

        Args:
            algorithm: The clustering algorithm to cluster with.
                (1) Leiden clustering: "leiden", "weighted_leiden", "modularity_leiden"
                    Leiden clustering, using modularity as quality function.
                (2) CPM Leiden: "cpm_leiden", "cpm_weighted_leiden"
                    Leiden clustering, using Constant Potts Model (CPM) as quality function.
                (3) Connected-components clustering: "cc"
                (4) Dense connected-components clustering: "dense_cc"
                (5) Spectral clustering: "spectral"
            similarity_threshold: The similarity threshold on which to create a graph.
                Any similarity threshold below the threshold will not create an edge.
            save_name: The name of the column in metadata to save motif clustering
                results into.
            **kwargs: Additional named arguments specific to the clustering algorithm of
                choice.

        Notes:
            Review the clustering algorithms available in utils/clustering.cluster().
        """
        self.metadata[save_name] = utils_clustering.cluster(
            similarity_matrix=self.similarity,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            **kwargs,
        )

    def clustering_quality(
        self, cluster_name: str = "cluster", with_names: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Produces a matrix that summarizes the quality of a particular clustering.

        Produce a matrix where diagonal entries represent lowest intra-cluster
          similarity and off-diagonal entries represent highest inter-cluster
          similarities.

        Args:
            cluster_name: The name of the column in metadata containing clustering
              annotations to group motifs by.
            with_names: Whether or not to return a raw quality matrix or a quality
              matrix with row/column labels in the form or a pd.DataFrame.

        Returns:
            A square np.ndarray or pd.DataFrame where diagonal entries represent lowest
              intra-cluster similarity and off-diagonal entries represent highest
              inter-cluster similarities. Returns an np.ndarray if with_names is False
              and a pd.DataFrame otherwise
        """
        # Cache cluster --> idxs dictionary
        ci_idxs = defaultdict(list)
        for i, c in enumerate(self.metadata[cluster_name]):
            ci_idxs[c].append(i)
        # Iterate through pairs of clusters
        clusters = sorted(ci_idxs.keys())
        quality = np.zeros((len(clusters), len(clusters)))
        for i, c1 in enumerate(clusters):
            c1_idxs = ci_idxs[c1]
            c1_similarity_slice = self.similarity[c1_idxs, :]
            for j, c2 in enumerate(clusters):
                if j < i:
                    continue
                c2_idxs = ci_idxs[c2]
                similarity_slice_ij = c1_similarity_slice[:, c2_idxs]
                if i == j:
                    quality[i, j] = np.min(similarity_slice_ij)
                else:
                    quality[i, j] = np.max(similarity_slice_ij)
                    quality[j, i] = quality[i, j]
        if with_names:
            return pd.DataFrame(quality, index=clusters, columns=clusters)
        else:
            return quality

    def cluster_averages(
        self,
        cluster_name: str = "cluster",
        aggregations: list[tuple[str]] = [("name", "count", "num_constituents")],
        weight_col: str | None = None,
        l2: bool | None = None,
        safe: bool = True,
    ) -> MotifCompendium:
        """Creates a MotifCompendium where each motif represents a cluster of motifs.

        For each cluster, the average aligned motif is computed. Specified metadata
          columns can be aggregated and retained in the new cluster. The average motifs
          and aggregated metadata are passed to build() and the resulting
          MotifCompendium in which each motif represents a cluster in the current
          MotifCompendium is returned.

        Args:
            cluster_name: Name of column in metadata, containing clustering
              annotations to group motifs by.
            aggregations: A list of tuples of string: ("source", "method", "save").
              "source": Name of column in metadata to aggregate.
              "method": Aggregation method to use. Ex: "count", "sum", "concatenate".
              "save": Name of column in new metadata, to save the aggregated data.
            weight_col: Name of column in metadata to use for weighted averaging.
            l2: Whether or not to use L2 normalization (instead of sqrt normalization)
              when computing motif similarity.
            safe: Whether or not to construct the MotifCompendium safely.

        Returns:
            A MotifCompendium where each entry represents a motif cluster in the current
              MotifCompendium.
        """
        # Set up aggregations
        aggregations_dicts = [
            {"source": x[0], "method": x[1], "save": x[2], "values": []}
            for x in aggregations
        ]
        # Cache cluster --> idxs dictionary
        cluster_idxs = defaultdict(list)
        for i, c in enumerate(self.metadata[cluster_name]):
            cluster_idxs[c].append(i)
        # Check weights
        if weight_col:
            if weight_col in self.metadata.columns:
                weights = self.metadata[weight_col]
            else:
                raise ValueError(f"{weight_col} is not a column in metadata.")
        else:
            weights = None
        # Prepare averaging
        clusters = sorted(cluster_idxs.keys())
        cluster_motif_avgs, cluster_names = [], []
        for c in clusters:
            # Cluster name
            cluster_names.append(f"{cluster_name}#{c}")
            # Cluster average motif
            c_idxs = cluster_idxs[c]
            motifs_c = self.motifs[c_idxs, :, :]
            alignment_fr_c = self.alignment_fr[c_idxs, :][:, c_idxs]
            alignment_h_c = self.alignment_h[c_idxs, :][:, c_idxs]
            # Average motifs
            motif_avg_c = utils_motif.average_motifs(
                motifs_c,
                alignment_fr_c,
                alignment_h_c,
                weights=weights.loc[c_idxs] if weights is not None else None,
            )
            cluster_motif_avgs.append(motif_avg_c)
            # Aggregations
            for agg_dict in aggregations_dicts:
                agg_c_data = self.metadata.loc[c_idxs, agg_dict["source"]]
                match agg_dict["method"]:
                    case "count":
                        agg_dict["values"].append(len(agg_c_data))
                    case "sum":
                        agg_dict["values"].append(np.sum(agg_c_data))
                    case "average" | "avg":
                        agg_dict["values"].append(np.mean(agg_c_data))
                    case "concatenate" | "concat":
                        agg_dict["values"].append(",".join(sorted(set(agg_c_data))))
                    case "unique":
                        agg_dict["values"].append(len(set(agg_c_data)))
                    case _:
                        raise ValueError(
                            f"{agg_dict['method']} is not a supported aggregation method."
                        )
        # Construct cluster average MotifCompendium
        cluster_motif_avgs = np.stack(cluster_motif_avgs, axis=0)
        metadata = pd.DataFrame()
        metadata["name"] = cluster_names
        for agg_dict in aggregations_dicts:
            metadata[agg_dict["save"]] = agg_dict["values"]
        return build(cluster_motif_avgs, metadata, l2, safe)

    def get_similarity_slice(
        self,
        slice1: pd.Series,
        slice2: pd.Series | None = None,
        with_names: bool = False,
    ) -> pd.DataFrame:
        """Extracts a subset of the similarity matrix.

        Takes in one or two conditions as pd.Series with dtype bool. Subsets the
          similarity matrix according to those conditions and returns the corresponding
          slice of the similarity matrix.

        Args:
            slice1: The first condition to subset on.
            slice2: The second condition to subset on.
            with_names: Whether or not to annotate the index/columns of the output
              DataFrame with motif names.

        Returns:
            A pd.DataFrame containing the similarity scores between two subsets of motifs.

        Notes:
            If only slice1 is provided then the similarity scores between the motifs
              specified by slice1 and all other motifs are returned.
        """
        if not (isinstance(slice1, pd.Series) and (slice1.dtype == bool)):
            raise TypeError("slice1 must be a pd.Series of dtype bool.")
        keep_idxs_1 = self.metadata[slice1].index.tolist()
        if slice2 is None:
            similarity_slice = self.similarity[keep_idxs_1, :]
            if with_names:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(self.metadata[slice1]["name"]),
                    columns=list(self.metadata["name"]),
                )
                return similarity_slice_df
            else:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(keep_idxs_1),
                    columns=list(self.metadata.index.tolist()),
                )
                return similarity_slice_df
        else:
            if not (isinstance(slice2, pd.Series) and (slice2.dtype == bool)):
                raise TypeError("slice2 must be a pd.Series of dtype bool.")
            keep_idxs_2 = self.metadata[slice2].index.tolist()
            similarity_slice = self.similarity[keep_idxs_1, :][:, keep_idxs_2]
            if with_names:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(self.metadata.loc[slice1, "name"]),
                    columns=list(self.metadata.loc[slice2, "name"]),
                )
                return similarity_slice_df
            else:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(keep_idxs_1),
                    columns=list(keep_idxs_2),
                )
                return similarity_slice_df

    def extend(self):
        """Add new motifs to the current MotifCompendium."""
        print("not yet implemented")
        assert False

    def assign_clusters_from_other(
        self,
        other: MotifCompendium,
        other_cluster_col: str = "cluster",
        save_col_sim: str = "mc_match_similarity",
        save_col_match: str = "mc_match",
        save_col_logo: str = "mc_match_logo",
        l2: bool | None = None,
    ) -> None:
        """Assign clusters to motifs based on an existing clustered MotifCompendium.

        Given another MotifCompendium that has already been clustered and assigned
          labels, compute similarity between all the motifs in this MotifCompendium
          and other. The highest similarity and closest motif match will be saved as
          columns save_col_sim and save_col_match in metadata.

        Args:
            other: The other MotifCompendium to compare against.
            other_cluster_col: The column in the other MotifCompendium with cluster
              names of each motif.
            save_col_sim: The column under which the highest similarity will be stored.
            save_col_match: The column under which the closest match will be stored.
            save_col_logo: The column under which the closest match logo will be stored.
            l2: Whether or not to use L2 normalization (instead of sqrt normalization)
              when computing motif similarity.

        Notes:
            Assumes that this MotifCompendium and other have the same dimensions for
              their motifs. (Ex: 4 and 4 or 8 and 8.)
        """
        if self.motifs.shape[2] != other.motifs.shape[2]:
            raise TypeError(
                "MotifCompendiums must have the same motif dimensionality to compare."
            )
        # Compute similarity
        mc_similarity, _, _ = utils_similarity.compute_similarities(
            [self.motifs, other.motifs], [(0, 1)], l2=l2
        )[0]
        closest_similarity = np.max(mc_similarity, axis=1)
        closest_index = np.argmax(mc_similarity, axis=1)
        # Save match information
        self[save_col_sim] = closest_similarity
        other_clusters = other.metadata[other_cluster_col].tolist()
        self[save_col_match] = [other_clusters[x] for x in closest_index]
        if "logo (fwd)" not in other.__images.columns:
            # Generate forward logos if not already generated
            motifs = (
                utils_motif.motif_8_to_4(other.motifs)
                if other.motifs.shape[2] == 8
                else other.motifs
            )
            motif_plotting_inputs = [
                utils_plotting.LogoPlottingInput(motif) for motif in motifs
            ]
            self.__images[save_col_logo] = [
                motif_input.utf8_plot
                for motif_input in utils_plotting.plot_many_motif_logos(
                    motif_plotting_inputs
                )
            ]
        else:
            # Copy forward logos from other MotifCompendium
            other_fwd_strings = other.__images["logo (fwd)"].tolist()
            self.__images[save_col_logo] = [other_fwd_strings[x] for x in closest_index]
