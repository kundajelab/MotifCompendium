from __future__ import annotations
from collections import defaultdict
import os

from bs4 import BeautifulSoup
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
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
    max_chunk: int | None = None,
    fast_plotting: bool | None = None,
    progress_bar: bool | None = None,
):
    """Set default values for max_chunk, max_cpus, and use_gpu.

    Args:
        max_cpus: The maximum number of CPUs to use. Used while loading multiple Modisco
          files, generating plots, and, when use_gpu is False, computing similarity.
        use_gpu: Whether or not to GPU accelerate similarity computations.
        max_chunk: The maximum number of motifs to compute similarity on at a time. SET
          TO -1 TO USE NO CHUNKING.
        fast_plotting: Whether or not to use fast plotting instead of logomaker when
          generating motif plots.
        progress_bar: Whether or not to show progress bars. (Currently only used when
          computing similarity).

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_cpus to the number of available
          CPUs.
        Currently, multi-GPU calculation is not supported.
        If memory constraints are not an issue, set max_chunk to -1 for faster
          performance. Otherwise, set max_chunk so that calculations fit in memory. For
          a GPU with ~12GB of memory, use max_chunk=1152.
    """
    if max_cpus is not None:
        utils_config.set_max_cpus(max_cpus)
    if use_gpu is not None:
        utils_config.set_use_gpu(use_gpu)
    if max_chunk is not None:
        utils_config.set_max_chunk(max_chunk)
    if fast_plotting is not None:
        utils_config.set_fast_plotting(fast_plotting)
    if progress_bar is not None:
        utils_config.set_progress_bar(progress_bar)


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
          'alignment_rc', and 'alignment_h', as well as a DataFrame called 'metadata'
          and another DataFrame called '__images'.
        Old objects may be incompatable with the newest version of the load function.
        Safe loading validates object integrity but may take significantly longer for
          large objects.
    """
    if not os.path.exists(file_loc):
        raise FileNotFoundError(f"File {file_loc} does not exist.")
    try:
        with h5py.File(file_loc, "r") as f:
            motifs = f["motifs"][:]
            similarity = f["similarity"][:]
            alignment_rc = f["alignment_rc"][:]
            alignment_h = f["alignment_h"][:]
        metadata = pd.read_hdf(file_loc, key="metadata")
        __images = pd.read_hdf(file_loc, key="__images")
    except:
        raise ValueError("File does not specify a MotifCompendium.")
    return MotifCompendium(
        motifs, similarity, alignment_rc, alignment_h, metadata, __images, safe
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
    if not os.path.exists(file_loc):
        raise FileNotFoundError(f"File {file_loc} does not exist.")
    try:
        with h5py.File(file_loc, "r") as f:
            motifs_shape = f["motifs"][:].shape
        metadata = pd.read_hdf(file_loc, key="metadata")
        __images = pd.read_hdf(file_loc, key="__images")
    except:
        raise ValueError("File does not specify a MotifCompendium.")
    print(
        f"MotifCompendium with {len(metadata)} motifs."
        + f"\n--- Motifs = {motifs_shape} ---\n"
        + f"\n--- Metadata ---\n{metadata}"
        + f"\n--- Images ---\n{list(__images.columns)}"
    )
    return metadata


def load_old_compendium(file_loc: str) -> MotifCompendium:
    """Loads an old MotifCompendium object from file.

    Args:
        file_loc: The file path of the out of data MotifCompendium object.

    Returns:
        A MotifCompendium object that is up to date.

    Notes:
        Assumes the file is an h5py file with a 'motifs' dataset and a DataFrame called
          'metadata'.
        This function is meant to be used for loading MotifCompendium objects saved
          before the 1.0 release.
        Objects will forcibly be loaded safely.
    """
    if not os.path.exists(file_loc):
        raise FileNotFoundError(f"File {file_loc} does not exist.")
    try:
        with h5py.File(file_loc, "r") as f:
            motifs = f["motifs"][:]
        metadata = pd.read_hdf(file_loc, key="metadata")
    except:
        raise ValueError("File does not specify a MotifCompendium.")
    # Recompute similarity on motifs
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs], [(0, 0)]
    )[0]
    np.fill_diagonal(similarity, 1)  # Sometimes diagonal is 0.999... but should be 1
    similarity = (similarity + similarity.T) / 2  # Ensure symmetric
    # Split metadata into images if needed
    metadata_columns = list(metadata.columns)
    image_columns = [x for x in metadata_columns if x.startswith("LOGOIMAGEDATA__")]
    nonimage_columns = [x for x in metadata_columns if x not in image_columns]
    metadata_nonimage = metadata[nonimage_columns]
    __images = pd.DataFrame(index=metadata.index)
    for image_column in image_columns:
        __images[image_column.split("LOGOIMAGEDATA__")[1]] = metadata[image_column]
    return MotifCompendium(
        motifs,
        similarity,
        alignment_rc,
        alignment_h,
        metadata_nonimage,
        __images,
        safe=True,
    )


def build(
    motifs: np.ndarray,
    metadata: pd.DataFrame | None = None,
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of motifs.

    Computes pairwise similarities on a set of motifs. Creates metadata if needed. Then,
      passes everything to the MotifCompendium constructor.

    Args:
        motifs: A stack of motifs of shape (N, L, 8/4).
        metadata: The metadata for all motifs. If None, it will be set to a DataFrame
          with generic motif names.
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
    utils_motif.validate_motif_stack_compendium(motifs)
    # Metadata
    if metadata is None:
        metadata = pd.DataFrame()
        metadata["name"] = [f"motif_{i}" for i in range(motifs.shape[0])]
    elif not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pd.DataFrame.")
    # Compute similarity
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs], [(0, 0)]
    )[0]
    np.fill_diagonal(similarity, 1)  # Sometimes diagonal is 0.999... but should be 1
    similarity = (similarity + similarity.T) / 2  # Ensure symmetric
    # Images
    __images = pd.DataFrame(index=metadata.index)
    # Construct object
    return MotifCompendium(
        motifs, similarity, alignment_rc, alignment_h, metadata, __images, safe
    )


def build_from_modisco(
    modisco_dict: dict[str, str],
    ic: bool = True,
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of Modisco outputs.

    Loads motifs and metadata from all Modisco outputs then passes them to build().

    Args:
        modisco_dict: A dictionary from model name to modisco file path.
        ic: Whether or not to apply information content scaling to Modisco motifs.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in all Modisco objects.

    Notes:
        Assumes the model names have no '-'s or '.'s in them.
        Using information content scaling is highly recommended.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    # Load from Modisco
    motifs, motif_names, seqlet_counts, model_names, posneg = (
        utils_loader.load_modiscos(modisco_dict, ic)
    )
    # Convert motifs to normalized 8-channel motifs
    motifs = utils_motif.motif_4_to_8(motifs)
    motifs /= np.sum(motifs, axis=(1, 2), keepdims=True)
    # Build metadata
    metadata = pd.DataFrame()
    metadata["name"] = motif_names
    metadata["num_seqlets"] = seqlet_counts
    metadata["model"] = model_names
    metadata["posneg"] = posneg
    # Construct object
    return build(
        motifs,
        metadata=metadata,
        safe=safe,
    )


def combine(
    compendiums: list[MotifCompendium],
    safe: bool = True,
) -> MotifCompendium:
    """Combines multiple MotifCompendium into one MotifCompendium.

    Computes similarity between each MotifCompendium's motifs to construct a single
      large similarity matrix. The metadata and images from each MotifCompendium are
      combined, as well, but an error will be thrown if the metadata and images of each
      MotifCompendium do not have the same columns. Then, the motifs, similarity and
      alignment matrices, and metadata are passed to the MotifCompendium constructor.

    Args:
        compendiums: A list of MotifCompendium objects.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs from all individual
          MotifCompendium.

    Notes:
        The metadata and __images of each MotifCompendium must contain the same columns.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    # Check inputs
    if not (
        isinstance(compendiums, list)
        and all([isinstance(x, MotifCompendium) for x in compendiums])
    ):
        raise TypeError("The input must be a list of MotifCompendium objects.")
    if len(compendiums) == 0:
        raise ValueError("The input list must contain at least one MotifCompendium.")
    # Confirm that the motifs from each MotifCompendium are compatible
    motifs_length = compendiums[0].motifs.shape[1]
    if not all([x.motifs.shape[1] == motifs_length for x in compendiums]):
        raise ValueError(
            "The motifs of the MotifCompendium objects must have the same length."
        )
    motifs_channels = compendiums[0].motifs.shape[2]
    if not all([x.motifs.shape[2] == motifs_channels for x in compendiums]):
        raise ValueError(
            "The motifs of the MotifCompendium objects must have the same channels."
        )
    # Confirm that the metadata and __images of each MotifCompendium has the same columns
    metadata_columns = compendiums[0].columns()
    if not all([x.columns() == metadata_columns] for x in compendiums):
        raise ValueError(
            "The metadata of each MotifCompendium must have the same columns."
            + "\n(Check mc.columns() for each MotifCompendium.)"
        )
    image_columns = compendiums[0].get_saved_images()
    if not all([x.get_saved_images() == image_columns] for x in compendiums):
        raise ValueError(
            "Each MotifCompendium must have the same saved images."
            + "\n(Check mc.get_saved_images() for each MotifCompendium.)"
        )
    n = len(compendiums)
    # Combine similarities
    motifs_list = [mc.motifs for mc in compendiums]
    calculations = []
    for i in range(n):
        for j in range(i + 1, n):
            calculations.append((i, j))
    similarity_results = utils_similarity.compute_similarities(
        motifs_list, calculations
    )
    similarity_block = [[None for _ in range(n)] for _ in range(n)]
    alignment_rc_block = [[None for _ in range(n)] for _ in range(n)]
    alignment_h_block = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_block[i][j] = compendiums[i].similarity
                alignment_rc_block[i][j] = compendiums[i].alignment_rc
                alignment_h_block[i][j] = compendiums[i].alignment_h
            elif i < j:
                (
                    similarity_block[i][j],
                    alignment_rc_block[i][j],
                    alignment_h_block[i][j],
                ) = similarity_results[calculations.index((i, j))]
            elif i > j:
                similarity_block[i][j] = similarity_block[j][i].T
                alignment_rc_block[i][j] = alignment_rc_block[j][i].T
                alignment_h_block[i][j] = alignment_h_block[j][i].T
    similarity = np.block(similarity_block)
    alignment_rc = np.block(alignment_rc_block)
    alignment_h = np.block(alignment_h_block)
    # Motifs
    motifs = np.concatenate(motifs_list, axis=0)
    # Metadata
    metadata = pd.concat([mc.metadata for mc in compendiums], ignore_index=True)
    # Images
    __images = pd.DataFrame()
    for images in compendiums[0].get_saved_images():
        __images[images] = pd.concat(
            [mc.get_images(images) for mc in compendiums], ignore_index=True
        )
    # Construct object
    return MotifCompendium(
        motifs,
        similarity,
        alignment_rc,
        alignment_h,
        metadata,
        __images,
        safe,
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
        motifs: A np.ndarray representing the motifs. Of shape (N, 30, 8/4).
          motifs[i, :, :] represents motif i.
        similarity: A np.ndarray of floats containing the pairwise similarity scores
          between each motif. Of shape (N, N). similarity[i, j] is the similarity
          between motif i and motif j and can be within [0, 1].
        alignment_rc: A np.ndarray of bools containing the reverse complement
          relationship between two motifs. Of shape (N, N). alignment_rc[i, j] is True
          if motif i should be reverse complemented to best align with motif j.
        alignment_h: A np.ndarray of ints containing the horizontal shift information
          between two motifs. Of shape (N, N). alignment_h[i, j] represents how many
          bases to the right motif j should be shifted (after being reverse complemented
          if needed) to best align with motif i. alignment_h[i] gives a vector of how
          motifs should be shifted to align with motif i.
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
        alignment_rc: np.ndarray,
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
            alignment_rc: A np.ndarray that is assigned to self.alignment_rc.
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
        self.alignment_rc = alignment_rc
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
            f.create_dataset("alignment_rc", data=self.alignment_rc)
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
        utils_motif.validate_motif_stack_compendium(self.motifs)
        # similarity
        if not (
            isinstance(self.similarity, np.ndarray)
            and self.similarity.dtype == np.single
        ):
            raise TypeError("self.similarity must be a np.ndarray of floats.")
        if not (
            (len(self.similarity.shape) == 2)
            and (self.similarity == self.similarity.T).all()
        ):
            raise ValueError("self.similarity must be a square transpose matrix.")
        if not (np.max(self.similarity) <= 1) and (np.min(self.similarity) >= 0):
            raise ValueError("self.similarity must have similarities between [0, 1].")
        if not (np.diag(self.similarity) == 1).all():
            raise ValueError("self.similarity must have 1s on the diagonal.")
        # alignment_rc
        if not (
            isinstance(self.alignment_rc, np.ndarray)
            and self.alignment_rc.dtype == np.bool_
        ):
            raise TypeError("self.alignment_rc must be a np.ndarray of bools.")
        if not (
            (len(self.alignment_rc.shape) == 2)
            and (self.alignment_rc == self.alignment_rc.T).all()
        ):
            raise ValueError("self.alignment_rc must be a square transpose matrix.")
        if not ((self.alignment_rc == 0) | (self.alignment_rc == 1)).all():
            raise ValueError("self.alignment_rc must have values being either 0 or 1.")
        # alignment_h
        if not (
            isinstance(self.alignment_h, np.ndarray)
            and self.alignment_h.dtype == np.byte
        ):
            raise TypeError("self.alignment_h must be a np.ndarray of ints.")
        if not (len(self.alignment_h.shape) == 2):
            raise ValueError("self.alignment_h must be a square matrix.")
        if not (
            (
                self.alignment_h
                == np.where(
                    self.alignment_rc == 0, -self.alignment_h.T, self.alignment_h.T
                )
            ).all()
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
        if not set(self.metadata.columns).isdisjoint(self.__images.columns):
            raise ValueError("self.metadata and self.__images must share no columns.")
        # shape matches
        if not (
            (self.motifs.shape[0] == self.similarity.shape[0])
            and (self.motifs.shape[0] == self.alignment_rc.shape[0])
            and (self.motifs.shape[0] == self.alignment_h.shape[0])
            and (self.motifs.shape[0] == len(self.metadata))
            and (self.motifs.shape[0] == len(self.__images))
        ):
            raise TypeError("Attribute shapes do not align.")

    def __eq__(self, other: MotifCompendium) -> bool:
        """Checks object equality between MotifCompendium."""
        if isinstance(other, MotifCompendium):
            return (
                (self.motifs == other.motifs).all()  # motifs equal
                and (self.similarity == other.similarity).all()  # similarity equal
                and (
                    self.alignment_rc == other.alignment_rc
                ).all()  # alignment_rc equal
                and (self.alignment_h == other.alignment_h).all()  # alignment_h equal
                and self.metadata.equals(other.metadata)  # metadata equal
                and (
                    (
                        sorted(self.get_saved_images())
                        == sorted(other.get_saved_images())
                    )
                    and all(
                        [
                            self.get_images(x) == other.get_images(x)
                            for x in self.get_saved_images()
                        ]
                    )
                )  # images equal (columns equal and each column value equal)
            )
        return False

    def __str__(self) -> str:
        """String representation of the MotifCompendium."""
        return (
            f"MotifCompendium with {len(self.metadata)} motifs."
            + f"\n--- Motifs = {self.motifs.shape} ---\n"
            + f"\n--- Metadata ---\n{self.metadata}"
            + f"\n--- Images ---\n{list(self.__images.columns)}"
        )

    def _repr_html_(self) -> str:
        """HTML representation of the MotifCompendium for display in notebooks."""
        html = f"<h2>MotifCompendium with {len(self.metadata)} motifs</h2>"
        # Motifs
        html += f"<h3>Motifs = {self.motifs.shape}</h3>"
        # Metadata
        html += "<h3>Metadata</h3>"
        html += self.metadata._repr_html_()
        # Images section
        html += "<h3>Images</h3>"
        if self.__images.columns.empty:
            html += "<p>No images available</p>"
        else:
            html += "<ul>"
            for col in self.__images.columns:
                html += f"<li>{col}</li>"
            html += "</ul>"
        return html

    def __len__(self) -> int:
        """Length of the MotifCompendium."""
        return self.motifs.shape[0]

    ########################
    # OBJECT MANIPULATIONS #
    ########################
    # MOTIFS
    def get_standard_motif_stack(self) -> np.ndarray:
        """Returns motifs in a standard (N, L, 4) shape."""
        if self.motifs.shape[2] == 4:
            return self.motifs
        return utils_motif.motif_8_to_4_signed(self.motifs)

    # METADATA
    def columns(self) -> list[str]:
        """Returns the columns of the metadata."""
        return list(self.metadata.columns)

    def rename_columns(self, mapper: dict[str, str]) -> None:
        """Renames the columns of the metadata in a Pandas-like syntax."""
        if not isinstance(mapper, dict):
            raise TypeError("Renaming must be done with a dictionary.")
        if not all([x in self.columns() for x in mapper.keys()]):
            raise KeyError("All keys in the mapper must be columns in the metadata.")
        self.metadata.rename(columns=mapper, inplace=True)

    def delete_column(self, column: str | list[str]) -> None:
        """Deletes the specified column(s) from the metadata."""
        if not (
            isinstance(column, str)
            or (isinstance(column, list) and all([isinstance(x, str) for x in column]))
        ):
            raise TypeError("column must be a string or a list of strings.")
        self.metadata.drop(column, axis=1, inplace=True)

    # IMAGES
    def get_saved_images(self) -> list[str]:
        """Returns a list of saved images in the MotifCompendium."""
        return list(self.__images.columns)

    def get_images(self, image_name: str) -> list[str]:
        """Returns a list of saved images as utf8 str in the MotifCompendium by column name."""
        if image_name not in self.__images.columns:
            raise KeyError(f"{image_name} is not a saved image.")
        return self.__images[image_name].tolist()

    def add_logos(self, motifs: np.ndarray, image_name: str) -> None:
        """Saves logos of the provided motifs as saved images.

        Args:
            motifs: The motifs to save logos for. Of shape (N, L, 4).
            image_name: The name of the images to save the logos as.
        """
        # Check inputs
        utils_motif.validate_motif_stack_standard(motifs)
        if not isinstance(image_name, str):
            raise TypeError("image_name must be a string.")
        if image_name in self.columns():
            raise ValueError(
                f"{image_name} is already a metadata column. Names may not overlap."
            )
        # Prepare plotting
        logo_plotting_inputs = [
            utils_plotting.LogoPlottingInput(motif=m) for m in motifs
        ]
        # Plot and save
        self.__images[image_name] = [
            motif_input.utf8_plot
            for motif_input in utils_plotting.plot_many_motif_logos(
                logo_plotting_inputs
            )
        ]

    def rename_images(self, mapper: dict[str, str]) -> None:
        """Renames the saved images in the MotifCompendium in a Pandas-like syntax."""
        if not isinstance(mapper, dict):
            raise TypeError("Renaming must be done with a dictionary.")
        if not all([x in self.get_saved_images() for x in mapper.keys()]):
            raise KeyError(
                "All keys in the mapper must an existing set of saved images."
            )
        self.__images.rename(columns=mapper, inplace=True)

    def delete_images(self, image_name: str | list[str]) -> None:
        """Deletes the specified saved images."""
        if not (
            isinstance(image_name, str)
            or (
                isinstance(image_name, list)
                and all([isinstance(x, str) for x in image_name])
            )
        ):
            raise TypeError("image_name must be a string or a list of strings.")
        self.__images.drop(image_name, axis=1, inplace=True)

    # SLICING AND SORTING
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
            __images_slice = self.__images.iloc[keep_idxs].reset_index(drop=True)
            # matrices
            motifs_slice = self.motifs[keep_idxs, :, :]
            similarity_slice = self.similarity[keep_idxs, :][:, keep_idxs]
            alignment_rc_slice = self.alignment_rc[keep_idxs, :][:, keep_idxs]
            alignment_h_slice = self.alignment_h[keep_idxs, :][:, keep_idxs]
            return MotifCompendium(
                motifs_slice,
                similarity_slice,
                alignment_rc_slice,
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

    def sort(
        self,
        by: str | list[str],
        ascending: bool | list[bool] = True,
        inplace: bool = False,
    ) -> MotifCompendium | None:
        """Sort the MotifCompendium.

        Sorts the MotifCompendium by the specified columns in the metadata.

        Args:
            by: The column or columns to sort by.
            ascending: Whether or not to sort in ascending order. Must have one per column.
            inplace: Whether or not to sort the MotifCompendium in place. If False,
              returns a new MotifCompendium. If True, the current MotifCompendium is
              reordered to be sorted.

        """
        # Check inputs
        if not (
            isinstance(by, str)
            or (isinstance(by, list) and all([isinstance(x, str) for x in by]))
        ):
            raise TypeError("by must be a string or a list of strings.")
        if isinstance(by, str):
            if by not in self.metadata.columns:
                raise KeyError(f"{by} not in metadata.")
        else:
            for x in by:
                if x not in self.metadata.columns:
                    raise KeyError(f"{x} not in metadata.")
        if not (
            isinstance(ascending, bool)
            or (
                isinstance(ascending, list)
                and all([isinstance(x, bool) for x in ascending])
            )
        ):
            raise TypeError("ascending must be a bool or a list of bools.")
        if isinstance(by, str):
            # by = string, ascending = bool --> allowed
            # by = string, ascending = list --> not allowed
            if not isinstance(ascending, bool):
                raise ValueError(
                    "ascending must be a single bool if sorting on a single column."
                )
        else:
            # by = list, ascending = bool --> expand ascending to match by
            if isinstance(ascending, bool):
                ascending = [ascending] * len(by)
            # by = list, ascending = list --> lengths must match
            if not (len(by) == len(ascending)):
                raise ValueError("by and ascending must be the same length.")
        # Sort
        metadata_sorted = self.metadata.sort_values(by=by, ascending=ascending)
        sorted_idx = list(metadata_sorted.index)
        # Create sorted attributes
        metadata_sorted.reset_index(drop=True, inplace=True)
        __images_sorted = self.__images.iloc[sorted_idx].reset_index(drop=True)
        motifs_sorted = self.motifs[sorted_idx, :, :]
        similarity_sorted = self.similarity[sorted_idx, :][:, sorted_idx]
        alignment_rc_sorted = self.alignment_rc[sorted_idx, :][:, sorted_idx]
        alignment_h_sorted = self.alignment_h[sorted_idx, :][:, sorted_idx]
        # Return
        if inplace:
            self.metadata = metadata_sorted
            self.__images = __images_sorted
            self.motifs = motifs_sorted
            self.similarity = similarity_sorted
            self.alignment_rc = alignment_rc_sorted
            self.alignment_h = alignment_h_sorted
        else:
            return MotifCompendium(
                motifs_sorted,
                similarity_sorted,
                alignment_rc_sorted,
                alignment_h_sorted,
                metadata_sorted,
                __images_sorted,
                safe=False,
            )

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

    ########################
    # CLUSTERING FUNCTIONS #
    ########################
    def cluster(
        self,
        algorithm: str = "cpm_leiden",
        similarity_threshold: float = 0.9,
        save_name: str = "cluster",
        cluster_on: str | None = None,
        cluster_within: str | None = None,
        cluster_within_on: tuple[str, str] | None = None,
        cluster_on_weight: str | None = None,
        **kwargs,
    ) -> None:
        """Cluster motifs.

        Cluster motifs using the algorithm of choice and the similarity threshold. The
            cluster assignment will be saved into the metadata in column save_name.

        Args:
            algorithm: The clustering algorithm to cluster with. Please see
              MotifCompendium.utils.clustering.cluster() for available algorithms.
            similarity_threshold: The similarity threshold above which motifs should be
              considered similar.
            save_name: The name of the column in the metadata to save motif clustering
                results into.
            cluster_on: The name of the column in metadata containing cluster
              annotations to group motifs by. If None, the clustering will be done on
              the entire MotifCompendium. If not None, the clustering will be done on
              the average motifs of the clusters in this column.
            cluster_within: The name of the column in metadata containing cluster
              annotations to perform clustering within. If None, the clustering will be
              done on the entire MotifCompendium. If not None, the clustering will be
              done per cluster in the cluster_within column. This will identify
              subclusters of the cluster_within_column.
            weight_col: The name of the column in metadata containing weights to use
              when averaging (for cluster_on). If None, all motifs are equally weighted.
            recursive: Whether or not to cluster recursively. If True, the clustering
            **kwargs: Additional named arguments specific to the clustering algorithm of
                choice.

        Notes:
            Review MotifCompendium.utils.clustering.cluster() for available clustering
              algorithms and algorithm-specific arguments.
        """
        # Check arguments
        if not (
            isinstance(similarity_threshold, float) and (0 <= similarity_threshold <= 1)
        ):
            raise ValueError("similarity_threshold must be a float between [0, 1].")
        
        # Cluster on
        if cluster_on:
            if not cluster_on in self.metadata.columns:
                raise KeyError(f"{cluster_on} not in metadata.")
            # Average, keeping cluster_within
            mc = self.cluster_averages(
                cluster_col=cluster_on,
                aggregations=[(cluster_within, "concat", cluster_within)],
                weight_col=weight_col,
            ) if cluster_within else self.cluster_averages(
                cluster_col=cluster_on,
                weight_col=weight_col,
            )
        else:
            mc = self
        
        # Cluster within
        if cluster_within:
            if not cluster_within in self.metadata.columns:
                raise KeyError(f"{cluster_within} not in metadata.")
            for c in set(mc[cluster_within]):
                mc_c = mc[mc[cluster_within] == c]
                c_idxs = mc.metadata[mc[cluster_within] == c].index.tolist()
                clusters_c = utils_clustering.cluster(
                    similarity_matrix=mc_c.similarity,
                    algorithm=algorithm,
                    similarity_threshold=similarity_threshold,
                    **kwargs,
                )
                clusters_c = [f"{c}-{c_sub}" for c_sub in clusters_c]
                mc.metadata.loc[c_idxs, save_name] = clusters_c
        else:
            # Cluster standard
            mc.metadata[save_name] = utils_clustering.cluster(
                similarity_matrix=mc.similarity,
                algorithm=algorithm,
                similarity_threshold=similarity_threshold,
                **kwargs,
            )
        
        # Cluster on: Map back to self
        if cluster_on:
            cluster_map = {
                row["source_cluster"]: row[save_name]
                for _, row in mc.metadata.iterrows()
            }
            self.metadata[save_name] = [cluster_map[c] for c in self.metadata[cluster_on]]

        # Recursive
        min_len = len(self)
        while recursive:
            cur_len = len(self.metadata[save_name].unique())
            # Check if clustering is done
            if cur_len == min_len:
                break
            else:
                min_len = cur_len
                self.cluster(
                    algorithm=algorithm,
                    similarity_threshold=similarity_threshold,
                    save_name=save_name,
                    cluster_on=save_name,
                    cluster_within=cluster_within,
                    weight_col=weight_col,
                    recursive=False,
                    **kwargs,
                )


    def clustering_quality(
        self, clustering: str, with_stats: bool = False
    ) -> pd.DataFrame:
        """Produces a pd.DataFrame that summarizes the quality of particular clustering.

        Produces a matrix where diagonal entries represent lowest intra-cluster
          similarities and off-diagonal entries represent highest inter-cluster
          similarities. This is useful for evaluating the quality of a clustering
          algorithm. Having high diagonal values and low off-diagonal values is better.
          The matrix is returned as a pd.DataFrame(). To access the original matrix, do
          .clustering_quality().to_numpy(). If with_stats is True, additional columns
          explicitly identifying the motifs driving low intra-cluster and high inter-
          cluster similarities will be added.

        Args:
            clustering: The name of the column in metadata containing cluster
              annotations to group motifs by.
            with_stats: Whether or not to compute and store extra quality statistics
              columns. If True, it adds the following columns, which have per-cluster
              information to the returned pd.DataFrame:
                - "lowest_internal_similarity": The lowest internal similarity.
                - "lowest_internal_similarity_motif1_name": The name of the first motif
                  contributing to the lowest internal similarity.
                - "lowest_internal_similarity_motif1_motif": The motif of the first
                  motif contributing to the lowest internal similarity.
                - "lowest_internal_similarity_motif2_name": The name of the second motif
                  contributing to the lowest internal similarity.
                - "lowest_internal_similarity_motif2_motif": The motif of the second
                  motif contributing to the lowest internal similarity.
                - "highest_external_similarity": The highest external similarity.
                - "highest_external_similarity_cluster": The cluster with which there is
                  a high external similarity.
                - "highest_external_similarity_motif_name": The name of the motif in the
                  external cluster that is driving high external similarity.
                - "highest_external_similarity_motif_motif": The motif of the motif in
                  the external cluster that is driving high external similarity.

        Returns:
            A pd.DataFrame containing information about the lowest intra-cluster
              similarities and the highest inter-cluster similarities per cluster. Each
              cluster has a row (index) and column corresponding to it. If with_stats is
              True, then additional columns summarizing the quality information per
              cluster are added as extra columns.

        Notes:
            If with_stats is True, it is assumed that "name" is a column in the
              MotifCompendium.
        """
        # Cache cluster --> idxs dictionary
        ci_idxs = defaultdict(list)
        for i, c in enumerate(self.metadata[clustering]):
            ci_idxs[c].append(i)
        # Iterate through pairs of clusters
        clusters = sorted(ci_idxs.keys())
        quality = np.zeros((len(clusters), len(clusters)))
        if with_stats:
            stats = pd.DataFrame(
                index=clusters,
                columns=[
                    "lowest_internal_similarity",
                    "lowest_internal_similarity_motif1_name",
                    "lowest_internal_similarity_motif1_motif",
                    "lowest_internal_similarity_motif2_name",
                    "lowest_internal_similarity_motif2_motif",
                    "highest_external_similarity",
                    "highest_external_similarity_cluster",
                    "highest_external_similarity_motif_name",
                    "highest_external_similarity_motif_motif",
                ],
            )
            motif_names = list(self["name"])
            motifs_standard = self.get_standard_motif_stack()
            external_similarity_culprit_idxs = np.zeros(
                (len(clusters), len(clusters)), dtype=np.int32
            )  # [i, j] --> in cluster i, which motif (in j) drove similarity
        for i, c1 in enumerate(clusters):
            c1_idxs = ci_idxs[c1]
            c1_similarity_slice = self.similarity[c1_idxs, :]
            for j, c2 in enumerate(clusters):
                # Lower triangle --> fill in during upper triangle calculation
                if j < i:
                    continue
                # Upper triangle + main diagonal
                c2_idxs = ci_idxs[c2]
                similarity_slice_ij = c1_similarity_slice[:, c2_idxs]
                # Main diagonal
                if i == j:
                    # Lowest internal similarity
                    min_idxs = np.unravel_index(
                        np.argmin(similarity_slice_ij), similarity_slice_ij.shape
                    )
                    quality[i, i] = similarity_slice_ij[min_idxs]
                    if with_stats:
                        culprit_idx_1 = c1_idxs[min_idxs[0]]
                        culprit_idx_2 = c1_idxs[min_idxs[1]]
                        stats.loc[c1, "lowest_internal_similarity"] = quality[i, i]
                        stats.loc[c1, "lowest_internal_similarity_motif1_name"] = (
                            motif_names[culprit_idx_1]
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif1_motif"] = (
                            motifs_standard[culprit_idx_1]
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif2_name"] = (
                            motif_names[culprit_idx_2]
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif2_motif"] = (
                            motifs_standard[culprit_idx_2]
                        )
                # Upper triangle
                else:
                    # Highest external similarity
                    max_idxs = np.unravel_index(
                        np.argmax(similarity_slice_ij), similarity_slice_ij.shape
                    )
                    quality[i, j] = similarity_slice_ij[max_idxs]
                    quality[j, i] = quality[i, j]
                    if with_stats:
                        culprit_idx_c1 = c1_idxs[max_idxs[0]]
                        external_similarity_culprit_idxs[j, i] = culprit_idx_c1
                        culprit_idx_c2 = c2_idxs[max_idxs[1]]
                        external_similarity_culprit_idxs[i, j] = culprit_idx_c2
        quality_df = pd.DataFrame(quality, index=clusters, columns=clusters)
        # Add stats if needed
        if with_stats:
            # Look at highest external similarities
            external_sim = quality * (1 - np.eye(quality.shape[0]))
            highest_external_sim_idx = external_sim.argmax(axis=0)
            stats["highest_external_similarity"] = [
                quality[i, highest_external_sim_idx[i]] for i in range(len(clusters))
            ]
            stats["highest_external_similarity_cluster"] = [
                clusters[x] for x in highest_external_sim_idx
            ]
            highest_external_similarity_motif_idxs = [
                external_similarity_culprit_idxs[i, highest_external_sim_idx[i]]
                for i in range(len(clusters))
            ]
            stats["highest_external_similarity_motif_name"] = [
                motif_names[x] for x in highest_external_similarity_motif_idxs
            ]
            stats["highest_external_similarity_motif_motif"] = [
                motifs_standard[x] for x in highest_external_similarity_motif_idxs
            ]
            # Concatenate columns
            quality_df = pd.concat([quality_df, stats], axis=1)
        # Return
        return quality_df

    def cluster_averages(
        self,
        clustering: str,
        aggregations: list[tuple[str]] = [("name", "count", "num_constituents")],
        weight_col: str | None = None,
        compute_quality_stats: bool = False,
    ) -> MotifCompendium:
        """Creates a MotifCompendium where each motif represents a cluster of motifs.

        For each cluster, the average aligned motif is computed. Specified metadata
          columns can be aggregated and retained in the new cluster. The average motifs
          and aggregated metadata are passed to build() and the resulting
          MotifCompendium in which each motif represents a cluster in the current
          MotifCompendium is returned.

        Args:
            clustering: The name of the column in metadata containing cluster
              annotations to group motifs by. Each row in the new MotifCompendium will
              correspond to a value in this column.
            aggregations: A list of tuples of strings: (source, method, save).
                source: The name of the column in the metadata to aggregate.
                method: The aggregation method to use. The possible choices are:
                    - "count": This option counts the number of values in the source
                        column for each cluster.
                    - "unique": This option counts the number of unique values in the
                        source column for each cluster.
                    - "sum": This option sums the values in the source column for each
                        cluster.
                    - "average" or "avg": This option averages the values in the source
                        column for each cluster.
                    - "concatenate" or "concat": This option lists all the unique values
                        in the source column for each cluster.
                    - "concat_counted": This option lists all the unique values and
                        their number of occurrences in the source column for each
                        cluster.
                save: The name of the column in the new metadata to save the aggregated
                  data to.
            weight_col: The name of the metadata column to be used to weight motifs when
              computing motif averages. Should be numeric.
            compute_quality_stats: Whether or not to compute quality statistics for the
              clustering. If True, the quality statistics will be saved in the metadata
              of the returned MotifCompendium.

        Returns:
            A MotifCompendium where each entry represents a motif cluster in the current
              MotifCompendium.

        Notes:
            Any non-supported aggregations types must be added manually to the cluster
              average MotifCompendium after it has been created.
            Turning compute_quality_stats off can save time but the quality statistics
              it generates are useful to have.
        """
        # Set up aggregations
        aggregations_dicts = []
        for x in aggregations:
            if x[2] in [
                "name",
                "source_cluster",
                "min_internal_similarity",
                "max_external_similarity",
            ]:
                raise ValueError(f"{x[2]} is a reserved column name.")
            aggregations_dicts.append(
                {"source": x[0], "method": x[1], "save": x[2], "values": []}
            )
        # Check weights
        if weight_col is not None:
            if weight_col in self.metadata.columns:
                weights = self.metadata[weight_col].to_numpy()
            else:
                raise ValueError(f"{weight_col} is not a column in metadata.")
        else:
            weights = None
        # Cache cluster --> idxs dictionary
        cluster_idxs = defaultdict(list)
        for i, c in enumerate(self.metadata[clustering]):
            cluster_idxs[c].append(i)
        # Perform averaging per cluster
        clusters = sorted(cluster_idxs.keys())
        cluster_motif_avgs, cluster_names = [], []
        for c in clusters:
            # Cluster name
            cluster_names.append(f"{clustering}#{c}")
            # Cluster average motif
            c_idxs = cluster_idxs[c]
            motifs_c = self.motifs[c_idxs, :, :]
            alignment_rc_c = self.alignment_rc[c_idxs, :][:, c_idxs][
                0, :
            ]  # vector of alignment
            alignment_h_c = self.alignment_h[c_idxs, :][:, c_idxs][
                0, :
            ]  # vector of alignment
            weights_c = (
                None if weights is None else weights[c_idxs]
            )  # vector of weights (if using weights)
            # Average motifs
            motif_avg_c = utils_motif.average_motifs(
                motifs_c, alignment_rc_c, alignment_h_c, weights=weights_c
            )
            cluster_motif_avgs.append(motif_avg_c)
            # Aggregations
            for agg_dict in aggregations_dicts:
                agg_c_data = self.metadata.loc[c_idxs, agg_dict["source"]]
                match agg_dict["method"]:
                    case "count":
                        agg_dict["values"].append(len(agg_c_data))
                    case "unique":
                        agg_dict["values"].append(len(set(agg_c_data)))
                    case "sum":
                        agg_dict["values"].append(np.sum(agg_c_data))
                    case "average" | "avg":
                        agg_dict["values"].append(np.mean(agg_c_data))
                    case "concatenate" | "concat":
                        agg_dict["values"].append(",".join(sorted(set(map(str,agg_c_data)))))
                    case "concat_counted":
                        val_counts = defaultdict(int)
                        for x in agg_c_data:
                            val_counts[x] += 1
                        val_strings = []
                        for x in sorted(val_counts.keys()):
                            val_strings.append(f"{x} ({val_counts[x]})")
                        agg_dict["values"].append(",".join(val_strings))
                    case _:
                        raise ValueError(
                            f"{agg_dict['method']} is not a supported aggregation method."
                        )
        # Construct cluster average MotifCompendium
        cluster_motif_avgs = np.stack(cluster_motif_avgs, axis=0)
        metadata = pd.DataFrame()
        metadata["name"] = cluster_names
        metadata["source_cluster"] = clusters
        for agg_dict in aggregations_dicts:
            metadata[agg_dict["save"]] = agg_dict["values"]
        mc_avg = build(cluster_motif_avgs, metadata, safe=False)
        # Compute quality statistics
        if compute_quality_stats:
            quality_df = self.clustering_quality(clustering, with_stats=True)
            mc_avg["lowest_internal_similarity"] = [
                f"{x:.3} ({y} vs {z})"
                for x, y, z in zip(
                    quality_df["lowest_internal_similarity"],
                    quality_df["lowest_internal_similarity_motif1_name"],
                    quality_df["lowest_internal_similarity_motif2_name"],
                )
            ]
            mc_avg.add_logos(
                np.stack(quality_df["lowest_internal_similarity_motif1_motif"]),
                "lowest_internal_similarity_motif1",
            )
            mc_avg.add_logos(
                np.stack(quality_df["lowest_internal_similarity_motif2_motif"]),
                "lowest_internal_similarity_motif2",
            )
            mc_avg["highest_external_similarity"] = [
                f"{x:.3} ({y}: {z})"
                for x, y, z in zip(
                    quality_df["highest_external_similarity"],
                    quality_df["highest_external_similarity_cluster"],
                    quality_df["highest_external_similarity_motif_name"],
                )
            ]
            cluster_revcomp = {c: i for i, c in enumerate(clusters)}
            mc_avg.add_logos(
                mc_avg.get_standard_motif_stack()[
                    [
                        cluster_revcomp[c]
                        for c in quality_df["highest_external_similarity_cluster"]
                    ]
                ],
                "highest_external_similarity_cluster",
            )
            mc_avg.add_logos(
                np.stack(quality_df["highest_external_similarity_motif_motif"]),
                "highest_external_similarity_motif",
            )
        return mc_avg

    ###########################
    # VIZUALIZATION FUNCTIONS #
    ###########################
    def motif_collection_html(
        self, html_out: str, group_by: str | list, average_motif: bool = True
    ) -> None:
        """Creates an html file displaying all motifs in the current MotifCompendium.

        Produces an html file at the specified location with all motifs from the current
          MotifCompendium. Motifs are grouped by the group_by field in the metadata.
          Group averages are plotted with a green background at the top of each group.

        Args:
            html_out: The path to save the html file.
            group_by: The column in the metadata to group motifs by for visualization.
              Alternatively, an externally provided list can fulfill the same role.
            average_motif: Whether or not to show the average motif per cluster.

        Notes:
            Assumes self.metadata has a "name" column.
            If you just want to plot cluster averages, consider doing
              mc.cluster_averages(group_by).motif_collection_html(html_out, "name").
        """
        # Prepare groups/names/motifs
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
        if "name" not in self.metadata.columns:
            raise KeyError("metadata must have a 'name' column.")
        names = list(self.metadata["name"])
        motifs = (
            utils_motif.motif_8_to_4_signed(self.motifs)
            if self.motifs.shape[2] == 8
            else self.motifs
        )
        # Group motifs
        motif_groups = dict()  # group name --> {motif name --> motif dict}
        group_seeds = dict()  # group name --> index of seed motif in group
        group_xmin_xmax = dict()  # group name --> group name --> (xmin, xmax) for group
        for i, x in enumerate(groups):
            if x in motif_groups:
                cluster_x_seed = group_seeds[x]
                x_rc = self.alignment_rc[cluster_x_seed, i]
                x_h = self.alignment_h[cluster_x_seed, i]
                motif_info_i = utils_plotting.LogoPlottingInput(
                    motifs[i],
                    revcomp=x_rc,
                    pos=x_h,
                    name=names[i],
                )
                group_xmin_xmax[x] = (
                    min(group_xmin_xmax[x][0], x_h),
                    max(
                        group_xmin_xmax[x][1],
                        x_h + motifs.shape[1],
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

    def summary_table_html(
        self, html_out: str, columns: None | list[str] = None, editable=False
    ) -> None:
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
              as columns in the summary table. If None, uses all columns.
            editable: Whether or not the table is editable.

        Notes:
            It is highly suggested that this function just be run on a MotifCompendium
              of motif clusters. Consider doing
              mc.cluster_averages(cluster).summary_table_html(html_out, summary_cols).
        """
        # Check columns
        all_columns = self.metadata.columns.tolist() + self.__images.columns.tolist()
        if columns is None:
            columns = list(self.metadata.columns)
        elif not all(c in all_columns for c in columns):
            missing_columns = [c for c in columns if c not in all_columns]
            raise KeyError(f"{missing_columns} not in metadata or saved images.")
        # If forward and reverse logos aren't in __images, create and add them
        if "logo (fwd)" not in self.get_saved_images():
            self.add_logos(self.get_standard_motif_stack(), "logo (fwd)")
        if "logo (rev)" not in self.get_saved_images():
            self.add_logos(self.get_standard_motif_stack()[:, ::-1, ::-1], "logo (rev)")
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
        utils_visualization.table_html(table_df, image_column, html_out, editable)

    def update_from_summary_table(self, html_loc: str) -> None:
        # Read the summary table HTML
        if not os.path.exists(html_loc):
            raise FileNotFoundError(f"{html_loc} does not exist.")
        with open(html_loc, "r") as f:
            html_content = f.read()
        # Get DataTable
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", {"id": "DataTable"})
        # Get headers (first row)
        headers = []
        image_columns = []
        for i, th in enumerate(table.find_all("th")):
            # If index column, skip
            if i == 0:
                headers.append("index")
                image_columns.append(True)  # Index acts as an image column
                continue
            # Check if column has sort buttons (non-image column)
            if th.find("button"):
                # Non-image columns have buttons that need to be removed
                header_name = th.text.strip().split("\n")[0]
                headers.append(header_name)
                image_columns.append(False)
            else:
                headers.append(th.text.strip())
                image_columns.append(True)
        # Get data rows (ignore first and second row)
        rows = []
        index = []
        for tr in table.find_all("tr")[2:]:  # Skip header and filter rows
            row_data = []
            for i, td in enumerate(tr.find_all("td")):
                if i == 0:
                    # First column is index
                    index.append(td.text.strip())
                elif not image_columns[i]:
                    # Skip image columns
                    val = td.text.strip()
                    try:
                        val = int(val)
                    except:
                        try:
                            val = float(val)
                        except:
                            pass
                    row_data.append(val)
            rows.append(row_data)
        # Create DataFrame
        non_image_headers = [
            h for h, is_img in zip(headers, image_columns) if not is_img
        ]
        index = pd.Index(index, dtype=int)
        df = pd.DataFrame(rows, index=index, columns=non_image_headers)
        df.sort_index(inplace=True)
        # Update metadata
        for col in df.columns:
            self.metadata[col] = df[col]

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
            if "name" not in self.metadata.columns:
                raise KeyError("metadata must have a 'name' column.")
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
    def add_motif_strings(
        self, name="motif_string", specificity=0.7, importance=1 / 30
    ) -> None:
        """Adds a column to the metadata that is a string representation of each motif.

        Adds a "motif_string" column to the metadata that contains the motif as a string.
        """
        if self.motifs.shape[2] == 8:
            motif_str_revstrs = utils_motif.motif_to_string(
                utils_motif.motif_8_to_4_unsigned(self.motifs), specificity, importance
            )
        else:
            motif_str_revstrs = utils_motif.motif_to_string(
                self.motifs, specificity, importance
            )
        self.metadata[name] = [f"{x[0]}<br/>{x[1]}" for x in motif_str_revstrs]

    def extend(self):
        """Add new motifs to the current MotifCompendium."""
        raise NotImplementedError("extend() has not been implemented yet.")

    def assign_label_from_motifs(
        self,
        other_motifs: np.ndarray,
        labels: list[str],
        utf8_images: list[str] | None = None,
        save_column_prefix: str = "match",
        max_submotifs: int = 1,
        min_score: float = 0.0,
    ) -> None:
        """
        Assign labels to motifs based on an external set of labeled motifs.

        Given an external set of motifs with labels, compute the closest matching
          motifs. Composite matching can be enabled by setting max_submotifs > 1. Labels
          are imported from the labeled motifs. utf8 images can also be imported if
          provided.

        Args:
            other_motifs: A np.ndarray motif stack to compare against. (M, L, 4)
            labels: A list of labels for each motif in other_motifs.
            utf8_images: A list of utf8 images for each motif in other_motifs.
            other_is_positive: Whether or not the other_motifs are positive motifs.
            save_column_prefix: The prefix to use for the saved columns.
              Will be saved as f"{save_column_prefix}_{score/name/logo}{i}"
            max_submotifs: The maximum number of submotifs to consider in a match.
            min_score: The minimum similarity score to consider as a match.
        """
        # Check arguments
        utils_motif.validate_motif_stack_similarity(other_motifs)
        if (utf8_images is not None) and not (
            (isinstance(utf8_images, list))
            and len(utf8_images) == other_motifs.shape[0]
        ):
            raise ValueError(
                "utf8_images must be a list of the same length as other_motifs."
            )
        my_motifs = self.motifs.copy()
        # Length dimensions: Resize motifs to match length
        if my_motifs.shape[1] > other_motifs.shape[1]:
            other_pad = self.motifs.shape[1] - other_motifs.shape[1]
            other_motifs = np.pad(other_motifs, ((0, 0), (0, other_pad), (0, 0)))
        elif my_motifs.shape[1] < other_motifs.shape[1]:
            my_pad = other_motifs.shape[1] - my_motifs.shape[1]
            my_motifs = np.pad(my_motifs, ((0, 0), (0, my_pad), (0, 0)))
        # Channel dimensions
        if my_motifs.shape[2] == 8 and other_motifs.shape[2] == 4:
            my_motifs = utils_motif.motif_8_to_4_unsigned(my_motifs)
        elif self.motifs.shape[2] == 4 and other_motifs.shape[2] == 8:
            other_motifs = utils_motif.motif_8_to_4_unsigned(other_motifs)
        # L2 normalize once
        my_motifs /= np.linalg.norm(my_motifs, axis=(1, 2), keepdims=True)
        other_motifs /= np.linalg.norm(other_motifs, axis=(1, 2), keepdims=True)
        # Find best match, per iteration
        match_scores = []
        match_labels = []
        match_motifs = []
        match_idxs = []
        match_mask = np.ones((my_motifs.shape[0],), dtype=bool)  # (N,)
        for i in range(max_submotifs):
            # Compute similarity
            sim, alignment_rc, alignment_h = utils_similarity.compute_similarities(
                [my_motifs, other_motifs], [(0, 1)]
            )[0]

            # Unscale L2 similarity, for dot product only
            sim = sim * (
                np.linalg.norm(my_motifs, axis=(1, 2))[:, np.newaxis]
                * np.linalg.norm(other_motifs, axis=(1, 2))[np.newaxis, :]
            )  # (N, M)
            # Identify matches, Scale score by i
            match_score = np.max(sim, axis=1) * np.sqrt(i + 1)  # (N,)
            match_idx = np.argmax(sim, axis=1)  # (N,)
            alignment_rc = alignment_rc[
                np.arange(alignment_rc.shape[0]), match_idx
            ]  # (N,)
            alignment_h = alignment_h[
                np.arange(alignment_h.shape[0]), match_idx
            ]  # (N,)
            match_motif = other_motifs[match_idx, :, :]
            # Remove matches below threshold
            match_mask = match_mask & (match_score >= min_score)  # (N,)
            match_score[~match_mask] = 0
            match_idx[~match_mask] = -1
            match_motif[~match_mask] = 0
            # Subtract best match
            my_motifs = utils_motif.remove_motif_component(
                my_motifs,
                match_motif,
                alignment_rc,
                alignment_h,
            )
            # Save match information
            match_label = [labels[x] if x >= 0 else "" for x in match_idx]
            if match_motif.shape[2] == 8:
                match_motif = utils_motif.motif_8_to_4_signed(match_motif)
            match_motifs.append(match_motif)
            match_scores.append(match_score)
            match_labels.append(match_label)
            match_idxs.append(match_idx)
        # Save match information
        for i in range(max_submotifs):
            self[f"{save_column_prefix}_score{i}"] = match_scores[i]  # Save scores
            self[f"{save_column_prefix}_name{i}"] = match_labels[i]  # Save labels
            # Save logos, matches only
            self.__images[f"{save_column_prefix}_logo{i}"] = ""
            match_idx = np.where(match_idxs[i] >= 0)[0]
            if utf8_images is None:
                # Generate forward logos if not provided
                match_motif = match_motifs[i][match_idx]
                self.add_logos(match_motif, f"{save_column_prefix}_logo{i}")
            else:
                # Copy forward logos if provided
                self.__images.loc[match_idx, f"{save_column_prefix}_logo{i}"] = [
                    utf8_images[x] for x in match_idx
                ]

    def assign_label_from_other(
        self,
        other: MotifCompendium,
        other_label_column: str = "name",
        save_column_prefix: str = "match",
        max_submotifs: int = 1,
        min_score: float = 0.0,
    ) -> None:
        """Assign clusters to motifs based on an already clustered MotifCompendium.

        Given another MotifCompendium that has already been clustered and assigned
          labels, compute similarity between all the motifs in this MotifCompendium
          and other, for max_submotif iterations. The highest similarity and closest
          motif match for each iteration will be saved as columns
          {save_column_prefix}_score{i} and {save_column_prefix}_name{i}.

        Args:
            other: The other MotifCompendium to compare against.
            other_label_column: The column in the other MotifCompendium to use as labels.
            save_column_prefix: The name of the column in the metadata to save matches to.
              Will be saved as f"{save_col_sim}_{score/name/logo}{i}".
            max_submotifs: The maximum number of submotifs to consider in a match.
            min_score: The minimum similarity score to consider as a match.

        Notes:
            The other MotifCompendium must have motifs length less than or equal to this
              MotifCompendium.
        """
        # Check if other_col_match exists in other MotifCompendium
        if other_label_column in other.metadata.columns:
            other_labels = other.metadata[other_label_column].tolist()
        else:
            raise KeyError(f"{other_label_column} not in other metadata.")
        # Check if forward logos in other MotifCompendium
        if "logo (fwd)" in other.get_saved_images():
            other_logos = other.get_images("logo (fwd)")
        else:
            other_logos = None
        # Assign labels
        self.assign_label_from_motifs(
            other_motifs=other.motifs,
            labels=other_labels,
            utf8_images=other_logos,
            save_column_prefix=save_column_prefix,
            max_submotifs=max_submotifs,
            min_score=min_score,
        )
