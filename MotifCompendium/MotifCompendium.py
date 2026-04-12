# DESIGNED BY SALIL DESHPANDE

from __future__ import annotations
from collections import defaultdict
import os
import warnings

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
    ic_scale: bool | None = None,
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
    if ic_scale is not None:
        utils_config.set_ic_scale(ic_scale)
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
    utils_motif.validate_motif_stack_standard(motifs)
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
    score_type: str = "contrib_scores",
    load_subpatterns: bool = False,
    modisco_region_width: int = 400,
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of Modisco outputs.

    Loads motifs and metadata from all Modisco outputs then passes them to build().

    Args:
        modisco_dict: A dictionary from model name to Modisco file path.
        load_subpatterns: Whether or not to load subpatterns from the Modisco file. If
          True, motifs will be loaded at the subpattern level. If False, motifs will be
          loaded at the pattern level.
        modisco_region_width: The region width used during Modisco. This argument only
          needs to be specified if using a non-standard region width.
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
    (   motifs,
        motif_names,
        seqlet_counts,
        model_names,
        posnegs,
        avgdist_summits,
    ) = utils_loader.load_modiscos(
        modisco_dict,
        score_type=score_type,
        load_subpatterns=load_subpatterns,
        modisco_region_width=modisco_region_width,
    )
    # Build metadata
    metadata = pd.DataFrame(
        {
            "name": motif_names,
            "num_seqlets": seqlet_counts,
            "model": model_names,
            "posneg": posnegs,
            "avg_dist_from_summit": avgdist_summits,
        }
    )
    # Construct object
    return build(
        motifs,
        metadata=metadata,
        safe=safe,
    )


def build_from_pfm(
    pfm_dict: dict[str, str],
    safe: bool = True,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of PFM files.

    Loads motifs and names from all PFM files then passes them to build().

    Args:
        pfm_dict: A dictionary mapping motif set name to PFM file path.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in all PFM files.

    Notes:
        Only accepts files in the PFM or MEME file formats.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    motifs, motif_names, file_names = utils_loader.load_pfms(pfm_dict)
    posneg = utils_motif.motif_posneg_sum(motifs)
    # Build metadata
    metadata = pd.DataFrame(
        {
            "name": motif_names,
            "posneg": posneg,
            "source": file_names,
        }
    )
    return build(
        motifs=motifs,
        metadata=metadata,
        safe=safe,
    )


def combine(
    compendiums: list[MotifCompendium],
    compendium_names: list[str] | None = None,
    safe: bool = True,
) -> MotifCompendium:
    """Combines multiple MotifCompendium into one MotifCompendium.

    Computes similarity between each MotifCompendium's motifs to construct a single
      large similarity matrix. The metadata and images from each MotifCompendium are
      combined, as well, but an error will be thrown if the metadata and images of each
      MotifCompendium do not have the same columns. Then, the motifs, similarity and
      alignment matrices, and metadata are passed to the MotifCompendium constructor.

    Args:
        compendiums: A list of MotifCompendium objects to combine.
        compendium_names: A list of names for each MotifCompendium. These names will be
          added as metadata to the combined MotifCompendium object in the
          'source_compendium' field.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs from all individual
          MotifCompendium.

    Notes:
        The 'source_compendium' column may not exist in any of the MotifCompendium
          objects passed into this function.
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
    if compendium_names is not None:
        if not (
            isinstance(compendium_names, list)
            and all([isinstance(x, str) for x in compendium_names])
        ):
            raise TypeError("compendium_names must be a list of strings.")
        if len(compendium_names) != len(compendiums):
            raise ValueError(
                "compendium_names must have the same length as compendiums."
            )
    # else:
    #     compendium_names = [f"compendium_{i}" for i in range(len(compendiums))]
    # Confirm that the motifs from each MotifCompendium are compatible
    # motifs_length = compendiums[0].motifs.shape[1]
    # if not all([x.motifs.shape[1] == motifs_length for x in compendiums]):
    #     raise ValueError(
    #         "The motifs of the MotifCompendium objects must have the same length."
    #     )
    motifs_channels = compendiums[0].motifs.shape[2]
    if not all([x.motifs.shape[2] == motifs_channels for x in compendiums]):
        raise ValueError(
            "The motifs of the MotifCompendium objects must have the same channels."
        )
    # Confirm that the metadata and __images of each MotifCompendium has the same columns
    metadata_columns = set(compendiums[0].columns())
    if not all(set(x.columns()) == metadata_columns for x in compendiums):
        raise ValueError(
            "The metadata of each MotifCompendium must have the same columns."
            + "\n(Check mc.columns() for each MotifCompendium.)"
        )
    image_columns = set(compendiums[0].images())
    if not all([set(x.images()) == image_columns for x in compendiums]):
        raise ValueError(
            "Each MotifCompendium must have the same saved images."
            + "\n(Check mc.images() for each MotifCompendium.)"
        )
    n = len(compendiums)
    # Prepare motifs
    motif_lengths = [mc.motifs.shape[1] for mc in compendiums]
    max_length = max(motif_lengths)
    motifs_list = [utils_motif.pad_motif(mc.motifs, max_length) for mc in compendiums]
    # Combine similarities
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
                alignment_h_delta = (
                    compendiums[i].alignment_rc * -(max_length - motif_lengths[i])
                ).astype(
                    compendiums[i].alignment_h.dtype
                )  # To account for motif resizing
                alignment_h_block[i][j] = compendiums[i].alignment_h + alignment_h_delta
            elif i < j:
                (
                    similarity_block[i][j],
                    alignment_rc_block[i][j],
                    alignment_h_block[i][j],
                ) = similarity_results[calculations.index((i, j))]
            elif i > j:
                similarity_block[i][j] = similarity_block[j][i].T
                alignment_rc_block[i][j] = alignment_rc_block[j][i].T
                alignment_h_block[i][j] = np.where(
                    alignment_rc_block[j][i].T == 0,
                    -alignment_h_block[j][i].T,
                    alignment_h_block[j][i].T,
                )
    similarity = np.block(similarity_block)
    alignment_rc = np.block(alignment_rc_block)
    alignment_h = np.block(alignment_h_block)
    # Motifs
    motifs = np.concatenate(motifs_list, axis=0)
    # Metadata
    metadata = pd.concat([mc.metadata for mc in compendiums], ignore_index=True)
    source_compendium = []
    if compendium_names is not None:
        for i, mc in enumerate(compendiums):
            source_compendium.extend([compendium_names[i]] * len(mc))
        metadata["source_compendium"] = source_compendium
    # Images
    __images = pd.DataFrame(index=metadata.index)
    for images in compendiums[0].images():
        __images[images] = pd.concat(
            [pd.Series(mc.get_images(images)) for mc in compendiums], ignore_index=True
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
          alignment_rc[i] gives a vector of how motifs should be reverse complemented to
          align with motif i.
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
            f.create_dataset("similarity", data=self.similarity.astype(np.single))
            f.create_dataset("alignment_rc", data=self.alignment_rc.astype(np.bool_))
            f.create_dataset("alignment_h", data=self.alignment_h.astype(np.short))
        self.metadata.to_hdf(save_loc, key="metadata", mode="a")
        self.__images.to_hdf(save_loc, key="__images", mode="a")

    def validate(self) -> None:
        """Verifies the integrity of the MotifCompendium.

        Checks the validity of each attribute (motifs, similarity, similarity_fb,
          similarity_h).

        Notes:
            This function can take a long time to run, especially for very large
              MotifCompendium.
        """
        # motifs
        utils_motif.validate_motif_stack_standard(self.motifs)
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
        if len(self.alignment_rc.shape) != 2:
            raise ValueError("self.alignment_rc must be a square matrix.")
        if not ((self.alignment_rc == 0) | (self.alignment_rc == 1)).all():
            raise ValueError("self.alignment_rc must have values being either 0 or 1.")
        if not (self.alignment_rc == self.alignment_rc.T).all():
            warnings.warn(
                "self.alignment_rc is not symmetric. This may be due to numerical instability (especially if you have very symmetric motifs).",
                RuntimeWarning,
            )
        # alignment_h
        if not (
            isinstance(self.alignment_h, np.ndarray)
            and self.alignment_h.dtype == np.short
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
            warnings.warn(
                "self.alignment_h should be symmetric for reverse complement motifs and skew-symmetric for motifs that are already aligned, but it is not. This is very rare and should not occur (unless you are working with PFMs, for which it happens often).",
                RuntimeWarning,
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
                    (sorted(self.images()) == sorted(other.images()))
                    and all(
                        [
                            self.get_images(x) == other.get_images(x)
                            for x in self.images()
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

    def delete_columns(self, column: str | list[str]) -> None:
        """Deletes the specified column(s) from the metadata."""
        if not (
            isinstance(column, str)
            or (isinstance(column, list) and all([isinstance(x, str) for x in column]))
        ):
            raise TypeError("column must be a string or a list of strings.")
        self.metadata.drop(column, axis=1, inplace=True)

    # IMAGES
    def images(self) -> list[str]:
        """Returns a list of the images saved in the MotifCompendium."""
        return list(self.__images.columns)

    def get_images(self, image_name: str) -> list[str]:
        """Returns a list of saved images as utf8 str in the MotifCompendium by column name."""
        if image_name not in self.__images.columns:
            raise KeyError(f"{image_name} is not a saved image.")
        return self.__images[image_name].tolist()

    def add_logos(
        self,
        motifs: np.ndarray,
        image_name: str,
        trim: bool | float | int = False,
        length: int | None = None,
    ) -> None:
        """Saves logos of the provided motifs as saved images.

        Args:
            motifs: The motifs to save logos for. Of shape (N, L, 4).
            image_name: The name of the images to save the logos as.
            trim: A bool or float/int indicating how the motif should be trimmed when
              plotting. If False, the motif will not be trimmed at all. If True, the
              motif will be trimmed at the flanks with a standard threshold of 1/L. If a
              number is provided, that number must be in [0, 1], and will define the
              trimming threshold. At a value of 0, only zero positions are trimmed and
              at a value of 1, all positions would be trimmed.
            length: An int indicating the length of the motif to be trimmed to, with the
              maximum importance. If None, no trimming is done.
        """
        # Check inputs
        utils_motif.validate_motif_stack_standard(motifs)
        if not (motifs.shape[0] == len(self)):
            raise ValueError(
                "The number of motifs must be the same as the number of motifs in the MotifCompendium."
            )
        if not isinstance(image_name, str):
            raise TypeError("image_name must be a string.")
        if image_name in self.columns():
            raise ValueError(
                f"{image_name} is already a metadata column. Names may not overlap."
            )
        # Prepare plotting
        logo_plotting_inputs = [
            utils_plotting.LogoPlottingInput(
                motif=m, 
                trim=trim,
                length=length,
            ) for m in motifs
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
        if not all([x in self.images() for x in mapper.keys()]):
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
    def __getitem__(self, key: str | pd.Series | slice | int | list[int]) -> pd.Series | MotifCompendium:
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
        elif isinstance(key, pd.Series) and pd.api.types.is_bool_dtype(key):
            keep_idxs = list(self.metadata[key].index)
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            if len(key) != len(self.metadata):
                raise ValueError("Boolean mask must be same length as metadata.")
            keep_idxs = list(np.where(key)[0])
        elif isinstance(key, list) and all(isinstance(x, bool) for x in key):
            if key.ndim != 1 or len(key) != len(self.metadata):
                raise ValueError("Boolean mask must be 1D and same length as metadata.")
            keep_idxs = [i for i, x in enumerate(key) if x]
        elif isinstance(key, slice):
            if len(key) != len(self.metadata):
                raise ValueError("Boolean mask must be same length as metadata.")
            keep_idxs = list(range(*key.indices(len(self.metadata))))
        elif isinstance(key, int):
            keep_idxs = [key]
        elif isinstance(key, list) and all(isinstance(i, int) for i in key):
            keep_idxs = key
        elif len(key) == 0:
            keep_idxs = []
        else:
            raise TypeError("MotifCompendium cannot be indexed by this.")

        # metadata
        metadata_slice = self.metadata.iloc[keep_idxs].reset_index(drop=True)
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
            if key in self.__images.columns:
                raise ValueError(
                    f"{key} is already a saved image. Names may not overlap."
                )
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
            ascending: Whether or not to sort in ascending order. Must have one per
              column.
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

    def copy(self) -> MotifCompendium:
        """Creates a copy of the MotifCompendium object."""
        return MotifCompendium(
            self.motifs.copy(),
            self.similarity.copy(),
            self.alignment_rc.copy(),
            self.alignment_h.copy(),
            self.metadata.copy(deep=True),
            self.__images.copy(deep=True),
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
            A pd.DataFrame containing the similarity scores between two subsets of
              motifs.

        Notes:
            If only slice1 is provided then the similarity scores between the motifs
              specified by slice1 and all other motifs are returned.
            If with_names is provided then it is assumed that the metadata has a "name"
              column.
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
        similarity_threshold: float | int = 0.9,
        save_name: str = "cluster",
        cluster_within: str | None = None,
        cluster_on: str | None = None,
        cluster_within_on: tuple[str, str] | None = None,
        cluster_on_weight: str | None = None,
        init_clustering_col: str | None = None,
        weight_col: str | None = None,
        largest_clusters_first: bool = True,
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
            cluster_within: The name of the column in metadata containing cluster
              annotations to perform clustering within. If not None, the clustering will
              be done per cluster in the cluster_within column. This will identify
              subclusters of the cluster_within column. It is guaranteed that if
              self.metadata.loc[i, cluster_within] != self.metadata.loc[j,
              cluster_within], then self.metadata.loc[i, save_name] !=
              self.metadata.loc[j, save_name].
            cluster_on: The name of the column in metadata containing cluster
              annotations to group motifs by. If not None, the clustering will be done
              on the average motifs of the clusters in this column. This will identify
              superclusters of the cluster_on column. It is guaranteed that if
              self.metadata.loc[i, cluster_on] == self.metadata.loc[j, cluster_on],
              then self.metadata.loc[i, save_name] == self.metadata.loc[j, save_name].
            cluster_within_on: A tuple of two strings both of which are columns in the
              metdata. The tuple, (cluster_within, cluster_on) will cluster within the
              cluster_within column and cluster on top of the cluster_on column.
            cluster_on_weight: The name of the column in metadata containing weights to
              use when averaging motifs while doing a cluster_on. If None, all motifs
              are equally weighted. cluster_on_weight can only be set when cluster_on or
              cluster_within_on is set.
            init_clustering_col: The name of the column in metadata containing initial cluster
              annotations to use as the initial clusters for algorithms that support
              initialization. If None, no initialization is used. (Cannot be used with 
              cluster_on or cluster_within_on as the initializations are averaged immediately;
              Only used if algorithm supports weighting. See 
              MotifCompendium.utils.clustering.cluster() for details.)
            weight_col: The name of the column in metadata containing weights to
              use when averaging motifs while doing a cluster_on. If None, all motifs
              are equally weighted. weight_col can only be set when cluster_on or
              cluster_within_on is set. (Only used if algorithm supports weighting. 
              See MotifCompendium.utils.clustering.cluster() for details.)
            largest_clusters_first: Whether or not the first clusters (0, 1, 2, ...)
              should be the largest clusters. If True, cluster 0 will be the largest
              cluster. If False, the cluster order will not relate to cluster size.
            **kwargs: Additional named arguments specific to the clustering algorithm of
                choice.

        Notes:
            Review MotifCompendium.utils.clustering.cluster() for available clustering
              algorithms and algorithm-specific arguments.
            Only one of cluster_within, cluster_on, or cluster_within_on can be used at
              once.
        """
        # Check arguments
        if not (
            isinstance(similarity_threshold, (float, int)) and (0 <= similarity_threshold <= 1)
        ):
            raise ValueError("similarity_threshold must be a float or int between [0, 1].")
        if weight_col and cluster_on_weight:
            cluster_on_weight = weight_col
            warnings.warn(
                "Both weight_col and cluster_on_weight are set. weight_col will be used and cluster_on_weight will be ignored. " \
                "Note that cluster_on_weight will be deprecated in favor of weight_col in the future.",
                FutureWarning,
            )
        elif cluster_on_weight:
            weight_col = cluster_on_weight
            warnings.warn(
                "cluster_on_weight is set. Note that cluster_on_weight will be deprecated in favor of weight_col in the future.",
                FutureWarning,
            )
        # Cluster within
        if (
            (cluster_within is not None)
            and (cluster_on is None)
            and (cluster_within_on is None)
        ):
            # Check inputs
            if not cluster_within in self.metadata.columns:
                raise KeyError(f"cluster_within {cluster_within} not in metadata.")
            if (weight_col is not None) and (
                weight_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"weight_col {weight_col} not in metadata."
                )
            if (init_clustering_col is not None) and (
                init_clustering_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"init_clustering_col {init_clustering_col} not in metadata."
                )
            # if cluster_on_weight is not None:
            #     raise ValueError(
            #         "cluster_on_weight can only be set when cluster_on or cluster_within_on is set."
            #     )
            # Prepare clustering
            clusters = pd.Series(-1, index=self.metadata.index)  # -1 for int
            num_clusters_so_far = 0
            # Cluster within each cluster_within cluster
            for c in set(self.metadata[cluster_within]):
                c_condition = self.metadata[cluster_within] == c
                c_idxs = list(c_condition[c_condition].index)
                # c_similarity = self.similarity[c_idxs, :][:, c_idxs]
                c_clusters = utils_clustering.cluster(
                    motifs=self[c_condition].motifs,
                    similarity_matrix=self[c_condition].similarity,
                    alignment_rc_matrix=self[c_condition].alignment_rc,
                    alignment_h_matrix=self[c_condition].alignment_h,
                    init_membership=self[c_condition].metadata[init_clustering_col].to_numpy() if init_clustering_col else None,
                    weights=self[c_condition].metadata[weight_col].to_numpy() if weight_col else None,
                    similarity_threshold=similarity_threshold,
                    algorithm=algorithm,
                    **kwargs,
                )
                c_clusters = [c + num_clusters_so_far for c in c_clusters]
                clusters[c_idxs] = c_clusters
                num_clusters_so_far += len(set(c_clusters))
            self.metadata[save_name] = clusters
        # Cluster on
        elif (
            (cluster_within is None)
            and (cluster_on is not None)
            and (cluster_within_on is None)
        ):
            # Check inputs
            if not cluster_on in self.metadata.columns:
                raise KeyError(f"cluster_on {cluster_on} not in metadata.")
            if (weight_col is not None) and (
                weight_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"weight_col {weight_col} not in metadata."
                )
            if init_clustering_col:
                raise ValueError(
                    "init_clustering_col cannot be used when cluster_on is set as the initializations are averaged immediately."
                )
            # Average
            mc_average = self.cluster_averages(
                clustering=cluster_on,
                weight_col=weight_col,
                aggregations=[
                        (weight_col, "sum", weight_col) 
                    ] if weight_col else [],
            )
            # Cluster
            mc_average.metadata["cluster"] = utils_clustering.cluster(
                motifs=mc_average.motifs,
                similarity_matrix=mc_average.similarity,
                alignment_rc_matrix=mc_average.alignment_rc,
                alignment_h_matrix=mc_average.alignment_h,
                weights=mc_average.metadata[weight_col].to_numpy() if weight_col else None,
                similarity_threshold=similarity_threshold,
                algorithm=algorithm,
                **kwargs,
            )
            # Map back to self
            cluster_map = {
                row["source_cluster"]: row["cluster"]
                for _, row in mc_average.metadata.iterrows()
            }
            self.metadata[save_name] = [
                cluster_map[c] for c in self.metadata[cluster_on]
            ]
        # Cluster within+on
        elif (
            (cluster_within is None)
            and (cluster_on is None)
            and (cluster_within_on is not None)
        ):
            # Check inputs
            if not (
                (type(cluster_within_on) == tuple)
                and (len(cluster_within_on) == 2)
                and (type(cluster_within_on[0]) == str)
                and (type(cluster_within_on[1]) == str)
            ):
                raise TypeError("cluster_within_on must be a tuple of two strings.")
            if not cluster_within_on[0] in self.metadata.columns:
                raise KeyError(
                    f"cluster_within_on-->cluster_within {cluster_within_on[0]} not in metadata."
                )
            if not cluster_within_on[1] in self.metadata.columns:
                raise KeyError(
                    f"cluster_within_on-->cluster_on {cluster_within_on[1]} not in metadata."
                )
            if (weight_col is not None) and (
                weight_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"weight_col {weight_col} not in metadata."
                )
            if init_clustering_col:
                raise ValueError(
                    "init_clustering_col cannot be used when cluster_on is set as the initializations are averaged immediately."
                )
            # Prepare clustering
            clusters = pd.Series(-1, index=self.metadata.index)  # -1 for int
            num_clusters_so_far = 0
            cluster_within, cluster_on = cluster_within_on
            # Cluster within each cluster_within cluster
            for c in set(self.metadata[cluster_within]):
                # Identify motifs corresponding to cluster_within
                c_condition = self.metadata[cluster_within] == c
                c_idxs = list(c_condition[c_condition].index)
                c_mc = self[c_condition]
                # Average for cluster_on
                c_mc_average = c_mc.cluster_averages(
                    clustering=cluster_on,
                    weight_col=weight_col,
                    aggregations=[
                        (weight_col, "sum", weight_col) 
                    ] if weight_col else [],
                )
                # Cluster
                c_mc_average.metadata["cluster"] = utils_clustering.cluster(
                    motifs=c_mc_average.motifs,
                    similarity_matrix=c_mc_average.similarity,
                    alignment_rc_matrix=c_mc_average.alignment_rc,
                    alignment_h_matrix=c_mc_average.alignment_h,
                    weights=c_mc_average.metadata[weight_col].to_numpy() if weight_col else None,
                    similarity_threshold=similarity_threshold,
                    algorithm=algorithm,
                    **kwargs,
                )
                # Map back to c_mc
                c_cluster_map = {
                    row["source_cluster"]: row["cluster"]
                    for _, row in c_mc_average.metadata.iterrows()
                }
                c_clusters = [c_cluster_map[c] for c in c_mc[cluster_on]]
                c_clusters = [c + num_clusters_so_far for c in c_clusters]
                num_clusters_so_far += len(set(c_clusters))
                # Map back to self
                clusters[c_idxs] = c_clusters
            self.metadata[save_name] = clusters
        # If all are None, cluster on entire MotifCompendium
        elif (
            (cluster_within is None)
            and (cluster_on is None)
            and (cluster_within_on is None)
        ):
            # Check inputs
            if (weight_col is not None) and (
                weight_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"weight_col {weight_col} not in metadata."
                )
            if (init_clustering_col is not None) and (
                init_clustering_col not in self.metadata.columns
            ):
                raise KeyError(
                    f"init_clustering_col {init_clustering_col} not in metadata."
                )
            # Cluster
            self.metadata[save_name] = utils_clustering.cluster(
                motifs=self.motifs,
                similarity_matrix=self.similarity,
                alignment_rc_matrix=self.alignment_rc,
                alignment_h_matrix=self.alignment_h,
                init_membership=self.metadata[init_clustering_col].to_numpy() if init_clustering_col else None,
                weights=self.metadata[weight_col].to_numpy() if weight_col else None,
                similarity_threshold=similarity_threshold,
                algorithm=algorithm,
                **kwargs,
            )
        # Otherwise, throw error
        else:
            raise ValueError(
                "Only one of cluster_within, cluster_on, or cluster_within_on can be used at once."
            )
        # Sort clusters by number of constituents
        if largest_clusters_first:
            sorted_clusters = self.metadata[save_name].value_counts().index.tolist()
            cluster_map = {
                old_cluster: new_cluster
                for new_cluster, old_cluster in enumerate(sorted_clusters)
            }
            self.metadata[save_name] = self.metadata[save_name].map(cluster_map)

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
                - "lowest_internal_similarity_motif1_logo": The motif of the first
                  motif contributing to the lowest internal similarity.
                - "lowest_internal_similarity_motif2_name": The name of the second motif
                  contributing to the lowest internal similarity.
                - "lowest_internal_similarity_motif2_logo": The logo of the second
                  motif contributing to the lowest internal similarity.
                - "highest_external_similarity": The highest external similarity.
                - "highest_external_similarity_cluster": The cluster within which the
                  highest external similarity motif is found in.
                - "highest_external_similarity_motif_name": The name of the motif in the
                  external cluster that is driving high external similarity.
                - "highest_external_similarity_motif_logo": The logo of the motif in
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
                    "lowest_internal_similarity_motif1_logo",
                    "lowest_internal_similarity_motif2_name",
                    "lowest_internal_similarity_motif2_logo",
                    "lowest_constituent_cluster_similarity",
                    "lowest_constituent_cluster_similarity_motif_name",
                    "lowest_constituent_cluster_similarity_logo",
                    "highest_external_similarity",
                    "highest_external_similarity_cluster",
                    "highest_external_similarity_motif_name",
                    "highest_external_similarity_motif_logo",
                ],
            )
            motif_names = list(self.metadata["name"])
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
                    )  # culprits of lowest internal similarity
                    quality[i, i] = similarity_slice_ij[min_idxs]
                    if with_stats:
                        culprit_1_idx = c1_idxs[min_idxs[0]]
                        culprit_1_rc = self.alignment_rc[
                            c1_idxs[0], culprit_1_idx
                        ]  # 0 b/c cluster aligned to first motif in cluster
                        culprit_2_idx = c1_idxs[min_idxs[1]]
                        culprit_2_rc = self.alignment_rc[
                            c1_idxs[0], culprit_2_idx
                        ]  # 0 b/c cluster aligned to first motif in cluster
                        stats.loc[c1, "lowest_internal_similarity"] = quality[i, i]
                        stats.loc[c1, "lowest_internal_similarity_motif1_name"] = (
                            motif_names[culprit_1_idx]
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif1_logo"] = (
                            motifs_standard[culprit_1_idx]
                            if not culprit_1_rc
                            else utils_motif.reverse_complement(
                                motifs_standard[culprit_1_idx]
                            )
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif2_name"] = (
                            motif_names[culprit_2_idx]
                        )
                        stats.loc[c1, "lowest_internal_similarity_motif2_logo"] = (
                            motifs_standard[culprit_2_idx]
                            if not culprit_2_rc
                            else utils_motif.reverse_complement(
                                motifs_standard[culprit_2_idx]
                            )
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
                        culprit_c1_idx = c1_idxs[
                            max_idxs[0]
                        ]  # culprit in c1 that is causing highest external similarity
                        external_similarity_culprit_idxs[j, i] = culprit_c1_idx
                        culprit_c2_idx = c2_idxs[
                            max_idxs[1]
                        ]  # culprit in c2 that is causing highest external similarity
                        external_similarity_culprit_idxs[i, j] = culprit_c2_idx
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
            stats["highest_external_similarity_motif_logo"] = [
                (
                    motifs_standard[y]
                    if ci_idxs[x] and not self.alignment_rc[ci_idxs[x][0], y]
                    else utils_motif.reverse_complement(motifs_standard[y])
                )
                for x, y in zip(
                    highest_external_sim_idx, highest_external_similarity_motif_idxs
                )
            ]  # align with respect to source cluster
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
                    - "average" or "avg" or "mean": This option averages the values in
                        the source column for each cluster.
                    - "concatenate" or "concat": This option lists all the unique values
                        in the source column for each cluster.
                    - "concat_counted": This option lists all the unique values and
                        their number of occurrences in the source column for each
                        cluster.
                save: The name of the column in the new metadata to save the aggregated
                  data to.
            weight_col: The name of the metadata column to be used to weight motifs when
              computing motif averages. The data in the weight_col should be numeric.
            compute_quality_stats: Whether or not to compute quality statistics for the
              clustering. If True, the quality statistics will be saved in the metadata
              of the returned MotifCompendium.
                - "best_match_similarity": The similarity of the cluster average motif to
                  its most similar cluster average motif (excluding itself). 
                - "best_match_cluster": The cluster of the most similar cluster average motif.
                - "best_match_cluster_logo": The logo of the most similar cluster average motif.
                - "lowest_constituent_cluster_similarity": The similarity of the cluster to 
                  its least similar constituent motif. ("radius")
                - "lowest_constituent_cluster_similarity_logo": The logo of the least similar
                  constituent motif.
                - "lowest_internal_similarity": The lowest similarity between all consituents
                  in the cluster. ("diameter")
                - "lowest_internal_similarity_motif1": Logo of one of the motif pair contributing 
                  to the lowest internal similarity.
                - "lowest_internal_similarity_motif2": Logo of the other motif pair contributing
                  to the lowest internal similarity.
                - "highest_external_similarity": The highest similarity between motifs in the 
                  cluster and motifs outside of the cluster.
                - "highest_external_similarity_logo": The logo of the motif outside of the cluster 
                  that has the highest external similarity.

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
            if weight_col not in self.metadata.columns:
                raise KeyError(f"{weight_col} not in metadata.")
            if not pd.api.types.is_numeric_dtype(self.metadata[weight_col]):
                raise TypeError(f"{weight_col} must be numeric.")
            weights = self.metadata[weight_col].to_numpy()
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
                agg_c_data = agg_c_data.dropna()  # Remove NaNs
                match agg_dict["method"]:
                    case "count":
                        agg_dict["values"].append(len(agg_c_data))
                    case "unique":
                        agg_dict["values"].append(len(set(agg_c_data)))
                    case "sum":
                        agg_dict["values"].append(np.sum(agg_c_data))
                    case "average" | "avg" | "mean":
                            agg_dict["values"].append(np.average(agg_c_data, weights=weights_c))
                    case "concatenate" | "concat":
                        agg_dict["values"].append(
                            ",".join(sorted(set(x.strip()
                                for val in agg_c_data
                                for x in str(val).split(",")
                                if x.strip())))
                            )
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
            # Most similar cluster
            best_match_idx = np.argmax(
                mc_avg.similarity - np.diag(np.diag(mc_avg.similarity)), axis=1
            )
            mc_avg["best_match_similarity"] = [
                f"{mc_avg.similarity[i, idx]:.3} ({mc_avg.metadata['name'][idx]})"
                for i, idx in enumerate(best_match_idx)
            ]
            mc_avg_standard_motifs = mc_avg.get_standard_motif_stack()
            mc_avg.add_logos(
                np.stack(
                    [
                        (
                            mc_avg_standard_motifs[x]
                            if not mc_avg.alignment_rc[i, x]
                            else utils_motif.reverse_complement(
                                mc_avg_standard_motifs[x]
                            )
                        )
                        for i, x in enumerate(best_match_idx)
                    ]
                ),
                "best_match_cluster",
                0,
            )
            # Best constituent-cluster match
            cluster_revcomp = {c: i for i, c in enumerate(clusters)}
            mc_mc_avg_similarity, _, _ = utils_similarity.compute_similarities(
                [self.motifs, mc_avg.motifs], [(0, 1)])[0]
            mc_mc_avg_membership = (
                self.metadata[clustering].to_numpy()[:, None] == mc_avg.metadata["source_cluster"].to_numpy()[None, :]
            )  # Membership matrix: True if motif i belongs to cluster j
            mc_mc_avg_similarity_masked = np.where(mc_mc_avg_membership, mc_mc_avg_similarity, np.inf)
            mc_mc_avg_similarity_min = np.min(mc_mc_avg_similarity_masked, axis=0)
            mc_mc_avg_similarity_idx = np.argmin(mc_mc_avg_similarity_masked, axis=0)
            mc_avg["lowest_constituent_cluster_similarity"] = [
                f"{x:.3} ({y})"
                for x, y in zip(
                    mc_mc_avg_similarity_min,
                    self.metadata["name"][mc_mc_avg_similarity_idx],
                )
            ]
            mc_avg.add_logos(
                np.stack([(
                        self.get_standard_motif_stack()[idx]
                        if not self.alignment_rc[idx, cluster_revcomp[c]]
                        else utils_motif.reverse_complement(
                            self.get_standard_motif_stack()[idx]
                        )
                    ) for idx, c in zip(mc_mc_avg_similarity_idx, clusters)
                ]),
                "lowest_constituent_cluster_similarity_logo",
                0,
            )
            # Actual quality
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
                np.stack(quality_df["lowest_internal_similarity_motif1_logo"]),
                "lowest_internal_similarity_motif1",
                0,
            )
            mc_avg.add_logos(
                np.stack(quality_df["lowest_internal_similarity_motif2_logo"]),
                "lowest_internal_similarity_motif2",
                0,
            )
            mc_avg["highest_external_similarity"] = [
                f"{x:.3} ({y}: {z})"
                for x, y, z in zip(
                    quality_df["highest_external_similarity"],
                    quality_df["highest_external_similarity_cluster"],
                    quality_df["highest_external_similarity_motif_name"],
                )
            ]
            mc_avg_standard_motifs = mc_avg.get_standard_motif_stack()
            external_cluster_rc = [
                mc_avg.alignment_rc[i, cluster_revcomp[c]]
                for i, c in enumerate(quality_df["highest_external_similarity_cluster"])
            ]
            mc_avg.add_logos(
                np.stack(
                    [
                        (
                            mc_avg_standard_motifs[cluster_revcomp[x]]
                            if not y
                            else utils_motif.reverse_complement(
                                mc_avg_standard_motifs[cluster_revcomp[x]]
                            )
                        )
                        for x, y in zip(
                            quality_df["highest_external_similarity_cluster"],
                            external_cluster_rc,
                        )
                    ]
                ),
                "highest_external_similarity_cluster",
                0,
            )
            mc_avg.add_logos(
                np.stack(
                    [
                        x if not y else utils_motif.reverse_complement(x)
                        for x, y in zip(
                            quality_df["highest_external_similarity_motif_logo"],
                            external_cluster_rc,
                        )
                    ]
                ),
                "highest_external_similarity_logo",
                0,
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
        motif_groups = (
            dict()
        )  # group name --> {motif name --> list of LogoPlottingInput}
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
        self,
        html_out: str,
        columns: None | list[str] = None,
        logo_trimming: bool | float | int = True,
        editable=False,
    ) -> None:
        """Creates an html file summarizing all motifs and metadata about them.

        Produces an html file at the specified location with all motifs from the current
          MotifCompendium. Each motif has one row in the summary table. Logos for the
          forward and reverse complement of each motif will be displayed in the first
          two columns. Columns from the current metadata as well as other images saved
          in the object (which can be viewed with MotifCompendium.images())
          can be displayed as columns in the summary table.

        Args:
            html_out: The path to save the html file.
            columns: The list of column names in the metadata or saved images to display
              as columns in the summary table. If None, uses all columns.
            logo_trimming: This argument is only relevant if save_images is True. A bool
              or float/int indicating how the motif should be trimmed when plotting. If
              False, the motif will not be trimmed at all. If True, the motif will be
              trimmed at the flanks with a standard threshold of 1/L. If a number is
              provided, that number must be in [0, 1], and will define the trimming
              threshold. At a value of 0, only zero positions are trimmed and at a value
              of 1, all positions would be trimmed.
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
        if "logo (fwd)" not in self.images():
            self.add_logos(
                self.get_standard_motif_stack(),
                "logo (fwd)",
                logo_trimming,
            )
        if "logo (rev)" not in self.images():
            self.add_logos(
                utils_motif.reverse_complement(self.get_standard_motif_stack()),
                "logo (rev)",
                logo_trimming,
            )
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
        """Read edited metadata back from a summary table HTML file.

        Parses the JSON data payload embedded in the HTML file (between the
        // DATA_START and // DATA_END markers) and updates self.metadata with
        any changes found in non-image columns.

        To capture in-browser edits, use the "Save Edits" button in the table
        (only visible when editable=True), which downloads a new HTML file with
        the modified data embedded. Pass that downloaded file to this method.

        Args:
            html_loc: Path to the HTML file previously generated by summary_table_html.

        Raises:
            FileNotFoundError: If html_loc does not exist.
            ValueError: If the DATA payload cannot be found (e.g. file from an older
                version of the tool; regenerate via summary_table_html).
        """
        import re

        if not os.path.exists(html_loc):
            raise FileNotFoundError(f"{html_loc} does not exist.")
        with open(html_loc, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Extract the JSON payload between the DATA_START / DATA_END markers
        match = re.search(
            r"//\s*DATA_START\s*\nconst DATA\s*=\s*([\s\S]*?);\s*\n\s*//\s*DATA_END",
            html_content,
        )
        if not match:
            raise ValueError(
                "Could not find the DATA payload in the HTML file. "
                "This file may have been generated by an older version of the tool; "
                "please regenerate it with summary_table_html()."
            )

        data = json.loads(match.group(1))
        columns = data["columns"]        # columns[0] is always 'index'
        rows_data = data["rows"]
        image_column = data["image_column"]

        # Identify text (non-image) columns, excluding the leading 'index' column
        index_col = columns[0]
        text_cols = [
            c
            for c, is_img in zip(columns[1:], image_column[1:])
            if not is_img
        ]

        # Build a DataFrame indexed by the original pandas index values
        index_values = [int(row[index_col]) for row in rows_data]
        records = [{c: row.get(c) for c in text_cols} for row in rows_data]
        df = pd.DataFrame(records, index=pd.Index(index_values, dtype=int))
        df.sort_index(inplace=True)

        # Coerce numeric columns where possible (mirrors original behaviour)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Write back only columns that already exist in metadata
        for col in df.columns:
            if col in self.metadata.columns:
                self.metadata[col] = df[col]

    def heatmap(
        self,
        similarity_threshold: float | None = None,
        annot: bool = False,
        label: bool = False,
        show: bool = False,
        save_loc: str | None = None,
    ) -> None:
        """Creates a heatmap of the similarity matrix.

        Produces a heatmap of the similarity matrix of this MotifCompendium with various
          formatting, display, and save options.

        Args:
            similarity_threshold: The minimum score below which no similarity values are
              shown. If None, all values are shown.
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
        if similarity_threshold is not None:
            if not (
                isinstance(similarity_threshold, float)
                and 0 <= similarity_threshold <= 1
            ):
                raise ValueError(
                    "similarity_threshold must be a float between 0 and 1."
                )
            heatmap_data = self.similarity * (self.similarity >= similarity_threshold)
        else:
            heatmap_data = self.similarity
        if label:
            if "name" not in self.metadata.columns:
                raise KeyError("metadata must have a 'name' column.")
            utils_plotting.plot_heatmap(
                heatmap_data,
                annot=annot,
                labels=list(self.metadata["name"]),
                show=show,
                save_loc=save_loc,
            )
        else:
            utils_plotting.plot_heatmap(
                heatmap_data, annot=annot, show=show, save_loc=save_loc
            )

    ######################
    # ANALYSIS FUNCTIONS #
    ######################
    def add_motif_strings(
        self,
        name: str = "motif_string",
        specificity: float = 0.7,
        importance: float | bool = True,
    ) -> None:
        """Adds a column to the metadata that is a string representation of each motif.

        Adds a column to the metadata that contains a text version of the forward and
          reverse complement of each motif in the MotifCompendium.

        Args:
            name: The name of the column to add to the metadata.
            specificity: The percentage of importance a base must have at a position to
              be included in the string.
            importance: The minimum level of importance a position must have to be
              included in the string.
        """
        unsigned_motifs = np.abs(self.get_standard_motif_stack())
        if importance is True:
            importance = 1 / unsigned_motifs.shape[1]
        motif_str_revstrs = utils_motif.motif_to_string(
            unsigned_motifs, specificity, importance
        )
        self.metadata[name] = [f"{x[0]}<br/>{x[1]}" for x in motif_str_revstrs]

    def symmetricness(self, name: str = "symmetricness") -> None:
        """Adds a column to the metadata that is the symmetricness of each motif.

        The symmetricness of a motif is its similarity with its reverse complement.

        Args:
            name: The name of the column to add to the metadata.
        """
        self.metadata[name] = np.diag(
            utils_similarity.compute_similarities(
                [self.motifs, utils_motif.reverse_complement(self.motifs)], [(0, 1)]
            )[0][0]
        )

    def extend(self):
        """Add new motifs to the current MotifCompendium."""
        raise NotImplementedError("extend() has not been implemented yet.")

    def assign_label_from_motifs(
        self,
        reference_motifs: np.ndarray,
        labels: list[str],
        min_score: float,
        max_submotifs: int = 1,
        label_unsigned: bool = True,
        save_images: bool = True,
        logo_trimming: bool | float | int = True,
        utf8_images: list[str] | None = None,
        save_col_prefix: str = "match",
    ) -> None:
        """
        Assign labels to motifs based on an external set of labeled motifs.

        Given an external reference set of motifs with labels, for each motif in this
          MotifCompendium, find the closest match that appears in the reference set. The
          match score and the label of the best match are saved in the metadata.
          Optionally, the logos of the matched motifs can be saved as images. These
          logos can be computed on the fly or passed in as utf8 images if they have been
          precomputed. Composite matching can be enabled by setting max_submotifs > 1.

        Args:
            reference_motifs: A np.ndarray motif stack of shape (M, L, 4) to compare
              against.
            labels: A list of labels for each motif in reference_motifs.
            min_score: The minimum similarity score to consider a match.
            max_submotifs: The maximum number of submotifs to consider in a match. If
              max_submotifs = 1, only a single match is given to each motif. If
              max_submotifs > 1, the best match for each motif can be from a combination
              of multiple reference motifs.
            label_unsigned: Whether or not to label indifferent of positive and negative
              signs. If True, negative motifs can be labeled by a positive reference motif.
              (i.e., indifferent to sign), by comparing the absolute value of both motifs. 
              If False, negative motifs can only labeled by a negative reference motif
              (i.e., sign matters).
            save_images: Whether or not to save the logos of the matched motifs. If
              True, the logos will appear as a saved image. If False, logos will not be
              saved as saved images.
            logo_trimming: This argument is only relevant if save_images is True. A bool
              or float/int indicating how the motif should be trimmed when plotting. If
              False, the motif will not be trimmed at all. If True, the motif will be
              trimmed at the flanks with a standard threshold of 1/L. If a number is
              provided, that number must be in [0, 1], and will define the trimming
              threshold. At a value of 0, only zero positions are trimmed and at a value
              of 1, all positions would be trimmed.
            utf8_images: A list of utf8 images for each motif in reference_motifs. If
              saved_images is True and utf8_images is None, the logos will be generated
              on the fly using the trimming option logo_trimming. If saved_images is
              True and utf8_images is a list of utf8 images, these images will be used
              as the logos for the matched motifs.
            save_col_prefix: The prefix to use for the saved columns. All saved columns
              and saved images generated from the labeling process will begin with
              save_col_prefix. These columns will have the structure
              f"{save_col_prefix}_{score/name/logo}{i}".
        """
        # Check arguments
        if not save_images and utf8_images is not None:
            raise ValueError("save_images is False but utf8_images is not None.")
        if (utf8_images is not None) and not (
            (isinstance(utf8_images, list))
            and len(utf8_images) == reference_motifs.shape[0]
        ):
            raise ValueError(
                "utf8_images must be a list of the same length as reference_motifs."
            )
        # Store as copys
        my_motifs = self.motifs.copy()
        reference_motifs = reference_motifs.copy()
        # Length dimensions: Resize motifs to match length
        max_length = max(my_motifs.shape[1], reference_motifs.shape[1])
        if my_motifs.shape[1] < max_length:
            my_pad = max_length - my_motifs.shape[1]
            my_motifs = np.pad(my_motifs, ((0, 0), (0, my_pad), (0, 0)))
        if reference_motifs.shape[1] < max_length:
            reference_pad = max_length - reference_motifs.shape[1]
            reference_motifs = np.pad(
                reference_motifs, ((0, 0), (0, reference_pad), (0, 0))
            )
        # Sign: Use absolute values
        if label_unsigned:
            my_motifs = np.abs(my_motifs)
            reference_motifs = np.abs(reference_motifs)

        # Initial: IC-scale (only once)
        reference_motifs_raw = reference_motifs
        if utils_config.get_ic_scale():
            ic_scale = True
            my_motifs = utils_motif.ic_scale(my_motifs)
            reference_motifs = utils_motif.ic_scale(reference_motifs)
            utils_config.set_ic_scale(False)  # Turn off IC scale for rest of function
        else:
            ic_scale = False
        # Initial: L2 normalize motifs (for motif-motif subtraction)
        my_motifs_norm = np.linalg.norm(my_motifs, axis=(1, 2), keepdims=True)
        my_motifs = np.divide(my_motifs, my_motifs_norm, where=my_motifs_norm!=0)
        reference_motifs_norm = np.linalg.norm(reference_motifs, axis=(1, 2), keepdims=True)
        reference_motifs = np.divide(reference_motifs, reference_motifs_norm, where=reference_motifs_norm!=0)

        # Find best match, per iteration
        match_scores = []
        match_labels = []
        match_motifs = []
        match_idxs = []
        match_mask = np.ones((my_motifs.shape[0],), dtype=bool)  # (N,)
        for i in range(max_submotifs):
            if len(my_motifs) > 0:
                # Compute similarity (L2 norm included during similarity calculation)
                sim, alignment_rc, alignment_h = utils_similarity.compute_similarities(
                    [my_motifs, reference_motifs], [(0, 1)]
                )[0]
                # Unscale L2 norm
                my_motifs_norm = np.linalg.norm(my_motifs, axis=(1, 2))[:, np.newaxis]
                reference_motifs_norm = np.linalg.norm(reference_motifs, axis=(1, 2))[np.newaxis, :]
                sim = sim * (
                    my_motifs_norm * reference_motifs_norm
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
                match_motif = reference_motifs_raw[match_idx]
                # Remove matches below threshold
                match_mask = match_mask & (match_score >= min_score)  # (N,)
                match_score[~match_mask] = 0
                match_idx[~match_mask] = -1
                match_motif[~match_mask] = 0
                # Subtract best match
                if match_mask.any():
                    my_motifs = utils_motif.remove_motif_component(
                        my_motifs,
                        match_motif,
                        alignment_rc,
                        alignment_h,
                    )
                # Remove motifs with no match
                my_motifs[~match_mask] = 0
            else:
                # No motifs left to match
                match_score = np.zeros((0,), dtype=float)
                match_idx = np.zeros((0,), dtype=int)
                match_motif = np.zeros((0, max_length, my_motifs.shape[2]), dtype=float)
            # Save match information
            match_label = [labels[x] if x >= 0 else None for x in match_idx]
            match_scores.append(match_score)
            match_idxs.append(match_idx)
            match_motifs.append(match_motif)
            match_labels.append(match_label)

        # Restore IC-scale setting
        if ic_scale:
            utils_config.set_ic_scale(True)

        # Save match information
        for i in range(max_submotifs):
            self.metadata[f"{save_col_prefix}_score{i}"] = match_scores[i]  # Save scores
            self.metadata[f"{save_col_prefix}_name{i}"] = match_labels[i]  # Save labels
            # Save logos, matches only
            if save_images:
                self.__images[f"{save_col_prefix}_logo{i}"] = "" # Initialize images
                match_idx = np.where(match_idxs[i] >= 0)[0]
                if utf8_images is None:
                    # Generate forward logos if not provided
                    self.add_logos(
                        match_motifs[i], f"{save_col_prefix}_logo{i}", logo_trimming
                    )
                else:
                    # Copy forward logos if provided
                    self.__images.loc[match_idx, f"{save_col_prefix}_logo{i}"] = [
                        utf8_images[x] if x >= 0 else ""
                        for x in match_idxs[i][match_idx]
                    ]