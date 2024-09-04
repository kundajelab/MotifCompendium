from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

import MotifCompendium.utils.clustering as utils_clustering
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.matrix as utils_matrix
import MotifCompendium.utils.plotting as utils_plotting
import MotifCompendium.utils.similarity as utils_similarity


######################################
# MOTIF COMPENDIUM FACTORY FUNCTIONS #
######################################
def load(file_loc: str, safe: bool = True) -> MotifCompendium:
    """Loads a MotifCompendium object from file.

    Args:
        file_loc: The path to the MotifCompendium file.
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        The corresponding MotifCompendium object.

    Notes:
        Assumes the file is an h5py file with datasets 'sims', 'logos', 'similarity',
          'alignment_fb', and 'alignment_h', as well as a DataFrame called 'metadata'.
        Old objects may be incompatable with the newest version of the load function.
        Safe loading validates object integrity but may take significantly longer for
          large objects.
    """
    with h5py.File(file_loc, "r") as f:
        sims = f["sims"][:]
        logos = f["logos"][:]
        similarity = f["similarity"][:]
        alignment_fb = f["alignment_fb"][:]
        alignment_h = f["alignment_h"][:]
    metadata = pd.read_hdf(file_loc, key="metadata")
    return MotifCompendium(
        sims, logos, similarity, alignment_fb, alignment_h, metadata, safe
    )


def build(
    sims: np.ndarray,
    logos: np.ndarray | None = None,
    metadata: pd.DataFrame | None = None,
    max_chunk: int | None = None,
    max_parallel: int | None = None,
    use_gpu: bool = False,
    l2: bool = False,
    safe: bool = False,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of motifs.

    Computes pairwise similarities on a set of motifs. Creates a metadata if needed.
      Then, passes everything to the MotifCompendium constructor.

    Args:
        sims: The motifs.
        logos: The visual representation of the motifs. If None, it will be set to be
          the sims.
        metadata: The metadata for all motifs. If None, it will be set to a DataFrame
          with generic motif names.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing similarity.
          If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization when
          computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in sims.

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_parallel to the number of available
          CPUs. Currently, multi-GPU calculation is not supported (max_parallel and
          use_gpu are incompatible).
        If memory constraints are not an issue, leave chunk as None for faster
          performance. Otherwise, decrease max_chunk until calculations fit in memory.
          For a GPU with ~12GB of memory, use max_chunk=1000.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    # Check sims
    utils_similarity.validate_sims(sims)
    # Logos
    if logos is None:
        logos = sims if sims.shape[2] == 4 else utils_similarity.sim8_to_sim4(sims)
    # Metadata
    if metadata is None:
        metadata = pd.DataFrame()
        metadata["name"] = [f"motif_{i}" for i in range(sims.shape[0])]
    # Compute similarity
    print("aligning")
    start = time.time()
    similarity, alignment_fb, alignment_h = utils_similarity.compute_similarities(
        [sims], [(0, 0)], max_chunk, max_parallel, use_gpu, l2=l2
    )[0]
    np.fill_diagonal(similarity, 1)
    print(f"completed {time.time() - start}")
    # Construct object
    return MotifCompendium(
        sims, logos, similarity, alignment_fb, alignment_h, metadata, safe
    )


def build_from_modisco(
    modisco_dict: dict[str, str],
    max_chunk: int | None = None,
    max_parallel: int | None = None,
    use_gpu: bool = False,
    ic: bool = False,
    l2: bool = False,
    safe: bool = False,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of Modisco outputs.

    Loads motifs and metadata from all Modisco outputs then passes them to build().

    Args:
        modisco_dict: A dictionary from model name to modisco file path.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing similarity.
          If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        ic: Whether or not to apply information content scaling to modisco motifs.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization when
          computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in all Modisco objects.

    Assumptions:
        The model names have no - or . in them.

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_parallel to the number of available
          CPUs. Currently, multi-GPU calculation is not supported (max_parallel and
          use_gpu are incompatible).
        If memory constraints are not an issue, leave chunk as None for faster
          performance. Otherwise, decrease max_chunk until calculations fit in memory.
          For a GPU with ~12GB of memory, use max_chunk=1000.
        Using information content scaling is highly recommended.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    sims, cwms, names, num_seqlets = utils_loader.load_modiscos(modisco_dict, ic=ic)
    metadata = pd.DataFrame()
    metadata["name"] = names
    metadata["num_seqlets"] = num_seqlets
    metadata["model"] = metadata["name"].str.split("-").str[0]
    metadata["posneg"] = metadata["name"].str.split(".").str[0].str.split("-").str[1]
    return build(
        sims,
        logos=cwms,
        metadata=metadata,
        max_chunk=max_chunk,
        max_parallel=max_parallel,
        use_gpu=use_gpu,
        l2=l2,
        safe=safe,
    )


def build_from_pfms(
    pfm_file: str,
    max_chunk: int | None = None,
    max_parallel: int | None = None,
    use_gpu: bool = False,
    l2: bool = False,
    safe: bool = False,
) -> MotifCompendium:
    """Builds a MotifCompendium object from a PFM file.

    Loads PFMs, converts them to PWMs, and then passes them to build().

    Args:
        pfm_file: The file path to a PFM text file.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing similarity.
          If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization when
          computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in the PFM file.

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_parallel to the number of available
          CPUs. Currently, multi-GPU calculation is not supported (max_parallel and
          use_gpu are incompatible).
        If memory constraints are not an issue, leave chunk as None for faster
          performance. Otherwise, decrease max_chunk until calculations fit in memory.
          For a GPU with ~12GB of memory, use max_chunk=1000.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    sims, names = utils_loader.load_pfm(pfm_file)
    logos = sims
    metadata = pd.DataFrame()
    metadata["name"] = names
    compendium = build(
        sims,
        logos=logos,
        metadata=metadata,
        max_chunk=max_chunk,
        max_parallel=max_parallel,
        use_gpu=use_gpu,
        l2=l2,
        safe=safe,
    )
    return compendium


def combine(
    compendiums: list[MotifCompendium],
    max_chunk: int | None = None,
    max_parallel: int | None = None,
    use_gpu: bool = False,
    l2: bool = False,
    safe: bool = False,
) -> MotifCompendium:
    """Combines multiple MotifCompendium into one MotifCompendium.

    Computes similarity between each MotifCompendium's motifs to construct a single
      large similarity matrix. Then, passes the concatenated set of motifs and overall
      similarity matrix to the MotifCompendium constructor.

    Args:
        compendiums: A list of MotifCompendium objects.
        max_chunk: The maximum number of motifs to compute similarity on at a time. If
          None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing similarity.
          If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
          only CPUs by default.
        l2: Whether or not to use L2 normalization (instead of sqrt normalization when
          computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs from all individual
          MotifCompendium.

    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
          parallelize across CPUs by setting max_parallel to the number of available
          CPUs. Currently, multi-GPU calculation is not supported (max_parallel and
          use_gpu are incompatible).
        If memory constraints are not an issue, leave chunk as None for faster
          performance. Otherwise, decrease max_chunk until calculations fit in memory.
          For a GPU with ~12GB of memory, use max_chunk=1000.
        Safe building validates object integrity but may take significantly longer for
          large objects.
    """
    print("not yet implemented")
    assert False
    n = len(compendiums)
    # SIMILARITIES
    sims_list = [mc.sims for mc in compendiums]
    calculations = []
    for i in range(n):
        for j in range(i + 1, n):
            calculations.append((i, j))
    similarity_results = utils_similarity.compute_similarities(
        sims_list, calculations, max_chunk, max_parallel, use_gpu
    )
    similarity_block = [[None for i in range(n)] for i in range(n)]
    alignment_fb_block = [[None for i in range(n)] for i in range(n)]
    alignment_h_block = [[None for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_block[i][j] = compendiums[i].similarity
                alignment_fb_block[i][j] = compendiums[i].alignment_fb
                alignment_h_block[i][j] = compendiums[i].alignment_h
            elif i > j:
                (
                    similarity_block[i][j],
                    alignment_fb_block[i][j],
                    alignment_h_block[i][j],
                ) = results_revcomp[(i, j)]
            else:
                similarity_block[i][j] = similarity_block[j][i].T
                alignment_fb_block[i][j] = alignemnt_fb_block[j][i].T
                alignment_h_block[i][j] = alignment_h_block[j][i].T
    similarity = np.block(similarity_block)
    alignment_fb = np.block(alignment_fb_block)
    alignment_h = np.block(alignment_h_block)
    # SIMS + LOGOS
    sims = np.concatenate(sims_list)
    logos = np.concatenate([mc.logos for mc in compendiums])
    # METADATA
    metadata = pd.concat([mc.metadata for mc in compendiums])
    return MotifCompendium(
        sims, logos, similarity, alignment_fb, alignment_h, metadata, safe
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
        sims: A np.ndarray representing the motifs. Of shape (N, 30, 8/4). sims[i, :, :]
          represents motif i.
        logos: A np.ndarray representing how each motif should be drawn by logomaker. Of
          shape (N, 30, 4). logos[i, :, :] represents motif i.
        similarity: A np.ndarray containing the pairwise similarity scores between each
          motif. Of shape (N, N). similarity[i, j] is the similarity between motif i and
          motif j.
        alignment_fb: A np.ndarray containing the forward/reverse complement
          relationship between two motifs. Of shape (N, N). alignment_fb[i, j] is 0/1 if
          motif i should/shouldn't be reverse complemented to best align with motif j.
        alignment_h: A np.ndarray containing the horizontal shift information between
          two motifs. Of shape (N, N). alignment_h[i, j] represents how many bases to
          the right motif i should be shifted to best align with motif j.
        metadata: A pd.DataFrame containing metadata for each motif. Of length N.
          metadata.iloc[i, :] contains metadata about motif i.
    """

    def __init__(
        self,
        sims: np.ndarray,
        logos: np.ndarray,
        similarity: np.ndarray,
        alignment_fb: np.ndarray,
        alignment_h: np.ndarray,
        metadata: np.ndarray,
        safe: bool,
    ) -> None:
        """MotifCompendium constructor.

        This constructor takes in already defined versions of each class attribute. If
          safe, validate() is run; otherwise, it is not.

        Args:
            sims: A np.ndarray that is assigned to self.sims.
            logos: A np.ndarray that is assigned to self.logos.
            similarity: A np.ndarray that is assigned to self.similarity.
            alignment_fb: A np.ndarray that is assigned to self.alignment_fb.
            alignment_h: A np.ndarray that is assigned to self.alignment_h.
            metadata: A pd.DataFrame that is assigned to self.metadata.
            safe: Whether or not to construct the MotifCompendium safely.

        Notes:
            In general, users should use factory functions and no access this
              constructor directly.
            Safe construction validates object integrity but may take significantly
              longer for large objects.
        """
        self.sims = sims
        self.logos = logos
        self.similarity = similarity
        self.alignment_fb = alignment_fb
        self.alignment_h = alignment_h
        self.metadata = metadata
        if safe:
            self.validate()

    def cluster(
        self,
        algorithm: str = "leiden",
        similarity_threshold: float = 0.8,
        save_name: str = "cluster",
        **kwargs,
    ) -> None:
        """Cluster motifs.

        Cluster motifs using the algorithm of choice and the similarity threshold. The
          cluster assignment will be saved into the metadata in column save_name.

        Args:
            algorithm: The clustering algorithm to cluster with.
            similarity_threshold: The similarity threshold for when two motifs are
              considered similar.
            save_name: The name of the column in metadata to save motif clustering
              results into.
            **kwargs: Additional named arguments specific to the clustering algorithm of
              choice.

        Notes:
            Review the clustering algorithms available in utils/clustering.cluster().
        """
        print("clustering")
        start = time.time()
        self.metadata[save_name] = utils_clustering.cluster(
            self.similarity, algorithm, similarity_threshold, **kwargs
        )
        print(f"completed {time.time() - start}")

    def clustering_quality(self, cluster_name: str = "cluster") -> np.ndarray:
        """Produces a matrix that summarizes the quality of a particular clustering.

        Produce a matrix where diagonal entries represent lowest intra-cluster
          similarity and off-diagonal entries represent highest inter-cluster
          similarities.

        Args:
            cluster_name: The name of the column in metadata containing clustering
              annotations to group motifs by.

        Returns:
            A square np.ndarray where diagonal entries represent lowest intra-cluster
              similarity and off-diagonal entries represent highest inter-cluster
              similarities.
        """
        clusters = sorted(set(self.metadata[cluster_name]))
        scores = np.zeros((len(clusters), len(clusters)))
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if j < i:
                    continue
                similarity_slice_ij = self.get_similarity_slice(
                    self[cluster_name] == c1, self[cluster_name] == c2
                )
                if i == j:
                    scores[i, j] = np.min(similarity_slice_ij)
                else:
                    scores[i, j] = np.max(similarity_slice_ij)
                    scores[j, i] = scores[i, j]
        return scores

    def cluster_averages(
        self,
        cluster_name: str = "cluster",
        max_chunk: int | None = None,
        max_parallel: int | None = None,
        use_gpu: bool = False,
        safe: bool = True,
        aggregate_on: list[str] = [],
    ) -> MotifCompendium:
        """Creates a MotifCompendium where each motif represents a cluster of motifs.

        For each cluster, the average aligned motif is computed. Then, a new
          MotifCompendium is built from those motifs. Certain metadata columns can be
          aggregated and retained in the new cluster.

        Args:
            cluster_name: The name of the column in metadata containing clustering
              annotations to group motifs by.
            max_chunk: The maximum number of motifs to compute similarity on at a time.
              If None, it will compute the entire similarity matrix at once.
            max_parallel: The maximum number of CPUs/GPUs to use for computing
              similarity. If None, it will not parallelize the computation at all.
            use_gpu: Whether or not to use GPUs to accelerate computing similarity. Uses
              only CPUs by default.
            safe: Whether or not to construct the MotifCompendium safely.
            aggregate_on: The list of columns in the metadata to aggregate on

        Returns:
            A MotifCompendium where each entry represents a motif cluster in the current
              MotifCompendium.

        Notes:
            Use GPU if possible to accelerate calculation (CuPy required.) Otherwise,
              parallelize across CPUs by setting max_parallel to the number of available
              CPUs. Currently, multi-GPU calculation is not supported (max_parallel and
              use_gpu are incompatible).
            If memory constraints are not an issue, leave chunk as None for faster
              performance. Otherwise, decrease max_chunk until calculations fit in
              memory. For a GPU with ~12GB of memory, use max_chunk=1000.
            Safe building validates object integrity but may take significantly longer
              for large objects.
        """
        sims, logos, names, num_constituents = [], [], [], []
        aggregations = {x: [] for x in aggregate_on}
        for c in sorted(set(self[cluster_name])):
            mc_c = self[self[cluster_name] == c]
            avg_motif_sims = utils_matrix.average_motifs(
                mc_c.sims, mc_c.alignment_fb, mc_c.alignment_h
            )
            avg_motif_logo = utils_matrix._8_to_4(avg_motif_sims)
            sims.append(avg_motif_sims)
            logos.append(avg_motif_logo)
            names.append(f"{cluster_name}#{c}")
            num_constituents.append(len(mc_c))
            for x in aggregate_on:
                aggregations[x].append(np.sum(mc_c[x]))
        sims = np.stack(sims, axis=0)
        logos = np.stack(logos, axis=0)
        metadata = pd.DataFrame()
        metadata["name"] = names
        metadata["num_constituents"] = num_constituents
        for x in aggregate_on:
            metadata[x] = aggregations[x]
        return build(sims, logos, metadata, max_chunk, max_parallel, use_gpu, safe=safe)

    def extend(self):
        """Add new motifs to the current MotifCompendium."""
        print("not yet implemented")

    def assign(self):
        """Assign clusters to a new set of motifs."""
        print("not yet implemented")

    ###
    # VISUALIZATION
    ###
    def create_html(
        self, html_out: str, group_by: str = "cluster", max_parallel: int = 16
    ) -> None:
        """Creates an html file with all motifs in the current MotifCompendium.

        Produces an html file at the specified location with all motifs from the current
          MotifCompendium. Motifs are grouped by the group_by field in the metadata.
          Group averages are plotted with a green background at the top of each group.

        Args:
            html_out: The path to save the html file.
            group_by: The column in the metadata to group motifs by for visualization.
            max_parallel: The maximum number of CPUs to use for parallelizing plotting.

        Notes:
            Plotting can take a long time. Increase max_parallel to improve plotting
              time.
            If you just want to plot cluster averages, consider doing
              mc.cluster_averages(group_by).create_html("name") with the appropriate
              CPU/GPU/chunking options.
        """
        print("visualizing")
        start = time.time()
        utils_plotting.create_html(
            self.logos,
            list(self.metadata[group_by]),
            self.alignment_fb,
            self.alignment_h,
            list(self.metadata["name"]),
            html_out,
            max_parallel=max_parallel,
        )
        print(f"completed {time.time() - start}")

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

    ###
    # GENERAL
    ###
    def save(self, save_loc: str) -> None:
        """Saves the MotifCompendium to file.

        Saves the current MotifCompendium to a compressed h5py file.

        Args:
            save_loc: Where to save the MotifCompendium to.
        """
        with h5py.File(save_loc, "w") as f:
            f.create_dataset("sims", data=self.sims)
            f.create_dataset("logos", data=self.logos)
            f.create_dataset("similarity", data=self.similarity)
            f.create_dataset("alignment_fb", data=self.alignment_fb)
            f.create_dataset("alignment_h", data=self.alignment_h)
        self.metadata.to_hdf(save_loc, key="metadata", mode="a")

    def validate(self) -> None:
        """Verifies the integrity of the MotifCompendium.

        Checks the validity of each attribute (sims, logos, similarity, similarity_fb,
          similarity_h).

        Notes:
            This function can take a long time to run.
        """
        # SIMS
        assert type(self.sims) == np.ndarray
        assert len(self.sims.shape) == 3
        assert self.sims.shape[1] == 30
        assert self.sims.shape[2] in [4, 8]
        assert (self.sims >= 0).all()
        assert np.allclose(self.sims.sum(axis=(1, 2)), 1)
        # LOGOS
        assert type(self.logos) == np.ndarray
        assert len(self.logos.shape) == 3
        assert self.logos.shape[1] == 30
        assert self.logos.shape[2] == 4
        # SIMILARITY
        assert type(self.similarity) == np.ndarray
        assert len(self.similarity.shape) == 2
        assert np.allclose(self.similarity, self.similarity.T)
        assert np.max(self.similarity) == 1
        assert (self.similarity >= 0).all()
        # ALIGNMENT_FB
        assert type(self.alignment_fb) == np.ndarray
        assert len(self.alignment_fb.shape) == 2
        assert np.allclose(self.alignment_fb, self.alignment_fb.T)
        assert ((self.alignment_fb == 0) | (self.alignment_fb == 1)).all()
        # ALIGNMENT_H
        assert type(self.alignment_h) == np.ndarray
        assert len(self.alignment_h.shape) == 2
        assert np.allclose(
            self.alignment_h,
            np.where(self.alignment_fb == 0, -self.alignment_h.T, self.alignment_h.T),
        )
        # METADATA
        assert type(self.metadata) == pd.DataFrame
        # SHAPE MATCHES
        assert self.sims.shape[0] == self.logos.shape[0]
        assert self.sims.shape[0] == self.similarity.shape[0]
        assert self.sims.shape[0] == self.alignment_fb.shape[0]
        assert self.sims.shape[0] == self.alignment_h.shape[0]
        assert self.sims.shape[0] == len(self.metadata)

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
            A DataFrame containing the similarity scores between two subsets of motifs.

        Notes:
            If only slice1 is provided then the similarity scores between the motifs
              specified by slice1 and all other motifs are returned.
        """
        assert isinstance(slice1, pd.Series)
        assert slice1.dtype == bool
        keep_idxs_1 = list(self.metadata[slice1].index)
        if slice2 is None:
            similarity_slice = self.similarity[keep_idxs_1, :]
            assert similarity_slice.shape == (len(keep_idxs_1), len(self))
            if with_names:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(self.metadata[slice1]["name"]),
                    columns=list(self.metadata["name"]),
                )
                return similarity_slice_df
            else:
                return similarity_slice
        else:
            assert isinstance(slice2, pd.Series)
            assert slice2.dtype == bool
            keep_idxs_2 = list(self.metadata[slice2].index)
            similarity_slice = self.similarity[keep_idxs_1, :][:, keep_idxs_2]
            assert similarity_slice.shape == (len(keep_idxs_1), len(keep_idxs_2))
            if with_names:
                similarity_slice_df = pd.DataFrame(
                    similarity_slice,
                    index=list(self.metadata[slice1]["name"]),
                    columns=list(self.metadata[slice2]["name"]),
                )
                return similarity_slice_df
            else:
                return similarity_slice

    def __str__(self) -> str:
        """String representation of the MotifCompendium.
        """
        return f"Motif Compendium with {len(self)} motifs.\n{self.metadata}"

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

        Note:
            When a subset of the MotifCompendium is returned, it is returned as a new
              MotifCompendium object and is not built safely.
        """
        if isinstance(key, str):
            return self.metadata[key]
        elif isinstance(key, pd.Series) and key.dtype == bool:
            metadata_slice = self.metadata[key]
            keep_idxs = list(metadata_slice.index)
            metadata_slice = metadata_slice.reset_index(drop=True)
            sims_slice = self.sims[keep_idxs, :, :]
            logos_slice = self.logos[keep_idxs, :, :]
            similarity_slice = self.similarity[keep_idxs, :][:, keep_idxs]
            alignment_fb_slice = self.alignment_fb[keep_idxs, :][:, keep_idxs]
            alignment_h_slice = self.alignment_h[keep_idxs, :][:, keep_idxs]
            return MotifCompendium(
                sims_slice,
                logos_slice,
                similarity_slice,
                alignment_fb_slice,
                alignment_h_slice,
                metadata_slice,
                safe=False,
            )
        else:
            raise TypeError("MotifCompendium cannot be indexed by this")

    def __setitem__(self, key: str, value) -> None:
        """Set the value of a column in the metadata.

        Allows adding or setting columns of the MotifCompendium with the same syntax as
          a Pandas DataFrame.

        Args:
            key: The name of the column to set.
            value: The value to set that column assignment to.

        Note:
            Works exactly like setting a pd.DataFrame column does.
        """
        if isinstance(key, str):
            self.metadata[key] = value
        else:
            raise TypeError("MotifCompendium assignments cannot be done like this")

    def __eq__(self, other: MotifCompendium) -> bool:
        """Length of the MotifCompendium.
        """
        if isinstance(other, MotifCompendium):
            return (
                np.allclose(self.sims, other.sims)
                and np.allclose(self.logos, other.logos)
                and np.allclose(self.similarity, other.similarity)
                and np.allclose(self.alignment_fb, other.alignment_fb)
                and np.allclose(self.alignment_h, other.alignment_h)
                and self.metadata.equals(other.metadata)
            )
        return False
