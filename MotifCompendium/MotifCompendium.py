import h5py
import numpy as np
import pandas as pd

from typing import Dict

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
        
    Assumptions:
        Assumes the file is an h5py file with datasets 'sims', 'logos',
        'similarity', 'alignment_fb', and 'alignment_h', as well as a
        DataFrame called 'metadata'.

    Notes:
        Old objects may be incompatable with the newest version of the load
        function.
        See MotifCompendium.__init__() for more details on safe.
    """
    with h5py.File(file_loc, 'r') as f:
        sims = f["sims"][:]
        logos = f["logos"][:]
        similarity = f["similarity"][:]
        alignment_fb = f["alignment_fb"][:]
        alignment_h = f["alignment_h"][:]
    metadata = pd.read_hdf(file_loc, key="metadata")
    return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h,
            metadata, safe)

def build(sims: np.ndarray, logos: np.ndarray = None,
        metadata: pd.DataFrame = None, max_chunk: int = None,
        max_parallel: int = None, use_gpu: bool = False, l2: bool = False,
        safe: bool = False) -> MotifCompendium:
    """Builds a MotifCompendium object from a set of motifs.

    Computes pairwise similarities on a set of motifs. Creates a metadata if
    needed. Then, passes everything to the MotifCompendium constructor.

    Args:
        sims: The motifs.
        logos: The visual representation of the motifs. If None, it will be set
          to be the sims.
        metadata: The metadata for all motifs. If None, it will be set to a
          DataFrame with generic motif names.
        max_chunk: The maximum number of motifs to compute similarity on at a
          time. If None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing
          similarity. If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity.
          Uses only CPUs by default.
        l2: Whether or not to use l2 normalization (instead of sqrt
          normalization when computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in sims.
        
    Notes:
        Use GPU if possible to accelerate calculation (CuPy required.)
        If memory constraints are not an issue, leave chunk as None for faster
        performance.
        For a GPU with ~12GB of memory, use max_chunk=1000.
        Currently, mutli-GPU calculation is not supported (max_parallel and
        use_gpu are incompatible).
        See MotifCompendium.__init__() for more details on safe.
    """
    # Check sims
    utils_similarity.validate_sims(sims)
    # Logos
    if logos is None:
        logos = (sims if sims.shape[2] == 4 
                else utils_similarity.sim8_to_sim4(sims))
    # Metadata
    if metadata is None:
        metadata = pd.DataFrame()
        metadata["name"] = [f"motif_{i}" for i in range(sims.shape[0])]
    # Compute similarity
    print("aligning"); start = time.time()
    (similarity,
     alignment_fb,
     alignment_h) = utils_similarity.compute_similarities([sims], [(0, 0)],
             max_chunk, max_parallel, use_gpu, l2=l2)[0]
    np.fill_diagonal(similarity, 1)
    print(f"completed {time.time() - start}")
    # Construct object
    return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h,
            metadata, safe)

def build_from_modisco(modisco_dict: Dict[str, str], max_chunk: int = None,
        max_parallel: int = None, use_gpu: bool = False, ic: bool = False,
        l2: bool = False, safe: bool = False):
    """Builds a MotifCompendium object from a set of Modisco outputs.

    Loads motifs and metadata from all Modisco reports then passes them to
    build().

    Args:
        modisco_dict: A dictionary from model name to modisco file path.
        max_chunk: The maximum number of motifs to compute similarity on at a
          time. If None, it will compute the entire similarity matrix at once.
        max_parallel: The maximum number of CPUs/GPUs to use for computing
          similarity. If None, it will not parallelize the computation at all.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity.
          Uses only CPUs by default.
        ic: Whether or not to apply information content scaling to modisco
          motifs.
        l2: Whether or not to use l2 normalization (instead of sqrt
          normalization when computing motif similarity).
        safe: Whether or not to construct the MotifCompendium safely.

    Returns:
        A MotifCompendium object containing all motifs in all modisco objects.

    Assumptions:
        The model names have no - or . in them.
        
    Notes:
        See build() for more details on max_chunk, max_parallel, and use_gpu.
        See MotifCompendium.__init__() for more details on safe.
        See utils_loader.ic_scale() for more details on ic.
    """
    (sims,
     cwms,
            names,
            num_seqlets) = utils_loader.load_modiscos(modisco_dict, ic=ic)
    metadata = pd.DataFrame()
    metadata["name"] = names
    metadata["num_seqlets"] = num_seqlets
    metadata["model"] = metadata["name"].str.split("-").str[0]
    metadata["posneg"] = metadata["name"].str.split(".").str[0]
                         .str.split("-").str[1]
    return build(sims, logos=cwms, metadata=metadata, max_chunk=max_chunk,
            max_parallel=max_parallel, use_gpu=use_gpu, l2=l2, safe=safe)

def build_from_pfms(pfm_file: str, max_chunk: int = None,
        max_parallel: int = None, use_gpu: bool = False, l2: bool = False,
        safe: bool = False):
    sims, names = utils_loader.load_pfm(pfm_file)
    logos = sims
    metadata = pd.DataFrame()
    metadata["name"] = names
    compendium = build(sims, logos=logos, metadata=metadata, max_chunk=max_chunk, max_parallel=max_parallel, use_gpu=use_gpu, l2=l2, safe=safe)
    return compendium

def combine(compendiums, max_chunk=None, max_parallel=None, use_gpu=False, safe=False):
    print("not yet implemented"); assert(False)
    n = len(compendiums)
    # SIMILARITIES
    sims_list = [mc.sims for mc in compendiums]
    calculations = []
    for i in range(n):
        for j in range(i+1, n):
            calculations.append((i, j))
    similarity_results = utils_similarity.compute_similarities(sims_list, calculations, max_chunk, max_parallel, use_gpu)
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
                similarity_block[i][j], alignment_fb_block[i][j], alignment_h_block[i][j] = results_revcomp[(i, j)]
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
    return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h, metadata, safe)


##########################
# MOTIF COMPENDIUM CLASS #
##########################
class MotifCompendium():
    def __init__(self, sims, logos, similarity, alignment_fb, alignment_h, metadata, safe):
        self.sims = sims
        self.logos = logos
        self.similarity = similarity
        self.alignment_fb = alignment_fb
        self.alignment_h = alignment_h
        self.metadata = metadata
        if safe:
            self.validate()

    def cluster(self, algorithm="leiden", similarity_threshold=0.8, save_name="cluster", **kwargs):
        print("clustering"); start = time.time()
        self.metadata[save_name] = utils_clustering.cluster(self.similarity, algorithm, similarity_threshold, **kwargs)
        print(f"completed {time.time() - start}")

    def clustering_quality(self, cluster_name="cluster"):
        clusters = sorted(set(self.metadata[cluster_name]))
        scores = np.zeros((len(clusters), len(clusters)))
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if j < i:
                    continue
                similarity_slice_ij = self.get_similarity_slice(self[cluster_name] == c1, self[cluster_name] == c2)
                if i == j:
                    scores[i, j] = np.min(similarity_slice_ij)
                else:
                    scores[i, j] = np.max(similarity_slice_ij)
                    scores[j, i] = scores[i, j]
        return scores

    def cluster_averages(self, cluster_name="cluster", max_chunk=None, max_parallel=None, use_gpu=False, safe=True, aggregate_on=[]):
        sims, logos, names, num_constituents = [], [], [], []
        aggregations = {x: [] for x in aggregate_on}
        for c in sorted(set(self[cluster_name])):
            mc_c = self[self[cluster_name] == c]
            avg_motif_sims = utils_matrix.average_motifs(mc_c.sims, mc_c.alignment_fb, mc_c.alignment_h)
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

    def extend(self, sims, metadata=None, max_chunk=None, max_parallel=None):
        print("not yet implemented")

    def assign(self, sims, n=5):
        print("not yet implemented")

    ###
    # VISUALIZATION
    ###
    def create_html(self, html_out, group_by="cluster", max_parallel=16):
        print("visualizing"); start = time.time()
        utils_plotting.create_html(self.logos, list(self.metadata[group_by]), self.alignment_fb, self.alignment_h, list(self.metadata["name"]), html_out, max_parallel=max_parallel)
        print(f"completed {time.time() - start}")

    def heatmap(self, annot=False, label=False, show=False, save_loc=None):
        if label:
            utils_plotting.plot_heatmap(self.similarity, annot=annot, labels=list(self.metadata["name"]), show=show, save_loc=save_loc)
        else:
            utils_plotting.plot_heatmap(self.similarity, annot=annot, show=show, save_loc=save_loc)

    ###
    # GENERAL
    ###
    def save(self, save_loc):
        with h5py.File(save_loc, 'w') as f:
            f.create_dataset("sims", data=self.sims)
            f.create_dataset("logos", data=self.logos)
            f.create_dataset("similarity", data=self.similarity)
            f.create_dataset("alignment_fb", data=self.alignment_fb)
            f.create_dataset("alignment_h", data=self.alignment_h)
        self.metadata.to_hdf(save_loc, key="metadata", mode="a")

    def validate(self):
        # SIMS
        assert(type(self.sims) == np.ndarray)
        assert(len(self.sims.shape) == 3)
        assert(self.sims.shape[1] == 30)
        assert(self.sims.shape[2] in [4, 8])
        assert((self.sims >= 0).all())
        assert(np.allclose(self.sims.sum(axis=(1, 2)), 1))
        # LOGOS	
        assert(type(self.logos) == np.ndarray)
        assert(len(self.logos.shape) == 3)
        assert(self.logos.shape[1] == 30)
        assert(self.logos.shape[2] == 4)
        # SIMILARITY
        assert(type(self.similarity) == np.ndarray)
        assert(len(self.similarity.shape) == 2)
        assert(np.allclose(self.similarity, self.similarity.T))
        assert(np.max(self.similarity) == 1)
        assert((self.similarity >= 0).all())
        # ALIGNMENT_FB
        assert(type(self.alignment_fb) == np.ndarray)
        assert(len(self.alignment_fb.shape) == 2)
        assert(np.allclose(self.alignment_fb, self.alignment_fb.T))
        assert(((self.alignment_fb == 0) | (self.alignment_fb == 1)).all())
        # ALIGNMENT_H
        assert(type(self.alignment_h) == np.ndarray)
        assert(len(self.alignment_h.shape) == 2)
        assert(np.allclose(self.alignment_h, np.where(self.alignment_fb == 0, -self.alignment_h.T, self.alignment_h.T)))
        # METADATA
        assert(type(self.metadata) == pd.DataFrame)
        # SHAPE MATCHES
        assert(self.sims.shape[0] == self.logos.shape[0])
        assert(self.sims.shape[0] == self.similarity.shape[0])
        assert(self.sims.shape[0] == self.alignment_fb.shape[0])
        assert(self.sims.shape[0] == self.alignment_h.shape[0])
        assert(self.sims.shape[0] == len(self.metadata))

    def get_similarity_slice(self, slice1, slice2=None, with_names=False):
        assert(isinstance(slice1, pd.Series))
        assert(slice1.dtype == bool)
        keep_idxs_1 = list(self.metadata[slice1].index)
        if slice2 is None:
            similarity_slice = self.similarity[keep_idxs_1, :]
            assert(similarity_slice.shape == (len(keep_idxs_1), len(self)))
            if with_names:
                similarity_slice_df = pd.DataFrame(similarity_slice, index=list(self.metadata[slice1]["name"]), columns=list(self.metadata["name"]))
                return similarity_slice_df
            else:
                return similarity_slice
        else:
            assert(isinstance(slice2, pd.Series))
            assert(slice2.dtype == bool)
            keep_idxs_2 = list(self.metadata[slice2].index)
            similarity_slice = self.similarity[keep_idxs_1, :][:, keep_idxs_2]
            assert(similarity_slice.shape == (len(keep_idxs_1), len(keep_idxs_2)))
            if with_names:
                similarity_slice_df = pd.DataFrame(similarity_slice, index=list(self.metadata[slice1]["name"]), columns=list(self.metadata[slice2]["name"]))
                return similarity_slice_df
            else:
                return similarity_slice

    def __str__(self):
        return f"Motif Compendium with {len(self)} motifs.\n{self.metadata}"

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, key):
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
            return MotifCompendium(sims_slice, logos_slice, similarity_slice, alignment_fb_slice, alignment_h_slice, metadata_slice, safe=False)
        else:
            raise TypeError("MotifCompendium cannot be indexed by this")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.metadata[key] = value
        else:
            raise TypeError("MotifCompendium assignments cannot be done like this")

    def __eq__(self, other):
        if isinstance(other, MotifCompendium):
            return np.allclose(self.sims, other.sims) and np.allclose(self.logos, other.logos) and np.allclose(self.similarity, other.similarity) and np.allclose(self.alignment_fb, other.alignment_fb) and np.allclose(self.alignment_h, other.alignment_h) and self.metadata.equals(other.metadata)
        return False

