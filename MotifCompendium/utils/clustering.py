import warnings

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
import scipy.sparse
import sklearn.cluster

import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.similarity as utils_similarity


######################
# CLUSTERING HANDLER #
######################
def cluster(
    motifs: np.ndarray,
    similarity_matrix: np.ndarray,
    alignment_rc_matrix: np.ndarray,
    alignment_h_matrix: np.ndarray,
    algorithm: str,
    similarity_threshold: float = 0.0,
    init_membership: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    **kwargs,
) -> list[int]:
    """Cluster a similarity matrix.

    Given a similarity matrix, a similarity threshold, and a choice of algorithm, return
      a clustering of the similarity matrix. The clustering is a list that assigns each
      index in the similarity matrix to a number number.

    Args:
        motifs: An array of motifs corresponding to the similarity matrix. (N, L, 4)
        similarity_matrix: A square numpy array representing the similarity matrix.
        alignment_rc_matrix: A square numpy array representing the reverse complement alignment matrix.
        alignment_h_matrix: A square numpy array representing the horizontal alignment matrix.
        similarity_threshold: The threshold above which two motifs are considered to be
          similar.
        algorithm: Which clustering algorithm to use. Supported options:
            - "leiden": Leiden clustering with Reichardt & Bornholdt's quality function
                and a configuration model as a null.
            - "cpm" or "cpm_leiden": Leiden clustering with the constant Potts model.
            - "cc": Connected-component clustering.
            - "dense_cc": Densely connected-component clustering.
            - "spectral": Spectral clustering.
            - "k_centroids": K-means-style clustering, by taking the mean of the cluster ("centroid").
            - "k_medoids": K-means-style clustering, by taking the medoid of the cluster ("medoid").
            - "k_mean_distance": K-means-style clustering, by taking mean distance to all constituents.
            - "k_median_distance": K-means-style clustering, by taking median distance to all constituents.
          Supportd only for K-means-style clustering algorithms.
        init_membership: A 1D numpy array representing the initial membership of each motif.
          If not specified, the initial membership will be determined by the clustering algorithm and seed.
          Not supported:
            - "cc": Connected-component clustering
            - "spectral": Spectral clustering
        weights: A 1D numpy array representing the weight of each motif.
            If not specified, all motifs will be weighted equally. 
            Supported only for K-means-style clustering algorithms: K-centroids, K-mean-distance
        **kwargs: Any additional arguments to be passed to the clustering algorithm of
          choice.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Please look through the possible clustering algorithm options in this function.
    """
    # Check arguments
    if not (
        isinstance(similarity_matrix, np.ndarray)
        and (similarity_matrix.ndim == 2)
        and (similarity_matrix.shape[0] == similarity_matrix.shape[1])
    ):
        raise ValueError(f"The similarity matrix must be a square matrix.")
    if similarity_matrix.shape[0] == 1:
        return [0]

    match algorithm.lower():
        # Leiden
        case "mod_leiden" | "leiden_mod" | "modularity_leiden" | "leiden_modularity" | "rb_leiden":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            # if utils_config.get_use_gpu():
            #     return modularity_leiden_clustering_gpu(weighted_adjacency_matrix, **kwargs)
            return rb_leiden_clustering_cpu(
                weighted_adjacency_matrix=weighted_adjacency_matrix,
                init_membership=init_membership,
                **kwargs
            )
        case "cpm" | "cpm_leiden" | "leiden_cpm" |"constant_potts_leiden" | "leiden_constant_potts":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return cpm_leiden_clustering(
                weighted_adjacency_matrix=weighted_adjacency_matrix,
                init_membership=init_membership,
                **kwargs
            )
        case "leiden" | "leidenalg":
            warnings.warn(
                "Defaulting Leiden algorithm: Constant Potts model", UserWarning
            )
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return cpm_leiden_clustering(
                weighted_adjacency_matrix=weighted_adjacency_matrix, 
                init_membership=init_membership,
                **kwargs
            )
        # Connected-components
        case "cc" | "connected_components":
            if init_membership is not None:
                warnings.warn("init_membership is provided but not supported for connected component clustering: Ignoring init_membership.", UserWarning)
            binary_adjacency_matrix = similarity_matrix >= similarity_threshold
            return cc_clustering(
                adjacency_matrix=binary_adjacency_matrix,
            )
        case "dcc" | "dense_cc" | "densely_connected_components":
            binary_adjacency_matrix = similarity_matrix >= similarity_threshold
            return densely_cc_clustering(
                adjacency_matrix=binary_adjacency_matrix,
                init_membership=init_membership, 
                **kwargs
            )
        # Spectral
        case "spectral":
            if init_membership is not None:
                warnings.warn("init_membership is provided but not supported for spectral clustering: Ignoring init_membership.", UserWarning)
            thresholded_similarity_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold)
            return spectral_clustering_k(
                similarity_matrix=thresholded_similarity_matrix, 
                **kwargs
            )
        # K-means
        case "k_centroids" | "kmeans_centroids" | "k_means_centroids":
            return k_centroids_clustering(
                motifs=motifs,
                similarity_matrix=similarity_matrix,
                alignment_rc_matrix=alignment_rc_matrix,
                alignment_h_matrix=alignment_h_matrix,
                init_membership=init_membership,
                weights=weights,
                **kwargs
            )
        case "k_medoids" | "kmedoids":
            return k_medoids_clustering(
                similarity_matrix=similarity_matrix,
                init_membership=init_membership,
                **kwargs
            )
        case "k_mean_distance" | "kmeans_distance" | "k_means_distance":
            return k_mean_distance_clustering(
                similarity_matrix=similarity_matrix, 
                init_membership=init_membership,
                weights=weights,
                **kwargs
            )
        case "k_median" | "k_median_distance" | "kmedians_distance":
            return k_median_distance_clustering(
                similarity_matrix=similarity_matrix,
                init_membership=init_membership,
                **kwargs
            )
        # Other
        case _:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


#####################
# LEIDEN CLUSTERING #
#####################
def rb_leiden_clustering_cpu(
    weighted_adjacency_matrix: np.ndarray,
    resolution_parameter: float = 1.0,
    n_iterations: int = -1,
    init_membership: np.ndarray = None,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """Perform Leiden clustering with R&B quality and a configuration null.

    Args:
        weighted_adjacency_matrix : A square weighted adjacency matrix.
        resolution_parameter: The resolution parameter for the Leiden algorithm. A
          resolution_parameter of 1 is equivalent to using modularity as the quality
          function.
        n_iterations: The number of iterations that Leiden is allowed to run. If
          n_iterations is -1, then Leiden will run until there is no longer any
          improvement in quality.
        init_membership: The initial membership assignment for the vertices.
          If specified, it will override and ignore seeds.
        seeds: Seeds with which to run Leiden. Each seed will correspond to an
          independent run of Leiden. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses Reichardt & Bornholdt's quality function with a configuration model as a
          a null. See leidenalg.RBConfigurationVertexPartition for more details.
    """
    # Check arguments
    if (isinstance(init_membership, np.ndarray) and
        init_membership.ndim == 1 and
        init_membership.shape[0] == weighted_adjacency_matrix.shape[0]
    ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        seeds = [None]  # No random seed
        init_membership = init_membership.tolist()  # Convert to list for leidenalg
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of vertices.")

    # Create igraph object
    # g = ig.Graph.Weighted_Adjacency(weighted_adjacency_matrix, mode="undirected")
    n_vertices = weighted_adjacency_matrix.shape[0]
    rows, cols = np.nonzero(weighted_adjacency_matrix)
    edges = list(zip(rows, cols))
    weights = weighted_adjacency_matrix[rows, cols]

    g = ig.Graph(n_vertices, edges=edges)
    g.es["weight"] = weights

    # Run RB Leiden clustering
    best_quality = None
    best_membership = None
    for seed in seeds:
        partition = la.find_partition(
            graph=g,
            partition_type=la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution_parameter,
            n_iterations=n_iterations,
            initial_membership=init_membership,
            seed=seed,
        )
        if best_quality is None or partition.quality() > best_quality:
            best_quality = partition.quality()
            best_membership = partition.membership
    return best_membership


def cpm_leiden_clustering(
    weighted_adjacency_matrix: np.ndarray,
    resolution_parameter: float = 1.0,
    n_iterations: int = -1,
    init_membership: np.ndarray = None,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """Perform Leiden clustering with the constant Potts model.

    Args:
        weighted_adjacency_matrix : A square weighted adjacency matrix.
        resolution_parameter: The resolution parameter for the Leiden algorithm.
        n_iterations: The number of iterations that Leiden is allowed to run. If
          n_iterations is -1, then Leiden will run until there is no longer any
          improvement in quality.
        init_membership: The initial membership assignment for the vertices.
          If specified, it will override and ignore seeds.
        seeds: Seeds with which to run Leiden. Each seed will correspond to an
          independent run of Leiden. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses the constant Potts model. See leidenalg.CPMVertexPartition for more
          details.
    """
    # Check arguments
    if (isinstance(init_membership, np.ndarray) and
        init_membership.ndim == 1 and
        init_membership.shape[0] == weighted_adjacency_matrix.shape[0]
    ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        seeds = [None]  # No random seed
        init_membership = init_membership.tolist()  # Convert to list for leidenalg
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of vertices.")

    # Create igraph object
    n_vertices = weighted_adjacency_matrix.shape[0]
    rows, cols = np.nonzero(weighted_adjacency_matrix)
    edges = list(zip(rows, cols))
    weights = weighted_adjacency_matrix[rows, cols]

    g = ig.Graph(n_vertices, edges=edges)
    g.es["weight"] = weights

    # Run CPM Leiden clustering
    best_quality = None
    best_membership = None
    for seed in seeds:
        partition = la.find_partition(
            graph=g,
            partition_type=la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=resolution_parameter,
            n_iterations=n_iterations,
            initial_membership=init_membership,
            seed=seed,
        )
        if best_quality is None or partition.quality() > best_quality:
            best_quality = partition.quality()
            best_membership = partition.membership
    return best_membership


def rb_leiden_clustering_gpu(
    weighted_adjacency_matrix: np.ndarray,
    resolution_parameter: float = 1.0,
    n_iterations: int = 100,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """
    Perform Leiden clustering with R&B quality and a configuration null, on GPU.

    Args:
        weighted_adjacency_matrix : A square weighted adjacency matrix.
        resolution_parameter: The resolution parameter for the Leiden algorithm. A
          resolution_parameter of 1 is equivalent to using modularity as the quality
          function.
        n_iterations: The number of iterations that Leiden is allowed to run. If
          n_iterations is -1, then Leiden will run until there is no longer any
          improvement in quality.
        seeds: Seeds with which to run Leiden. Each seed will correspond to an
          independent run of Leiden. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses Leiden method described in: Traag, V. A., Waltman, L., & van Eck, N. J.
          (2019). From Louvain to Leiden: guaranteeing well-connected communities.
          Scientific reports, 9(1), 5233. doi: 10.1038/s41598-019-41695-z. See
          cugraph.leiden for more details.
    """
    raise NotImplementedError("GPU version not yet implemented.")
    """
    import cugraph
    import cudf
    
    # Convert adjacency matrix to sparse format
    sparse_matrix = scipy.sparse.coo_matrix(adjacency_matrix)
    
    # Create a cuGraph graph from edge list
    for int_prec, flt_prec in [(np.int64, np.float64), (np.int64, np.float32), (np.int32, np.float32)]:
        try:
            G = cugraph.Graph()
            G.from_cudf_edgelist(cudf.DataFrame({
                        'src': sparse_matrix.row.astype(int_prec),
                        'dst': sparse_matrix.col.astype(int_prec),
                        'weight': sparse_matrix.data.astype(flt_prec),
                    }), source='src', destination='dst', edge_attr='weight', renumber=False)
            break
        except Exception as e:
            print(f"Failed to convert adjacency matrix with {int_prec} and {flt_prec}: {e}")
            continue
    else:
        raise ValueError("Failed to convert adjacency matrix to cuDF.")
    
    # Run Leiden clustering
    best_quality = None
    best_membership = None
    for seed in seeds:
        partition, quality = cugraph.leiden(G, 
            resolution=resolution_parameter, 
            max_iter=n_iterations, 
            random_state=seed
        )
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_membership = partition['partition'].to_numpy().tolist()
    return best_membership
    """


#################
# CC CLUSTERING #
#################
def cc_clustering(
    adjacency_matrix: np.ndarray
) -> list[int]:
    """Find connected components in an adjacency matrix."""
    N = adjacency_matrix.shape[0]
    clustering = np.zeros((N,), dtype=int)
    current_cluster = 1
    # Iterate through all nodes
    for current_node in range(N):
        # If already clustered, continue
        if clustering[current_node] != 0:
            continue
        # Otherwise, create new cluster
        considering = np.zeros((N,))
        considering[current_node] = 1
        found_cc = False
        # Keep expanding component until it no longer changes
        while not found_cc:
            new_considering = (adjacency_matrix @ considering) > 0
            found_cc = (new_considering == considering).all()
            considering = new_considering
        # Assign cluster values
        clustering += current_cluster * considering
        # Move to next cluster
        current_cluster += 1
    # Return
    assert (clustering > 0).all()
    return (clustering - 1).tolist()


def densely_cc_clustering(
    adjacency_matrix: np.ndarray,
    density: float = 1.0,
    init_membership: np.ndarray = None,
    seed: int = 100
) -> list[int]:
    """Find densely connected components in a greedy fashion in an adjacency matrix."""
    # Check arguments
    if not (isinstance(density, float) and (0 < density <= 1)):
        raise ValueError("Density must be a float between 0 and 1.")
    # Initialize order of nodes:
    #   If init_membership is provided, order based on membership. 
    #   If seed is 0, do not shuffle.
    N = adjacency_matrix.shape[0]
    if (isinstance(init_membership, np.ndarray) 
        and init_membership.ndim == 1 
        and init_membership.shape[0] == N
    ):
        _, node_order = np.unique(init_membership, return_index=True)
        node_order = np.argsort(node_order)  # Order nodes by first occurrence of their membership
    elif seed == 0:
        warnings.warn("Seed set as zero. Existing node order used without shuffling", UserWarning)
        node_order = np.arange(N)
    else:
        node_order = np.random.default_rng(seed).permutation(N)
    # Cluster
    clustering = np.zeros((N,), dtype=int)
    current_cluster = 1
    for i, current_node in enumerate(node_order):
        # If already clustered, continue
        if clustering[current_node] != 0:
            continue
        # Otherwise, create new cluster
        clustering[current_node] = current_cluster
        new_cluster_idxs = [current_node]
        # Check all remaining nodes in a single pass for clustering
        for consider_node in node_order[i + 1 :]:
            num_edges = np.sum(adjacency_matrix[consider_node, new_cluster_idxs])
            # If sufficient edge density, add to cluster
            if num_edges / len(new_cluster_idxs) >= density:
                clustering[consider_node] = current_cluster
                new_cluster_idxs.append(consider_node)
        # Move to next cluster
        current_cluster += 1
    # Return
    assert (clustering > 0).all()
    return (clustering - 1).tolist()


#######################
# SPECTRAL CLUSTERING #
#######################
def spectral_clustering_k(
    similarity_matrix: np.ndarray,
    k: int = 40,
    cluster_qr: bool = False
) -> list[int]:
    """Spectral clustering on a similarity matrix for a fixed number of clusters."""
    if cluster_qr:
        clustering = (
            sklearn.cluster.SpectralClustering(
                k, affinity="precomputed", assign_labels="cluster_qr"
            )
            .fit(similarity_matrix)
            .labels_
        )
    else:
        clustering = (
            sklearn.cluster.SpectralClustering(k, affinity="precomputed")
            .fit(similarity_matrix)
            .labels_
        )
    return clustering


############################
# K-MEANS-STYLE CLUSTERING #
############################
## HELPER FUNCTIONS ##
def _initialize_anchors(
    similarity_matrix: np.ndarray,
    method: str,
    k: int,
    seed: int | None,
) -> np.ndarray:
    """Initialize anchor indices for each cluster from an NxN similarity matrix.
    
    Args:
        similarity_matrix: An NxN numpy array representing similarities between N points.
        method: The method for initializing anchors. Supported options:
            - "random": Pick k random indices as anchors.
            - "maximin": Pick the first anchor at random, then pick subsequent anchors
              iteratively by picking the point with maximum distance to the closest
              chosen anchor.
            - "kmeans++": Pick the first anchor at random, then pick subsequent anchors
              iteratively by picking points with probability proportional to the square of
              distance to the closest chosen anchor.
        k: The number of anchors/clusters to initialize.
        seed: The random seed to use for initialization.
    
    Returns:
        Anchor indices: A numpy array of shape (k,) where each element is an integer 
          representing the index of the anchor point for that cluster.
    """
    N = similarity_matrix.shape[0]
    rng = np.random.default_rng(seed)

    # Random: Pick k random indices as anchors
    if method == "random":
        return rng.choice(np.arange(N), size=k, replace=False).astype(int)  # (k,)

    # Pick at random, then use maximin or kmeans++ to pick subsequent anchors
    first = rng.integers(0, N)
    chosen = [first]
    while len(chosen) < k:
        dist = 1 - similarity_matrix[:, chosen].max(axis=1)
        dist = np.clip(dist, 0, None)
        dist[np.array(chosen)] = 0
        # If no remaining points: Pick random remaining point
        if dist.sum() == 0:
            remaining = np.setdiff1d(np.arange(N), np.array(chosen), assume_unique=False)
            chosen.append(int(rng.choice(remaining)))
        # Maximin: Pick point with maximum distance to the closest chosen anchor
        elif method == "maximin":
            chosen.append(int(np.argmax(dist)))
        # Kmeans++: Pick point with probability proportional to the square of distance to closest chosen anchor
        else:  # kmeans++
            probs = dist / dist.sum()
            chosen.append(int(rng.choice(np.arange(N), p=probs)))
    
    # Return: k anchor points
    return np.array(chosen, dtype=int)  # (k,)

def _initialize_membership(
    similarity_matrix: np.ndarray,
    k: int,
    seed: int | None,
    method: str,
) -> np.ndarray:
    """Initialize memberships for each point from an NxN similarity matrix, by initializing k anchors 
    and assigning each point to the cluster of the nearest anchor.
    
    Args:
        similarity_matrix: An NxN numpy array representing similarities between N points.
        method: The method for initializing anchors. Supported options:
            - "random": Pick k random indices as anchors.
            - "maximin": Pick the first anchor at random, then pick subsequent anchors
              iteratively by picking the point with maximum distance to the closest
              chosen anchor.
            - "kmeans++": Pick the first anchor at random, then pick subsequent anchors
              iteratively by picking points with probability proportional to the square of
              distance to the closest chosen anchor.
        k: The number of anchors/clusters to initialize.
        seed: The random seed to use for initialization.
    
    Returns:
        Membership: A numpy array of shape (N,) where each element is an integer 
          representing the cluster that index corresponds to. All elements with the same
          value have been assigned to the same cluster.
        """
    anchors = _initialize_anchors(
        similarity_matrix=similarity_matrix,
        method=method,
        k=k,
        seed=seed,
    )
    # Return: Membership per point, by assigning each point to the cluster of the nearest anchor
    return similarity_matrix[:, anchors].argmax(axis=1)  # (N,)

def _update_assignment_threshold(
    membership: np.ndarray,
    similarity_matrix_motif_cluster: np.ndarray,
    assignment_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign points to clusters, if new assignment improvement in similarity is greater than the threshold."""
    membership_hyp = similarity_matrix_motif_cluster.argmax(axis=1)  # (N,)
    similarity_motif_cluster_hyp = similarity_matrix_motif_cluster.max(axis=1)  # (N,)
    similarity_motif_cluster_curr = similarity_matrix_motif_cluster[
        np.arange(similarity_matrix_motif_cluster.shape[0]), membership
    ]  # (N,)
    similarity_motif_cluster_diff = similarity_motif_cluster_hyp - similarity_motif_cluster_curr
    # Update
    membership_update = np.where(
        similarity_motif_cluster_diff >= assignment_threshold,
        membership_hyp,
        membership,
    )  # (N,)
    similarity_motif_cluster_update = np.where(
        similarity_motif_cluster_diff >= assignment_threshold,
        similarity_motif_cluster_hyp,
        similarity_motif_cluster_curr,
    )  # (N,)
    return membership_update, similarity_motif_cluster_update


## K-CENTROIDS CLUSTERING ##
def k_centroids_clustering(
    motifs: np.ndarray,
    similarity_matrix: np.ndarray,
    alignment_rc_matrix: np.ndarray,
    alignment_h_matrix: np.ndarray,
    init_membership: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    k: int = None,
    init_method: str = "kmeans++",
    assignment_threshold: float = 0.0,
    n_iterations: int = -1,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """K-means clustering, by taking the mean of the cluster ("centroid") as the cluster representation, 
    and assigning cluster membership to motifs by assigning the cluster with the closest centroid.
    
    Args:
        motifs: An array of motifs corresponding to the similarity matrix. (N, L, 4)
        similarity_matrix: A square numpy array representing similarities between N motifs.
        alignment_rc_matrix: A square numpy array representing reverse complement alignments between N motifs.
        alignment_h_matrix: A square numpy array representing horizontal alignments between N motifs.
        init_membership: A 1D numpy array representing the initial membership of each motif.
        weights: A 1D numpy array representing the weight of each motif.
          If not specified, all motifs will be weighted equally. 
          The length of weights must be equal to the number of motifs.
        k: The number of clusters to find. Must be specified if init_membership is not specified. 
          Otherwise, k will be set to the number of unique clusters in init_membership.
        init_method: The method for initializing cluster centroids. Supported options:
          - "random": Pick k random indices as initial centroids.
          - "kmeans++": Pick the first centroid at random, then pick subsequent centroids
            iteratively by picking points with probability proportional to the square of distance 
            to the closest chosen centroid.
          - "maximin": Pick the first centroid at random, then pick subsequent centroids iteratively 
            by picking the point with maximum distance to the closest
            chosen centroid.
        assignment_threshold: The minimum increase in similarity required for a motif to switch to
          a new cluster. If the closest centroid is not at least this much more similar, 
          then the motif will not switch clusters.
        n_iterations: The max number of iterations to run k-means, per run/seed. 
          If -1, run until convergence.
        seeds: Seeds with which to run clustering. Each seed will correspond to an
          independent run of clustering. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of clustering that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.
    """
    # Import libraries (avoid circular imports)
    import MotifCompendium
    N = similarity_matrix.shape[0]

    # Initialization: Similarity, Motif, Membership
    if (isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
        ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of motifs.")

    # Weights
    if weights is None:
        weights = np.ones((N,))
    elif not (isinstance(weights, np.ndarray) and (weights.ndim == 1) and (weights.shape[0] == N)):
        raise ValueError("weights must be a numpy array of floats of length equal to the number of motifs.")

    # Run K-centroids clustering:
    global_membership = []
    global_score = 0
    for seed in seeds:
        # Initialize
        if init_method is not None:
            init_membership = _initialize_membership(
                similarity_matrix=similarity_matrix,
                k=k,
                seed=seed,
                method=init_method,
            )
        # Build MotifCompendium object
        mc = MotifCompendium.MotifCompendium(
            motifs,
            similarity_matrix,
            alignment_rc_matrix,
            alignment_h_matrix,
            pd.DataFrame({
                "membership": init_membership,
                "weights": weights,
            }),
            pd.DataFrame(),
            safe=False,
        )
        # Run clustering
        membership_old = init_membership
        score_old = 0
        iteration = 0
        while iteration != n_iterations:
            # Compute cluster representations: Centroids
            mc_average = mc.cluster_averages(
                clustering="membership",
                weight_col="weights",
                aggregations=[],
            )
            # Calculate distance matrix: Motif to clusters
            similarity_matrix_motif_cluster, _, _ = utils_similarity.compute_similarities(
                [mc.motifs, mc_average.motifs], [(0, 1)]
            )[0]  # (N, k)
            # Update membership, distance, and score, with threshold
            membership_new, similarity_motif_cluster = _update_assignment_threshold(
                membership=membership_old,
                similarity_matrix_motif_cluster=similarity_matrix_motif_cluster,
                assignment_threshold=assignment_threshold,
            )  # (N,) (N,)
            mc["membership"] = membership_new
            score_new = similarity_motif_cluster.sum()
            # Check for convergence: Across steps
            if np.array_equal(membership_old, membership_new):
                break
            # Update best score and membership
            else:
                membership_old = membership_new
                score_old = score_new
                iteration += 1
        # Check for convergence: Across seeds
        if score_new >= global_score:
            global_membership = membership_new
            global_score = score_new
    # Remove temp columns
    mc.delete_columns(["membership", "weights"])

    return global_membership.tolist()


## K-MEDOIDS CLUSTERING ##
def _find_k_medoids(
    memberships: np.ndarray,
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """Update each cluster medoid to maximize in-cluster similarity."""
    # One-hot encode memberships: shape (N, k)
    one_hot_membership = (
        memberships[:, None] == np.arange(len(np.unique(memberships)))[None, :]
    ).astype(float)  # (N, k)
    # cluster_sums[i, c] = sum of similarity to all points in cluster c
    cluster_sums = similarity_matrix @ one_hot_membership  # (N, k)
    # Mask out points not belonging to each cluster
    cluster_sums[~one_hot_membership.astype(bool)] = -np.inf
    # Medoid: argmax per cluster
    medoids = cluster_sums.argmax(axis=0)  # (k,)
    return medoids.astype(int)


def k_medoids_clustering(
    similarity_matrix: np.ndarray,
    init_membership: np.ndarray | None = None,
    k: int = None,
    init_method: str = "kmeans++",
    assignment_threshold: float = 0.0,
    n_iterations: int = -1,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """K-medoids clustering, by taking the motif with the closest similarity to all other motifs 
    in the cluster as the cluster representation ("medoid"), and assigning cluster membership 
    to motifs by assigning the cluster with the closest medoid.
    
    Args:
        similarity_matrix: A np.ndarray similarity matrix of shape (N, N)
        init_membership: A np.ndarray of integers representing the initial, starting membership of each node.
        k: The number of clusters to find. Must be specified if init_membership is not specified. 
          Otherwise, k will be set to the number of unique clusters in init_membership.
        init_method: The method for initializing cluster centroids. Supported options:
          - "random": Pick k random indices as initial centroids.
          - "kmeans++": Pick the first centroid at random, then pick subsequent centroids
            iteratively by picking points with probability proportional to the square of distance 
            to the closest chosen centroid.
          - "maximin": Pick the first centroid at random, then pick subsequent centroids iteratively 
            by picking the point with maximum distance to the closest
            chosen centroid.
        assignment_threshold: The minimum increase in similarity required for a motif to switch to
          a new cluster. If the closest centroid is not at least this much more similar, 
          then the motif will not switch clusters.
        n_iterations: The max number of iterations to run k-means, per run/seed. 
          If -1, run until convergence.
        seeds: Seeds with which to run clustering. Each seed will correspond to an
          independent run of clustering. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of clustering that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.
    """
    # Initialization: Membership
    N = similarity_matrix.shape[0]
    if (isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
        ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of motifs.")

    # Run K-medoids clustering:
    global_membership = []
    global_score = 0
    for seed in seeds:
        # Initialize: 
        if init_method is not None:
            init_membership = _initialize_membership(
                similarity_matrix=similarity_matrix,
                k=k,
                seed=seed,
                method=init_method,
            )

        # Run clustering
        membership_old = init_membership
        score_old = 0
        iteration = 0
        while iteration != n_iterations:
            # Compute cluster representations: Medoids
            medoids = _find_k_medoids(
                memberships=membership_old,
                similarity_matrix=similarity_matrix,
            )
            # Calculate distance matrix: Motif to clusters
            similarity_matrix_motif_cluster = similarity_matrix[:, medoids]  # (N, k)
            # Update membership, distance, and score, with threshold
            membership_new, similarity_motif_cluster = _update_assignment_threshold(
                membership=membership_old,
                similarity_matrix_motif_cluster=similarity_matrix_motif_cluster,
                assignment_threshold=assignment_threshold,
            )  # (N,) (N,)
            score_new = similarity_motif_cluster.sum()
            # Check for convergence: Across steps
            if np.array_equal(membership_old, membership_new):
                break
            # Update best score and membership
            else:
                membership_old = membership_new
                score_old = score_new
                iteration += 1
        # Check for convergence: Across seeds
        if score_new >= global_score:
            global_membership = membership_new
            global_score = score_new

    return global_membership.tolist()


## K-MEAN DISTANCE CLUSTERING ##
def _calculate_k_mean_distance(
    memberships: np.ndarray,
    similarity_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Caclulate mean distance from each point in each cluster, weighted by the weights of each point."""
    N = similarity_matrix.shape[0]
    k = len(np.unique(memberships))
    # One-hot encode memberships:
    one_hot_membership = (
        memberships[:, None] == np.arange(k)[None, :]
    ).astype(float)  # (N, k)
    weighted_membership = one_hot_membership * weights[:, None]  # (N, k)
    # Calculate weighted mean distance
    weight_sums = weighted_membership.sum(axis=0)  # (k,)
    k_mean_distance = (similarity_matrix @ weighted_membership) / weight_sums  # (N, k)
    return k_mean_distance.astype(float)  # (N, k)


def k_mean_distance_clustering(
    similarity_matrix: np.ndarray,
    init_membership: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    k: int = None,
    init_method: str = "kmeans++",
    assignment_threshold: float = 0.0,
    n_iterations: int = -1,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """K-means clustering, by taking the mean distance across all motifs in the cluster as the 
    representative distance to each cluster ("mean distance"), and assigning cluster membership
    to motifs by assigning the cluster with the closest mean distance.
    
    Args:
        similarity_matrix: A square, symmetric similarity matrix.
        init_membership: A np.ndarray of integers representing the initial, starting membership of each node.
        weights: A np.ndarray of floats representing the weight of each motif.
          If not specified, all motifs will be weighted equally. 
          The length of weights must be equal to the number of motifs.
        k: The number of clusters to find. Must be specified if init_membership is not specified. 
          Otherwise, k will be set to the number of unique clusters in init_membership.
        init_method: The method for initializing cluster centroids. Supported options:
          - "random": Pick k random indices as initial centroids.
          - "kmeans++": Pick the first centroid at random, then pick subsequent centroids
            iteratively by picking points with probability proportional to the square of distance 
            to the closest chosen centroid.
          - "maximin": Pick the first centroid at random, then pick subsequent centroids iteratively 
            by picking the point with maximum distance to the closest
            chosen centroid.
        assignment_threshold: The minimum increase in similarity required for a motif to switch to
          a new cluster. If the closest centroid is not at least this much more similar, 
          then the motif will not switch clusters.
        n_iterations: The max number of iterations to run k-means, per run/seed. 
          If -1, run until convergence.
        seeds: Seeds with which to run clustering. Each seed will correspond to an
          independent run of clustering. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of clustering that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.
    """
    # Initialization: Membership
    N = similarity_matrix.shape[0]
    if (isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
        ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of motifs.")

    # Weights
    if weights is None:
        weights = np.ones((N,))
    elif not (isinstance(weights, np.ndarray) and (weights.ndim == 1) and (weights.shape[0] == N)):
        raise ValueError("weights must be a numpy array of floats of length equal to the number of motifs.")

    # Run K-mean distance clustering:
    global_membership = []
    global_score = 0
    for seed in seeds:
        # Initialize
        if init_method is not None:
            init_membership = _initialize_membership(
                similarity_matrix=similarity_matrix,
                k=k,
                seed=seed,
                method=init_method,
            )

        # Run clustering
        membership_old = init_membership
        score_old = 0
        iteration = 0
        while iteration != n_iterations:
            # Calculate distance matrix: Motif to clusters
            similarity_matrix_motif_cluster = _calculate_k_mean_distance(
                memberships=membership_old,
                similarity_matrix=similarity_matrix,
                weights=weights,
            )  # (N, k)
            # Update membership, distance, and score, with threshold
            membership_new, similarity_motif_cluster = _update_assignment_threshold(
                membership=membership_old,
                similarity_matrix_motif_cluster=similarity_matrix_motif_cluster,
                assignment_threshold=assignment_threshold,
            )  # (N,) (N,)
            score_new = similarity_motif_cluster.sum()
            # Check for convergence: Across steps
            if np.array_equal(membership_old, membership_new):
                break
            # Update best score and membership
            else:
                membership_old = membership_new
                score_old = score_new
                iteration += 1
        # Check for convergence: Across seeds
        if score_new >= global_score:
            global_membership = membership_new
            global_score = score_new

    return global_membership.tolist()


## K-MEDIAN DISTANCE CLUSTERING ##
def _calculate_k_median_distance(
    memberships: np.ndarray,
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """Caclulate median distance from each point in each cluster."""
    N = similarity_matrix.shape[0]
    k = len(np.unique(memberships))
    k_median_distance = np.zeros((N, k))  # (N, k)
    for c in range(k):
        cluster_idxs = np.where(memberships == c)[0]
        if len(cluster_idxs) == 0:
            k_median_distance[:, c] = 0
        else:
            k_median_distance[:, c] = np.median(similarity_matrix[:, cluster_idxs], axis=1)
    return k_median_distance.astype(float)  # (N, k)


def k_median_distance_clustering(
    similarity_matrix: np.ndarray,
    init_membership: np.ndarray | None = None,
    k: int = None,
    init_method: str = "kmeans++",
    assignment_threshold: float = 0.0,
    n_iterations: int = -1,
    seeds: list[int] = [100, 200],
) -> list[int]:
    """K-medoid clustering, by taking the median distance across all motifs in the cluster as the 
    representative distance to each cluster ("median distance"), and assigning cluster membership
    to motifs by assigning the cluster with the closest median distance.
    
    Args:
        similarity_matrix: A square, symmetric similarity matrix.
        init_membership: A np.ndarray of integers representing the initial, starting membership of each node.
        k: The number of clusters to find. Must be specified if init_membership is not specified. 
          Otherwise, k will be set to the number of unique clusters in init_membership.
        init_method: The method for initializing cluster centroids. Supported options:
          - "random": Pick k random indices as initial centroids.
          - "kmeans++": Pick the first centroid at random, then pick subsequent centroids
            iteratively by picking points with probability proportional to the square of distance 
            to the closest chosen centroid.
          - "maximin": Pick the first centroid at random, then pick subsequent centroids iteratively 
            by picking the point with maximum distance to the closest
            chosen centroid.
        assignment_threshold: The minimum increase in similarity required for a motif to switch to
          a new cluster. If the closest centroid is not at least this much more similar, 
          then the motif will not switch clusters.
        n_iterations: The max number of iterations to run k-means, per run/seed. 
          If -1, run until convergence.
        seeds: Seeds with which to run clustering. Each seed will correspond to an
          independent run of clustering. The clustering from the run with the highest
          quality will be returned. The length of seeds is equal to the number of
          runs of clustering that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.
    """
    # Initialize: Membership
    N = similarity_matrix.shape[0]
    if (isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
        ):
        _, init_membership = np.unique(init_membership, return_inverse=True) # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(f"init_membership must be a numpy array of integers of length equal to the number of motifs.")

    # Run K-median distance clustering:
    global_membership = []
    global_score = 0
    for seed in seeds:
        # Initialize
        if init_method is not None:
            init_membership = _initialize_membership(
                similarity_matrix=similarity_matrix,
                k=k,
                seed=seed,
                method=init_method,
            )
        
        # Run clustering
        membership_old = init_membership
        score_old = 0
        iteration = 0
        while iteration != n_iterations:
            # Calculate distance matrix: Motif to clusters
            similarity_matrix_motif_cluster = _calculate_k_median_distance(
                memberships=membership_old,
                similarity_matrix=similarity_matrix,
            )  # (N, k)
            # Update membership, distance, and score, with threshold
            membership_new, similarity_motif_cluster = _update_assignment_threshold(
                membership=membership_old,
                similarity_matrix_motif_cluster=similarity_matrix_motif_cluster,
                assignment_threshold=assignment_threshold,
            )  # (N,) (N,)
            score_new = similarity_motif_cluster.sum()
            # Check for convergence: Across steps
            if np.array_equal(membership_old, membership_new):
                break
            # Update best score and membership
            else:
                membership_old = membership_new
                score_old = score_new
                iteration += 1
        # Check for convergence: Across seeds
        if score_new >= global_score:
            global_membership = membership_new
            global_score = score_new

    return global_membership.tolist()
