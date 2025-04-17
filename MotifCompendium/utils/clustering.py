import igraph as ig
import leidenalg as la
import numpy as np
import scipy.sparse
import sklearn.cluster

from MotifCompendium.utils.config import get_use_gpu


######################
# CLUSTERING HANDLER #
######################
def cluster(
    similarity_matrix: np.ndarray,
    similarity_threshold: float,
    algorithm: str,
    **kwargs,
) -> list[int]:
    """Cluster a similarity matrix.

    Given a similarity matrix, a similarity threshold, and a choice of algorithm, return
      a clustering of the similarity matrix. The clustering is a list that assigns each
      index in the similarity matrix to a number number.

    Args:
        similarity_matrix: A square similarity matrix.
        similarity_threshold: The threshold above which two motifs are considered to be
          similar.
        algorithm: Which clustering algorithm to use. Supported options:
            - "leiden": Leiden clustering with Reichardt & Bornholdt's quality function
                and a configuration model as a null.
            - "cpm" or "cpm_leiden": Leiden clustering with the constant Potts model.
            - "cc": Connected-component clustering.
            - "dense_cc": Densely connected-component clustering.
            - "spectral": Spectral clustering.
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
        case "mod_leiden" | "modularity_leiden" | "rb_leiden":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            if get_use_gpu():
                print("Warning: GPU version not yet implemented. Falling back to CPU.")
                return rb_leiden_clustering_cpu(weighted_adjacency_matrix, **kwargs)
            else:
                return rb_leiden_clustering_cpu(weighted_adjacency_matrix, **kwargs)
        case "cpm" | "cpm_leiden" | "constant_potts_leiden":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return cpm_leiden_clustering(weighted_adjacency_matrix, **kwargs)
        case "leiden" | "leidenalg":
            print(
                "Warning: Falling back to default Leiden algorithm: constant Potts model"
            )
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return cpm_leiden_clustering(weighted_adjacency_matrix, **kwargs)
        # Connected-components
        case "cc" | "connected_components":
            adjacency_matrix = similarity_matrix >= similarity_threshold
            return cc_clustering(adjacency_matrix)
        case "dcc" | "dense_cc":
            adjacency_matrix = similarity_matrix >= similarity_threshold
            return densely_cc_clustering(adjacency_matrix, **kwargs)
        # Spectral
        case "spectral":
            return spectral_clustering_k(similarity_matrix, **kwargs)
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
        seeds: Seeds with which to run Leiden. Each seed will corresponding to an
          independent run of Leiden. The clustering from the run with the highest
          quailty will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses Reichardt & Bornholdt's quality function with a configuration model as a
          a null. See leidenalg.RBConfigurationVertexPartition for more details.
    """
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
    seeds: list[int] = [100, 200],
) -> list[int]:
    """Perform Leiden clustering with the constant Potts model.

    Args:
        weighted_adjacency_matrix : A square weighted adjacency matrix.
        resolution_parameter: The resolution parameter for the Leiden algorithm.
        n_iterations: The number of iterations that Leiden is allowed to run. If
          n_iterations is -1, then Leiden will run until there is no longer any
          improvement in quality.
        seeds: Seeds with which to run Leiden. Each seed will corresponding to an
          independent run of Leiden. The clustering from the run with the highest
          quailty will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses the constant Potts model. See leidenalg.CPMVertexPartition for more details.
    """
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
            seed=seed,
        )
        if best_quality is None or partition.quality() > best_quality:
            best_quality = partition.quality()
            best_membership = partition.membership
    return best_membership


def modularity_leiden_clustering_gpu(
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
        seeds: Seeds with which to run Leiden. Each seed will corresponding to an
          independent run of Leiden. The clustering from the run with the highest
          quailty will be returned. The length of seeds is equal to the number of
          runs of Leiden that are performed.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Uses Leiden method described in: Traag, V. A., Waltman, L., & van Eck, N. J. (2019).
        From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports,
        9(1), 5233. doi: 10.1038/s41598-019-41695-z. See cugraph.leiden for more details.
    """
    ValueError("GPU version not yet implemented.")
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
def cc_clustering(adjacency_matrix: np.ndarray) -> list[int]:
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
    adjacency_matrix: np.ndarray, density: float = 1.0, seed: int = 1
) -> list[int]:
    """Find densely connected components in a greedy fashion in an adjacency matrix."""
    if not (isinstance(density, float) and (0 <= density <= 1)):
        raise ValueError("Density must be a float between 0 and 1.")
    N = adjacency_matrix.shape[0]
    clustering = np.zeros((N,), dtype=int)
    current_cluster = 1
    # Set random order of nodes
    node_order = (
        np.arange(N) if seed == 0 else np.random.default_rng(seed).permutation(N)
    )  # Don't shuffle ordering on seed of 0
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
    similarity_matrix: np.ndarray, k: int = 40, cluster_qr: bool = False
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
