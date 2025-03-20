import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.cluster


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
    match algorithm:
        # Leiden
        case "leiden":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return leiden_clustering(weighted_adjacency_matrix, **kwargs)
        case "cpm" | "cpm_leiden":
            weighted_adjacency_matrix = similarity_matrix * (
                similarity_matrix >= similarity_threshold
            )
            return cpm_leiden_clustering(weighted_adjacency_matrix, **kwargs)
        # Connected-component
        case "cc":
            adjacency_matrix = similarity_matrix >= similarity_threshold
            return cc_clustering(adjacency_matrix)
        case "dense_cc":
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
def leiden_clustering(
    weighted_adjacency_matrix: np.ndarray,
    resolution_parameter: float = 1.0,
    n_iterations: int = -1,
    seeds: list[int] = [1]
) -> list[int]:
    """Perform Leiden clustering with R&B's quality and a configuration null.
    
    Args:
        weighted_adjacency_matrix: A square weighted adjacency matrix.
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
    g = ig.Graph.Weighted_Adjacency(weighted_adjacency_matrix, mode="undirected")
    best_quality = None
    best_membership = None
    for seed in len(seeds):
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
    seeds: list[int] = [1]
) -> list[int]:
    """Perform Leiden clustering with the constant Potts model.
    
    Args:
        weighted_adjacency_matrix: A square weighted adjacency matrix.
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
    g = ig.Graph.Weighted_Adjacency(weighted_adjacency_matrix, mode="undirected")
    best_quality = None
    best_membership = None
    for seed in len(seeds):
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


#################
# CC CLUSTERING #
#################
def cc_clustering(adjacency_matrix: np.ndarray) -> list[int]:
    """Find connected components in an adjacency matrix."""
    N = adjacency_matrix.shape[0]
    clustering = [False] * N
    current_cluster = 1
    for current_node in range(N):
        if not clustering[current_node]:
            # create new cluster
            clustering[current_node] = current_cluster
            new_cluster_idxs = [current_node]
            for consider_node in range(current_node + 1, N):
                if adjacency_matrix[consider_node, new_cluster_idxs].any():
                    clustering[consider_node] = current_cluster
                    new_cluster_idxs.append(consider_node)
            current_cluster += 1
    print(f"found {current_cluster} clusters in {N} nodes")
    for i in range(N):
        assert clustering[i]
    return [x - 1 for x in clustering]


def densely_cc_clustering(adjacency_matrix: np.ndarray) -> list[int]:
    """Find densely connected components in a greedy fashion in an adjacency matrix."""
    N = adjacency_matrix.shape[0]
    clustering = [False] * N
    current_cluster = 1
    for current_node in range(N):
        if not clustering[current_node]:
            # create new cluster
            clustering[current_node] = current_cluster
            new_cluster_idxs = [current_node]
            for consider_node in range(current_node + 1, N):
                if adjacency_matrix[consider_node, new_cluster_idxs].all():
                    clustering[consider_node] = current_cluster
                    new_cluster_idxs.append(consider_node)
            current_cluster += 1
    print(f"found {current_cluster} clusters in {N} nodes")
    for i in range(N):
        assert clustering[i]
    return [x - 1 for x in clustering]


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
