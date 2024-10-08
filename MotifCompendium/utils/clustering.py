import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import sklearn.cluster


######################
# CLUSTERING HANDLER #
######################
def cluster(
    similarity_matrix: SimilarityMatrix,
    algorithm: str,
    similarity_threshold: float,
    **kwargs,
) -> list(int):
    """Cluster a similarity matrix.

    Given a similarity matrix, a similarity threshold, and an algorithm choice, return
      a clustering of the similarity matrix. The clustering is a list that assigns each
      index in the similarity matrix to a number number.

    Args:
        similarity_matrix: A square similarity matrix.
        algorithm: A string representing which clustering algorithm to use.
        similarity_threshold: The threshold above which two motifs are considered to be
          similar.
        **kwargs: Any additional arguments to be passed to the clustering algorithm of
          choice.

    Returns:
        A list of integers where each element represents the cluster that index
          corresponds to. All elements with the same value have been assigned to the
          same cluster.

    Notes:
        Please look through the possible clustering algorithm options in this function.
    """
    # Leiden
    if algorithm == "leiden":
        adjacency_matrix = 1 * (similarity_matrix >= similarity_threshold)
        return leiden_clustering(adjacency_matrix)
    elif algorithm == "cpm_leiden":
        adjacency_matrix = 1 * (similarity_matrix >= similarity_threshold)
        kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["resolution"]}
        return cpm_leiden_clustering(adjacency_matrix, **kwargs_filtered)
    # CC
    elif algorithm == "cc":
        adjacency_matrix = similarity_matrix >= similarity_threshold
        return cc_clustering(adjacency_matrix)
    elif algorithm == "greedy_cc":
        adjacency_matrix = similarity_matrix >= similarity_threshold
        return greedy_cc_clustering(adjacency_matrix)
    # Spectral
    elif algorithm == "spectral":
        kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["k", "cluster_qr"]}
        return spectral_clustering_k(similarity_matrix, **kwargs_filtered)


#####################
# LEIDEN CLUSTERING #
#####################
def leiden_clustering(adjacency_matrix):
    g = ig.Graph.Adjacency(adjacency_matrix)
    partition = la.find_partition(g, la.ModularityVertexPartition)
    return partition.membership


def cpm_leiden_clustering(adjacency_matrix, resolution=0.8):
    g = ig.Graph.Adjacency(adjacency_matrix)
    partition = la.find_partition(
        g, la.CPMVertexPartition, resolution_parameter=resolution
    )
    return partition.membership


#######################
# SPECTRAL CLUSTERING #
#######################
def spectral_clustering_k(
    similarity_matrix, k=40, cluster_qr=False
):
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


#################
# CC CLUSTERING #
#################
def cc_clustering(adjacency_matrix):
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


def greedy_cc_clustering(adjacency_matrix):
    # FINDING GREEDY DENSELY-CONNECTED COMPONENTS
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