import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import sklearn.cluster


######################
# CLUSTERING HANDLER #
######################
def cluster(
    similarity_matrix: np.ndarray,
    algorithm: str,
    similarity_threshold: float = 0.9,
    **kwargs,
    ) -> list[int]:
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
    if algorithm in ["leiden", "weighted_leiden", "cpm_leiden"]:
        return run_leiden_clustering(similarity_matrix, algorithm, **kwargs_filtered)

    # CC
    elif algorithm == "cc":
        adjacency_matrix = similarity_matrix >= similarity_threshold
        return cc_clustering(adjacency_matrix)
    elif algorithm == "greedy_cc":
        adjacency_matrix = similarity_matrix >= similarity_threshold
        return greedy_cc_clustering(adjacency_matrix)
    
    # Spectral
    elif algorithm == "spectral":
        kwargs_filtered = {
            k: v for k, v in enumerate(kwargs) if k in ["k", "cluster_qr"]
        }
        return spectral_clustering_k(similarity_matrix, **kwargs_filtered)
    
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

#####################
# LEIDEN CLUSTERING #
#####################

def create_sparse_csr(similarity_matrix, similarity_threshold=0.5):
	"""Create a sparse version of the similarity matrix, 
      by filtering out scores below a similarity_threshold"""
	sparse_similarity_matrix = sparse.csr_matrix(np.where(similarity_matrix >= similarity_threshold, similarity_matrix, 0))
	return sparse_similarity_matrix

def run_leiden_clustering(similarity_matrix: np.ndarray, 
    algorithm: str,
    similarity_threshold: float,
    n_seeds: int = 2, 
    n_iterations: int = -1, 
    resolution_parameter: float = 1.0, 
    **kwargs
    ) -> np.ndarray:
	"""Run Leiden clustering.

    Types of Leiden clustering:
        (1) Standard Leiden clustering: Modularity, Resolution parameter = 1.0
        (2) Weighted Leiden clustering: Modularity, Resolution parameter enabled
        (3) CPM Leiden clustering: CPM Quality, Resolution parameter enabled

    Args:
        similarity_matrix (np.ndarray): A square similarity matrix.
        algorithm (str): A string representing which clustering algorithm to use.
        similarity_threshold (float): The threshold above which two motifs are considered to be
          similar.
        n_seeds (int, optional): Number of seeds
        n_iterations (int, optional): Number of clustering iterations
        resolution_parameter (float, optional)
        **kwargs: Any additional arguments to be passed to the clustering algorithm of
          choice.
        
    Returns:
        A np.ndarray of memberships
        """
    # Open kwargs
    for key, value in kwargs.items():
        if key == "n_seeds":
            n_seeds = value
        if key == "n_iterations":
            n_iterations = value
        if key == "resolution":
            resolution_parameter = value
    
    # Convert similarity matrix into sparse csr matrix
    sparse_similarity_matrix = create_sparse_csr(similarity_matrix, similarity_threshold)

    # Convert sparse matrix to edge list
	n_vertices = sparse_similarity_matrix.shape[0]  # Assuming square similarity matrix
	rows, cols = sparse_similarity_matrix.nonzero() # Construct edges from non-zero entries
	edges = list(zip(rows, cols))
	weights = sparse_similarity_matrix.data

	# Create igraph object
	g = ig.Graph(n_vertices, edges=edges)
	g.es['weight'] = weights # Assign edge attribute

	best_quality = None
	best_membership = None

    for seed in range(1, n_seeds+1):
        if algorithm == "leiden":
            partition = leiden_clustering(g, n_iterations, seed)
        elif algorithm == "weighted_leiden":
            partition = weighted_leiden_clustering(g, partition_type, resolution_parameter, n_iterations, seed)
        elif algorithm == "cpm_leiden":
            partition = cpm_leiden_clustering(g, partition_type, resolution_parameter, n_iterations, seed)
        else:
            raise ValueError(f"Unsupported Leiden clustering type: {partition_type}")

		quality = partition.quality()
		membership = np.array(partition.membership)
		
		if best_quality is None or quality > best_quality:
			best_quality = quality
			best_membership = membership

	return best_membership

def leiden_clustering(graph: ig.Graph, 
    n_iterations: int, 
    seed: int
    ):
    """Standard Leiden clustering: Modularity, Resolution parameter = 1.0"""
    partition_type = la.ModularityVertexPartition
    partition = la.find_partition(
            graph=g,
            partition_type=partition_type,
            weights='weight', # Use edge attribute of graph
            n_iterations=n_iterations,
            seed=seed*100)
    return partition

def weighted_leiden_clustering(g: ig.Graph, 
    resolution_parameter: float, 
    n_iterations: int,
    seed: int
    ):
    """Weighted Leiden clustering: Modularity, Resolution parameter enabled"""
    partition_type = la.RBConfigurationVertexPartition
    partition = la.find_partition(
        graph=g,
        partition_type=partition_type,
        weights='weight', # Use edge attribute of graph
        resolution_parameter=resolution_parameter,
        n_iterations=n_iterations,
        seed=seed*100)
    return partition.membership

def cpm_leiden_clustering(g: ig.Graph, 
    resolution_parameter: float, 
    n_iterations: int,
    seed: int
    ):
    """Constant Potts Model (CPM) Leiden clustering: CPM Qualty, Resolution parameter enabled"""
    partition_type = la.CPMVertexPartition
    partition = la.find_partition(
        graph=g,
        partition_type=partition_type,
        weights='weight', # Use edge attribute of graph
        resolution_parameter=resolution_parameter,
        n_iterations=n_iterations,
        seed=seed*100)
    return partition.membership


#######################
# SPECTRAL CLUSTERING #
#######################
def spectral_clustering_k(similarity_matrix, k=40, cluster_qr=False):
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
