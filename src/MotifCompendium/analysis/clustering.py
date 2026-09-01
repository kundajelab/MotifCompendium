import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot

from MotifCompendium import MotifCompendium as MotifCompendiumClass
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif


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
    if (
        isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
    ):
        _, init_membership = np.unique(
            init_membership, return_inverse=True
        )  # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(
            f"init_membership must be a numpy array of integers of length equal to the number of motifs."
        )

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
    one_hot_membership = (memberships[:, None] == np.arange(k)[None, :]).astype(
        float
    )  # (N, k)
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
    if (
        isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
    ):
        _, init_membership = np.unique(
            init_membership, return_inverse=True
        )  # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(
            f"init_membership must be a numpy array of integers of length equal to the number of motifs."
        )

    # Weights
    if weights is None:
        weights = np.ones((N,))
    elif not (
        isinstance(weights, np.ndarray)
        and (weights.ndim == 1)
        and (weights.shape[0] == N)
    ):
        raise ValueError(
            "weights must be a numpy array of floats of length equal to the number of motifs."
        )

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
            k_median_distance[:, c] = np.median(
                similarity_matrix[:, cluster_idxs], axis=1
            )
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
    if (
        isinstance(init_membership, np.ndarray)
        and (init_membership.ndim == 1)
        and (init_membership.shape[0] == N)
    ):
        _, init_membership = np.unique(
            init_membership, return_inverse=True
        )  # Remap to contiguous 0-based indices
        k = len(np.unique(init_membership))  # Set k
        seeds = [None]  # No random seed, run once
        init_method = None  # No initialization method
    elif init_membership is not None:
        raise ValueError(
            f"init_membership must be a numpy array of integers of length equal to the number of motifs."
        )

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
