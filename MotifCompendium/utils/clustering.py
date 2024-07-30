import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics

import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import random


#####################
# LEIDEN CLUSTERING #
#####################
def leiden_clustering(adjacency_matrix):
	import igraph as ig
	import leidenalg as la
	g = ig.Graph.Adjacency(adjacency_matrix)
	partition = la.find_partition(g, la.ModularityVertexPartition)
	return partition.membership

def cpm_leiden_clustering(adjacency_matrix, resolution=0.8):
	import igraph as ig
	import leidenalg as la
	g = ig.Graph.Adjacency(adjacency_matrix)
	partition = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=resolution)
	return partition.membership

def leiden_clustering_weights(adjacency_matrix, weights):
	import igraph as ig
	import leidenalg as la
	g = ig.Graph.Adjacency(adjacency_matrix)
	partition = la.find_partition(g, la.ModularityVertexPartition, weights=weights)
	return partition.membership

def cpm_leiden_clustering_weights(adjacency_matrix, weights, resolution=0.8):
	import igraph as ig
	import leidenalg as la
	g = ig.Graph.Adjacency(adjacency_matrix)
	partition = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=resolution, weights=weights)
	return partition.membership

#######################
# SPECTRAL CLUSTERING #
#######################
'''
def similarity_asw(similarity_matrix, clustering):
	worst_asw = 1
	for cluster in sorted(set(clustering)):
		c_idxs = [i for i, c in enumerate(clustering) if c == cluster]
		similarity_submatrix = similarity_matrix[c_idxs, :][:, c_idxs]
		c_asw = np.min(similarity_submatrix)
		# print(f"\t\t{cluster}-->{c_asw}")
		if c_asw < worst_asw:
			worst_asw = c_asw
	return worst_asw

def spectral_clustering_k(similarity_matrix, k, traditional_asw=True):
	clustering = sklearn.cluster.SpectralClustering(k, affinity="precomputed").fit(similarity_matrix).labels_
	if traditional_asw:
		asw = sklearn.metrics.silhouette_score(1 - similarity_matrix, clustering, metric="precomputed")
	else:
		asw = similarity_asw(similarity_matrix, clustering)
	return asw, clustering

def spectral_clustering_ks(similarity_matrix, ks):
	best_k = -1
	best_asw = 0
	best_clustering = None
	for k in ks:
		if (k == 1) or (k >= similarity_matrix.shape[0]):
			continue
		asw_k, clustering_k = spectral_clustering_k(similarity_matrix, k)
		if asw_k > best_asw:
			best_k = k
			best_asw = asw_k
			best_clustering = clustering_k
	return best_k, best_asw, best_clustering

def spectral_subclustering(similarity_matrix, cluster_idxs, similarity_threshold):
	similarity_submatrix = similarity_matrix[cluster_idxs, :][:, cluster_idxs]
	if np.min(similarity_submatrix) >= similarity_threshold:
		return [cluster_idxs]
	for k in range(2, len(cluster_idxs)):
		if k >= len(cluster_idxs):
			continue
		asw, clustering = spectral_clustering_k(similarity_submatrix, k, traditional_asw=False)
		if asw >= similarity_threshold:
			return [[cluster_idxs[i] for i, c in enumerate(clustering) if c == cluster] for cluster in sorted(set(clustering))]
	return [[idx] for idx in cluster_idxs]

def merge_clusters(clusters, similarity_matrix, similarity_threshold):
	merged_clusters = []
	cluster_processed = [False] * len(clusters)
	for ci, c in enumerate(clusters):
		if not cluster_processed[ci]:
			cluster_processed[ci] = True
			current_combined = c
			for cj in range(ci+1, len(clusters)):
				if not cluster_processed[cj]:
					compare_cluster = clusters[cj]
					if np.min(similarity_matrix[current_combined, :][:, compare_cluster]) > similarity_threshold:
						current_combined += compare_cluster	
						cluster_processed[cj] = True
			merged_clusters.append(current_combined)
	assert(all(cluster_processed))
	return merged_clusters

def spectral_clustering(similarity_matrix, similarity_threshold=0.8):
	N = similarity_matrix.shape[0]
	# FIND BEST K (COURSE-GRAINED)
	best_k, _, _ = spectral_clustering_ks(similarity_matrix, [5*(i+1) for i in range(60)])
	# FIND BEST K (FINE-GRAINED)
	best_k, best_asw, best_clustering = spectral_clustering_ks(similarity_matrix, [best_k-4, best_k-3, best_k-2, best_k-1, best_k, best_k+1, best_k+2, best_k+3, best_k+4])
	assert(len(best_clustering) == N)
	# CLUSTER
	clusters = [[i for i, x in enumerate(best_clustering) if x == c] for c in set(best_clustering)]
	clusters = [x[1] for x in sorted([(len(c), c) for c in clusters], reverse=True)]
	# SUBCLUSTER
	final_clusters = []
	for c in clusters:
		subcluster_list = spectral_subclustering(similarity_matrix, c, similarity_threshold)
		for sc in subcluster_list:
			final_clusters.append(sc)
	assert(sum([len(x) for x in final_clusters]) == N)

	# RECOMBINE
	final_clusters = merge_clusters(final_clusters, similarity_matrix, similarity_threshold)

	# CHECK RESULTS
	assert(sum([len(x) for x in final_clusters]) == N)
	return final_clusters
'''

def spectral_clustering_k(similarity_matrix, similarity_threshold=0.8, k=40, cluster_qr=False):
	if cluster_qr:
		clustering = sklearn.cluster.SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr").fit(similarity_matrix).labels_
	else:
		clustering = sklearn.cluster.SpectralClustering(k, affinity="precomputed").fit(similarity_matrix).labels_
	return clustering


#################
# CC CLUSTERING #
#################
'''
def connected_components(adjacency_matrix):
	check_error(adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "error: adjacency matrix not square")
	N = adjacency_matrix.shape[0]
	nodes_considered = [False] * N
	ccs = []
	current_node = 0
	while current_node != N:
		if not nodes_considered[current_node]:
			nodes_considered[current_node] = True
			current_cc = [current_node]
			currently_considering = [current_node]
			while len(currently_considering) > 0:
				considering_node = currently_considering.pop(0)
				connected_nodes = adjacency_matrix[considering_node, :].nonzero()[0]
				for i in connected_nodes:
					if not nodes_considered[i]:
						nodes_considered[i] = True
						current_cc.append(i)
						currently_considering.append(i)
			ccs.append(current_cc)
		current_node += 1
	ccs = [x[1] for x in sorted([(len(c), c) for c in ccs], reverse=True)]
	assert(sum([len(x) for x in ccs]) == N)
	return ccs

def cc_clustering(similarity_matrix, similarity_threshold=0.8):
	# COARSE GRAINED CLUSTERING
	N = similarity_matrix.shape[0]
	adjacency_matrix = similarity_matrix > similarity_threshold
	ccs = connected_components(adjacency_matrix)
	# FINE GRAINED CLUSTERING
	final_clusters = []
	for cc in ccs:
		subcluster_list = spectral_subclustering(similarity_matrix, cc, similarity_threshold)
		for sc in subcluster_list:
			final_clusters.append(sc)
	# CHECK RESULTS
	assert(sum([len(x) for x in final_clusters]) == N)
	return final_clusters
'''

def cc_clustering(adjacency_matrix):	
	N = adjacency_matrix.shape[0]
	clustering = [False] * N
	current_cluster = 1
	for current_node in range(N):
		if not clustering[current_node]:
			# create new cluster
			clustering[current_node] = current_cluster
			new_cluster_idxs = [current_node]
			for consider_node in range(current_node+1, N):
				if adjacency_matrix[consider_node, new_cluster_idxs].any():
					clustering[consider_node] = current_cluster
					new_cluster_idxs.append(consider_node)
			current_cluster += 1
	print(f"found {current_cluster} clusters in {N} nodes")
	for i in range(N):
		assert(clustering[i])
	return [x-1 for x in clustering]

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
			for consider_node in range(current_node+1, N):
				if adjacency_matrix[consider_node, new_cluster_idxs].all():
					clustering[consider_node] = current_cluster
					new_cluster_idxs.append(consider_node)
			current_cluster += 1
	print(f"found {current_cluster} clusters in {N} nodes")
	for i in range(N):
		assert(clustering[i])
	return [x-1 for x in clustering]

def _cc_clustering_random(adjacency_matrix, seed):
	# FINDING GREEDY DENSELY-CONNECTED COMPONENTS
	N = adjacency_matrix.shape[0]
	clustering = [False] * N
	current_cluster = 1
	# RANDOM NODE ORDER
	random.seed(seed)
	node_order = random.sample(list(range(N)), k=N)
	# print(f"seed {seed}: {node_order[:5]}")
	# CONTINUE
	for current_node_idx in range(N):
		current_node = node_order[current_node_idx]
		if not clustering[current_node]:
			# create new cluster
			clustering[current_node] = current_cluster
			new_cluster_idxs = [current_node]
			for consider_node_idx in range(current_node_idx+1, N):
				consider_node = node_order[consider_node_idx]
				if adjacency_matrix[consider_node, new_cluster_idxs].all():
					clustering[consider_node] = current_cluster
					new_cluster_idxs.append(consider_node)
			current_cluster += 1
	print(f"found {current_cluster} clusters in {N} nodes")
	for i in range(N):
		assert(clustering[i])
	return [x-1 for x in clustering]

def _clustering_to_adjacency(clustering):
	# create clustering dict
	clustering_dict = dict()
	for i, c in enumerate(clustering):
		if c in clustering_dict:
			clustering_dict[c].append(i)
		else:
			clustering_dict[c] = [i]
	# create adjacency
	N = len(clustering)
	adjacency = np.zeros((N, N))
	# fill adjacency
	for c, idxs in clustering_dict.items():
		for i in idxs:
			for j in idxs:
				adjacency[i, j] = 1	
	assert((adjacency == adjacency.T).all() and np.max(adjacency) == 1)
	return adjacency

def cc_leiden(adjacency_matrix, n_iter=100):
	clustering_adjacency = np.load("clustering_adjacency.npy")
	'''
	clustering_adjacency = np.zeros(adjacency_matrix.shape)
	for i in range(n_iter):
		if i == 0:
			clustering_i = cc_clustering(adjacency_matrix)
		else:
			clustering_i = _cc_clustering_random(adjacency_matrix, i)
		clustering_adjacency += _clustering_to_adjacency(clustering_i)
	'''
	clustering = leiden_clustering(clustering_adjacency)
	print(max(clustering))
	return clustering

def cc_stable(adjacency_matrix, n_iter=100):
	clustering_adjacency = np.load("clustering_adjacency.npy")
	'''
	clustering_adjacency = np.zeros(adjacency_matrix.shape)
	for i in range(n_iter):
		if i == 0:
			clustering_i = cc_clustering(adjacency_matrix)
		else:
			clustering_i = _cc_clustering_random(adjacency_matrix, i)
		clustering_adjacency += _clustering_to_adjacency(clustering_i)
	# np.save("clustering_adjacency.npy", clustering_adjacency)
	assert(False)
	'''
	print(np.count_nonzero(clustering_adjacency))
	sns.histplot(clustering_adjacency[clustering_adjacency > 0])
	plt.show()
	assert(False)
	stable_adjacency = clustering_adjacency >= 5
	return cc_clustering(stable_adjacency)


######################
# CLUSTERING HANDLER #
######################
def cluster(similarity_matrix, algorithm, similarity_threshold, **kwargs):
	# Leiden
	if algorithm == "leiden":
		adjacency_matrix = 1*(similarity_matrix >= similarity_threshold)
		return leiden_clustering(adjacency_matrix)

	elif algorithm == "cpm_leiden":
		adjacency_matrix = 1*(similarity_matrix >= similarity_threshold)
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["resolution"]}
		return cpm_leiden_clustering(adjacency_matrix, **kwargs_filtered)

	elif algorithm == "leiden_step":
		adjacency_matrices = [1*(similarity_matrix >= x) for x in [1, 0.95, 0.9, 0.85, 0.80, 0.75, 0.7]]
		adjacency_matrix = np.sum(adjacency_matrices, axis=0)
		return leiden_clustering(adjacency_matrix)

	elif algorithm == "cpm_leiden_step":
		adjacency_matrices = [1*(similarity_matrix >= x) for x in [1, 0.95, 0.9, 0.85, 0.80, 0.75, 0.7]]
		adjacency_matrix = np.sum(adjacency_matrices, axis=0)
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["resolution"]}
		return cpm_leiden_clustering(adjacency_matrix, **kwargs_filtered)

	elif algorithm == "leiden_weights":
		adjacency_matrix = np.ones(similarity_matrix.shape)
		adjacency_matrix = 1*(similarity_matrix >= similarity_threshold)
		return leiden_clustering_weights(adjacency_matrix, similarity_matrix*adjacency_matrix)

	elif algorithm == "cpm_leiden_weights":
		adjacency_matrices = [1*(similarity_matrix >= x) for x in [1, 0.95, 0.9, 0.85, 0.80, 0.75, 0.7]]
		adjacency_matrix = np.sum(adjacency_matrices, axis=0)
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["resolution"]}
		return cpm_leiden_clustering_weights(adjacency_matrix, similarity_matrix, **kwargs_filtered)

	# CC
	elif algorithm == "cc":	
		adjacency_matrix = similarity_matrix >= similarity_threshold
		return cc_clustering(adjacency_matrix)
	elif algorithm == "greedy_cc":
		adjacency_matrix = similarity_matrix >= similarity_threshold
		return greedy_cc_clustering(adjacency_matrix)
	elif algorithm == "cc_stable":
		print("not yet implemented"); assert(False)
		adjacency_matrix = similarity_matrix >= similarity_threshold
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["n_iters"]}
		return cc_stable_clustering(adjacency_matrix)

	elif algorithm == "cc_leiden":
		print("not yet implemented"); assert(False)
		adjacency_matrix = similarity_matrix >= similarity_threshold
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["n_iters"]}
		return cc_leiden_clustering(adjacency_matrix)

	# Spectral
	elif algorithm == "spectral":
		print("not yet implemented"); assert(False)

	elif algorithm == "spectral_k":
		kwargs_filtered = {k: v for k, v in enumerate(kwargs) if k in ["k", "cluster_qr"]}
		return spectral_clustering_k(similarity_matrix, similarity_threshold, **kwargs)	
	
	else:
		print("that clustering algorithm has not been implemented yet...")
		assert(False)

