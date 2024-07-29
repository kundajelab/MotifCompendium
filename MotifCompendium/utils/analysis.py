import h5py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os


#######################
# SIMILARITY ANALYSES #
#######################
def plot_similarity_distribution(mc, save_loc, vals=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05], n_per=5):
	N = len(mc)
	similarity = mc.similarity
	clustering = [False for _ in range(N)]
	for val in vals:
		print(f"--- finding {val} ---")
		indices = np.where((similarity <= val + 0.01) & (similarity >= val - 0.01))
		indices = list(zip(indices[0], indices[1]))
		p = 1
		for i, j in indices:
			if not (clustering[i] or clustering[j]):
				print(f"\t{p}) Found {i} {j} --> {similarity[i, j]}")
				clustering[i] = f"Similarity {val} example {p}"
				clustering[j] = f"Similarity {val} example {p}"
				p += 1
			if p > n_per:
				break
	clustering_series = pd.Series(clustering)
	mc_distribution = mc[clustering_series != False]
	from .plotting import create_html
	create_html(mc_distribution.logos, list(clustering_series[clustering_series != False]), mc_distribution.alignment_fb, mc_distribution.alignment_h, mc_distribution.metadata["name"], save_loc, average=False)


def plot_ground_truth_mismatch(mc, ground_truth, save_loc, similarity_threshold=0.8, max_examples=100, quality=None):
	if quality is None:
		quality = mc.clustering_quality(ground_truth)
		# else assume (for now) that quality was derived from ground truth
	# Log similarity errors
	names, clusters = [], []
	mps = sorted(set(mc[ground_truth]))
	n_examples = 0
	for i, mp in enumerate(mps):
		if quality[i, i] >= similarity_threshold:
			continue
		mp_select = mc[ground_truth] == mp
		similarity_slice_ii_df = mc.get_similarity_slice(mp_select, mp_select, with_names=True)
		similarity_slice_ii_df_stacked = similarity_slice_ii_df.stack()
		row_label, col_label = similarity_slice_ii_df_stacked.idxmin()
		names.append(row_label)
		names.append(col_label)
		clusters += [f"Low internal similarity {mp} ({quality[i, i]:.3})"]*2
		n_examples += 1
		if n_examples >= (max_examples)//2:
			break
	# High external similarity
	n_examples = 0
	for i, mp1 in enumerate(mps):
		for j, mp2 in enumerate(mps):
			if j <= i:
				continue
			if quality[i, j] < similarity_threshold:
				continue
			similarity_slice_ij_df = mc.get_similarity_slice(mc[ground_truth] == mp1, mc[ground_truth] == mp2, with_names=True)
			similarity_slice_ij_df_stacked = similarity_slice_ij_df.stack()
			row_label, col_label = similarity_slice_ij_df_stacked.idxmax()
			names.append(row_label)
			names.append(col_label)
			clusters += [f"High external similarity {mp1} & {mp2} ({quality[i, j]:.3})"]*2
			n_examples += 1
			if n_examples >= (max_examples)//2:
				break
		if n_examples >= (max_examples)//2:
			break
	# Map back and html
	names_revmap = {x: i for i, x in enumerate(list(mc["name"]))}
	idxs = [names_revmap[x] for x in names]
	# Prep for plotting
	html_cwms = mc.logos[idxs, :, :]
	html_sim_fb = mc.alignment_fb[idxs, :][:, idxs]
	html_sim_alignments = mc.alignment_h[idxs, :][:, idxs]
	from .plotting import create_html
	create_html(html_cwms, clusters, html_sim_fb, html_sim_alignments, names, save_loc, average=False, parallel=False)


def judge_clustering(mc, clustering, base_saveloc):
	print("getting quality")
	if os.path.exists(f"{base_saveloc}_quality.npy"):
		clustering_quality = np.load(f"{base_saveloc}_quality.npy")
	else:
		clustering_quality = mc.clustering_quality(clustering)
		np.save(f"{base_saveloc}_quality.npy", clustering_quality)
	print("plotting")
	fig, axs = plt.subplots(2, 1, sharex=True)
	sns.histplot(np.diag(clustering_quality), ax=axs[0], stat="proportion", kde=True)
	axs[0].set_title("worst intra-cluster similarities")
	n_clusters = clustering_quality.shape[0]
	triu = [clustering_quality[i, j] for i in range(n_clusters) for j in range(i+1, n_clusters)]
	sns.histplot(triu, ax=axs[1], stat="proportion", kde=True)
	axs[1].set_title("best inter-cluster similarities")
	axs[1].set_xlabel("similarity")
	plt.suptitle(f"{clustering} ({n_clusters} clusters)")
	plt.savefig(f"{base_saveloc}.png")
	plt.close()


#######################
# DOWNSTREAM ANALYSES #
#######################
def plot_unique_per_cluster(mc, clustering, save_loc):
	clusters = sorted(set(mc[clustering]))
	motif_names = []
	cluster_names = []
	for c in clusters:
		similarity_contrast_c_df = mc.get_similarity_slice(mc[clustering] == c, mc[clustering] != c, with_names=True)
		c_best_similarities = similarity_contrast_c_df.max(axis=1)
		most_unique = c_best_similarities.idxmin()
		most_unique_similarity = c_best_similarities.min()
		motif_names.append(most_unique)
		cluster_names.append(f"{c} ({most_unique_similarity:.3})")
	# Map back and html
	names_revmap = {x: i for i, x in enumerate(list(mc["name"]))}
	idxs = [names_revmap[x] for x in motif_names]
	# Prep for plotting
	html_cwms = mc.logos[idxs, :, :]
	html_sim_fb = mc.alignment_fb[idxs, :][:, idxs]
	html_sim_alignments = mc.alignment_h[idxs, :][:, idxs]
	from .plotting import create_html
	create_html(html_cwms, cluster_names, html_sim_fb, html_sim_alignments, motif_names, save_loc, average=False, parallel=False)


def cluster_grouping_upset_plot(mc, clustering, grouping, save_loc, **kwargs):
	metadata = mc.metadata
	membership_lists = [list(set(metadata[metadata[clustering] == c][grouping])) for c in set(metadata[clustering])]
	import upsetplot
	clusters_by_grouping = upsetplot.from_memberships(membership_lists)
	upsetplot.UpSet(clusters_by_grouping, subset_size="count", **kwargs).plot()
	plt.savefig(save_loc, bbox_inches='tight')
	plt.close()


def export_clusters_modisco(mc, cluster_name, save_loc, **kwargs):
	mc_cluster_avg = mc.cluster_averages(cluster_name, **kwargs) # kwargs for new avg similarity calculations
	pos_neg = np.sum(mc_cluster_avg.logos, axis=(1, 2)) > 0
	pos_neg = ["pos" if x > 0 else "neg" for x in pos_neg]
	mc_cluster_avg.metadata["pos_neg"] = pos_neg
	with h5py.File(save_loc, 'w') as f:
		f.attrs["window_size"] = 30
		# Positive
		if "pos" in mc_cluster_avg["pos_neg"]:
			pos_group = f.create_group("pos_patterns")
			mc_cluster_avg_pos = mc_cluster_avg[mc_cluster_avg["pos_neg"] == "pos"]
			for i in range(len(mc_cluster_avg_pos)):
				name = mc_cluster_avg_pos.loc[i, "name"]
				cwm = mc_cluster_avg_pos.logos[i, :, :]
				pos_cluster = pos_group.create_group(name)
				pos_cluster.create_dataset("contrib_scores", data=cwm)
		# Negative
		if "neg" in mc_cluster_avg["pos_neg"]:
			neg_group = f.create_group("neg_patterns")
			mc_cluster_avg_neg = mc_cluster_avg[mc_cluster_avg["pos_neg"] == "neg"]
			for i in range(len(mc_cluster_avg_neg)):
				name = mc_cluster_avg_neg.loc[i, "name"]
				cwm = mc_cluster_avg_neg.logos[i, :, :]
				neg_cluster = neg_group.create_group(name)
				neg_cluster.create_dataset("contrib_scores", data=cwm)

