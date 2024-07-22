import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os


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

