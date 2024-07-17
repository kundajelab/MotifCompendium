import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os

import MotifCompendium

def judge_clustering(mc, name):
	print("getting quality")
	if os.path.exists(f"quality_{name}.npy"):
		clustering_quality = np.load(f"quality_{name}.npy")
	else:
		clustering_quality = mc.clustering_quality(name)
		np.save(f"quality_{name}.npy", clustering_quality)
	print("plotting")
	fig, axs = plt.subplots(2, 1, sharex=True)
	sns.histplot(np.diag(clustering_quality), ax=axs[0], stat="proportion", kde=True)
	axs[0].set_title("worst intra-cluster similarities")
	n_clusters = clustering_quality.shape[0]
	triu = [clustering_quality[i, j] for i in range(n_clusters) for j in range(i+1, n_clusters)]
	sns.histplot(triu, ax=axs[1], stat="proportion", kde=True)
	axs[1].set_title("best inter-cluster similarities")
	axs[1].set_xlabel("similarity")
	plt.suptitle(f"{name} ({n_clusters} clusters)")
	plt.savefig(f"{name}.png")
	plt.close()

###
# CODE START
###
'''
print("loading")
mc = MotifCompendium.load("selin_compendium.mc")
print(mc)

print("clustering")
mc.cluster(algorithm="leiden", save_name="leiden")
mc.cluster(algorithm="leiden_step", save_name="leiden_step")
# mc.cluster(algorithm="leiden_weights", save_name="leiden_weights")
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
	mc.cluster(algorithm="cpm_leiden", save_name=f"cpm_leiden_{r}")
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
	mc.cluster(algorithm="cpm_leiden_step", save_name=f"cpm_leiden_step_{r}")
# for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
# 	mc.cluster(algorithm="cpm_leiden_weights", save_name=f"cpm_leiden_weights_{r}")
mc.cluster(algorithm="cc", save_name="cc")
print(mc)
mc.save("selin_compendium_clustered.mc")
'''
print("judging")
mc = MotifCompendium.load("selin_compendium_clustered.mc")
judge_clustering(mc, "merged_pattern")
judge_clustering(mc, "annotation")
judge_clustering(mc, "leiden")
judge_clustering(mc, "leiden_step")
# judge_clustering(mc, "leiden_weights")
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
	judge_clustering(mc, f"cpm_leiden_{r}")
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
	judge_clustering(mc, f"cpm_leiden_step_{r}")
# for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
# 	judge_clustering(mc, f"cpm_leiden_weights_{r}")
judge_clustering(mc, "cc")

