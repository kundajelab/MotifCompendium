import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial
import seaborn as sns

import os

import MotifCompendium
import utils_plotting


###
# CODE START
###
file_name = "selin_compendium_clustered.mc"
mc = MotifCompendium.load(file_name)

scores = np.load("quality_merged_pattern.npy")

sns.heatmap(scores)
plt.show()

internal_scores = np.diag(scores)
min_internal_score = np.min(internal_scores)
print(min_internal_score)
plt.hist(internal_scores)
plt.show()

print("--- low internal scores ---")
for i, mp in enumerate(mps):
	if internal_scores[i] >= 0.8:
		continue
	similarity_slice_ii_df = z.get_similarity_slice(z["merged_pattern"] == mp, z["merged_pattern"] == mp, with_names=True)
	similarity_slice_ii_df_stacked = similarity_slice_ii_df.stack()
	row_label, col_label = similarity_slice_ii_df_stacked.idxmin()
	print(f"{mp}: {row_label} {col_label} {internal_scores[i]}")
	# z[z["merged_pattern"] == mp].create_html(f"/oak/stanford/groups/akundaje/salil512/web/temp/modisco_clustering_tests/{mp}.html")

for i, mp1 in enumerate(mps):
	for j, mp2 in enumerate(mps):
		if not (mp1 == test_match_1 and mp2 == test_match_2):
			continue
		similarity_slice_ij_df = z.get_similarity_slice(z["merged_pattern"] == mp1, z["merged_pattern"] == mp2, with_names=True)
		similarity_slice_ij_df_stacked = similarity_slice_ij_df.stack()
		row_label, col_label = similarity_slice_ij_df_stacked.idxmax()
		print(f"{mp1} ({row_label}) {mp2} ({col_label}): {scores[i, j]}")
		assert(False)

input("")

print("--- high external scores ---")
for i, mp1 in enumerate(mps):
	for j, mp2 in enumerate(mps):
		if j <= i:
			continue
		# if scores[i, j] > min_internal_score:
		if scores[i, j] > 0.8:
			similarity_slice_ij_df = z.get_similarity_slice(z["merged_pattern"] == mp1, z["merged_pattern"] == mp2, with_names=True)
			similarity_slice_ij_df_stacked = similarity_slice_ij_df.stack()
			row_label, col_label = similarity_slice_ij_df_stacked.idxmax()
			print(f"{mp1} ({row_label}) {mp2} ({col_label}): {scores[i, j]}")
			input("")

