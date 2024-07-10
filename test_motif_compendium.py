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

# ADULT HEART
'''
modisco_dir = "/oak/stanford/groups/akundaje/salil512/adult_heart/model_training/modisco"
modisco_files = {pseudobulk: f"{modisco_dir}/{pseudobulk}/modisco_output.h5" for pseudobulk in os.listdir(modisco_dir)}
output_loc = f"/oak/stanford/groups/akundaje/salil512/web/temp/adult_heart_modisco_clustering"
'''

# ADULT HEART SUBSET
'''
modisco_dir = "/oak/stanford/groups/akundaje/salil512/adult_heart/model_training/modisco"
modisco_files = {pseudobulk: f"{modisco_dir}/{pseudobulk}/modisco_output.h5" for pseudobulk in os.listdir(modisco_dir)}
modisco_files = {x: modisco_files[x] for x in ["cardiomyocyte_all", "fibroblast_all"]}
output_loc = f"/oak/stanford/groups/akundaje/salil512/web/temp/adult_heart_subset_modisco_clustering"
'''

# CARDIOID
'''
modisco_dir = "/oak/stanford/groups/akundaje/ryanzhao/cardioid/modisco"
modisco_files = {pseudobulk: f"{modisco_dir}/{pseudobulk}/modisco_output.h5" for pseudobulk in os.listdir(modisco_dir)}
output_loc = f"/oak/stanford/groups/akundaje/salil512/web/temp/cardioid_modisco_clustering"
'''

# VIVEK
'''
modisco_dir = "/oak/stanford/groups/akundaje/vir/tfatlas/modisco/release_run_1/meanshap/ENCSR000EGX"
modisco_files = {"ENCSR000EGX_profile": f"{modisco_dir}/profile/modisco_results.h5", "ENCSR000EGX_counts": f"{modisco_dir}/counts/modisco_results.h5"}
output_loc = f"/oak/stanford/groups/akundaje/salil512/web/temp/vivek_modisco_clustering"
'''

# ZIWEI
'''
modisco_dir = "/oak/stanford/groups/akundaje/projects/CRC_finemap/output/7_29_2023_shap_scores/"
modisco_files = {pseudobulk: f"{modisco_dir}/{pseudobulk}/count_modisco.50kseqlets.h5" for pseudobulk in os.listdir(modisco_dir) if pseudobulk not in [".interpret.args.json", ".htaccess", "primary_colon_bias", "TA1_bias", "primary_colon_TA1_bias", "primary_colon_K562_bias", "primary_colon", "Unknown"]}
output_loc = f"/oak/stanford/groups/akundaje/salil512/web/temp/ziwei_modisco_clustering"
'''

# SELIN
# '''
allowed_modiscos = pd.read_csv("/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/01-models/qc/chrombpnet_models_keep2.tsv", sep="\t", names=["cluster", "folds", "name"])
allowed_modiscos_set = set(allowed_modiscos["cluster"])
modisco_dir = "/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/01-models/modisco/bias_Heart_c0_thresh0.4"
modisco_files = {cluster: f"{modisco_dir}/{cluster}/counts_modisco_output.h5" for cluster in os.listdir(modisco_dir) if cluster in allowed_modiscos_set}
# '''

# MotifCompendium.build_from_modisco(modisco_files, max_chunk=1000, use_gpu=True)
# MotifCompendium.build_from_modisco(modisco_files, max_chunk=1000, max_parallel=2, use_gpu=True)
# MotifCompendium.build_from_modisco(modisco_files, max_chunk=500, use_gpu=True)
# MotifCompendium.build_from_modisco(modisco_files, max_chunk=500, max_parallel=2, use_gpu=True)


# z = MotifCompendium.build_from_modisco(modisco_files, max_chunk=200, max_parallel=8, use_gpu=True)
# z1 = MotifCompendium.build_from_modisco(modisco_files, max_chunk=1000, max_parallel=2, use_gpu=True)
# z2 = MotifCompendium.build_from_modisco(modsco_files, max_chunk=1000, use_gpu=True)
# print(z1 == z2)
assert(False)

for i in range(1, 6):
	print(f"--- {10*i} ---")
	MotifCompendium.build_from_modisco(modisco_files, max_chunk=10*i, max_parallel=32)
assert(False)
z2 = MotifCompendium.build_from_modisco(modisco_files, max_chunk=30)
print(z1 == z2)
assert(False)

file_name = "selin_compendium.mc"

z = MotifCompendium.load(file_name)
assert(False)

'''
z = MotifCompendium.build_from_modisco(modisco_files)
print(z.similarity)
z.save(file_name)
assert(False)
'''

'''
z = MotifCompendium.load(file_name)

z.metadata["celltype"] = z.metadata["name"].str.split("-").str[0]
z.metadata["organ"] = z.metadata["celltype"].str.split("_").str[0]
z.metadata["posneg"] = z.metadata["name"].str.split("-").str[1].str.split(".").str[0]

ground_truth = pd.read_csv("/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/02-meta/modisco_compiled_anno/modisco_merged_reports.tsv", sep="\t")
ground_truth["name"] = ground_truth["component_celltype"] + "-" + ground_truth["pattern_class"].str.split("_").str[0] + "." + ground_truth["pattern"]
ground_truth_name = list(ground_truth["name"])
ground_truth_merged_pattern = list(ground_truth["merged_pattern"])
ground_truth_map = {ground_truth_name[i]: ground_truth_merged_pattern[i] for i in range(len(ground_truth_name))}
z.metadata["merged_pattern"] = [ground_truth_map[x] for x in list(z.metadata["name"])]
z.save(file_name)
assert(False)
'''

'''
z = MotifCompendium.load(file_name)
print(z)
print(z.sims.shape, z.logos.shape, z.similarity.shape, z.alignment_fb.shape, z.alignment_h.shape)
z_muscle = z[z["organ"] == "Muscle"]
print(z_muscle)
print(z_muscle.sims.shape, z_muscle.logos.shape, z_muscle.similarity.shape, z_muscle.alignment_fb.shape, z_muscle.alignment_h.shape)
print(z)
print(z.sims.shape, z.logos.shape, z.similarity.shape, z.alignment_fb.shape, z.alignment_h.shape)
assert(False)
'''

'''
z = MotifCompendium.load(file_name)
mps = sorted(set(z["merged_pattern"]))
scores = np.zeros((len(mps), len(mps)))
for i, mp1 in enumerate(mps):
	for j, mp2 in enumerate(mps):
		if j < i:
			continue
		similarity_slice_ij = z.get_similarity_slice(z["merged_pattern"] == mp1, z["merged_pattern"] == mp2)
		if i == j:
			scores[i, j] = np.min(similarity_slice_ij)
		else:
			scores[i, j] = np.max(similarity_slice_ij)
			scores[j, i] = np.max(similarity_slice_ij)
df = pd.DataFrame(scores, index=mps, columns=mps)
np.save("selin_cutoff_merged_pattern_similarity.npy", scores)
sns.heatmap(df)
plt.show()
assert(False)
'''

# '''
z = MotifCompendium.load(file_name)
z_select = z[z["name"].isin(["Eye_c13-neg.pattern_0", "Heart_c13-neg.pattern_1"])]
print(z_select.similarity)
print(z_select.alignment_fb)
print(z_select.alignment_h)
eye_cwm = z_select.logos[0, :, :]
heart_cwm =  z_select.logos[1, :, :][::-1, ::-1]

np.set_printoptions(linewidth=400, precision=5)
eye_sim = z_select.sims[0, :, :]
heart_sim =  z_select.sims[1, :, :][::-1, ::-1]
print(eye_sim)
print(heart_sim)
# eye_sim[eye_sim < 1/240] = 0
# eye_sim /= np.sum(eye_sim)
# heart_sim[heart_sim < 1/240] = 0
# heart_sim /= np.sum(heart_sim)
# print(eye_sim)
# print(heart_sim)
product = np.sqrt(eye_sim)*np.sqrt(heart_sim)
print(product)
product_per_pos = np.sum(product, axis=1)
print(product_per_pos)
total_sum = np.sum(product_per_pos)
print(total_sum)

utils_plotting.plot_motifs([eye_cwm, heart_cwm])
assert(False)
# '''

z = MotifCompendium.load(file_name)
z.metadata["cluster"] = z.metadata["merged_pattern"]
mps = sorted(set(z["merged_pattern"]))
scores = np.load("selin_cutoff_merged_pattern_similarity.npy")
df = pd.DataFrame(scores, index=mps, columns=mps)
sns.heatmap(df)
plt.show()

internal_scores = np.diag(scores)
min_internal_score = np.min(internal_scores)
print(min_internal_score)
plt.hist(internal_scores)
plt.show()

assert(False)

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

