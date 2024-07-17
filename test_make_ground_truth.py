import pandas as pd

import os

import MotifCompendium


###
# CODE START
###
allowed_modiscos = pd.read_csv("/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/01-models/qc/chrombpnet_models_keep2.tsv", sep="\t", names=["cluster", "folds", "name"])
allowed_modiscos_set = set(allowed_modiscos["cluster"])
modisco_dir = "/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/01-models/modisco/bias_Heart_c0_thresh0.4"
modisco_files = {cluster: f"{modisco_dir}/{cluster}/counts_modisco_output.h5" for cluster in os.listdir(modisco_dir) if cluster in allowed_modiscos_set}

mc = MotifCompendium.build_from_modisco(modisco_files, max_chunk=1000, use_gpu=True)

ground_truth = pd.read_csv("/oak/stanford/groups/akundaje/sjessa/projects/HDMA/chrombpnet/output/02-meta/modisco_compiled_anno/modisco_merged_reports.tsv", sep="\t")
ground_truth["name"] = ground_truth["component_celltype"] + "-" + ground_truth["pattern_class"].str.split("_").str[0] + "." + ground_truth["pattern"]
ground_truth_name = list(ground_truth["name"])
ground_truth_merged_pattern = list(ground_truth["merged_pattern"])
ground_truth_map = {ground_truth_name[i]: ground_truth_merged_pattern[i] for i in range(len(ground_truth_name))}
mc.metadata["merged_pattern"] = [ground_truth_map[x] for x in list(mc["name"])]

selin_annotations = pd.read_csv("selin_annotations.tsv", sep="\t")
selin_annotations_merged_pattern = list(selin_annotations["pattern"])
selin_annotations_annotation = list(selin_annotations["annotation"])
selin_annotations_map = {selin_annotations_merged_pattern[i]: selin_annotations_annotation[i] for i in range(len(selin_annotations))}
mc.metadata["annotation"] = [selin_annotations_map[x] for x in list(mc["merged_pattern"])]

print(mc)
mc.save("selin_compendium.mc")

