#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: combine the TF, Chromatin and ProCap atlasclusters into ENCODE clusters.

Inputs are the output roots of the three atlas pipelines, each <dir>/{counts,profile,combined}:

  --tfatlas-dir      pipeline/bpnet/pipeline_bpnet.py            (TF atlas)
  --chromatlas-dir   pipeline/chrombpnet/pipeline_chrombpnet.py  (Chromatin atlas)
  --procapatlas-dir  pipeline/procapnet/pipeline_procapnet.py    (ProCap atlas)

The atlasclusters are force-clustered (DCC + k-means, recursively) into ENCODE clusters, which
are annotated, filtered and exported; the results are written back into the atlas pipelines'
directories. pipeline_encode_full.py runs the atlas pipelines and this stage in one process.
"""
# Annotations are lazily evaluated: the helper signatures reference per-atlas
# config classes (configs.MotifFilterArgs) that only exist as profile variants.
from __future__ import annotations

import os
import sys
import time
import re
import argparse
import logging
import gzip
import json
import dataclasses
from argparse import Namespace

import numpy as np
import pandas as pd
import pysam

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config

import importlib.util
# Configs: this script's own configs.py, imported under a name of its own (not "configs"), so that the
# atlas pipelines can be imported side by side (see pipeline/encode/pipeline_encode_full.py)
_configs_spec = importlib.util.spec_from_file_location(
    "configs_encode", os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.py")
)
configs = importlib.util.module_from_spec(_configs_spec)
sys.modules[_configs_spec.name] = configs
_configs_spec.loader.exec_module(configs)



#### ARGUMENTS ---------------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="ENCODE MotifCompendium: combine the TF, Chromatin and ProCap atlasclusters into ENCODE clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory of the ENCODE combine stage.")
    parser.add_argument("--mcid-version", type=str, default="V0", help="ENCODE MCID version.")

    # Input: the atlas pipelines' output roots, each <dir>/{counts,profile,combined}
    parser.add_argument("--tfatlas-dir", type=str, required=True,
                        help="Output root of the TF atlas pipeline (pipeline/bpnet/pipeline_bpnet.py).")
    parser.add_argument("--chromatlas-dir", type=str, required=True,
                        help="Output root of the Chromatin atlas pipeline (pipeline/chrombpnet/pipeline_chrombpnet.py).")
    parser.add_argument("--procapatlas-dir", type=str, required=True,
                        help="Output root of the ProCap atlas pipeline (pipeline/procapnet/pipeline_procapnet.py).")

    # Input: Reference databases
    parser.add_argument("--mc-ref-qc", type=str, required=True,
                        help="ENCODE QC reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-direct", type=str, required=True,
                        help="ENCODE QC + in vitro reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-clustered-hr-metadata", type=str, required=True,
                        help="Metadata TSV of the human-readable clustered reference.")
    parser.add_argument("--ref-invitro", type=str, required=True,
                        help="Reference in-vitro PFM/MotifCompendium.")
    parser.add_argument("--column-desc-tsv", type=str, required=True,
                        help="TSV describing HTML summary table column names.")

    # Input: initial metadata of each atlas (source annotations, hits annotation)
    parser.add_argument("--tf-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the TF atlas.")
    parser.add_argument("--chrom-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the Chromatin atlas.")
    parser.add_argument("--procap-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the ProCap atlas.")

    # Input: QC (each optional: its QC step is skipped without it), curation, hits annotation
    parser.add_argument("--mc-qc-remove-removed", type=str, default=None,
                        help="(Optional) Removed MotifCompendium object (.mc) from the prior QC-remove run.")
    parser.add_argument("--mc-qc-pass-filtered", type=str, default=None,
                        help="(Optional) Filtered MotifCompendium object (.mc) from the prior QC-pass run.")
    parser.add_argument("--mc-qc-pass-lowq", type=str, default=None,
                        help="(Optional) Low-quality MotifCompendium object (.mc) from the prior QC-pass run.")
    parser.add_argument("--metadata-edited", type=str, required=True,
                        help="Manually edited filtered-metadata CSV.")
    parser.add_argument("--hg38-fasta", type=str, required=True,
                        help="hg38 genome FASTA, for hits annotation.")
    parser.add_argument("--gencode-gtf", type=str, required=True,
                        help="GENCODE annotation GTF (.gz), for hits annotation.")

    # Compute
    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000,
                        help="Maximum number of motifs to process at a time. Set to -1 for no chunking.")
    parser.add_argument("--var-chunk", action="store_true",
                        help="Starting from max_chunk, dynamically reduce chunk size based on available memory.")
    parser.add_argument("--unsafe", action="store_false", dest="safe",
                        help="Do not check the validity of the MotifCompendium objects.")
    parser.add_argument("--no-ic", action="store_false", dest="ic",
                        help="Do not scale motifs by information content.")
    parser.add_argument("--fast-plot", action="store_true",
                        help="Reduce the resolution of logo images for faster plotting.")
    parser.add_argument("--export-modisco", action="store_true", help="Also export TF-MoDISco HDF5 files (default: off).")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ---------------------------------------------------------
def set_default_configs():
    """Bind the ENCODE config classes (configs.PROFILES["ENCODE"]) to module globals."""
    global MetadataCols, TFAnnotationCols, CellTypeCols, ReferenceMatchCols, LabelCols
    global MotifMatchArgs, DirectTypeArgs, FilterArgs, MotifFilterArgs
    global ClusterArgs, VisualizeArgs, HitsCols, EncodeCombineArgs, MotifCompendiumArgs

    for name, cls in configs.PROFILES["ENCODE"].items():
        globals()[name] = cls()


#### ARG ALIASES -------------------------------------------------------------
# run_encode_combine() reads args.<ref_name>; this CLI exposes the same inputs under the flag
# names of pipeline_encode_full.py. stage_args() mirrors each one onto the reference name.
ARG_ALIASES = {
    "ref_qc":                     "mc_ref_qc",
    "ref_direct":                 "mc_ref_direct",
    "ref_clustered_hr_metadata":  "mc_ref_clustered_hr_metadata",
    "hg38":                       "hg38_fasta",
    "gencode":                    "gencode_gtf",
    "mc_qc_removed":              "mc_qc_remove_removed",
    "mc_qc_passed":               "mc_qc_pass_filtered",
    "mc_qc_lowq":                 "mc_qc_pass_lowq",
}

#### PATHS / PROVENANCE -----------------------------------------------------
def stage_args(args, **overrides):
    """A per-stage args namespace: the shared CLI args plus stage-specific wiring."""
    d = vars(args).copy()
    d.update(overrides)
    # Mirror this CLI's flag names onto the names the confirmed stage bodies expect
    for ref_name, cli_name in ARG_ALIASES.items():
        if d.get(ref_name) is None and cli_name in d:
            d[ref_name] = d[cli_name]
    return Namespace(**d)


def collect_configs_snapshot():
    """Snapshot every dataclass in configs.py, exactly as instantiated for this run.

    Nothing in the pipeline mutates the config dataclasses at runtime, so instantiating them
    fresh here reproduces the values the run actually used.
    """
    snapshot = {}
    for config_name in sorted(dir(configs)):
        if config_name.startswith("_"):
            continue
        config_cls = getattr(configs, config_name)
        if not (isinstance(config_cls, type) and dataclasses.is_dataclass(config_cls)):
            continue
        try:
            snapshot[config_name] = dataclasses.asdict(config_cls())
        except Exception as exc:  # keep the dump best-effort rather than failing the run
            snapshot[config_name] = f"<could not serialize: {exc}>"
    return snapshot


def collect_previous_args(upstream_dirs):
    """Load the args.json of each upstream run so provenance chains across pipeline stages.

    Args:
        upstream_dirs: iterable of (label, run_output_dir) pairs. Each run's args.json is
          expected at <run_output_dir>/metadata/args.json.
    """
    previous = {}
    for label, run_dir in upstream_dirs:
        if not run_dir:
            continue
        prev_path = os.path.join(run_dir, "metadata", "args.json")
        if not os.path.exists(prev_path):
            previous[label] = f"<not found: {prev_path}>"
            continue
        try:
            with open(prev_path) as prev_handle:
                previous[label] = json.load(prev_handle)
        except Exception as exc:
            previous[label] = f"<could not read {prev_path}: {exc}>"
    return previous


#### FUNCTIONS ---------------------------------------------------------------
def assign_direct_composite(
    target_mc: MotifCompendium.MotifCompendium,
    direct_reference_mc: MotifCompendium.MotifCompendium,
    indirect_reference_mc: MotifCompendium.MotifCompendium,
    min_score: float,
    max_submotifs: int = 1,
    label_unsigned: bool = True,
    save_images: bool = True,
    logo_trimming: bool | float | int = True,
    save_col_prefix: str = "match",
) -> None:
    # Check arguments
    my_motifs = target_mc.motifs.copy() # (N,)
    direct_reference_motifs = direct_reference_mc.motifs.copy() # (M_1,)
    indirect_reference_motifs = indirect_reference_mc.motifs.copy() # (M_2,)
    # Sign: Use absolute values
    if label_unsigned:
        my_motifs = np.abs(my_motifs)
        direct_reference_motifs = np.abs(direct_reference_motifs)
        indirect_reference_motifs = np.abs(indirect_reference_motifs)
    # Length: Resize motifs to match length (L,)
    max_length = max(my_motifs.shape[1], direct_reference_motifs.shape[1], indirect_reference_motifs.shape[1])
    if my_motifs.shape[1] < max_length:
        my_pad = max_length - my_motifs.shape[1]
        my_motifs = np.pad(my_motifs, ((0, 0), (0, my_pad), (0, 0)))
    if direct_reference_motifs.shape[1] < max_length:
        direct_pad = max_length - direct_reference_motifs.shape[1]
        direct_reference_motifs = np.pad(direct_reference_motifs, ((0, 0), (0, direct_pad), (0, 0)))
    if indirect_reference_motifs.shape[1] < max_length:
        indirect_pad = max_length - indirect_reference_motifs.shape[1]
        indirect_reference_motifs = np.pad(indirect_reference_motifs, ((0, 0), (0, indirect_pad), (0, 0)))

    # Save raw references before IC-scaling and L2 normalization
    direct_reference_motifs_raw = direct_reference_motifs
    indirect_reference_motifs_raw = indirect_reference_motifs

    # Initial: IC-scale (only once)
    if utils_config.get_ic_scale():
        ic_scale = True
        my_motifs = utils_motif.ic_scale(my_motifs)
        direct_reference_motifs = utils_motif.ic_scale(direct_reference_motifs)
        indirect_reference_motifs = utils_motif.ic_scale(indirect_reference_motifs)
        utils_config.set_ic_scale(False)  # Turn off IC scale for rest of function
    else:
        ic_scale = False

    # Initial: L2 normalize motifs (comparable scales)
    my_motifs_norm = np.linalg.norm(my_motifs, axis=(1, 2), keepdims=True)
    my_motifs = np.divide(my_motifs, my_motifs_norm, where=my_motifs_norm!=0)
    
    direct_reference_motifs_norm = np.linalg.norm(direct_reference_motifs, axis=(1, 2), keepdims=True)
    direct_reference_motifs = np.divide(direct_reference_motifs, direct_reference_motifs_norm,
              where=direct_reference_motifs_norm!=0)
    
    indirect_reference_motifs_norm = np.linalg.norm(indirect_reference_motifs, axis=(1, 2), keepdims=True)
    indirect_reference_motifs = np.divide(indirect_reference_motifs, indirect_reference_motifs_norm,
              where=indirect_reference_motifs_norm!=0)

    # Combine direct + indirect reference
    reference_motifs = np.concatenate(
        [direct_reference_motifs, indirect_reference_motifs], axis=0
    )  # (M_1 + M_2, L, 4)
    reference_motifs_raw = np.concatenate(
        [direct_reference_motifs_raw, indirect_reference_motifs_raw], axis=0
    )  # (M_1 + M_2, L, 4)
    reference_metadata_name_list = (
        direct_reference_mc.metadata[MetadataCols.name_col].tolist() + indirect_reference_mc.metadata[MetadataCols.name_col].tolist()
    )

    # Find best match, per iteration
    match_scores = []
    match_labels = []
    match_motifs = []
    match_idxs = []
    match_mask = np.ones((my_motifs.shape[0],), dtype=bool)  # (N,)

    ## Direct match:
    # Compute similarity
    sim, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [my_motifs, direct_reference_motifs], [(0, 1)]
    )[0]

    # Scale
    my_motifs_norm = np.linalg.norm(my_motifs, axis=(1, 2))[:, np.newaxis]
    direct_reference_motifs_norm = np.linalg.norm(direct_reference_motifs, axis=(1, 2))[np.newaxis, :]
    sim = sim * (my_motifs_norm * direct_reference_motifs_norm)  # unscale L2 norm
    
    # Identify matches, Scale score by i
    match_score = np.max(sim, axis=1) * np.sqrt(1)  # (N,)
    match_idx = np.argmax(sim, axis=1)  # (N,)
    alignment_rc = alignment_rc[
        np.arange(alignment_rc.shape[0]), match_idx
    ]  # (N,)
    alignment_h = alignment_h[
        np.arange(alignment_h.shape[0]), match_idx
    ]  # (N,)
    match_motif = direct_reference_motifs_raw[match_idx]
    # Remove matches below threshold
    match_mask = match_mask & (match_score >= min_score)  # (N,)
    match_score[~match_mask] = 0
    match_idx[~match_mask] = -1
    match_motif[~match_mask] = 0
    # Subtract best match
    my_motifs = utils_motif.remove_motif_component(
        my_motifs,
        match_motif,
        alignment_rc,
        alignment_h,
    )
    # Remove motifs with no match
    my_motifs[~match_mask] = 0
    # Save match information
    match_label = [direct_reference_mc.metadata[MetadataCols.name_col].tolist()[x] if x >= 0 else None for x in match_idx]
    match_motifs.append(match_motif)
    match_scores.append(match_score)
    match_labels.append(match_label)
    match_idxs.append(match_idx)

    ## Composite match:
    for i in range(1, max_submotifs):
        if len(my_motifs) > 0:
            # Compute similarity
            sim, alignment_rc, alignment_h = utils_similarity.compute_similarities(
                [my_motifs, reference_motifs], [(0, 1)]
            )[0]
            # Unscale L2 norm
            my_motifs_norm = np.linalg.norm(my_motifs, axis=(1, 2))[:, np.newaxis]
            reference_motifs_norm = np.linalg.norm(reference_motifs, axis=(1, 2))[np.newaxis, :]
            sim = sim * (
                my_motifs_norm * reference_motifs_norm
            )  # (N, M)
            
            # Identify matches, Scale score by i
            match_score = np.max(sim, axis=1) * np.sqrt(i + 1)  # (N,)
            match_idx = np.argmax(sim, axis=1)  # (N,)
            alignment_rc = alignment_rc[
                np.arange(alignment_rc.shape[0]), match_idx
            ]  # (N,)
            alignment_h = alignment_h[
                np.arange(alignment_h.shape[0]), match_idx
            ]  # (N,)
            match_motif = reference_motifs_raw[match_idx]  # (N, L, 4)
            
            # Remove matches below threshold
            match_mask = match_mask & (match_score >= min_score)  # (N,)
            match_score[~match_mask] = 0  # (N,)
            match_idx[~match_mask] = -1  # (N,)
            match_motif[~match_mask] = 0  # (N, L, 4)
    
            # Subtract best match
            if match_mask.any():
                my_motifs = utils_motif.remove_motif_component(
                    my_motifs,
                    match_motif,
                    alignment_rc,
                    alignment_h,
                )
                # Remove motifs with no match
                my_motifs[~match_mask] = 0
                # Save match information
                match_label = [reference_metadata_name_list[x] if x >= 0 else None for x in match_idx]
                match_motifs.append(match_motif)
                match_scores.append(match_score)
                match_labels.append(match_label)
                match_idxs.append(match_idx)
            else:
                break
        else:
            break

    # Restore IC-scale setting
    if ic_scale:
        utils_config.set_ic_scale(True)
    
    # Save match information
    for i in range(len(match_scores)):
        target_mc.metadata[f"{save_col_prefix}_score{i}"] = match_scores[i]  # Save scores
        target_mc.metadata[f"{save_col_prefix}_name{i}"] = match_labels[i]  # Save labels
        # Save logos, matches only
        if save_images:
            # Generate forward logos if not provided
            target_mc.add_logos(
                match_motifs[i], f"{save_col_prefix}_logo{i}", logo_trimming
            )


def aggregate_weighted_avg(mc: MotifCompendium, mc_cluster: MotifCompendium, 
        cluster_col: str, agg_col: str, new_col: str, weight_col: str, ) -> None:
    """
    Aggregate the MotifCompendium object by weighted average.
    
    Args:
        mc (MotifCompendium): The sub MotifCompendium object.
        mc_cluster (MotifCompendium): The cluster averaged MotifCompendium object.
        cluster_col (str): The column name of clusterings.
        agg_col (str): The column name to aggregate.
        new_col (str): The column name to save the new weighted average.
        weight_col (str): The column name to use for weights.
        
    Returns:
        None
    """
    weighted_avgs = []
    for cluster_id in mc_cluster["source_cluster"]:
        mc_subset = mc[mc[cluster_col] == cluster_id]
        if len(mc_subset) == 0:
            weighted_avgs.append(np.nan)
            continue
        
        mask = mc_subset[agg_col].notna()
        if not mask.any():
            weighted_avgs.append(np.nan)
            continue
        
        values = mc_subset.metadata.loc[mask, agg_col]
        weights = mc_subset.metadata.loc[mask, weight_col]
        weighted_avg = (values * weights).sum() / weights.sum()
        weighted_avgs.append(weighted_avg)
    
    mc_cluster[new_col] = weighted_avgs


def apply_filter_threshold(
    mc: MotifCompendium,
    flag_col: str,
    flag_type_col: str,
    metric: str,
    operation: str,
    threshold: int | float | bool | str | list | set | tuple,
    invert: bool = False,
    override: bool = False,
    flag_type: str | None = None,
) -> None:
    """
    Apply a filter threshold to the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        flag_col (str): The column name to use for the filter flag.
        flag_type_col (str): The column name to use for the flag type.
        metric (str): The metric to use for the filter.
        operation (str): The operation to use for the filter. One of: <, <=, >, >=, ==, !=
        threshold (int | float | bool | str | list | set | tuple): The threshold to use for the filter.
        flag_type (str): The type of flag to apply.
        invert (bool): Whether to invert the filter or not.
        override (bool): Whether to override the existing flag or not.
    Returns:
        None
    """
    # Filters on the number of FiNeMo hits: Skipped if no FiNeMo paths are provided (see add_num_hits)
    if metric == MetadataCols.num_hits_col and metric not in mc.columns():
        logging.info(f"Skipping filter on {metric}: no FiNeMo paths provided")
        return

    # Apply filter
    if operation == "<":
        filter_result = (mc[metric] < threshold)
    elif operation == "<=":
        filter_result = (mc[metric] <= threshold)
    elif operation == ">":
        filter_result = (mc[metric] > threshold)
    elif operation == ">=":
        filter_result = (mc[metric] >= threshold)
    elif operation == "==":
        filter_result = (mc[metric] == threshold)
    elif operation == "!=":
        filter_result = (mc[metric] != threshold)
    elif operation == "isna":
        filter_result = (mc[metric].isna())
    elif operation == "notna":
        filter_result = (mc[metric].notna())
    elif operation == "contains":
        if isinstance(threshold, str):
            filter_result = mc[metric].str.contains(threshold, na=False)
        elif isinstance(threshold, (list, set, tuple)):
            filter_result = mc[metric].apply(lambda x: pd.notna(x) and any(t in x for t in threshold))
        else:
            raise ValueError("Threshold must be a string, list, set, or tuple for 'contains' operation.")
    elif operation == "isin":
        if not isinstance(threshold, (list, set, tuple)):
            raise ValueError("Threshold must be a list, set, or tuple for 'isin' operation.")
        filter_result = (mc[metric].isin(threshold))
    else:
        raise ValueError("Invalid operation for filter threshold.")
    
    # Invert: True -> False, False -> True
    if invert is True:
        filter_result = ~filter_result
    
    # Standard: True | False = True
    if override is False:
        mc[flag_col] = mc[flag_col] | filter_result

    # Override: True & False = False
    elif override is True:
        mc[flag_col] = mc[flag_col] & filter_result
    
    # Apply filter type
    if flag_type is not None:
        mc.metadata.loc[filter_result, flag_type_col] = flag_type


def apply_filter_clusters(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply cluster filters to the MotifCompendium object.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs): The filter arguments.
        filter_flag_col (str): The column name to use for the filter flag.

    Returns:
        None
    """
    # Filter #1: Soft filters
    for filter_args in motif_filter_args.soft_filters:
        if filter_args.apply_cluster:
            logging.info(f"Applying cluster filter: {filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=filter_args.metric,
                operation=filter_args.operation,
                threshold=filter_args.threshold,
                invert=filter_args.invert,
                override=filter_args.override,
                flag_type=filter_args.flag_type,
            )

    # Filter #2: Soft override: Any match.  See apply_filter_motifs() for
    # the two supported profile schemas and their ordering.
    soft_override_filters = getattr(
        motif_filter_args,
        "soft_override_filters",
        getattr(motif_filter_args, "override_filters", ()),
    )
    for soft_override_filter_args in soft_override_filters:
        if soft_override_filter_args.apply_cluster:
            logging.info(f"Applying cluster override filter: {soft_override_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=soft_override_filter_args.metric,
                operation=soft_override_filter_args.operation,
                threshold=soft_override_filter_args.threshold,
                invert=soft_override_filter_args.invert,
                override=soft_override_filter_args.override,
            )

    # Filter #3: Moderate filters
    for moderate_filter_args in getattr(motif_filter_args, "moderate_filters", ()):
        if moderate_filter_args.apply_cluster:
            logging.info(f"Applying cluster moderate filter: {moderate_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=moderate_filter_args.metric,
                operation=moderate_filter_args.operation,
                threshold=moderate_filter_args.threshold,
                invert=moderate_filter_args.invert,
                override=moderate_filter_args.override,
                flag_type=moderate_filter_args.flag_type,
            )
    
    # Filter #4: Moderate override: Direct match
    for moderate_override_filter_args in getattr(motif_filter_args, "moderate_override_filters", ()):
        if moderate_override_filter_args.apply_cluster:
            logging.info(f"Applying cluster moderate override filter: {moderate_override_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=moderate_override_filter_args.metric,
                operation=moderate_override_filter_args.operation,
                threshold=moderate_override_filter_args.threshold,
                invert=moderate_override_filter_args.invert,
                override=moderate_override_filter_args.override,
            )

    # Filter #5: Apply hard filters
    for hard_filter_args in motif_filter_args.hard_filters:
        if hard_filter_args.apply_cluster:
            logging.info(f"Applying cluster hard filter: {hard_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=hard_filter_args.metric,
                operation=hard_filter_args.operation,
                threshold=hard_filter_args.threshold,
                invert=hard_filter_args.invert,
                override=hard_filter_args.override,
                flag_type=hard_filter_args.flag_type,
            )


#### HITS INDEXING: WORKERS ---------------------------------------------------
# Each FiNeMo hit with the ENCODE cluster of its motif, as an indexed BED (bgzip + tabix); counts and profile
# hits of an experiment are then collapsed into one BED. From sherlock/runs/encode/finemo/apply_mc/update_hits_bed.py.
HITS_BED_COLS = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
HITS_INDEXED_COLS = HITS_BED_COLS + ['category', 'confidence', 'head', 'name_tfmodisco', 'hits_bed_cluster']
HITS_CONFIDENCES = ['high', 'medium', 'low', 'flag']  # label_confidence, best first
HITS_CONFIDENCES_SIMPLE = ['high', 'medium', 'low', '']
HITS_COLORS = {'high': '0,128,0', 'medium': '255,255,0', 'low': '255,0,0'}  # Green, yellow, red; others grey
HITS_GREY = '128,128,128'  # Flag, non-canonical, filtered
HITS_NAME_UNKNOWN = 'unvalidated'
HITS_NAME_NONE = 'None'
HITS_CATEGORY_NONCANONICAL = 'non-canonical'
HITS_MOTIF_COLS = ['name_tfmodisco', 'hits_bed_name', 'hits_bed_cluster', 'label_confidence', 'category', 'flagged']


def _hits_pattern_key(posneg, names):
    # "pos" / "neg" + "." + pattern number (the trailing "pattern_{#}"), as in add_num_hits() of the atlas pipelines
    return posneg.astype(str).str[:3] + "." + names.astype(str).str.extract(r"(?:^|[^A-Za-z])pattern_(\d+)$", expand=False)


def _write_bed_tabix(bed_df, bed_gz_path):
    # Sort, write, then bgzip + tabix-index: writes bed_gz_path and bed_gz_path.tbi (removes the plain BED)
    os.makedirs(os.path.dirname(bed_gz_path), exist_ok=True)
    bed_plain_path = bed_gz_path.removesuffix(".gz")
    bed_df.sort_values(['chrom', 'start', 'end']).to_csv(bed_plain_path, sep="\t", header=False, index=False)
    pysam.tabix_index(bed_plain_path, preset="bed", force=True)


def _index_hits_bed(hits_bed_in, hits_bed_out, motif_lookup, head):
    """FiNeMo hits of one experiment and head, each with the ENCODE cluster of its motif.

    motif_lookup: pos/neg pattern key -> HITS_MOTIF_COLS values. Hits of a motif not in it are unvalidated,
    without a cluster (None), flagged and grey, as in update_hits_bed.py.
    """
    hits_df = pd.read_csv(hits_bed_in, sep="\t", header=None, usecols=range(6), names=HITS_BED_COLS[:6], keep_default_na=False)
    motifs = pd.DataFrame.from_dict(motif_lookup, orient="index", columns=HITS_MOTIF_COLS)
    matched = motifs.reindex(_hits_pattern_key(hits_df['name'].str.split(".").str[0], hits_df['name']).to_numpy())
    confidence = matched['label_confidence'].fillna(HITS_CONFIDENCES[-1]).to_numpy()
    category = matched['category'].fillna(HITS_CATEGORY_NONCANONICAL).to_numpy()
    flagged = matched['flagged'].fillna(True).astype(bool).to_numpy()

    hits_df['name_tfmodisco'] = matched['name_tfmodisco'].fillna(pd.Series(hits_df['name'].to_numpy(), index=matched.index)).to_numpy()
    hits_df['name'] = matched['hits_bed_name'].fillna(HITS_NAME_UNKNOWN).to_numpy()
    hits_df['hits_bed_cluster'] = matched['hits_bed_cluster'].fillna(HITS_NAME_NONE).to_numpy()
    hits_df['thickStart'] = hits_df['start']
    hits_df['thickEnd'] = hits_df['end']
    hits_df['head'] = head
    hits_df['category'] = category
    hits_df['confidence'] = confidence
    item_rgb = pd.Series(confidence).map(HITS_COLORS).fillna(HITS_GREY).to_numpy()
    item_rgb[(category == HITS_CATEGORY_NONCANONICAL) | flagged] = HITS_GREY
    hits_df['itemRgb'] = item_rgb
    _write_bed_tabix(hits_df[HITS_INDEXED_COLS], hits_bed_out)
    return len(hits_df)


def _collapse_overlapping_hits(df, deduplicate_col):
    """Collapse overlapping hits sharing chrom + deduplicate_col (not for None); keep the most confident label."""
    df = df.sort_values(['chrom', deduplicate_col, 'start', 'end']).reset_index(drop=True)
    confidence_rank = {confidence: i for i, confidence in enumerate(HITS_CONFIDENCES)}

    chroms = df['chrom'].to_numpy()
    starts = df['start'].to_numpy()
    ends = df['end'].to_numpy()
    names = df['name'].to_numpy()
    scores = df['score'].to_numpy()
    strands = df['strand'].to_numpy()
    rgbs = df['itemRgb'].to_numpy()
    categories = df['category'].to_numpy()
    confidences = df['confidence'].to_numpy()
    heads = df['head'].astype(str).to_numpy()
    tfmodiscos = df['name_tfmodisco'].astype(str).to_numpy()
    clusters = df['hits_bed_cluster'].to_numpy()
    conditions = df[deduplicate_col].to_numpy()

    merged_rows = []
    n = len(df)
    i = 0
    while i < n:
        curr_chrom, curr_start, curr_end = chroms[i], starts[i], ends[i]
        curr_name, curr_score, curr_strand, curr_rgb = names[i], scores[i], strands[i], rgbs[i]
        curr_category, curr_confidence = categories[i], confidences[i]
        curr_cluster, curr_condition = clusters[i], conditions[i]
        curr_heads, curr_tfmodisco = [heads[i]], [tfmodiscos[i]]

        j = i + 1
        while (
            j < n
            and chroms[j] == curr_chrom
            and conditions[j] == curr_condition
            and curr_condition != HITS_NAME_NONE
            and starts[j] < curr_end
        ):
            curr_end = max(curr_end, ends[j])
            curr_heads.append(heads[j])
            curr_tfmodisco.append(tfmodiscos[j])
            if confidence_rank.get(curr_confidence, float('inf')) > confidence_rank.get(confidences[j], float('inf')):
                curr_name, curr_score, curr_strand, curr_rgb = names[j], scores[j], strands[j], rgbs[j]
                curr_category, curr_confidence = categories[j], confidences[j]
            j += 1

        merged_rows.append({
            'chrom': curr_chrom, 'start': curr_start, 'end': curr_end,
            'name': curr_name, 'score': curr_score, 'strand': curr_strand,
            'thickStart': curr_start, 'thickEnd': curr_end, 'itemRgb': curr_rgb,
            'category': curr_category, 'confidence': curr_confidence,
            'head': ';'.join(dict.fromkeys(curr_heads)),
            'name_tfmodisco': ';'.join(dict.fromkeys(curr_tfmodisco)),
            'hits_bed_cluster': curr_cluster,
        })
        i = j

    return pd.DataFrame(merged_rows, columns=HITS_INDEXED_COLS)


def _combine_hits_beds(hits_beds_in, hits_bed_out):
    """Indexed hits of an experiment's heads, combined; overlapping hits of an ENCODE cluster collapsed."""
    hits_df = pd.concat(
        [pd.read_csv(path, sep="\t", header=None, names=HITS_INDEXED_COLS, keep_default_na=False) for path in hits_beds_in],
        axis=0, ignore_index=True,
    )
    hits_df = _collapse_overlapping_hits(hits_df, 'hits_bed_cluster')
    _write_bed_tabix(hits_df, hits_bed_out)
    return len(hits_df)


#### HITS ANNOTATION: WORKERS ------------------------------------------------
# ProcessPoolExecutor pickles the function that it runs, so that function must be module-level
# (a function nested in run_encode_combine() cannot be pickled). run_encode_combine() puts its
# annotation closure here before the (forked) pool starts, so the workers inherit it.
_HITS_WORKER = {}


def _hits_worker_init(hg38_fasta_path):
    # Each worker opens its own handle: pysam.FastaFile isn't safe to share across processes
    # (the underlying fd's read position would race)
    import pysam
    _HITS_WORKER["fasta"] = pysam.FastaFile(hg38_fasta_path)


def _hits_worker_run(hits_tsv, peaks_bed, hits_tsv_annotated):
    return _HITS_WORKER["annotate_row"](hits_tsv, peaks_bed, hits_tsv_annotated, _HITS_WORKER["fasta"])


def _summarize_hits_file(hits_tsv_annotated):
    # Per annotated hits file: hits per (cluster, peak type), and distances to the TSS / peak summit per cluster
    cols = pd.read_csv(
        hits_tsv_annotated,
        sep="\t",
        usecols=["hits_bed_cluster", "peakType", "distToTSS", "distToPeakSummit"],
    )
    peak_type_counts = cols.groupby(['hits_bed_cluster', 'peakType']).size().unstack(fill_value=0)
    dist_values = {
        cluster: (g['distToTSS'].to_numpy(), g['distToPeakSummit'].to_numpy())
        for cluster, g in cols.groupby('hits_bed_cluster')
    }
    return peak_type_counts, dist_values


def run_encode_combine(args):
    """ENCODE: combine the TF, Chromatin and ProCap atlasclusters into ENCODE clusters"""


    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    # A run log is ALWAYS written into <output-dir>/logs/, as in the per-head pipeline.
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_{os.path.basename(os.path.normpath(args.output_dir))}"
        f"_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.out",
    )
    # force=True: each stage replaces the previous stage's handlers (incl. its log file)
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    )
    logging.info(f'Run started: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logging.info(f"Writing run log to: {log_path}")

    # Set compute options
    if args.verbose:
        logging.info(
            f"Setting compute options:\n"
            f"  max_chunk={args.max_chunk},\n"
            f"  max_cpus={args.max_cpus},\n"
            f"  use_gpu={args.use_gpu},\n"
            f"  ic_scale={args.ic},\n"
            f"  fast_plot={args.fast_plot},\n"
        )

    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        ic_scale=args.ic,
        fast_plotting=args.fast_plot,
    )

    # Check output
    if args.verbose:
        logging.info(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    ### PATHS --------------------------------------------------------------------
    ## Derived identifiers
    unique_id_initial = f'{MetadataCols.unique_id_col}.{args.mcid_version}'

    ## Derived clustering column names
    encodecluster_cluster_col = f"encode_rec_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}"

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ## Input: TF atlas
    mc_atlascluster_filtered_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_atlascluster_all.mc")

    mc_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_metacluster_all.mc")

    mc_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_metacluster_all.mc")


    ## Input: Chromatin atlas
    mc_atlascluster_filtered_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_atlascluster_all.mc")

    mc_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_metacluster_all.mc")

    mc_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_metacluster_all.mc")


    ## Input: ProCap atlas
    mc_atlascluster_filtered_procap_path = os.path.join(args.procap_combined_dir, "mc", "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_procap_path = os.path.join(args.procap_combined_dir, "mc", "motifcompendium_atlascluster_all.mc")

    mc_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_metacluster_all.mc")

    mc_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_metacluster_all.mc")


    ## Output: MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")

    mc_atlascluster_combined_path = os.path.join(mc_dir, "motifcompendium_atlascluster_combined.mc")
    mc_encode_path = os.path.join(mc_dir, "motifcompendium_encode_all.mc")
    mc_encode_filtered_path = os.path.join(mc_dir, "motifcompendium_encode_filtered.mc")
    mc_encode_removed_path = os.path.join(mc_dir, "motifcompendium_encode_removed.mc")

    ## Output: HTML summary table
    html_dir = os.path.join(args.output_dir, "html")

    html_encode_path = os.path.join(html_dir, "motifcompendium_encode_all.html")
    html_encode_filtered_path = os.path.join(html_dir, "motifcompendium_encode_filtered.html")
    html_encode_removed_path = os.path.join(html_dir, "motifcompendium_encode_removed.html")
    html_encode_edited_path = os.path.join(html_dir, "motifcompendium_encode_all_edited.html")
    html_encode_edited_filtered_path = os.path.join(html_dir, "motifcompendium_encode_filtered_edited.html")
    html_encode_edited_removed_path = os.path.join(html_dir, "motifcompendium_encode_removed_edited.html")

    ## Output: Export - MoDISco HDF5, MEME
    export_dir = os.path.join(args.output_dir, "export")

    modisco_encode_path = os.path.join(export_dir, "motifcompendium_encode_all.h5")
    modisco_encode_filtered_path = os.path.join(export_dir, "motifcompendium_encode_filtered.h5")
    modisco_encode_removed_path = os.path.join(export_dir, "motifcompendium_encode_removed.h5")

    meme_encode_path = os.path.join(export_dir, "motifcompendium_encode_all.meme.txt")
    meme_encode_filtered_path = os.path.join(export_dir, "motifcompendium_encode_filtered.meme.txt")
    meme_encode_removed_path = os.path.join(export_dir, "motifcompendium_encode_removed.meme.txt")

    ## Output: Metadata
    metadata_dir = os.path.join(args.output_dir, "metadata")

    metadata_atlascluster_combined_path = os.path.join(metadata_dir, "motifcompendium_atlascluster_combined_metadata.tsv")
    metadata_encode_path = os.path.join(metadata_dir, "motifcompendium_encode_all_metadata.tsv")
    metadata_encode_filtered_path = os.path.join(metadata_dir, "motifcompendium_encode_filtered_metadata.tsv")
    metadata_encode_removed_path = os.path.join(metadata_dir, "motifcompendium_encode_removed_metadata.tsv")
    metadata_motif_index_path = os.path.join(metadata_dir, "motifcompendium_encode_motif_index.tsv")
    metadata_motif_category_path = os.path.join(metadata_dir, "motifcompendium_encode_motif_category.tsv")

    args_json_path = os.path.join(metadata_dir, "args.json")

    ## Output: Hits annotation
    metadata_hits_path = os.path.join(metadata_dir, "motifcompendium_encode_metadata_hits.tsv")
    hits_annotated_dir = os.path.join(export_dir, "hits_annotated")

    ## Per-atlas lookup: upstream pipeline directory, by atlas / head
    mc_dirs = {
        EncodeCombineArgs.tfatlas_id: {
            'combined': args.tf_combined_dir,
            'counts': args.tf_counts_dir,
            'profile': args.tf_profile_dir,
        },
        EncodeCombineArgs.chromatlas_id: {
            'combined': args.chrom_combined_dir,
            'counts': args.chrom_counts_dir,
            'profile': args.chrom_profile_dir,
        },
        EncodeCombineArgs.procapatlas_id: {
            'combined': args.procap_combined_dir,
            'counts': args.procap_counts_dir,
            'profile': args.procap_profile_dir,
        },
    }

    # ## Combine atlascluster
    logging.info("Loading TF, Chromatin and ProCap atlascluster compendia...")

    # ### Load MotifCompendium
    # Load MotifCompendium
    mc_atlascluster_filtered_tf = MotifCompendium.load(mc_atlascluster_filtered_tf_path, safe=args.safe)
    mc_atlascluster_filtered_chrom = MotifCompendium.load(mc_atlascluster_filtered_chrom_path, safe=args.safe)
    mc_atlascluster_filtered_procap = MotifCompendium.load(mc_atlascluster_filtered_procap_path, safe=args.safe)

    # The atlases' own selection: once an ENCODE combine has written its final flags back,
    # motifcompendium_atlascluster_filtered.mc holds those instead, and this stage would recluster an already-filtered selection
    for mc_input, mc_input_path in (
        (mc_atlascluster_filtered_tf, mc_atlascluster_filtered_tf_path),
        (mc_atlascluster_filtered_chrom, mc_atlascluster_filtered_chrom_path),
        (mc_atlascluster_filtered_procap, mc_atlascluster_filtered_procap_path),
    ):
        if VisualizeArgs.encodecluster_name_col in mc_input.columns():
            raise ValueError(
                f"{mc_input_path} holds the final flags of an earlier ENCODE combine ({VisualizeArgs.encodecluster_name_col}), "
                "not its atlas's own selection: rerun that atlas (its per-head and combine stages) before this stage."
            )

    # Add TF, Chrom, ProCap to source, names, constituents
    mc_atlascluster_filtered_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_atlascluster_filtered_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_atlascluster_filtered_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id

    # Preserve all columns, None if not specified
    all_metadata_cols = set(mc_atlascluster_filtered_tf.columns()) | set(mc_atlascluster_filtered_chrom.columns()) | set(mc_atlascluster_filtered_procap.columns())

    for mc in [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom, mc_atlascluster_filtered_procap]:
        for col in sorted(all_metadata_cols):
            if col not in mc.columns():
                mc[col] = None

    # Number of FiNeMo hits: NaN, not None, for an atlas without FiNeMo paths (e.g. ProCap), so that the
    # ENCODE cluster sums stay numeric (see the ENCODE cluster num_hits below).
    if MetadataCols.num_hits_col in all_metadata_cols:
        for mc in [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom, mc_atlascluster_filtered_procap]:
            mc[MetadataCols.num_hits_col] = pd.to_numeric(mc[MetadataCols.num_hits_col], errors="coerce")

    # Delete all images, except logos
    keep_images = ['logo (fwd)', 'logo (rev)']

    for mc in [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom, mc_atlascluster_filtered_procap]:
        remove_images = set(mc.images()) - set(keep_images)
        mc.delete_images(list(remove_images))


    # ### Import reference
    logging.info("Loading reference motif databases...")

    # Load Ref MotifCompendium
    if args.ref_qc.endswith(".mc"):
        mc_ref_qc = MotifCompendium.load(args.ref_qc, safe=args.safe)
    else:
        mc_ref_qc = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref_qc))[0]: args.ref_qc},
            safe=args.safe,
        )
    if args.ref_invitro.endswith(".mc"):
        mc_ref_invitro = MotifCompendium.load(args.ref_invitro, safe=args.safe)
    else:
        mc_ref_invitro = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref_invitro))[0]: args.ref_invitro},
            safe=args.safe,
        )
    if args.ref_direct:
        if args.ref_direct.endswith(".mc"):
            mc_ref_direct = MotifCompendium.load(args.ref_direct, safe=args.safe)
        else:
            mc_ref_direct = MotifCompendium.build_from_pfm(
                pfm_dict={os.path.splitext(os.path.basename(args.ref_direct))[0]: args.ref_direct},
                safe=args.safe,
            )
    else:
        # Direct: Combine MotifCompendium
        for col in list(set(mc_ref_qc.columns() + mc_ref_invitro.columns())):
            for mc_ref in [mc_ref_qc, mc_ref_invitro,]:
                if col not in mc_ref.columns():
                    mc_ref[col] = None

        mc_ref_direct = MotifCompendium.combine(
            [mc_ref_qc, mc_ref_invitro,],
            ['reference-qc', 'reference-invitro',],
            safe=args.safe,
        )

    # Remove multimers
    mc_ref_qc = mc_ref_qc[~mc_ref_qc[MetadataCols.name_col].str.contains('::')]
    mc_ref_direct = mc_ref_direct[~mc_ref_direct[MetadataCols.name_col].str.contains('::')]

    # IC scale
    mc_ref_qc.motifs = utils_motif.ic_scale(mc_ref_qc.motifs)
    mc_ref_direct.motifs = utils_motif.ic_scale(mc_ref_direct.motifs)

    # Reference: Non-ZNF (including CTCF, REST)
    mc_ref_qc_nonznf_mask = ~(
        mc_ref_qc[TFAnnotationCols.tfclass_family_name_col].astype(str).str.contains("zinc", case=False, na=False)
        | mc_ref_qc[TFAnnotationCols.humantf_dbd_col].astype(str).str.contains("C2H2", case=False, na=False)
        | mc_ref_qc[TFAnnotationCols.family_col].astype(str).str.contains("ZNF|ZSCAN|ZKSCAN|ZBTB", case=False, na=False)
    ) | mc_ref_qc[TFAnnotationCols.family_col].isin(['CTCF', 'REST'])
    mc_ref_qc_nonznf = mc_ref_qc[mc_ref_qc_nonznf_mask]

    # Annotate motifs: Human-readable
    mc_ref_clustered_hr_metadata = pd.read_csv(args.ref_clustered_hr_metadata, sep="\t")

    mc_ref_clustered_hr_metadata[MetadataCols.name_col] = mc_ref_clustered_hr_metadata[MetadataCols.name_col].str.split("_").str[0]  # Remove "_{#}"
    mc_ref_clustered_hr_metadata['motifs_name'] = mc_ref_clustered_hr_metadata['motifs'].str.split(",")
    mc_ref_clustered_hr_metadata_expand = mc_ref_clustered_hr_metadata.explode("motifs_name")

    mc_ref_qc_hr_dict = dict(zip(mc_ref_clustered_hr_metadata_expand['motifs_name'], mc_ref_clustered_hr_metadata_expand[MetadataCols.name_col]))
    mc_ref_qc_tf_dict = dict(zip(mc_ref_qc[MetadataCols.name_col], mc_ref_qc[ReferenceMatchCols.ref_tf_col]))

    mc_ref_qc[LabelCols.label_col] = mc_ref_qc[MetadataCols.name_col].map(mc_ref_qc_hr_dict)


    # #### Pairwise similarity
    # Combine MotifCompendium
    mc_atlascluster_combined = MotifCompendium.combine(
        [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom, mc_atlascluster_filtered_procap],
        safe=args.safe,
    )


    # #### Cluster: ENCODE clusters
    logging.info("Force-clustering TF, Chromatin and ProCap atlasclusters into ENCODE clusters...")
    # Force-cluster
    mc_atlascluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_force,
        similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
        save_name=encodecluster_cluster_col,
        largest_clusters_first=True,
        density=ClusterArgs.cluster_force_density,
        seed=ClusterArgs.seeds[0],
    )
    # Reassign: K-means
    mc_atlascluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=encodecluster_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=encodecluster_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_encode = mc_atlascluster_combined[encodecluster_cluster_col]
    logging.info(f"Unique clusters: {len(set(membership_encode))}")

    # Cluster: Recursively
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Force-cluster: DCC
        mc_atlascluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_on=encodecluster_cluster_col,
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=encodecluster_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-means
        mc_atlascluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=encodecluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=encodecluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_encode = mc_atlascluster_combined[encodecluster_cluster_col]
        if args.verbose:
            logging.info(f"  Unique clusters: {len(set(new_membership_encode))}")
        if np.array_equal(membership_encode, new_membership_encode):
            logging.info(f"Final unique clusters: {len(set(new_membership_encode))}")
            break
        else:
            membership_encode = new_membership_encode


    # # ENCODE Atlas

    # #### Average
    # Average MotifCompendium
    aggregations_encodecluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.aggregations:
        if agg_col_in in mc_atlascluster_combined.columns():
            aggregations_encodecluster.append((agg_col_in, agg, agg_col_out))

    mc_encode = mc_atlascluster_combined.cluster_averages(
        clustering=encodecluster_cluster_col,
        aggregations=aggregations_encodecluster,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Number of FiNeMo hits: NaN, not 0, for an ENCODE cluster without hits information (only atlasclusters of
    # an atlas without FiNeMo paths, e.g. ProCap; cluster_averages() sums the non-NaN values, so 0), so that
    # 1_minhits does not flag it. A mixed cluster sums the hits of its atlasclusters that have them.
    if MetadataCols.num_hits_col in mc_encode.columns():
        num_hits_encode = mc_atlascluster_combined.metadata.groupby(encodecluster_cluster_col)[MetadataCols.num_hits_col].sum(min_count=1)
        mc_encode[MetadataCols.num_hits_col] = mc_encode[ClusterArgs.source_cluster_col].map(num_hits_encode)

    # Weighted aggregations
    weighted_aggregations_encodecluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.weighted_aggregations:
        if agg_col_in in mc_atlascluster_combined.columns():
            weighted_aggregations_encodecluster.append((agg_col_in, agg, agg_col_out))

    for (agg_col_old, _, agg_col_new) in weighted_aggregations_encodecluster:
        aggregate_weighted_avg(
            mc=mc_atlascluster_combined,
            mc_cluster=mc_encode,
            cluster_col=encodecluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )

    # Merge aggregations
    merge_aggregations_encodecluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_encode.columns():
            merge_aggregations_encodecluster.append(aggregate)
        if agg_col_new not in mc_encode.columns():
            mc_encode.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_encodecluster:
        mc_encode.metadata[agg_col_new] = mc_encode.metadata[[agg_col_old, agg_col_new]].apply(
            lambda row: ",".join(dict.fromkeys(
                part
                for agg in row
                if pd.notna(agg) and agg != ""
                for part in agg.split(",")
                if part not in ("", "None")
            )),
            axis=1
        )


    # #### Calculations
    ## Calculations
    mc_encode[MetadataCols.total_contrib_score_ratio_col] = mc_encode[MetadataCols.total_contrib_score_col] / mc_encode[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_encode[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_encode.get_standard_motif_stack())  # Positive / negative
    mc_encode[MetadataCols.composite_col] = mc_encode[MetadataCols.composite_col].astype(bool)

    mc_encode.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlet_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_encode.sort(by=[direct_type_col, composite_col], ascending=True, inplace=True)  # Sort by direct type, composite
    mc_encode.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_encode[MetadataCols.unique_id_col] = unique_id_initial + f".C4.TF_Chrom_ProCap.counts_profile." + mc_encode.metadata.index.map('{:08d}'.format)

    ### Aggregate: Annotations
    mc_tfatlas_metadata_updates = []
    mc_tfatlas_images_updated = pd.DataFrame(index=mc_encode[MetadataCols.unique_id_col])
    mc_tfatlas_images_existing = set(mc_encode.images())

    for i, row in mc_encode.metadata.iterrows():
        # MotifCompendium: Subset
        mc_tfatlas_i = mc_encode[[i]]
        ref_name_constituents_i = row[ReferenceMatchCols.ref_name_constit_all_col].split(",") if pd.notna(row[ReferenceMatchCols.ref_name_constit_all_col]) else []
        direct_ref_name_constituents_i = row[ReferenceMatchCols.direct_ref_name_constit_all_col].split(",") if pd.notna(row[ReferenceMatchCols.direct_ref_name_constit_all_col]) else []
        tf_i = row[DirectTypeArgs.tf_direct_col].split(",") if pd.notna(row[DirectTypeArgs.tf_direct_col]) else []
        tf_family_i = row[DirectTypeArgs.family_direct_col].split(",") if pd.notna(row[DirectTypeArgs.family_direct_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        if ref_name_constituents_i:
            mc_ref_qc_i = mc_ref_qc[mc_ref_qc[MetadataCols.name_col].isin(ref_name_constituents_i)]

            # Annotate with best reference constituents
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_tfatlas_i,
                assign_from_mc=mc_ref_qc_i,
                save_col_prefix=ReferenceMatchCols.best_ref_constit_col,
                min_score=0,
                max_submotifs=max_submotifs_i,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )

        ## [Direct] Reference annotations 
        if direct_ref_name_constituents_i:
            # Reference direct: Subset (monomer)
            mc_ref_direct_i = mc_ref_direct[
                (mc_ref_direct[MetadataCols.name_col].isin(direct_ref_name_constituents_i)) &
                ((mc_ref_direct[ReferenceMatchCols.ref_tf_col].isin(tf_i)) |
                 (mc_ref_direct[ReferenceMatchCols.ref_family_col].isin(tf_family_i)))
            ]
        
            # Reference: Subset (multimer: homo, hetero)
            mc_ref_i = mc_ref_direct[mc_ref_direct[MetadataCols.name_col].isin(ref_name_constituents_i)]  # Ref direct = Ref QC + Ref Invitro

            # Annotate with best reference constituents
            if len(mc_ref_direct_i) > 0:
                assign_direct_composite(
                    target_mc=mc_tfatlas_i,
                    direct_reference_mc=mc_ref_direct_i,
                    indirect_reference_mc=mc_ref_i,
                    save_col_prefix=ReferenceMatchCols.best_direct_ref_constit_col,
                    min_score=0,
                    max_submotifs=max_submotifs_i,
                    label_unsigned=True,
                    save_images=True,
                    logo_trimming=0,
                )
    
        ## Update: Metadata, Images
        mc_tfatlas_metadata_updates.append(mc_tfatlas_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_tfatlas_i.images() if image_col not in mc_tfatlas_images_existing]
        mc_tfatlas_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_tfatlas_i.get_images(image_col),
                index=mc_tfatlas_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_tfatlas_images_updated = mc_tfatlas_images_updated.combine_first(mc_tfatlas_i_new_images)
        mc_tfatlas_images_updated.update(mc_tfatlas_i_new_images)

    # Apply updates: Metadata, Images
    mc_tfatlas_metadata_updates_df = pd.concat(mc_tfatlas_metadata_updates)
    mc_encode.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_encode.metadata = mc_encode.metadata.combine_first(mc_tfatlas_metadata_updates_df)
    mc_encode.metadata.update(mc_tfatlas_metadata_updates_df)
    mc_encode.metadata.reset_index(inplace=True)
    del mc_tfatlas_metadata_updates, mc_tfatlas_metadata_updates_df

    for image_col in mc_tfatlas_images_updated.columns:
        mc_encode.set_images(
            image_name=image_col,
            images=mc_tfatlas_images_updated[image_col].where(
                pd.notna(mc_tfatlas_images_updated[image_col]), None).to_list()
        )
    del mc_tfatlas_images_updated, mc_tfatlas_images_existing


    # #### Annotate
    # ### Annotate: Meta-clusters
    # Initialize
    # Experiment: Determined by motif; Composite: Determined by cluster
    mc_encode[LabelCols.label_col] = MetadataCols.name_unknown  # Initial readable name: Unknown
    # mc_encode[direct_label_col] = None  # Inherit TF Atlas
    mc_encode[LabelCols.label_core_col] = None
    mc_encode[LabelCols.direct_label_core_col] = None
    mc_encode[LabelCols.label_flank_col] = None
    # mc_encode[ref_readable_name_col] = None

    ## [General] Label meta-clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_encode,
        assign_from_mc=mc_ref_qc,
        from_label_col=MetadataCols.name_col,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=True,
        logo_trimming=0,
    )

    ## Annotate: Centroids (Indirect)
    # Map human-readable name
    ref_readable_name_cols = [f'{ReferenceMatchCols.ref_readable_col}_name{i}' for i in range(MotifMatchArgs.max_submotifs)]
    ref_score_cols = [f'{ReferenceMatchCols.ref_col}_score{i}' for i in range(MotifMatchArgs.max_submotifs)]
    for i, (ref_readable_name_col, ref_score_col) in enumerate(zip(ref_readable_name_cols, ref_score_cols)):    
        mc_encode[ref_readable_name_col] = mc_encode[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # cateogory: Mask
    mask_direct_encodecluster = mc_encode[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
    mask_nondirect_encodecluster = ~mask_direct_encodecluster
    mask_monomer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_encode[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_encode[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_encode[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_encode[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_encodecluster_nondirect = (
        mask_dimer_encodecluster_nondirect | mask_trimer_encodecluster_nondirect
    )
    mask_matched_atlascluster_nondirect = (
        mask_monomer_encodecluster_nondirect | mask_dimer_encodecluster_nondirect | mask_trimer_encodecluster_nondirect
    )

    # Dimer-style notation
    mc_encode.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_col] = mc_encode.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_encode.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_readable_col] = mc_encode.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_encode.metadata.loc[mask_dimer_encodecluster_nondirect, LabelCols.label_flank_col] = mc_encode.metadata.loc[mask_dimer_encodecluster_nondirect, ref_readable_name_cols[1]]
    mc_encode.metadata.loc[mask_trimer_encodecluster_nondirect, LabelCols.label_flank_col] = mc_encode.metadata.loc[mask_trimer_encodecluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_encode.metadata.loc[mask_trimer_encodecluster_nondirect, ref_readable_name_cols[2]]
    # mc_encode.metadata.loc[mask_dimer_encodecluster_nondirect, num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_encode.metadata.loc[mask_trimer_encodecluster_nondirect, num_subunit_col] = 3
    # mc_encode.metadata[composite_col] = mc_encode.metadata[num_subunit_col] > min_composite_subunit

    # # Trimer-style notation
    # mc_encode[ref_readable_name_col] = mc_encode.metadata[
    #     [
    #         f'{ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

    # Concatenate all unique annotations
    mc_encode[ReferenceMatchCols.ref_name_col] = mc_encode.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### Filter
    # Calculate filters
    mc_encode[FilterArgs.filter_flag_col] = False
    utils_analysis.calculate_filters(
        mc=mc_encode,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )

    # Apply filters
    apply_filter_clusters(
        mc=mc_encode,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # ENCODE filter: Unvalidated, Indirect, from only one experiment with under 250 seqlets
    encode_filter_mask = (
        (mc_encode[LabelCols.label_col] == MetadataCols.name_unknown)  # Unvalidated
        & ~(mc_encode[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:2]), na=False))  # Keep all Direct TFs A, B-NZ, regardless of QC score
        & ~(mc_encode[MetadataCols.model_col].str.contains(","))  # 
        & (mc_encode[MetadataCols.num_seqlet_col] < 250)
    )
    mc_encode.metadata.loc[encode_filter_mask, FilterArgs.filter_flag_col] = True


    # #### Filter: QC - Remove
    if args.mc_qc_removed:
        # Load QC MotifCompendium: ENCODE QC Filters
        mc_qc_removed = MotifCompendium.load(args.mc_qc_removed, safe=args.safe)

        # Scale: Off
        ic_scale = False
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Assign label: From QC Removed
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_encode,
            assign_from_mc=mc_qc_removed[~(
                (mc_qc_removed[LabelCols.label_readable_col] == FilterArgs.filter_flag_types[-1])
                & (mc_qc_removed[MetadataCols.num_seqlet_col] < 250)
            )],  # Do not use: Unvalidated, less than 250 seqlets
            from_label_col=LabelCols.label_readable_col,  # After v21: Change to prename_col to label_readable_col,
            save_col_prefix=MetadataCols.qc_col,
            min_score=FilterArgs.qc_min_score_removed,
            max_submotifs=1,
            label_unsigned=False,
            save_images=False,
        )

        # Scale: On
        ic_scale = True
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Assign QC name: simple name, readable label, filter type
        mask_qc = (mc_encode[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_removed)
        mc_encode.metadata.loc[mask_qc, LabelCols.label_readable_col] = mc_encode.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_encode.metadata.loc[mask_qc, LabelCols.label_col] = mc_encode.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_encode.metadata.loc[mask_qc, FilterArgs.filter_flag_type_col] = mc_encode.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_encode.metadata.loc[mask_qc, MetadataCols.checked_col] = True

        # Filter:
        mask_filter = (
            mc_encode[FilterArgs.filter_flag_col] | 
            ((mc_encode[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_removed) &  # Flag if matches to removed motifs
            ~(mc_encode[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:2]), na=False)))  # Keep all Direct TFs A, B-NZ, regardless of QC score
        )
        mc_encode.metadata.loc[mask_filter, FilterArgs.filter_flag_col] = True

        # Update: Filter type, Core (Direct), Core, Flank, Label confidence
        mc_encode.metadata.loc[mask_filter, FilterArgs.filter_flag_type_col] = mc_encode.metadata.loc[mask_filter, LabelCols.label_col]
        mc_encode.metadata.loc[mask_filter, LabelCols.direct_label_core_col] = None
        mc_encode.metadata.loc[mask_filter, LabelCols.label_core_col] = None
        mc_encode.metadata.loc[mask_filter, LabelCols.label_flank_col] = None
        mc_encode.metadata.loc[mask_filter, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    # Remove flagged clusters
    mc_encode_filtered = mc_encode[~mc_encode[FilterArgs.filter_flag_col]]
    mc_encode_removed = mc_encode[mc_encode[FilterArgs.filter_flag_col]]


    # #### Clustering (for sorting only)
    ## Cluster: For sorting only
    # Cluster: Leiden CPM
    logging.info("Clustering ENCODE clusters (for sorting only)...")
    mc_encode_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_encode_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        init_clustering_col=VisualizeArgs.sort_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_encode_filtered[VisualizeArgs.sort_col]
    logging.info(f"Unique clusters: {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_encode_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_on=VisualizeArgs.sort_col,
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            n_iterations=ClusterArgs.n_iterations,
            seeds=ClusterArgs.seeds,
        )
        # Reassign: K-means
        mc_encode_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_encode_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters: {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_encode_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_on=VisualizeArgs.sort_col,
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-means
        mc_encode_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
    
        # Check convergence
        new_membership_cluster = mc_encode_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters: {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # #### Final name: ENCODE Atlas
    # Count occurrence: Core TF family (Direct)
    mc_atlascluster_combined[LabelCols.direct_label_core_readable_col_set] = mc_atlascluster_combined[LabelCols.direct_label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_atlascluster_explode_core_readable_direct = mc_atlascluster_combined.metadata.explode(LabelCols.direct_label_core_readable_col_set)

    # Count occurrence: Core TF family
    mc_atlascluster_combined[LabelCols.label_core_readable_col_set] = mc_atlascluster_combined[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_atlascluster_explode_core_readable = mc_atlascluster_combined.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_atlascluster_combined[LabelCols.label_flank_col_set] = mc_atlascluster_combined[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_atlascluster_explode_flank = mc_atlascluster_combined.metadata.explode(LabelCols.label_flank_col_set)

    ## Count occurrences: Create pivot table
    # Core: Direct: Monomer
    mc_atlascluster_core_readable_direct_monomer_pivot = pd.pivot_table(
        mc_atlascluster_explode_core_readable_direct.loc[~mc_atlascluster_explode_core_readable_direct[MetadataCols.composite_col]],
        index=encodecluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_atlascluster_core_readable_direct_monomer_pivot = mc_atlascluster_core_readable_direct_monomer_pivot.div(
        mc_atlascluster_core_readable_direct_monomer_pivot.sum(axis=1), axis=0)

    # Core: Direct: All
    mc_atlascluster_core_readable_direct_pivot = pd.pivot_table(
        mc_atlascluster_explode_core_readable_direct,
        index=encodecluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_atlascluster_core_readable_direct_pivot = mc_atlascluster_core_readable_direct_pivot.div(
        mc_atlascluster_core_readable_direct_pivot.sum(axis=1), axis=0)

    # Core: All
    mc_atlascluster_core_readable_pivot = pd.pivot_table(
        mc_atlascluster_explode_core_readable,
        index=encodecluster_cluster_col,
        columns=LabelCols.label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_atlascluster_core_readable_pivot = mc_atlascluster_core_readable_pivot.div(
        mc_atlascluster_core_readable_pivot.sum(axis=1), axis=0)

    # Composite
    mc_atlascluster_flank_pivot = pd.pivot_table(
        mc_atlascluster_explode_flank,
        index=encodecluster_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_atlascluster_flank_pivot = mc_atlascluster_flank_pivot.div(
        mc_atlascluster_flank_pivot.sum(axis=1), axis=0)

    def is_valid(x):
        """Pandas-safe validity check"""
        return pd.notna(x) and x != ""


    def assign_direct_label_core(row):
        """Assign label using weighted occurrence from constituents"""
        # Initialize
        direct_label_core = None
        direct_label_core_readable = None
        source_cluster = row[ClusterArgs.source_cluster_col]

        # 1: Assign Direct TF (if only one Direct TF)
        tf_direct = row[DirectTypeArgs.tf_direct_col]
        tf_family_direct = row[DirectTypeArgs.family_direct_col]
        if is_valid(tf_direct) and ("," not in str(tf_direct)):
            direct_label_core = str(tf_direct).strip()

            if is_valid(tf_family_direct):
                direct_label_core_readable = str(tf_family_direct).strip()
            else:
                direct_label_core_readable = direct_label_core
    
        # 2: Assign Direct Core TF family (if only one Direct TF family)
        direct_label_core_constituents = row[LabelCols.direct_label_core_constit_all_col]
        if (
            pd.isna(direct_label_core_readable)
            and pd.isna(direct_label_core)
            and is_valid(direct_label_core_constituents)
            and ("," not in str(direct_label_core_constituents))
        ):
            direct_label_core = str(direct_label_core_constituents).strip()
            direct_label_core_readable = direct_label_core

        # 3: Assign Direct Core TF family (Direct: monomer only)
        if (
            pd.isna(direct_label_core_readable)
            and pd.isna(direct_label_core)
            and (source_cluster in mc_atlascluster_core_readable_direct_monomer_pivot.index)
        ):
            core_readable_direct_monomer_frac = mc_atlascluster_core_readable_direct_monomer_pivot.loc[source_cluster]

            if core_readable_direct_monomer_frac.sum() > 0:
                for major_family_frac in np.arange(VisualizeArgs.max_major_family_frac, VisualizeArgs.major_family_increment, VisualizeArgs.major_family_increment):
                    major_family_mask = core_readable_direct_monomer_frac >= major_family_frac
                    if major_family_mask.any():
                        direct_label_core = "-".join(
                            core_readable_direct_monomer_frac[major_family_mask]
                            .sort_values(ascending=False)
                            .index[:VisualizeArgs.max_major_family_count]
                        )
                        direct_label_core_readable = direct_label_core
                        break

        # 4: Assign Direct Core TF family (Direct: all)
        if (
            pd.isna(direct_label_core_readable)
            and pd.isna(direct_label_core)
            and (source_cluster in mc_atlascluster_core_readable_direct_pivot.index)
        ):
            core_readable_direct_frac = mc_atlascluster_core_readable_direct_pivot.loc[source_cluster]

            if core_readable_direct_frac.sum() > 0:
                for major_family_frac in np.arange(VisualizeArgs.max_major_family_frac, VisualizeArgs.major_family_increment, VisualizeArgs.major_family_increment):
                    major_family_mask = core_readable_direct_frac >= major_family_frac
                    if major_family_mask.any():
                        direct_label_core = "-".join(
                            core_readable_direct_frac[major_family_mask]
                            .sort_values(ascending=False)
                            .index[:VisualizeArgs.max_major_family_count]
                        )
                        direct_label_core_readable = direct_label_core
                        break

        return direct_label_core, direct_label_core_readable


    def assign_label_core(row):
        # Initialize
        label_core = None
        label_core_readable = None
        source_cluster = row[ClusterArgs.source_cluster_col]

        # 1: Assign core TF (if only one TF)
        label_core_constituents = row[LabelCols.label_core_constit_all_col]
        if (
            is_valid(label_core_constituents)
            and ("," not in str(label_core_constituents))
        ):
            label_core = str(label_core_constituents).strip()
            label_core_readable = label_core
    
        # 2: Combine core TFs
        if (
            pd.isna(label_core_readable)
            and pd.isna(label_core)
            and (source_cluster in mc_atlascluster_core_readable_pivot.index)
        ):
            label_core_readable_frac = mc_atlascluster_core_readable_pivot.loc[source_cluster]
            if label_core_readable_frac.sum() > 0:
                for major_family_frac in np.arange(VisualizeArgs.max_major_family_frac, VisualizeArgs.major_family_increment, VisualizeArgs.major_family_increment):
                    major_family_mask = label_core_readable_frac >= major_family_frac
                    if major_family_mask.any():
                        label_core = "-".join(
                            label_core_readable_frac[major_family_mask]
                            .sort_values(ascending=False)
                            .index[:VisualizeArgs.max_major_family_count]
                        )
                        label_core_readable = label_core
                        break

        return label_core, label_core_readable


    def assign_label_flank(row):
        # Initialize
        label_flank = None
        source_cluster = row[ClusterArgs.source_cluster_col]

        # 1: Assign Flank TF (if only one Flank TF)
        label_flank_constituents = row[LabelCols.label_flank_constit_all_col]
        if (
            is_valid(label_flank_constituents)
            and ("," not in str(label_flank_constituents))
        ):
            label_flank = str(label_flank_constituents).strip()

        # 2: Combine composite TF families
        if (
            pd.isna(label_flank)
            and (source_cluster in mc_atlascluster_flank_pivot.index)
        ):
            composite_family_frac = mc_atlascluster_flank_pivot.loc[source_cluster]
            if composite_family_frac.sum() > 0:
                for major_family_frac in np.arange(VisualizeArgs.max_major_family_frac, VisualizeArgs.major_family_increment, VisualizeArgs.major_family_increment):
                    major_family_mask = composite_family_frac >= major_family_frac
                    if major_family_mask.any():
                        label_flank = "-".join(
                            composite_family_frac[major_family_mask]
                            .sort_values(ascending=False)
                            .index[:VisualizeArgs.max_major_family_count]
                        )
                        break

        return label_flank


    def compose_label(row, label_core, flank_fn):
        """Flank and combined label of a core: {core}::{flank} for a composite whose flank resolves, else {core}."""
        if not is_valid(label_core):
            return None, None
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col]) or is_valid(row[LabelCols.label_flank_constit_all_col])  # Centroid or constituent flank
        if mask_composite and mask_has_composite:
            label_flank = flank_fn(row)
            if is_valid(label_flank):
                return label_flank, str(label_core) + "::" + str(label_flank)
            return label_flank, str(label_core)  # Flank unresolved: Core only
        return None, str(label_core)


    def centroid_flank(row):
        """Flank of an annotated centroid: its own, else its constituents'."""
        return row[LabelCols.label_flank_col] if is_valid(row[LabelCols.label_flank_col]) else assign_label_flank(row)


    def assign_labels(row):
        """Direct label (Rule 1) and label (Rules 1-4) of a row, by column: one pass, so every rule reads the row as annotated."""
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        ## Rule 1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if is_valid(direct_type) and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4]):
            label_core, label_core_readable = assign_direct_label_core(row)
            label_flank, label_combined = compose_label(row, label_core, assign_label_flank)
        direct_labels = {
            LabelCols.direct_label_core_col: label_core,
            LabelCols.direct_label_core_readable_col: label_core_readable,
            LabelCols.direct_label_col: label_combined,
        }

        ## Rule 2: Indirect, with annotated centroid: Use centroid's core (and flank, else constituent flank)
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(row[LabelCols.label_core_col])
            and is_valid(row[LabelCols.label_core_readable_col])
        ):
            label_core = str(row[LabelCols.label_core_col])
            label_core_readable = str(row[LabelCols.label_core_readable_col])
            label_flank, label_combined = compose_label(row, label_core, centroid_flank)

        ## Rule 3: Indirect, with annotated constituents: Use constituent's annotations
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(row[LabelCols.label_core_constit_all_col])
        ):
            label_core, label_core_readable = assign_label_core(row)
            label_flank, label_combined = compose_label(row, label_core, assign_label_flank)

        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown

        return {
            **direct_labels,
            LabelCols.label_core_col: label_core,
            LabelCols.label_core_readable_col: label_core_readable,
            LabelCols.label_flank_col: label_flank,
            LabelCols.label_col: label_combined,
        }

    # Add human-readable labels: Direct (Rule 1) and label (Rules 1-4), in one pass over the same rows
    label_cols = [
        LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.direct_label_col,
        LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
    ]
    mc_encode_filtered.metadata[label_cols] = pd.DataFrame(
        mc_encode_filtered.metadata.apply(assign_labels, axis=1).tolist(),
        index=mc_encode_filtered.metadata.index,
        columns=label_cols,
    )
    # Name: Metacluster - Removed
    mc_encode_removed[LabelCols.label_col] = mc_encode_removed[FilterArgs.filter_flag_type_col]


    # #### Name
    ## Label readable
    mc_encode_filtered[LabelCols.label_readable_col] = mc_encode_filtered[LabelCols.label_col]
    mc_encode_removed[LabelCols.label_readable_col] = mc_encode_removed[LabelCols.label_col]


    # #### Apply: QC - Passed
    if args.mc_qc_passed:
        # Load QC MotifCompendium: ENCODE QC Passed
        mc_qc_passed = MotifCompendium.load(args.mc_qc_passed, safe=args.safe)

        # Scale: Off
        ic_scale = False
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Assign label: From QC Filtered
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_encode_filtered,
            assign_from_mc=mc_qc_passed,
            from_label_col=LabelCols.label_readable_col,  # After v21: Change to label_readable_col
            save_col_prefix=MetadataCols.qc_col,
            min_score=FilterArgs.qc_min_score_filtered,
            max_submotifs=1,
            label_unsigned=False,
            save_images=False,
        )

        # Scale: On
        ic_scale = True
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Update label
        mask_qc = mc_encode_filtered[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_filtered
        mc_encode_filtered.metadata.loc[mask_qc, LabelCols.label_readable_col] = mc_encode_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_encode_filtered.metadata.loc[mask_qc, LabelCols.label_col] = mc_encode_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace("-like", "", regex=True)

        mc_encode_filtered.metadata.loc[mask_qc, MetadataCols.composite_col] = mc_encode_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0'].str.contains("::")
        mask_qc_composite = (mask_qc) & (mc_encode_filtered[MetadataCols.composite_col])
        mc_encode_filtered.metadata.loc[mask_qc_composite, MetadataCols.num_subunit_col] = 2

        mc_encode_filtered.metadata.loc[mask_qc, MetadataCols.checked_col] = True


    # #### Final Update: Manual

    # - E2F, GMEB, SP140: E2F/CpG
    # - ZNF: ZNF/CpG
    # - GATA, FOX: Homo-dimer
    # - ETV-NFAT, ETS: Homo-dimer
    # - AP1
    # - Unvalidated: Below 200 seqlets
    # - Composite
    if args.metadata_edited:
        # Load edited metadata
        mc_encode_metadata_edited = pd.read_csv(args.metadata_edited, index_col=0)

        # ``label`` is the canonical editable QC field.  Accept
        # ``label_readable`` only for legacy curated metadata exports, such
        # as the v23 sheet used for this run.
        if LabelCols.label_col in mc_encode_metadata_edited.columns:
            edited_label_col = LabelCols.label_col
        elif LabelCols.label_readable_col in mc_encode_metadata_edited.columns:
            edited_label_col = LabelCols.label_readable_col
            mc_encode_metadata_edited[LabelCols.label_col] = mc_encode_metadata_edited[edited_label_col]
        else:
            raise ValueError(
                "Edited metadata must contain the canonical "
                f"'{LabelCols.label_col}' column."
            )

        # Apply Filter: Removed
        mc_encode_metadata_edited[FilterArgs.filter_flag_col] = mc_encode_metadata_edited[edited_label_col].isin(FilterArgs.filter_flag_types[:-1])

        ## Update metadata: From flag_remove, label
        # Filtered: Filter type, Core, Flank, Composite, num_subunit
        mask_filtered = ~mc_encode_metadata_edited[FilterArgs.filter_flag_col]
        mc_encode_metadata_edited.loc[mask_filtered, FilterArgs.filter_flag_type_col] = None
        mc_encode_metadata_edited.loc[mask_filtered, LabelCols.label_core_col] = mc_encode_metadata_edited.loc[mask_filtered, LabelCols.label_col].str.split("::").str[0]

        # Composite
        mc_encode_metadata_edited[MetadataCols.num_subunit_col] = mc_encode_metadata_edited[LabelCols.label_col].str.split("::").str.len()
        mc_encode_metadata_edited[MetadataCols.composite_col] = (mc_encode_metadata_edited[MetadataCols.num_subunit_col] > 1)
        masK_composite = mc_encode_metadata_edited[MetadataCols.composite_col] & mask_filtered
        mc_encode_metadata_edited.loc[masK_composite, LabelCols.label_flank_col] = mc_encode_metadata_edited.loc[masK_composite, LabelCols.label_col].str.split("::").str[1]

        # Removed: Filter type, Core (Direct), Core, Flank, Label confidence
        mask_removed = mc_encode_metadata_edited[FilterArgs.filter_flag_col]
        mc_encode_metadata_edited.loc[mask_removed, FilterArgs.filter_flag_type_col] = mc_encode_metadata_edited.loc[mask_removed, LabelCols.label_col]
        mc_encode_metadata_edited.loc[mask_removed, LabelCols.direct_label_core_col] = None
        mc_encode_metadata_edited.loc[mask_removed, LabelCols.label_core_col] = None
        mc_encode_metadata_edited.loc[mask_removed, LabelCols.label_flank_col] = None
        mc_encode_metadata_edited.loc[mask_removed, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

        # Keep a legacy display column synchronized when it exists; ``label``
        # remains the canonical field for all QC updates.
        if LabelCols.label_readable_col in mc_encode_metadata_edited.columns:
            mc_encode_metadata_edited[LabelCols.label_readable_col] = mc_encode_metadata_edited[LabelCols.label_col]

        # Update MotifCompendium metadata
        mc_encode_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_encode_metadata_edited.set_index(MetadataCols.unique_id_col, inplace=True)

        mc_encode_filtered.metadata.update(mc_encode_metadata_edited)

        mc_encode_filtered.metadata.reset_index(inplace=True)
        mc_encode_metadata_edited.reset_index(inplace=True)


    # #### Manual: Rules
    ## Manage: Prename, simple name, label readable
    # Swap "high-GC/AT", "ETV-NFAT"
    mc_encode_filtered[LabelCols.label_readable_col] = mc_encode_filtered[LabelCols.label_readable_col].str.replace("GC/AT-bias", "high-GC/AT", regex=True)
    mc_encode_filtered[LabelCols.label_readable_col] = mc_encode_filtered[LabelCols.label_readable_col].str.replace("ETV-NFAT", "ETV-ETS", regex=True)

    ## Label: TF names only
    # Remove "{TF}-like" from readable label
    mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace("-like", "", regex=True)
    # Substitute: AP1 to JUN-FOS
    mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace("AP1", "JUN-FOS", regex=True)
    # Remove: HD/
    mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace("HD/", "", regex=True)
    # Remove: Non-canonical
    for filter_flag_type in FilterArgs.filter_flag_types:
        mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace(filter_flag_type, "", regex=True)

    # Remove: Delimiters
    mc_encode_filtered[LabelCols.label_col] = mc_encode_filtered[LabelCols.label_col].str.replace(r'(::|/|-|_)$', '', regex=True)

    # CpG
    mask_cpg = mc_encode_filtered[LabelCols.label_readable_col].str.contains("/CpG", na=False)
    mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.label_readable_col] = np.where(
        mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col].notna()
        & (mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] != ""),
        mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] + "/CpG",
        "CpG"
    )

    mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.label_col] = np.where(
        mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col].notna()
        & (mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] != ""),
        mc_encode_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col],
        "CpG"
    )

    # Filter:
    mask_filter = (
        mc_encode_filtered[FilterArgs.filter_flag_col] | 
        # mc_encode_filtered[direct_type_col].isin(filter_flag_types[:-1]) |
        mc_encode_filtered[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types[:-1])
    )
    mc_encode_filtered.metadata.loc[mask_filter, FilterArgs.filter_flag_col] = True
    mc_encode_filtered.metadata.loc[mask_filter, FilterArgs.filter_flag_type_col] = mc_encode_filtered.metadata.loc[mask_filter, LabelCols.label_readable_col]

    # Remove flagged clusters
    mc_encode_filtered_removed = mc_encode_filtered[mc_encode_filtered[FilterArgs.filter_flag_col]]
    mc_encode_filtered = mc_encode_filtered[~mc_encode_filtered[FilterArgs.filter_flag_col]]

    mc_encode_filtered_removed.metadata = mc_encode_filtered_removed.metadata[mc_encode_removed.columns()]
    for img in mc_encode_filtered_removed.images():
        if img not in mc_encode_removed.images():
            mc_encode_filtered_removed.delete_images(img)

    # Removed: Combine
    mc_encode_removed = MotifCompendium.combine(
        [mc_encode_removed, mc_encode_filtered_removed],
        safe=args.safe,
    )


    # #### Annotate: Best label
    ### Annotate: Best label
    mc_encode_filtered_metadata_updated = mc_encode_filtered.metadata.set_index(MetadataCols.unique_id_col)
    mc_encode_filtered_images_updated = pd.DataFrame(index=mc_encode_filtered_metadata_updated.index)
    mc_encode_filtered_images_existing = set(mc_encode_filtered.images())

    # Update Core, Core (Direct), Flank
    mc_encode_filtered[LabelCols.label_core_col] = np.where(
        mc_encode_filtered[LabelCols.label_col].str.contains("::", na=False),
        mc_encode_filtered[LabelCols.label_col].str.split("::").str[0],
        mc_encode_filtered[LabelCols.label_col]
    )
    mc_encode_filtered[LabelCols.direct_label_core_col] = np.where(
        mc_encode_filtered[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False),
        np.where(
            mc_encode_filtered[LabelCols.label_col].str.contains("::", na=False),
            mc_encode_filtered[LabelCols.label_col].str.split("::").str[0],
            mc_encode_filtered[LabelCols.label_col]
        ),
        None
    )
    mc_encode_filtered[LabelCols.label_flank_col] = np.where(
        mc_encode_filtered[LabelCols.label_col].str.contains("::", na=False),
        mc_encode_filtered[LabelCols.label_col].str.split("::").str[1],
        None
    )

    for i, row in mc_encode_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_encode_filtered_i = mc_encode_filtered[[i]]
        label_core_i = re.split(r'[-/]', row[LabelCols.label_core_col]) if pd.notna(row[LabelCols.label_core_col]) else []
        direct_label_core_i = re.split(r'[-/]', row[LabelCols.direct_label_core_col]) if pd.notna(row[LabelCols.direct_label_core_col]) else []
        label_flank_i = re.split(r'[-/]', row[LabelCols.label_flank_col]) if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        # Reference: Subset
        if direct_label_core_i:
            mc_ref_core_i = mc_ref_direct[
                (mc_ref_direct[ReferenceMatchCols.ref_tf_col].isin(direct_label_core_i)) |
                (mc_ref_direct[ReferenceMatchCols.ref_family_col].isin(direct_label_core_i))
            ]
        else:
            mc_ref_core_i = mc_ref_qc[
                (mc_ref_qc[ReferenceMatchCols.ref_tf_col].isin(label_core_i)) |
                (mc_ref_qc[ReferenceMatchCols.ref_family_col].isin(label_core_i))
            ]
        mc_ref_composite_i = mc_ref_qc[
            (mc_ref_qc[ReferenceMatchCols.ref_tf_col].isin(label_flank_i)) |
            (mc_ref_qc[ReferenceMatchCols.ref_family_col].isin(label_flank_i))
        ]

        # Annotate with name references
        if len(mc_ref_core_i) > 0:
            assign_direct_composite(
                target_mc=mc_encode_filtered_i,
                direct_reference_mc=mc_ref_core_i,
                indirect_reference_mc=mc_ref_composite_i,
                save_col_prefix=LabelCols.best_label_ref_col,
                min_score=0,
                max_submotifs=max_submotifs_i if len(mc_ref_composite_i) > 0 else 1,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )
    
        ## Update: Metadata, Images
        mc_encode_filtered_metadata_updated = mc_encode_filtered_metadata_updated.combine_first(mc_encode_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_encode_filtered_metadata_updated.update(mc_encode_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        new_image_cols = [image_col for image_col in mc_encode_filtered_i.images() if image_col.startswith(LabelCols.best_label_ref_col)]
        mc_encode_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_encode_filtered_i.get_images(image_col),
                index=mc_encode_filtered_i.metadata.set_index(MetadataCols.unique_id_col).index,
            )
            for image_col in new_image_cols
        })
        mc_encode_filtered_images_updated = mc_encode_filtered_images_updated.combine_first(mc_encode_filtered_i_new_images)
        mc_encode_filtered_images_updated.update(mc_encode_filtered_i_new_images)

    # Apply updates: Metadata, Images
    mc_encode_filtered.metadata = mc_encode_filtered_metadata_updated.reset_index()
    for image_col in mc_encode_filtered_images_updated.columns:
        mc_encode_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_encode_filtered_images_updated[image_col]],
        )
    del mc_encode_filtered_metadata_updated, mc_encode_filtered_images_updated, mc_encode_filtered_images_existing


    # #### Label confidence
    # Final rules
    mc_encode_filtered[MetadataCols.composite_col] = mc_encode_filtered[LabelCols.label_readable_col].str.contains("::")

    ## Set label readable type
    # Filtered
    label_conf_conditions = [
        mc_encode_filtered[DirectTypeArgs.direct_type_col].str.contains(DirectTypeArgs.direct_types[0], na=False),  # High: A
        mc_encode_filtered[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[1:3]), na=False),  # Medium: B-NZ, B-Z
        ~mc_encode_filtered[FilterArgs.filter_flag_col], # Low: Not flagged
        mc_encode_filtered[FilterArgs.filter_flag_col],   # Flagged: Flagged
    ]
    mc_encode_filtered[LabelCols.label_conf_col] = np.select(
        label_conf_conditions,  # High, Medium, low, Flagged
        LabelCols.label_confs, # High, Medium, low, Flagged
        default=LabelCols.label_confs[-1]  # Remaining: Flagged
    )

    # Removed
    mc_encode_removed[LabelCols.label_conf_col] = LabelCols.label_confs[-1]


    # #### Apply: QC - Low quality
    if args.mc_qc_lowq:
        # Load QC MotifCompendium: ENCODE QC Filters
        mc_qc_lowq = MotifCompendium.load(args.mc_qc_lowq, safe=args.safe)

        # Scale: Off
        ic_scale = False
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Assign label: From QC Filtered
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_encode_filtered,
            assign_from_mc=mc_qc_lowq,
            from_label_col=LabelCols.label_readable_col,  # After v21: Change to prename_col to label_readable_col,
            save_col_prefix=MetadataCols.qc_col,
            min_score=FilterArgs.qc_min_score_filtered,
            max_submotifs=1,
            label_unsigned=False,
            save_images=False,
        )

        # Scale: On
        ic_scale = True
        MotifCompendium.set_compute_options(
            max_chunk=args.max_chunk,
            max_cpus=args.max_cpus,
            use_gpu=args.use_gpu,
            ic_scale=ic_scale,
            fast_plotting=args.fast_plot,
        )

        # Update label
        mask_qc = mc_encode_filtered[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_filtered
        mc_encode_filtered.metadata.loc[mask_qc, MetadataCols.checked_col] = True
        mc_encode_filtered.metadata.loc[mask_qc, LabelCols.label_conf_col] = LabelCols.label_confs[2]


    # #### Naming
    ## Simple name: ({+_-})_{prename}_{H_M_L}
    sign = np.where(mc_encode_filtered[MetadataCols.posneg_col] == "pos", "pos", "neg")
    conf = np.select([
        mc_encode_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[0],
        mc_encode_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[1],
        mc_encode_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[2],
        ],
        LabelCols.label_confs_simple[:-1],
        default=LabelCols.label_confs_simple[2],
    )

    mc_encode_filtered[MetadataCols.prename_col] = np.where(
        ~mc_encode_filtered[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types),
        (mc_encode_filtered[LabelCols.label_readable_col]
            + "_" + sign
            + "_" + conf),
        mc_encode_filtered[LabelCols.label_readable_col],
    )

    mc_encode_removed[LabelCols.label_readable_col] = mc_encode_removed[FilterArgs.filter_flag_type_col]
    mc_encode_removed[MetadataCols.prename_col] = mc_encode_removed[FilterArgs.filter_flag_type_col]

    #### Update original MotifCompendium
    logging.info("Propagating ENCODE cluster labels to all levels...")
    ### (5) ENCODE cluster:
    # Combine: Update columns
    mc_encodees = [mc_encode_filtered, mc_encode_removed]

    mc_encode.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_encode_images_updated = pd.DataFrame(index=mc_encode.metadata.index)
    mc_encode_images_existing = set(mc_encode.images())

    # Build updates: Metadata, Images
    for mc_encode_i in mc_encodees:
        # Metadata
        mc_encode.metadata = mc_encode.metadata.combine_first(mc_encode_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_encode.metadata.update(mc_encode_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
        new_image_cols = [image_col for image_col in mc_encode_i.images() if image_col not in mc_encode_images_existing]
        mc_encode_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_encode_i.get_images(image_col),
                index=mc_encode_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_encode_images_updated = mc_encode_images_updated.combine_first(mc_encode_i_new_images)
        mc_encode_images_updated.update(mc_encode_i_new_images)

    # Apply updates: Metadata, Images
    mc_encode.metadata.reset_index(inplace=True)
    for image_col in mc_encode_images_updated.columns:
        mc_encode.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_encode_images_updated[image_col]],
        )
    del mc_encode_images_updated, mc_encode_images_existing

    # # Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_encode_i in mc_encodees:
    #     metadata_cols = list(set(metadata_cols + mc_encode_i.columns()))
    #     image_cols = list(set(image_cols + mc_encode_i.images()))
    # for mc_encode_i in mc_encodees:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_encode_i.columns():
    #             mc_encode_i.metadata[metadata_col] = None
    #     for logo_i in ['logo (fwd)', 'logo (rev)']:
    #         mc_encode_i.add_logos(
    #             motifs=mc_encode_i.motifs,
    #             image_name=logo_i,
    #             trim=0,
    #         )
    #     for image_col in image_cols:
    #         if image_col not in mc_encode_i.images():
    #             mc_encode_i.add_logos(
    #                 motifs=np.zeros_like(mc_encode_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Update original MotifCompendium: Metacluster
    # mc_encode = MotifCompendium.combine(mc_encodees, safe=safe)


    # #### Assign name
    ## Rename: {prename_col}_{counter}
    mc_encode.metadata[MetadataCols.counter_col] = mc_encode.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_encode.metadata[MetadataCols.name_new_col] = mc_encode.metadata[MetadataCols.prename_col] + "_" + mc_encode.metadata[MetadataCols.counter_col].astype(str)
    mc_encode.metadata[MetadataCols.name_col] = mc_encode.metadata[MetadataCols.name_new_col]
    mc_encode.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    # Simple name = Prename
    mc_encode[MetadataCols.simple_name_col] = mc_encode[MetadataCols.prename_col]

    # Map: Atlascluster to ENCODE cluster
    atlascluster2encode_name_map = dict(zip(mc_encode[ClusterArgs.source_cluster_col], mc_encode[MetadataCols.name_col]))
    atlascluster2encode_simple_map = dict(zip(mc_encode[ClusterArgs.source_cluster_col], mc_encode[MetadataCols.simple_name_col]))

    mc_atlascluster_combined[VisualizeArgs.encodecluster_name_col] = mc_atlascluster_combined[encodecluster_cluster_col].map(atlascluster2encode_name_map)
    mc_atlascluster_combined[MetadataCols.simple_name_col] = mc_atlascluster_combined[encodecluster_cluster_col].map(atlascluster2encode_simple_map).fillna(mc_atlascluster_combined[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)


    # #### Visualize HTML
    # # Add: Notes
    # mc_encode.metadata[notes_col] = None
    # mc_encode.metadata[checked_col] = None

    # Create HTML
    html_columns_encodecluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_encode.columns() + mc_encode.images())]
    html_column_desc_dict_encodecluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_encodecluster}

    os.makedirs(os.path.dirname(html_encode_path), exist_ok=True)
    mc_encode.summary_table_html(
        html_out=html_encode_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    os.makedirs(os.path.dirname(html_encode_filtered_path), exist_ok=True)
    mc_encode_filtered = mc_encode[~mc_encode[FilterArgs.filter_flag_col]]
    mc_encode_filtered.summary_table_html(
        html_out=html_encode_filtered_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    os.makedirs(os.path.dirname(html_encode_removed_path), exist_ok=True)
    mc_encode_removed = mc_encode[mc_encode[FilterArgs.filter_flag_col]]
    mc_encode_removed.summary_table_html(
        html_out=html_encode_removed_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    # # Create HTML: Edited
    # html_columns_encodecluster = [html_col for html_col in html_columns if html_col in (mc_encode.columns() + mc_encode.images())]
    # html_column_desc_dict_encodecluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_encodecluster}

    # os.makedirs(os.path.dirname(html_encode_edited_path), exist_ok=True)
    # mc_encode.summary_table_html(
    #     html_out=html_encode_edited_path,
    #     columns=html_columns_encodecluster,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_encodecluster,
    # )

    # os.makedirs(os.path.dirname(html_encode_edited_filtered_path), exist_ok=True)
    # mc_encode_filtered = mc_encode[~mc_encode[filter_flag_col]]
    # mc_encode_filtered.summary_table_html(
    #     html_out=html_encode_edited_filtered_path,
    #     columns=html_columns_encodecluster+['checked', f'{qc_col}_name0', f'{qc_col}_score0'],
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_encodecluster,
    # )

    # os.makedirs(os.path.dirname(html_encode_edited_removed_path), exist_ok=True)
    # mc_encode_removed = mc_encode[mc_encode[filter_flag_col]]
    # mc_encode_removed.summary_table_html(
    #     html_out=html_encode_edited_removed_path,
    #     columns=html_columns_encodecluster + [filter_flag_col, filter_flag_type_col, f'{qc_col}_score0'],
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_encodecluster,
    # )

    # Clean up
    mc_encode.metadata = mc_encode.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_encode_filtered.metadata = mc_encode_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_encode_removed.metadata = mc_encode_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_atlascluster_combined.metadata = mc_atlascluster_combined.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')

    # Save MotifCompendium: ENCODE
    os.makedirs(os.path.dirname(mc_encode_path), exist_ok=True)
    mc_encode.save(mc_encode_path)

    os.makedirs(os.path.dirname(mc_encode_filtered_path), exist_ok=True)
    mc_encode_filtered.save(mc_encode_filtered_path)

    os.makedirs(os.path.dirname(mc_encode_removed_path), exist_ok=True)
    mc_encode_removed.save(mc_encode_removed_path)

    os.makedirs(os.path.dirname(mc_atlascluster_combined_path), exist_ok=True)
    mc_atlascluster_combined.save(mc_atlascluster_combined_path)

    # Load MotifCompendium: ENCODE
    # mc_atlascluster_combined = MotifCompendium.load(mc_atlascluster_combined_path, safe=safe)


    # ### Export
    # Export as MEME txt
    os.makedirs(os.path.dirname(meme_encode_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_encode,
        name_col=MetadataCols.name_col,
        save_loc=meme_encode_path,
        trim_importance=(1/(2*mc_encode.motifs.shape[1])),
    )

    os.makedirs(os.path.dirname(meme_encode_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_encode_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_encode_filtered_path,
        trim_importance=(1/(2*mc_encode_filtered.motifs.shape[1])),
    )

    os.makedirs(os.path.dirname(meme_encode_removed_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_encode_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_encode_removed_path,
        trim_importance=(1/(2*mc_encode_removed.motifs.shape[1])),
    )

    # Metadata TSV
    os.makedirs(os.path.dirname(metadata_atlascluster_combined_path), exist_ok=True)
    mc_atlascluster_combined.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_atlascluster_combined_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(metadata_encode_path), exist_ok=True)
    mc_encode.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_encode_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(metadata_encode_path), exist_ok=True)
    mc_encode_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_encode_filtered_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(metadata_encode_path), exist_ok=True)
    mc_encode_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_encode_removed_path, sep="\t", index=False)


    # ### Motif index
    # Load MotifCompendium
    mc_counts_tf = MotifCompendium.load(mc_counts_tf_path, safe=args.safe)
    mc_cluster_counts_tf = MotifCompendium.load(mc_cluster_counts_tf_path, safe=args.safe)
    mc_metacluster_counts_tf = MotifCompendium.load(mc_metacluster_counts_tf_path, safe=args.safe)

    mc_profile_tf = MotifCompendium.load(mc_profile_tf_path, safe=args.safe)
    mc_cluster_profile_tf = MotifCompendium.load(mc_cluster_profile_tf_path, safe=args.safe)
    mc_metacluster_profile_tf = MotifCompendium.load(mc_metacluster_profile_tf_path, safe=args.safe)

    mc_atlascluster_tf = MotifCompendium.load(mc_atlascluster_tf_path, safe=args.safe)


    mc_counts_chrom = MotifCompendium.load(mc_counts_chrom_path, safe=args.safe)
    mc_cluster_counts_chrom = MotifCompendium.load(mc_cluster_counts_chrom_path, safe=args.safe)
    mc_metacluster_counts_chrom = MotifCompendium.load(mc_metacluster_counts_chrom_path, safe=args.safe)

    mc_profile_chrom = MotifCompendium.load(mc_profile_chrom_path, safe=args.safe)
    mc_cluster_profile_chrom = MotifCompendium.load(mc_cluster_profile_chrom_path, safe=args.safe)
    mc_metacluster_profile_chrom = MotifCompendium.load(mc_metacluster_profile_chrom_path, safe=args.safe)

    mc_atlascluster_chrom = MotifCompendium.load(mc_atlascluster_chrom_path, safe=args.safe)
    mc_counts_procap = MotifCompendium.load(mc_counts_procap_path, safe=args.safe)
    mc_cluster_counts_procap = MotifCompendium.load(mc_cluster_counts_procap_path, safe=args.safe)
    mc_metacluster_counts_procap = MotifCompendium.load(mc_metacluster_counts_procap_path, safe=args.safe)

    mc_profile_procap = MotifCompendium.load(mc_profile_procap_path, safe=args.safe)
    mc_cluster_profile_procap = MotifCompendium.load(mc_cluster_profile_procap_path, safe=args.safe)
    mc_metacluster_profile_procap = MotifCompendium.load(mc_metacluster_profile_procap_path, safe=args.safe)

    mc_atlascluster_procap = MotifCompendium.load(mc_atlascluster_procap_path, safe=args.safe)

    # FiNeMo and peaks paths, per experiment and head, for the hits indexing and annotation below
    hits_finemo_path_cols = [MetadataCols.id_col, MetadataCols.head_col, MetadataCols.finemo_path_col]
    hits_finemo_paths_df = pd.concat(
        [
            mc_head.metadata[[col for col in hits_finemo_path_cols + [MetadataCols.peaks_path_col] if col in mc_head.columns()]]
            .assign(**{VisualizeArgs.atlas_col: atlas_id})
            for atlas_id, mc_head in (
                (EncodeCombineArgs.tfatlas_id, mc_counts_tf), (EncodeCombineArgs.tfatlas_id, mc_profile_tf),
                (EncodeCombineArgs.chromatlas_id, mc_counts_chrom), (EncodeCombineArgs.chromatlas_id, mc_profile_chrom),
                (EncodeCombineArgs.procapatlas_id, mc_counts_procap), (EncodeCombineArgs.procapatlas_id, mc_profile_procap),
            )
            if MetadataCols.finemo_path_col in mc_head.columns()
        ] or [pd.DataFrame(columns=hits_finemo_path_cols + [VisualizeArgs.atlas_col])],
        ignore_index=True,
    )
    if MetadataCols.peaks_path_col not in hits_finemo_paths_df.columns:
        hits_finemo_paths_df[MetadataCols.peaks_path_col] = None

    # The legacy Chrom profile indexed compendium predates the atlascluster
    # annotation that the corresponding TF profile already carries. Its
    # motifs still retain their metacluster assignment, so restore the
    # derived atlascluster name before applying the shared ENCODE mapping.
    if VisualizeArgs.atlascluster_name_col not in mc_profile_chrom.columns():
        metacluster_to_atlascluster = dict(zip(
            mc_metacluster_profile_chrom[MetadataCols.name_col],
            mc_metacluster_profile_chrom[VisualizeArgs.atlascluster_name_col],
        ))
        mc_profile_chrom[VisualizeArgs.atlascluster_name_col] = mc_profile_chrom[
            VisualizeArgs.metacluster_name_col
        ].map(metacluster_to_atlascluster)
        if mc_profile_chrom[VisualizeArgs.atlascluster_name_col].notna().sum() == 0:
            raise ValueError(
                "Could not recover atlascluster_name for the legacy Chrom "
                "profile compendium from its metacluster assignments."
            )

    # Update Atlasclusters
    for mc_atlascluster_i in [mc_atlascluster_tf, mc_atlascluster_chrom, mc_atlascluster_procap]:
        for metadata_col in mc_atlascluster_combined.columns():
            if metadata_col not in mc_atlascluster_i.columns():
                mc_atlascluster_i.metadata[metadata_col] = None
        mc_atlascluster_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_atlascluster_i.metadata.update(mc_atlascluster_combined.metadata.set_index(MetadataCols.unique_id_col))
        mc_atlascluster_i.metadata.reset_index(inplace=True)

    ## Map: Atlascluster to ENCODE cluster: Name, Simple name
    mc_atlascluster_tf[VisualizeArgs.encodecluster_name_col] = mc_atlascluster_tf[encodecluster_cluster_col].map(atlascluster2encode_name_map)
    mc_atlascluster_tf[MetadataCols.simple_name_col] = mc_atlascluster_tf[encodecluster_cluster_col].map(atlascluster2encode_simple_map).fillna(mc_atlascluster_tf[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)

    mc_atlascluster_chrom[VisualizeArgs.encodecluster_name_col] = mc_atlascluster_chrom[encodecluster_cluster_col].map(atlascluster2encode_name_map)
    mc_atlascluster_chrom[MetadataCols.simple_name_col] = mc_atlascluster_chrom[encodecluster_cluster_col].map(atlascluster2encode_simple_map).fillna(mc_atlascluster_chrom[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)
    mc_atlascluster_procap[VisualizeArgs.encodecluster_name_col] = mc_atlascluster_procap[encodecluster_cluster_col].map(atlascluster2encode_name_map)
    mc_atlascluster_procap[MetadataCols.simple_name_col] = mc_atlascluster_procap[encodecluster_cluster_col].map(atlascluster2encode_simple_map).fillna(mc_atlascluster_procap[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)

    ## Map: Motif, Cluster, Metacluster to ENCODE cluster: Name, Simple name
    metacluster2atlascluster_encode_map = dict(zip(mc_atlascluster_combined[MetadataCols.name_col], mc_atlascluster_combined[VisualizeArgs.encodecluster_name_col]))
    metacluster2atlascluster_simple_map = dict(zip(mc_atlascluster_combined[MetadataCols.name_col], mc_atlascluster_combined[MetadataCols.simple_name_col]))

    ##  Apply map: Motif, Cluster, Metacluster to ENCODE cluster: Name, Simple name
    mc_metacluster_counts_tf[VisualizeArgs.encodecluster_name_col] = mc_metacluster_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_counts_tf[MetadataCols.simple_name_col] = mc_metacluster_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_metacluster_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_tf[VisualizeArgs.encodecluster_name_col] = mc_cluster_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_counts_tf[MetadataCols.simple_name_col] = mc_cluster_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_cluster_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_tf[VisualizeArgs.encodecluster_name_col] = mc_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_counts_tf[MetadataCols.simple_name_col] = mc_counts_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_metacluster_profile_tf[VisualizeArgs.encodecluster_name_col] = mc_metacluster_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_profile_tf[MetadataCols.simple_name_col] = mc_metacluster_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_metacluster_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_tf[VisualizeArgs.encodecluster_name_col] = mc_cluster_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_profile_tf[MetadataCols.simple_name_col] = mc_cluster_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_cluster_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_tf[VisualizeArgs.encodecluster_name_col] = mc_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_profile_tf[MetadataCols.simple_name_col] = mc_profile_tf[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')


    mc_metacluster_counts_chrom[VisualizeArgs.encodecluster_name_col] = mc_metacluster_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_counts_chrom[MetadataCols.simple_name_col] = mc_metacluster_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_metacluster_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_chrom[VisualizeArgs.encodecluster_name_col] = mc_cluster_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_counts_chrom[MetadataCols.simple_name_col] = mc_cluster_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_cluster_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_chrom[VisualizeArgs.encodecluster_name_col] = mc_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_counts_chrom[MetadataCols.simple_name_col] = mc_counts_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_metacluster_profile_chrom[VisualizeArgs.encodecluster_name_col] = mc_metacluster_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_profile_chrom[MetadataCols.simple_name_col] = mc_metacluster_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_metacluster_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_chrom[VisualizeArgs.encodecluster_name_col] = mc_cluster_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_profile_chrom[MetadataCols.simple_name_col] = mc_cluster_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_cluster_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_chrom[VisualizeArgs.encodecluster_name_col] = mc_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_profile_chrom[MetadataCols.simple_name_col] = mc_profile_chrom[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_metacluster_counts_procap[VisualizeArgs.encodecluster_name_col] = mc_metacluster_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_counts_procap[MetadataCols.simple_name_col] = mc_metacluster_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_metacluster_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_procap[VisualizeArgs.encodecluster_name_col] = mc_cluster_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_counts_procap[MetadataCols.simple_name_col] = mc_cluster_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_cluster_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_procap[VisualizeArgs.encodecluster_name_col] = mc_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_counts_procap[MetadataCols.simple_name_col] = mc_counts_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_metacluster_profile_procap[VisualizeArgs.encodecluster_name_col] = mc_metacluster_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_metacluster_profile_procap[MetadataCols.simple_name_col] = mc_metacluster_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_metacluster_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_metacluster_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_metacluster_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_procap[VisualizeArgs.encodecluster_name_col] = mc_cluster_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_cluster_profile_procap[MetadataCols.simple_name_col] = mc_cluster_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_cluster_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_cluster_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_procap[VisualizeArgs.encodecluster_name_col] = mc_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_encode_map)
    mc_profile_procap[MetadataCols.simple_name_col] = mc_profile_procap[VisualizeArgs.atlascluster_name_col].map(metacluster2atlascluster_simple_map).fillna(mc_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    ## Filter flag: If no encodecluster_name_col, then True
    mc_counts_tf[FilterArgs.filter_flag_col] = ~mc_counts_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_counts_tf[FilterArgs.filter_flag_col] = ~mc_cluster_counts_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_counts_tf[FilterArgs.filter_flag_col] = ~mc_metacluster_counts_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_profile_tf[FilterArgs.filter_flag_col] = ~mc_profile_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_profile_tf[FilterArgs.filter_flag_col] = ~mc_cluster_profile_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_profile_tf[FilterArgs.filter_flag_col] = ~mc_metacluster_profile_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_atlascluster_tf[FilterArgs.filter_flag_col] = ~mc_atlascluster_tf[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])


    mc_counts_chrom[FilterArgs.filter_flag_col] = ~mc_counts_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_counts_chrom[FilterArgs.filter_flag_col] = ~mc_cluster_counts_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_counts_chrom[FilterArgs.filter_flag_col] = ~mc_metacluster_counts_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_profile_chrom[FilterArgs.filter_flag_col] = ~mc_profile_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_profile_chrom[FilterArgs.filter_flag_col] = ~mc_cluster_profile_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_profile_chrom[FilterArgs.filter_flag_col] = ~mc_metacluster_profile_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_atlascluster_chrom[FilterArgs.filter_flag_col] = ~mc_atlascluster_chrom[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_counts_procap[FilterArgs.filter_flag_col] = ~mc_counts_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_counts_procap[FilterArgs.filter_flag_col] = ~mc_cluster_counts_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_counts_procap[FilterArgs.filter_flag_col] = ~mc_metacluster_counts_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_profile_procap[FilterArgs.filter_flag_col] = ~mc_profile_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_cluster_profile_procap[FilterArgs.filter_flag_col] = ~mc_cluster_profile_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
    mc_metacluster_profile_procap[FilterArgs.filter_flag_col] = ~mc_metacluster_profile_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    mc_atlascluster_procap[FilterArgs.filter_flag_col] = ~mc_atlascluster_procap[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])

    ## Confidence: If no encodecluster_name_col, then Flag
    mc_encode_filtered_metadata = mc_encode_filtered.metadata.set_index(MetadataCols.name_col)

    mc_counts_tf.metadata[LabelCols.label_conf_col] = mc_counts_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_tf.metadata[LabelCols.label_conf_col] = mc_cluster_counts_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_counts_tf.metadata[LabelCols.label_conf_col] = mc_metacluster_counts_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_tf.metadata[LabelCols.label_conf_col] = mc_profile_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_tf.metadata[LabelCols.label_conf_col] = mc_cluster_profile_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_profile_tf.metadata[LabelCols.label_conf_col] = mc_metacluster_profile_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_atlascluster_tf.metadata[LabelCols.label_conf_col] = mc_atlascluster_tf.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])


    mc_counts_chrom.metadata[LabelCols.label_conf_col] = mc_counts_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_chrom.metadata[LabelCols.label_conf_col] = mc_cluster_counts_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_counts_chrom.metadata[LabelCols.label_conf_col] = mc_metacluster_counts_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_chrom.metadata[LabelCols.label_conf_col] = mc_profile_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_chrom.metadata[LabelCols.label_conf_col] = mc_cluster_profile_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_profile_chrom.metadata[LabelCols.label_conf_col] = mc_metacluster_profile_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_atlascluster_chrom.metadata[LabelCols.label_conf_col] = mc_atlascluster_chrom.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_counts_procap.metadata[LabelCols.label_conf_col] = mc_counts_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_procap.metadata[LabelCols.label_conf_col] = mc_cluster_counts_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_counts_procap.metadata[LabelCols.label_conf_col] = mc_metacluster_counts_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_procap.metadata[LabelCols.label_conf_col] = mc_profile_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_procap.metadata[LabelCols.label_conf_col] = mc_cluster_profile_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_metacluster_profile_procap.metadata[LabelCols.label_conf_col] = mc_metacluster_profile_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_atlascluster_procap.metadata[LabelCols.label_conf_col] = mc_atlascluster_procap.metadata[VisualizeArgs.encodecluster_name_col].map(mc_encode_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    # Motif index
    mc_motif_index = pd.concat([
        mc_counts_tf.metadata,
        mc_profile_tf.metadata,
        mc_counts_chrom.metadata,
        mc_profile_chrom.metadata,
        mc_counts_procap.metadata,
        mc_profile_procap.metadata,
    ], axis=0).reset_index(drop=True)

    # Add motif category type: "direct-tf", "indirect-tf", "tf", "non-canonical"
    mc_motif_index[DirectTypeArgs.category_type_col] = np.select(
        [
            # TF Atlas: Direct
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (mc_motif_index[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (~mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Indirect
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (~mc_motif_index[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (~mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Unvalidated
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (~mc_motif_index[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Non-canonical
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & ((mc_motif_index[FilterArgs.filter_flag_col] == True)
             | (~mc_motif_index[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
               ))),
            # Chromatin Atlas: Canonical
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (~mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Unvalidated
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # Chromatin Atlas: Non-canonical
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & ((mc_motif_index[FilterArgs.filter_flag_col] == True)
             | (~mc_motif_index[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
               ))),
            # ProCap Atlas: Canonical
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (~mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # ProCap Atlas: Unvalidated
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & (mc_motif_index[FilterArgs.filter_flag_col] == False)
             & (mc_motif_index[VisualizeArgs.encodecluster_name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # ProCap Atlas: Non-canonical
            ((mc_motif_index[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & ((mc_motif_index[FilterArgs.filter_flag_col] == True)
             | (~mc_motif_index[VisualizeArgs.encodecluster_name_col].isin(mc_encode_filtered[MetadataCols.name_col])
               ))),
        ],
        [
            DirectTypeArgs.category_types[0],
            DirectTypeArgs.category_types[1],
            DirectTypeArgs.category_types[3],
            DirectTypeArgs.category_types[4],
            DirectTypeArgs.category_types[2],
            DirectTypeArgs.category_types[3],
            DirectTypeArgs.category_types[4],
            DirectTypeArgs.category_types[2],
            DirectTypeArgs.category_types[3],
            DirectTypeArgs.category_types[4],
        ],
        default=DirectTypeArgs.category_types[4]
    )

    # Export: Motif index
    mc_motif_index[VisualizeArgs.motif_index_cols].to_csv(metadata_motif_index_path, sep="\t", index=False)
    # mc_motif_index[motif_cateogory_cols].to_csv(mc_motif_cateogory_tsv, sep="\t", index=False)

    # MotifCompendium objects
    mc_alls = {
        'TF': {
            'combined': mc_atlascluster_tf,
            'counts': {
                'metacluster': mc_metacluster_counts_tf,
                'cluster': mc_cluster_counts_tf,
                'motif': mc_counts_tf,
            },
            'profile': {
                'metacluster': mc_metacluster_profile_tf,
                'cluster': mc_cluster_profile_tf,
                'motif': mc_profile_tf,
            },
        },
        'Chrom': {
            'combined': mc_atlascluster_chrom,
            'counts': {
                'metacluster': mc_metacluster_counts_chrom,
                'cluster': mc_cluster_counts_chrom,
                'motif': mc_counts_chrom,
            },
            'profile': {
                'metacluster': mc_metacluster_profile_chrom,
                'cluster': mc_cluster_profile_chrom,
                'motif': mc_profile_chrom,
            },
        },
        'ProCap': {
            'combined': mc_atlascluster_procap,
            'counts': {
                'metacluster': mc_metacluster_counts_procap,
                'cluster': mc_cluster_counts_procap,
                'motif': mc_counts_procap,
            },
            'profile': {
                'metacluster': mc_metacluster_profile_procap,
                'cluster': mc_cluster_profile_procap,
                'motif': mc_profile_procap,
            },
        },
    }

    level_suffixs = {
        'metacluster': EncodeCombineArgs.suffix_metacluster,
        'cluster': EncodeCombineArgs.suffix_cluster,
        'motif': EncodeCombineArgs.suffix_motif,
    }
    mc_suffixs = {
        atlas: {'combined': EncodeCombineArgs.suffix_atlascluster, 'counts': level_suffixs, 'profile': level_suffixs}
        for atlas in mc_alls
    }

    # Clean up
    for atlas, atlas_dict in mc_alls.items():
        # Combined
        mc_combined = atlas_dict['combined']
        mc_combined.metadata = mc_combined.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')

        # Head
        for head in ['counts', 'profile']:
            head_dict = atlas_dict[head]

            # Level
            for level, mc_level in head_dict.items():
                mc_level.metadata = mc_level.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')

    # Save MotifCompendium over a file that later stages load.  On this filesystem, h5py can exceptionally raise
    # EACCES while closing a very large file.  Write transactionally so a failed close cannot corrupt it; retaining the
    # previous file is safer than replacing it with an unverified partial one.
    def save_replace(mc, save_path):
        # MotifCompendium.save appends ``.mc`` when absent, so retain that
        # extension on the temporary path to make the replacement target
        # unambiguous.
        temp_path = f"{save_path}.tmp.mc"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            mc.save(temp_path)
            os.replace(temp_path, save_path)
        except RuntimeError as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if "unable to close file" not in str(exc):
                raise
            logging.warning(
                "h5py close reported EACCES while writing %s; retaining the "
                "previous compendium and continuing.",
                save_path,
            )


    # Export per-atlas compendia, back into the upstream atlas pipeline directories.
    # One pass over every (atlas, head, level, flag) combination writes all four
    # artefacts, so each flag slice is taken once instead of once per artefact.
    #   File name: {file_general}{level_suffix}{flag_suffix}{suffix}{filetype}: the atlas pipelines' own files, overwritten
    logging.info("Exporting per-atlas compendia, back into the atlas pipeline directories...")


    def export_level_flag(mc_level_flag, mc_level_dir, mc_level_suffix, suffix_flag, level):
        """Write MEME / metadata TSV / HTML / MoDISco H5 (--export-modisco) / .mc for one flag slice."""

        def out_path(subdir, suffix, suffix_filetype):
            file_name = (EncodeCombineArgs.file_general + mc_level_suffix + suffix_flag
                         + suffix + suffix_filetype)
            path = os.path.join(mc_level_dir, subdir, file_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path

        # MEME txt
        path = out_path('export', '', '.meme.txt')
        utils_analysis.export_compendium_meme(
            mc=mc_level_flag,
            name_col=MetadataCols.name_col,
            save_loc=path,
            trim_importance=(1/(2*mc_level_flag.motifs.shape[1])),
        )
        if args.verbose:
            logging.info(f'Saved at: {path}')

        # MoDISco HDF5
        if args.export_modisco:
            path = out_path('export', '', '.h5')
            utils_analysis.export_compendium_modisco(
                mc=mc_level_flag,
                name_col=MetadataCols.name_col,
                save_loc=path,
            )
            if args.verbose:
                logging.info(f'Saved at: {path}')

        # Metadata TSV
        path = out_path('metadata', EncodeCombineArgs.suffix_metadata, '.tsv')
        mc_level_flag.metadata.to_csv(path, sep="\t", index=False)
        if args.verbose:
            logging.info(f'Saved at: {path}')

        # HTML summary table: Clusters only, motifs have no logos to summarise
        if level != 'motif':
            path = out_path('html', '', '.html')
            html_columns_level = [html_col for html_col in VisualizeArgs.html_columns
                                  if html_col in (mc_level_flag.columns() + mc_level_flag.images())]
            mc_level_flag.summary_table_html(
                html_out=path,
                columns=html_columns_level,
                logo_trimming=VisualizeArgs.html_logo_trim,
                editable=True,
                column_desc_dict=html_column_desc_dict_encodecluster,
            )
            if args.verbose:
                logging.info(f'Saved at: {path}')

        # MotifCompendium object
        path = out_path('mc', '', '.mc')
        save_replace(mc_level_flag, path)
        if args.verbose:
            logging.info(f'Saved at: {path}')


    for atlas, atlas_dict in mc_alls.items():
        # Combined (atlascluster) level
        level = 'combined'
        mc_level = atlas_dict[level]
        for suffix_flag, flag_bool in EncodeCombineArgs.flags:
            export_level_flag(
                mc_level_flag=mc_level[mc_level[FilterArgs.filter_flag_col] == flag_bool],
                mc_level_dir=mc_dirs[atlas][level],
                mc_level_suffix=mc_suffixs[atlas][level],
                suffix_flag=suffix_flag,
                level=level,
            )

        # Head: counts, profile
        for head in ['counts', 'profile']:
            for level, mc_level in atlas_dict[head].items():
                for suffix_flag, flag_bool in EncodeCombineArgs.flags:
                    export_level_flag(
                        mc_level_flag=mc_level[mc_level[FilterArgs.filter_flag_col] == flag_bool],
                        mc_level_dir=mc_dirs[atlas][head],
                        mc_level_suffix=mc_suffixs[atlas][head][level],
                        suffix_flag=suffix_flag,
                        level=level,
                    )

    # Save MotifCompendium: with the ENCODE clusters, over the atlases' motifcompendium_*_all.mc
    for mc, save_path in (
        (mc_counts_tf, mc_counts_tf_path),
        (mc_cluster_counts_tf, mc_cluster_counts_tf_path),
        (mc_metacluster_counts_tf, mc_metacluster_counts_tf_path),
        (mc_profile_tf, mc_profile_tf_path),
        (mc_cluster_profile_tf, mc_cluster_profile_tf_path),
        (mc_metacluster_profile_tf, mc_metacluster_profile_tf_path),
        (mc_atlascluster_tf, mc_atlascluster_tf_path),
        (mc_counts_chrom, mc_counts_chrom_path),
        (mc_cluster_counts_chrom, mc_cluster_counts_chrom_path),
        (mc_metacluster_counts_chrom, mc_metacluster_counts_chrom_path),
        (mc_profile_chrom, mc_profile_chrom_path),
        (mc_cluster_profile_chrom, mc_cluster_profile_chrom_path),
        (mc_metacluster_profile_chrom, mc_metacluster_profile_chrom_path),
        (mc_atlascluster_chrom, mc_atlascluster_chrom_path),
        (mc_counts_procap, mc_counts_procap_path),
        (mc_cluster_counts_procap, mc_cluster_counts_procap_path),
        (mc_metacluster_counts_procap, mc_metacluster_counts_procap_path),
        (mc_profile_procap, mc_profile_procap_path),
        (mc_cluster_profile_procap, mc_cluster_profile_procap_path),
        (mc_metacluster_profile_procap, mc_metacluster_profile_procap_path),
        (mc_atlascluster_procap, mc_atlascluster_procap_path),
    ):
        save_replace(mc, save_path)


    #### Aggregate: Metadata, hits ----------------------------------
    logging.info("Aggregating metadata and motif hits...")

    # Load the original per-atlas metadata.  This stage has no ``--metadata``
    # argument: the final ENCODE compendium contains motifs from both atlases,
    # so the source annotations must be assembled from their two initial
    # metadata tables rather than from the curated QC-edit sheet.
    def read_initial_metadata(metadata_path, atlas_name):
        if metadata_path.endswith(".csv"):
            metadata_df = pd.read_csv(metadata_path, dtype="string")
        elif metadata_path.endswith(".tsv"):
            metadata_df = pd.read_csv(metadata_path, sep="\t", dtype="string")
        else:
            raise ValueError(
            "Initial metadata must be a CSV or TSV file: "
            f"{metadata_path}"
            )
        metadata_df[VisualizeArgs.atlas_col] = atlas_name
        return metadata_df

    encode_metadata_df = pd.concat(
        (
            read_initial_metadata(args.tf_init_metadata, EncodeCombineArgs.tfatlas_id),
            read_initial_metadata(args.chrom_init_metadata, EncodeCombineArgs.chromatlas_id),
            read_initial_metadata(args.procap_init_metadata, EncodeCombineArgs.procapatlas_id),
        ),
        ignore_index=True,
        sort=False,
    )

    # Metadata: Recompile
    metadata_recompile_cols = [
        MetadataCols.biosample_col,
        CellTypeCols.celltype_col,
        CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col,
        CellTypeCols.slim_organ_col,
        CellTypeCols.slim_dev_col,
        CellTypeCols.slim_system_col,
    ]
    required_metadata_cols = [MetadataCols.id_col, *metadata_recompile_cols]
    missing_metadata_cols = [
        col for col in required_metadata_cols if col not in encode_metadata_df.columns
    ]
    if missing_metadata_cols:
        raise ValueError(
            "Initial atlas metadata is missing required columns: "
            + ", ".join(missing_metadata_cols)
        )

    # Clean up metadata
    encode_metadata_df[metadata_recompile_cols] = (
        encode_metadata_df[metadata_recompile_cols]
        .astype("string")
        .apply(lambda s: (
            s.str.replace(r"[ '\t]+", "-", regex=True)
                .str.replace(r"-+", "-", regex=True)
                .str.strip("-")
        ))
    )

    # Recompile metadata: 
    def join_unique(values):
        values = values.dropna().astype(str)
        return ",".join(sorted(set(values)))

    def recompile_metadata(mc_encode_i, encode_metadata_df, recompile_cols):
        rows_to_ids = (
            mc_encode_i.metadata[[MetadataCols.id_col]]
            .assign(_row=mc_encode_i.metadata.index)
            .assign(_encid=lambda d: d[MetadataCols.id_col].astype(str).str.split(","))
            .explode("_encid")
        )

        merged = rows_to_ids.merge(
            encode_metadata_df,
            left_on="_encid",
            right_on=MetadataCols.id_col,
            how="left",
            suffixes=("", "_source"),
        )

        compiled = (
            merged.groupby("_row")[recompile_cols]
            .agg(join_unique)
            .reindex(mc_encode_i.metadata.index)
            .fillna("")
        )

        mc_encode_i.metadata.loc[:, recompile_cols] = compiled

    # Recompile metadata
    for mc_encode_i in (mc_encode_filtered, mc_encode_removed):
        recompile_metadata(mc_encode_i, encode_metadata_df, metadata_recompile_cols)
        recompile_metadata(mc_encode_i, encode_metadata_df, metadata_recompile_cols)


    # ### Annotate hits
    # Hits bed indexed
    category_type_simple_col = 'category'
    label_conf_simple_col = 'confidence'
    name_tfmodisco_col = 'name_tfmodisco'
    hits_bed_cluster_col = 'hits_bed_cluster'  # Matched to: encodecluster_name_col

    NARROWPEAKS_CUSTOM_COLS = [
        'chrom',
        'start',
        'end',
        'name',
        'score',
        'strand',
        'thickStart',
        'thickEnd',
        'itemRgb',
        category_type_simple_col,
        label_conf_simple_col,
        MetadataCols.head_col,
        name_tfmodisco_col,
        hits_bed_cluster_col,
    ]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Index hits, per experiment: its FiNeMo hits (both heads), each with the ENCODE cluster of its motif, as
    # indexed BEDs (bgzip + tabix) in <export>/hits_indexed: {ID}.{head}.hits.indexed.bed.gz and, both heads
    # collapsed, {ID}.hits.indexed.combined.bed.gz (from sherlock/runs/encode/finemo/apply_mc/update_hits_bed.py).
    # FiNeMo paths: the motifs' finemo_path (from the atlas pipelines' --{head}-finemo-paths), read above.
    logging.info("Indexing FiNeMo hits with their ENCODE clusters...")
    hits_indexed_dir = os.path.join(export_dir, "hits_indexed")
    hits_index_df = mc_motif_index.copy()

    # Hits BED name and ENCODE cluster, per motif
    flagged = hits_index_df[FilterArgs.filter_flag_col].fillna(True).astype(bool)
    simple_name = hits_index_df[MetadataCols.simple_name_col]
    encode_name = hits_index_df[VisualizeArgs.encodecluster_name_col]
    confidence_simple = pd.Series(np.select(
        [hits_index_df[LabelCols.label_conf_col] == confidence for confidence in HITS_CONFIDENCES],
        HITS_CONFIDENCES_SIMPLE,
        default=HITS_CONFIDENCES_SIMPLE[-1],
    ), index=hits_index_df.index)
    hits_bed_name = simple_name.fillna(HITS_NAME_UNKNOWN)
    hits_bed_cluster = encode_name.fillna(simple_name).fillna(HITS_NAME_NONE)

    # TF atlas: Direct motifs named by their TF; others by their ENCODE simple name
    mask_tf = hits_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id
    mask_direct = hits_index_df[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])
    mask = mask_tf & ~flagged & mask_direct
    hits_bed_name[mask] = (hits_index_df[MetadataCols.tf_col] + "_" + hits_index_df[MetadataCols.posneg_col] + "_" + confidence_simple)[mask]
    mask = mask_tf & (flagged | ~mask_direct) & simple_name.astype(bool)
    hits_bed_name[mask] = simple_name[mask]
    hits_bed_cluster[mask_tf & ~flagged] = encode_name[mask_tf & ~flagged]

    # Chromatin, ProCap atlases: ENCODE simple name
    mask = hits_index_df[VisualizeArgs.atlas_col].isin([EncodeCombineArgs.chromatlas_id, EncodeCombineArgs.procapatlas_id]) & simple_name.astype(bool)
    hits_bed_name[mask] = simple_name[mask]
    hits_bed_cluster[mask] = encode_name[mask]

    # Flagged: Simple name = filter flag type (CpG, tandem-repeats, high-GC/AT, low-complexity, unvalidated)
    hits_bed_name[flagged] = simple_name[flagged]
    hits_bed_cluster[flagged] = simple_name[flagged]
    hits_index_df["hits_bed_name"] = hits_bed_name.fillna(simple_name).fillna(HITS_NAME_UNKNOWN)
    hits_index_df["hits_bed_cluster"] = hits_bed_cluster.fillna(encode_name).fillna(simple_name).fillna(HITS_NAME_NONE)
    hits_index_df["flagged"] = flagged
    hits_index_df["pattern_key"] = _hits_pattern_key(hits_index_df[MetadataCols.posneg_col], hits_index_df[MetadataCols.name_col])

    # Index: per experiment and head with a FiNeMo path
    hits_models_df = (
        hits_finemo_paths_df.dropna(subset=[MetadataCols.finemo_path_col])
        .drop_duplicates(subset=[MetadataCols.id_col, MetadataCols.head_col])
        [[MetadataCols.id_col, MetadataCols.head_col, VisualizeArgs.atlas_col, MetadataCols.finemo_path_col]]
    )
    if hits_models_df.empty:
        logging.warning("No FiNeMo paths (finemo_path) in the atlas compendia: hits are not indexed or annotated.")
    # Peaks used to call each experiment's hits, for the annotation below (the counts head's, else the profile's)
    hits_peaks_paths = (
        hits_finemo_paths_df.dropna(subset=[MetadataCols.peaks_path_col])
        .sort_values(MetadataCols.head_col)
        .drop_duplicates(subset=[MetadataCols.id_col])
        .set_index(MetadataCols.id_col)[MetadataCols.peaks_path_col]
        .to_dict()
    )
    n_workers_index = max(1, min(args.max_cpus, len(os.sched_getaffinity(0))))
    hits_indexed_paths = {}
    with ProcessPoolExecutor(max_workers=n_workers_index, mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {}
        for _, row in hits_models_df.iterrows():
            model_hits_index = hits_index_df[
                (hits_index_df[MetadataCols.id_col] == row[MetadataCols.id_col])
                & (hits_index_df[MetadataCols.head_col] == row[MetadataCols.head_col])
            ].dropna(subset=["pattern_key"]).drop_duplicates(subset=["pattern_key"])
            motif_lookup = dict(zip(model_hits_index["pattern_key"], zip(
                model_hits_index[MetadataCols.name_col], model_hits_index["hits_bed_name"], model_hits_index["hits_bed_cluster"],
                model_hits_index[LabelCols.label_conf_col], model_hits_index[DirectTypeArgs.category_type_col], model_hits_index["flagged"],
            )))
            hits_bed_out = os.path.join(hits_indexed_dir, f"{row[MetadataCols.id_col]}.{row[MetadataCols.head_col]}.hits.indexed.bed.gz")
            future = executor.submit(_index_hits_bed, row[MetadataCols.finemo_path_col], hits_bed_out, motif_lookup, row[MetadataCols.head_col])
            futures[future] = (row[MetadataCols.id_col], row[MetadataCols.head_col], hits_bed_out)
        for future in as_completed(futures):
            encid, head, hits_bed_out = futures[future]
            try:
                future.result()
                hits_indexed_paths.setdefault(encid, {})[head] = hits_bed_out
            except Exception as e:
                logging.warning(f"Indexing hits failed for {encid} ({head}): {e}")

    # Combine: both heads of an experiment, overlapping hits of an ENCODE cluster collapsed
    hits_combined_paths = {}
    with ProcessPoolExecutor(max_workers=n_workers_index, mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {}
        for encid, head_paths in hits_indexed_paths.items():
            if len(head_paths) == 1:
                hits_combined_paths[encid] = next(iter(head_paths.values()))
                continue
            hits_bed_out = os.path.join(hits_indexed_dir, f"{encid}.hits.indexed.combined.bed.gz")
            futures[executor.submit(_combine_hits_beds, [head_paths[head] for head in sorted(head_paths)], hits_bed_out)] = (encid, hits_bed_out)
        for future in as_completed(futures):
            encid, hits_bed_out = futures[future]
            try:
                future.result()
                hits_combined_paths[encid] = hits_bed_out
            except Exception as e:
                logging.warning(f"Combining indexed hits failed for {encid}: {e}")
    logging.info(f"Indexed hits: {len(hits_combined_paths)} experiments")

    # Hits metadata: per experiment, its atlas and indexed hits (for the annotation below)
    metadata_hits_df = (
        hits_models_df.drop_duplicates(subset=[MetadataCols.id_col])
        .set_index(MetadataCols.id_col)[[VisualizeArgs.atlas_col]]
    )
    metadata_hits_df["hits_indexed"] = metadata_hits_df.index.map(hits_combined_paths)
    metadata_hits_df = metadata_hits_df.dropna(subset=["hits_indexed"])
    os.makedirs(os.path.dirname(metadata_hits_path), exist_ok=True)
    metadata_hits_df.to_csv(metadata_hits_path, sep="\t")


    # Load: Gene annotation (GENCODE v38, the same GENCODE track underlying
    # TxDb.Hsapiens.UCSC.hg38.knownGene used by ArchR::createGeneAnnotation)
    gtf_cols = ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    gencode_gtf_df = pd.read_csv(
        args.gencode,
        sep="\t",
        comment="#",
        header=None,
        names=gtf_cols,
        usecols=['chrom', 'feature', 'start', 'end', 'strand', 'attribute'],
    )

    def gtf_extract_attr(attribute, key):
        return attribute.str.extract(fr'{key} "([^"]+)"')[0]

    # Genes: one row per gene body (spans all its transcripts), like GenomicFeatures::genes(TxDb)
    genes_df = gencode_gtf_df[gencode_gtf_df['feature'] == 'gene'].copy()
    genes_df['gene_id'] = gtf_extract_attr(genes_df['attribute'], 'gene_id')
    genes_df['symbol'] = gtf_extract_attr(genes_df['attribute'], 'gene_name').fillna(genes_df['gene_id'])
    genes_df = genes_df[['chrom', 'start', 'end', 'strand', 'symbol']].reset_index(drop=True)

    # Exons: every exon of every transcript, like unlist(exonsBy(TxDb, by="tx"))
    exons_df = gencode_gtf_df[gencode_gtf_df['feature'] == 'exon'][['chrom', 'start', 'end', 'strand']].reset_index(drop=True)

    # TSS: one row per unique transcript start position (strand-aware), like
    # unique(resize(transcripts(TxDb), width=1, fix="start"))
    transcripts_df = gencode_gtf_df[gencode_gtf_df['feature'] == 'transcript'].copy()
    transcripts_df['transcript_name'] = gtf_extract_attr(transcripts_df['attribute'], 'transcript_name')
    transcripts_df['tss_pos'] = np.where(transcripts_df['strand'] == '+', transcripts_df['start'], transcripts_df['end'])
    tss_df = (
        transcripts_df
        .drop_duplicates(subset=['chrom', 'tss_pos', 'strand'])
        .rename(columns={'tss_pos': 'pos', 'transcript_name': 'label'})
        [['chrom', 'pos', 'strand', 'label']]
        .reset_index(drop=True)
    )

    del gencode_gtf_df, transcripts_df
    logging.info(f"{len(genes_df)} genes, {len(exons_df)} exons, {len(tss_df)} unique TSS")

    # Functions: Genome annotation
    # Reimplements ArchR:::.fastAnnoPeaks (genomic annotation: promoter/exonic/
    # intronic/distal, distance to nearest gene/TSS, GC/N content) in pure Python.

    class IntervalIndex:
        """Per-chromosome index answering: does point p overlap ANY [start, end] interval?"""

        def __init__(self, chrom, start, end):
            self.by_chrom = {}
            df = pd.DataFrame({'chrom': np.asarray(chrom), 'start': np.asarray(start), 'end': np.asarray(end)})
            for chrom_i, g in df.groupby('chrom'):
                order = np.argsort(g['start'].to_numpy(), kind='mergesort')
                starts_sorted = g['start'].to_numpy()[order]
                cummax_end = np.maximum.accumulate(g['end'].to_numpy()[order])
                self.by_chrom[chrom_i] = (starts_sorted, cummax_end)

        def overlaps_any(self, chrom_arr, pos_arr):
            result = np.zeros(len(pos_arr), dtype=bool)
            df = pd.DataFrame({'chrom': chrom_arr, 'pos': pos_arr, '_i': np.arange(len(pos_arr))})
            for chrom_i, g in df.groupby('chrom'):
                if chrom_i not in self.by_chrom:
                    continue
                starts_sorted, cummax_end = self.by_chrom[chrom_i]
                p = g['pos'].to_numpy()
                j = np.searchsorted(starts_sorted, p, side='right') - 1
                valid = j >= 0
                hit = np.zeros(len(p), dtype=bool)
                hit[valid] = cummax_end[j[valid]] >= p[valid]
                result[g['_i'].to_numpy()] = hit
            return result


    class NearestPointIndex:
        """Per-chromosome nearest-1bp-point index: distance + label of the closest point to each query."""

        def __init__(self, chrom, pos, label):
            self.by_chrom = {}
            df = pd.DataFrame({'chrom': np.asarray(chrom), 'pos': np.asarray(pos), 'label': np.asarray(label)})
            for chrom_i, g in df.groupby('chrom'):
                order = np.argsort(g['pos'].to_numpy(), kind='mergesort')
                self.by_chrom[chrom_i] = (g['pos'].to_numpy()[order], g['label'].to_numpy()[order])

        def nearest(self, chrom_arr, pos_arr):
            # IRanges-style distance between two 1bp points: max(0, |a - b| - 1)
            dist = np.full(len(pos_arr), np.nan)
            label = np.full(len(pos_arr), None, dtype=object)
            df = pd.DataFrame({'chrom': chrom_arr, 'pos': pos_arr, '_i': np.arange(len(pos_arr))})
            for chrom_i, g in df.groupby('chrom'):
                if chrom_i not in self.by_chrom:
                    continue
                pos_sorted, labels_sorted = self.by_chrom[chrom_i]
                p = g['pos'].to_numpy()
                j = np.searchsorted(pos_sorted, p)
                j_left = np.clip(j - 1, 0, len(pos_sorted) - 1)
                j_right = np.clip(j, 0, len(pos_sorted) - 1)
                d_left = np.maximum(np.abs(p - pos_sorted[j_left]) - 1, 0)
                d_right = np.maximum(np.abs(p - pos_sorted[j_right]) - 1, 0)
                use_left = d_left <= d_right
                ii = g['_i'].to_numpy()
                dist[ii] = np.where(use_left, d_left, d_right)
                label[ii] = labels_sorted[np.where(use_left, j_left, j_right)]
            return dist, label


    def classify_peak_type(chrom_arr, pos_arr, promoter_index, gene_index, exon_index):
        op = promoter_index.overlaps_any(chrom_arr, pos_arr)
        og = gene_index.overlaps_any(chrom_arr, pos_arr)
        oe = exon_index.overlaps_any(chrom_arr, pos_arr)
        peak_type = np.full(len(pos_arr), 'Distal', dtype=object)
        peak_type[og & oe] = 'Exonic'
        peak_type[og & ~oe] = 'Intronic'
        peak_type[op] = 'Promoter'  # promoter overrides exonic/intronic, matching ArchR
        return peak_type


    def compute_gc_n_content(chrom_arr, start_arr, end_arr, fasta):
        # 1-based inclusive [start, end] -> pysam 0-based half-open fetch
        gc = np.empty(len(start_arr))
        n_frac = np.empty(len(start_arr))
        for i in range(len(start_arr)):
            seq = fasta.fetch(chrom_arr[i], int(start_arr[i]) - 1, int(end_arr[i])).upper()
            length = len(seq)
            gc[i] = round((seq.count('G') + seq.count('C')) / length, 4)
            n_frac[i] = round(seq.count('N') / length, 4)
        return gc, n_frac

    # Build: Genome annotation indices
    gene_tss_pos = np.where(genes_df['strand'] == '+', genes_df['start'], genes_df['end'])

    gene_index = IntervalIndex(genes_df['chrom'], genes_df['start'], genes_df['end'])
    exon_index = IntervalIndex(exons_df['chrom'], exons_df['start'], exons_df['end'])
    promoter_region = 2000  # ArchR promoterRegion = c(2000, 2000), symmetric around gene TSS
    promoter_index = IntervalIndex(genes_df['chrom'], gene_tss_pos - promoter_region, gene_tss_pos + promoter_region)
    gene_tss_index = NearestPointIndex(genes_df['chrom'], gene_tss_pos, genes_df['symbol'])
    tss_index = NearestPointIndex(tss_df['chrom'], tss_df['pos'], tss_df['label'])

    hg38_fasta = pysam.FastaFile(args.hg38)

    # Functions: Annotate hits
    def annotate_hits(hits_df, peaks_bed, fasta):
        # if these exist already, drop so we can overwrite
        cols_drop = ['distToGeneStart', 'nearestGene', 'peakType', 'distToTSS', 'nearestTSS', 'GC', 'N', 'distToPeakSummit']
        hits = hits_df.drop(columns=[c for c in cols_drop if c in hits_df.columns]).copy()

        # convert 0-based (BED) to 1-based (GRanges-style) start
        hits['start'] = hits['start'] + 1
        chrom_arr = hits['chrom'].to_numpy()
        start_arr = hits['start'].to_numpy()
        end_arr = hits['end'].to_numpy()
        hit_center = start_arr + (end_arr - start_arr) // 2

        # genomic annotation: distance/nearest gene, peak type, distance/nearest TSS
        dist_gene, nearest_gene = gene_tss_index.nearest(chrom_arr, hit_center)
        hits['distToGeneStart'] = dist_gene
        hits['nearestGene'] = nearest_gene

        hits['peakType'] = classify_peak_type(chrom_arr, hit_center, promoter_index, gene_index, exon_index)

        dist_tss, nearest_tss = tss_index.nearest(chrom_arr, hit_center)
        hits['distToTSS'] = dist_tss
        hits['nearestTSS'] = nearest_tss

        # nucleotide content over the full hit interval
        hits['GC'], hits['N'] = compute_gc_n_content(chrom_arr, start_arr, end_arr, fasta)

        # distance to peak summit (peaks are fixed-width windows around a summit,
        # so the center of the peak is the summit)
        peaks_df = pd.read_csv(peaks_bed, sep="\t", header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
        peaks_df['start'] = peaks_df['start'] + 1
        peak_center = peaks_df['start'].to_numpy() + (peaks_df['end'].to_numpy() - peaks_df['start'].to_numpy()) // 2
        peak_summit_index = NearestPointIndex(peaks_df['chrom'], peak_center, peak_center)
        dist_summit, _ = peak_summit_index.nearest(chrom_arr, hit_center)
        hits['distToPeakSummit'] = dist_summit

        # back to 0-based (BED) start
        hits['start'] = hits['start'] - 1
        return hits

    def annotate_hits_row(hits_tsv, peaks_bed, hits_tsv_annotated, fasta):
        hits_df = pd.read_csv(
            hits_tsv,
            sep="\t",
            header=None,
            names=NARROWPEAKS_CUSTOM_COLS,
            keep_default_na=False,
        )
        hits_anno_df = annotate_hits(hits_df, peaks_bed, fasta)
        hits_anno_df.to_csv(hits_tsv_annotated, sep="\t", index=False)
        return len(hits_anno_df)

    # The (forked) workers inherit this closure; the pool runs the module-level _hits_worker_run
    _HITS_WORKER["annotate_row"] = annotate_hits_row


    # Annotate hits, per motif -- parallelized across experiments (rows); each row
    # is independent (its own hits/peaks files), so this parallelizes cleanly.
    os.makedirs(hits_annotated_dir, exist_ok=True)
    n_workers = min(8, len(os.sched_getaffinity(0)))
    mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_hits_worker_init, initargs=(args.hg38,), mp_context=mp_context) as executor:
        futures = {}
        for i, row in metadata_hits_df.iterrows():
            hits_tsv = row["hits_indexed"]
            if pd.isna(hits_tsv) or not os.path.exists(hits_tsv):
                logging.warning("Skipping missing hits file for %s: %s", i, hits_tsv)
                continue
            # Indexed hits only (NARROWPEAKS_CUSTOM_COLS: each hit with its ENCODE cluster); a raw FiNeMo
            # hits file has no ENCODE cluster to summarize hits by
            with (gzip.open(hits_tsv, "rt") if hits_tsv.endswith(".gz") else open(hits_tsv)) as hits_handle:
                n_hits_cols = len(hits_handle.readline().rstrip("\n").split("\t"))
            if n_hits_cols != len(NARROWPEAKS_CUSTOM_COLS):
                logging.warning(
                    "Skipping hit annotation for %s: %s has %d columns, not the %d of indexed hits",
                    i, hits_tsv, n_hits_cols, len(NARROWPEAKS_CUSTOM_COLS),
                )
                continue

            # Peaks used to call this experiment's hits (peaks_path: the atlas pipelines' --{head}-peaks-paths)
            peaks_bed = hits_peaks_paths.get(i)
            if peaks_bed is None:
                logging.warning("Skipping hit annotation for %s: no peaks (peaks_path)", i)
                continue
            if pd.isna(peaks_bed) or not os.path.exists(peaks_bed):
                logging.warning(
                    "Skipping hit annotation for %s: missing peaks %s",
                    i, peaks_bed,
                )
                continue
            hits_tsv_annotated = os.path.join(hits_annotated_dir, f"{i}.hits.annotated.tsv.gz")

            future = executor.submit(_hits_worker_run, hits_tsv, peaks_bed, hits_tsv_annotated)
            futures[future] = i

        for n_done, future in enumerate(as_completed(futures), start=1):
            i = futures[future]
            try:
                n_hits = future.result()
                if args.verbose:
                    logging.info(f"[{n_done}/{len(futures)}] {i}: {n_hits} hits")
            except Exception as e:
                if args.verbose:
                    logging.info(f"[{n_done}/{len(futures)}] {i}: FAILED - {e}")

    # Functions: Compile hits summary per motif
    # Compile hits summary per motif -- pool per-hit annotations (peakType,
    # distToTSS, distToPeakSummit) across every experiment that contributes to the
    # same final combined motif cluster (hits_bed_cluster / ENCODE_name),
    # parallelized across annotated files
    from collections import defaultdict

    hits_tsv_annotated_paths = [
        os.path.join(hits_annotated_dir, f"{i}.hits.annotated.tsv.gz")
        for i in metadata_hits_df.index
    ]
    hits_tsv_annotated_paths = [p for p in hits_tsv_annotated_paths if os.path.exists(p)]

    peak_type_counts_total = None
    dist_tss_by_cluster = defaultdict(list)
    dist_summit_by_cluster = defaultdict(list)

    n_workers = min(8, len(os.sched_getaffinity(0)))
    mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(_summarize_hits_file, p): p for p in hits_tsv_annotated_paths}
        for n_done, future in enumerate(as_completed(futures), start=1):
            p = futures[future]
            try:
                peak_type_counts, dist_values = future.result()
            except Exception as e:
                if args.verbose:
                    logging.info(f"[{n_done}/{len(futures)}] {p}: FAILED - {e}")
                continue

            peak_type_counts_total = (
                peak_type_counts if peak_type_counts_total is None
                else peak_type_counts_total.add(peak_type_counts, fill_value=0)
            )
            for cluster, (tss_vals, summit_vals) in dist_values.items():
                dist_tss_by_cluster[cluster].append(tss_vals)
                dist_summit_by_cluster[cluster].append(summit_vals)

            if n_done % 500 == 0 or n_done == len(futures):
                if args.verbose:
                    logging.info(f"[{n_done}/{len(futures)}] files compiled")

    # Hits summary, per-cluster hit columns and figures: only if any indexed hits were annotated
    hits_summarized = peak_type_counts_total is not None
    if not hits_summarized:
        logging.warning(
            "No indexed hits were annotated: the hits summary and the region-type / distance-to-TSS "
            "figures are skipped (the FiNeMo hits need their ENCODE clusters: NARROWPEAKS_CUSTOM_COLS)."
        )
    else:
        peak_type_counts_total = peak_type_counts_total.fillna(0).astype(int)
        peak_type_counts_total.columns = [f"n_hits_{c}" for c in peak_type_counts_total.columns]
        peak_type_counts_total["n_hits_total"] = peak_type_counts_total.sum(axis=1)

        # nanmean/nanmedian: a small fraction of hits land on unplaced/alt scaffolds that
        # GENCODE's primary-chromosome annotation doesn't cover, giving NaN distToTSS for
        # just those hits -- skip them rather than letting one NaN blank out a whole motif
        dist_tss_concat = {c: np.concatenate(v) for c, v in dist_tss_by_cluster.items()}
        dist_summit_concat = {c: np.concatenate(v) for c, v in dist_summit_by_cluster.items()}

        dist_summary = pd.DataFrame({
            'mean_distToTSS': {c: np.nanmean(v) for c, v in dist_tss_concat.items()},
            'median_distToTSS': {c: np.nanmedian(v) for c, v in dist_tss_concat.items()},
            'mean_distToPeakSummit': {c: np.nanmean(v) for c, v in dist_summit_concat.items()},
            'median_distToPeakSummit': {c: np.nanmedian(v) for c, v in dist_summit_concat.items()},
        })
        dist_summary[HitsCols.distToTSS_values_col] = pd.Series(dist_tss_concat, dtype=object)  # full distribution, for box plots

        hits_summary_df = (
            peak_type_counts_total.join(dist_summary, how='outer')
            .reset_index()
            .rename(columns={'index': 'hits_bed_cluster'})
        )
        logging.info(f"{len(hits_summary_df)} combined motif clusters summarized")

        # Recompile hits summary
        def recompile_hits_summary(mc_encode_i, hits_summary_df):
            summary_cols = [c for c in hits_summary_df.columns if c != 'hits_bed_cluster']
            hits_summary_lookup = hits_summary_df.set_index('hits_bed_cluster')

            # if these exist already, drop so we can overwrite
            mc_encode_i.metadata = mc_encode_i.metadata.drop(
                columns=[c for c in summary_cols if c in mc_encode_i.metadata.columns]
            )

            compiled = hits_summary_lookup.reindex(mc_encode_i.metadata[MetadataCols.name_col]).reset_index(drop=True)
            compiled.index = mc_encode_i.metadata.index

            # no hits found for this cluster -> 0 counts (not NaN); mean/median stay NaN
            count_cols = [c for c in summary_cols if c.startswith('n_hits_')]
            compiled[count_cols] = compiled[count_cols].fillna(0).astype(int)

            mc_encode_i.metadata[summary_cols] = compiled

        # Recompile hits summary
        for mc_encode_i in (mc_encode_filtered, mc_encode_removed):
            recompile_hits_summary(mc_encode_i, hits_summary_df)

            # Calculate fractions
            mc_encode_i[HitsCols.hits_exonic_frac_col] = mc_encode_i[HitsCols.hits_exonic_col] / mc_encode_i[HitsCols.hits_total_col]
            mc_encode_i[HitsCols.hits_intronic_frac_col] = mc_encode_i[HitsCols.hits_intronic_col] / mc_encode_i[HitsCols.hits_total_col]
            mc_encode_i[HitsCols.hits_promoter_frac_col] = mc_encode_i[HitsCols.hits_promoter_col] / mc_encode_i[HitsCols.hits_total_col]
            mc_encode_i[HitsCols.hits_distal_frac_col] = mc_encode_i[HitsCols.hits_distal_col] / mc_encode_i[HitsCols.hits_total_col]

        # Functions: Plot per-motif figures
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import MotifCompendium.utils.plotting as utils_plotting

        cmap_peaktype = {
            'Exonic': '#1b9e77',
            'Promoter': '#7570b3',
            'Intronic': '#d95f02',
            'Distal': '#999999',
        }

        def plot_region_type_barplot(fracs):
            # fracs: dict {peakType: fraction}; renders a single horizontal stacked bar
            fig, ax = plt.subplots(figsize=(4, 0.6))
            left = 0
            for peak_type, color in cmap_peaktype.items():
                frac = fracs.get(peak_type, 0)
                if pd.isna(frac):
                    frac = 0
                ax.barh(0, frac, left=left, color=color, edgecolor='white')
                left += frac
            ax.set_xlim(0, 1)
            ax.axis('off')
            return fig

        def plot_disttotss_boxplot(values, xlim=None):
            fig, ax = plt.subplots(figsize=(2.2, 0.8))

            values = np.asarray(values, dtype=float) / 1_000  # bp → kb
            values = values[~np.isnan(values)]

            if len(values) == 0:
                ax.axis("off")
                return fig

            ax.boxplot(values, vert=False, showfliers=False, widths=0.6)
            ax.set_yticks([])
            ax.set_xlabel(None)
            ax.tick_params(labelsize=6)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if xlim is not None:
                ax.set_xlim(np.asarray(xlim) / 1_000)  # supply limits in bp

            fig.tight_layout(pad=0.1)
            return fig

        # Set max xlim for distToTSS boxplots: round up to 1 significant figure the 90th percentile of all distToTSS values across all motifs
        def round_up_to_1_sigfig(x):
            if pd.isna(x) or x <= 0:
                return x
            exp = np.floor(np.log10(abs(x)))
            factor = 10 ** (-exp)
            return np.ceil(x * factor) / factor

        disttotss_max = round_up_to_1_sigfig(
            np.nanpercentile(
                np.concatenate([
                    np.asarray(values, dtype=float)
                    for values in mc_encode_filtered[HitsCols.distToTSS_values_col]
                    if isinstance(values, np.ndarray) and values.size > 0
                ]),
                90,
            )
        )

        # Plot per-motif figures
        for mc_encode_i in (mc_encode_filtered, mc_encode_removed):
            region_type_images = []
            disttotss_images = []
            for _, row in mc_encode_i.metadata.iterrows():
                fracs = {
                    'Exonic': row.get(HitsCols.hits_exonic_frac_col),
                    'Promoter': row.get(HitsCols.hits_promoter_frac_col),
                    'Intronic': row.get(HitsCols.hits_intronic_frac_col),
                    'Distal': row.get(HitsCols.hits_distal_frac_col),
                }
                if all(pd.isna(v) for v in fracs.values()):
                    region_type_images.append(None)
                else:
                    fig = plot_region_type_barplot(fracs)
                    region_type_images.append(utils_plotting.encode_figure_as_utf8(fig))

                values = row.get(HitsCols.distToTSS_values_col)
                if values is None or (not hasattr(values, '__len__')) or len(values) == 0:
                    disttotss_images.append(None)
                else:
                    fig = plot_disttotss_boxplot(values, xlim=(0, disttotss_max))
                    disttotss_images.append(utils_plotting.encode_figure_as_utf8(fig))

            mc_encode_i.set_images(HitsCols.region_type_barplot_col, region_type_images)
            mc_encode_i.set_images(HitsCols.distToTSS_boxplot_col, disttotss_images)

        # Drop columns: distToTSS_values (full distribution, not needed for HTML)
        mc_encode_filtered.metadata.drop(columns=[HitsCols.distToTSS_values_col], inplace=True)
        mc_encode_removed.metadata.drop(columns=[HitsCols.distToTSS_values_col], inplace=True)

    # Create HTML
    html_columns_encodecluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_encode_filtered.columns() + mc_encode_filtered.images())] + [
        html_col for html_col in [
            MetadataCols.posneg_col,
            HitsCols.hits_total_col,
            HitsCols.hits_exonic_frac_col,
            HitsCols.hits_promoter_frac_col,
            HitsCols.hits_intronic_frac_col,
            HitsCols.hits_distal_frac_col,
            HitsCols.region_type_barplot_col,
            HitsCols.distToTSS_boxplot_col,
        ]  # Add new figure columns here
        if html_col in (mc_encode_filtered.columns() + mc_encode_filtered.images())
    ]
    html_column_desc_dict_encodecluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_encodecluster}

    os.makedirs(os.path.dirname(html_encode_filtered_path), exist_ok=True)
    mc_encode_filtered.summary_table_html(
        html_out=html_encode_filtered_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    os.makedirs(os.path.dirname(html_encode_removed_path), exist_ok=True)
    mc_encode_removed.summary_table_html(
        html_out=html_encode_removed_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    # Metadata TSV
    os.makedirs(os.path.dirname(metadata_encode_path), exist_ok=True)
    mc_encode_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_encode_filtered_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(metadata_encode_path), exist_ok=True)
    mc_encode_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_encode_removed_path, sep="\t", index=False)

    # Save MotifCompendium: ENCODE, with the hits annotation (over the ones saved before it)
    save_replace(mc_encode_filtered, mc_encode_filtered_path)
    save_replace(mc_encode_removed, mc_encode_removed_path)

    # Args JSON
    args_record = {
        # --- Full provenance (added 2026-09-10) ---------------------------------
        # Every parser argument, verbatim
        "parser_args": vars(args),
        # Upstream provenance: every per-head / combined run this ENCODE combine consumes
        "previous_args": collect_previous_args([
            ("tf_counts", args.tf_counts_dir),
            ("tf_profile", args.tf_profile_dir),
            ("tf_combined", args.tf_combined_dir),
            ("chrom_counts", args.chrom_counts_dir),
            ("chrom_profile", args.chrom_profile_dir),
            ("chrom_combined", args.chrom_combined_dir),
            ("procap_counts", args.procap_counts_dir),
            ("procap_profile", args.procap_profile_dir),
            ("procap_combined", args.procap_combined_dir),
            ("previous_run_this_output_dir", args.output_dir),
        ]),
        # Every dataclass in configs.py as used by this run
        "configs": collect_configs_snapshot(),
        # --- Curated summary (pre-existing keys) --------------------------------
        # Compute options
        "max_chunk": args.max_chunk,
        "max_cpus": args.max_cpus,
        "use_gpu": args.use_gpu,
        "safe": args.safe,
        "ic_scale": args.ic,
        "fast_plot": args.fast_plot,

        # Parameters: Column names
        "name_col": MetadataCols.name_col,
        "simple_name_col": MetadataCols.simple_name_col,
        "prename_col": MetadataCols.prename_col,
        "name_new_col": MetadataCols.name_new_col,
        "label_col": LabelCols.label_col,
        "direct_label_col": LabelCols.direct_label_col,
        "name_constituents_col": MetadataCols.name_constituents_col,
        "name_unknown": MetadataCols.name_unknown,
        "counter_col": MetadataCols.counter_col,
        "qc_col": MetadataCols.qc_col,
        "unique_id_col": MetadataCols.unique_id_col,
        "unique_id_initial": unique_id_initial,

        # Experiment
        "id_col": MetadataCols.id_col,
        "tf_col": MetadataCols.tf_col,
        "tf_encode_col": MetadataCols.tf_encode_col,
        "model_col": MetadataCols.model_col,
        "head_col": MetadataCols.head_col,
        "accession_col": MetadataCols.accession_col,
        "biosample_col": MetadataCols.biosample_col,
        "assay_col": MetadataCols.assay_col,
        "znf_col": MetadataCols.znf_col,

        # Composites
        "composite_col": MetadataCols.composite_col,
        "num_subunit_col": MetadataCols.num_subunit_col,
        "num_subunit_direct_col": MetadataCols.num_subunit_direct_col,

        # TF-MoDISco
        "num_seqlet_col": MetadataCols.num_seqlet_col,
        "num_constituents_col": MetadataCols.num_constituents_col,
        "posneg_col": MetadataCols.posneg_col,
        "avgcontrib_score_col": MetadataCols.avgcontrib_score_col,
        "total_contrib_score_col": MetadataCols.total_contrib_score_col,
        "group_total_contrib_score_col": MetadataCols.group_total_contrib_score_col,
        "total_contrib_score_ratio_col": MetadataCols.total_contrib_score_ratio_col,
        "avgdist_summit_col": MetadataCols.avgdist_summit_col,

        # Parameters: TF annotation
        "family_col": TFAnnotationCols.family_col,
        "tfclass_family_name_col": TFAnnotationCols.tfclass_family_name_col,
        "tfclass_family_id_col": TFAnnotationCols.tfclass_family_id_col,
        "humantf_dbd_col": TFAnnotationCols.humantf_dbd_col,

        # Parameters: Cell type annotation
        "celltype_col": CellTypeCols.celltype_col,
        "organ_col": CellTypeCols.organ_col,
        "slim_cell_col": CellTypeCols.slim_cell_col,
        "slim_organ_col": CellTypeCols.slim_organ_col,
        "slim_dev_col": CellTypeCols.slim_dev_col,
        "slim_system_col": CellTypeCols.slim_system_col,

        # Annotation
        # "label_initial": label_initial,
        "label_col": LabelCols.label_col,
        "label_readable_col": LabelCols.label_readable_col,
        "label_constit_1_col": LabelCols.label_constit_1_col,
        "label_constit_all_col": LabelCols.label_constit_all_col,
        "direct_label_constit_1_col": LabelCols.direct_label_constit_1_col,
        "direct_label_constit_all_col": LabelCols.direct_label_constit_all_col,
        "label_core_col": LabelCols.label_core_col,
        "label_core_col_set": LabelCols.label_core_col_set,
        "label_core_constit_1_col": LabelCols.label_core_constit_1_col,
        "label_core_constit_all_col": LabelCols.label_core_constit_all_col,
        "direct_label_core_col": LabelCols.direct_label_core_col,
        "direct_label_core_col_set": LabelCols.direct_label_core_col_set,
        "direct_label_core_constit_1_col": LabelCols.direct_label_core_constit_1_col,
        "direct_label_core_constit_all_col": LabelCols.direct_label_core_constit_all_col,
        "label_flank_col": LabelCols.label_flank_col,
        "label_flank_col_set": LabelCols.label_flank_col_set,
        "label_flank_constit_1_col": LabelCols.label_flank_constit_1_col,
        "label_flank_constit_all_col": LabelCols.label_flank_constit_all_col,
        "label_conf_col": LabelCols.label_conf_col,
        "label_confs": LabelCols.label_confs,
        "best_label_ref_col": LabelCols.best_label_ref_col,

        # Parameters: Reference / Direct
        "ref_tf_col": ReferenceMatchCols.ref_tf_col,
        "ref_family_col": ReferenceMatchCols.ref_family_col,
        "ref_col": ReferenceMatchCols.ref_col,
        "ref_name_col": ReferenceMatchCols.ref_name_col,
        "ref_name_constit_1_col": ReferenceMatchCols.ref_name_constit_1_col,
        "ref_name_constit_all_col": ReferenceMatchCols.ref_name_constit_all_col,
        "best_ref_constit_col": ReferenceMatchCols.best_ref_constit_col,
        "ref_readable_col": ReferenceMatchCols.ref_readable_col,
        # "ref_readable_name_col": ref_readable_name_col,
        # "ref_readable_name_constit_1_col": ref_readable_name_constit_1_col,
        # "ref_readable_name_constit_all_col": ref_readable_name_constit_all_col,
        "direct_ref_col": ReferenceMatchCols.direct_ref_col,
        "direct_ref_name_col": ReferenceMatchCols.direct_ref_name_col,
        "direct_ref_name_constit_1_col": ReferenceMatchCols.direct_ref_name_constit_1_col,
        "direct_ref_name_constit_all_col": ReferenceMatchCols.direct_ref_name_constit_all_col,
        "best_direct_ref_constit_col": ReferenceMatchCols.best_direct_ref_constit_col,
        "direct_ref_readable_col": ReferenceMatchCols.direct_ref_readable_col,
        # "direct_ref_readable_name_col": direct_ref_readable_name_col,
        # "direct_ref_readable_name_constit_1_col": direct_ref_readable_name_constit_1_col,
        # "direct_ref_readable_name_constit_all_col": direct_ref_readable_name_constit_all_col,
        "znf_ref_name_col": ReferenceMatchCols.znf_ref_name_col,
        "znf_ref_name_constit_1_col": ReferenceMatchCols.znf_ref_name_constit_1_col,
        "znf_ref_name_constit_all_col": ReferenceMatchCols.znf_ref_name_constit_all_col,
        "nonznf_ref_col": ReferenceMatchCols.nonznf_ref_col,
        "nonznf_ref_readable_col": ReferenceMatchCols.nonznf_ref_readable_col,
        "min_single_score": MotifMatchArgs.min_single_score,
        "min_composite_score": MotifMatchArgs.min_composite_score,
        "max_submotifs": MotifMatchArgs.max_submotifs,
        "min_composite_subunit": MotifMatchArgs.min_composite_subunit,
        "direct_type_col": DirectTypeArgs.direct_type_col,
        "tf_direct_col": DirectTypeArgs.tf_direct_col,
        "family_direct_col": DirectTypeArgs.family_direct_col,
        "direct_types": DirectTypeArgs.direct_types,

        # Parameters: Filtering
        "filter_flag_col": FilterArgs.filter_flag_col,
        "filter_flag_type_col": FilterArgs.filter_flag_type_col,
        "filter_trim_importance": FilterArgs.filter_trim_importance,
        "filter_trim_length": FilterArgs.filter_trim_length,
        "filter_ic_scale": FilterArgs.filter_ic_scale,
        "filter_flag_types": FilterArgs.filter_flag_types,
        "qc_min_score_removed": FilterArgs.qc_min_score_removed,
        "qc_min_score_filtered": FilterArgs.qc_min_score_filtered,

        # Parameters: Clustering
        "cluster_algorithm_main": ClusterArgs.cluster_algorithm_main,
        "cluster_algorithm_force": ClusterArgs.cluster_algorithm_force,
        "cluster_algorithm_kmeans": ClusterArgs.cluster_algorithm_kmeans,
        "cluster_sim_threshold": ClusterArgs.cluster_sim_threshold,
        "cluster_sim_threshold_force": ClusterArgs.cluster_sim_threshold_force,
        "cluster_force_density": ClusterArgs.cluster_force_density,
        "cluster_sim_assignment_threshold_kmeans": ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        "cluster_max_iter_main": ClusterArgs.cluster_max_iter_main,
        "cluster_max_iter_force": ClusterArgs.cluster_max_iter_force,
        "encodecluster_cluster_col": encodecluster_cluster_col,
        "average_weight_col": ClusterArgs.average_weight_col,
        "source_cluster_col": ClusterArgs.source_cluster_col,
        "n_iterations": ClusterArgs.n_iterations,
        "seeds": ClusterArgs.seeds,

        # Parameters: HTML
        "html_logo_trim": VisualizeArgs.html_logo_trim,
        "motif_string_col": VisualizeArgs.motif_string_col,
        "motif_string_importance": VisualizeArgs.motif_string_importance,
        "motif_string_specificity": VisualizeArgs.motif_string_specificity,

        # Cluster names
        "cluster_name_col": VisualizeArgs.cluster_name_col,
        "metacluster_name_col": VisualizeArgs.metacluster_name_col,
        "atlascluster_name_col": VisualizeArgs.atlascluster_name_col,
        "encodecluster_name_col": VisualizeArgs.encodecluster_name_col,
        "atlas_col": VisualizeArgs.atlas_col,
        "sort_col": VisualizeArgs.sort_col,
        "max_major_family_frac": VisualizeArgs.max_major_family_frac,
        "major_family_increment": VisualizeArgs.major_family_increment,
        "max_major_family_count": VisualizeArgs.max_major_family_count,

        # Aggregation
        "aggregations": ClusterArgs.aggregations,
        "weighted_aggregations": ClusterArgs.weighted_aggregations,

        # Motif index
        "motif_index_cols": VisualizeArgs.motif_index_cols,

        # Run log
        "log_path": log_path,
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


#### MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()
    set_default_configs()

    # Pipeline-level logging (stdout only); run_encode_combine() sets up the same logging again
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("ENCODE pipeline: run started.")

    # Inputs: the atlas pipelines' stage directories
    atlas_dirs = {"tf": args.tfatlas_dir, "chrom": args.chromatlas_dir, "procap": args.procapatlas_dir}
    stage_dir_args = {}
    for atlas_key, atlas_dir in atlas_dirs.items():
        stage_dir_args.update({
            f"{atlas_key}_dir": os.path.join(atlas_dir, "combined"),
            f"{atlas_key}_combined_dir": os.path.join(atlas_dir, "combined"),
            f"{atlas_key}_counts_dir": os.path.join(atlas_dir, "counts"),
            f"{atlas_key}_profile_dir": os.path.join(atlas_dir, "profile"),
        })
        atlascluster_path = os.path.join(atlas_dir, "combined", "mc", "motifcompendium_atlascluster_filtered.mc")
        if not os.path.exists(atlascluster_path):
            raise FileNotFoundError(f"Atlas pipeline output not found (has it completed?): {atlascluster_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.time:
        start_time = time.time()
    run_encode_combine(stage_args(args, **stage_dir_args))
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    logging.info(f'ENCODE pipeline: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
