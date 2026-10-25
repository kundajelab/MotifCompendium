#!/usr/bin/env python
# coding: utf-8
"""
BPNet (TF ChIP-seq): end-to-end MotifCompendium pipeline.

Runs the full hierarchy for a collection of TF ChIP-seq BPNet models from a single CLI,
in order:

  1. counts   - counts head  : motif -> cluster -> meta-cluster
  2. profile  - profile head : motif -> cluster -> meta-cluster
  3. combine  - counts + profile meta-clusters -> atlasclusters

Each head is labeled by TF type (direct / non-direct, ZNF composite handling via a
modified Smith-Waterman alignment against a B1H zinc-finger reference), clustered
per-TF and per-pattern, and built into meta-clusters. The combine stage merges the
two heads' meta-clusters 1:1 by pairwise similarity above the force threshold and
assigns labels and QC flags at the atlascluster level. Every level is exported as
MotifCompendium/HTML/MEME summaries.

Every stage directory is derived from a single --output-dir root, and the combine stage
reads the counts and profile stage directories directly:

  <root>/counts       <root>/profile       <root>/combined

Use --stages to run a subset (e.g. `--stages combine` to re-run only the combine stage on
existing counts/profile outputs under the same root).
"""
import os
import sys
import re
import argparse
import logging
import concurrent.futures
import json
import dataclasses
import time
from argparse import Namespace

import numpy as np
import pandas as pd

import h5py

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.composite as utils_composite

import importlib.util
# Configs: this script's own configs.py, imported under a name of its own (not "configs"), so that the
# atlas pipelines can be imported side by side (see pipeline/encode/pipeline_encode_full.py)
_configs_spec = importlib.util.spec_from_file_location(
    "configs_bpnet", os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.py")
)
configs = importlib.util.module_from_spec(_configs_spec)
sys.modules[_configs_spec.name] = configs
_configs_spec.loader.exec_module(configs)


#### STAGES ---------------------------------------------------------
# Stage key -> label
STAGES = {
    "counts":  "BPNet: counts head",
    "profile": "BPNet: profile head",
    "combine": "BPNet: combine counts + profile",
}
STAGE_ORDER = list(STAGES)


#### ARGUMENTS ------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline for a collection of TF ChIP-seq BPNet models, end to end (counts head -> profile head -> combine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline control
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER, help="Subset of stages to run, in pipeline order.")
    parser.add_argument("--update-constituents", choices=["now", "defer", "only"], default="now",
                        help="The combine stage's update of the levels below the atlasclusters (each head's metacluster, cluster "
                             "and motif compendia: their atlascluster; the atlascluster motif index). now: by the combine "
                             "stage; defer: not (the levels below stay as the per-head stages wrote them, e.g. until the ENCODE "
                             "pipeline's last level is made); only: that update alone, from the combine's saved outputs (no "
                             "stage is run).")
    parser.add_argument("--save-all", action="store_true",
                        help="Also save every level's removed motifs/clusters (motifcompendium_*_removed: .mc, HTML, MEME, "
                             "MoDISco H5); the _all and _filtered ones are always saved.")

    # Run type
    parser.add_argument("-vr", "--mcid-version", type=str, default=None, required=True, help="The version to use in the unique identifier (e.g., V1), for every stage.")

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Root output directory; the counts/profile/combined stage directories are derived from it.")

    # Input: per head (counts, profile)
    for head in ("counts", "profile"):
        parser.add_argument(f"--{head}-input-mc", type=str, default=None, help=f"{head} head: path to the input MotifCompendium object.")
        parser.add_argument(f"--{head}-modisco-h5s", nargs="+", type=str, default=None, help=f"{head} head: path(s) to the input TF-MoDISco (lite) H5 file(s).")
        parser.add_argument(f"--{head}-input-names", nargs="+", type=str, default=None, help=f"{head} head: nickname(s) of the input TF-MoDISco (lite) H5 file(s).")
        parser.add_argument(f"--{head}-metadata", type=str, default=None, help=f"{head} head: path to the metadata file, per input file or motif: CSV, TSV format.")
        parser.add_argument(f"--{head}-finemo-paths", type=str, default=None, help=f"{head} head: path to the FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
        parser.add_argument(f"--{head}-peaks-paths", type=str, default=None, help=f"{head} head: path to the peaks paths file (TSV, no header: input name, peaks BED), recorded per motif (peaks_path) for the ENCODE hits annotation.")
    parser.add_argument("-sh", "--input-subpatterns", action="store_true", help="Use subpatterns (instead of main patterns) when loading input TF-MoDISco (lite) H5 file(s).")

    # Reference databases
    parser.add_argument("--ref-qc", type=str, required=True, help="Path to the ENCODE QC reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--ref-invitro", type=str, required=True, help="Path to the ENCODE in vitro reference PFM MotifCompendium object (.mc), used to label direct motifs.")
    parser.add_argument("--ref-znf-b1h", type=str, required=True, help="Path to the B1H zinc-finger reference MotifCompendium object (.mc), used for Smith-Waterman ZNF alignment.")
    parser.add_argument("--ref-clustered-hr-metadata", type=str, required=True, help="Path to the metadata TSV for --mc-ref-clustered-hr.")
    parser.add_argument("--ref-direct", type=str, help="Path to the ENCODE QC + in vitro reference PFM MotifCompendium object (.mc), used to label direct motifs.")

    # Metadata
    parser.add_argument("--column-desc-tsv", type=str, required=True, help="Path to the TSV describing HTML summary table column names.")
    parser.add_argument("--tf-class", type=str, default=None, help="Path to the Transcription Factor Class file (TSV format).")
    parser.add_argument("--human-tf", type=str, default=None, help="Path to the Human Transcription Factor file (CSV format).")

    # Compute
    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("--var-chunk", action="store_true", help="Starting from max_chunk, dynamically reduce chunk size, based on available CPU/GPU memory.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Do not check the validity of the MotifCompendium objects.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not scale motifs by information content.")
    parser.add_argument("--fast-plot", action="store_true", help="Reduce the resolution of logo images for faster plotting.")
    parser.add_argument("--export-modisco", action="store_true", help="Also export TF-MoDISco HDF5 files (default: off).")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ------------------------------------------------------------
def set_default_configs():
    global MotifCompendiumArgs, TFMoDIScoCols, MetadataCols, TFAnnotationCols, ReferenceMatchCols, LabelCols, MotifMatchArgs
    global DirectTypeArgs, SmithWatermanArgs, FilterArgs, MotifFilterArgs, AtlasFilterArgs, ClusterArgs, VisualizeArgs

    MotifCompendiumArgs = configs.MotifCompendiumArgs()
    TFMoDIScoCols = configs.TFMoDIScoCols()
    MetadataCols = configs.MetadataCols()
    TFAnnotationCols = configs.TFAnnotationCols()
    ReferenceMatchCols = configs.ReferenceMatchCols()
    LabelCols = configs.LabelCols()
    MotifMatchArgs = configs.MotifMatchArgs()
    DirectTypeArgs = configs.DirectTypeArgs()
    SmithWatermanArgs = configs.SmithWatermanArgs()
    FilterArgs = configs.FilterArgs()
    # The per-head stages use MotifFilterArgs (per-head pipeline thresholds); the combine
    # (atlascluster) stage uses AtlasFilterArgs (combine_counts-profile.ipynb thresholds).
    # See configs.AtlasFilterArgs.
    MotifFilterArgs = configs.MotifFilterArgs()
    AtlasFilterArgs = configs.AtlasFilterArgs()
    ClusterArgs = configs.ClusterArgs()
    VisualizeArgs = configs.VisualizeArgs()


#### PATHS ----------------------------------------------------------
def stage_dirs(root):
    """Fixed per-stage directory layout, derived from the --output-dir root."""
    return {
        "counts":  os.path.join(root, "counts"),
        "profile": os.path.join(root, "profile"),
        "combine": os.path.join(root, "combined"),
    }


def stage_args(args, **overrides):
    """A per-stage args namespace: the shared CLI args plus stage-specific wiring."""
    d = vars(args).copy()
    d.update(overrides)
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


#### FUNCTIONS -------------------------------------------------------------
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


def apply_filter_initial(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs | configs.AtlasFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply motif filters to the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs | configs.AtlasFilterArgs): The filter arguments.
        filter_flag_col (str): The column name to use for the filter flag.
        
    Returns:
        None
    """
    # Filter #1: Hard filters
    for hard_filter_args in motif_filter_args.hard_filters:
        if hard_filter_args.apply_motif:
            logging.info(f"Applying hard filter: {hard_filter_args.name}")
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

def apply_filter_motifs(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs | configs.AtlasFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply motif filters to the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs | configs.AtlasFilterArgs): The filter arguments.
        filter_flag_col (str): The column name to use for the filter flag.
        
    Returns:
        None
    """
    # Filter #1: Soft filters
    for filter_args in motif_filter_args.soft_filters:
        if filter_args.apply_motif:
            logging.info(f"Applying filter: {filter_args.name}")
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

    # Filter #2: Soft override: Any match
    for soft_override_filter_args in motif_filter_args.soft_override_filters:
        if soft_override_filter_args.apply_motif:
            logging.info(f"Applying override filter: {soft_override_filter_args.name}")
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
    for moderate_filter_args in motif_filter_args.moderate_filters:
        if moderate_filter_args.apply_motif:
            logging.info(f"Applying moderate filter: {moderate_filter_args.name}")
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
    for moderate_override_filter_args in motif_filter_args.moderate_override_filters:
        if moderate_override_filter_args.apply_motif:
            logging.info(f"Applying moderate override filter: {moderate_override_filter_args.name}")
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
    
    # Filter #5: Hard filters
    for hard_filter_args in motif_filter_args.hard_filters:
        if hard_filter_args.apply_motif:
            logging.info(f"Applying hard filter: {hard_filter_args.name}")
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

def apply_filter_clusters(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs | configs.AtlasFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply cluster filters to the MotifCompendium object.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs | configs.AtlasFilterArgs): The filter arguments.
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

    # Filter #2: Soft override: Any match
    for soft_override_filter_args in motif_filter_args.soft_override_filters:
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
    for moderate_filter_args in motif_filter_args.moderate_filters:
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
    for moderate_override_filter_args in motif_filter_args.moderate_override_filters:
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


def add_num_hits(
    mc: MotifCompendium,
    finemo_paths: str | None,
    num_hits_col: str,
    finemo_path_col: str = "finemo_path",
    name_col: str = "name",
    posneg_col: str = "posneg",
    model_col: str = "model",
    max_workers: int = 1,
) -> bool:
    """
    Add the number of FiNeMo hits per motif, counted in its model's hits file, if a FiNeMo paths file is provided.

    finemo_paths is a TSV without header: per model (input name), its FiNeMo hits file, as BED (the motif name in
    the 4th column, e.g. hits.bed.gz) or FiNeMo hits TSV (a motif_name column). Hits are matched to motifs by
    pos/neg and pattern number, because the TF-MoDISco pattern names can be renamed after FiNeMo ran (e.g.
    pos_patterns.pattern_0 vs. pos_patterns.ENCSR000AHD_ChIP_CTCF_MCF-7_ENCSR867WPY_counts_pattern_0). A motif
    without hits has 0.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        finemo_paths (str | None): Path to the FiNeMo paths file (TSV, no header: model, FiNeMo hits file).
        num_hits_col (str): The column name to save the number of hits to.
        finemo_path_col (str): The column name to save each motif's FiNeMo hits file to.
        name_col (str): The column name of the motif names.
        posneg_col (str): The column name of pos/neg.
        model_col (str): The column name of the models.
        max_workers (int): Number of hits files to read at a time.

    Returns:
        bool: True if num_hits_col was added. False if no FiNeMo paths file is provided, in which
            case the filters on num_hits_col are skipped (see apply_filter_threshold).
    """
    if not finemo_paths:
        logging.info(f"No FiNeMo paths provided: {num_hits_col} is not calculated, and its filters are skipped.")
        return False

    # FiNeMo hits file per model
    finemo_df = pd.read_csv(finemo_paths, sep="\t", header=None, usecols=[0, 1], names=["model", "finemo"], dtype=str)
    if finemo_df["model"].duplicated().any():
        raise ValueError(f"Models are listed more than once in the FiNeMo paths file: {finemo_paths}")
    model_finemo = mc[model_col].astype(str).map(finemo_df.set_index("model")["finemo"])
    if model_finemo.isna().any():
        missing_models = mc.metadata.loc[model_finemo.isna().to_numpy(), model_col].unique()
        raise ValueError(f"No FiNeMo path in {finemo_paths} for {len(missing_models)} model(s): {', '.join(map(str, missing_models[:10]))}")

    def pattern_key(posneg: pd.Series, names: pd.Series) -> pd.Series:
        # "pos" / "neg" + "." + pattern number (from the trailing "pattern_{#}"; subpatterns do not match)
        return posneg.str[:3] + "." + names.str.extract(r"(?:^|[^A-Za-z])pattern_(\d+)$", expand=False)

    motif_keys = pattern_key(mc[posneg_col], mc[name_col])
    if motif_keys.isna().any():
        unkeyed = mc.metadata.loc[motif_keys.isna().to_numpy(), name_col]
        raise ValueError(f"{len(unkeyed)} motif name(s) without a pattern number (subpatterns?), e.g.: {', '.join(unkeyed[:5])}")

    def count_hits(hits_path: str) -> pd.Series:
        # Hits per pos/neg pattern, in one hits file (an empty file: no hits)
        try:
            if hits_path.endswith((".bed", ".bed.gz")):
                motif_names = pd.read_csv(hits_path, sep="\t", header=None, usecols=[3])[3]
            else:
                motif_names = pd.read_csv(hits_path, sep="\t", usecols=["motif_name"])["motif_name"]
        except pd.errors.EmptyDataError:
            return pd.Series(dtype=int)
        # Count per motif name, then key the names (far fewer than the hits)
        name_counts = motif_names.astype(str).value_counts()
        names = name_counts.index.to_series()
        keys = pattern_key(names.str.split(".").str[0], names)
        if keys.isna().any():
            raise ValueError(f"FiNeMo hits with motif names that are not pos/neg patterns ({', '.join(names[keys.isna()][:5])}): {hits_path}")
        return name_counts.groupby(keys.to_numpy()).sum()

    # Count: the hits of each model's hits file
    hits_paths = model_finemo.unique()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        hits_counts = dict(zip(hits_paths, executor.map(count_hits, hits_paths)))

    # Match: (hits file, pos/neg pattern number); a motif without hits has 0
    hits_lookup = pd.concat(hits_counts)
    motif_index = pd.MultiIndex.from_arrays([model_finemo, motif_keys])
    mc[num_hits_col] = hits_lookup.reindex(motif_index).fillna(0).astype(int).to_numpy()
    mc[finemo_path_col] = model_finemo.to_numpy()

    # Check: the hits are of their model's motifs (a hits file none of whose hits match is a mismatch)
    matched = hits_lookup.index.isin(motif_index)
    if (~matched).any():
        paths_matched = set(hits_lookup.index[matched].get_level_values(0))
        paths_unmatched = sorted(set(hits_lookup.index[~matched].get_level_values(0)) - paths_matched)
        if paths_unmatched:
            raise ValueError(f"No FiNeMo hit matches a motif of its model in {len(paths_unmatched)} hits file(s), e.g.: {paths_unmatched[0]}")
        logging.warning(f"{int(hits_lookup[~matched].sum())} FiNeMo hits of {int((~matched).sum())} pattern(s) are not of their model's motifs (ignored).")

    logging.info(f"Added {num_hits_col}, counted in {len(hits_counts)} FiNeMo hits files: {(mc[num_hits_col] == 0).sum()} of {len(mc)} motifs have no hits.")
    return True


def add_input_paths(
    mc: MotifCompendium,
    paths_tsv: str | None,
    path_col: str,
    model_col: str = "model",
) -> None:
    """
    Add a per-model input path to every motif (e.g. the peaks of its model), if a paths file is provided.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        paths_tsv (str | None): Path to the paths file (TSV, no header: model, path).
        path_col (str): The column name to save the paths to.
        model_col (str): The column name of the models.
    """
    if not paths_tsv:
        return
    paths_df = pd.read_csv(paths_tsv, sep="\t", header=None, usecols=[0, 1], names=["model", "path"], dtype=str)
    if paths_df["model"].duplicated().any():
        raise ValueError(f"Models are listed more than once in: {paths_tsv}")
    mc[path_col] = mc[model_col].astype(str).map(paths_df.set_index("model")["path"]).to_numpy()
    missing_models = mc.metadata.loc[mc[path_col].isna(), model_col].unique()
    if len(missing_models):
        logging.warning(f"No {path_col} in {paths_tsv} for {len(missing_models)} model(s): {', '.join(map(str, missing_models[:10]))}")


def save_replace(mc, path):
    """Save mc over path, an input of later stages: to a temporary file first, then replace path with it."""
    temp_path = f"{path}.tmp.mc"  # MotifCompendium.save appends .mc when absent
    if os.path.exists(temp_path):
        os.remove(temp_path)
    mc.save(temp_path)
    os.replace(temp_path, path)


#### STAGE BODIES -----------------------------------------------------------

def run_tf_sequential(args):
    """TF atlas, one head: motif -> cluster -> meta-cluster"""


    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    # A run log is ALWAYS written into <output-dir>/logs/, regardless of how this
    # script was launched, so a finished run is always auditable.
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
        logging.info("Checking output directory...")
        logging.info(f"Output directory: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        logging.info(f"Creating output directory: {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)

    ### PATHS --------------------------------------------------------------------
    ## Derived identifiers
    head_type = args.head_type
    unique_id_initial = f'{MetadataCols.unique_id_col}.{args.mcid_version}'
    label_initial = MetadataCols.atlas_type + "-" + head_type

    ## Derived clustering column names
    cluster_direct_cluster_col = f"Within-{ClusterArgs.direct_cluster_within}_{ClusterArgs.cluster_algorithm_main}-{ClusterArgs.cluster_sim_threshold}_rec_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}"
    cluster_nondirect_cluster_col = f"Within-{ClusterArgs.nondirect_cluster_within}_{ClusterArgs.cluster_algorithm_main}-{ClusterArgs.cluster_sim_threshold}_rec_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}"
    cluster_cluster_col = f"Within-{ClusterArgs.direct_cluster_within}-{ClusterArgs.nondirect_cluster_within}_{ClusterArgs.cluster_algorithm_main}-{ClusterArgs.cluster_sim_threshold}_rec_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}"
    metacluster_cluster_col = f"{ClusterArgs.cluster_algorithm_main}-{ClusterArgs.cluster_sim_threshold}_rec_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}"

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ## MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")
    os.makedirs(mc_dir, exist_ok=True)

    # Motifs
    mc_motif_init_path = os.path.join(mc_dir, "motifcompendium_motif_init.mc")
    mc_motif_path = os.path.join(mc_dir, "motifcompendium_motif_all.mc")
    mc_motif_filtered_path = os.path.join(mc_dir, "motifcompendium_motif_filtered.mc")
    mc_motif_removed_path = os.path.join(mc_dir, "motifcompendium_motif_removed.mc")

    # Clusters
    mc_cluster_path = os.path.join(mc_dir, "motifcompendium_cluster_all.mc")
    mc_cluster_filtered_path = os.path.join(mc_dir, "motifcompendium_cluster_filtered.mc")
    mc_cluster_removed_path = os.path.join(mc_dir, "motifcompendium_cluster_removed.mc")

    # Metaclusters
    mc_metacluster_path = os.path.join(mc_dir, "motifcompendium_metacluster_all.mc")
    mc_metacluster_filtered_path = os.path.join(mc_dir, "motifcompendium_metacluster_filtered.mc")
    mc_metacluster_removed_path = os.path.join(mc_dir, "motifcompendium_metacluster_removed.mc")

    ## HTML summary table
    html_dir = os.path.join(args.output_dir, "html")

    html_motif_removed_path = os.path.join(html_dir, "motifcompendium_motif_removed.html")

    html_cluster_path = os.path.join(html_dir, "motifcompendium_cluster_all.html")
    html_cluster_filtered_path = os.path.join(html_dir, "motifcompendium_cluster_filtered.html")
    html_cluster_removed_path = os.path.join(html_dir, "motifcompendium_cluster_removed.html")
    html_cluster_direct_path = os.path.join(html_dir, "motifcompendium_cluster_direct.html")
    html_cluster_nondirect_path = os.path.join(html_dir, "motifcompendium_cluster_nondirect.html")

    html_metacluster_path = os.path.join(html_dir, "motifcompendium_metacluster_all.html")
    html_metacluster_filtered_path = os.path.join(html_dir, "motifcompendium_metacluster_filtered.html")
    html_metacluster_removed_path = os.path.join(html_dir, "motifcompendium_metacluster_removed.html")

    ## Export: MoDISco HDF5, MEME
    export_dir = os.path.join(args.output_dir, "export")

    modisco_motif_path = os.path.join(export_dir, "motifcompendium_motif_all.h5")
    modisco_motif_filtered_path = os.path.join(export_dir, "motifcompendium_motif_filtered.h5")
    modisco_motif_removed_path = os.path.join(export_dir, "motifcompendium_motif_removed.h5")

    modisco_cluster_path = os.path.join(export_dir, "motifcompendium_cluster_all.h5")
    modisco_cluster_filtered_path = os.path.join(export_dir, "motifcompendium_cluster_filtered.h5")
    modisco_cluster_removed_path = os.path.join(export_dir, "motifcompendium_cluster_removed.h5")

    modisco_metacluster_path = os.path.join(export_dir, "motifcompendium_metacluster_all.h5")
    modisco_metacluster_filtered_path = os.path.join(export_dir, "motifcompendium_metacluster_filtered.h5")
    modisco_metacluster_removed_path = os.path.join(export_dir, "motifcompendium_metacluster_removed.h5")

    meme_motif_path = os.path.join(export_dir, "motifcompendium_motif_all.meme.txt")
    meme_motif_filtered_path = os.path.join(export_dir, "motifcompendium_motif_filtered.meme.txt")
    meme_motif_removed_path = os.path.join(export_dir, "motifcompendium_motif_removed.meme.txt")

    meme_cluster_path = os.path.join(export_dir, "motifcompendium_cluster_all.meme.txt")
    meme_cluster_filtered_path = os.path.join(export_dir, "motifcompendium_cluster_filtered.meme.txt")
    meme_cluster_removed_path = os.path.join(export_dir, "motifcompendium_cluster_removed.meme.txt")

    meme_metacluster_path = os.path.join(export_dir, "motifcompendium_metacluster_all.meme.txt")
    meme_metacluster_filtered_path = os.path.join(export_dir, "motifcompendium_metacluster_filtered.meme.txt")
    meme_metacluster_removed_path = os.path.join(export_dir, "motifcompendium_metacluster_removed.meme.txt")

    ## Export: Metadata
    metadata_dir = os.path.join(args.output_dir, "metadata")

    metadata_motif_path = os.path.join(metadata_dir, "motifcompendium_motif_all_metadata.tsv")
    metadata_cluster_path = os.path.join(metadata_dir, "motifcompendium_cluster_all_metadata.tsv")
    metadata_metacluster_path = os.path.join(metadata_dir, "motifcompendium_metacluster_all_metadata.tsv")
    metadata_motif_index_path = os.path.join(metadata_dir, "motifcompendium_motif_index.tsv")

    args_json_path = os.path.join(metadata_dir, "args.json")

    ### BUILD --------------------------------------------------------------------
    # Load the input MotifCompendium object
    if args.input_mc:
        # Check input
        if args.verbose:
            logging.info("Checking input paths...")
        if not os.path.exists(args.input_mc):
            logging.error(f"Input MotifCompendium object not found: {args.input_mc}")
            raise FileNotFoundError(f"Input MotifCompendium object not found: {args.input_mc}")

        # Load MotifCompendium object
        if args.verbose:
            logging.info(f"Loading input MotifCompendium object: {args.input_mc}...")
        if args.time:
            start_time = time.time()
        mc = MotifCompendium.load(args.input_mc, safe=args.safe)
        if args.verbose:
            logging.info(
                f"Completed loading MotifCompendium object:\n"
                f"  Total number of motifs: {len(mc)}\n"
                f"  Metadata columns: {mc.columns()}"
            )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Load the input Modisco H5 files
    elif args.input_modisco_h5s:
        # Check input
        if args.verbose:
            logging.info("Checking input paths...")
        failed_h5s = []
        for h5 in args.input_modisco_h5s:
            # Check if H5 file exists
            if not os.path.exists(h5):
                logging.error(f"Input H5 file not found: {h5}")
                failed_h5s.append(h5)

            # Check if H5 file is not empty
            elif os.path.getsize(h5) <= MotifCompendiumArgs.min_h5_size_bytes:
                logging.error(f"Input H5 file is empty: {h5}")
                failed_h5s.append(h5)
            
            # Check if H5 file is a valid Modisco file
            else:
                with h5py.File(h5, "r") as f:
                    if "pos_patterns" not in f and "neg_patterns" not in f:
                        logging.error(f"Input H5 file is not a valid Modisco file: {h5}")
                        failed_h5s.append(h5)

        if failed_h5s:
            logging.error(f"Failed to load the following H5 files: {', '.join(failed_h5s)}")
            raise ValueError(f"Failed to load the following H5 files: {', '.join(failed_h5s)}")

        # Build MotifCompendium object
        modisco_names = args.input_names if args.input_names is not None else [os.path.basename(h5) for h5 in args.input_modisco_h5s]
        modisco_dict = dict(zip(modisco_names, args.input_modisco_h5s))
        if args.verbose:
            logging.info("Loading input H5 files...")
        while True:
            try:
                if args.verbose:
                    logging.info("Building MotifCompendium object from input H5 files...")
                if args.time:
                    start_time = time.time()
                mc = MotifCompendium.build_from_modisco(
                    modisco_dict=modisco_dict,
                    load_subpatterns=args.input_subpatterns,
                    modisco_region_width=TFMoDIScoCols.region_width,
                    safe=args.safe,
                )
                if args.verbose:
                    logging.info(
                        f"Completed building MotifCompendium object:\n"
                        f"  Total number of motifs: {len(mc)}\n"
                        f"  Metadata columns: {mc.columns()}"
                    )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")
                break

            except Exception as e:
                logging.error(f"Error: {e}")
                if not args.var_chunk:
                    raise ValueError("Error building MotifCompendium object.")

                args.max_chunk = args.max_chunk - 100
                MotifCompendium.set_compute_options(max_chunk=args.max_chunk)
                logging.info(f"Retrying with max_chunk={args.max_chunk}...")
                if args.max_chunk <= 0:
                    logging.critical("Error building MotifCompendium object. max_chunk has reached 0.")
                    raise ValueError("Error building MotifCompendium object.")

        # Input paths, per model: TF-MoDISco H5
        mc[MetadataCols.modisco_path_col] = mc[MetadataCols.model_col].map(modisco_dict)

    # Load the input PFM files
    elif args.input_pfms:
        # Check input
        if args.verbose:
            logging.info("Checking input paths...")
        failed_pfms = []
        for pfm in args.input_pfms:
            # Check if PFM file exists
            if not os.path.exists(pfm):
                logging.error(f"Input PFM file not found: {pfm}")
                failed_pfms.append(pfm)

            # Check if PFM file is not empty
            elif os.path.getsize(pfm) <= 0:
                logging.error(f"Input PFM file is empty: {pfm}")
                failed_pfms.append(pfm)

        if failed_pfms:
            logging.error(f"Failed to load the following PFM files: {', '.join(failed_pfms)}")
            raise ValueError(f"Failed to load the following PFM files: {', '.join(failed_pfms)}")

        # Build MotifCompendium object from PFMs
        if args.verbose:
            logging.info("Loading input PFM files...")
        if args.time:
            start_time = time.time()
        mc = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(pfm))[0]: pfm for pfm in args.input_pfms},
            safe=args.safe,
        )
        if args.verbose:
            logging.info(
                f"Completed building MotifCompendium object from PFMs:\n"
                f"  Total number of motifs: {len(mc)}\n"
                f"  Metadata columns: {mc.columns()}"
            )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        # Input paths, per model: PFM file
        pfm_dict = {os.path.splitext(os.path.basename(pfm))[0]: pfm for pfm in args.input_pfms}
        mc[MetadataCols.pfm_path_col] = mc["source"].map(pfm_dict)

    else:
        logging.error("Input MotifCompendium object or H5 files must be provided.")
        raise ValueError("Input MotifCompendium object or H5 files must be provided.")

    # Load metadata
    if args.metadata:
        if args.verbose:
            logging.info(f"Loading metadata: {args.metadata}...")
        if args.metadata.endswith(".csv"):
            metadata_df = pd.read_csv(args.metadata)
        elif args.metadata.endswith(".tsv"):
            metadata_df = pd.read_csv(args.metadata, sep="\t")
        else:
            logging.error("Metadata file must be a CSV or TSV file.")
            raise ValueError("Metadata file must be a CSV or TSV file.")
        
        # Metadata cannot contain special column names
        for col in MotifCompendiumArgs.reserved_metadata_columns:
            if col in metadata_df.columns:
                logging.error(f"Metadata file cannot contain column name: {col}")
                raise ValueError(f"Metadata file cannot contain column name: {col}")

        # Single-valued annotations (one per model) are concatenated into ","-separated lists when motifs are
        # aggregated, so a "," inside a value would split it (e.g. celltype "Jurkat,-Clone-E6-1" -> "Jurkat",
        # "-Clone-E6-1"): its parts are joined with "-" instead (biosample "Jurkat-Clone-E6-1")
        for col in [MetadataCols.biosample_col]:
            if col in metadata_df.columns and metadata_df[col].dtype == object:
                mask_comma = metadata_df[col].str.contains(",", na=False)
                if mask_comma.any():
                    logging.warning(f"{col}: ',' inside {mask_comma.sum()} value(s) replaced by '-', e.g. {metadata_df.loc[mask_comma, col].iloc[0]}")
                    metadata_df[col] = metadata_df[col].str.replace(r"\s*,[\s-]*", "-", regex=True)

        # Merge metadata
        if len(metadata_df) == len(mc):
            # Metadata per motif
            mc.metadata = mc.metadata.merge(
                metadata_df,
                left_index=True,
                right_index=True,
                how="left",
                suffixes=("", "_drop"),
            )
        else:
            # Metadata per model
            if "model" not in mc.metadata.columns:
                if "source" in mc.metadata.columns:
                    mc.metadata = mc.metadata.merge(
                        metadata_df,
                        left_on="source",
                        right_on=metadata_df.columns[0],
                        how="left",
                        suffixes=("", "_drop"),
                    )
                else:
                    logging.error("MotifCompendium metadata does not contain 'model' or 'source' column for merging metadata.")
                    raise ValueError("MotifCompendium metadata does not contain 'model' or 'source' column for merging metadata.")
            else:
                mc.metadata = mc.metadata.merge(
                    metadata_df,
                    left_on="model",
                    right_on=metadata_df.columns[0],
                    how="left",
                    suffixes=("", "_drop"),
                )

        # Drop duplicate columns
        mc.metadata = mc.metadata.loc[:, ~mc.metadata.columns.str.endswith("_drop")]
        mc.metadata = mc.metadata.loc[:, ~mc.metadata.columns.duplicated()]

        if args.verbose:
            logging.info(f"Metadata columns: {mc.columns()}")

    # FiNeMo hits: Number of hits per motif, if a FiNeMo paths file is provided
    add_num_hits(
        mc=mc,
        finemo_paths=args.finemo_paths,
        num_hits_col=MetadataCols.num_hits_col,
        finemo_path_col=MetadataCols.finemo_path_col,
        name_col=MetadataCols.name_col,
        posneg_col=MetadataCols.posneg_col,
        model_col=MetadataCols.model_col,
        max_workers=args.max_cpus,
    )

    # Peaks, per model (for the ENCODE hits annotation), if a peaks paths file is provided
    add_input_paths(mc, args.peaks_paths, MetadataCols.peaks_path_col, model_col=MetadataCols.model_col)

    # Add columns: Assay ChIP
    mc[MetadataCols.assay_col] = 'ChIP'
    # mc_znf[MetadataCols.assay_col] = 'ChIP'


    #### PREPROCESS -----------------------------------------------------------------
    ## Family: Annotate TF families
    _FAMILIES = [
        'ARID', 'ARNT', 'BCL', 'BHLH', 'CEBP', 'CREB', 'CDX', 'DMRT', 'ESRR', 'ELS', 'ETS', 'ETL', 'ETV', 'E2F', 
        'FOS', 'FOX', 'GABP', 'GATA', 'GFI', 'HMG', 'HNF', 'HOX',
        'JUN', 'KIAA', 'MAF', 'MAX', 'MEF', 'NFAT', 'NFE', 'NFI', 'NFY', 'NANOG', 'NR', 'NRF', 'NEURO', 'NKX',
        'PAX', 'PHO', 'POU', 'PPAR', 'PRDM', 'REL', 'ROR', 'RXR', 
        'SP', 'SPI', 'STAT', 'TCF', 'CTCF', 'TFAP', 'TFCP', 'TSC22', 'TBX',
        'YY', 'ZBTB', 'ZFP', 'ZNF', 'ZSCAN',
    ]

    def annotate_family(metadata_df, tf_col=MetadataCols.tf_col):
        # Family: Remove number at end
        metadata_df[TFAnnotationCols.family_col] = metadata_df[tf_col].apply(
            lambda x: x[:-3] if (len(x) > 3 and x[-3:].isdigit()) else (x[:-2] if x[-2:].isdigit() else (x[:-1] if x[-1:].isdigit() else x))
        )

        # Family: If matched with known family
        for family in _FAMILIES:
            metadata_df.loc[metadata_df[tf_col].str.upper().str.startswith(family), TFAnnotationCols.family_col] = family

        # Remove TFAnnotationCols.ref_tf_col from family name
        # metadata_df[TFAnnotationCols.family_col] = metadata_df[TFAnnotationCols.family_col].apply(lambda x: x[2:] if x.startswith(TFAnnotationCols.ref_tf_col) else x)

        # MX = MAX
        metadata_df.loc[metadata_df[tf_col].str.contains("MX"), TFAnnotationCols.family_col] = "MAX"


    # TF: Annotate
    logging.info("Annotating TF families...")
    annotate_family(mc.metadata)

    ## TF Class
    if args.tf_class:
        # Load TF Class
        tfclass_df = pd.read_csv(args.tf_class, sep="\t")
        hs_df = tfclass_df[tfclass_df['Organism'] == 'Homo_sapiens']

        def match_family(target):
            matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]
            if matches.empty:
                if target.startswith("TF"):
                    target = target[2:]
                    matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]
                elif target.startswith("T"):
                    target = target[1:]
                    matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]

            if not matches.empty:
                return matches.iloc[0]['Family Name']
            return None

        def match_family_id(target):
            matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]
            if matches.empty:
                if target.startswith("TF"):
                    target = target[2:]
                    matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]
                elif target.startswith("T"):
                    target = target[1:]
                    matches = tfclass_df[tfclass_df['Genus Name'].str.contains(str(target), case=False, na=False) | tfclass_df['TF Name'].str.contains(str(target), case=False, na=False)]

            if not matches.empty:
                return matches.iloc[0]['Family ID']
            return None
        
        def annotate_tfclass(metadata_df, tf_col):
            # Annotate family type and ID
            metadata_df[TFAnnotationCols.tfclass_family_name_col] = metadata_df[tf_col].apply(match_family)
            metadata_df[TFAnnotationCols.tfclass_family_id_col] = metadata_df[tf_col].apply(match_family_id)
            
            # Clean tfclass_family_name_col
            metadata_df[TFAnnotationCols.tfclass_family_name_col] = metadata_df[TFAnnotationCols.tfclass_family_name_col].str.split().str.join('-')

        # TF Class: Annotate family
        logging.info("Annotating from: TF Class...")
        annotate_tfclass(mc.metadata, MetadataCols.tf_col)

    ## Human Transcription Factor
    if args.human_tf:
        # Load Human TF CSV
        humantf_df = pd.read_csv(args.human_tf)

        def annotate_humantf(metadata_df, tf_col):
            # Annotate DBD
            dbd_map = humantf_df.set_index('HGNC symbol')['DBD'].to_dict()
            metadata_df[TFAnnotationCols.humantf_dbd_col] = metadata_df[tf_col].map(dbd_map)

            # Clean DBD
            metadata_df[TFAnnotationCols.humantf_dbd_col] = metadata_df[TFAnnotationCols.humantf_dbd_col].str.replace(';', '').str.replace(' ', '-')
        
        # Human TF: Annotate
        logging.info("Annotating from: Human Transcription Factor...")
        annotate_humantf(mc.metadata, MetadataCols.tf_col)


    # Rename motifs: Original TF-Modisco motif name
    mc[MetadataCols.name_col] = (
        mc[MetadataCols.posneg_col]
        + "_" + "patterns."
        + mc[MetadataCols.id_col]
        + "_" + mc[MetadataCols.assay_col]
        + "_" + mc[MetadataCols.tf_col]
        + "_" + mc[MetadataCols.biosample_col].str.replace(" ", "-").str.replace("/", "-")
        + "_" + mc[MetadataCols.accession_col]
        + "_" + mc[MetadataCols.head_col]
        + "_" + "pattern"
        + "_" + mc[MetadataCols.name_col].str.split("_").str[-1]
    )
    # E.g., 
    # From: ENCSR590KEQ_profile-pos.ENCSR590KEQ_ChIP_ARNT_GM12878_ENCSR728JEK_profile_pattern_0
    # To: pos_patterns.ENCSR828NCB_ChIP_GATAD2B_GM12878_ENCSR308RCC_profile_pattern_9


    # # Label: ZNF vs. Non-ZNF
    # # ZNF: All ZNF
    # mc_znf[MetadataCols.znf_col] = 'znf'

    # ENCODE: Identify ZNFs
    mc_znf_mask = (
        mc[TFAnnotationCols.tfclass_family_name_col].astype(str).str.contains("zinc|ZF", case=False, na=False)
        | mc[TFAnnotationCols.humantf_dbd_col].astype(str).str.contains("C2H2", case=False, na=False)
        | mc[TFAnnotationCols.family_col].astype(str).str.contains("ZNF|ZSCAN|ZKSCAN|ZBTB", case=False, na=False)
        | mc[TFAnnotationCols.family_col].astype(str).str.lower().str.startswith("z", na=False)
    )
    mc[MetadataCols.znf_col] = np.where(mc_znf_mask, 'znf', 'non-znf')
    mc_znf = mc[mc_znf_mask]


    ## Add: Unique ID
    mc[MetadataCols.unique_id_col] = unique_id_initial + f".C0.{MetadataCols.atlas_type}.{head_type}." + mc.metadata.index.map('{:08d}'.format)


    # ### Calculate metrics
    logging.info("Calculating motif metrics and filter metrics...")
    # Calculate contribution score
    mc[MetadataCols.avgcontrib_score_col] = mc.motifs.sum(axis=(1,2))
    mc[MetadataCols.total_contrib_score_col] = mc[MetadataCols.num_seqlets_col] * mc[MetadataCols.avgcontrib_score_col]
    mc[MetadataCols.group_total_contrib_score_col] = mc.metadata.groupby(MetadataCols.id_col)[MetadataCols.total_contrib_score_col].transform('sum')
    mc[MetadataCols.total_contrib_score_ratio_col] = mc[MetadataCols.total_contrib_score_col] / mc[MetadataCols.group_total_contrib_score_col]


    # Calculate filter metrics
    mc[FilterArgs.filter_flag_col] = False
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )
    # Calculate filters: No scaling, trim zeros
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=['posneg_inverted', 'truncated'],
        trim_importance=0,
    )

    # Save MotifCompendium: Initial checkpoint (pre-annotation)
    logging.info(f"Saving initial MotifCompendium object: {mc_motif_init_path}...")
    os.makedirs(os.path.dirname(mc_motif_init_path), exist_ok=True)
    mc.save(mc_motif_init_path)

    # ## Annotate motifs
    logging.info("Annotating motifs against the reference...")
    # - (1) **Direct ALPHA**: Direct motif
    #     - Matched with known *in vitro* or CORE-only PFM
    # - (2) **Direct BETA**: TF Paralog, ZNF B1H
    #     - (For TFs without/no match with Direct ALPHA)
    #     - Non-ZNFs: Matched with annotated paralog's *in vivo* or *in vitro* PFM (non-ZNFs)
    #     - ZNFs: Matched with ZNF B1H via Modified Smith-Waterman
    # - (3) **Direct GAMMA**: Top total contrib score
    #     - (For TFs without/no match with Direct ALPHA, BETA)
    #     - Top k=1 total contrib score (= num_seqlets x avg_contrib)
    # - (4) **Indirect**: Others
    #     - Remaining motifs, annotated by human-readable
    # - (5) **Removed**: Filtered out
    #     - Removed motifs, by motif filters

    # ### Import: All reference databases
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
    if args.ref_znf_b1h.endswith(".mc"):
        mc_b1h = MotifCompendium.load(args.ref_znf_b1h, safe=args.safe)
    else:
        mc_b1h = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref_znf_b1h))[0]: args.ref_znf_b1h},
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

    # # TF: Annotate
    mc_refs = [mc_ref_qc, mc_ref_invitro]
    for mc_ref_i in mc_refs:
        annotate_tfclass(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)
        annotate_humantf(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)
        annotate_family(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)

    # Remove multimers
    mc_ref_qc = mc_ref_qc[~mc_ref_qc[MetadataCols.name_col].str.contains('::')]
    mc_ref_direct = mc_ref_direct[~mc_ref_direct[MetadataCols.name_col].str.contains('::')]

    # IC scale
    mc_ref_qc.motifs = utils_motif.ic_scale(mc_ref_qc.motifs)
    mc_ref_direct.motifs = utils_motif.ic_scale(mc_ref_direct.motifs)
    mc_b1h.motifs = utils_motif.ic_scale(mc_b1h.motifs)

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
    mc_ref_qc_tf_dict = dict(zip(mc_ref_qc[MetadataCols.name_col], mc_ref_qc[TFAnnotationCols.ref_tf_col]))

    mc_ref_qc[LabelCols.label_col] = mc_ref_qc[MetadataCols.name_col].map(mc_ref_qc_hr_dict)

    # ### Annotate: Motifs
    # **(4) Indirect: Others**
    # - Reference: Reference QC (human-readable)
    # - Reference: Reference QC (non-ZNF; human-readable)
    # 
    # **(5) Removed**
    # - By motif filters
    # 
    # **(1) Direct ALPHA: Direct motif**
    # - Per TF:
    #     - Reference: TF's Reference *in vitro* + QC [direct monomer, homo-multimer]
    #     - Reference: TF's Reference *in vitro* + QC + All Reference QC (human-readable) [direct hetero-multimer]
    # 
    # **(2) Direct BETA: TF Paralog, ZNF B1H**
    # - (Non-ZNFs) Per TF:
    #     - Reference: TF family's Reference QC (human-readable) [direct monomer, homo-multimer]
    #     - Reference: TF family's Reference QC (human-readable) + All Reference QC (human-readable) [direct hetero-multimer]
    # - (ZNFs) Per TF:
    #     - From NOT annotated (in (4)): min single, min composite
    #     - Rank by Modified Smith-Waterman
    #     - Threshold: 0.7
    #     - If none, top 1 between 0.5 - 0.6
    # 
    # **(3) Direct GAMMA: Top total contrib score**
    # - (Otherwise)
    # - Top k=1 total contrib score (= num_seqlets x avg_contrib)

    # #### (5-1) Removed
    # - Initial remove: Pos/neg inverted, truncated

    # Apply filters
    apply_filter_initial(
        mc=mc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )


    # Removed: mc = mc_qc + mc_removed
    mc_removed = mc[mc[FilterArgs.filter_flag_col]]
    mc_qc = mc[~mc[FilterArgs.filter_flag_col]]

    # #### (4) Indirect: Others
    # Generally, annotate all motifs, with human-readable names

    ## Initialize columns
    mc_qc.metadata = mc_qc.metadata.assign(**{
        DirectTypeArgs.direct_type_col:                None,
        DirectTypeArgs.tf_direct_col:                  None,
        DirectTypeArgs.family_direct_col:              None,
        MetadataCols.num_subunit_col:                1,
        MetadataCols.num_subunit_direct_col:         None,
        MetadataCols.composite_col:                  False,
        LabelCols.label_core_col:                 None,
        LabelCols.direct_label_core_col:          None,
        LabelCols.label_core_readable_col:        None,
        LabelCols.direct_label_core_readable_col: None,
        LabelCols.label_flank_col:                None,
    }).copy()

    ## [General] Label motifs
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_qc,
        assign_from_mc=mc_ref_qc,
        from_label_col=MetadataCols.name_col,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=False,
        logo_trimming=0,
    )

    ## Annotate:
    # Map human-readable name
    ref_readable_name_cols = [f'{ReferenceMatchCols.ref_readable_col}_name{i}' for i in range(MotifMatchArgs.max_submotifs)]
    ref_score_cols = [f'{ReferenceMatchCols.ref_col}_score{i}' for i in range(MotifMatchArgs.max_submotifs)]
    for i, (ref_readable_name_col, ref_score_col) in enumerate(zip(ref_readable_name_cols, ref_score_cols)):    
        mc_qc[ref_readable_name_col] = mc_qc[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_monomer_cluster = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_cluster = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_cluster = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_cluster = (
        mask_dimer_cluster | mask_trimer_cluster
    )
    mask_matched_cluster = (
        mask_monomer_cluster | mask_dimer_cluster | mask_trimer_cluster
    )

    # Dimer-style notation
    mc_qc.metadata.loc[mask_matched_cluster, LabelCols.label_core_col] = mc_qc.metadata.loc[mask_matched_cluster, ref_readable_name_cols[0]]
    mc_qc.metadata.loc[mask_matched_cluster, LabelCols.label_core_readable_col] = mc_qc.metadata.loc[mask_matched_cluster, ref_readable_name_cols[0]]
    mc_qc.metadata.loc[mask_dimer_cluster, LabelCols.label_flank_col] = mc_qc.metadata.loc[mask_dimer_cluster, ref_readable_name_cols[1]]
    mc_qc.metadata.loc[mask_trimer_cluster, LabelCols.label_flank_col] = mc_qc.metadata.loc[mask_trimer_cluster, ref_readable_name_cols[1]] + "::" + mc_qc.metadata.loc[mask_trimer_cluster, ref_readable_name_cols[2]]
    mc_qc.metadata.loc[mask_dimer_cluster, MetadataCols.num_subunit_col] = 2
    mc_qc.metadata.loc[mask_trimer_cluster, MetadataCols.num_subunit_col] = 3
    mc_qc.metadata[MetadataCols.composite_col] = mc_qc.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit  # Composite: True, False


    # Concatenate all unique annotations
    mc_qc[ReferenceMatchCols.ref_name_col] = mc_qc.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

    ## [General] Label motifs (Non-ZNF)
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_qc,
        assign_from_mc=mc_ref_qc_nonznf,
        from_label_col=MetadataCols.name_col,
        save_col_prefix=ReferenceMatchCols.nonznf_ref_col,
        min_score=MotifMatchArgs.min_single_score, # MotifMatchArgs.min_composite_score
        max_submotifs=1, # MotifMatchArgs.max_submotifs
        label_unsigned=True,
        save_images=False,
        logo_trimming=0,
    )

    # Map human-readable name
    for submotif_i in range(1): # range(MotifMatchArgs.max_submotifs)
        mc_qc[f'{ReferenceMatchCols.nonznf_ref_readable_col}_name{submotif_i}'] = mc_qc[f'{ReferenceMatchCols.nonznf_ref_col}_name{submotif_i}'].map(mc_ref_qc_hr_dict)

    # #### (2) Direct motifs: BETA (ZNF) - B1H alignment
    ## Run Modified Smith-Waterman
    # Record: Score, alignment
    mc_qc.metadata[SmithWatermanArgs.msw_score_col] = 0.0
    mc_qc.metadata[SmithWatermanArgs.msw_sim_col] = 0.0
    mc_qc.metadata[SmithWatermanArgs.msw_orient_col] = None
    mc_qc.metadata[SmithWatermanArgs.msw_align_col] = None
    mc_qc.metadata[SmithWatermanArgs.msw_action_col] = None
    aligned_motifs = np.zeros((mc_qc.motifs.shape[0], mc_qc.motifs.shape[1] + 2 * (SmithWatermanArgs.finger_length - 1), mc_qc.motifs.shape[2]))

    # Per TF
    for i, row_b1h in mc_b1h.metadata.iterrows():
        ## TF
        tf_i = row_b1h[TFAnnotationCols.ref_tf_col]
        logging.info(f"Processing: {tf_i}")
        
        ## B1H
        mc_b1h_i = mc_b1h[[i]]
        name_b1h = row_b1h[MetadataCols.name_col]
        b1h_ppm_ic_i = utils_motif.trim_motif(mc_b1h_i.motifs[0], 0) # (L, 4), trim zeros
        logging.info(f"  B1H: {name_b1h}")

        ## CWMs
        mc_qc_i = mc_qc[mc_qc.metadata[MetadataCols.tf_col] == tf_i]
        cwm_i = mc_qc_i.motifs
        
        if len(mc_qc_i) > 0:
            ## Run dynamic programming: Per CWM
            for j, row_cwm in mc_qc_i.metadata.iterrows():
                name_cwm = row_cwm[MetadataCols.name_col]
                cwm_ij = utils_motif.trim_motif(cwm_i[j] , 0) # (L, 4), trim zeros
                # logging.info(f"  CWM: {name_cwm}")

                (msw_score, sim_score, msw_aligned_b1h, msw_orient, msw_align_b1h, msw_align_cwm, msw_scale_f, msw_action, _, _,) = utils_composite.run_modified_smith_waterman_alignment(
                    target_motif=cwm_ij,
                    reference_motif=b1h_ppm_ic_i,
                    sim_threshold=SmithWatermanArgs.msw_sim_threshold,
                    overlap_penalty_f1=SmithWatermanArgs.overlap_penalty_f1,
                    overlap_penalty_f2=SmithWatermanArgs.overlap_penalty_f2,
                    skip_penalty_f1=SmithWatermanArgs.skip_penalty_f1, 
                    skip_penalty_f2=SmithWatermanArgs.skip_penalty_f2,
                    skip_penalty_l1=SmithWatermanArgs.skip_penalty_l1, 
                    skip_penalty_l2=SmithWatermanArgs.skip_penalty_l2,
                    finger_length=SmithWatermanArgs.finger_length,
                )
                
                # Aligned B1H motif (collapsed)
                msw_aligned_b1h_collapse = msw_aligned_b1h.sum(axis=0)
                msw_aligned_b1h_collapse = utils_motif.pad_motif(
                    msw_aligned_b1h_collapse[np.newaxis, :, :], aligned_motifs.shape[1])

                # Record, in original MotifCompendium
                org_idx = (mc_qc.metadata[MetadataCols.name_col] == name_cwm).idxmax()
                if msw_score > row_cwm[SmithWatermanArgs.msw_score_col]:
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_score_col] = msw_score
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_sim_col] = sim_score
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_orient_col] = msw_orient
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_align_col] = "\n".join("".join(str(msw_align_pos)) for msw_align_pos in [msw_align_b1h, msw_align_cwm])
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_action_col] = ";".join(msw_action)
                    mc_qc.metadata.loc[org_idx, SmithWatermanArgs.msw_name_col] = name_b1h
                    aligned_motifs[org_idx] = msw_aligned_b1h_collapse # Save aligned motif

    # Visualize aligned motifs, in original MotifCompendium
    mc_qc.add_logos(
        motifs=aligned_motifs,
        image_name=SmithWatermanArgs.msw_logo_col,
        trim=0,
    )

    # #### (1,2,3) Direct motifs: ALPHA, BETA (non-ZNF), BETA (ZNF), GAMMA
    ## [Direct] Label motifs
    mc_metadata_updates = []
    for i, tf in enumerate(mc_qc[MetadataCols.tf_col].unique()):
        # Slice: MotifCompendium
        mc_i = mc_qc[mc_qc[MetadataCols.tf_col] == tf]
        tf_family = mc_i.metadata[TFAnnotationCols.family_col].unique()[0]
        logging.info(f"Processing {i+1}: {tf} ({len(mc_i)})")
        
        # Slice: Reference
        mc_ref_alpha_i = mc_ref_direct[mc_ref_direct[TFAnnotationCols.ref_tf_col] == tf]
        mc_ref_beta_nzf_i = mc_ref_qc[mc_ref_qc[TFAnnotationCols.ref_family_col] == tf_family]
        mc_ref_beta_zf_i = mc_ref_qc[mc_ref_qc[TFAnnotationCols.ref_family_col] != tf_family]
        
        # Check history
        check_alpha = False
        check_beta_nzf = False
        check_beta_zf = False
        check_gamma = False

        # Initialize mc_direct_is
        mc_direct_alpha_i = pd.DataFrame() # For len() = 0
        mc_direct_beta_nzf_i = pd.DataFrame() # For len() = 0
        mc_direct_beta_zf_i = pd.DataFrame() # For len() = 0

        ## [Direct ALPHA] Label motifs, using invitro+qc
        if len(mc_ref_alpha_i) > 0  and len(mc_i) > 0:
            ## Check ALPHA
            logging.info(f"  Direct Alpha ref: {len(mc_ref_alpha_i)}")
            check_alpha = True
            
            ## (1) Homo: Monomer, homo-multimer
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_i,
                assign_from_mc=mc_ref_alpha_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs,
                label_unsigned=True,
                save_images=False,
                logo_trimming=0,
            )

            # Annotate human-readable name: TF family
            for submotif_i in range(MotifMatchArgs.max_submotifs):
                mc_i.metadata.loc[
                    mc_i.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score,
                    f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                ] = tf_family
            
            ## (2) Hetero-multimer
            # Re-label "homo-monomer" to monomer, hetero-multimer
            mc_i_monomer = mc_i[
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score0"] > MotifMatchArgs.min_composite_score) &
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score1"] < MotifMatchArgs.min_composite_score) &
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score2"] < MotifMatchArgs.min_composite_score)
            ]
            if len(mc_i_monomer) > 0:
                assign_direct_composite(
                    target_mc=mc_i_monomer,
                    direct_reference_mc=mc_ref_alpha_i,
                    indirect_reference_mc=mc_ref_qc,
                    save_col_prefix=ReferenceMatchCols.direct_ref_col,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs,
                    label_unsigned=True,
                    save_images=False,
                    logo_trimming=0,
                )
                # Annotate human-readable name: TF family
                for submotif_i in range(1, MotifMatchArgs.max_submotifs):  # Keep first as TF
                    mc_i_monomer.metadata.loc[
                        mc_i_monomer.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score,
                        f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                    ] = mc_i_monomer[f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}'].map(mc_ref_qc_hr_dict)

                # Assign true monomer + hetero-multimer back to original mc_i
                mc_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
                mc_i.metadata = mc_i.metadata.combine_first(mc_i_monomer.metadata.set_index(MetadataCols.unique_id_col))
                mc_i.metadata.update(mc_i_monomer.metadata.set_index(MetadataCols.unique_id_col))
                mc_i.metadata.reset_index(inplace=True)

            ## Direct: Type, name, readable name
            # Concatenate all unique direct reference annotations
            mc_i[ReferenceMatchCols.direct_ref_name_col] = mc_i.metadata[
                [f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
            ].agg(
                lambda x: ",".join(pd.unique(x.dropna().astype(str))),
                axis=1
            )

            # Build up masks, per subunit
            mask_subunit_df = pd.DataFrame({submotif_i: (
                    mc_i[f"{ReferenceMatchCols.direct_ref_col}_score{submotif_i}"] > (MotifMatchArgs.min_single_score if submotif_i == 0 else MotifMatchArgs.min_composite_score)
                ) for submotif_i in range(MotifMatchArgs.max_submotifs)
            }, index=mc_i.metadata.index)

            # Subunit: Names
            direct_ref_readable_name_subunit_df = pd.concat([
                mc_i.metadata[f"{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}"]
                for submotif_i in range(MotifMatchArgs.max_submotifs)], axis=1
            )

            direct_ref_readable_name_subunit_df = direct_ref_readable_name_subunit_df.where(mask_subunit_df.values)
            direct_ref_readable_composite = (
                direct_ref_readable_name_subunit_df.iloc[:, 1:]
                .fillna("")
                .agg("-".join, axis=1)
                .str.strip("-")
                .str.replace("--", "-", regex=False)
            )

            # Subunit: Number
            num_subunit = mask_subunit_df.sum(axis=1)

            # Mask: Any direct motifs
            mask_match = mask_subunit_df.any(axis=1)
            mc_i.metadata.loc[mask_match, DirectTypeArgs.direct_type_col]                = DirectTypeArgs.direct_types[0]  # alpha
            mc_i.metadata.loc[mask_match, DirectTypeArgs.tf_direct_col]                  = tf
            mc_i.metadata.loc[mask_match, DirectTypeArgs.family_direct_col]              = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.label_core_col]                 = tf
            mc_i.metadata.loc[mask_match, LabelCols.direct_label_core_col]          = tf
            mc_i.metadata.loc[mask_match, LabelCols.label_core_readable_col]        = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.direct_label_core_readable_col] = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.label_flank_col]                = direct_ref_readable_composite[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.num_subunit_col]                = num_subunit[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.num_subunit_direct_col]         = num_subunit[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.composite_col]                  = (num_subunit[mask_match] > MotifMatchArgs.min_composite_subunit).values
            
            # Select direct: Alpha
            mc_direct_alpha_i = mc_i[mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]]
            mc_direct_alpha_monomer_i = mc_i[
                (mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]) & (mc_i[MetadataCols.num_subunit_direct_col] <= MotifMatchArgs.min_composite_subunit)
            ]
            mc_direct_alpha_multimer_i = mc_i[
                (mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]) & (mc_i[MetadataCols.num_subunit_direct_col] > MotifMatchArgs.min_composite_subunit)
            ]
            logging.info(f"  Alpha (monomer): {len(mc_direct_alpha_monomer_i)} (ref: {len(mc_ref_alpha_i)})")
            logging.info(f"  Alpha (multimer): {len(mc_direct_alpha_multimer_i)} (ref: {len(mc_ref_alpha_i)})")

            # Update original MotifCompendium metadata
            for mc_direct_alpha_subunit_i in [mc_direct_alpha_monomer_i, mc_direct_alpha_multimer_i]:
                mc_metadata_updates.append(mc_direct_alpha_subunit_i.metadata.set_index(MetadataCols.unique_id_col))

            # Continue with unannotated motifs
            mc_i = mc_i[mc_i[DirectTypeArgs.direct_type_col] != DirectTypeArgs.direct_types[0]]
        
        ## [Direct BETA: Non-ZNF] Label motifs, using paralog QC (with human-readable name)
        if (len(mc_direct_alpha_i) == 0) and (len(mc_ref_beta_nzf_i) > 0) and (mc_i[MetadataCols.znf_col] == 'non-znf').all() and len(mc_i) > 0:
            ## Check BETA: Non-ZNF
            logging.info(f"  Direct Beta ref (non-ZNF): {len(mc_ref_beta_nzf_i)}")
            check_beta_nzf = True
            
            ## (1) Monomer, homo-multimer
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_i,
                assign_from_mc=mc_ref_beta_nzf_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs,
                label_unsigned=True,
                save_images=False,
                logo_trimming=0,
            )

            # Annotate human-readable name: TF family
            for submotif_i in range(MotifMatchArgs.max_submotifs):
                mc_i.metadata.loc[
                    mc_i.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score,
                    f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                ] = tf_family
            
            ## (2) Hetero-multimer
            # Re-label "homo-monomer" to monomer, hetero-multimer
            mc_i_monomer = mc_i[
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score0"] > MotifMatchArgs.min_composite_score) &
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score1"] < MotifMatchArgs.min_composite_score) &
                (mc_i[f"{ReferenceMatchCols.direct_ref_col}_score2"] < MotifMatchArgs.min_composite_score)
            ]
            if len(mc_i_monomer) > 0:
                assign_direct_composite(
                    target_mc=mc_i_monomer,
                    direct_reference_mc=mc_ref_beta_nzf_i,
                    indirect_reference_mc=mc_ref_qc,
                    save_col_prefix=ReferenceMatchCols.direct_ref_col,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs,
                    label_unsigned=True,
                    save_images=False,
                    logo_trimming=0,
                )
                # Annotate human-readable name: TF family
                for submotif_i in range(1, MotifMatchArgs.max_submotifs):  # Keep first as TF family
                    mc_i_monomer.metadata.loc[
                        mc_i_monomer.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score,
                        f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                    ] = mc_i_monomer[f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}'].map(mc_ref_qc_hr_dict)
                

                # Assign true monomer + hetero-multimer back to original mc_i
                mc_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
                mc_i.metadata = mc_i.metadata.combine_first(mc_i_monomer.metadata.set_index(MetadataCols.unique_id_col))
                mc_i.metadata.update(mc_i_monomer.metadata.set_index(MetadataCols.unique_id_col))
                mc_i.metadata.reset_index(inplace=True)

            ## Direct: Type, name, readable name
            # Concatenate all unique direct reference annotations
            mc_i[ReferenceMatchCols.direct_ref_name_col] = mc_i.metadata[
                [f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
            ].agg(
                lambda x: ",".join(pd.unique(x.dropna().astype(str))),
                axis=1
            )

            # Build up masks, per subunit
            mask_subunit_df = pd.DataFrame({submotif_i: (
                    mc_i[f"{ReferenceMatchCols.direct_ref_col}_score{submotif_i}"] > (
                        MotifMatchArgs.min_single_score if submotif_i == 0 else MotifMatchArgs.min_composite_score)) 
                for submotif_i in range(MotifMatchArgs.max_submotifs)
            }, index=mc_i.metadata.index)

            # Subunit: Names
            direct_ref_readable_name_subunit_df = pd.concat([
                mc_i.metadata[f"{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}"]
                for submotif_i in range(MotifMatchArgs.max_submotifs)], axis=1
            )

            direct_ref_readable_name_subunit_df = direct_ref_readable_name_subunit_df.where(mask_subunit_df.values)
            direct_ref_readable_composite = (
                direct_ref_readable_name_subunit_df.iloc[:, 1:]
                .fillna("")
                .agg("-".join, axis=1)
                .str.strip("-")
                .str.replace("--", "-", regex=False)
            )

            # Subunit: Number
            num_subunit = mask_subunit_df.sum(axis=1)

            # Mask: Any direct motifs
            mask_match = mask_subunit_df.any(axis=1)
            mc_i.metadata.loc[mask_match, DirectTypeArgs.direct_type_col]                = DirectTypeArgs.direct_types[1]  # beta (non-ZNF)
            mc_i.metadata.loc[mask_match, DirectTypeArgs.tf_direct_col]                  = tf
            mc_i.metadata.loc[mask_match, DirectTypeArgs.family_direct_col]              = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.label_core_col]                 = tf
            mc_i.metadata.loc[mask_match, LabelCols.direct_label_core_col]          = tf
            mc_i.metadata.loc[mask_match, LabelCols.label_core_readable_col]        = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.direct_label_core_readable_col] = tf_family
            mc_i.metadata.loc[mask_match, LabelCols.label_flank_col]                = direct_ref_readable_composite[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.num_subunit_col]                = num_subunit[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.num_subunit_direct_col]         = num_subunit[mask_match].values
            mc_i.metadata.loc[mask_match, MetadataCols.composite_col]                  = (num_subunit[mask_match] > MotifMatchArgs.min_composite_subunit).values

            # Select direct: Beta Non-ZNF
            mc_direct_beta_nzf_i = mc_i[mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]]
            mc_direct_beta_nzf_monomer_i = mc_i[
                (mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]) & (mc_i[MetadataCols.num_subunit_direct_col] <= MotifMatchArgs.min_composite_subunit)
            ]
            mc_direct_beta_nzf_multimer_i = mc_i[
                (mc_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]) & (mc_i[MetadataCols.num_subunit_direct_col] > MotifMatchArgs.min_composite_subunit)
            ]
            logging.info(f"  Beta Non-ZNF (monomer): {len(mc_direct_beta_nzf_monomer_i)} (ref: {len(mc_ref_beta_nzf_i)})")
            logging.info(f"  Beta Non-ZNF (multimer): {len(mc_direct_beta_nzf_multimer_i)} (ref: {len(mc_ref_beta_nzf_i)})")

            # Update original MotifCompendium metadata
            for mc_direct_beta_nzf_subunit_i in [mc_direct_beta_nzf_monomer_i, mc_direct_beta_nzf_multimer_i]:
                mc_metadata_updates.append(mc_direct_beta_nzf_subunit_i.metadata.set_index(MetadataCols.unique_id_col))

            # Continue with unannotated motifs
            mc_i = mc_i[mc_i[DirectTypeArgs.direct_type_col] != DirectTypeArgs.direct_types[1]]
        
        ## [Direct BETA: ZNF] Select by Modified Smith-Waterman
        elif (len(mc_direct_alpha_i) == 0) and (mc_i[MetadataCols.znf_col] == 'znf').all() and (mc_i[SmithWatermanArgs.msw_score_col].notna()).any() and len(mc_i) > 0:
            ## Check BETA: ZNF
            logging.info(f"  Direct Beta ref (ZNF)")
            check_beta_zf = True
            
            ## Match: Non-ZNFs
            mc_znf_i = mc_i[~(mc_i[f'{ReferenceMatchCols.nonznf_ref_col}_score0'] > MotifMatchArgs.min_single_score)]
            
            # Sort by Modified Smith-Waterman score
            mc_znf_i.sort(
                by=SmithWatermanArgs.msw_score_col,
                ascending=False,
                inplace=True,
            )
            
            ## Identify direct motif
            # (1) Above threshold
            mask_match = mc_znf_i[SmithWatermanArgs.msw_score_col] > SmithWatermanArgs.min_msw_score

            # (2) Top k = 1, above threshold
            if not mask_match.any():
                idx_2 = mc_znf_i.metadata[mc_znf_i[SmithWatermanArgs.msw_score_col] > SmithWatermanArgs.min_topk_msw_score].index
                for k in range(SmithWatermanArgs.topk, 0, -1):
                    if len(idx_2) >= k:
                        mask_match = mc_znf_i.metadata.index.isin(idx_2[:k])
                        break
            
            # Update metadata
            mc_znf_i.metadata.loc[mask_match, DirectTypeArgs.direct_type_col]                = DirectTypeArgs.direct_types[2]  # beta (ZNF)
            mc_znf_i.metadata.loc[mask_match, DirectTypeArgs.tf_direct_col]                  = tf
            mc_znf_i.metadata.loc[mask_match, DirectTypeArgs.family_direct_col]              = tf_family
            mc_znf_i.metadata.loc[mask_match, MetadataCols.num_subunit_col]                = 1.0
            mc_znf_i.metadata.loc[mask_match, MetadataCols.num_subunit_direct_col]         = 1.0
            mc_znf_i.metadata.loc[mask_match, MetadataCols.composite_col]                  = False
            mc_znf_i.metadata.loc[mask_match, LabelCols.label_core_col]                 = tf
            mc_znf_i.metadata.loc[mask_match, LabelCols.direct_label_core_col]          = tf
            mc_znf_i.metadata.loc[mask_match, LabelCols.label_core_readable_col]        = tf_family
            mc_znf_i.metadata.loc[mask_match, LabelCols.direct_label_core_readable_col] = tf_family
            mc_znf_i.metadata.loc[mask_match, LabelCols.label_flank_col]                = None
            mc_znf_i.metadata.loc[mask_match, ReferenceMatchCols.znf_ref_name_col]               = mc_znf_i.metadata.loc[mask_match, SmithWatermanArgs.msw_name_col].values
            
            # Select direct: Beta ZNF
            mc_direct_beta_zf_i = mc_znf_i[mc_znf_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]]
            logging.info(f"  Beta ZNF: {len(mc_direct_beta_zf_i)} ")

            # Update original MotifCompendium metadata
            mc_metadata_updates.append(mc_direct_beta_zf_i.metadata.set_index(MetadataCols.unique_id_col))
            
            # Continue with unannotated motifs
            mc_i = mc_i[~mc_i[MetadataCols.name_col].isin(mc_direct_beta_zf_i[MetadataCols.name_col])]
        
        ## [Direct GAMMA] Top total contrib
        if not check_alpha and not check_beta_nzf and not check_beta_zf and len(mc_i) > 0:
            # Check GAMMA
            check_gamma = True
            
            # Remove ANY matched motifs
            mc_gamma_i = mc_i[~(
                (mc_i[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score) |
                (mc_i[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score) |
                (mc_i[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
            )]

            # If GAMMA exists (any non-annotated motifs)
            if len(mc_gamma_i) > 0:
                # Sort by total contrib
                mc_gamma_i.sort(
                    by=SmithWatermanArgs.gamma_col,
                    ascending=False,
                    inplace=True,
                )
        
                # Select top k
                for k in range(SmithWatermanArgs.topk, 0, -1):
                    if len(mc_gamma_i) >= k:
                        # Final direct type, name, readable name
                        mask_match = mc_gamma_i.metadata.iloc[:k].index
                        mc_gamma_i.metadata.loc[mask_match, DirectTypeArgs.direct_type_col]                = DirectTypeArgs.direct_types[3]  # gamma
                        mc_gamma_i.metadata.loc[mask_match, DirectTypeArgs.tf_direct_col]                  = tf
                        mc_gamma_i.metadata.loc[mask_match, DirectTypeArgs.family_direct_col]              = tf_family
                        mc_gamma_i.metadata.loc[mask_match, MetadataCols.num_subunit_col]                = 1.0
                        mc_gamma_i.metadata.loc[mask_match, MetadataCols.num_subunit_direct_col]         = 1.0
                        mc_gamma_i.metadata.loc[mask_match, MetadataCols.composite_col]                  = False
                        mc_gamma_i.metadata.loc[mask_match, LabelCols.label_core_col]                 = tf
                        mc_gamma_i.metadata.loc[mask_match, LabelCols.direct_label_core_col]          = tf
                        mc_gamma_i.metadata.loc[mask_match, LabelCols.label_core_readable_col]        = tf_family
                        mc_gamma_i.metadata.loc[mask_match, LabelCols.direct_label_core_readable_col] = tf_family
                        mc_gamma_i.metadata.loc[mask_match, LabelCols.label_flank_col]                = None
                        break
        
                # Gamma: Store in list
                mc_direct_gamma_i = mc_gamma_i[mc_gamma_i[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]]
                logging.info(f"  Gamma: {len(mc_direct_gamma_i)} ")

                # Update original MotifCompendium metadata
                mc_metadata_updates.append(mc_direct_gamma_i.metadata.set_index(MetadataCols.unique_id_col))

                # Continue with unannotated motifs
                mc_i = mc_i[mc_i[DirectTypeArgs.direct_type_col] != DirectTypeArgs.direct_types[3]]
            
        # Update original MotifCompendium metadata
        mc_metadata_updates.append(mc_i.metadata.set_index(MetadataCols.unique_id_col))

    # Update: Metadata
    mc_metadata_updates_df = pd.concat(mc_metadata_updates)
    mc_metadata_updates_df = mc_metadata_updates_df[
        ~mc_metadata_updates_df.index.duplicated(keep='last')
    ]
    mc_qc.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_qc.metadata = mc_qc.metadata.combine_first(mc_metadata_updates_df)
    mc_qc.metadata.update(mc_metadata_updates_df)
    mc_qc.metadata.reset_index(inplace=True)
    del mc_metadata_updates_df, mc_metadata_updates

    # Concatenate all unique annotations
    mc_qc[ReferenceMatchCols.direct_ref_name_col] = mc_qc.metadata[
        [f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### (5) Motifs: Removed
    # Apply filters
    apply_filter_motifs(
        mc=mc_qc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Update metadata: Original MotifCompendium mc (up to mc_qc)
    mc.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc.metadata = mc.metadata.combine_first(mc_qc.metadata.set_index(MetadataCols.unique_id_col))
    mc.metadata.update(mc_qc.metadata.set_index(MetadataCols.unique_id_col))
    mc.metadata.reset_index(inplace=True)


    # Removed: mc = mc_qc + mc_removed = (mc_filtered + mc_removed) + mc_removed = mc_filtered + (mc_removed + mc_removed)
    mc_removed = mc[mc[MetadataCols.name_col].isin(
        set(mc_qc[mc_qc[FilterArgs.filter_flag_col]][MetadataCols.name_col])
        | set(mc_removed[MetadataCols.name_col])
    )]
    mc_removed[FilterArgs.filter_flag_col] = True
    mc_filtered = mc_qc[~mc_qc[MetadataCols.name_col].isin(mc_removed[MetadataCols.name_col])]

    # #### (4) Motifs: Indirect
    # Slice MotifCompendiums
    mc_direct = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]

    mc_direct_alpha_monomer = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]) & (mc_filtered[MetadataCols.num_subunit_direct_col] == MotifMatchArgs.min_composite_subunit)]
    mc_direct_alpha_multimer = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]) & (mc_filtered[MetadataCols.num_subunit_direct_col] > MotifMatchArgs.min_composite_subunit)]

    mc_direct_beta_nzf_monomer = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]) & (mc_filtered[MetadataCols.num_subunit_direct_col] == MotifMatchArgs.min_composite_subunit)]
    mc_direct_beta_nzf_multimer = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]) & (mc_filtered[MetadataCols.num_subunit_direct_col] > MotifMatchArgs.min_composite_subunit)]
    mc_direct_beta_zf = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2])]

    mc_direct_gamma = mc_filtered[(mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3])]

    mc_nondirect = mc_filtered[~mc_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]
    mc_nondirect[DirectTypeArgs.direct_type_col] = DirectTypeArgs.direct_types[4]  # Indirect: X

    # ## Clustering: Motifs
    logging.info("Clustering motifs, within TF...")
    # - Order:
    #     - **(1) Direct ALPHA**
    #     - **(2-1) Direct BETA (non-ZNF)**
    #     - **(2-2) Direct BETA (ZNF)**
    #     - **(3) Direct GAMMA**
    #     - **(4) Non-direct**
    # - Process:
    #     - Cluster within boundary: `ClusterArgs.direct_cluster_within`
    #     - Initial clustering
    #     - Recursive clustering
    #     - Force clustering


    # Group
    mc_direct_is = {
        DirectTypeArgs.direct_alpha_mononer_key: mc_direct_alpha_monomer,
        DirectTypeArgs.direct_alpha_multimer_key:mc_direct_alpha_multimer,
        DirectTypeArgs.direct_beta_nzf_monomer_key: mc_direct_beta_nzf_monomer,
        DirectTypeArgs.direct_beta_nzf_multimer_key: mc_direct_beta_nzf_multimer,
        DirectTypeArgs.direct_beta_zf_key: mc_direct_beta_zf,
        DirectTypeArgs.direct_gamma_key: mc_direct_gamma,
    }

    membership_direct_is = { # Initialize, with names
        DirectTypeArgs.direct_alpha_mononer_key: mc_direct_alpha_monomer[MetadataCols.name_col],
        DirectTypeArgs.direct_alpha_multimer_key: mc_direct_alpha_multimer[MetadataCols.name_col],
        DirectTypeArgs.direct_beta_nzf_monomer_key: mc_direct_beta_nzf_monomer[MetadataCols.name_col],
        DirectTypeArgs.direct_beta_nzf_multimer_key: mc_direct_beta_nzf_multimer[MetadataCols.name_col],
        DirectTypeArgs.direct_beta_zf_key: mc_direct_beta_zf[MetadataCols.name_col],
        DirectTypeArgs.direct_gamma_key: mc_direct_gamma[MetadataCols.name_col],
    }

    direct_type_keys = [
        DirectTypeArgs.direct_alpha_mononer_key,
        DirectTypeArgs.direct_alpha_multimer_key,
        DirectTypeArgs.direct_beta_nzf_monomer_key,
        DirectTypeArgs.direct_beta_nzf_multimer_key,
        DirectTypeArgs.direct_beta_zf_key,
        DirectTypeArgs.direct_gamma_key,
    ]


    # Check all MCs: Direct
    for direct_type_key in list(direct_type_keys):
        mc_direct_i = mc_direct_is[direct_type_key]
        membership_i = membership_direct_is[direct_type_key]
        logging.info(f"{direct_type_key}: {len(mc_direct_i)} motifs, {len(set(membership_i))} clusters")
        
        # Drop empty MCs
        if len(mc_direct_i) == 0:
            del mc_direct_is[direct_type_key]
            del membership_direct_is[direct_type_key]
            direct_type_keys.remove(direct_type_key)
            logging.info(f"  -> Dropped (empty)")

    # Check all MCs: Non-direct
    membership_nondirect = mc_nondirect[MetadataCols.name_col]
    logging.info(f"Non-direct: {len(mc_nondirect)} motifs, {len(set(membership_nondirect))} clusters")

    ## Cluster MotifCompendium: Main
    # Direct
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        # Cluster: Leiden CPM
        mc_direct_i.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within=ClusterArgs.direct_cluster_within,
            save_name=cluster_direct_cluster_col,
            largest_clusters_first=True,
            n_iterations=ClusterArgs.n_iterations,
            seeds=ClusterArgs.seeds,
        )
        # Reassign: K-means
        mc_direct_i.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.direct_cluster_within,
            save_name=cluster_direct_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=cluster_direct_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Membership
        membership_i = mc_direct_i[cluster_direct_cluster_col]
        if args.verbose:
            logging.info(f"Unique clusters ({direct_type_key}): {len(set(membership_i))}")
        
        # Reassign
        mc_direct_is[direct_type_key] = mc_direct_i
        membership_direct_is[direct_type_key] = membership_i

    # Non-direct
    # Cluster: Leiden CPM
    mc_nondirect.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        cluster_within=ClusterArgs.nondirect_cluster_within,
        save_name=cluster_nondirect_cluster_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )
    # Reassign: K-means
    mc_nondirect.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        cluster_within=ClusterArgs.nondirect_cluster_within,
        save_name=cluster_nondirect_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=cluster_nondirect_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_nondirect = mc_nondirect[cluster_nondirect_cluster_col]
    logging.info(f"Unique clusters (Non-direct): {len(set(membership_nondirect))}")


    ## Recursively cluster: Main
    # Direct
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        membership_i = membership_direct_is[direct_type_key]
        for i in range(ClusterArgs.cluster_max_iter_main):
            logging.info(f"Recursively clustering: {i}")
            # Cluster: CPM
            mc_direct_i.cluster(
                algorithm=ClusterArgs.cluster_algorithm_main,
                similarity_threshold=ClusterArgs.cluster_sim_threshold,
                cluster_within_on=(ClusterArgs.direct_cluster_within, cluster_direct_cluster_col),
                cluster_on_weight=ClusterArgs.average_weight_col,
                save_name=cluster_direct_cluster_col,
                largest_clusters_first=True,
                n_iterations=ClusterArgs.n_iterations,
                seeds=ClusterArgs.seeds,
            )
            # Reassign: K-means
            mc_direct_i.cluster(
                algorithm=ClusterArgs.cluster_algorithm_kmeans,
                cluster_within=ClusterArgs.direct_cluster_within,
                save_name=cluster_direct_cluster_col,
                largest_clusters_first=True,
                init_clustering_col=cluster_direct_cluster_col,
                weight_col=ClusterArgs.average_weight_col,
                assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
                n_iterations=ClusterArgs.n_iterations,
            )

            # Check convergence
            new_membership_i = mc_direct_i[cluster_direct_cluster_col]
            logging.info(f"  Unique clusters (Direct {direct_type_key}): {len(set(new_membership_i))}")
            if np.array_equal(membership_i, new_membership_i):
                # Reassign
                mc_direct_is[direct_type_key] = mc_direct_i
                membership_direct_is[direct_type_key] = new_membership_i
                logging.info(f"Final unique clusters ({direct_type_key}): {len(set(new_membership_i))}")
                break
            else:
                membership_i = new_membership_i

    # Non-direct
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_nondirect.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within_on=(ClusterArgs.nondirect_cluster_within, cluster_nondirect_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=cluster_nondirect_cluster_col,
            largest_clusters_first=True,
            n_iterations=ClusterArgs.n_iterations,
            seeds=ClusterArgs.seeds,
        )

        # Reassign: K-means
        mc_nondirect.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.nondirect_cluster_within,
            save_name=cluster_nondirect_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=cluster_nondirect_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_nondirect = mc_nondirect[cluster_nondirect_cluster_col]
        if args.verbose:
            logging.info(f"  Unique clusters (Non-direct): {len(set(new_membership_nondirect))}")
        if np.array_equal(membership_nondirect, new_membership_nondirect):
            logging.info(f"Final unique clusters (Non-direct): {len(set(new_membership_nondirect))}")
            break
        else:
            membership_nondirect = new_membership_nondirect


    ## Force-cluster
    # Direct
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        membership_i = membership_direct_is[direct_type_key]
        for i in range(ClusterArgs.cluster_max_iter_force):
            logging.info(f"Recursively clustering: {i}")
            # Cluster: DCC
            mc_direct_i.cluster(
                algorithm=ClusterArgs.cluster_algorithm_force,
                similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
                cluster_within_on=(ClusterArgs.direct_cluster_within, cluster_direct_cluster_col),
                cluster_on_weight=ClusterArgs.average_weight_col,
                save_name=cluster_direct_cluster_col,
                largest_clusters_first=True,
                density=ClusterArgs.cluster_force_density,
                seed=ClusterArgs.seeds[0],
            )
            # Reassign: K-means
            mc_direct_i.cluster(
                algorithm=ClusterArgs.cluster_algorithm_kmeans,
                cluster_within=ClusterArgs.direct_cluster_within,
                save_name=cluster_direct_cluster_col,
                largest_clusters_first=True,
                init_clustering_col=cluster_direct_cluster_col,
                weight_col=ClusterArgs.average_weight_col,
                assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
                n_iterations=ClusterArgs.n_iterations,
            )

            new_membership_i = mc_direct_i[cluster_direct_cluster_col]
            logging.info(f"  Unique clusters (Direct {direct_type_key}): {len(set(new_membership_i))}")
            if np.array_equal(membership_i, new_membership_i):
                # Reassign
                mc_direct_is[direct_type_key] = mc_direct_i
                membership_direct_is[direct_type_key] = new_membership_i
                logging.info(f"Final unique clusters ({direct_type_key}): {len(set(new_membership_i))}")
                break
            else:
                membership_i = new_membership_i

    # Non-direct
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_nondirect.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.nondirect_cluster_within, cluster_nondirect_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=cluster_nondirect_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )

        # Reassign: K-means
        mc_nondirect.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.nondirect_cluster_within,
            save_name=cluster_nondirect_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=cluster_nondirect_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_nondirect = mc_nondirect[cluster_nondirect_cluster_col]
        if args.verbose:
            logging.info(f"  Unique clusters (Non-direct): {len(set(new_membership_nondirect))}")
        if np.array_equal(membership_nondirect, new_membership_nondirect):
            logging.info(f"Final unique clusters (Non-direct): {len(set(new_membership_nondirect))}")
            break
        else:
            membership_nondirect = new_membership_nondirect

    # Final check all MCs
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        membership_i = membership_direct_is[direct_type_key]
        logging.info(f"{direct_type_key}: {len(mc_direct_i)} motifs, {len(set(membership_i))} clusters")
    logging.info(f"Non-direct: {len(mc_nondirect)} motifs, {len(set(new_membership_nondirect))} clusters")


    ## Re-number clustering
    max_cluster = -1
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        mc_direct_i[cluster_direct_cluster_col] = mc_direct_i[cluster_direct_cluster_col] + max_cluster + 1
        max_cluster = mc_direct_i[cluster_direct_cluster_col].max()
    mc_nondirect[cluster_nondirect_cluster_col] = mc_nondirect[cluster_nondirect_cluster_col] + max_cluster + 1

    # Update: Direct - Map cluster number to motifs (from mc_direct_i to mc_direct)
    name_cluster_dict = {}
    for direct_type_key in direct_type_keys:
        mc_direct_i = mc_direct_is[direct_type_key]
        if len(mc_direct_i) > 0:
            name_cluster_dict.update(dict(zip(mc_direct_i[MetadataCols.name_col], mc_direct_i[cluster_direct_cluster_col])))
    mc_direct[cluster_direct_cluster_col] = mc_direct[MetadataCols.name_col].map(name_cluster_dict)

    # Combine clustering numbering
    mc_direct[cluster_cluster_col] = mc_direct[cluster_direct_cluster_col]
    mc_nondirect[cluster_cluster_col] = mc_nondirect[cluster_nondirect_cluster_col]


    # Update metadata: Original MotifCompendium 
    for mc_i in [mc_direct, mc_nondirect]:
        for col in mc_i.columns():
            if col not in mc_filtered.columns():
                mc_filtered.metadata[col] = None

    # Update
    mc_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_filtered.metadata.update(mc_direct.metadata.set_index(MetadataCols.unique_id_col))
    mc_filtered.metadata.update(mc_nondirect.metadata.set_index(MetadataCols.unique_id_col))
    mc_filtered.metadata.reset_index(inplace=True)


    # # CLUSTERS
    # ## Create clusters, by averaging
    logging.info("Creating clusters, by averaging...")
    # - Clusters:
    #     - (1) Direct ALPHA
    #     - (2-1) Direct BETA (non-ZNF)
    #     - (2-2) Direct BETA (ZNF)
    #     - (3) Direct GAMMA
    #     - (4) Non-direct

    # Average MotifCompendium
    aggregations_cluster = []
    for i, aggregate in enumerate(ClusterArgs.aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_filtered.columns():
            aggregations_cluster.append(aggregate)

    mc_cluster = mc_filtered.cluster_averages(
        clustering=cluster_cluster_col,
        aggregations=aggregations_cluster,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Weighted ClusterArgs.aggregations
    weighted_aggregations_cluster = []
    for i, aggregate in enumerate(ClusterArgs.weighted_aggregations):
        (agg_col_old, _, agg_col_new) = aggregate
        if agg_col_old in mc_filtered.columns():
            weighted_aggregations_cluster.append(aggregate)

    for (agg_col_old, _, agg_col_new) in weighted_aggregations_cluster:
        aggregate_weighted_avg(
            mc=mc_filtered,
            mc_cluster=mc_cluster,
            cluster_col=cluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )

    # Merge ClusterArgs.aggregations
    merge_aggregations_cluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_cluster.columns():
            merge_aggregations_cluster.append(aggregate)
        if agg_col_new not in mc_cluster.columns():
            mc_cluster.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_cluster:
        mc_cluster.metadata[agg_col_new] = mc_cluster.metadata[[agg_col_old, agg_col_new]].apply(
            lambda row: ",".join(dict.fromkeys(
                part
                for agg in row
                if pd.notna(agg) and agg != ""
                for part in agg.split(",")
                if part not in ("", "None")
            )),
            axis=1
        )


    # ### Calculations: Clusters
    mc_cluster[MetadataCols.total_contrib_score_ratio_col] = mc_cluster[MetadataCols.total_contrib_score_col] / mc_cluster[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_cluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_cluster.get_standard_motif_stack())  # Positive / negative
    mc_cluster.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_cluster.sort(by=[DirectTypeArgs.direct_type_col, MetadataCols.composite_col], ascending=True, inplace=True)  # Sort by direct type, composite
    mc_cluster.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_cluster[MetadataCols.unique_id_col] = unique_id_initial + f".C1.{MetadataCols.atlas_type}.{head_type}." + mc_cluster.metadata.index.map('{:08d}'.format)


    ### Aggregate: Annotations
    mc_cluster_metadata_updates = []
    mc_cluster_images_updated = pd.DataFrame(index=mc_cluster[MetadataCols.unique_id_col])
    mc_cluster_images_existing = set(mc_cluster.images())

    for i, row in mc_cluster.metadata.iterrows():
        # MotifCompendium: Subset
        mc_cluster_i = mc_cluster[[i]]
        ref_name_constituents_i = row[ReferenceMatchCols.ref_name_constit_all_col].split(",") if pd.notna(row[ReferenceMatchCols.ref_name_constit_all_col]) else []
        direct_ref_name_constituents_i = row[ReferenceMatchCols.direct_ref_name_constit_all_col].split(",") if pd.notna(row[ReferenceMatchCols.direct_ref_name_constit_all_col]) else []
        tf_i = row[DirectTypeArgs.tf_direct_col]
        tf_family_i = row[DirectTypeArgs.family_direct_col]
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        if ref_name_constituents_i:
            mc_ref_qc_i = mc_ref_qc[mc_ref_qc[MetadataCols.name_col].isin(ref_name_constituents_i)]

            # Annotate with best reference constituents
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_i,
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
            if direct_type_i == DirectTypeArgs.direct_types[0]:  # ALPHA
                mc_ref_direct_i = mc_ref_direct[
                    (mc_ref_direct[MetadataCols.name_col].isin(direct_ref_name_constituents_i)) &
                    (mc_ref_direct[TFAnnotationCols.ref_tf_col] == tf_i)
                ]
            if direct_type_i == DirectTypeArgs.direct_types[1]:  # BETA (Non-ZNF)
                mc_ref_direct_i = mc_ref_direct[
                    (mc_ref_direct[MetadataCols.name_col].isin(direct_ref_name_constituents_i)) &
                    (mc_ref_direct[TFAnnotationCols.ref_family_col] == tf_family_i)
                ]
            
            # Reference: Subset (multimer: homo, hetero)
            mc_ref_i = mc_ref_direct[mc_ref_direct[MetadataCols.name_col].isin(ref_name_constituents_i)]  # Ref direct = Ref QC + Ref Invitro

            # Annotate with best reference constituents
            if len(mc_ref_direct_i) > 0:
                assign_direct_composite(
                    target_mc=mc_cluster_i,
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
        mc_cluster_metadata_updates.append(mc_cluster_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_cluster_i.images() if image_col not in mc_cluster_images_existing]
        mc_cluster_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_i.get_images(image_col),
                index=mc_cluster_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_images_updated = mc_cluster_images_updated.combine_first(mc_cluster_i_new_images)
        mc_cluster_images_updated.update(mc_cluster_i_new_images)

    # Apply updates: Metadata, Images
    mc_cluster_metadata_updates_df = pd.concat(mc_cluster_metadata_updates)
    mc_cluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster.metadata = mc_cluster.metadata.combine_first(mc_cluster_metadata_updates_df)
    mc_cluster.metadata.update(mc_cluster_metadata_updates_df)
    mc_cluster.metadata.reset_index(inplace=True)
    del mc_cluster_metadata_updates_df, mc_cluster_metadata_updates

    for image_col in mc_cluster_images_updated.columns:
        mc_cluster.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_images_updated[image_col]]
        )
    del mc_cluster_images_updated, mc_cluster_images_existing

    # #### Annotate (General)
    # Initialize
    mc_cluster.metadata = mc_cluster.metadata.assign(**{
        # DirectTypeArgs.direct_type_col:                None,
        # DirectTypeArgs.tf_direct_col:                  None,
        # DirectTypeArgs.family_direct_col:              None,
        MetadataCols.num_subunit_col:                1,
        MetadataCols.num_subunit_direct_col:         None,
        MetadataCols.composite_col:                  False,
        LabelCols.label_col:                      MetadataCols.name_unknown,
        LabelCols.direct_label_col:               None,
        LabelCols.label_core_col:                 None,
        LabelCols.direct_label_core_col:          None,
        LabelCols.label_core_readable_col:        None,
        LabelCols.direct_label_core_readable_col: None,
        LabelCols.label_flank_col:                None,
    }).copy()


    ## [General] Label clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_cluster,
        assign_from_mc=mc_ref_qc,
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
        mc_cluster[ref_readable_name_col] = mc_cluster[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_nondirect_cluster = ~mc_cluster[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])
    mask_monomer_cluster_nondirect = (
        mask_nondirect_cluster
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_cluster_nondirect = (
        mask_nondirect_cluster
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_cluster_nondirect = (
        mask_nondirect_cluster
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_cluster_nondirect = (
        mask_dimer_cluster_nondirect | mask_trimer_cluster_nondirect
    )
    mask_matched_cluster_nondirect = (
        mask_monomer_cluster_nondirect | mask_dimer_cluster_nondirect | mask_trimer_cluster_nondirect
    )

    # Dimer-style notation
    mc_cluster.metadata.loc[mask_matched_cluster_nondirect, LabelCols.label_core_col] = mc_cluster.metadata.loc[mask_matched_cluster_nondirect, ref_readable_name_cols[0]]
    mc_cluster.metadata.loc[mask_matched_cluster_nondirect, LabelCols.label_core_readable_col] = mc_cluster.metadata.loc[mask_matched_cluster_nondirect, ref_readable_name_cols[0]]
    mc_cluster.metadata.loc[mask_dimer_cluster_nondirect, LabelCols.label_flank_col] = mc_cluster.metadata.loc[mask_dimer_cluster_nondirect, ref_readable_name_cols[1]]
    mc_cluster.metadata.loc[mask_trimer_cluster_nondirect, LabelCols.label_flank_col] = mc_cluster.metadata.loc[mask_trimer_cluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_cluster.metadata.loc[mask_trimer_cluster_nondirect, ref_readable_name_cols[2]]
    mc_cluster.metadata.loc[mask_dimer_cluster_nondirect, MetadataCols.num_subunit_col] = 2
    mc_cluster.metadata.loc[mask_trimer_cluster_nondirect, MetadataCols.num_subunit_col] = 3
    mc_cluster.metadata[MetadataCols.composite_col] = mc_cluster.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit  # Composite: True, False


    # Concatenate all unique annotations
    mc_cluster[ReferenceMatchCols.ref_name_col] = mc_cluster.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### Calculate filters
    # Calculate filters
    mc_cluster[FilterArgs.filter_flag_col] = False

    utils_analysis.calculate_filters(
        mc=mc_cluster,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )

    # Calculate filters: No scaling, trim zeros
    utils_analysis.calculate_filters(
        mc=mc_cluster,
        metric_list=['posneg_inverted', 'truncated'],
        trim_importance=0,
    )

    # ### Annotate: Clusters (Direct)
    # #### Direct: Annotate (Alpha)
    ## [Direct: Alpha]
    logging.info("Cluster: Direct ALPHA")

    # Select: Direct Alpha
    mc_cluster_direct_alpha = mc_cluster[mc_cluster[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]]

    # Direct type, name, readable name (Default)
    mc_cluster_direct_alpha.metadata = mc_cluster_direct_alpha.metadata.assign(**{
        DirectTypeArgs.direct_type_col:                    DirectTypeArgs.direct_types[0],
        DirectTypeArgs.tf_direct_col:                      mc_cluster_direct_alpha[MetadataCols.tf_col],
        DirectTypeArgs.family_direct_col:                  mc_cluster_direct_alpha[TFAnnotationCols.family_col],
        MetadataCols.num_subunit_col:                    1,
        MetadataCols.num_subunit_direct_col:             1,
        MetadataCols.composite_col:                      False,
        LabelCols.label_core_col:                     mc_cluster_direct_alpha[MetadataCols.tf_col],
        LabelCols.direct_label_core_col:              mc_cluster_direct_alpha[MetadataCols.tf_col],
        LabelCols.label_core_readable_col:            mc_cluster_direct_alpha[TFAnnotationCols.family_col],
        LabelCols.direct_label_core_readable_col:     mc_cluster_direct_alpha[TFAnnotationCols.family_col],
        LabelCols.label_flank_col:                    None,
    }).copy()

    # mc_cluster_direct_alpha_is = []
    mc_cluster_direct_alpha_metadata_updates = []
    mc_cluster_direct_alpha_images_updated = pd.DataFrame(index=mc_cluster_direct_alpha[MetadataCols.unique_id_col])
    mc_cluster_direct_alpha_images_existing = set(mc_cluster_direct_alpha.images())

    for i, tf in enumerate(mc_cluster_direct_alpha[MetadataCols.tf_col].unique()):
        # Slice: MotifCompendium
        mc_cluster_direct_alpha_i = mc_cluster_direct_alpha[mc_cluster_direct_alpha[MetadataCols.tf_col] == tf]
        tf_family = mc_cluster_direct_alpha_i[TFAnnotationCols.family_col].unique()[0]
        logging.info(f"Processing {i+1}: {tf} in {tf_family} ({len(mc_cluster_direct_alpha_i)})")

        # Slice: Reference
        mc_ref_alpha_i = mc_ref_direct[mc_ref_direct[TFAnnotationCols.ref_tf_col] == tf]
        logging.info(f"  Alpha ref: {len(mc_ref_alpha_i)}")

        # Label motifs
        if len(mc_ref_alpha_i) > 0:
            ## (1) Annotate: Monomer, homo-multimer
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_direct_alpha_i,
                assign_from_mc=mc_ref_alpha_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )

            # (Repeat) Direct: Ensure at least one readable name
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_direct_alpha_i,
                assign_from_mc=mc_ref_alpha_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=0, 
                max_submotifs=1,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )

            # Annotate human-readable name: TF family
            for submotif_i in range(MotifMatchArgs.max_submotifs):
                mc_cluster_direct_alpha_i.metadata.loc[
                    mc_cluster_direct_alpha_i.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score, 
                    f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                ] = tf_family
            
            ## (2) Hetero-multimer
            # Re-label "homo-monomer" to monomer, hetero-multimer
            mc_cluster_direct_alpha_i_monomer = mc_cluster_direct_alpha_i[
                (mc_cluster_direct_alpha_i[f"{ReferenceMatchCols.direct_ref_col}_score0"] > MotifMatchArgs.min_composite_score) &
                (mc_cluster_direct_alpha_i[f"{ReferenceMatchCols.direct_ref_col}_score1"] < MotifMatchArgs.min_composite_score) &
                (mc_cluster_direct_alpha_i[f"{ReferenceMatchCols.direct_ref_col}_score2"] < MotifMatchArgs.min_composite_score)
            ]
            mc_cluster_direct_alpha_i_multimer = mc_cluster_direct_alpha_i[
                ~(mc_cluster_direct_alpha_i[MetadataCols.name_col].isin(mc_cluster_direct_alpha_i_monomer[MetadataCols.name_col]))
            ]
            if len(mc_cluster_direct_alpha_i_monomer) > 0:
                # Store: Metadata, Images
                mc_cluster_direct_alpha_i_metadata_updates = []
                mc_cluster_direct_alpha_i_images_updated = pd.DataFrame(index=mc_cluster_direct_alpha_i[MetadataCols.unique_id_col])

                ## Annotate: Hetero-multimer
                assign_direct_composite(
                    target_mc=mc_cluster_direct_alpha_i_monomer,
                    direct_reference_mc=mc_ref_alpha_i,
                    indirect_reference_mc=mc_ref_qc,
                    save_col_prefix=ReferenceMatchCols.direct_ref_col,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs,
                    label_unsigned=True,
                    save_images=True,
                    logo_trimming=0,
                )

                # Annotate human-readable name: HR name
                for submotif_i in range(1, MotifMatchArgs.max_submotifs):  # Keep first as TF
                    mc_cluster_direct_alpha_i_monomer.metadata.loc[
                        mc_cluster_direct_alpha_i_monomer.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score, 
                        f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                    ] = mc_cluster_direct_alpha_i_monomer[f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}'].map(mc_ref_qc_hr_dict)

                ## Update Cluster_i MotifCompendium: Metadata, Images
                # Metadata
                mc_cluster_direct_alpha_i_subunits = [
                    mc_cluster_direct_alpha_i_monomer, 
                    mc_cluster_direct_alpha_i_multimer
                ]
                mc_cluster_direct_alpha_i_subunit_metadatas = [
                    mc_cluster_direct_alpha_i_monomer.metadata,
                    mc_cluster_direct_alpha_i_multimer.metadata,
                ]
                for mc_cluster_direct_alpha_i_subunit_metadata in mc_cluster_direct_alpha_i_subunit_metadatas:
                    mc_cluster_direct_alpha_i_metadata_updates.append(mc_cluster_direct_alpha_i_subunit_metadata.set_index(MetadataCols.unique_id_col))
                # Images
                for mc_cluster_direct_alpha_i_subunit, mc_cluster_direct_alpha_i_subunit_metadata in zip(mc_cluster_direct_alpha_i_subunits, mc_cluster_direct_alpha_i_subunit_metadatas):
                    new_image_cols = [image_col for image_col in mc_cluster_direct_alpha_i_subunit.images() if image_col not in mc_cluster_direct_alpha_images_existing]
                    mc_cluster_direct_alpha_i_subunit_new_images = pd.DataFrame({
                        image_col: pd.Series(
                            mc_cluster_direct_alpha_i_subunit.get_images(image_name=image_col),
                            index=mc_cluster_direct_alpha_i_subunit_metadata[MetadataCols.unique_id_col],
                        )
                        for image_col in new_image_cols
                    })
                    mc_cluster_direct_alpha_i_images_updated = mc_cluster_direct_alpha_i_images_updated.combine_first(mc_cluster_direct_alpha_i_subunit_new_images)
                    mc_cluster_direct_alpha_i_images_updated.update(mc_cluster_direct_alpha_i_subunit_new_images)
                
                # Apply update: Metadata, Images
                mc_cluster_direct_alpha_i_metadata_updates_df = pd.concat(mc_cluster_direct_alpha_i_metadata_updates)
                mc_cluster_direct_alpha_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
                mc_cluster_direct_alpha_i.metadata = mc_cluster_direct_alpha_i.metadata.combine_first(mc_cluster_direct_alpha_i_metadata_updates_df)
                mc_cluster_direct_alpha_i.metadata.update(mc_cluster_direct_alpha_i_metadata_updates_df)
                mc_cluster_direct_alpha_i.metadata.reset_index(inplace=True)
                del mc_cluster_direct_alpha_i_metadata_updates_df, mc_cluster_direct_alpha_i_metadata_updates

                for image_col in mc_cluster_direct_alpha_i_images_updated.columns:
                    mc_cluster_direct_alpha_i.set_images(
                        image_name=image_col,
                        images=[x if pd.notna(x) else None
                            for x in mc_cluster_direct_alpha_i_images_updated[image_col]]
                    )
        
            # Direct: Composite, Core, Core (Direct), Composite subunits
            mask_subunit_df = pd.DataFrame({submotif_i: 
                (mc_cluster_direct_alpha_i.metadata[f"{ReferenceMatchCols.direct_ref_col}_score{submotif_i}"] > (
                    MotifMatchArgs.min_single_score if submotif_i == 0 else MotifMatchArgs.min_composite_score)
                ) for submotif_i in range(MotifMatchArgs.max_submotifs)
            }, index=mc_cluster_direct_alpha_i.metadata.index)
            # At least 1 (the core): the cluster is direct by its motifs, even if its average motif no longer
            # matches the TF above min_single_score (no subunit counted: 0)
            num_subunit = mask_subunit_df.sum(axis=1).clip(lower=1)
            direct_ref_readable_name_subunit_df = pd.concat([
                mc_cluster_direct_alpha_i.metadata[f"{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}"]
                    for submotif_i in range(1, MotifMatchArgs.max_submotifs)
                ], axis=1
            ).where(mask_subunit_df.iloc[:, 1:].fillna(False).astype(bool))
            
            mc_cluster_direct_alpha_i.metadata = mc_cluster_direct_alpha_i.metadata.assign(**{
                MetadataCols.num_subunit_col:                    num_subunit,
                MetadataCols.num_subunit_direct_col:             num_subunit,
                MetadataCols.composite_col:                      num_subunit > MotifMatchArgs.min_composite_subunit,
                LabelCols.label_core_col:                     tf,
                LabelCols.direct_label_core_col:              tf,
                LabelCols.label_core_readable_col:            tf_family,
                LabelCols.direct_label_core_readable_col:     tf_family,
                LabelCols.label_flank_col:                    (
                    direct_ref_readable_name_subunit_df
                    .fillna("")
                    .astype(str)
                    .apply(lambda row: "-".join(x for x in row if x), axis=1)
                ),
            }).copy()

            ## Update Cluster MotifCompendium: Metadata, Images
            # Metadata
            mc_cluster_direct_alpha_metadata_updates.append(mc_cluster_direct_alpha_i.metadata.set_index(MetadataCols.unique_id_col))
            # Images
            new_image_cols = [image_col for image_col in mc_cluster_direct_alpha_i.images() if image_col not in mc_cluster_direct_alpha_images_existing]
            mc_cluster_direct_alpha_i_new_images = pd.DataFrame({
                image_col: pd.Series(
                    mc_cluster_direct_alpha_i.get_images(image_name=image_col),
                    index=mc_cluster_direct_alpha_i.metadata[MetadataCols.unique_id_col],
                )
                for image_col in new_image_cols
            })
            mc_cluster_direct_alpha_images_updated = mc_cluster_direct_alpha_images_updated.combine_first(mc_cluster_direct_alpha_i_new_images)
            mc_cluster_direct_alpha_images_updated.update(mc_cluster_direct_alpha_i_new_images)

        # else:
        #     logging.info(f"  No Alpha ref found")
        #     for iter in range(MotifMatchArgs.max_submotifs):
        #         mc_cluster_direct_alpha_i.metadata[
        #             [f"{ReferenceMatchCols.direct_ref_col}_score{iter}", 
        #              f"{ReferenceMatchCols.direct_ref_col}_name{iter}", 
        #              f"{ReferenceMatchCols.direct_ref_col}_target{iter}",
        #             ]] = None
        #         mc_cluster_direct_alpha_i.add_logos(
        #             motifs=np.zeros_like(mc_cluster_direct_alpha_i.get_standard_motif_stack()),
        #             image_name=f"{ReferenceMatchCols.direct_ref_col}_logo{iter}",
        #             trim=False,
        #         )
        #
        # # Store in list
        # mc_cluster_direct_alpha_is.append(mc_cluster_direct_alpha_i)

    # # Recombine
    # mc_cluster_direct_alpha = MotifCompendium.combine(mc_cluster_direct_alpha_is, safe=safe)
    # del mc_cluster_direct_alpha_is

    # Apply update: Metadata, Images
    mc_cluster_direct_alpha_metadata_updates_df = pd.concat(mc_cluster_direct_alpha_metadata_updates)
    mc_cluster_direct_alpha.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_direct_alpha.metadata = mc_cluster_direct_alpha.metadata.combine_first(mc_cluster_direct_alpha_metadata_updates_df)
    mc_cluster_direct_alpha.metadata.update(mc_cluster_direct_alpha_metadata_updates_df)
    mc_cluster_direct_alpha.metadata.reset_index(inplace=True)
    del mc_cluster_direct_alpha_metadata_updates_df, mc_cluster_direct_alpha_metadata_updates

    for image_col in mc_cluster_direct_alpha_images_updated.columns:
        mc_cluster_direct_alpha.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_direct_alpha_images_updated[image_col]]
        )
    del mc_cluster_direct_alpha_images_updated, mc_cluster_direct_alpha_images_existing


    # Concatenate all unique direct annotations
    mc_cluster_direct_alpha[ReferenceMatchCols.direct_ref_name_col] = mc_cluster_direct_alpha.metadata[
        [f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

    ## Direct: Beta (Non-ZNF)
    logging.info("Cluster: Direct BETA (non-ZNF)")

    # Select: Direct Beta (Non-ZNF)
    mc_cluster_direct_beta_nzf = mc_cluster[mc_cluster[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]]

    # Direct type, name, readable name (Default)
    mc_cluster_direct_beta_nzf.metadata = mc_cluster_direct_beta_nzf.metadata.assign(**{
        DirectTypeArgs.direct_type_col:                    DirectTypeArgs.direct_types[1],
        DirectTypeArgs.tf_direct_col:                      mc_cluster_direct_beta_nzf[MetadataCols.tf_col],
        DirectTypeArgs.family_direct_col:                  mc_cluster_direct_beta_nzf[TFAnnotationCols.family_col],
        MetadataCols.num_subunit_col:                    1,
        MetadataCols.num_subunit_direct_col:             1,
        MetadataCols.composite_col:                      False,
        LabelCols.label_core_col:                     mc_cluster_direct_beta_nzf[MetadataCols.tf_col],
        LabelCols.label_core_readable_col:            mc_cluster_direct_beta_nzf[TFAnnotationCols.family_col],
        LabelCols.direct_label_core_col:              mc_cluster_direct_beta_nzf[MetadataCols.tf_col],
        LabelCols.direct_label_core_readable_col:     mc_cluster_direct_beta_nzf[TFAnnotationCols.family_col],
        LabelCols.label_flank_col:                    None,
    }).copy()

    # mc_cluster_direct_beta_nzf_is = []
    mc_cluster_direct_beta_nzf_metadata_updates = []
    mc_cluster_direct_beta_nzf_images_updated = pd.DataFrame(index=mc_cluster_direct_beta_nzf[MetadataCols.unique_id_col])
    mc_cluster_direct_beta_nzf_images_existing = set(mc_cluster_direct_beta_nzf.images())

    for i, tf in enumerate(mc_cluster_direct_beta_nzf[MetadataCols.tf_col].unique()):
        # Slice: MotifCompendium
        mc_cluster_direct_beta_nzf_i = mc_cluster_direct_beta_nzf[mc_cluster_direct_beta_nzf[MetadataCols.tf_col] == tf]
        tf_family = mc_cluster_direct_beta_nzf_i[TFAnnotationCols.family_col].unique()[0]
        logging.info(f"Processing {i+1}: {tf} in {tf_family} ({len(mc_cluster_direct_beta_nzf_i)})")
        
        # Slice: Reference
        mc_ref_beta_nzf_i = mc_ref_qc[mc_ref_qc[TFAnnotationCols.family_col] == tf_family]
        logging.info(f"  Beta ref: {len(mc_ref_beta_nzf_i)}")

        # Label motifs
        if len(mc_ref_beta_nzf_i) > 0:
            ## (1) Annotate: Monomer, homo-multimer
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_direct_beta_nzf_i,
                assign_from_mc=mc_ref_beta_nzf_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=MotifMatchArgs.min_composite_score, # Direct: Ensure readable name
                max_submotifs=MotifMatchArgs.max_submotifs,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )

            # (Repeat) Direct: Ensure readable name
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_direct_beta_nzf_i,
                assign_from_mc=mc_ref_beta_nzf_i,
                from_label_col=MetadataCols.name_col,
                save_col_prefix=ReferenceMatchCols.direct_ref_col,
                min_score=0,
                max_submotifs=1,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )
            
            # Annotate human-readable name: TF
            for submotif_i in range(MotifMatchArgs.max_submotifs):
                mc_cluster_direct_beta_nzf_i.metadata.loc[
                    mc_cluster_direct_beta_nzf_i.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score,
                    f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                ] = tf_family
            
            ## (2) Hetero-multimer
            # Re-label "homo-monomer" to monomer, hetero-multimer
            mc_cluster_direct_beta_nzf_i_monomer = mc_cluster_direct_beta_nzf_i[
                (mc_cluster_direct_beta_nzf_i[f"{ReferenceMatchCols.direct_ref_col}_score0"] > MotifMatchArgs.min_composite_score) &
                (mc_cluster_direct_beta_nzf_i[f"{ReferenceMatchCols.direct_ref_col}_score1"] < MotifMatchArgs.min_composite_score) &
                (mc_cluster_direct_beta_nzf_i[f"{ReferenceMatchCols.direct_ref_col}_score2"] < MotifMatchArgs.min_composite_score)
            ]
            mc_cluster_direct_beta_nzf_i_multimer = mc_cluster_direct_beta_nzf_i[
                ~(mc_cluster_direct_beta_nzf_i[MetadataCols.name_col].isin(mc_cluster_direct_beta_nzf_i_monomer[MetadataCols.name_col]))
            ]
            if len(mc_cluster_direct_beta_nzf_i_monomer) > 0:
                # Store: Metadata, Images
                mc_cluster_direct_beta_nzf_i_metadata_updates = []
                mc_cluster_direct_beta_nzf_i_images_updated = pd.DataFrame(index=mc_cluster_direct_beta_nzf_i[MetadataCols.unique_id_col])

                ## Annotate: Hetero-multimer
                assign_direct_composite(
                    target_mc=mc_cluster_direct_beta_nzf_i_monomer,
                    direct_reference_mc=mc_ref_beta_nzf_i,
                    indirect_reference_mc=mc_ref_qc,
                    save_col_prefix=ReferenceMatchCols.direct_ref_col,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs,
                    label_unsigned=True,
                    save_images=True,
                    logo_trimming=0,
                )
                # Annotate human-readable name: HR name
                for submotif_i in range(1, MotifMatchArgs.max_submotifs):  # Keep first as TF family
                    mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[
                        mc_cluster_direct_beta_nzf_i_monomer.metadata[f'{ReferenceMatchCols.direct_ref_col}_score{submotif_i}'] > MotifMatchArgs.min_composite_score, 
                        f'{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}'
                    ] = mc_cluster_direct_beta_nzf_i_monomer[f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}'].map(mc_ref_qc_hr_dict)

                ## Update Cluster_i MotifCompendium: Metadata, Images
                # Metadata
                mc_cluster_direct_beta_nzf_i_subunits = [
                    mc_cluster_direct_beta_nzf_i_monomer, 
                    mc_cluster_direct_beta_nzf_i_multimer
                ]
                mc_cluster_direct_beta_nzf_i_subunit_metadatas = [
                    mc_cluster_direct_beta_nzf_i_monomer.metadata,
                    mc_cluster_direct_beta_nzf_i_multimer.metadata,
                ]
                for mc_cluster_direct_beta_nzf_i_subunit_metadata in mc_cluster_direct_beta_nzf_i_subunit_metadatas:
                    mc_cluster_direct_beta_nzf_i_metadata_updates.append(mc_cluster_direct_beta_nzf_i_subunit_metadata.set_index(MetadataCols.unique_id_col))
                # Images
                for mc_cluster_direct_beta_nzf_i_subunit, mc_cluster_direct_beta_nzf_i_subunit_metadata in zip(mc_cluster_direct_beta_nzf_i_subunits, mc_cluster_direct_beta_nzf_i_subunit_metadatas):
                    new_image_cols = [image_col for image_col in mc_cluster_direct_beta_nzf_i_subunit.images() if image_col not in mc_cluster_direct_beta_nzf_images_existing]
                    mc_cluster_direct_beta_nzf_i_subunit_new_images = pd.DataFrame({
                        image_col: pd.Series(
                            mc_cluster_direct_beta_nzf_i_subunit.get_images(image_name=image_col),
                            index=mc_cluster_direct_beta_nzf_i_subunit_metadata[MetadataCols.unique_id_col],
                        )
                        for image_col in new_image_cols
                    })
                    mc_cluster_direct_beta_nzf_i_images_updated = mc_cluster_direct_beta_nzf_i_images_updated.combine_first(mc_cluster_direct_beta_nzf_i_subunit_new_images)
                    mc_cluster_direct_beta_nzf_i_images_updated.update(mc_cluster_direct_beta_nzf_i_subunit_new_images)
                
                # Apply update: Metadata, Images
                mc_cluster_direct_beta_nzf_i_metadata_updates_df = pd.concat(mc_cluster_direct_beta_nzf_i_metadata_updates)
                mc_cluster_direct_beta_nzf_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
                mc_cluster_direct_beta_nzf_i.metadata = mc_cluster_direct_beta_nzf_i.metadata.combine_first(mc_cluster_direct_beta_nzf_i_metadata_updates_df)
                mc_cluster_direct_beta_nzf_i.metadata.update(mc_cluster_direct_beta_nzf_i_metadata_updates_df)
                mc_cluster_direct_beta_nzf_i.metadata.reset_index(inplace=True)
                del mc_cluster_direct_beta_nzf_i_metadata_updates_df, mc_cluster_direct_beta_nzf_i_metadata_updates

                for image_col in mc_cluster_direct_beta_nzf_i_images_updated.columns:
                    mc_cluster_direct_beta_nzf_i.set_images(
                        image_name=image_col,
                        images=[x if pd.notna(x) else None
                            for x in mc_cluster_direct_beta_nzf_i_images_updated[image_col]]
                    )
        
            # Direct: Composite, Core, Core (Direct), Composite subunits
            mask_subunit_df = pd.DataFrame({submotif_i: 
                (mc_cluster_direct_beta_nzf_i.metadata[f"{ReferenceMatchCols.direct_ref_col}_score{submotif_i}"] > (
                    MotifMatchArgs.min_single_score if submotif_i == 0 else MotifMatchArgs.min_composite_score)
                ) for submotif_i in range(MotifMatchArgs.max_submotifs)
            }, index=mc_cluster_direct_beta_nzf_i.metadata.index)
            # At least 1 (the core): the cluster is direct by its motifs, even if its average motif no longer
            # matches the TF above min_single_score (no subunit counted: 0)
            num_subunit = mask_subunit_df.sum(axis=1).clip(lower=1)
            direct_ref_readable_name_subunit_df = pd.concat([
                mc_cluster_direct_beta_nzf_i.metadata[f"{ReferenceMatchCols.direct_ref_readable_col}_name{submotif_i}"]
                    for submotif_i in range(1, MotifMatchArgs.max_submotifs)
                ], axis=1
            ).where(mask_subunit_df.iloc[:, 1:].fillna(False).astype(bool))
            
            mc_cluster_direct_beta_nzf_i.metadata = mc_cluster_direct_beta_nzf_i.metadata.assign(**{
                MetadataCols.num_subunit_col:                    num_subunit,
                MetadataCols.num_subunit_direct_col:             num_subunit,
                MetadataCols.composite_col:                      num_subunit > MotifMatchArgs.min_composite_subunit,
                LabelCols.label_core_col:                     tf,
                LabelCols.direct_label_core_col:              tf,
                LabelCols.label_core_readable_col:            tf_family,
                LabelCols.direct_label_core_readable_col:     tf_family,
                LabelCols.label_flank_col:                    (
                    direct_ref_readable_name_subunit_df
                    .fillna("")
                    .astype(str)
                    .apply(lambda row: "-".join(x for x in row if x), axis=1)
                ),
            }).copy()

            ## Update Cluster MotifCompendium: Metadata, Images
            # Metadata
            mc_cluster_direct_beta_nzf_metadata_updates.append(mc_cluster_direct_beta_nzf_i.metadata.set_index(MetadataCols.unique_id_col))
            # Images
            new_image_cols = [image_col for image_col in mc_cluster_direct_beta_nzf_i.images() if image_col not in mc_cluster_direct_beta_nzf_images_existing]
            mc_cluster_direct_beta_nzf_i_new_images = pd.DataFrame({
                image_col: pd.Series(
                    mc_cluster_direct_beta_nzf_i.get_images(image_name=image_col),
                    index=mc_cluster_direct_beta_nzf_i.metadata[MetadataCols.unique_id_col],
                )
                for image_col in new_image_cols
            })
            mc_cluster_direct_beta_nzf_images_updated = mc_cluster_direct_beta_nzf_images_updated.combine_first(mc_cluster_direct_beta_nzf_i_new_images)
            mc_cluster_direct_beta_nzf_images_updated.update(mc_cluster_direct_beta_nzf_i_new_images)

    # Apply update: Metadata, Images
    mc_cluster_direct_beta_nzf_metadata_updates_df = pd.concat(mc_cluster_direct_beta_nzf_metadata_updates)
    mc_cluster_direct_beta_nzf.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_direct_beta_nzf.metadata = mc_cluster_direct_beta_nzf.metadata.combine_first(mc_cluster_direct_beta_nzf_metadata_updates_df)
    mc_cluster_direct_beta_nzf.metadata.update(mc_cluster_direct_beta_nzf_metadata_updates_df)
    mc_cluster_direct_beta_nzf.metadata.reset_index(inplace=True)
    del mc_cluster_direct_beta_nzf_metadata_updates_df, mc_cluster_direct_beta_nzf_metadata_updates

    for image_col in mc_cluster_direct_beta_nzf_images_updated.columns:
        mc_cluster_direct_beta_nzf.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_direct_beta_nzf_images_updated[image_col]]
        )
    del mc_cluster_direct_beta_nzf_images_updated, mc_cluster_direct_beta_nzf_images_existing

    # Concatenate all unique direct annotations
    mc_cluster_direct_beta_nzf[ReferenceMatchCols.direct_ref_name_col] = mc_cluster_direct_beta_nzf.metadata[
        [f'{ReferenceMatchCols.direct_ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    ## Direct: Beta (ZNF)
    logging.info("Cluster: Direct BETA (ZNF)")

    # Select: Direct Beta (ZNF)
    mc_cluster_direct_beta_zf = mc_cluster[mc_cluster[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]]

    # Direct type, name, readable name (Default)
    mc_cluster_direct_beta_zf.metadata = mc_cluster_direct_beta_zf.metadata.assign(**{
        DirectTypeArgs.direct_type_col:        DirectTypeArgs.direct_types[2],
        DirectTypeArgs.tf_direct_col:          mc_cluster_direct_beta_zf[MetadataCols.tf_col],
        DirectTypeArgs.family_direct_col:      mc_cluster_direct_beta_zf[TFAnnotationCols.family_col],
        MetadataCols.num_subunit_col:        1,
        MetadataCols.num_subunit_direct_col: 1,
        MetadataCols.composite_col:          False,
        LabelCols.label_core_col:         mc_cluster_direct_beta_zf[MetadataCols.tf_col],
        LabelCols.direct_label_core_col:  mc_cluster_direct_beta_zf[MetadataCols.tf_col],
        LabelCols.label_core_readable_col: mc_cluster_direct_beta_zf[TFAnnotationCols.family_col],
        LabelCols.direct_label_core_readable_col: mc_cluster_direct_beta_zf[TFAnnotationCols.family_col],
        LabelCols.label_flank_col:        None,
    }).copy()

    ### MODIFIED SMITH-WATERMAN ALIGNMENT
    mc_b1h_select = mc_b1h[mc_b1h[TFAnnotationCols.ref_tf_col].isin(mc_cluster_direct_beta_zf[MetadataCols.tf_col].unique())]

    ## Run Modified Smith-Waterman
    aligned_motifs = np.zeros((
        mc_cluster_direct_beta_zf.motifs.shape[0], 
        mc_cluster_direct_beta_zf.motifs.shape[1] + 2 * (SmithWatermanArgs.finger_length - 1), 
        mc_cluster_direct_beta_zf.motifs.shape[2]
    ))

    for i, row_b1h in mc_b1h_select.metadata.iterrows():
        tf_i = row_b1h[TFAnnotationCols.ref_tf_col]
        logging.info(f"Processing: {tf_i}")
        
        ## B1H
        mc_b1h_i = mc_b1h_select[[i]]
        name_b1h = row_b1h[MetadataCols.name_col]
        b1h_ppm_ic_i = utils_motif.trim_motif(mc_b1h_i.motifs[0], 0) # (L, 4), trim zeros

        ## CWMs
        mc_znf_i = mc_cluster_direct_beta_zf[mc_cluster_direct_beta_zf.metadata[MetadataCols.tf_col] == tf_i]
        cwm_i = mc_znf_i.motifs
        
        if len(mc_znf_i) > 0:
            ## Run dynamic programming: Per CWM
            for j, row_cwm in mc_znf_i.metadata.iterrows():
                name_cwm = row_cwm[MetadataCols.name_col]
                cwm_ij = utils_motif.trim_motif(cwm_i[j] , 0) # (L, 4), trim zeros
                logging.info(f"  CWM: {name_cwm}")
                
                (msw_score, sim_score, msw_aligned_b1h, msw_orient, msw_align_b1h, msw_align_cwm, msw_scale_f, msw_action, _, _) = utils_composite.run_modified_smith_waterman_alignment(
                    target_motif=cwm_ij,
                    reference_motif=b1h_ppm_ic_i,
                    sim_threshold=SmithWatermanArgs.msw_sim_threshold,
                    overlap_penalty_f1=SmithWatermanArgs.overlap_penalty_f1,
                    overlap_penalty_f2=SmithWatermanArgs.overlap_penalty_f2,
                    skip_penalty_f1=SmithWatermanArgs.skip_penalty_f1, 
                    skip_penalty_f2=SmithWatermanArgs.skip_penalty_f2,
                    skip_penalty_l1=SmithWatermanArgs.skip_penalty_l1, 
                    skip_penalty_l2=SmithWatermanArgs.skip_penalty_l2,
                    finger_length=SmithWatermanArgs.finger_length,
                )

                # Aligned B1H motif (collapsed)
                msw_aligned_b1h_collapse = msw_aligned_b1h.sum(axis=0)
                msw_aligned_b1h_collapse = np.pad(msw_aligned_b1h_collapse,
                    ((0, aligned_motifs.shape[1] - msw_aligned_b1h_collapse.shape[0]), (0, 0)),
                    mode='constant')
                cwm_ij = np.pad(cwm_ij,
                    ((0, aligned_motifs.shape[1] - cwm_ij.shape[0]), (0, 0)),
                    mode='constant')

                # Record, in original MotifCompendium
                org_idx = (mc_cluster_direct_beta_zf.metadata[MetadataCols.name_col] == name_cwm).idxmax()
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, SmithWatermanArgs.msw_score_col] = msw_score
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, SmithWatermanArgs.msw_sim_col] = sim_score
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, SmithWatermanArgs.msw_align_col] = "\n".join("".join(str(msw_align_pos)) for msw_align_pos in [msw_align_b1h, msw_align_cwm])
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, SmithWatermanArgs.msw_action_col] = ";".join(msw_action)
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, SmithWatermanArgs.msw_name_col] = name_b1h
                mc_cluster_direct_beta_zf.metadata.loc[org_idx, ReferenceMatchCols.znf_ref_name_col] = name_b1h

                # Save aligned motif
                aligned_motifs[org_idx] = msw_aligned_b1h_collapse

    # Visualize aligned motifs
    mc_cluster_direct_beta_zf.add_logos(
        motifs=aligned_motifs,
        image_name=SmithWatermanArgs.msw_logo_col,
        trim=0,
    )

    ## Direct GAMMA
    # Final direct type, name, readable name
    mc_cluster_direct_gamma = mc_cluster[mc_cluster[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]]

    # Direct type, name, readable name (Default)
    mc_cluster_direct_gamma.metadata = mc_cluster_direct_gamma.metadata.assign(**{
        DirectTypeArgs.direct_type_col:                    DirectTypeArgs.direct_types[3],  # C
        DirectTypeArgs.tf_direct_col:                      mc_cluster_direct_gamma[MetadataCols.tf_col],
        DirectTypeArgs.family_direct_col:                  mc_cluster_direct_gamma[TFAnnotationCols.family_col],
        MetadataCols.num_subunit_col:                    1,
        MetadataCols.num_subunit_direct_col:             1,
        MetadataCols.composite_col:                      False,
        LabelCols.label_core_col:                     mc_cluster_direct_gamma[MetadataCols.tf_col],
        LabelCols.direct_label_core_col:              mc_cluster_direct_gamma[MetadataCols.tf_col],
        LabelCols.label_core_readable_col:            mc_cluster_direct_gamma[TFAnnotationCols.family_col],
        LabelCols.direct_label_core_readable_col:     mc_cluster_direct_gamma[TFAnnotationCols.family_col],
        LabelCols.label_flank_col:                    None,
    }).copy()

    # #### Indirect: Annotate
    ### Non-direct
    # Select: Non-direct
    mc_cluster_nondirect = mc_cluster[~mc_cluster[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]  # Not: A, B-NZ, B-Z, C
    mc_cluster_nondirect.metadata = mc_cluster_nondirect.metadata.assign(**{
        # DirectTypeArgs.direct_type_col:                  Can be X, U, removed
        DirectTypeArgs.tf_direct_col:                      None,
        DirectTypeArgs.family_direct_col:                  None,
        MetadataCols.num_subunit_col:                    1,
        MetadataCols.num_subunit_direct_col:             1,
        MetadataCols.composite_col:                      False,
        LabelCols.direct_label_core_col:              mc_cluster_nondirect[MetadataCols.tf_col],
        LabelCols.direct_label_core_readable_col:     mc_cluster_nondirect[TFAnnotationCols.family_col],
    }).copy()

    ## Indirect
    # Apply filters
    apply_filter_clusters(
        mc=mc_cluster_nondirect,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Remove flagged clusters
    mc_cluster_removed = mc_cluster_nondirect[mc_cluster_nondirect[FilterArgs.filter_flag_col]]
    mc_cluster_nondirect_filtered = mc_cluster_nondirect[~mc_cluster_nondirect[FilterArgs.filter_flag_col]]  # Remove flagged clusters from cluster_nondirect

    # Annotate
    mc_cluster_nondirect_filtered[DirectTypeArgs.direct_type_col] = DirectTypeArgs.direct_types[4]  # Indirect: X (including unvalidated)

    # #### Cluster (Direct): Direct names
    # Group
    mc_cluster_direct_is = {
        DirectTypeArgs.direct_alpha_key: mc_cluster_direct_alpha,
        DirectTypeArgs.direct_beta_nzf_key: mc_cluster_direct_beta_nzf,
        DirectTypeArgs.direct_beta_zf_key: mc_cluster_direct_beta_zf,
        DirectTypeArgs.direct_gamma_key: mc_cluster_direct_gamma,
    }

    direct_type_keys = [
        DirectTypeArgs.direct_alpha_key,
        DirectTypeArgs.direct_beta_nzf_key,
        DirectTypeArgs.direct_beta_zf_key,
        DirectTypeArgs.direct_gamma_key,
    ]

    ## Direct cluster label: TF-{counts/profile}-{TF-ChIP}_{A/B/C}{direct_score}_{tf_core}::{tf_composite}
    ref_score_cols = [f"{ReferenceMatchCols.direct_ref_col}_score{iter}" for iter in range(MotifMatchArgs.max_submotifs)]
    composite_score_cols = [f"{ReferenceMatchCols.direct_ref_col}_score{iter}" for iter in range(1, MotifMatchArgs.max_submotifs)]

    for i, direct_type_key in enumerate(direct_type_keys):
        mc_cluster_direct_i = mc_cluster_direct_is[direct_type_key]
        
        # Boolean conditions    
        mask_composite = mc_cluster_direct_i[MetadataCols.composite_col]
        mask_has_composite = mc_cluster_direct_i[LabelCols.label_flank_col].notna()
        if direct_type_key in [DirectTypeArgs.direct_alpha_key, DirectTypeArgs.direct_beta_nzf_key]:  # Alpha, Beta non-ZNF:
            score_col = (mc_cluster_direct_i[ref_score_cols].fillna(0).max(axis=1) * 100).astype(int).astype(str)
        elif direct_type_key == DirectTypeArgs.direct_beta_zf_key:  # Beta ZNF:
            score_col = mc_cluster_direct_i[SmithWatermanArgs.msw_sim_col].fillna(0).apply(lambda x: int(x * 100)).astype(str)
        elif direct_type_key == DirectTypeArgs.direct_gamma_key:  # Gamma:
            score_col = mc_cluster_direct_i[SmithWatermanArgs.gamma_col].fillna(0).apply(lambda x: int(x * 100)).astype(str)
        
        # Assign readable label:
        mc_cluster_direct_i[LabelCols.direct_label_col] = (
            mc_cluster_direct_i[LabelCols.direct_label_core_col]
            + np.where(
                mask_composite & mask_has_composite,
                "::" + mc_cluster_direct_i[LabelCols.label_flank_col],
                ""
            )
        )
        mc_cluster_direct_i[MetadataCols.prename_col] = (
            label_initial
            + "-"
            + mc_cluster_direct_i[DirectTypeArgs.tf_direct_col]
            + "_" + mc_cluster_direct_i[DirectTypeArgs.direct_type_col]
            + score_col
            + "_"
            + mc_cluster_direct_i[LabelCols.direct_label_col]
        )
        mc_cluster_direct_i[LabelCols.label_col] = mc_cluster_direct_i[LabelCols.direct_label_col]
        mc_cluster_direct_i[MetadataCols.simple_name_col] = mc_cluster_direct_i[LabelCols.label_col]
        if direct_type_key == DirectTypeArgs.direct_alpha_key:
            label_conf_i = LabelCols.label_confs[0]
        elif direct_type_key in (DirectTypeArgs.direct_beta_nzf_key, DirectTypeArgs.direct_beta_zf_key):
            label_conf_i = LabelCols.label_confs[1]
        elif direct_type_key == DirectTypeArgs.direct_gamma_key:
            label_conf_i = LabelCols.label_confs[2]
        mc_cluster_direct_i[LabelCols.label_conf_col] = label_conf_i
        
        # Reassign
        mc_cluster_direct_is[direct_type_key] = mc_cluster_direct_i

    #### Direct: ALL
    mc_cluster_direct = mc_cluster[mc_cluster[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]

    mc_cluster_direct.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_direct_images_updated = pd.DataFrame(index=mc_cluster_direct.metadata.index)
    mc_cluster_direct_images_existing = set(mc_cluster_direct.images())

    # Build updates: Metadata, Images
    for direct_type_key in direct_type_keys:
        # Direct: Type
        mc_cluster_direct_i = mc_cluster_direct_is[direct_type_key]
        # Metadata
        mc_cluster_direct.metadata = mc_cluster_direct.metadata.combine_first(mc_cluster_direct_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_cluster_direct.metadata.update(mc_cluster_direct_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
        new_image_cols = [image_col for image_col in mc_cluster_direct_i.images() if image_col not in mc_cluster_direct_images_existing]
        mc_cluster_direct_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_direct_i.get_images(image_col),
                index=mc_cluster_direct_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_direct_images_updated = mc_cluster_direct_images_updated.combine_first(mc_cluster_direct_i_new_images)
        mc_cluster_direct_images_updated.update(mc_cluster_direct_i_new_images)

    # Apply updates: Metadata, Images
    mc_cluster_direct.metadata.reset_index(inplace=True)
    for image_col in mc_cluster_direct_images_updated.columns:
        mc_cluster_direct.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_direct_images_updated[image_col]]
        )
    del mc_cluster_direct_images_updated, mc_cluster_direct_images_existing

    # # Cluster (Direct): Ensure metadata columns are consistent
    # mc_cluster_metadata_cols = set()
    # mc_cluster_image_cols = set()

    # for direct_type_key in direct_type_keys:
    #     mc_cluster_direct_i = mc_cluster_direct_is[direct_type_key]
    #     for metadata_col in mc_cluster_direct_i.columns():
    #         mc_cluster_metadata_cols.add(metadata_col)
    #     for image_col in mc_cluster_direct_i.images():
    #         mc_cluster_image_cols.add(image_col)

    # for direct_type_key in direct_type_keys:
    #     mc_cluster_direct_i = mc_cluster_direct_is[direct_type_key]
    #     for metadata_col in mc_cluster_metadata_cols:
    #         if metadata_col not in mc_cluster_direct_i.columns():
    #             mc_cluster_direct_i.metadata[metadata_col] = None
    #     for image_col in mc_cluster_image_cols:
    #         if image_col not in mc_cluster_direct_i.images():
    #             mc_cluster_direct_i.add_logos(
    #                 motifs=np.zeros_like(mc_cluster_direct_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )
    #     # Reassign
    #     mc_cluster_direct_is[direct_type_key] = mc_cluster_direct_i

    # # Combine MotifCompendiums
    # mc_cluster_direct = MotifCompendium.combine(
    #     list(mc_cluster_direct_is.values()), 
    #     safe=safe)

    ## Non-direct cluster label: TF-{counts/profile}-{TF-ChIP}_X{score}_{tf_core}::{tf_composite}
    # Boolean conditions
    ref_score_cols = [f"{ReferenceMatchCols.ref_col}_score{iter}" for iter in range(MotifMatchArgs.max_submotifs)]
    composite_score_cols = [f"{ReferenceMatchCols.ref_col}_score{iter}" for iter in range(1, MotifMatchArgs.max_submotifs)]

    mask_indirect = mc_cluster_nondirect_filtered[LabelCols.label_core_col].notna()
    mask_composite = mc_cluster_nondirect_filtered[MetadataCols.composite_col]
    mask_has_composite = mc_cluster_nondirect_filtered[LabelCols.label_flank_col].notna()

    ## Assign readable label
    # Indirect
    mc_cluster_nondirect_filtered[LabelCols.label_col] = (
        np.where(
            mask_indirect,
            mc_cluster_nondirect_filtered[LabelCols.label_core_col],
            ""
        )
        + np.where(
            mask_indirect & mask_composite & mask_has_composite,
            "::" + mc_cluster_nondirect_filtered[LabelCols.label_flank_col],
            ""
        )
    )
    mc_cluster_nondirect_filtered[MetadataCols.prename_col] = (
        label_initial
        + "-"
        + mc_cluster_nondirect_filtered[MetadataCols.tf_col]
        + "_" + mc_cluster_nondirect_filtered[DirectTypeArgs.direct_type_col]
        + (mc_cluster_nondirect_filtered[ref_score_cols].fillna(0).max(axis=1) * 100).astype(int).astype(str)
        + "_"
        + mc_cluster_nondirect_filtered[LabelCols.label_col]
    )
    mc_cluster_nondirect_filtered[MetadataCols.simple_name_col] = mc_cluster_nondirect_filtered[LabelCols.label_col]

    mc_cluster_nondirect_filtered.metadata.loc[mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[2]
    mc_cluster_nondirect_filtered.metadata.loc[~mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    ## Removed: TF-{counts/profile}-{TF-ChIP}_{FilterArgs.filter_flag_type_col}
    mc_cluster_removed[LabelCols.label_col] = mc_cluster_removed[FilterArgs.filter_flag_type_col]
    mc_cluster_removed[MetadataCols.prename_col] = (
        label_initial
        + "-"
        + mc_cluster_removed[MetadataCols.tf_col]
        + "_" 
        + mc_cluster_removed[LabelCols.label_col]
    )
    mc_cluster_removed[MetadataCols.simple_name_col] = mc_cluster_removed[LabelCols.label_col]
    mc_cluster_removed[LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    # #### Merge Final MotifCompendium: Clusters
    # Clusters: Direct, Indirect
    mc_cluster_filtereds = [mc_cluster_direct, mc_cluster_nondirect_filtered]
    mc_cluster_filtered = mc_cluster[mc_cluster[MetadataCols.name_col].isin({
        name
        for mc_cluster_i in mc_cluster_filtereds
        for name in mc_cluster_i[MetadataCols.name_col]
    })]

    # Update Metadata, Images
    mc_cluster_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_filtered_images_updated = pd.DataFrame(index=mc_cluster_filtered.metadata.index)
    mc_cluster_filtered_images_existing = set(mc_cluster_filtered.images())

    # Build updates: Metadata, Images
    for mc_cluster_filtered_i in mc_cluster_filtereds:
        # Metadata
        mc_cluster_filtered.metadata = mc_cluster_filtered.metadata.combine_first(mc_cluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_cluster_filtered.metadata.update(mc_cluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
        new_image_cols = [image_col for image_col in mc_cluster_filtered_i.images() if image_col not in mc_cluster_filtered_images_existing]
        mc_cluster_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_filtered_i.get_images(image_name=image_col),
                index=mc_cluster_filtered_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_filtered_images_updated = mc_cluster_filtered_images_updated.combine_first(mc_cluster_filtered_i_new_images)
        mc_cluster_filtered_images_updated.update(mc_cluster_filtered_i_new_images)

    # Apply updates: Images
    mc_cluster_filtered.metadata.reset_index(inplace=True)
    for image_col in mc_cluster_filtered_images_updated.columns:
        mc_cluster_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_filtered_images_updated[image_col]]
        )
    del mc_cluster_filtered_images_updated, mc_cluster_filtered_images_existing

    ### Annotate: Best label
    mc_cluster_filtered_metadata_updates = []
    mc_cluster_filtered_images_updated = pd.DataFrame(index=mc_cluster_filtered[MetadataCols.unique_id_col])
    mc_cluster_filtered_images_existing = set(mc_cluster_filtered.images())

    for i, row in mc_cluster_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_cluster_filtered_i = mc_cluster_filtered[[i]]
        label_core_i = row[LabelCols.label_core_col].split("-") if pd.notna(row[LabelCols.label_core_col]) else []
        direct_label_core_i = row[LabelCols.direct_label_core_col].split("-") if pd.notna(row[LabelCols.direct_label_core_col]) else []
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        # Reference: Subset
        if direct_label_core_i:
            mc_ref_core_i = mc_ref_direct[
                (mc_ref_direct[TFAnnotationCols.ref_tf_col].isin(direct_label_core_i)) |
                (mc_ref_direct[TFAnnotationCols.ref_family_col].isin(direct_label_core_i))
            ]
        else:
            mc_ref_core_i = mc_ref_qc[
                (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_core_i)) |
                (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_core_i))
            ]
        mc_ref_composite_i = mc_ref_qc[
            (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_flank_i)) |
            (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_flank_i))
        ]

        # Annotate with name references
        if len(mc_ref_core_i) > 0:
            assign_direct_composite(
                target_mc=mc_cluster_filtered_i,
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
        mc_cluster_filtered_metadata_updates.append(mc_cluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_cluster_filtered_i.images() if image_col not in mc_cluster_filtered_images_existing]
        mc_cluster_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_filtered_i.get_images(image_col),
                index=mc_cluster_filtered_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_filtered_images_updated = mc_cluster_filtered_images_updated.combine_first(mc_cluster_filtered_i_new_images)
        mc_cluster_filtered_images_updated.update(mc_cluster_filtered_i_new_images)

    # Apply updates: Metadata, Images
    mc_cluster_filtered_metadata_updates_df = pd.concat(mc_cluster_filtered_metadata_updates)
    mc_cluster_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_filtered.metadata = mc_cluster_filtered.metadata.combine_first(mc_cluster_filtered_metadata_updates_df)
    mc_cluster_filtered.metadata.update(mc_cluster_filtered_metadata_updates_df)
    mc_cluster_filtered.metadata.reset_index(inplace=True)
    del mc_cluster_filtered_metadata_updates_df, mc_cluster_filtered_metadata_updates

    for image_col in mc_cluster_filtered_images_updated.columns:
        mc_cluster_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_filtered_images_updated[image_col]]
        )
    del mc_cluster_filtered_images_updated, mc_cluster_filtered_images_existing


    # ## META-CLUSTER
    # ### Clustering: Clusters
    logging.info("Clustering clusters into meta-clusters...")
    # - Process:
    #     - Cluster (without boundary)
    #     - Initial clustering
    #     - Recursive clustering
    #     - Force clustering

    ## Cluster MotifCompendium
    # Cluster: Leiden CPM
    mc_cluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        cluster_within=ClusterArgs.metacluster_within,
        save_name=metacluster_cluster_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_cluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        cluster_within=ClusterArgs.metacluster_within,
        save_name=metacluster_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=metacluster_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_cluster_filtered[metacluster_cluster_col]
    logging.info(f"Unique clusters (Indirect): {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_cluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within_on=(ClusterArgs.metacluster_within, metacluster_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=metacluster_cluster_col,
            largest_clusters_first=True,
            n_iterations=ClusterArgs.n_iterations,
            seeds=ClusterArgs.seeds,
        )
        # Reassign: K-means
        mc_cluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.metacluster_within,
            save_name=metacluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=metacluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_cluster_filtered[metacluster_cluster_col]
        if args.verbose:
            logging.info(f"Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_cluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.metacluster_within, metacluster_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=metacluster_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-means
        mc_cluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.metacluster_within,
            save_name=metacluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=metacluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
        
        # Check convergence
        new_membership_cluster = mc_cluster_filtered[metacluster_cluster_col]
        if args.verbose:
            logging.info(f"Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster

    # ### Create metaclusters, by averaging
    logging.info("Creating meta-clusters, by averaging...")
    # Average MotifCompendium
    aggregations_metacluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.aggregations:
        if agg_col_in in mc_cluster_filtered.columns():
            aggregations_metacluster.append((agg_col_in, agg, agg_col_out))

    mc_metacluster = mc_cluster_filtered.cluster_averages(
        clustering=metacluster_cluster_col,
        aggregations=aggregations_metacluster,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Weighted ClusterArgs.aggregations
    weighted_aggregations_metacluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.weighted_aggregations:
        if agg_col_in in mc_cluster_filtered.columns():
            weighted_aggregations_metacluster.append((agg_col_in, agg, agg_col_out))

    for (agg_col_old, _, agg_col_new) in weighted_aggregations_metacluster:
        aggregate_weighted_avg(
            mc=mc_cluster_filtered,
            mc_cluster=mc_metacluster,
            cluster_col=metacluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )

    # Merge ClusterArgs.aggregations
    merge_aggregations_metacluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_metacluster.columns():
            merge_aggregations_metacluster.append(aggregate)
        if agg_col_new not in mc_metacluster.columns():
            mc_metacluster.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_metacluster:
        mc_metacluster.metadata[agg_col_new] = mc_metacluster.metadata[[agg_col_old, agg_col_new]].apply(
            lambda row: ",".join(dict.fromkeys(
                part
                for agg in row
                if pd.notna(agg) and agg != ""
                for part in agg.split(",")
                if part not in ("", "None")
            )),
            axis=1
        )


    # #### Calculations: Metaclusters
    ## Calculations
    mc_metacluster[MetadataCols.total_contrib_score_ratio_col] = mc_metacluster[MetadataCols.total_contrib_score_col] / mc_metacluster[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_metacluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_metacluster.get_standard_motif_stack())  # Positive / negative
    mc_metacluster[MetadataCols.composite_col] = mc_metacluster[MetadataCols.composite_col].astype(bool)

    mc_metacluster.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_metacluster.sort(by=[DirectTypeArgs.direct_type_col, MetadataCols.composite_col], ascending=True, inplace=True)  # Sort by direct type, composite
    mc_metacluster.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_metacluster[MetadataCols.unique_id_col] = unique_id_initial + f".C2.{MetadataCols.atlas_type}.{head_type}." + mc_metacluster.metadata.index.map('{:08d}'.format)


    ### Aggregate: Annotations
    mc_metacluster_metadata_updates = []
    mc_metacluster_images_updated = pd.DataFrame(index=mc_metacluster[MetadataCols.unique_id_col])
    mc_metacluster_images_existing = set(mc_metacluster.images())

    for i, row in mc_metacluster.metadata.iterrows():
        # MotifCompendium: Subset
        mc_metacluster_i = mc_metacluster[[i]]
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
                assign_to_mc=mc_metacluster_i,
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
                ((mc_ref_direct[TFAnnotationCols.ref_tf_col].isin(tf_i)) |
                (mc_ref_direct[TFAnnotationCols.ref_family_col].isin(tf_family_i)))
            ]
            
            # Reference: Subset (multimer: homo, hetero)
            mc_ref_i = mc_ref_direct[mc_ref_direct[MetadataCols.name_col].isin(ref_name_constituents_i)]  # Ref direct = Ref QC + Ref Invitro

            # Annotate with best reference constituents
            if len(mc_ref_direct_i) > 0:
                assign_direct_composite(
                    target_mc=mc_metacluster_i,
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
        mc_metacluster_metadata_updates.append(mc_metacluster_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_metacluster_i.images() if image_col not in mc_metacluster_images_existing]
        mc_metacluster_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_metacluster_i.get_images(image_col),
                index=mc_metacluster_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_metacluster_images_updated = mc_metacluster_images_updated.combine_first(mc_metacluster_i_new_images)
        mc_metacluster_images_updated.update(mc_metacluster_i_new_images)

    # Apply updates: Metadata, Images
    mc_metacluster_metadata_updates_df = pd.concat(mc_metacluster_metadata_updates)
    mc_metacluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_metacluster.metadata = mc_metacluster.metadata.combine_first(mc_metacluster_metadata_updates_df)
    mc_metacluster.metadata.update(mc_metacluster_metadata_updates_df)
    mc_metacluster.metadata.reset_index(inplace=True)
    del mc_metacluster_metadata_updates_df, mc_metacluster_metadata_updates

    for image_col in mc_metacluster_images_updated.columns:
        mc_metacluster.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_metacluster_images_updated[image_col]]
        )
    del mc_metacluster_images_updated, mc_metacluster_images_existing

    # ### Annotate: Meta-clusters
    logging.info("Annotating meta-clusters against the reference...")
    # Initialize
    # Experiment: Determined by motif; Composite: Determined by cluster
    mc_metacluster.metadata = mc_metacluster.metadata.assign(**{
        # DirectTypeArgs.direct_type_col:                None,
        # DirectTypeArgs.tf_direct_col:                  None,
        # DirectTypeArgs.family_direct_col:              None,
        # MetadataCols.num_subunit_col:                1,
        # MetadataCols.num_subunit_direct_col:         None,
        # MetadataCols.composite_col:                  False,
        LabelCols.label_col:                      MetadataCols.name_unknown,
        LabelCols.direct_label_col:               None,
        LabelCols.label_core_col:                 None,
        LabelCols.direct_label_core_col:          None,
        LabelCols.label_core_readable_col:        None,
        LabelCols.direct_label_core_readable_col: None,
        LabelCols.label_flank_col:                None,
    }).copy()


    # [General] Label clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_metacluster,
        assign_from_mc=mc_ref_qc,
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
        mc_metacluster[ref_readable_name_col] = mc_metacluster[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_nondirect_metacluster = ~mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
    mask_monomer_metacluster_nondirect = (
        mask_nondirect_metacluster
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_metacluster_nondirect = (
        mask_nondirect_metacluster
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_metacluster_nondirect = (
        mask_nondirect_metacluster
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_metacluster_nondirect = (
        mask_dimer_metacluster_nondirect | mask_trimer_metacluster_nondirect
    )
    mask_matched_metacluster_nondirect = (
        mask_monomer_metacluster_nondirect | mask_dimer_metacluster_nondirect | mask_trimer_metacluster_nondirect
    )

    # Dimer-style notation
    mc_metacluster.metadata.loc[mask_matched_metacluster_nondirect, LabelCols.label_core_col] = mc_metacluster.metadata.loc[mask_matched_metacluster_nondirect, ref_readable_name_cols[0]]
    mc_metacluster.metadata.loc[mask_matched_metacluster_nondirect, LabelCols.label_core_readable_col] = mc_metacluster.metadata.loc[mask_matched_metacluster_nondirect, ref_readable_name_cols[0]]
    mc_metacluster.metadata.loc[mask_dimer_metacluster_nondirect, LabelCols.label_flank_col] = mc_metacluster.metadata.loc[mask_dimer_metacluster_nondirect, ref_readable_name_cols[1]]
    mc_metacluster.metadata.loc[mask_trimer_metacluster_nondirect, LabelCols.label_flank_col] = mc_metacluster.metadata.loc[mask_trimer_metacluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_metacluster.metadata.loc[mask_trimer_metacluster_nondirect, ref_readable_name_cols[2]]
    # mc_metacluster.metadata.loc[mask_dimer_metacluster_nondirect, MetadataCols.num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_metacluster.metadata.loc[mask_trimer_metacluster_nondirect, MetadataCols.num_subunit_col] = 3
    # mc_metacluster.metadata[MetadataCols.composite_col] = mc_metacluster.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit



    # Concatenate all unique annotations
    mc_metacluster[ReferenceMatchCols.ref_name_col] = mc_metacluster.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # ### Filter: Meta-clusters
    # Calculate filters
    mc_metacluster[FilterArgs.filter_flag_col] = False

    utils_analysis.calculate_filters(
        mc=mc_metacluster,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )

    # Calculate filters: No scaling, trim zeros
    utils_analysis.calculate_filters(
        mc=mc_metacluster,
        metric_list=['posneg_inverted', 'truncated',],
        trim_importance=0,
    )


    # Apply filters
    apply_filter_clusters(
        mc=mc_metacluster,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )


    # Remove flagged clusters
    mc_metacluster_removed = mc_metacluster[
        (mc_metacluster[FilterArgs.filter_flag_col]) & (
            ~mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
        )
    ]
    mc_metacluster_filtered = mc_metacluster[
        ~((mc_metacluster[FilterArgs.filter_flag_col]) & (
            ~mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
        ))
    ]


    ## Cluster: For sorting only
    logging.info("Clustering meta-clusters (for sorting only)...")
    # Cluster: Leiden CPM
    mc_metacluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_metacluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        init_clustering_col=VisualizeArgs.sort_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_metacluster_filtered[VisualizeArgs.sort_col]
    if args.verbose:
        logging.info(f"Unique clusters (for sorting only): {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_metacluster_filtered.cluster(
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
        mc_metacluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_metacluster_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters (for sorting only): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_metacluster_filtered.cluster(
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
        mc_metacluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
        
        # Check convergence
        new_membership_cluster = mc_metacluster_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters (for sorting only): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster

    # ### Final meta-cluster names
    logging.info("Assigning final meta-cluster names...")
    # Do not use Gamma for names: Remove direct family name: Direct Gamma
    mc_cluster_filtered_name = mc_cluster_filtered.copy()
    mask_direct_c = mc_cluster_filtered_name[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]
    mc_cluster_filtered_name.metadata.loc[mask_direct_c, LabelCols.label_core_col] = None
    mc_cluster_filtered_name.metadata.loc[mask_direct_c, LabelCols.direct_label_core_col] = None
    mc_cluster_filtered_name.metadata.loc[mask_direct_c, LabelCols.label_core_readable_col] = None
    mc_cluster_filtered_name.metadata.loc[mask_direct_c, LabelCols.direct_label_core_readable_col] = None
    mc_cluster_filtered_name.metadata.loc[mask_direct_c, LabelCols.label_flank_col] = None

    # Count occurrence: Core readable TF, TF family (Direct)
    mc_cluster_filtered_name[LabelCols.direct_label_core_readable_col_set] = mc_cluster_filtered_name[LabelCols.direct_label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_cluster_explode_core_readable_direct = mc_cluster_filtered_name.metadata.explode(LabelCols.direct_label_core_readable_col_set)

    # Count occurrence: Core readable TF, TF family
    mc_cluster_filtered_name[LabelCols.label_core_readable_col_set] = mc_cluster_filtered_name[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_cluster_explode_core_readable = mc_cluster_filtered_name.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Flank readable TF family
    mc_cluster_filtered_name[LabelCols.label_flank_col_set] = mc_cluster_filtered_name[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_cluster_explode_readable_composite = mc_cluster_filtered_name.metadata.explode(LabelCols.label_flank_col_set)


    ## Count occurrences: Create pivot table
    # Core: Direct: Monomer
    mc_cluster_core_readable_direct_monomer_pivot = pd.pivot_table(
        mc_cluster_explode_core_readable_direct.loc[~mc_cluster_explode_core_readable_direct[MetadataCols.composite_col]],
        index=metacluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_cluster_core_readable_direct_monomer_pivot = mc_cluster_core_readable_direct_monomer_pivot.div(
        mc_cluster_core_readable_direct_monomer_pivot.sum(axis=1), axis=0)

    # Core: Direct: All
    mc_cluster_core_readable_direct_pivot = pd.pivot_table(
        mc_cluster_explode_core_readable_direct,
        index=metacluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_cluster_core_readable_direct_pivot = mc_cluster_core_readable_direct_pivot.div(
        mc_cluster_core_readable_direct_pivot.sum(axis=1), axis=0)

    # Core: All
    mc_cluster_core_readable_pivot = pd.pivot_table(
        mc_cluster_explode_core_readable,
        index=metacluster_cluster_col,
        columns=LabelCols.label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_cluster_core_readable_pivot = mc_cluster_core_readable_pivot.div(
        mc_cluster_core_readable_pivot.sum(axis=1), axis=0)

    # Composite
    mc_cluster_flank_pivot = pd.pivot_table(
        mc_cluster_explode_readable_composite,
        index=metacluster_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_cluster_flank_pivot = mc_cluster_flank_pivot.div(
        mc_cluster_flank_pivot.sum(axis=1), axis=0)


    # Match order of metaclusters
    mc_cluster_core_readable_direct_monomer_pivot = mc_cluster_core_readable_direct_monomer_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_cluster_core_readable_direct_pivot = mc_cluster_core_readable_direct_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_cluster_core_readable_pivot = mc_cluster_core_readable_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_cluster_flank_pivot = mc_cluster_flank_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')

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
            and (source_cluster in mc_cluster_core_readable_direct_monomer_pivot.index)
        ):
            core_readable_direct_monomer_frac = mc_cluster_core_readable_direct_monomer_pivot.loc[source_cluster]
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
            and (source_cluster in mc_cluster_core_readable_direct_pivot.index)
        ):
            core_readable_direct_frac = mc_cluster_core_readable_direct_pivot.loc[source_cluster]
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
            and (source_cluster in mc_cluster_core_readable_pivot.index)
        ):
            label_core_readable_frac = mc_cluster_core_readable_pivot.loc[source_cluster]
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
            and (source_cluster in mc_cluster_flank_pivot.index)
        ):
            composite_family_frac = mc_cluster_flank_pivot.loc[source_cluster]
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


    ### Label: Metacluster - Filtered
    # Add human-readable label (Direct only)
    # Add human-readable labels: Direct (Rule 1) and label (Rules 1-4), in one pass over the same rows
    label_cols = [
        LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.direct_label_col,
        LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
    ]
    mc_metacluster_filtered.metadata[label_cols] = pd.DataFrame(
        mc_metacluster_filtered.metadata.apply(assign_labels, axis=1).tolist(),
        index=mc_metacluster_filtered.metadata.index,
        columns=label_cols,
    )
    ### Label: Metacluster - Removed
    mc_metacluster_removed[LabelCols.label_col] = mc_metacluster_removed[FilterArgs.filter_flag_type_col]

    ## Prename
    mc_metacluster_filtered[MetadataCols.prename_col] = label_initial + "_" + mc_metacluster_filtered[LabelCols.label_col]
    mc_metacluster_removed[MetadataCols.prename_col] = label_initial + "_" + mc_metacluster_removed[LabelCols.label_col]

    ## Simple name
    mc_metacluster_filtered[MetadataCols.simple_name_col] = mc_metacluster_filtered[LabelCols.label_col]
    mc_metacluster_removed[MetadataCols.simple_name_col] = mc_metacluster_removed[LabelCols.label_col]

    ### Annotate: Best label
    mc_metacluster_filtered_metadata_updates = []
    mc_metacluster_filtered_images_updated = pd.DataFrame(index=mc_metacluster_filtered[MetadataCols.unique_id_col])
    mc_metacluster_filtered_images_existing = set(mc_metacluster_filtered.images())

    for i, row in mc_metacluster_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_metacluster_filtered_i = mc_metacluster_filtered[[i]]
        label_core_i = row[LabelCols.label_core_col].split("-") if pd.notna(row[LabelCols.label_core_col]) else []
        direct_label_core_i = row[LabelCols.direct_label_core_col].split("-") if pd.notna(row[LabelCols.direct_label_core_col]) else []
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        # Reference: Subset
        if direct_label_core_i:
            mc_ref_core_i = mc_ref_direct[
                (mc_ref_direct[TFAnnotationCols.ref_tf_col].isin(direct_label_core_i)) |
                (mc_ref_direct[TFAnnotationCols.ref_family_col].isin(direct_label_core_i))
            ]
        else:
            mc_ref_core_i = mc_ref_qc[
                (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_core_i)) |
                (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_core_i))
            ]
        mc_ref_composite_i = mc_ref_qc[
            (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_flank_i)) |
            (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_flank_i))
        ]

        # Annotate with name references
        if len(mc_ref_core_i) > 0:
            assign_direct_composite(
                target_mc=mc_metacluster_filtered_i,
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
        mc_metacluster_filtered_metadata_updates.append(mc_metacluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_metacluster_filtered_i.images() if image_col not in mc_metacluster_filtered_images_existing]
        mc_metacluster_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_metacluster_filtered_i.get_images(image_col),
                index=mc_metacluster_filtered_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_metacluster_filtered_images_updated = mc_metacluster_filtered_images_updated.combine_first(mc_metacluster_filtered_i_new_images)
        mc_metacluster_filtered_images_updated.update(mc_metacluster_filtered_i_new_images)

    # Apply updates: Metadata, Images
    mc_metacluster_filtered_metadata_updates_df = pd.concat(mc_metacluster_filtered_metadata_updates)
    mc_metacluster_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_metacluster_filtered.metadata = mc_metacluster_filtered.metadata.combine_first(mc_metacluster_filtered_metadata_updates_df)
    mc_metacluster_filtered.metadata.update(mc_metacluster_filtered_metadata_updates_df)
    mc_metacluster_filtered.metadata.reset_index(inplace=True)
    del mc_metacluster_filtered_metadata_updates, mc_metacluster_filtered_metadata_updates_df

    for image_col in mc_metacluster_filtered_images_updated.columns:
        mc_metacluster_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_metacluster_filtered_images_updated[image_col]]
        )
    del mc_metacluster_filtered_images_updated, mc_metacluster_filtered_images_existing

    #### Update original MotifCompendium
    logging.info("Propagating cluster / meta-cluster labels to all levels...")
    ### (3) Metacluster:
    # Update original Metacluster MotifCompendium: Metadata, Images
    mc_metaclusters = [mc_metacluster_filtered, mc_metacluster_removed]

    mc_metacluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_metacluster_images_updated = pd.DataFrame(index=mc_metacluster.metadata.index)
    mc_metacluster_images_existing = set(mc_metacluster.images())

    # Build updates: Metadata, Images
    for mc_metacluster_i in mc_metaclusters:
        # Metadata
        mc_metacluster.metadata = mc_metacluster.metadata.combine_first(mc_metacluster_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_metacluster.metadata.update(mc_metacluster_i.metadata.set_index(MetadataCols.unique_id_col))

        # Images
        new_image_cols = [image_col for image_col in mc_metacluster_i.images() if image_col not in mc_metacluster_images_existing]
        mc_metacluster_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_metacluster_i.get_images(image_col),
                index=mc_metacluster_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_metacluster_images_updated = mc_metacluster_images_updated.combine_first(mc_metacluster_i_new_images)
        mc_metacluster_images_updated.update(mc_metacluster_i_new_images)

    # Apply updates: Metadata, Images
    mc_metacluster.metadata.reset_index(inplace=True)
    for image_col in mc_metacluster_images_updated.columns:
        mc_metacluster.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_metacluster_images_updated[image_col]]
        )
    del mc_metacluster_images_updated, mc_metacluster_images_existing

    # # Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_metacluster_i in mc_metaclusters:
    #     metadata_cols = list(set(metadata_cols + mc_metacluster_i.columns()))
    #     image_cols = list(set(image_cols + mc_metacluster_i.images()))
    # for mc_metacluster_i in mc_metaclusters:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_metacluster_i.columns():
    #             mc_metacluster_i.metadata[metadata_col] = None
    #     for logo_i in ['logo (fwd)', 'logo (rev)']:
    #         mc_metacluster_i.add_logos(
    #             motifs=mc_metacluster_i.motifs,
    #             image_name=logo_i,
    #             trim=VisualizeArgs.html_logo_trim,
    #         )
    #     for image_col in image_cols:
    #         if image_col not in mc_metacluster_i.images():
    #             mc_metacluster_i.add_logos(
    #                 motifs=np.zeros_like(mc_metacluster_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Metaclusters: Combine
    # mc_metacluster = MotifCompendium.combine(mc_metaclusters, safe=safe)

    # Set label readable type
    label_conf_conditions = [
        mc_metacluster[DirectTypeArgs.direct_type_col].str.contains(DirectTypeArgs.direct_types[0], na=False),  # High: A
        mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[1:3]), na=False),  # Medium: B-NZ, B-Z
        mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[3:]), na=False),  # Low: C, X
        ~mc_metacluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types), na=False)  # Other: Flag
    ]

    mc_metacluster[LabelCols.label_conf_col] = np.select(
        label_conf_conditions,
        LabelCols.label_confs, # High, Medium, Low, Flag
        default=LabelCols.label_confs[-1],  # Remaining: Flag
    )


    ## Rename: {readable_name}_{counter}
    mc_metacluster.metadata[MetadataCols.counter_col] = mc_metacluster.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_metacluster.metadata[MetadataCols.name_new_col] = mc_metacluster.metadata[MetadataCols.prename_col] + "_" + mc_metacluster.metadata[MetadataCols.counter_col].astype(str)
    mc_metacluster.metadata[MetadataCols.name_col] = mc_metacluster.metadata[MetadataCols.name_new_col]
    mc_metacluster.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    ## Map: Cluster to Metacluster: Name, Simple name
    cluster2metacluster_metacluster_map = dict(zip(mc_metacluster[ClusterArgs.source_cluster_col], mc_metacluster[MetadataCols.name_col]))
    cluster2metacluster_simple_map = dict(zip(mc_metacluster[ClusterArgs.source_cluster_col], mc_metacluster[MetadataCols.simple_name_col]))

    mc_cluster_filtered[VisualizeArgs.metacluster_name_col] = mc_cluster_filtered[metacluster_cluster_col].map(cluster2metacluster_metacluster_map)
    mc_cluster_filtered[MetadataCols.simple_name_col] = mc_cluster_filtered[metacluster_cluster_col].map(cluster2metacluster_simple_map)


    #### (2) Cluster:
    # Update original Metacluster MotifCompendium: Metadata, Images
    mc_clusters = [mc_cluster_filtered, mc_cluster_removed]

    mc_cluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_cluster_images_updated = pd.DataFrame(index=mc_cluster.metadata.index)
    mc_cluster_images_existing = set(mc_cluster.images())

    # Build updates: Metadata, Images
    for mc_cluster_i in mc_clusters:
        # Metadata
        mc_cluster.metadata = mc_cluster.metadata.combine_first(mc_cluster_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_cluster.metadata.update(mc_cluster_i.metadata.set_index(MetadataCols.unique_id_col))

        # Images
        new_image_cols = [image_col for image_col in mc_cluster_i.images() if image_col not in mc_cluster_images_existing]
        mc_cluster_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_i.get_images(image_col),
                index=mc_cluster_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_images_updated = mc_cluster_images_updated.combine_first(mc_cluster_i_new_images)
        mc_cluster_images_updated.update(mc_cluster_i_new_images)

    # Apply updates: Metadata, Images
    mc_cluster.metadata.reset_index(inplace=True)
    for image_col in mc_cluster_images_updated.columns:
        mc_cluster.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_images_updated[image_col]]
        )
    del mc_cluster_images_updated, mc_cluster_images_existing

    # # Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_cluster_i in mc_clusters:
    #     metadata_cols = list(set(metadata_cols + mc_cluster_i.columns()))
    #     image_cols = list(set(image_cols + mc_cluster_i.images()))
    # for mc_cluster_i in mc_clusters:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_cluster_i.columns():
    #             mc_cluster_i.metadata[metadata_col] = None
    #     for logo_i in ['logo (fwd)', 'logo (rev)']:
    #         mc_cluster_i.add_logos(
    #             motifs=mc_cluster_i.motifs,
    #             image_name=logo_i,
    #             trim=VisualizeArgs.html_logo_trim,
    #         )
    #     for image_col in image_cols:
    #         if image_col not in mc_cluster_i.images():
    #             mc_cluster_i.add_logos(
    #                 motifs=np.zeros_like(mc_cluster_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Update original MotifCompendium: Cluster
    # mc_cluster = MotifCompendium.combine(mc_clusters, safe=safe)

    # ## Finalize clusters
    # #### Names
    # Add counter to name
    mc_cluster.metadata[MetadataCols.counter_col] = mc_cluster.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount()
    mc_cluster.metadata[MetadataCols.counter_col] = mc_cluster.metadata[MetadataCols.counter_col].fillna(0).astype(int)
    mc_cluster.metadata[MetadataCols.name_new_col] = mc_cluster.metadata[MetadataCols.prename_col] + "_" + mc_cluster.metadata[MetadataCols.counter_col].astype(str)
    mc_cluster.metadata[MetadataCols.name_col] = mc_cluster.metadata[MetadataCols.name_new_col]
    mc_cluster.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    ## Map: Motif to Cluster: Name, Cluster, Simple name
    motif2cluster_cluster_map = dict(zip(mc_cluster[ClusterArgs.source_cluster_col], mc_cluster[MetadataCols.name_col]))
    motif2cluster_metacluster_map = dict(zip(mc_cluster[ClusterArgs.source_cluster_col], mc_cluster[VisualizeArgs.metacluster_name_col]))
    motif2cluster_simple_map = dict(zip(mc_cluster[ClusterArgs.source_cluster_col], mc_cluster[MetadataCols.simple_name_col]))

    mc_filtered[VisualizeArgs.cluster_name_col] = mc_filtered[cluster_cluster_col].map(motif2cluster_cluster_map)
    mc_filtered[VisualizeArgs.metacluster_name_col] = mc_filtered[cluster_cluster_col].map(motif2cluster_metacluster_map)
    mc_filtered[MetadataCols.simple_name_col] = mc_filtered[cluster_cluster_col].map(motif2cluster_simple_map)


    ### (1) Motif:
    # Combine: Update columns
    mcs = [mc_filtered, mc_removed]

    # Update original MotifCompendium metadata
    mc.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    for mc_i in mcs:
        mc.metadata = mc.metadata.combine_first(mc_i.metadata.set_index(MetadataCols.unique_id_col))
        mc.metadata.update(mc_i.metadata.set_index(MetadataCols.unique_id_col))
    mc.metadata.reset_index(inplace=True)

    # Delete all images: To save memory
    for mc_i in mcs + [mc]:
        for image_col in mc_i.images():
            mc_i.delete_images(image_col)


    #### PREPARE EXPORTS ------------------------------------------------------------
    # All MotifCompendium objects are final from here on: add the sort columns,
    # slice the filtered / removed / direct views, and resolve the HTML columns once.
    logging.info("Preparing final MotifCompendium objects...")

    ## (3) Metaclusters
    mc_metacluster_filtered = mc_metacluster[~mc_metacluster[FilterArgs.filter_flag_col]]
    mc_metacluster_removed = mc_metacluster[mc_metacluster[FilterArgs.filter_flag_col]]

    ## (2) Clusters: Sort by metacluster
    mc_cluster[VisualizeArgs.sort_col] = mc_cluster[metacluster_cluster_col]
    mc_cluster_filtered = mc_cluster[~mc_cluster[FilterArgs.filter_flag_col]]
    mc_cluster_removed = mc_cluster[mc_cluster[FilterArgs.filter_flag_col]]
    mc_cluster_direct = mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]
    mc_cluster_nondirect_filtered = mc_cluster_filtered[~mc_cluster_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]

    ## (1) Motifs: Sort by cluster
    mc[VisualizeArgs.sort_col] = mc[cluster_cluster_col]
    mc_filtered = mc[~mc[FilterArgs.filter_flag_col]]
    mc_removed = mc[mc[FilterArgs.filter_flag_col]]
    mc_direct = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]
    mc_nondirect = mc_filtered[~mc_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]

    # Removed motifs: Sample a subset, for the HTML summary table only
    # (capped at the number of removed motifs: sample(n) raises if n exceeds the population)
    n_removed_sample = min(VisualizeArgs.html_removed_sample_size, len(mc_removed))
    mc_removed_sample = mc_removed[list(
        mc_removed.metadata.sample(n=n_removed_sample,
                                   random_state=ClusterArgs.seeds[0]).index
    )] if n_removed_sample > 0 else None

    ## HTML summary table columns, per level
    html_columns_metacluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_metacluster.columns() + mc_metacluster.images())]
    html_column_desc_dict_metacluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_metacluster}

    html_columns_cluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_cluster.columns() + mc_cluster.images())]
    html_column_desc_dict_cluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_cluster}

    if args.verbose:
        logging.info(
            f"Final object sizes:\n"
            f"  Motifs:       {len(mc)} ({len(mc_filtered)} filtered, {len(mc_removed)} removed; "
            f"{len(mc_direct)} direct, {len(mc_nondirect)} non-direct)\n"
            f"  Clusters:     {len(mc_cluster)} ({len(mc_cluster_filtered)} filtered, {len(mc_cluster_removed)} removed; "
            f"{len(mc_cluster_direct)} direct, {len(mc_cluster_nondirect_filtered)} non-direct)\n"
            f"  Metaclusters: {len(mc_metacluster)} ({len(mc_metacluster_filtered)} filtered, {len(mc_metacluster_removed)} removed)"
        )


    #### EXPORT --------------------------------------------------------------------
    ## SAVE: MotifCompendium object
    logging.info(f"Saving MotifCompendium objects to: {mc_dir}...")
    os.makedirs(mc_dir, exist_ok=True)

    # (1) Motifs
    mc.save(mc_motif_path)
    mc_filtered.save(mc_motif_filtered_path)
    if args.save_all:
        mc_removed.save(mc_motif_removed_path)

    # (2) Clusters
    mc_cluster.save(mc_cluster_path)
    mc_cluster_filtered.save(mc_cluster_filtered_path)
    if args.save_all:
        mc_cluster_removed.save(mc_cluster_removed_path)

    # (3) Metaclusters
    mc_metacluster.save(mc_metacluster_path)
    mc_metacluster_filtered.save(mc_metacluster_filtered_path)
    if args.save_all:
        mc_metacluster_removed.save(mc_metacluster_removed_path)


    ## HTML summary table
    logging.info(f"Saving HTML summary tables to: {html_dir}...")
    os.makedirs(html_dir, exist_ok=True)

    # (1) Motifs: Removed only (sampled subset)
    if args.save_all:
        if mc_removed_sample is not None:
            mc_removed_sample.summary_table_html(
                html_out=html_motif_removed_path,
                columns=(mc_removed_sample.columns() + mc_removed_sample.images()),
                logo_trimming=VisualizeArgs.html_logo_trim,
                editable=True,
                column_desc_dict=html_column_desc_dict_metacluster,
            )

    # (2) Clusters
    mc_cluster.summary_table_html(
        html_out=html_cluster_path,
        columns=[FilterArgs.filter_flag_col] + html_columns_cluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_cluster,
    )
    mc_cluster_filtered.summary_table_html(
        html_out=html_cluster_filtered_path,
        columns=html_columns_cluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_cluster,
    )
    if args.save_all:
        mc_cluster_removed.summary_table_html(
            html_out=html_cluster_removed_path,
            columns=[col for col in html_columns_cluster if col in (mc_cluster_removed.columns() + mc_cluster_removed.images())],
            logo_trimming=VisualizeArgs.html_logo_trim,
            editable=True,
            column_desc_dict=html_column_desc_dict_cluster,
        )
    mc_cluster_direct.summary_table_html(
        html_out=html_cluster_direct_path,
        columns=html_columns_cluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_cluster,
    )
    mc_cluster_nondirect_filtered.summary_table_html(
        html_out=html_cluster_nondirect_path,
        columns=html_columns_cluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_cluster,
    )

    # (3) Metaclusters
    mc_metacluster.summary_table_html(
        html_out=html_metacluster_path,
        columns=[FilterArgs.filter_flag_col] + html_columns_metacluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_metacluster,
    )
    mc_metacluster_filtered.summary_table_html(
        html_out=html_metacluster_filtered_path,
        columns=html_columns_metacluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_metacluster,
    )
    if args.save_all:
        mc_metacluster_removed.summary_table_html(
            html_out=html_metacluster_removed_path,
            columns=[col for col in html_columns_metacluster if col in (mc_metacluster_removed.columns() + mc_metacluster_removed.images())],
            logo_trimming=VisualizeArgs.html_logo_trim,
            editable=True,
            column_desc_dict=html_column_desc_dict_metacluster,
        )


    ## EXPORT: MoDISco HDF5
    if args.export_modisco:
        logging.info(f"Saving MoDISco HDF5 files to: {export_dir}...")
        os.makedirs(export_dir, exist_ok=True)

        # (1) Motifs
        utils_analysis.export_compendium_modisco(
            mc=mc,
            name_col=MetadataCols.name_col,
            save_loc=modisco_motif_path,
        )
        utils_analysis.export_compendium_modisco(
            mc=mc_filtered,
            name_col=MetadataCols.name_col,
            save_loc=modisco_motif_filtered_path,
        )
        if args.save_all:
            utils_analysis.export_compendium_modisco(
                mc=mc_removed,
                name_col=MetadataCols.name_col,
                save_loc=modisco_motif_removed_path,
            )

        # (2) Clusters
        utils_analysis.export_compendium_modisco(
            mc=mc_cluster,
            name_col=MetadataCols.name_col,
            save_loc=modisco_cluster_path,
        )
        utils_analysis.export_compendium_modisco(
            mc=mc_cluster_filtered,
            name_col=MetadataCols.name_col,
            save_loc=modisco_cluster_filtered_path,
        )
        if args.save_all:
            utils_analysis.export_compendium_modisco(
                mc=mc_cluster_removed,
                name_col=MetadataCols.name_col,
                save_loc=modisco_cluster_removed_path,
            )

        # (3) Metaclusters
        utils_analysis.export_compendium_modisco(
            mc=mc_metacluster,
            name_col=MetadataCols.name_col,
            save_loc=modisco_metacluster_path,
        )
        utils_analysis.export_compendium_modisco(
            mc=mc_metacluster_filtered,
            name_col=MetadataCols.name_col,
            save_loc=modisco_metacluster_filtered_path,
        )
        if args.save_all:
            utils_analysis.export_compendium_modisco(
                mc=mc_metacluster_removed,
                name_col=MetadataCols.name_col,
                save_loc=modisco_metacluster_removed_path,
            )


    ## EXPORT: MEME txt
    logging.info(f"Saving MEME txt files to: {export_dir}...")
    os.makedirs(export_dir, exist_ok=True)

    # (1) Motifs
    utils_analysis.export_compendium_meme(
        mc=mc,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_path,
        trim_importance=(1/(2*mc.motifs.shape[1])),
    )
    utils_analysis.export_compendium_meme(
        mc=mc_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_filtered_path,
        trim_importance=(1/(2*mc_filtered.motifs.shape[1])),
    )
    if args.save_all:
        utils_analysis.export_compendium_meme(
            mc=mc_removed,
            name_col=MetadataCols.name_col,
            save_loc=meme_motif_removed_path,
            trim_importance=(1/(2*mc_removed.motifs.shape[1])),
        )

    # (2) Clusters
    utils_analysis.export_compendium_meme(
        mc=mc_cluster,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_path,
        trim_importance=(1/(2*mc_cluster.motifs.shape[1])),
    )
    utils_analysis.export_compendium_meme(
        mc=mc_cluster_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_filtered_path,
        trim_importance=(1/(2*mc_cluster_filtered.motifs.shape[1])),
    )
    if args.save_all:
        utils_analysis.export_compendium_meme(
            mc=mc_cluster_removed,
            name_col=MetadataCols.name_col,
            save_loc=meme_cluster_removed_path,
            trim_importance=(1/(2*mc_cluster_removed.motifs.shape[1])),
        )

    # (3) Metaclusters
    utils_analysis.export_compendium_meme(
        mc=mc_metacluster,
        name_col=MetadataCols.name_col,
        save_loc=meme_metacluster_path,
        trim_importance=(1/(2*mc_metacluster.motifs.shape[1])),
    )
    utils_analysis.export_compendium_meme(
        mc=mc_metacluster_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_metacluster_filtered_path,
        trim_importance=(1/(2*mc_metacluster_filtered.motifs.shape[1])),
    )
    if args.save_all:
        utils_analysis.export_compendium_meme(
            mc=mc_metacluster_removed,
            name_col=MetadataCols.name_col,
            save_loc=meme_metacluster_removed_path,
            trim_importance=(1/(2*mc_metacluster_removed.motifs.shape[1])),
        )


    ## EXPORT: Metadata TSV
    logging.info(f"Saving metadata to: {metadata_dir}...")
    os.makedirs(metadata_dir, exist_ok=True)

    mc.metadata.to_csv(metadata_motif_path, sep="\t", index=False)
    mc_cluster.metadata.to_csv(metadata_cluster_path, sep="\t", index=False)
    mc_metacluster.metadata.to_csv(metadata_metacluster_path, sep="\t", index=False)

    # Motif index
    mc.metadata[VisualizeArgs.motif_index_cols].to_csv(metadata_motif_index_path, sep="\t", index=False)

    # Arguments JSON
    args_record = {
        # --- Full provenance (added 2026-09-10) ---------------------------------
        # Every parser argument, verbatim
        "parser_args": vars(args),
        # Upstream provenance: the previous run in this output dir, and the run that produced --input-mc
        "previous_args": collect_previous_args([
            ("previous_run_this_output_dir", args.output_dir),
            ("input_mc_run", os.path.dirname(os.path.dirname(args.input_mc)) if args.input_mc else None),
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
        "prename_col": MetadataCols.prename_col,
        "simple_name_col": MetadataCols.simple_name_col,
        "name_new_col": MetadataCols.name_new_col,
        "name_constituents_col": MetadataCols.name_constituents_col,
        "name_unknown": MetadataCols.name_unknown,
        "counter_col": MetadataCols.counter_col,
        "qc_col": MetadataCols.qc_col,
        "unique_id_col": MetadataCols.unique_id_col,
        "unique_id_initial": unique_id_initial,
        
        # TF ChIP-seq
        "atlas_type": MetadataCols.atlas_type,
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

        # Direct motif
        "tf_direct_col": DirectTypeArgs.tf_direct_col,
        "label_flank_col": LabelCols.label_flank_col,
        "direct_type_col": DirectTypeArgs.direct_type_col,
        "family_direct_col": DirectTypeArgs.family_direct_col,
        
        # TF-MoDISco
        "num_seqlets_col": MetadataCols.num_seqlets_col,
        "num_constituents_col": MetadataCols.num_constituents_col,
        "posneg_col": MetadataCols.posneg_col,
        "avgcontrib_score_col": MetadataCols.avgcontrib_score_col,
        "total_contrib_score_col": MetadataCols.total_contrib_score_col,
        "group_total_contrib_score_col": MetadataCols.group_total_contrib_score_col,
        "total_contrib_score_ratio_col": MetadataCols.total_contrib_score_ratio_col,
        "avgdist_summit_col": MetadataCols.avgdist_summit_col,
        "family_col": TFAnnotationCols.family_col,
        "tfclass_family_name_col": TFAnnotationCols.tfclass_family_name_col,
        "tfclass_family_id_col": TFAnnotationCols.tfclass_family_id_col,
        "humantf_dbd_col": TFAnnotationCols.humantf_dbd_col,
        
        # Annotation
        "label_col": LabelCols.label_col,
        "label_constit_1_col": LabelCols.label_constit_1_col,
        "label_constit_all_col": LabelCols.label_constit_all_col,

        "direct_label_col": LabelCols.direct_label_col,
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

        "label_core_readable_col": LabelCols.label_core_readable_col,
        "label_core_readable_col_set": LabelCols.label_core_readable_col_set,
        "label_core_readable_constit_1_col": LabelCols.label_core_readable_constit_1_col,
        "label_core_readable_constit_all_col": LabelCols.label_core_readable_constit_all_col,

        "direct_label_core_readable_col": LabelCols.direct_label_core_readable_col,
        "direct_label_core_readable_col_set": LabelCols.direct_label_core_readable_col_set,
        "direct_label_core_readable_constit_1_col": LabelCols.direct_label_core_readable_constit_1_col,
        "direct_label_core_readable_constit_all_col": LabelCols.direct_label_core_readable_constit_all_col,

        "label_flank_col": LabelCols.label_flank_col,
        "label_flank_col_set": LabelCols.label_flank_col_set,
        "label_flank_constit_1_col": LabelCols.label_flank_constit_1_col,
        "label_flank_constit_all_col": LabelCols.label_flank_constit_all_col,

        "label_conf_col": LabelCols.label_conf_col,
        "label_confs": LabelCols.label_confs,

        "best_label_ref_col": LabelCols.best_label_ref_col,

        # Parameters: Reference / Direct
        "ref_tf_col": TFAnnotationCols.ref_tf_col,
        "ref_family_col": TFAnnotationCols.ref_family_col,
        "ref_col": ReferenceMatchCols.ref_col,
        "ref_name_col": ReferenceMatchCols.ref_name_col,
        "ref_name_constit_1_col": ReferenceMatchCols.ref_name_constit_1_col,
        "ref_name_constit_all_col": ReferenceMatchCols.ref_name_constit_all_col,
        "best_ref_constit_col": ReferenceMatchCols.best_ref_constit_col,
        "ref_readable_col": ReferenceMatchCols.ref_readable_col,
        # "ref_readable_name_col": ref_readable_name_col,
        # "ref_readable_name_constit_1_col": ref_readable_name_constit_1_col,
        # "ref_readable_name_constit_all_col": ref_readable_name_constit_all_col,
        "nonznf_ref_col": ReferenceMatchCols.nonznf_ref_col,
        "nonznf_ref_readable_col": ReferenceMatchCols.nonznf_ref_readable_col,
        "min_single_score": MotifMatchArgs.min_single_score,
        "min_composite_score": MotifMatchArgs.min_composite_score,
        "max_submotifs": MotifMatchArgs.max_submotifs,
        "min_composite_subunit": MotifMatchArgs.min_composite_subunit,
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
        "direct_types": DirectTypeArgs.direct_types,
        "direct_alpha_key": DirectTypeArgs.direct_alpha_key,
        "direct_alpha_mononer_key": DirectTypeArgs.direct_alpha_mononer_key,
        "direct_alpha_multimer_key": DirectTypeArgs.direct_alpha_multimer_key,
        "direct_beta_nzf_key": DirectTypeArgs.direct_beta_nzf_key,
        "direct_beta_nzf_monomer_key": DirectTypeArgs.direct_beta_nzf_monomer_key,
        "direct_beta_nzf_multimer_key": DirectTypeArgs.direct_beta_nzf_multimer_key,
        "direct_beta_zf_key": DirectTypeArgs.direct_beta_zf_key,
        "direct_gamma_key": DirectTypeArgs.direct_gamma_key,

        # Parameters: Smith-Waterman (ZNF)
        "msw_score_col": SmithWatermanArgs.msw_score_col,
        "msw_name_col": SmithWatermanArgs.msw_name_col,
        "msw_sim_col": SmithWatermanArgs.msw_sim_col,
        "msw_orient_col": SmithWatermanArgs.msw_orient_col,
        "msw_align_col": SmithWatermanArgs.msw_align_col,
        "msw_action_col": SmithWatermanArgs.msw_action_col,
        "msw_logo_col": SmithWatermanArgs.msw_logo_col,
        "msw_sim_threshold": SmithWatermanArgs.msw_sim_threshold,
        "overlap_penalty_f1": SmithWatermanArgs.overlap_penalty_f1,
        "overlap_penalty_f2": SmithWatermanArgs.overlap_penalty_f2,
        "skip_penalty_f1": SmithWatermanArgs.skip_penalty_f1,
        "skip_penalty_f2": SmithWatermanArgs.skip_penalty_f2,
        "skip_penalty_l1": SmithWatermanArgs.skip_penalty_l1,
        "skip_penalty_l2": SmithWatermanArgs.skip_penalty_l2,
        "finger_length": SmithWatermanArgs.finger_length,
        "min_msw_score": SmithWatermanArgs.min_msw_score,
        "min_topk_msw_score": SmithWatermanArgs.min_topk_msw_score,
        "topk": SmithWatermanArgs.topk,

        # Parameters: Gamma
        "gamma_col": SmithWatermanArgs.gamma_col,

        # Parameters: Clustering
        "cluster_algorithm_main": ClusterArgs.cluster_algorithm_main,
        "cluster_sim_threshold": ClusterArgs.cluster_sim_threshold,
        "cluster_sim_threshold_force": ClusterArgs.cluster_sim_threshold_force,
        "cluster_force_density": ClusterArgs.cluster_force_density,
        "cluster_sim_assignment_threshold_kmeans": ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        "direct_cluster_within": ClusterArgs.direct_cluster_within,
        "nondirect_cluster_within": ClusterArgs.nondirect_cluster_within,
        "metacluster_within": ClusterArgs.metacluster_within,
        "cluster_direct_cluster_col": cluster_direct_cluster_col,
        "cluster_nondirect_cluster_col": cluster_nondirect_cluster_col,
        "cluster_cluster_col": cluster_cluster_col,
        "metacluster_cluster_col": metacluster_cluster_col,
        "cluster_max_iter_main": ClusterArgs.cluster_max_iter_main,
        "cluster_max_iter_force": ClusterArgs.cluster_max_iter_force,
        "cluster_algorithm_force": ClusterArgs.cluster_algorithm_force,
        "cluster_algorithm_kmeans": ClusterArgs.cluster_algorithm_kmeans,
        "average_weight_col": ClusterArgs.average_weight_col,
        "source_cluster_col": ClusterArgs.source_cluster_col,
        "n_iterations": ClusterArgs.n_iterations,
        "seeds": ClusterArgs.seeds,

        # Parameters: Filtering
        "filter_flag_col": FilterArgs.filter_flag_col,
        "filter_flag_type_col": FilterArgs.filter_flag_type_col,
        "filter_trim_importance": FilterArgs.filter_trim_importance,
        "filter_trim_length": FilterArgs.filter_trim_length,
        "filter_ic_scale": FilterArgs.filter_ic_scale,
        "filter_flag_types": FilterArgs.filter_flag_types,
        "motif_string_col": VisualizeArgs.motif_string_col,
        "motif_string_importance": VisualizeArgs.motif_string_importance,
        "motif_string_specificity": VisualizeArgs.motif_string_specificity,

        # Parameters: Hierarchy
        "cluster_name_col": VisualizeArgs.cluster_name_col,
        "metacluster_name_col": VisualizeArgs.metacluster_name_col,
        "sort_col": VisualizeArgs.sort_col,
        "max_major_family_frac": VisualizeArgs.max_major_family_frac,
        "major_family_increment": VisualizeArgs.major_family_increment,
        "max_major_family_count": VisualizeArgs.max_major_family_count,

        # Parameters: HTML
        "html_logo_trim": VisualizeArgs.html_logo_trim,

        # Aggregation
        "aggregations": ClusterArgs.aggregations,
        "weighted_aggregations": ClusterArgs.weighted_aggregations,

        # Motif index
        "motif_index_cols": VisualizeArgs.motif_index_cols,

        # Motif filter args
        **MotifFilterArgs.to_dict(),

        # Run log
        "log_path": log_path,
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    # # Final overview
    # Total motifs
    logging.info(f"Total motifs: {len(mc)}")
    logging.info(f"  Direct motifs: {len(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])])}")
    logging.info(f"    Direct (A) motifs: {len(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]])}")
    logging.info(f"    Direct (B-NZ) motifs: {len(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]])}")
    logging.info(f"    Direct (B-Z) motifs: {len(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]])}")
    logging.info(f"    Direct (C) motifs: {len(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]])}")
    logging.info(f"  Non-direct motifs: {len(mc[~mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])])}")
    logging.info(f"    Indirect motifs: {len(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[4]])}")    
    logging.info(f"  Removed motifs: {len(mc_removed)}")


    # Total ENCIDS
    logging.info(f"Total ENCIDs: {mc[MetadataCols.id_col].nunique()}")
    logging.info(f"  Direct ENCIDs: {mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.id_col].nunique()}")
    logging.info(f"    Direct (A) ENCIDs: {mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.id_col].nunique()}")
    logging.info(f"    Direct (B-NZ) ENCIDs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.id_col].unique()))}")
    logging.info(f"    Direct (B-Z) ENCIDs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.id_col].unique()))}")
    logging.info(f"    Direct (C) ENCIDs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:3])][MetadataCols.id_col].unique()))}")
    logging.info(f"  Non-direct ENCIDs: {len(set(mc[~mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.id_col].unique()))}")
    logging.info(f"    Indirect ENCIDs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[4]][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.id_col].unique()))}")
    logging.info(f"  Removed ENCIDs: {len(set(mc[mc[DirectTypeArgs.direct_type_col].isna()][MetadataCols.id_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.id_col].unique()))}")


    # Total TFs
    logging.info(f"Total TFs: {mc[MetadataCols.tf_col].nunique()}")
    logging.info(f"  Direct TFs: {mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.tf_col].nunique()}")
    logging.info(f"    Direct (A) TFs: {mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.tf_col].nunique()}")
    logging.info(f"    Direct (B-NZ) TFs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]][MetadataCols.tf_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.tf_col].unique()))}")
    logging.info(f"    Direct (B-Z) TFs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]][MetadataCols.tf_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]][MetadataCols.tf_col].unique()))}")
    logging.info(f"    Direct (C) TFs: {len(set(mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]][MetadataCols.tf_col].unique()) - set(mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:3])][MetadataCols.tf_col].unique()))}")
    logging.info(f"  Non-direct TFs: {mc[~mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.tf_col].nunique()}")
    logging.info(f"    Indirect TFs: {mc[mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[4]][MetadataCols.tf_col].nunique() - mc[mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])][MetadataCols.tf_col].nunique()}")
    logging.info(f"  Removed TFs: {mc_removed[MetadataCols.tf_col].nunique()}")


    # Total ZNFs
    logging.info(f"Total ZNFs (ENCODE): {mc[mc[MetadataCols.znf_col] == 'znf'][MetadataCols.tf_col].nunique()}")
    logging.info(f"  With B1H: {len(set(mc_b1h[TFAnnotationCols.ref_tf_col].unique()) & set(mc[MetadataCols.tf_col].unique()))}")
    logging.info(f"  With a Known Direct A: {len(set(mc[mc[MetadataCols.znf_col] == 'znf'][MetadataCols.tf_col].unique()) & set(mc_ref_direct[TFAnnotationCols.ref_tf_col].unique()))}")
    logging.info(f"  In Direct (A): {mc[(mc[MetadataCols.znf_col] == 'znf') & (mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0])][MetadataCols.tf_col].nunique()}")
    logging.info(f"  In Direct (B-NZ): {mc[(mc[MetadataCols.znf_col] == 'znf') & (mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1])][MetadataCols.tf_col].nunique()}")
    logging.info(f"  In Direct (C): {mc[(mc[MetadataCols.znf_col] == 'znf') & (mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3])][MetadataCols.tf_col].nunique()}")
    logging.info(f"    With B1H: {len(set(mc_b1h[TFAnnotationCols.ref_tf_col].unique()) & set(mc[(mc[MetadataCols.znf_col] == 'znf') & (mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3])][MetadataCols.tf_col].unique()))}")
    logging.info(f"  In Non-direct only: {len(set(mc[(mc[MetadataCols.znf_col] == 'znf') & (mc[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))][MetadataCols.tf_col].unique()) - set(mc[(mc[MetadataCols.znf_col] == 'znf') & ~(mc[DirectTypeArgs.direct_type_col].isna())][MetadataCols.tf_col].unique()))}")
    logging.info(f"    With B1H: {len(set(mc_b1h[TFAnnotationCols.ref_tf_col].unique()) - set(mc[(mc[MetadataCols.znf_col] == 'znf') & ~(mc[DirectTypeArgs.direct_type_col].isna())][MetadataCols.tf_col].unique()))}")


    # Total clusters
    logging.info(f"Total clusters: {len(mc_cluster_filtered)}")
    logging.info(f"  Direct clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col].notna()])}")
    logging.info(f"    Direct (A) clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]])}")
    logging.info(f"    Direct (B-NZ) clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]])}")
    logging.info(f"    Direct (B-Z) clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]])}")
    logging.info(f"    Direct (C) clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]])}")
    logging.info(f"  Non-direct clusters: {len(mc_cluster_filtered[~mc_cluster_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])])}")
    logging.info(f"    Indirect clusters: {len(mc_cluster_filtered[mc_cluster_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[4]])}")
    logging.info(f"  Removed clusters: {len(mc_cluster_removed)}")

    logging.info(f'Run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


def update_constituents(args, mc_metacluster_combined, metacluster2atlas_name_map, metacluster2atlas_simple_map):
    """
    The levels below the atlasclusters: each head's metacluster, cluster and motif compendia (motifcompendium_*_all.mc)
    get their atlascluster (name, simple name), by their metacluster; then the atlascluster motif index. Run by the
    combine stage (--update-constituents now), or on its own from the combine's saved outputs (--update-constituents only:
    run_tf_constituents()).
    """
    atlascluster_cluster_col = f"rec_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}"
    mc_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_metacluster_all.mc")
    mc_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_metacluster_all.mc")
    metadata_motif_index_path = os.path.join(args.output_dir, "metadata", "motifcompendium_atlascluster_motif_index.tsv")
    os.makedirs(os.path.dirname(metadata_motif_index_path), exist_ok=True)

    # ### Motif index
    # Load MotifCompendium
    mc_metacluster_counts = MotifCompendium.load(mc_metacluster_counts_path, safe=args.safe)
    mc_metacluster_profile = MotifCompendium.load(mc_metacluster_profile_path, safe=args.safe)

    mc_cluster_counts = MotifCompendium.load(mc_cluster_counts_path, safe=args.safe)
    mc_cluster_profile = MotifCompendium.load(mc_cluster_profile_path, safe=args.safe)

    mc_counts = MotifCompendium.load(mc_counts_path, safe=args.safe)
    mc_profile = MotifCompendium.load(mc_profile_path, safe=args.safe)

    for mc_count_i in [mc_metacluster_counts, mc_cluster_counts, mc_counts]:
        if 'head' not in mc_count_i.columns():
            mc_count_i['head'] = 'counts'
    for mc_profile_i in [mc_metacluster_profile, mc_cluster_profile, mc_profile]:
        if 'head' not in mc_profile_i.columns():
            mc_profile_i['head'] = 'profile'

    # Update Metaclusters
    for mc_metacluster_i in [mc_metacluster_counts, mc_metacluster_profile]:
        for metadata_col in mc_metacluster_combined.columns():
            if metadata_col not in mc_metacluster_i.columns():
                mc_metacluster_i.metadata[metadata_col] = None
        mc_metacluster_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_metacluster_i.metadata.update(mc_metacluster_combined.metadata.set_index(MetadataCols.unique_id_col))
        mc_metacluster_i.metadata.reset_index(inplace=True)

    ## Map: Metacluster to Atlas-cluster: Name, Simple name
    mc_metacluster_counts[VisualizeArgs.atlascluster_name_col] = mc_metacluster_counts[atlascluster_cluster_col].map(metacluster2atlas_name_map)
    mc_metacluster_counts[MetadataCols.simple_name_col] = mc_metacluster_counts[atlascluster_cluster_col].map(metacluster2atlas_simple_map).fillna(mc_metacluster_counts[MetadataCols.simple_name_col])

    mc_metacluster_profile[VisualizeArgs.atlascluster_name_col] = mc_metacluster_profile[atlascluster_cluster_col].map(metacluster2atlas_name_map)
    mc_metacluster_profile[MetadataCols.simple_name_col] = mc_metacluster_profile[atlascluster_cluster_col].map(metacluster2atlas_simple_map).fillna(mc_metacluster_profile[MetadataCols.simple_name_col])

    ## Map: Motif, Cluster to Atlas-cluster
    cluster2metacluster_atlascluster_map = dict(zip(mc_metacluster_combined[MetadataCols.name_col], mc_metacluster_combined[VisualizeArgs.atlascluster_name_col]))
    cluster2metacluster_simple_map = dict(zip(mc_metacluster_combined[MetadataCols.name_col], mc_metacluster_combined[MetadataCols.simple_name_col]))

    mc_cluster_counts[VisualizeArgs.atlascluster_name_col] = mc_cluster_counts[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_atlascluster_map)
    mc_cluster_counts[MetadataCols.simple_name_col] = mc_cluster_counts[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_simple_map).fillna(mc_cluster_counts[MetadataCols.simple_name_col])
    mc_cluster_profile[VisualizeArgs.atlascluster_name_col] = mc_cluster_profile[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_atlascluster_map)
    mc_cluster_profile[MetadataCols.simple_name_col] = mc_cluster_profile[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_simple_map).fillna(mc_cluster_profile[MetadataCols.simple_name_col])

    mc_counts[VisualizeArgs.atlascluster_name_col] = mc_counts[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_atlascluster_map)
    mc_counts[MetadataCols.simple_name_col] = mc_counts[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_simple_map).fillna(mc_counts[MetadataCols.simple_name_col])
    mc_profile[VisualizeArgs.atlascluster_name_col] = mc_profile[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_atlascluster_map)
    mc_profile[MetadataCols.simple_name_col] = mc_profile[VisualizeArgs.metacluster_name_col].map(cluster2metacluster_simple_map).fillna(mc_profile[MetadataCols.simple_name_col])

    # Save MotifCompendium: with the motif index, over the per-head inputs (motifcompendium_*_all.mc)
    save_replace(mc_metacluster_counts, mc_metacluster_counts_path)
    save_replace(mc_metacluster_profile, mc_metacluster_profile_path)

    save_replace(mc_cluster_counts, mc_cluster_counts_path)
    save_replace(mc_cluster_profile, mc_cluster_profile_path)

    save_replace(mc_counts, mc_counts_path)
    save_replace(mc_profile, mc_profile_path)

    # Motif index
    mc_motif_index = pd.concat([
        mc_counts.metadata,
        mc_profile.metadata,
    ], axis=0).reset_index(drop=True)

    # Export: Motif index
    mc_motif_index[VisualizeArgs.motif_index_cols_atlascluster].to_csv(metadata_motif_index_path, sep="\t", index=False)


def run_tf_constituents(args):
    """--update-constituents only: the combine stage's update of the levels below the atlasclusters (update_constituents()), from its
    saved outputs (<output-dir>/mc: motifcompendium_atlascluster_all.mc, motifcompendium_metacluster_combined.mc)."""
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_{os.path.basename(os.path.normpath(args.output_dir))}_constituents"
        f"_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.out",
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    )
    logging.info(f"Run started (the levels below the atlasclusters only): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logging.info(f"Writing run log to: {log_path}")
    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        ic_scale=args.ic,
        fast_plotting=args.fast_plot,
    )
    mc_dir = os.path.join(args.output_dir, "mc")
    mc_atlascluster = MotifCompendium.load(os.path.join(mc_dir, "motifcompendium_atlascluster_all.mc"), safe=args.safe)
    mc_metacluster_combined = MotifCompendium.load(os.path.join(mc_dir, "motifcompendium_metacluster_combined.mc"), safe=args.safe)
    update_constituents(
        args,
        mc_metacluster_combined,
        dict(zip(mc_atlascluster[ClusterArgs.source_cluster_col], mc_atlascluster[MetadataCols.name_col])),
        dict(zip(mc_atlascluster[ClusterArgs.source_cluster_col], mc_atlascluster[MetadataCols.simple_name_col])),
    )
    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def run_tf_combine(args):
    """TF atlas: combine the counts and profile meta-clusters into atlasclusters"""


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
    label_initial = MetadataCols.atlas_type

    ## Derived clustering column name
    atlascluster_cluster_col = f"rec_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}"

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ## Input: counts head
    mc_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_filtered_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_metacluster_filtered.mc")
    mc_metacluster_counts_path = os.path.join(args.counts_dir, "mc", "motifcompendium_metacluster_all.mc")

    ## Input: profile head
    mc_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_metacluster_filtered_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_metacluster_filtered.mc")
    mc_metacluster_profile_path = os.path.join(args.profile_dir, "mc", "motifcompendium_metacluster_all.mc")

    ## Output: MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")

    mc_metacluster_combined_path = os.path.join(mc_dir, "motifcompendium_metacluster_combined.mc")
    mc_atlascluster_path = os.path.join(mc_dir, "motifcompendium_atlascluster_all.mc")
    mc_atlascluster_filtered_path = os.path.join(mc_dir, "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_removed_path = os.path.join(mc_dir, "motifcompendium_atlascluster_removed.mc")

    ## Output: HTML summary table
    html_dir = os.path.join(args.output_dir, "html")

    html_atlascluster_path = os.path.join(html_dir, "motifcompendium_atlascluster_all.html")
    html_atlascluster_filtered_path = os.path.join(html_dir, "motifcompendium_atlascluster_filtered.html")
    html_atlascluster_removed_path = os.path.join(html_dir, "motifcompendium_atlascluster_removed.html")

    ## Output: Export - MoDISco HDF5, MEME
    export_dir = os.path.join(args.output_dir, "export")

    modisco_atlascluster_path = os.path.join(export_dir, "motifcompendium_atlascluster_all.h5")
    modisco_atlascluster_filtered_path = os.path.join(export_dir, "motifcompendium_atlascluster_filtered.h5")
    modisco_atlascluster_removed_path = os.path.join(export_dir, "motifcompendium_atlascluster_removed.h5")

    meme_atlascluster_path = os.path.join(export_dir, "motifcompendium_atlascluster_all.meme.txt")
    meme_atlascluster_filtered_path = os.path.join(export_dir, "motifcompendium_atlascluster_filtered.meme.txt")
    meme_atlascluster_removed_path = os.path.join(export_dir, "motifcompendium_atlascluster_removed.meme.txt")

    ## Output: Metadata
    metadata_dir = os.path.join(args.output_dir, "metadata")

    metadata_atlascluster_path = os.path.join(metadata_dir, "motifcompendium_atlascluster_all_metadata.tsv")
    metadata_motif_index_path = os.path.join(metadata_dir, "motifcompendium_atlascluster_motif_index.tsv")
    args_json_path = os.path.join(metadata_dir, "args.json")


    # ## Initial Meta-clusters

    # #### Load MotifCompendium
    # Load MotifCompendium
    logging.info(f"Loading meta-cluster MotifCompendium objects (counts): {mc_metacluster_filtered_counts_path}")
    mc_metacluster_filtered_counts = MotifCompendium.load(mc_metacluster_filtered_counts_path, safe=args.safe)
    logging.info(f"  Meta-clusters: {len(mc_metacluster_filtered_counts)}")
    logging.info(f"Loading meta-cluster MotifCompendium objects (profile): {mc_metacluster_filtered_profile_path}")
    mc_metacluster_filtered_profile = MotifCompendium.load(mc_metacluster_filtered_profile_path, safe=args.safe)

    # The per-head selection: once the ENCODE combine has written its final flags back, motifcompendium_*_filtered.mc
    # holds those instead, and this stage would combine an already-filtered selection
    for mc_input, mc_input_path in (
        (mc_metacluster_filtered_counts, mc_metacluster_filtered_counts_path),
        (mc_metacluster_filtered_profile, mc_metacluster_filtered_profile_path),
    ):
        if VisualizeArgs.encodecluster_name_col in mc_input.columns():
            raise ValueError(
                f"{mc_input_path} holds the final flags of an ENCODE combine ({VisualizeArgs.encodecluster_name_col}), "
                "not the per-head selection: rerun the per-head stages (counts, profile) before this stage."
            )
    logging.info(f"  Meta-clusters: {len(mc_metacluster_filtered_profile)}")


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
    if args.ref_znf_b1h.endswith(".mc"):
        mc_b1h = MotifCompendium.load(args.ref_znf_b1h, safe=args.safe)
    else:
        mc_b1h = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref_znf_b1h))[0]: args.ref_znf_b1h},
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
    mc_b1h.motifs = utils_motif.ic_scale(mc_b1h.motifs)

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
    mc_ref_qc_tf_dict = dict(zip(mc_ref_qc[MetadataCols.name_col], mc_ref_qc[TFAnnotationCols.ref_tf_col]))

    mc_ref_qc[LabelCols.label_col] = mc_ref_qc[MetadataCols.name_col].map(mc_ref_qc_hr_dict)


    # ### Combine: Meta-clusters
    logging.info("Combining counts + profile meta-clusters...")

    # #### Pairwise similarity
    # Combine MotifCompendium
    mc_metacluster_combined = MotifCompendium.combine(
        [mc_metacluster_filtered_counts, mc_metacluster_filtered_profile],
        safe=args.safe,
    )
    logging.info(f"  Combined meta-clusters: {len(mc_metacluster_combined)}")


    # #### Combine
    # Force-cluster
    mc_metacluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_force,
        similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
        cluster_within=ClusterArgs.atlascluster_within,
        save_name=atlascluster_cluster_col,
        largest_clusters_first=True,
        seed=ClusterArgs.seeds[0],
    )
    # Reassign: K-mean distance
    mc_metacluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        cluster_within=ClusterArgs.atlascluster_within,
        save_name=atlascluster_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=atlascluster_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
    )
    membership_atlas = mc_metacluster_combined[atlascluster_cluster_col]
    logging.info(f"Unique clusters (Indirect): {len(set(membership_atlas))}")

    # Cluster: Recursively
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Force-cluster
        mc_metacluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.atlascluster_within, atlascluster_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=atlascluster_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-mean distance
        mc_metacluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.atlascluster_within,
            save_name=atlascluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=atlascluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        )

        # Check convergence
        new_membership_atlas = mc_metacluster_combined[atlascluster_cluster_col]
        if args.verbose:
            logging.info(f"  Unique clusters (Indirect): {len(set(new_membership_atlas))}")
        if np.array_equal(membership_atlas, new_membership_atlas):
            logging.info(f"Final unique clusters (Indirect): {len(set(new_membership_atlas))}")
            break
        else:
            membership_atlas = new_membership_atlas


    # ## Final: TF Atlas
    # #### Average
    logging.info("Creating atlasclusters, by averaging...")
    # 
    # Average MotifCompendium
    aggregations_atlascluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.aggregations:
        if agg_col_in in mc_metacluster_combined.columns():
            aggregations_atlascluster.append((agg_col_in, agg, agg_col_out))

    mc_tfatlas = mc_metacluster_combined.cluster_averages(
        clustering=atlascluster_cluster_col,
        aggregations=aggregations_atlascluster,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Weighted aggregations
    weighted_aggregations_atlascluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.weighted_aggregations:
        if agg_col_in in mc_metacluster_combined.columns():
            weighted_aggregations_atlascluster.append((agg_col_in, agg, agg_col_out))

    for (agg_col_old, _, agg_col_new) in weighted_aggregations_atlascluster:
        aggregate_weighted_avg(
            mc=mc_metacluster_combined,
            mc_cluster=mc_tfatlas,
            cluster_col=atlascluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )

    # Merge aggregations
    merge_aggregations_atlascluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_tfatlas.columns():
            merge_aggregations_atlascluster.append(aggregate)
        if agg_col_new not in mc_tfatlas.columns():
            mc_tfatlas.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_atlascluster:
        mc_tfatlas.metadata[agg_col_new] = mc_tfatlas.metadata[[agg_col_old, agg_col_new]].apply(
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
    mc_tfatlas[MetadataCols.total_contrib_score_ratio_col] = mc_tfatlas[MetadataCols.total_contrib_score_col] / mc_tfatlas[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_tfatlas[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_tfatlas.get_standard_motif_stack())  # Positive / negative
    mc_tfatlas[MetadataCols.composite_col] = mc_tfatlas[MetadataCols.composite_col].astype(bool)

    mc_tfatlas.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_tfatlas.sort(by=[direct_type_col, composite_col], ascending=True, inplace=True)  # Sort by direct type, composite
    mc_tfatlas.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_tfatlas[MetadataCols.unique_id_col] = unique_id_initial + f".C3.{MetadataCols.atlas_type}.counts_profile." + mc_tfatlas.metadata.index.map('{:08d}'.format)

    ### Aggregate: Annotations
    mc_tfatlas_metadata_updates = []
    mc_tfatlas_images_updated = pd.DataFrame(index=mc_tfatlas[MetadataCols.unique_id_col])
    mc_tfatlas_images_existing = set(mc_tfatlas.images())

    for i, row in mc_tfatlas.metadata.iterrows():
        # MotifCompendium: Subset
        mc_tfatlas_i = mc_tfatlas[[i]]
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
                ((mc_ref_direct[TFAnnotationCols.ref_tf_col].isin(tf_i)) |
                 (mc_ref_direct[TFAnnotationCols.ref_family_col].isin(tf_family_i)))
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
    mc_tfatlas.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_tfatlas.metadata = mc_tfatlas.metadata.combine_first(mc_tfatlas_metadata_updates_df)
    mc_tfatlas.metadata.update(mc_tfatlas_metadata_updates_df)
    mc_tfatlas.metadata.reset_index(inplace=True)
    del mc_tfatlas_metadata_updates, mc_tfatlas_metadata_updates_df

    for image_col in mc_tfatlas_images_updated.columns:
        mc_tfatlas.set_images(
            image_name=image_col,
            images=mc_tfatlas_images_updated[image_col].where(
                pd.notna(mc_tfatlas_images_updated[image_col]), None).to_list()
        )
    del mc_tfatlas_images_updated, mc_tfatlas_images_existing


    # #### Annotate
    # ### Annotate: Meta-clusters
    logging.info("Annotating atlasclusters against the reference...")
    # Initialize
    mc_tfatlas[LabelCols.label_col] = MetadataCols.name_unknown  # Initial readable name: Unknown
    mc_tfatlas[LabelCols.direct_label_col] = None
    mc_tfatlas[LabelCols.label_core_col] = None
    mc_tfatlas[LabelCols.direct_label_core_col] = None
    mc_tfatlas[LabelCols.label_core_readable_col] = None
    mc_tfatlas[LabelCols.direct_label_core_readable_col] = None
    mc_tfatlas[LabelCols.label_flank_col] = None
    # mc_tfatlas[ref_readable_name_col] = None

    ## [General] Label meta-clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_tfatlas,
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
        mc_tfatlas[ref_readable_name_col] = mc_tfatlas[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_nondirect_atlascluster = ~mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
    mask_monomer_atlascluster_nondirect = (
        mask_nondirect_atlascluster
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_atlascluster_nondirect = (
        mask_nondirect_atlascluster
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_atlascluster_nondirect = (
        mask_nondirect_atlascluster
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_tfatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_atlascluster_nondirect = (
        mask_dimer_atlascluster_nondirect | mask_trimer_atlascluster_nondirect
    )
    mask_matched_atlascluster_nondirect = (
        mask_monomer_atlascluster_nondirect | mask_dimer_atlascluster_nondirect | mask_trimer_atlascluster_nondirect
    )

    # Dimer-style notation
    mc_tfatlas.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_col] = mc_tfatlas.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_tfatlas.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_readable_col] = mc_tfatlas.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_tfatlas.metadata.loc[mask_dimer_atlascluster_nondirect, LabelCols.label_flank_col] = mc_tfatlas.metadata.loc[mask_dimer_atlascluster_nondirect, ref_readable_name_cols[1]]
    mc_tfatlas.metadata.loc[mask_trimer_atlascluster_nondirect, LabelCols.label_flank_col] = mc_tfatlas.metadata.loc[mask_trimer_atlascluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_tfatlas.metadata.loc[mask_trimer_atlascluster_nondirect, ref_readable_name_cols[2]]
    # mc_tfatlas.metadata.loc[mask_dimer_atlascluster_nondirect, num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_tfatlas.metadata.loc[mask_trimer_atlascluster_nondirect, num_subunit_col] = 3
    # mc_tfatlas.metadata[composite_col] = mc_tfatlas.metadata[num_subunit_col] > min_composite_subunit

    # # Trimer-style notation
    # mc_tfatlas[ref_readable_name_col] = mc_tfatlas.metadata[
    #     [
    #         f'{ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

    # Concatenate all unique annotations
    mc_tfatlas[ReferenceMatchCols.ref_name_col] = mc_tfatlas.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### Filter
    logging.info("Filtering atlasclusters...")
    # Calculate filters
    mc_tfatlas[FilterArgs.filter_flag_col] = False
    utils_analysis.calculate_filters(
        mc=mc_tfatlas,
        metric_list=AtlasFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )

    # Apply filters
    apply_filter_clusters(
        mc=mc_tfatlas,
        motif_filter_args=AtlasFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Remove flagged clusters
    mc_tfatlas_removed = mc_tfatlas[
        (mc_tfatlas[FilterArgs.filter_flag_col]) & (~mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False))
    ]
    mc_tfatlas_filtered = mc_tfatlas[~(
        (mc_tfatlas[FilterArgs.filter_flag_col]) & (~mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False))
    )]


    # #### Clustering (for sorting only)
    ## Cluster: For sorting only
    logging.info("Clustering atlasclusters (for sorting only)...")
    # Cluster: Leiden CPM
    mc_tfatlas_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_tfatlas_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        init_clustering_col=VisualizeArgs.sort_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_tfatlas_filtered[VisualizeArgs.sort_col]
    if args.verbose:
        logging.info(f"Unique clusters (for sorting only): {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_tfatlas_filtered.cluster(
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
        mc_tfatlas_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_tfatlas_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters (for sorting only): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_tfatlas_filtered.cluster(
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
        mc_tfatlas_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
    
        # Check convergence
        new_membership_cluster = mc_tfatlas_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters (for sorting only): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # #### Name: Aggregate TF, Composite TF family
    # Count occurrence: Core TF family (Direct)
    mc_metacluster_combined[LabelCols.direct_label_core_readable_col_set] = mc_metacluster_combined[LabelCols.direct_label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_metacluster_explode_core_readable_direct = mc_metacluster_combined.metadata.explode(LabelCols.direct_label_core_readable_col_set)

    # Count occurrence: Core TF family
    mc_metacluster_combined[LabelCols.label_core_readable_col_set] = mc_metacluster_combined[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_metacluster_explode_core_readable = mc_metacluster_combined.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_metacluster_combined[LabelCols.label_flank_col_set] = mc_metacluster_combined[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_metacluster_explode_flank = mc_metacluster_combined.metadata.explode(LabelCols.label_flank_col_set)

    ## Count occurrences: Create pivot table
    # Core: Direct: Monomer
    mc_metacluster_core_readable_direct_monomer_pivot = pd.pivot_table(
        mc_metacluster_explode_core_readable_direct.loc[~mc_metacluster_explode_core_readable_direct[MetadataCols.composite_col]],
        index=atlascluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_metacluster_core_readable_direct_monomer_pivot = mc_metacluster_core_readable_direct_monomer_pivot.div(
        mc_metacluster_core_readable_direct_monomer_pivot.sum(axis=1), axis=0)

    # Core: Direct: All
    mc_metacluster_core_readable_direct_pivot = pd.pivot_table(
        mc_metacluster_explode_core_readable_direct,
        index=atlascluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_metacluster_core_readable_direct_pivot = mc_metacluster_core_readable_direct_pivot.div(
        mc_metacluster_core_readable_direct_pivot.sum(axis=1), axis=0)

    # Core: All
    mc_metacluster_core_readable_pivot = pd.pivot_table(
        mc_metacluster_explode_core_readable,
        index=atlascluster_cluster_col,
        columns=LabelCols.label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_metacluster_core_readable_pivot = mc_metacluster_core_readable_pivot.div(
        mc_metacluster_core_readable_pivot.sum(axis=1), axis=0)

    # Composite
    mc_metacluster_flank_pivot = pd.pivot_table(
        mc_metacluster_explode_flank,
        index=atlascluster_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_metacluster_flank_pivot = mc_metacluster_flank_pivot.div(
        mc_metacluster_flank_pivot.sum(axis=1), axis=0)

    # Match order of metaclusters
    mc_metacluster_core_readable_direct_monomer_pivot = mc_metacluster_core_readable_direct_monomer_pivot.reindex(
        mc_tfatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_metacluster_core_readable_direct_pivot = mc_metacluster_core_readable_direct_pivot.reindex(
        mc_tfatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_metacluster_core_readable_pivot = mc_metacluster_core_readable_pivot.reindex(
        mc_tfatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_metacluster_flank_pivot = mc_metacluster_flank_pivot.reindex(
        mc_tfatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')

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
            and (source_cluster in mc_metacluster_core_readable_direct_monomer_pivot.index)
        ):
            core_readable_direct_monomer_frac = mc_metacluster_core_readable_direct_monomer_pivot.loc[source_cluster]

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
            and (source_cluster in mc_metacluster_core_readable_direct_pivot.index)
        ):
            core_readable_direct_frac = mc_metacluster_core_readable_direct_pivot.loc[source_cluster]

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
            and (source_cluster in mc_metacluster_core_readable_pivot.index)
        ):
            label_core_readable_frac = mc_metacluster_core_readable_pivot.loc[source_cluster]
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
            and (source_cluster in mc_metacluster_flank_pivot.index)
        ):
            composite_family_frac = mc_metacluster_flank_pivot.loc[source_cluster]
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

    # Label: Metacluster - Filtered
    # Add human-readable labels: Direct (Rule 1) and label (Rules 1-4), in one pass over the same rows
    label_cols = [
        LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.direct_label_col,
        LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
    ]
    mc_tfatlas_filtered.metadata[label_cols] = pd.DataFrame(
        mc_tfatlas_filtered.metadata.apply(assign_labels, axis=1).tolist(),
        index=mc_tfatlas_filtered.metadata.index,
        columns=label_cols,
    )
    # Label: Metacluster - Removed
    mc_tfatlas_removed[LabelCols.label_col] = mc_tfatlas_removed[FilterArgs.filter_flag_type_col]

    ## Prename
    mc_tfatlas_filtered[MetadataCols.prename_col] = label_initial + "_" + mc_tfatlas_filtered[LabelCols.label_col]
    mc_tfatlas_removed[MetadataCols.prename_col] = label_initial + "_" + mc_tfatlas_removed[LabelCols.label_col]

    ## Simple name
    mc_tfatlas_filtered[MetadataCols.simple_name_col] = mc_tfatlas_filtered[LabelCols.label_col]
    mc_tfatlas_removed[MetadataCols.simple_name_col] = mc_tfatlas_removed[LabelCols.label_col]

    ### Annotate: Best label
    mc_tfatlas_filtered_metadata_updates = []
    mc_tfatlas_filtered_images_updated = pd.DataFrame(index=mc_tfatlas_filtered[MetadataCols.unique_id_col])
    mc_tfatlas_filtered_images_existing = set(mc_tfatlas_filtered.images())

    for i, row in mc_tfatlas_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_tfatlas_filtered_i = mc_tfatlas_filtered[[i]]
        label_core_i = row[LabelCols.label_core_col].split("-") if pd.notna(row[LabelCols.label_core_col]) else []
        direct_label_core_i = row[LabelCols.direct_label_core_col].split("-") if pd.notna(row[LabelCols.direct_label_core_col]) else []
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        # Reference: Subset
        if direct_label_core_i:
            mc_ref_core_i = mc_ref_direct[
                (mc_ref_direct[TFAnnotationCols.ref_tf_col].isin(direct_label_core_i)) |
                (mc_ref_direct[TFAnnotationCols.ref_family_col].isin(direct_label_core_i))
            ]
        else:
            mc_ref_core_i = mc_ref_qc[
                (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_core_i)) |
                (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_core_i))
            ]
        mc_ref_composite_i = mc_ref_qc[
            (mc_ref_qc[TFAnnotationCols.ref_tf_col].isin(label_flank_i)) |
            (mc_ref_qc[TFAnnotationCols.ref_family_col].isin(label_flank_i))
        ]

        # Annotate with name references
        if len(mc_ref_core_i) > 0:
            assign_direct_composite(
                target_mc=mc_tfatlas_filtered_i,
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
        mc_tfatlas_filtered_metadata_updates.append(mc_tfatlas_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
    
        new_image_cols = [image_col for image_col in mc_tfatlas_filtered_i.images() if image_col not in mc_tfatlas_filtered_images_existing]
        mc_tfatlas_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_tfatlas_filtered_i.get_images(image_col),
                index=mc_tfatlas_filtered_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_tfatlas_filtered_images_updated = mc_tfatlas_filtered_images_updated.combine_first(mc_tfatlas_filtered_i_new_images)
        mc_tfatlas_filtered_images_updated.update(mc_tfatlas_filtered_i_new_images)

    # Apply updates: Metadata, Images
    mc_tfatlas_filtered_metadata_updates_df = pd.concat(mc_tfatlas_filtered_metadata_updates)
    mc_tfatlas_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_tfatlas_filtered.metadata = mc_tfatlas_filtered.metadata.combine_first(mc_tfatlas_filtered_metadata_updates_df)
    mc_tfatlas_filtered.metadata.update(mc_tfatlas_filtered_metadata_updates_df)
    mc_tfatlas_filtered.metadata.reset_index(inplace=True)
    del mc_tfatlas_filtered_metadata_updates, mc_tfatlas_filtered_metadata_updates_df

    for image_col in mc_tfatlas_filtered_images_updated.columns:
        mc_tfatlas_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_tfatlas_filtered_images_updated[image_col]],
        )
    del mc_tfatlas_filtered_images_updated, mc_tfatlas_filtered_images_existing


    # #### Update MotifCompendium
    #### Update original MotifCompendium
    ### (4) Atlascluster:
    # Combine: Update columns
    mc_tfatlases = [mc_tfatlas_filtered, mc_tfatlas_removed]

    mc_tfatlas.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_tfatlas_images_updated = pd.DataFrame(index=mc_tfatlas.metadata.index)
    mc_tfatlas_images_existing = set(mc_tfatlas.images())

    # Build updates: Metadata, Images
    for mc_tfatlas_i in mc_tfatlases:
        # Metadata
        mc_tfatlas.metadata = mc_tfatlas.metadata.combine_first(mc_tfatlas_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_tfatlas.metadata.update(mc_tfatlas_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
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
    mc_tfatlas.metadata.reset_index(inplace=True)
    for image_col in mc_tfatlas_images_updated.columns:
        mc_tfatlas.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_tfatlas_images_updated[image_col]],
        )
    del mc_tfatlas_images_updated, mc_tfatlas_images_existing


    # Set label readable type
    label_conf_conditions = [
        mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains(DirectTypeArgs.direct_types[0], na=False),  # High: A
        mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[1:3]), na=False),  # Medium: B-NZ, B-Z
        mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[3:]), na=False),  # Low: C, X
        ~mc_tfatlas[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types), na=False)  # Other: Flag
    ]

    mc_tfatlas[LabelCols.label_conf_col] = np.select(
        label_conf_conditions,
        LabelCols.label_confs, # High, Medium, Low, Flag
        default=LabelCols.label_confs[-1]  # Remaining: Flag
    )

    ## Rename: {readable_name}_{counter}
    mc_tfatlas.metadata[MetadataCols.counter_col] = mc_tfatlas.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_tfatlas.metadata[MetadataCols.name_new_col] = mc_tfatlas.metadata[MetadataCols.prename_col] + "_" + mc_tfatlas.metadata[MetadataCols.counter_col].astype(str)
    mc_tfatlas.metadata[MetadataCols.name_col] = mc_tfatlas.metadata[MetadataCols.name_new_col]
    mc_tfatlas.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    ## Map: Metacluster to Atlas-cluster: Name, Simple name
    metacluster2atlas_name_map = dict(zip(mc_tfatlas[ClusterArgs.source_cluster_col], mc_tfatlas[MetadataCols.name_col]))
    metacluster2atlas_simple_map = dict(zip(mc_tfatlas[ClusterArgs.source_cluster_col], mc_tfatlas[MetadataCols.simple_name_col]))

    mc_metacluster_combined[VisualizeArgs.atlascluster_name_col] = mc_metacluster_combined[atlascluster_cluster_col].map(metacluster2atlas_name_map)
    mc_metacluster_combined[MetadataCols.simple_name_col] = mc_metacluster_combined[atlascluster_cluster_col].map(metacluster2atlas_simple_map)


    # #### Visualize HTML
    ## Create HTML
    html_columns_atlascluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_tfatlas.columns() + mc_tfatlas.images())]
    html_column_desc_dict_atlascluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_atlascluster}

    os.makedirs(os.path.dirname(html_atlascluster_path), exist_ok=True)
    mc_tfatlas.summary_table_html(
        html_out=html_atlascluster_path,
        columns=[FilterArgs.filter_flag_col]+html_columns_atlascluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_atlascluster,
    )

    os.makedirs(os.path.dirname(html_atlascluster_filtered_path), exist_ok=True)
    mc_tfatlas_filtered = mc_tfatlas[~mc_tfatlas[FilterArgs.filter_flag_col]]
    mc_tfatlas_filtered.summary_table_html(
        html_out=html_atlascluster_filtered_path,
        columns=html_columns_atlascluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_atlascluster,
    )

    if args.save_all:
        os.makedirs(os.path.dirname(html_atlascluster_removed_path), exist_ok=True)
    mc_tfatlas_removed = mc_tfatlas[mc_tfatlas[FilterArgs.filter_flag_col]]
    if args.save_all:
        mc_tfatlas_removed.summary_table_html(
            html_out=html_atlascluster_removed_path,
            columns=[col for col in VisualizeArgs.html_columns if col in (mc_tfatlas_removed.columns() + mc_tfatlas_removed.images())],
            logo_trimming=VisualizeArgs.html_logo_trim,
            editable=True,
            column_desc_dict=html_column_desc_dict_atlascluster,
        )

    # os.makedirs(os.path.dirname(html_tfatlas_monomer_path), exist_ok=True)
    # mc_tfatlas_monomer = mc_tfatlas_filtered[~mc_tfatlas_filtered[composite_col]]
    # mc_tfatlas_monomer.summary_table_html(
    #     html_out=html_tfatlas_monomer_path,
    #     columns=html_columns_atlascluster,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_atlascluster,
    # )

    # os.makedirs(os.path.dirname(html_tfatlas_multimer_path), exist_ok=True)
    # mc_tfatlas_multimer = mc_tfatlas_filtered[mc_tfatlas_filtered[composite_col]]
    # mc_tfatlas_multimer.summary_table_html(
    #     html_out=html_tfatlas_multimer_path,
    #     columns=html_columns_atlascluster,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_atlascluster,
    # )


    # #### Save MotifCompendium
    # Save MotifCompendium: TF Atlas
    os.makedirs(os.path.dirname(mc_metacluster_combined_path), exist_ok=True)
    mc_metacluster_combined.save(mc_metacluster_combined_path)

    os.makedirs(os.path.dirname(mc_atlascluster_path), exist_ok=True)
    mc_tfatlas.save(mc_atlascluster_path)

    os.makedirs(os.path.dirname(mc_atlascluster_filtered_path), exist_ok=True)
    mc_tfatlas_filtered.save(mc_atlascluster_filtered_path)

    if args.save_all:
        os.makedirs(os.path.dirname(mc_atlascluster_removed_path), exist_ok=True)
        mc_tfatlas_removed.save(mc_atlascluster_removed_path)

    # os.makedirs(os.path.dirname(mc_tfatlas_monomer_path), exist_ok=True)
    # mc_tfatlas_monomer.save(mc_tfatlas_monomer_path)

    # os.makedirs(os.path.dirname(mc_tfatlas_multimer_path), exist_ok=True)
    # mc_tfatlas_multimer.save(mc_tfatlas_multimer_path)


    # ### Export
    # Export as MEME txt
    os.makedirs(os.path.dirname(meme_atlascluster_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_tfatlas,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_path,
        trim_importance=(1/(2*mc_tfatlas.motifs.shape[1])),
    )

    os.makedirs(os.path.dirname(meme_atlascluster_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_tfatlas_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_filtered_path,
        trim_importance=(1/(2*mc_tfatlas_filtered.motifs.shape[1])),
    )

    if args.save_all:
        os.makedirs(os.path.dirname(meme_atlascluster_removed_path), exist_ok=True)
        utils_analysis.export_compendium_meme(
            mc=mc_tfatlas_removed,
            name_col=MetadataCols.name_col,
            save_loc=meme_atlascluster_removed_path,
            trim_importance=(1/(2*mc_tfatlas_removed.motifs.shape[1])),
        )

    ## EXPORT: MoDISco HDF5
    if args.export_modisco:
        logging.info("Saving MoDISco HDF5 files...")
        os.makedirs(os.path.dirname(modisco_atlascluster_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_tfatlas,
            name_col=MetadataCols.name_col,
            save_loc=modisco_atlascluster_path,
        )
        utils_analysis.export_compendium_modisco(
            mc=mc_tfatlas_filtered,
            name_col=MetadataCols.name_col,
            save_loc=modisco_atlascluster_filtered_path,
        )
        if args.save_all:
            utils_analysis.export_compendium_modisco(
                mc=mc_tfatlas_removed,
                name_col=MetadataCols.name_col,
                save_loc=modisco_atlascluster_removed_path,
            )

    # Metadata TSV
    os.makedirs(metadata_dir, exist_ok=True)
    mc_tfatlas.metadata.to_csv(metadata_atlascluster_path, sep="\t", index=False)


    # #### The levels below: each head's metaclusters, clusters and motifs (their atlascluster), the motif index
    if args.update_constituents == "now":
        update_constituents(args, mc_metacluster_combined, metacluster2atlas_name_map, metacluster2atlas_simple_map)
    else:
        logging.info("The levels below the atlasclusters: not updated (--update-constituents defer; later: --update-constituents only)")

    # Args JSON
    args_record = {
        # --- Full provenance (added 2026-09-10) ---------------------------------
        # Every parser argument, verbatim
        "parser_args": vars(args),
        # Upstream provenance: the per-head runs this combine consumes
        "previous_args": collect_previous_args([
            ("counts", args.counts_dir),
            ("profile", args.profile_dir),
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
        "prename_col": MetadataCols.prename_col,
        "simple_name_col": MetadataCols.simple_name_col,
        "name_new_col": MetadataCols.name_new_col,
        "name_constituents_col": MetadataCols.name_constituents_col,
        "name_unknown": MetadataCols.name_unknown,
        "counter_col": MetadataCols.counter_col,
        "qc_col": MetadataCols.qc_col,
        "unique_id_col": MetadataCols.unique_id_col,
        "unique_id_initial": unique_id_initial,

        # Experiment
        "atlas_type": MetadataCols.atlas_type,
        "id_col": MetadataCols.id_col,
        "tf_col": MetadataCols.tf_col,
        "tf_encode_col": MetadataCols.tf_encode_col,
        "model_col": MetadataCols.model_col,
        "head_col": MetadataCols.head_col,
        "accession_col": MetadataCols.accession_col,
        "biosample_col": MetadataCols.biosample_col,
        "assay_col": MetadataCols.assay_col,
        "znf_col": MetadataCols.znf_col,

        # Direct motif
        "composite_col": MetadataCols.composite_col,
        "num_subunit_col": MetadataCols.num_subunit_col,
        "num_subunit_direct_col": MetadataCols.num_subunit_direct_col,
        "tf_direct_col": DirectTypeArgs.tf_direct_col,
        "label_flank_col": LabelCols.label_flank_col,
        "direct_type_col": DirectTypeArgs.direct_type_col,
        "family_direct_col": DirectTypeArgs.family_direct_col,

        # TF-MoDISco
        "num_seqlets_col": MetadataCols.num_seqlets_col,
        "num_constituents_col": MetadataCols.num_constituents_col,
        "posneg_col": MetadataCols.posneg_col,
        "avgcontrib_score_col": MetadataCols.avgcontrib_score_col,
        "total_contrib_score_col": MetadataCols.total_contrib_score_col,
        "group_total_contrib_score_col": MetadataCols.group_total_contrib_score_col,
        "total_contrib_score_ratio_col": MetadataCols.total_contrib_score_ratio_col,
        "avgdist_summit_col": MetadataCols.avgdist_summit_col,
        "family_col": TFAnnotationCols.family_col,
        "tfclass_family_name_col": TFAnnotationCols.tfclass_family_name_col,
        "tfclass_family_id_col": TFAnnotationCols.tfclass_family_id_col,
        "humantf_dbd_col": TFAnnotationCols.humantf_dbd_col,

        # Annotation
        "label_initial": label_initial,
        "label_col": LabelCols.label_col,
        "label_constit_1_col": LabelCols.label_constit_1_col,
        "label_constit_all_col": LabelCols.label_constit_all_col,
        "direct_label_col": LabelCols.direct_label_col,
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
        "label_flank_col_set": LabelCols.label_flank_col_set,
        "label_flank_constit_1_col": LabelCols.label_flank_constit_1_col,
        "label_flank_constit_all_col": LabelCols.label_flank_constit_all_col,
        "label_conf_col": LabelCols.label_conf_col,
        "label_confs": LabelCols.label_confs,
        "best_label_ref_col": LabelCols.best_label_ref_col,

        # Parameters: Reference / Direct
        "ref_tf_col": TFAnnotationCols.ref_tf_col,
        "ref_family_col": TFAnnotationCols.ref_family_col,
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
        "nonznf_ref_readable_col": ReferenceMatchCols.nonznf_ref_readable_col,
        "nonznf_ref_col": ReferenceMatchCols.nonznf_ref_col,
        "min_single_score": MotifMatchArgs.min_single_score,
        "min_composite_score": MotifMatchArgs.min_composite_score,
        "max_submotifs": MotifMatchArgs.max_submotifs,
        "min_composite_subunit": MotifMatchArgs.min_composite_subunit,
        "direct_types": DirectTypeArgs.direct_types,

        # Parameters: Clustering
        "cluster_algorithm_main": ClusterArgs.cluster_algorithm_main,
        "cluster_algorithm_force": ClusterArgs.cluster_algorithm_force,
        "cluster_algorithm_kmeans": ClusterArgs.cluster_algorithm_kmeans,
        "cluster_sim_threshold": ClusterArgs.cluster_sim_threshold,
        "cluster_sim_threshold_force": ClusterArgs.cluster_sim_threshold_force,
        "cluster_force_density": ClusterArgs.cluster_force_density,
        "cluster_sim_assignment_threshold_kmeans": ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        "atlascluster_within": ClusterArgs.atlascluster_within,
        "atlascluster_cluster_col": atlascluster_cluster_col,
        "cluster_max_iter_main": ClusterArgs.cluster_max_iter_main,
        "cluster_max_iter_force": ClusterArgs.cluster_max_iter_force,
        "average_weight_col": ClusterArgs.average_weight_col,
        "source_cluster_col": ClusterArgs.source_cluster_col,
        "n_iterations": ClusterArgs.n_iterations,
        "seeds": ClusterArgs.seeds,

        # Parameters: Filtering
        "filter_flag_col": FilterArgs.filter_flag_col,
        "filter_flag_type_col": FilterArgs.filter_flag_type_col,
        "filter_trim_importance": FilterArgs.filter_trim_importance,
        "filter_trim_length": FilterArgs.filter_trim_length,
        "filter_ic_scale": FilterArgs.filter_ic_scale,
        "filter_flag_types": FilterArgs.filter_flag_types,
        "motif_string_col": VisualizeArgs.motif_string_col,
        "motif_string_importance": VisualizeArgs.motif_string_importance,
        "motif_string_specificity": VisualizeArgs.motif_string_specificity,

        # Parameters: Hierarchy
        "cluster_name_col": VisualizeArgs.cluster_name_col,
        "metacluster_name_col": VisualizeArgs.metacluster_name_col,
        "atlascluster_name_col": VisualizeArgs.atlascluster_name_col,
        "sort_col": VisualizeArgs.sort_col,
        "max_major_family_frac": VisualizeArgs.max_major_family_frac,
        "major_family_increment": VisualizeArgs.major_family_increment,
        "max_major_family_count": VisualizeArgs.max_major_family_count,

        # Parameters: HTML
        "html_logo_trim": VisualizeArgs.html_logo_trim,

        # Aggregation
        "aggregations": ClusterArgs.aggregations,
        "weighted_aggregations": ClusterArgs.weighted_aggregations,

        # Motif index
        "motif_index_cols": VisualizeArgs.motif_index_cols_atlascluster,

        # Run log
        "log_path": log_path,
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


#### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()

    set_default_configs()

    ### OPTIONS -----------------------------------------------------------------
    # Pipeline-level logging (stdout only); each stage body sets up the same logging again
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Stages requested, kept in pipeline order
    selected = [s for s in STAGE_ORDER if s in set(args.stages)]

    os.makedirs(args.output_dir, exist_ok=True)
    DIRS = stage_dirs(args.output_dir)

    if args.update_constituents == "only":
        logging.info("BPNet pipeline: the update of the levels below the atlasclusters only (--update-constituents only).")
        run_tf_constituents(stage_args(args, output_dir=DIRS["combine"], counts_dir=DIRS["counts"], profile_dir=DIRS["profile"]))
        sys.exit(0)

    logging.info("BPNet pipeline: run started.")
    logging.info(f"Output root: {args.output_dir}")
    logging.info(f"Stages to run ({len(selected)}/{len(STAGE_ORDER)}): {', '.join(selected)}")

    ### RUN ---------------------------------------------------------------------
    for stage in selected:
        stage_dir = DIRS[stage]
        logging.info("=" * 78)
        logging.info(f"STAGE {STAGE_ORDER.index(stage) + 1}/{len(STAGE_ORDER)}: {STAGES[stage]}")
        logging.info(f"  output: {stage_dir}")
        logging.info("=" * 78)
        if args.time:
            stage_start = time.time()

        os.makedirs(stage_dir, exist_ok=True)

        # --- 1 & 2: per-head sequential clustering ----------------------------
        if stage in ("counts", "profile"):
            sa = stage_args(
                args,
                output_dir=stage_dir,
                head_type=stage,
                input_mc=getattr(args, f"{stage}_input_mc"),
                input_modisco_h5s=getattr(args, f"{stage}_modisco_h5s"),
                input_names=getattr(args, f"{stage}_input_names"),
                input_pfms=None,
                metadata=getattr(args, f"{stage}_metadata"),
                finemo_paths=getattr(args, f"{stage}_finemo_paths"),
                peaks_paths=getattr(args, f"{stage}_peaks_paths"),
            )
            run_tf_sequential(sa)

        # --- 3: combine the two heads into atlasclusters ----------------------
        else:
            sa = stage_args(
                args,
                output_dir=stage_dir,
                counts_dir=DIRS["counts"],
                profile_dir=DIRS["profile"],
            )
            run_tf_combine(sa)

        if args.time:
            logging.info(f"Stage '{stage}' time taken: {time.time() - stage_start:.2f}s")

    logging.info("=" * 78)
    logging.info(f'BPNet pipeline: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
