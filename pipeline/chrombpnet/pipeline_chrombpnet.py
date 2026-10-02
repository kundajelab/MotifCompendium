#!/usr/bin/env python
# coding: utf-8
"""
Sequential hierarchical clustering of the Chromatin Atlas ENCODE motif compendium:
per-cell-type clustering of motifs, followed by a second clustering pass over
cell-type clusters into meta-clusters, with reference-based labeling and
MotifCompendium/HTML/MEME export at each level.
"""
import os
import re
import sys
import time

import argparse
import logging
import json
import dataclasses
import h5py

import numpy as np
import pandas as pd

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config

import configs


#### ARGUMENTS ------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(description="Pipeline for a collection of DNase/ATAC-seq ChromBPNet models.")

    # Run type
    parser.add_argument("-vr", "--mcid-version", type=str, default=None, required=True, help="The version to use in the unique identifier (e.g., V1).")
    parser.add_argument("-hd", "--head-type", type=str, default=None, required=True, help="The head type to use (options: counts, profile).")

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")

    # Input
    parser.add_argument("-im", "--input-mc", type=str, default=None, help="Path to the input MotifCompendium object.")
    parser.add_argument("-ih", "--input-modisco-h5s", nargs="+", type=str, default=None, help="Path(s) to the input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-sh", "--input-subpatterns", action="store_true", help="Use subpatterns (instead of main patterns) when loading input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-ip", "--input-pfms", nargs="+", type=str, default=None, help="Path(s) to the input motif PFM files, in PFM or MEME format.")
    parser.add_argument("-in", "--input-names", nargs="+", type=str, default=None, help="Nickname(s) of the input file(s): TF-MoDISco (lite) H5, PFMs, MEME.")

    # Reference databases
    parser.add_argument("--ref-qc", type=str, required=True, help="Path to the ENCODE QC reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--ref-clustered-hr-metadata", type=str, required=True, help="Path to the metadata TSV for --mc-ref-clustered-hr.")

    # Metadata
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, per input file or motif: CSV, TSV format.")
    parser.add_argument("--column-desc-tsv", type=str, required=True, help="Path to the TSV describing HTML summary table column names.")

    # Compute
    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("--var-chunk", action="store_true", help="Starting from max_chunk, dynamically reduce chunk size, based on available CPU/GPU memory.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Do not check the validity of the MotifCompendium objects.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not scale motifs by information content.")
    parser.add_argument("--fast-plot", action="store_true", help="Reduce the resolution of logo images for faster plotting.")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ------------------------------------------------------------
def set_default_configs():
    global MotifCompendiumArgs, TFMoDIScoCols, MetadataCols, CellTypeCols, ReferenceMatchCols, LabelCols, MotifMatchArgs
    global DirectTypeArgs, FilterArgs, MotifFilterArgs, ClusterArgs, VisualizeArgs

    MotifCompendiumArgs = configs.MotifCompendiumArgs()
    TFMoDIScoCols = configs.TFMoDIScoCols()
    MetadataCols = configs.MetadataCols()
    CellTypeCols = configs.CellTypeCols()
    ReferenceMatchCols = configs.ReferenceMatchCols()
    LabelCols = configs.LabelCols()
    MotifMatchArgs = configs.MotifMatchArgs()
    DirectTypeArgs = configs.DirectTypeArgs()
    FilterArgs = configs.FilterArgs()
    MotifFilterArgs = configs.MotifFilterArgs()
    ClusterArgs = configs.ClusterArgs()
    VisualizeArgs = configs.VisualizeArgs()


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

# %%


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
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply motif filters to the MotifCompendium object.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs): The filter arguments.
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
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """
    Apply motif filters to the MotifCompendium object.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        motif_filter_args (configs.MotifFilterArgs): The filter arguments.
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
    for override_filter_args in motif_filter_args.override_filters:
        if override_filter_args.apply_motif:
            logging.info(f"Applying override filter: {override_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=override_filter_args.metric,
                operation=override_filter_args.operation,
                threshold=override_filter_args.threshold,
                invert=override_filter_args.invert,
                override=override_filter_args.override,
            )
    
    # Filter #3: Hard filters
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
        filter_flag_type_col (str): The column name to use for the filter flag type.
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

    # Filter #2: Override: Any match
    for override_filter_args in motif_filter_args.override_filters:
        if override_filter_args.apply_cluster:
            logging.info(f"Applying cluster override filter: {override_filter_args.name}")
            apply_filter_threshold(
                mc=mc,
                flag_col=filter_flag_col,
                flag_type_col=filter_flag_type_col,
                metric=override_filter_args.metric,
                operation=override_filter_args.operation,
                threshold=override_filter_args.threshold,
                invert=override_filter_args.invert,
                override=override_filter_args.override,
            )

    # Filter #3: Apply hard filters
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


#### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()

    set_default_configs()

    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    # A run log is ALWAYS written into <output-dir>/logs/, regardless of how this
    # script was launched, so a finished run is always auditable. Previously the log
    # existed only if the caller redirected stdout, so interactively-started runs
    # left no record at all (see CHANGELOG 2026-09-09).
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_{os.path.basename(os.path.normpath(args.output_dir))}"
        f"_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.out",
    )
    logging.basicConfig(
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
    cluster_cluster_col = f"Within-{ClusterArgs.cluster_within}_{ClusterArgs.cluster_algorithm_main}-{ClusterArgs.cluster_sim_threshold}_rec_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}"
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
    mc_cluster_init_path = os.path.join(mc_dir, "motifcompendium_cluster_init.mc")
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

        # Merge metadata
        n_motifs_premerge = len(mc)
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

        # Guard: the "per model" / "per source" branches above merge how="left" on a
        # right key that is not checked for uniqueness, so a metadata file with
        # repeated keys would silently DUPLICATE motif rows and corrupt every
        # downstream count. run_sequential_cluster-celltype.py has no merge at all
        # (its metadata arrives baked into the init .mc), so this hazard is specific
        # to this script - fail loudly instead of clustering a corrupted set.
        if len(mc.metadata) != n_motifs_premerge:
            logging.critical(
                f"Metadata merge changed the number of motifs: "
                f"{n_motifs_premerge} -> {len(mc.metadata)}. "
                f"The metadata key column ('{metadata_df.columns[0]}') is most likely "
                f"not unique in {args.metadata}."
            )
            raise ValueError(
                f"Metadata merge changed the number of motifs "
                f"({n_motifs_premerge} -> {len(mc.metadata)})."
            )

        # Guard: a mis-keyed merge leaves the clustering boundary column all-NaN,
        # which silently collapses the within-cell-type clustering groups.
        if ClusterArgs.cluster_within in mc.metadata.columns:
            n_missing_within = int(mc.metadata[ClusterArgs.cluster_within].isna().sum())
            if n_missing_within:
                logging.warning(
                    f"{n_missing_within} / {len(mc.metadata)} motifs have no "
                    f"'{ClusterArgs.cluster_within}' value after the metadata merge; "
                    f"clustering within '{ClusterArgs.cluster_within}' will treat these "
                    f"as a single group."
                )

        if args.verbose:
            logging.info(f"Metadata columns: {mc.columns()}")

    ## Rename motifs: Original TF-Modisco motif name
    # From scratch
    mc[MetadataCols.name_col] = (
        mc[MetadataCols.posneg_col]
        + "_" + "patterns."
        + mc[MetadataCols.id_col]
        + "_" + mc[MetadataCols.assay_col]
        + "_" + mc[MetadataCols.biosample_col].str.replace(" ", "-")
        + "_" + mc[MetadataCols.accession_col]
        + "_" + mc[MetadataCols.head_col]
        + "_" + "pattern"
        + "_" + mc[MetadataCols.name_col].str.split("_").str[-1]
    )

    # E.g., 
    # From: ENCSR155NPL_counts-pos.ENCSR155NPL_DNASE_renal-cortex-interstitium_ENCSR596EZJ_counts_pattern_0
    # To: pos_patterns.ENCSR155NPL_DNASE_renal-cortex-interstitium_ENCSR596EZJ_counts_pattern_0

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

    # Remove multimers
    mc_ref_qc = mc_ref_qc[~mc_ref_qc[MetadataCols.name_col].str.contains('::')]

    # IC scale
    mc_ref_qc.motifs = utils_motif.ic_scale(mc_ref_qc.motifs)

    # Annotate motifs: Human-readable
    mc_ref_clustered_hr_metadata = pd.read_csv(args.ref_clustered_hr_metadata, sep="\t")

    mc_ref_clustered_hr_metadata[MetadataCols.name_col] = mc_ref_clustered_hr_metadata[MetadataCols.name_col].str.split("_").str[0]  # Remove "_{#}"
    mc_ref_clustered_hr_metadata['motifs_name'] = mc_ref_clustered_hr_metadata['motifs'].str.split(",")
    mc_ref_clustered_hr_metadata_expand = mc_ref_clustered_hr_metadata.explode("motifs_name")

    mc_ref_qc_hr_dict = dict(zip(mc_ref_clustered_hr_metadata_expand['motifs_name'], mc_ref_clustered_hr_metadata_expand[MetadataCols.name_col]))
    mc_ref_qc_tf_dict = dict(zip(mc_ref_qc[MetadataCols.name_col], mc_ref_qc[ReferenceMatchCols.ref_tf_col]))

    mc_ref_qc[LabelCols.label_col] = mc_ref_qc[MetadataCols.name_col].map(mc_ref_qc_hr_dict)


    # #### Remove: Initial
    # - Initial remove: Pos/neg inverted, truncated
    # Apply filters
    apply_filter_initial(
        mc=mc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Removed
    mc_removed = mc[mc[FilterArgs.filter_flag_col]]
    mc_qc = mc[~mc[FilterArgs.filter_flag_col]]

    # ### Annotate: Motifs
    logging.info("Annotating motifs against the reference...")
    ## Initialize columns
    mc_qc.metadata = mc_qc.metadata.assign(**{
        MetadataCols.num_subunit_col:            1,
        MetadataCols.composite_col:              False,
        LabelCols.label_core_col:             None,
        LabelCols.label_core_readable_col:    None,
        LabelCols.label_flank_col:            None,
    }).copy()

    ## [General] Label motifs
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_qc,
        assign_from_mc=mc_ref_qc,
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
    mask_monomer_mc = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_mc = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_mc = (
        (mc_qc[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_qc[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_mc = (
        mask_dimer_mc | mask_trimer_mc
    )
    mask_matched_mc = (
        mask_monomer_mc | mask_dimer_mc | mask_trimer_mc
    )

    # Dimer-style notation
    mc_qc.metadata.loc[mask_matched_mc, LabelCols.label_core_col] = mc_qc.metadata.loc[mask_matched_mc, ref_readable_name_cols[0]]
    mc_qc.metadata.loc[mask_matched_mc, LabelCols.label_core_readable_col] = mc_qc.metadata.loc[mask_matched_mc, ref_readable_name_cols[0]]
    mc_qc.metadata.loc[mask_dimer_mc, LabelCols.label_flank_col] = mc_qc.metadata.loc[mask_dimer_mc, ref_readable_name_cols[1]]
    mc_qc.metadata.loc[mask_trimer_mc, LabelCols.label_flank_col] = mc_qc.metadata.loc[mask_trimer_mc, ref_readable_name_cols[1]] + "::" + mc_qc.metadata.loc[mask_trimer_mc, ref_readable_name_cols[2]]
    mc_qc.metadata.loc[mask_dimer_mc, MetadataCols.num_subunit_col] = 2
    mc_qc.metadata.loc[mask_trimer_mc, MetadataCols.num_subunit_col] = 3
    mc_qc.metadata[MetadataCols.composite_col] = mc_qc.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit  # Composite: True, False


    # Concatenate all unique annotations
    mc_qc[ReferenceMatchCols.ref_name_col] = mc_qc.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

    # ### Filter: Motifs
    logging.info("Filtering motifs...")
    # Apply filters
    apply_filter_motifs(
        mc=mc_qc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Removed, Filtered
    mc_removed = mc[mc[MetadataCols.name_col].isin(
        set(mc_qc[mc_qc[FilterArgs.filter_flag_col]][MetadataCols.name_col])
        | set(mc_removed[MetadataCols.name_col])
    )]
    mc_removed[FilterArgs.filter_flag_col] = True
    mc_filtered = mc_qc[~mc_qc[FilterArgs.filter_flag_col]]

    # Annotate
    mc_filtered[DirectTypeArgs.direct_type_col] = DirectTypeArgs.direct_types[4]  # Indirect: X (including unvalidated)

    # ## Clustering: Motifs
    logging.info("Clustering motifs, within cell type...")
    # - Process:
    #     - Cluster within boundary: `celltype`
    #     - Initial clustering
    #     - Recursive clustering
    #     - Force clustering

    ## Cluster MotifCompendium
    # Cluster: Leiden CPM
    mc_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        cluster_within=ClusterArgs.cluster_within,
        save_name=cluster_cluster_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )
    # Reassign: K-means
    mc_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        cluster_within=ClusterArgs.cluster_within,
        save_name=cluster_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=cluster_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )

    membership_cluster = mc_filtered[cluster_cluster_col]
    logging.info(f"Unique clusters (Indirect): {len(set(membership_cluster))}")


    # Recursively cluster:
    for i in range(ClusterArgs.cluster_max_iter_main):
        logging.info(f"Recursively clustering: {i}")
        # Cluster: CPM
        mc_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within_on=(ClusterArgs.cluster_within, cluster_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=cluster_cluster_col,
            largest_clusters_first=True,
            n_iterations=ClusterArgs.n_iterations,
            seeds=ClusterArgs.seeds,
        )
        # Reassign: K-means
        mc_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.cluster_within,
            save_name=cluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=cluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_filtered[cluster_cluster_col]
        logging.info(f"  Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            logging.info(f"Final unique clusters (Indirect): {len(set(new_membership_cluster))}")
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
        logging.info(f"Recursively clustering: {i}")
        # Cluster: DCC
        mc_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.cluster_within, cluster_cluster_col),
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=cluster_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-means
        mc_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            cluster_within=ClusterArgs.cluster_within,
            save_name=cluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=cluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_filtered[cluster_cluster_col]
        logging.info(f"  Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            logging.info(f"Final unique clusters (Indirect): {len(set(new_membership_cluster))}")
            break
        else:
            membership_cluster = new_membership_cluster


    # # CLUSTERS
    # ## Create clusters, by averaging
    logging.info("Creating clusters, by averaging...")
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

    # ### Calculations: Cluster
    mc_cluster[MetadataCols.total_contrib_score_ratio_col] = mc_cluster[MetadataCols.total_contrib_score_col] / mc_cluster[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_cluster.metadata[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_cluster.get_standard_motif_stack())

    mc_cluster.sort(by=[MetadataCols.composite_col, MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_cluster.sort(by=[MetadataCols.composite_col], ascending=True, inplace=True)  # Sort by composite
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
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1

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


    # ### Annotate: Clusters
    logging.info("Annotating clusters against the reference...")
    # Initialize columns
    mc_cluster.metadata = mc_cluster.metadata.assign(**{
        MetadataCols.num_subunit_col:            1,
        MetadataCols.composite_col:              False,
        LabelCols.label_col:                  MetadataCols.name_unknown,
        LabelCols.label_core_col:             None,
        LabelCols.label_core_readable_col:    None,
        LabelCols.label_flank_col:            None,
    }).copy()


    # [General] Label clusters
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

    ## Annotate: Centroids
    # Map human-readable name
    ref_readable_name_cols = [f'{ReferenceMatchCols.ref_readable_col}_name{i}' for i in range(MotifMatchArgs.max_submotifs)]
    ref_score_cols = [f'{ReferenceMatchCols.ref_col}_score{i}' for i in range(MotifMatchArgs.max_submotifs)]
    for i, (ref_readable_name_col, ref_score_col) in enumerate(zip(ref_readable_name_cols, ref_score_cols)):    
        mc_cluster[ref_readable_name_col] = mc_cluster[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_monomer_cluster = (
        (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_cluster = (
        (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_cluster = (
        (mc_cluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_cluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_cluster = (
        mask_dimer_cluster | mask_trimer_cluster
    )
    mask_matched_cluster = (
        mask_monomer_cluster | mask_dimer_cluster | mask_trimer_cluster
    )

    # Dimer-style notation
    mc_cluster.metadata.loc[mask_matched_cluster, LabelCols.label_core_col] = mc_cluster.metadata.loc[mask_matched_cluster, ref_readable_name_cols[0]]
    mc_cluster.metadata.loc[mask_matched_cluster, LabelCols.label_core_readable_col] = mc_cluster.metadata.loc[mask_matched_cluster, ref_readable_name_cols[0]]
    mc_cluster.metadata.loc[mask_dimer_cluster, LabelCols.label_flank_col] = mc_cluster.metadata.loc[mask_dimer_cluster, ref_readable_name_cols[1]]
    mc_cluster.metadata.loc[mask_trimer_cluster, LabelCols.label_flank_col] = mc_cluster.metadata.loc[mask_trimer_cluster, ref_readable_name_cols[1]] + "::" + mc_cluster.metadata.loc[mask_trimer_cluster, ref_readable_name_cols[2]]
    mc_cluster.metadata.loc[mask_dimer_cluster, MetadataCols.num_subunit_col] = 2
    mc_cluster.metadata.loc[mask_trimer_cluster, MetadataCols.num_subunit_col] = 3
    mc_cluster.metadata[MetadataCols.composite_col] = mc_cluster.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit  # Composite: True, False


    # Concatenate all unique annotations
    mc_cluster[ReferenceMatchCols.ref_name_col] = mc_cluster.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

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

    # ### Filter: Clusters
    logging.info("Filtering clusters...")
    # Apply filters
    apply_filter_clusters(
        mc=mc_cluster,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Remove flagged clusters
    mc_cluster_removed = mc_cluster[mc_cluster[FilterArgs.filter_flag_col]]
    mc_cluster_filtered = mc_cluster[~mc_cluster[FilterArgs.filter_flag_col]]

    ## Cluster label: {cell_type}_X{score}_{tf_core}::{tf_composite}
    # Boolean conditions
    ref_score_cols = [f"{ReferenceMatchCols.ref_col}_score{iter}" for iter in range(MotifMatchArgs.max_submotifs)]
    composite_score_cols = [f"{ReferenceMatchCols.ref_col}_score{iter}" for iter in range(1, MotifMatchArgs.max_submotifs)]

    mask_indirect = mc_cluster_filtered[LabelCols.label_core_col].notna()
    mask_composite = mc_cluster_filtered[MetadataCols.composite_col]
    mask_has_composite = mc_cluster_filtered[LabelCols.label_flank_col].notna()

    # Filtered: Assign readable label
    mc_cluster_filtered.metadata[DirectTypeArgs.direct_type_col] = DirectTypeArgs.direct_types[4]  # Indirect: X (including unvalidated)
    mc_cluster_filtered[LabelCols.label_col] = (
        np.where(
            mask_indirect,
            mc_cluster_filtered[LabelCols.label_core_col],
            ""
        )
        + np.where(
            mask_indirect & mask_composite & mask_has_composite,
            "::" + mc_cluster_filtered[LabelCols.label_flank_col],
            ""
        )
    )
    mc_cluster_filtered[MetadataCols.prename_col] = (
        label_initial
        + "-"
        + mc_cluster_filtered[CellTypeCols.celltype_col]
        + "_" + mc_cluster_filtered[DirectTypeArgs.direct_type_col]
        + (mc_cluster_filtered[ref_score_cols].fillna(0).max(axis=1) * 100).astype(int).astype(str)
        + "_"
        + mc_cluster_filtered[LabelCols.label_col]
    )
    mc_cluster_filtered[MetadataCols.simple_name_col] = mc_cluster_filtered[LabelCols.label_col]

    mc_cluster_filtered.metadata.loc[mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[2]
    mc_cluster_filtered.metadata.loc[~mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    ## Removed: Assign readable label
    mc_cluster_removed[LabelCols.label_col] = mc_cluster_removed[FilterArgs.filter_flag_type_col]
    mc_cluster_removed[MetadataCols.prename_col] = (
        label_initial
        + "-"
        + mc_cluster_removed[CellTypeCols.celltype_col]
        + "_" 
        + mc_cluster_removed[LabelCols.label_col]
    )
    mc_cluster_removed[MetadataCols.simple_name_col] = mc_cluster_removed[LabelCols.label_col]
    mc_cluster_removed[LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    ### Annotate: Best label
    mc_cluster_filtered_metadata_updates = []
    mc_cluster_filtered_images_updated = pd.DataFrame(index=mc_cluster_filtered[MetadataCols.unique_id_col])
    mc_cluster_filtered_images_existing = set(mc_cluster_filtered.images())

    for i, row in mc_cluster_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_cluster_filtered_i = mc_cluster_filtered[[i]]
        label_core_i = row[LabelCols.label_core_col].split("-") if pd.notna(row[LabelCols.label_core_col]) else []
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1

        ## [General] Reference annotations
        # Reference: Subset
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


    # Save MotifCompendium: Cluster
    mc_cluster_filtered.save(mc_cluster_init_path)


    # ## META-CLUSTERS
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
        logging.info(f"  Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            logging.info(f"Final unique clusters (Indirect): {len(set(new_membership_cluster))}")
            break
        else:
            membership_cluster = new_membership_cluster


    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter_force):
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
        logging.info(f"  Unique clusters (Indirect): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            logging.info(f"Final unique clusters (Indirect): {len(set(new_membership_cluster))}")
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
    # Calculations
    mc_metacluster[MetadataCols.total_contrib_score_ratio_col] = mc_metacluster[MetadataCols.total_contrib_score_col] / mc_metacluster[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_metacluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_metacluster.get_standard_motif_stack())  # Positive / negative
    mc_metacluster[MetadataCols.composite_col] = mc_metacluster[MetadataCols.composite_col].astype(bool)

    mc_metacluster.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_metacluster.sort(by=[MetadataCols.composite_col], ascending=True, inplace=True)  # Sort by composite
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
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1

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
    # Initialize columns
    # Experiment: Determined by motif; Composite: Determined by cluster
    mc_metacluster.metadata = mc_metacluster.metadata.assign(**{
        LabelCols.label_col:                  MetadataCols.name_unknown,
        LabelCols.label_core_col:             None,
        LabelCols.label_core_readable_col:    None,
        LabelCols.label_flank_col:            None,
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

    ## Annotate: Centroids
    # Map human-readable name
    ref_readable_name_cols = [f'{ReferenceMatchCols.ref_readable_col}_name{i}' for i in range(MotifMatchArgs.max_submotifs)]
    ref_score_cols = [f'{ReferenceMatchCols.ref_col}_score{i}' for i in range(MotifMatchArgs.max_submotifs)]
    for i, (ref_readable_name_col, ref_score_col) in enumerate(zip(ref_readable_name_cols, ref_score_cols)):    
        mc_metacluster[ref_readable_name_col] = mc_metacluster[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_monomer_metacluster = (
        (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_metacluster = (
        (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_metacluster = (
        (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_metacluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_metacluster = (
        mask_dimer_metacluster | mask_trimer_metacluster
    )
    mask_matched_metacluster = (
        mask_monomer_metacluster | mask_dimer_metacluster | mask_trimer_metacluster
    )

    # Dimer-style notation
    mc_metacluster.metadata.loc[mask_matched_metacluster, LabelCols.label_core_col] = mc_metacluster.metadata.loc[mask_matched_metacluster, ref_readable_name_cols[0]]
    mc_metacluster.metadata.loc[mask_matched_metacluster, LabelCols.label_core_readable_col] = mc_metacluster.metadata.loc[mask_matched_metacluster, ref_readable_name_cols[0]]
    mc_metacluster.metadata.loc[mask_dimer_metacluster, LabelCols.label_flank_col] = mc_metacluster.metadata.loc[mask_dimer_metacluster, ref_readable_name_cols[1]]
    mc_metacluster.metadata.loc[mask_trimer_metacluster, LabelCols.label_flank_col] = mc_metacluster.metadata.loc[mask_trimer_metacluster, ref_readable_name_cols[1]] + "::" + mc_metacluster.metadata.loc[mask_trimer_metacluster, ref_readable_name_cols[2]]
    # mc_metacluster.metadata.loc[mask_dimer_metacluster, MetadataCols.num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_metacluster.metadata.loc[mask_trimer_metacluster, MetadataCols.num_subunit_col] = 3
    # mc_metacluster.metadata[MetadataCols.composite_col] = mc_metacluster.metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit


    # Concatenate all unique annotations
    mc_metacluster[ReferenceMatchCols.ref_name_col] = mc_metacluster.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

    # #### Filter: Meta-clusters
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
    mc_metacluster_removed = mc_metacluster[mc_metacluster[FilterArgs.filter_flag_col]]
    mc_metacluster_filtered = mc_metacluster[~mc_metacluster[FilterArgs.filter_flag_col]]


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
    # Count occurrence: Core TF family
    mc_cluster_filtered[LabelCols.label_core_readable_col_set] = mc_cluster_filtered[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_cluster_explode_core_readable = mc_cluster_filtered.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_cluster_filtered[LabelCols.label_flank_col_set] = mc_cluster_filtered[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_cluster_explode_readable_composite = mc_cluster_filtered.metadata.explode(LabelCols.label_flank_col_set)

    ## Count occurrences: Create pivot table
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
    mc_cluster_core_readable_pivot = mc_cluster_core_readable_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_cluster_flank_pivot = mc_cluster_flank_pivot.reindex(
        mc_metacluster_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')


    def is_valid(x):
        """Pandas-safe validity check"""
        return pd.notna(x) and x != ""

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

    def assign_label(row):
        # Initialize
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col]) or is_valid(row[LabelCols.label_flank_constit_all_col])  # Centroid or constituent flank
        
        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_core_readable = row[LabelCols.label_core_readable_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]

        
        ## Rule 2: Indirect, with annotated centroid: Use centroid's core
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(row_label_core)
            and is_valid(row_label_core_readable)
        ):
            label_core = str(row_label_core)
            label_core_readable = str(row_label_core_readable)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = row_label_flank if is_valid(row_label_flank) else assign_label_flank(row)  # Centroid flank, else constituent flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
                else:
                    label_combined = str(label_core)  # Flank unresolved: Core only
            elif is_valid(label_core):
                label_combined = str(label_core)
        
        ## Rule 3: Indirect, with annotated constituents: Use constituent's annotations
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(row_label_core_constituents)
        ):
            label_core, label_core_readable = assign_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
                else:
                    label_combined = str(label_core)  # Flank unresolved: Core only
            elif is_valid(label_core):
                label_combined = str(label_core)
        
        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown
        
        return label_core, label_core_readable, label_flank, label_combined


    # Add human-readable label
    mc_metacluster_filtered.metadata[[LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col]] = pd.DataFrame(
        mc_metacluster_filtered.metadata.apply(assign_label, axis=1).tolist(),
        index=mc_metacluster_filtered.metadata.index
    )

    ### Name: Metacluster - Removed
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
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1

        ## [General] Reference annotations
        # Reference: Subset
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
    del mc_metacluster_filtered_metadata_updates_df, mc_metacluster_filtered_metadata_updates

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


    # Readable label: Type
    mask_indirect = ~mc_metacluster[LabelCols.label_col].isin(FilterArgs.filter_flag_types)
    mc_metacluster.metadata.loc[mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[2]
    mc_metacluster.metadata.loc[~mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[-1]


    # Rename: {readable_name}_{counter}
    mc_metacluster.metadata[MetadataCols.counter_col] = mc_metacluster.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_metacluster.metadata[MetadataCols.name_new_col] = mc_metacluster.metadata[MetadataCols.prename_col] + "_" + mc_metacluster.metadata[MetadataCols.counter_col].astype(str)
    mc_metacluster.metadata[MetadataCols.name_col] = mc_metacluster.metadata[MetadataCols.name_new_col]
    mc_metacluster.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    ## Map: Cluster to Metacluster: Name, Simple name
    cluster2metacluster_metacluster_map = dict(zip(mc_metacluster[ClusterArgs.source_cluster_col], mc_metacluster[MetadataCols.name_col]))
    cluster2metacluster_simple_map = dict(zip(mc_metacluster[ClusterArgs.source_cluster_col], mc_metacluster[MetadataCols.simple_name_col]))

    mc_cluster_filtered[VisualizeArgs.metacluster_name_col] = mc_cluster_filtered[metacluster_cluster_col].map(cluster2metacluster_metacluster_map)
    mc_cluster_filtered[MetadataCols.simple_name_col] = mc_cluster_filtered[metacluster_cluster_col].map(cluster2metacluster_simple_map)

    ### (2) Cluster:
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
                index=mc_cluster_i.metadata[MetadataCols.unique_id_col],
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


    # Rename: {readable_name}_{counter}
    mc_cluster.metadata[MetadataCols.counter_col] = mc_cluster.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
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
    metadata_cols = []
    for mc_i in mcs:
        metadata_cols = list(set(metadata_cols + mc_i.columns()))

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
    # slice the filtered / removed views, and resolve the HTML columns once.
    logging.info("Preparing final MotifCompendium objects...")

    ## (3) Metaclusters
    mc_metacluster_filtered = mc_metacluster[~mc_metacluster[FilterArgs.filter_flag_col]]
    mc_metacluster_removed = mc_metacluster[mc_metacluster[FilterArgs.filter_flag_col]]

    ## (2) Clusters: Sort by metacluster
    mc_cluster[VisualizeArgs.sort_col] = mc_cluster[metacluster_cluster_col]
    mc_cluster_filtered = mc_cluster[~mc_cluster[FilterArgs.filter_flag_col]]
    mc_cluster_removed = mc_cluster[mc_cluster[FilterArgs.filter_flag_col]]

    ## (1) Motifs: Sort by cluster
    mc[VisualizeArgs.sort_col] = mc[cluster_cluster_col]
    mc_filtered = mc[~mc[FilterArgs.filter_flag_col]]
    mc_removed = mc[mc[FilterArgs.filter_flag_col]]

    # Removed motifs: Sample a subset, for the HTML summary table only
    mc_removed_sample = mc_removed[list(
        mc_removed.metadata.sample(n=VisualizeArgs.html_removed_sample_size,
                                   random_state=ClusterArgs.seeds[0]).index
    )]

    ## HTML summary table columns, per level
    html_columns_metacluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_metacluster.columns() + mc_metacluster.images())]
    html_column_desc_dict_metacluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_metacluster}

    html_columns_cluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_cluster.columns() + mc_cluster.images())]
    html_column_desc_dict_cluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_cluster}

    if args.verbose:
        logging.info(
            f"Final object sizes:\n"
            f"  Motifs:       {len(mc)} ({len(mc_filtered)} filtered, {len(mc_removed)} removed)\n"
            f"  Clusters:     {len(mc_cluster)} ({len(mc_cluster_filtered)} filtered, {len(mc_cluster_removed)} removed)\n"
            f"  Metaclusters: {len(mc_metacluster)} ({len(mc_metacluster_filtered)} filtered, {len(mc_metacluster_removed)} removed)"
        )


    #### EXPORT --------------------------------------------------------------------
    ## SAVE: MotifCompendium object
    logging.info(f"Saving MotifCompendium objects to: {mc_dir}...")
    os.makedirs(mc_dir, exist_ok=True)

    # (1) Motifs
    mc.save(mc_motif_path)
    mc_filtered.save(mc_motif_filtered_path)
    mc_removed.save(mc_motif_removed_path)

    # (2) Clusters
    mc_cluster.save(mc_cluster_path)
    mc_cluster_filtered.save(mc_cluster_filtered_path)
    mc_cluster_removed.save(mc_cluster_removed_path)

    # (3) Metaclusters
    mc_metacluster.save(mc_metacluster_path)
    mc_metacluster_filtered.save(mc_metacluster_filtered_path)
    mc_metacluster_removed.save(mc_metacluster_removed_path)


    ## HTML summary table
    logging.info(f"Saving HTML summary tables to: {html_dir}...")
    os.makedirs(html_dir, exist_ok=True)

    # (1) Motifs: Removed only (sampled subset)
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
    mc_cluster_removed.summary_table_html(
        html_out=html_cluster_removed_path,
        columns=[col for col in html_columns_cluster if col in (mc_cluster_removed.columns() + mc_cluster_removed.images())],
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
    mc_metacluster_removed.summary_table_html(
        html_out=html_metacluster_removed_path,
        columns=[col for col in html_columns_metacluster if col in (mc_metacluster_removed.columns() + mc_metacluster_removed.images())],
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_metacluster,
    )


    ## EXPORT: MoDISco HDF5
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
    utils_analysis.export_compendium_modisco(
        mc=mc_metacluster_removed,
        name_col=MetadataCols.name_col,
        save_loc=modisco_metacluster_removed_path,
    )


    ## EXPORT: MEME txt
    logging.info(f"Saving MEME txt files to: {export_dir}...")

    # (1) Motifs
    utils_analysis.export_compendium_meme(
        mc=mc,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_filtered_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_removed_path,
    )

    # (2) Clusters
    utils_analysis.export_compendium_meme(
        mc=mc_cluster,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_cluster_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_filtered_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_cluster_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_removed_path,
    )

    # (3) Metaclusters
    utils_analysis.export_compendium_meme(
        mc=mc_metacluster,
        name_col=MetadataCols.name_col,
        save_loc=meme_metacluster_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_metacluster_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_metacluster_filtered_path,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_metacluster_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_metacluster_removed_path,
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
        "unique_id_col": MetadataCols.unique_id_col,
        "unique_id_initial": unique_id_initial,
        
        # Experiment
        "atlas_type": MetadataCols.atlas_type,
        "id_col": MetadataCols.id_col,
        "assay_col": MetadataCols.assay_col,
        "model_col": MetadataCols.model_col,
        "head_col": MetadataCols.head_col,
        "file_col": MetadataCols.file_col,
        "accession_col": MetadataCols.accession_col,
        "biosample_col": MetadataCols.biosample_col,

        # Annotation
        "label_initial": label_initial,
        "label_col": LabelCols.label_col,
        "label_constit_1_col": LabelCols.label_constit_1_col,
        "label_constit_all_col": LabelCols.label_constit_all_col,
        "label_core_col": LabelCols.label_core_col,
        "label_core_col_set": LabelCols.label_core_col_set,
        "label_core_constit_1_col": LabelCols.label_core_constit_1_col,
        "label_core_constit_all_col": LabelCols.label_core_constit_all_col,
        "label_flank_col": LabelCols.label_flank_col,
        "label_flank_col_set": LabelCols.label_flank_col_set,
        "label_flank_constit_1_col": LabelCols.label_flank_constit_1_col,
        "label_flank_constit_all_col": LabelCols.label_flank_constit_all_col,
        "label_conf_col": LabelCols.label_conf_col,
        "label_confs": LabelCols.label_confs,
        "best_label_ref_col": LabelCols.best_label_ref_col,

        # Composites
        "composite_col": MetadataCols.composite_col,
        "num_subunit_col": MetadataCols.num_subunit_col,
        
        # TF-MoDISco
        "num_seqlets_col": MetadataCols.num_seqlets_col,
        "num_constituents_col": MetadataCols.num_constituents_col,
        "posneg_col": MetadataCols.posneg_col,
        "avgcontrib_score_col": MetadataCols.avgcontrib_score_col,
        "total_contrib_score_col": MetadataCols.total_contrib_score_col,
        "group_total_contrib_score_col": MetadataCols.group_total_contrib_score_col,
        "total_contrib_score_ratio_col": MetadataCols.total_contrib_score_ratio_col,
        "avgdist_summit_col": MetadataCols.avgdist_summit_col,

        # Parameters: Cell type annotations
        "celltype_col": CellTypeCols.celltype_col,
        "organ_col": CellTypeCols.organ_col,
        "slim_cell_col": CellTypeCols.slim_cell_col,
        "slim_organ_col": CellTypeCols.slim_organ_col,
        "slim_dev_col": CellTypeCols.slim_dev_col,
        "slim_system_col": CellTypeCols.slim_system_col,

        # Parameters: Annotation / Reference
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
        "ref_znf_col": ReferenceMatchCols.ref_znf_col,
        "ref_znf_name_col": ReferenceMatchCols.ref_znf_name_col,
        "ref_znf_name_constit_1_col": ReferenceMatchCols.ref_znf_name_constit_1_col,
        "ref_znf_name_constit_all_col": ReferenceMatchCols.ref_znf_name_constit_all_col,
        "ref_nonznf_readable_col": ReferenceMatchCols.ref_nonznf_readable_col,
        "ref_nonznf_col": ReferenceMatchCols.ref_nonznf_col,
        "min_single_score": MotifMatchArgs.min_single_score,
        "min_composite_score": MotifMatchArgs.min_composite_score,
        "max_submotifs": MotifMatchArgs.max_submotifs,
        "min_composite_subunit": MotifMatchArgs.min_composite_subunit,
        "direct_type_col": DirectTypeArgs.direct_type_col,
        "direct_types": DirectTypeArgs.direct_types,

        # Parameters: Clustering
        "cluster_algorithm_main": ClusterArgs.cluster_algorithm_main,
        "cluster_algorithm_force": ClusterArgs.cluster_algorithm_force,
        "cluster_algorithm_kmeans": ClusterArgs.cluster_algorithm_kmeans,
        "cluster_sim_threshold": ClusterArgs.cluster_sim_threshold,
        "cluster_sim_threshold_force": ClusterArgs.cluster_sim_threshold_force,
        "cluster_force_density": ClusterArgs.cluster_force_density,
        "cluster_sim_assignment_threshold_kmeans": ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        "cluster_within": ClusterArgs.cluster_within,
        "metacluster_within": ClusterArgs.metacluster_within,
        "cluster_cluster_col": cluster_cluster_col,
        "metacluster_cluster_col": metacluster_cluster_col,
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
        "metacluster_name": VisualizeArgs.metacluster_name,
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
    }

    # Provenance: record exactly which inputs this run consumed and how big each
    # level came out. Without this, a finished run cannot be traced back to its
    # inputs - which is what made the v19-vs-v20 baseline mix-up expensive to
    # diagnose (see CHANGELOG 2026-09-09): the .mc outputs alone do not say which
    # init compendium / MoDISco H5 set produced them.
    _size_targets = [
        "mc", "mc_qc", "mc_filtered", "mc_removed",
        "mc_cluster", "mc_cluster_filtered", "mc_cluster_removed",
        "mc_metacluster", "mc_metacluster_filtered", "mc_metacluster_removed",
    ]
    _scope = locals()
    object_sizes = {}
    for _name in _size_targets:
        _obj = _scope.get(_name)
        if _obj is None:
            continue
        try:
            object_sizes[_name] = len(_obj)
        except TypeError:
            pass

    args_record["run"] = {
        "argv": sys.argv,
        "cli_args": {k: v for k, v in sorted(vars(args).items())},
        "input_mode": (
            "input_mc" if args.input_mc
            else "input_modisco_h5s" if args.input_modisco_h5s
            else "input_pfms" if args.input_pfms
            else None
        ),
        "n_input_files": (
            len(args.input_modisco_h5s) if args.input_modisco_h5s
            else len(args.input_pfms) if args.input_pfms
            else 1 if args.input_mc
            else 0
        ),
        "log_path": log_path,
        "object_sizes": object_sizes,
        "completed": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)


    logging.info(f'Run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')