#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: combine the TF, Chromatin and ProCap modelclusters into crossclusters.

Inputs are the output roots of the three atlas pipelines, each <dir>/{counts,profile,combined}:

  --tfatlas-dir      pipeline/bpnet/pipeline_bpnet.py            (TF atlas)
  --chromatlas-dir   pipeline/chrombpnet/pipeline_chrombpnet.py  (Chromatin atlas)
  --procapatlas-dir  pipeline/procapnet/pipeline_procapnet.py    (ProCap atlas)

The modelclusters are force-clustered (DCC + k-means, recursively) into crossclusters, which
are annotated, filtered, decomposed into reference monomers (composites) and exported; the
results are written back into the atlas pipelines' directories. The composites and their
subunits are written to --composite-dir, for the next stages:

  pipeline_encode_composite_ism.py  ISM cooperativity of each composite, per model
  pipeline_encode_composite.py      merge the composites by their cooperativity: the final
                                    ENCODE MotifCompendium, exported again to every level
  pipeline_encode_hits.py           FiNeMo hits indexed and annotated with the final crossclusters

pipeline_encode_full.py runs the atlas pipelines and these stages in one process.
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
import json
import dataclasses
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
    "configs_encode", os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.py")
)
configs = importlib.util.module_from_spec(_configs_spec)
sys.modules[_configs_spec.name] = configs
_configs_spec.loader.exec_module(configs)



#### ARGUMENTS ---------------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="ENCODE MotifCompendium: combine the TF, Chromatin and ProCap modelclusters into crossclusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory of the ENCODE combine stage.")
    parser.add_argument("--mcid-version", type=str, default="V0", help="ENCODE MCID version.")
    parser.add_argument("--update-constituents", choices=["now", "defer", "only"], default="now",
                        help="This stage's update of the levels below its own (every atlas level and the motif indexes: their crosscluster, final flags and confidence). now: by this stage; defer: not (they stay as they were, e.g. until the last level is made: run_pipeline_encode_propagate.sh); only: that update alone, from this stage's saved outputs (nothing else is run).")
    parser.add_argument("--save-all", action="store_true",
                        help="Also save every level's removed clusters (motifcompendium_*_removed: .mc, HTML, MEME, metadata "
                             "TSV); the _all and _filtered ones are always saved.")

    # Input: the atlas pipelines' output roots, each <dir>/{counts,profile,combined}
    parser.add_argument("--tfatlas-dir", type=str, required=True,
                        help="Output root of the TF atlas pipeline (pipeline/bpnet/pipeline_bpnet.py).")
    parser.add_argument("--chromatlas-dir", type=str, required=True,
                        help="Output root of the Chromatin atlas pipeline (pipeline/chrombpnet/pipeline_chrombpnet.py).")
    parser.add_argument("--procapatlas-dir", type=str, required=True,
                        help="Output root of the ProCap atlas pipeline (pipeline/procapnet/pipeline_procapnet.py).")

    # Output: composites (the inputs of pipeline_encode_composite_ism.py)
    parser.add_argument("--composite-dir", type=str, default=None,
                        help="Output directory of the composites (default: 'composite', next to --output-dir; kept out of "
                             "--output-dir, which a rerun of this stage clears).")
    parser.add_argument("--all-paths-tsv", type=str, default=None,
                        help="(Optional) Model metadata TSV (raw/all_paths.tsv of composite/build_all_paths.py), for the "
                             "model, assay and atlas of each model in the per-model composite metadata.")

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
    parser.add_argument("--hg38-fasta", type=str, default=None,
                        help="Not used: the hits are annotated by pipeline_encode_hits.py (accepted for existing launchers).")
    parser.add_argument("--gencode-gtf", type=str, default=None,
                        help="Not used: the hits are annotated by pipeline_encode_hits.py (accepted for existing launchers).")

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
    global ClusterArgs, VisualizeArgs, HitsCols, EncodeCombineArgs, MotifCompendiumArgs, CompositeArgs

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



#### PATHS --------------------------------------------------------------------
def crosscluster_paths(args):
    """Inputs (the atlas pipelines' compendia) and outputs (<output-dir>) of the ENCODE stages, by name."""
    ## Input: TF atlas
    mc_modelcluster_filtered_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_modelcluster_filtered.mc")
    mc_modelcluster_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_modelcluster_all.mc")

    mc_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_counts_tf_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_contribcluster_all.mc")

    mc_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_profile_tf_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_contribcluster_all.mc")


    ## Input: Chromatin atlas
    mc_modelcluster_filtered_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_modelcluster_filtered.mc")
    mc_modelcluster_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_modelcluster_all.mc")

    mc_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_counts_chrom_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_contribcluster_all.mc")

    mc_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_profile_chrom_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_contribcluster_all.mc")


    ## Input: ProCap atlas
    mc_modelcluster_filtered_procap_path = os.path.join(args.procap_combined_dir, "mc", "motifcompendium_modelcluster_filtered.mc")
    mc_modelcluster_procap_path = os.path.join(args.procap_combined_dir, "mc", "motifcompendium_modelcluster_all.mc")

    mc_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_counts_procap_path = os.path.join(args.procap_counts_dir, "mc", "motifcompendium_contribcluster_all.mc")

    mc_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_motif_all.mc")
    mc_cluster_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_cluster_all.mc")
    mc_contribcluster_profile_procap_path = os.path.join(args.procap_profile_dir, "mc", "motifcompendium_contribcluster_all.mc")


    ## Output: MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")

    mc_modelcluster_combined_path = os.path.join(mc_dir, "motifcompendium_modelcluster_combined.mc")
    mc_crosscluster_path = os.path.join(mc_dir, "motifcompendium_crosscluster_all.mc")
    mc_crosscluster_filtered_path = os.path.join(mc_dir, "motifcompendium_crosscluster_filtered.mc")
    mc_crosscluster_removed_path = os.path.join(mc_dir, "motifcompendium_crosscluster_removed.mc")

    ## Output: HTML summary table
    html_dir = os.path.join(args.output_dir, "html")

    html_crosscluster_path = os.path.join(html_dir, "motifcompendium_crosscluster_all.html")
    html_crosscluster_filtered_path = os.path.join(html_dir, "motifcompendium_crosscluster_filtered.html")
    html_crosscluster_removed_path = os.path.join(html_dir, "motifcompendium_crosscluster_removed.html")
    html_crosscluster_edited_path = os.path.join(html_dir, "motifcompendium_crosscluster_all_edited.html")
    html_crosscluster_edited_filtered_path = os.path.join(html_dir, "motifcompendium_crosscluster_filtered_edited.html")
    html_crosscluster_edited_removed_path = os.path.join(html_dir, "motifcompendium_crosscluster_removed_edited.html")

    ## Output: Export - MoDISco HDF5, MEME
    export_dir = os.path.join(args.output_dir, "export")

    modisco_crosscluster_path = os.path.join(export_dir, "motifcompendium_crosscluster_all.h5")
    modisco_crosscluster_filtered_path = os.path.join(export_dir, "motifcompendium_crosscluster_filtered.h5")
    modisco_crosscluster_removed_path = os.path.join(export_dir, "motifcompendium_crosscluster_removed.h5")

    meme_crosscluster_path = os.path.join(export_dir, "motifcompendium_crosscluster_all.meme.txt")
    meme_crosscluster_filtered_path = os.path.join(export_dir, "motifcompendium_crosscluster_filtered.meme.txt")
    meme_crosscluster_removed_path = os.path.join(export_dir, "motifcompendium_crosscluster_removed.meme.txt")

    ## Output: Metadata
    metadata_dir = os.path.join(args.output_dir, "metadata")

    metadata_modelcluster_combined_path = os.path.join(metadata_dir, "motifcompendium_modelcluster_combined_metadata.tsv")
    metadata_crosscluster_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_all_metadata.tsv")
    metadata_crosscluster_filtered_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_filtered_metadata.tsv")
    metadata_crosscluster_removed_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_removed_metadata.tsv")
    metadata_motif_index_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_motif_index.tsv")
    level_name_col = VisualizeArgs.crosscluster_name_col  # Each motif's cluster of the level, in its motif index (the hits')
    metadata_motif_category_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_motif_category.tsv")

    args_json_path = os.path.join(metadata_dir, "args.json")

    ## Output: Hits annotation
    metadata_hits_path = os.path.join(metadata_dir, "motifcompendium_crosscluster_metadata_hits.tsv")
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
    return SimpleNamespace(**{name: value for name, value in locals().items() if name != "args"})


def composite_paths(composite_dir):
    """Composite outputs (composite/{identify,validate}_composite_motifs.ipynb): <composite-dir>/{mc,export,html,metadata,ism}."""
    return SimpleNamespace(
        composite_dir=composite_dir,
        ism_dir=os.path.join(composite_dir, "ism"),
        # Composites (pipeline_encode.py); the per-motif and per-model tables are the ISM stage's inputs
        mc_composite_path=os.path.join(composite_dir, "mc", "motifcompendium_encode_composite.mc"),
        metadata_composite_path=os.path.join(composite_dir, "export", "motifcompendium_encode_composite_metadata.tsv"),
        metadata_composite_model_path=os.path.join(composite_dir, "export", "motifcompendium_encode_composite_metadata_model.tsv"),
        html_msw_path=os.path.join(composite_dir, "html", "motifcompendium_crosscluster_msw.html"),
        html_composite_path=os.path.join(composite_dir, "html", "motifcompendium_encode_composite.html"),
        # ISM cooperativity (the ISM evidence and the classified composites: --save-all) and the merged composites
        # (pipeline_encode_composite.py)
        mc_composite_ism_motifs_init_path=os.path.join(composite_dir, "mc", "motifcompendium_encode_composite_ism_motifs_init.mc"),
        mc_composite_ism_motifs_path=os.path.join(composite_dir, "mc", "motifcompendium_encode_composite_ism_motifs.mc"),
        mc_composite_ism_path=os.path.join(composite_dir, "mc", "motifcompendium_encode_composite_ism.mc"),
        mc_composite_ism_dedup_path=os.path.join(composite_dir, "mc", "motifcompendium_encode_composite_ism_deduplicated.mc"),
        metadata_composite_ism_motifs_path=os.path.join(composite_dir, "export", "motifcompendium_encode_composite_ism_motifs_metadata.tsv"),
        metadata_composite_ism_motifs_agg_path=os.path.join(composite_dir, "export", "motifcompendium_encode_composite_ism_motifs_metadata_agg.tsv"),
        html_composite_ism_path=os.path.join(composite_dir, "html", "motifcompendium_encode_composite_ism.html"),
        html_composite_ism_dedup_path=os.path.join(composite_dir, "html", "motifcompendium_encode_composite_ism_deduplicated.html"),
        # The encodecluster level (C5; pipeline_encode_composite.py): one compendium (nothing is removed at this
        # level), the membership of the crossclusters (C4), the motif index, the hits annotation (pipeline_encode_hits.py)
        metadata_dir=os.path.join(composite_dir, "metadata"),
        export_dir=os.path.join(composite_dir, "export"),
        mc_encodecluster_path=os.path.join(composite_dir, "mc", "motifcompendium_encode.mc"),
        html_encodecluster_path=os.path.join(composite_dir, "html", "motifcompendium_encode.html"),
        metadata_encodecluster_path=os.path.join(composite_dir, "metadata", "motifcompendium_encode_metadata.tsv"),
        meme_encodecluster_path=os.path.join(composite_dir, "export", "motifcompendium_encode.meme.txt"),
        metadata_encodecluster_membership_path=os.path.join(composite_dir, "metadata", "motifcompendium_encode_membership.tsv"),
        metadata_encodecluster_motif_index_path=os.path.join(composite_dir, "metadata", "motifcompendium_encode_motif_index.tsv"),
        metadata_hits_path=os.path.join(composite_dir, "metadata", "motifcompendium_encode_metadata_hits.tsv"),
        hits_annotated_dir=os.path.join(composite_dir, "export", "hits_annotated"),
    )


def hits_level_paths(args, cpaths, level):
    """crosscluster_paths(args), with {mc,html,metadata,meme}_level{,_filtered,_removed}_path those of the level the hits
    are indexed and summarized by (pipeline_encode_hits.py --level): the crossclusters' (C4) or, level "encodecluster",
    the encodecluster level's (C5; cpaths: composite_paths()), whose motif index, name column and hits outputs then also
    replace the crossclusters'. The encodecluster level is one compendium (its "filtered" one): no removed one (None)."""
    paths = crosscluster_paths(args)
    for kind in ("mc", "html", "metadata", "meme"):
        if level == "encodecluster":
            level_path = getattr(cpaths, f"{kind}_encodecluster_path")
            level_paths = (level_path, level_path, None)
        else:
            level_paths = tuple(getattr(paths, f"{kind}_crosscluster{suffix}_path") for suffix in ("", "_filtered", "_removed"))
        for suffix, level_path in zip(("", "_filtered", "_removed"), level_paths):
            setattr(paths, f"{kind}_level{suffix}_path", level_path)
    if level == "encodecluster":
        paths.metadata_motif_index_path = cpaths.metadata_encodecluster_motif_index_path
        paths.level_name_col = VisualizeArgs.encodecluster_name_col
        for name in ("metadata_dir", "export_dir", "metadata_hits_path", "hits_annotated_dir"):
            setattr(paths, name, getattr(cpaths, name))
    return paths


def atlas_stage_dir_args(tfatlas_dir, chromatlas_dir, procapatlas_dir):
    """The atlas pipelines' stage directories ({atlas}_{combined,counts,profile}_dir), from their output roots."""
    stage_dir_args = {}
    for atlas_key, atlas_dir in {"tf": tfatlas_dir, "chrom": chromatlas_dir, "procap": procapatlas_dir}.items():
        stage_dir_args.update({
            f"{atlas_key}_dir": os.path.join(atlas_dir, "combined"),
            f"{atlas_key}_combined_dir": os.path.join(atlas_dir, "combined"),
            f"{atlas_key}_counts_dir": os.path.join(atlas_dir, "counts"),
            f"{atlas_key}_profile_dir": os.path.join(atlas_dir, "profile"),
        })
        modelcluster_path = os.path.join(atlas_dir, "combined", "mc", "motifcompendium_modelcluster_filtered.mc")
        if not os.path.exists(modelcluster_path):
            raise FileNotFoundError(f"Atlas pipeline output not found (has it completed?): {modelcluster_path}")
    return stage_dir_args


def crosscluster_cluster_col_name():
    """The crosscluster of each modelcluster (mc_modelcluster_combined), as named by its clustering."""
    return f"crosscluster_rec_{ClusterArgs.cluster_algorithm_force}-{ClusterArgs.cluster_sim_threshold_force}_{ClusterArgs.cluster_algorithm_kmeans}-{ClusterArgs.cluster_sim_assignment_threshold_kmeans}"


def default_composite_dir(output_dir):
    """<run>/encode, next to <run>/cross: the composites, their ISM runs and the encodecluster level (C5),
    kept out of the ENCODE output directory, which a rerun clears."""
    return os.path.join(os.path.dirname(os.path.normpath(output_dir)), CompositeArgs.composite_dir_name)


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


#### COMPOSITES: FUNCTIONS ----------------------------------------------------
# From composite/identify_composite_motifs.ipynb, up to the ISM cooperativity runs (pipeline_encode_composite_ism.py).
def prepare_composite_reference(mc_ref_direct, ref_clustered_hr_metadata_path, length_filter=True):
    """
    Direct reference monomers for the composite decomposition, with their reference cluster and its human-readable
    label: without the reference composites, the long non-ZNF motifs (length_filter) and the IRFs (but one).

    Returns:
        (mc_ref_direct_monomer, reference cluster name per reference motif, reference cluster label per reference motif)
    """
    # Annotate motifs: Human-readable
    mc_ref_clustered_hr_metadata = pd.read_csv(ref_clustered_hr_metadata_path, sep="\t")

    mc_ref_clustered_hr_metadata[ReferenceMatchCols.ref_readable_col] = mc_ref_clustered_hr_metadata[MetadataCols.name_col].str.split("_").str[0]  # Remove "_{#}"
    mc_ref_clustered_hr_metadata['motifs_name'] = mc_ref_clustered_hr_metadata['motifs'].str.split(",")
    mc_ref_clustered_hr_metadata_expand = mc_ref_clustered_hr_metadata.explode("motifs_name")

    mc_ref_qc_cluster_name_dict = dict(zip(mc_ref_clustered_hr_metadata_expand['motifs_name'], mc_ref_clustered_hr_metadata_expand[MetadataCols.name_col]))
    mc_ref_qc_cluster_label_dict = dict(zip(mc_ref_clustered_hr_metadata_expand['motifs_name'], mc_ref_clustered_hr_metadata_expand[ReferenceMatchCols.ref_readable_col]))

    # Reference: Without composites
    mc_ref_direct_monomer = mc_ref_direct[~mc_ref_direct[MetadataCols.name_col].isin(CompositeArgs.ref_composites)]
    mc_ref_direct_monomer[CompositeArgs.reference_cluster_col] = mc_ref_direct_monomer[MetadataCols.name_col].map(mc_ref_qc_cluster_name_dict)
    mc_ref_direct_monomer[LabelCols.label_col] = mc_ref_direct_monomer[MetadataCols.name_col].map(mc_ref_qc_cluster_label_dict)

    # Add motif string
    idx_to_acgt = np.array(["", "A", "C", "G", "T"])
    for rc in ["fwd", "rev"]:
        if rc == "rev":
            motifs = utils_motif.reverse_complement(mc_ref_direct_monomer.motifs)
        else:
            motifs = mc_ref_direct_monomer.motifs

        # Identify argmax (1-indexed)
        mc_ref_argmax = motifs.argmax(axis=-1) + 1          # shape (N, L), values 1-4
        all_zero = (motifs == 0).all(axis=-1)                  # shape (N, L)

        mc_ref_idxmax = np.where(all_zero, 0, mc_ref_argmax)

        # Convert idx to ACGT string
        mc_ref_strings = np.array([
            "".join(idx_to_acgt[row])
            for row in mc_ref_idxmax
        ])

        # Add motif string to MotifCompendium
        mc_ref_direct_monomer[f"{VisualizeArgs.motif_string_col}_{rc}"] = mc_ref_strings

    # Remove long non-ZNF motifs
    mask_match_name = (
        mc_ref_direct_monomer.metadata[MetadataCols.name_col].str.contains(CompositeArgs.monomer_long_tf_names, na=False)
        | mc_ref_direct_monomer.metadata[TFAnnotationCols.family_col].isin(CompositeArgs.monomer_long_families)
    )
    mask_match_nonznf = ~(
        mc_ref_direct_monomer.metadata[TFAnnotationCols.tfclass_family_name_col].astype(str).str.contains("zinc|ZF", case=False, na=False)
        | mc_ref_direct_monomer.metadata[TFAnnotationCols.humantf_dbd_col].astype(str).str.contains("C2H2", case=False, na=False)
        | mc_ref_direct_monomer.metadata[TFAnnotationCols.family_col].astype(str).str.contains("ZNF|ZSCAN|ZKSCAN|ZBTB", case=False, na=False)
        | mc_ref_direct_monomer.metadata[TFAnnotationCols.family_col].astype(str).str.lower().str.startswith("z", na=False)
    )
    mask_match = mask_match_name | mask_match_nonznf
    mask_length = mc_ref_direct_monomer.metadata[f"{VisualizeArgs.motif_string_col}_fwd"].str.len() > CompositeArgs.monomer_max_length
    if length_filter:
        mc_ref_direct_monomer = mc_ref_direct_monomer[~(mask_match & mask_length)]

    # Remove all IRF, except one
    mc_ref_direct_monomer = mc_ref_direct_monomer[~((mc_ref_direct_monomer[MetadataCols.name_col].str.contains("IRF")) & (mc_ref_direct_monomer[MetadataCols.name_col] != CompositeArgs.monomer_irf_keep))]

    return mc_ref_direct_monomer, mc_ref_qc_cluster_name_dict, mc_ref_qc_cluster_label_dict


def assign_msw_composites(mc_crosscluster, mc_ref_composite, ic_scale, verbose=False):
    """
    Modified Smith-Waterman (Bellman DP): decompose each motif into reference monomers (bellman_* columns and the
    aligned-reference logo). A motif that is not high or medium confidence becomes a composite (with its core and
    flank labels from the matched reference clusters) when the decomposition finds more than one subunit at a
    similarity of at least min_single_score, and stops being one otherwise.
    """
    # Motifs and their metadata rows are matched by position below
    mc_crosscluster.metadata.reset_index(drop=True, inplace=True)

    # Initialize columns
    mc_crosscluster.metadata[CompositeArgs.bellman_score_col] = 0.0
    mc_crosscluster.metadata[CompositeArgs.bellman_sim_col] = 0.0
    mc_crosscluster.metadata[CompositeArgs.bellman_orient_col] = None
    mc_crosscluster.metadata[CompositeArgs.bellman_align_col] = None
    mc_crosscluster.metadata[CompositeArgs.bellman_name_col] = None
    mc_crosscluster.metadata[CompositeArgs.bellman_label_col] = None

    mc_ref_composite_motifs = mc_ref_composite.motifs  # (F, M, 4)
    ref_len = mc_ref_composite_motifs.shape[1]  # M
    # Aligned reference motifs: 2*M + L - 2 columns, with the ENCODE motif starting at column M-1
    aligned_motifs = np.zeros((mc_crosscluster.motifs.shape[0], max(mc_crosscluster.motifs.shape[1], ref_len) + 2 * ref_len - 2, mc_crosscluster.motifs.shape[2]))

    ## Run dynamic programming: Per ENCODE MotifCompendium motif
    for i, row_crosscluster in mc_crosscluster.metadata.iterrows():
        name_i = row_crosscluster[MetadataCols.name_col]
        unique_id_i = row_crosscluster[MetadataCols.unique_id_col]
        composite_i = row_crosscluster[MetadataCols.composite_col]
        label_conf_i = row_crosscluster[LabelCols.label_conf_col]
        motif_i = mc_crosscluster.motifs[i]
        motif_i_trimmed = utils_motif.trim_motif(motif_i , 0) # (L, 4), trim zeros

        # Pad to match length of reference motifs
        if motif_i_trimmed.shape[0] < mc_ref_composite.motifs.shape[1]:
            pad_len = mc_ref_composite.motifs.shape[1] - motif_i_trimmed.shape[0]
            motif_i_trimmed = np.pad(motif_i_trimmed, ((0, pad_len), (0, 0)))  # Pad with zeros at the end

        if verbose:
            logging.info(f"Processing: {name_i}")

        (bellman_score, bellman_sim_score, bellman_aligned_ref_stack, bellman_aligned_ref_idx, bellman_aligned_ref_rc, bellman_aligned_ref_h, bellman_aligned_ref_scale) = utils_composite.run_modified_smith_waterman_composite(
            target_motif=motif_i_trimmed,
            subunit_motif_stack=mc_ref_composite_motifs,
            sim_threshold=CompositeArgs.bellman_sim_min_threshold,
            overlap_penalty_fraction=CompositeArgs.bellman_overlap_penalty_fraction,
            match_count_max=CompositeArgs.bellman_match_count_max,
            match_count_penalty=CompositeArgs.bellman_match_count_penalty,
            match_count_fraction=CompositeArgs.bellman_match_count_fraction,
            subunit_motif_trim=CompositeArgs.bellman_subunit_motif_trim,
            label_unsigned=True,
            allow_reverse_comp=True,
            ic_scale=ic_scale,
            target_similarity=CompositeArgs.bellman_target_similarity,
            collapse=CompositeArgs.bellman_collapse,
        )

        # Aligned reference motif (collapsed)
        bellman_aligned_ref_collapse = bellman_aligned_ref_stack.sum(axis=0)  # (2*M + L - 2, 4)
        bellman_aligned_ref_collapse = utils_motif.pad_motif(
            bellman_aligned_ref_collapse[np.newaxis, :, :], aligned_motifs.shape[1])

        # Record, in original MotifCompendium
        org_idx = (mc_crosscluster.metadata[MetadataCols.unique_id_col] == unique_id_i).idxmax()
        if bellman_score > row_crosscluster[CompositeArgs.bellman_score_col]:
            # Identify reference motifs used in alignment: One per match (a reference motif can match more than once)
            match_idx = np.where(bellman_aligned_ref_scale > 0)[0]
            match_idx_sort = match_idx[np.argsort(-bellman_aligned_ref_scale[match_idx], kind="stable")]  # Sort by descending scale factor
            ref_idx_sort = bellman_aligned_ref_idx[match_idx_sort]  # Positions in mc_ref_composite
            ref_num = len(ref_idx_sort)
            ref_names = mc_ref_composite.metadata[MetadataCols.name_col].iloc[ref_idx_sort].to_list()
            ref_names_combined = "::".join(ref_names)
            ref_orient_combined = "::".join("rev" if rc else "for" for rc in bellman_aligned_ref_rc[match_idx_sort])  # Per reference motif, as ref_names
            ref_start_combined = "::".join(str(h - (ref_len - 1)) for h in bellman_aligned_ref_h[match_idx_sort])  # Start in motif_i_trimmed, per reference motif
            ref_cluster_names = mc_ref_composite.metadata[CompositeArgs.reference_cluster_col].iloc[ref_idx_sort].to_list()  # If using mc_ref
            ref_cluster_labels = mc_ref_composite.metadata[LabelCols.label_col].iloc[ref_idx_sort].to_list()
            ref_cluster_names_combined = "::".join(ref_cluster_names)
            ref_cluster_labels_combined = "::".join(ref_cluster_labels)

            # Bellman: Composite condition
            bellman_composite = (
                (ref_num > 1)
                and (bellman_sim_score >= MotifMatchArgs.min_single_score)
            )

            # Update metadata: Bellman
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_score_col] = bellman_score
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_sim_col] = bellman_sim_score
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_orient_col] = ref_orient_combined
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_align_col] = ref_start_combined
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_name_col] = ref_names_combined
            mc_crosscluster.metadata.loc[org_idx, CompositeArgs.bellman_label_col] = ref_cluster_labels_combined

            # Update metadata: Composite
            if label_conf_i not in LabelCols.label_confs[0:2]:  # Don't update high, medium
                # Prior composites + flip only. `not`, never `~`: composite_i comes from iterrows() as a Python bool,
                # for which ~True == -2 and ~False == -1 are both truthy (every Bellman composite was relabeled)
                if (bellman_composite and not composite_i) or (not bellman_composite and composite_i):
                    mc_crosscluster.metadata.loc[org_idx, MetadataCols.composite_col] = bellman_composite
                    mc_crosscluster.metadata.loc[org_idx, MetadataCols.num_subunit_col] = ref_num
                    mc_crosscluster.metadata.loc[org_idx, LabelCols.label_core_col] = ref_cluster_labels[0] if ref_num > 0 else None  # If using mc_ref
                    mc_crosscluster.metadata.loc[org_idx, LabelCols.label_core_readable_col] = ref_cluster_labels[0] if ref_num > 0 else None
                    mc_crosscluster.metadata.loc[org_idx, LabelCols.label_flank_col] = "::".join(ref_cluster_labels[1:]) if bellman_composite else None
                    mc_crosscluster.metadata.loc[org_idx, LabelCols.label_col] = mc_crosscluster.metadata.loc[org_idx, LabelCols.label_core_readable_col] + "::" + mc_crosscluster.metadata.loc[org_idx, LabelCols.label_flank_col] if bellman_composite else mc_crosscluster.metadata.loc[org_idx, LabelCols.label_core_readable_col]

            aligned_motifs[org_idx] = bellman_aligned_ref_collapse # Save aligned motif

    # Visualize aligned motifs, in original MotifCompendium
    mc_crosscluster.add_logos(
        motifs=aligned_motifs,
        image_name=CompositeArgs.bellman_logo_col,
        trim=0,
    )


def write_back_rows(mc, mc_rows, rows, columns, image_names):
    """Set columns and images of mc's rows (positions) to those of mc_rows (mc[rows]), None and NaN included."""
    for col in columns:
        values = mc_rows.metadata[col].to_numpy()
        if col not in mc.metadata:
            mc.metadata[col] = None
        if mc.metadata[col].dtype != values.dtype:
            mc.metadata[col] = mc.metadata[col].astype(object)
        mc.metadata.iloc[rows, mc.metadata.columns.get_loc(col)] = values
    for image_name in image_names:
        images = list(mc.get_images(image_name)) if image_name in mc.images() else [None] * len(mc)
        for row, image in zip(rows, mc_rows.get_images(image_name)):
            images[row] = image
        mc.set_images(image_name, images)


def annotate_family(metadata_df, tf_col, family_col):
    # Family: Remove number at end
    metadata_df[family_col] = metadata_df[tf_col].apply(
        lambda x: x[:-3] if (len(x) > 3 and x[-3:].isdigit()) else (x[:-2] if x[-2:].isdigit() else (x[:-1] if x[-1:].isdigit() else x))
    )

    # Family: If matched with known family
    for family in CompositeArgs.families:
        metadata_df.loc[metadata_df[tf_col].str.upper().str.contains(family), family_col] = family

    # MX = MAX
    metadata_df.loc[metadata_df[tf_col].str.contains("MX"), family_col] = "MAX"


def identify_composites(mc_crosscluster, mc_ref_direct_monomer, mc_ref_qc_cluster_name_dict, mc_ref_direct_monomer_all):
    """
    The composites of mc_crosscluster (its filtered crossclusters), each with its reference subunits: name, cluster,
    score, alignment, start, end and motif string per subunit, and the spacing between the first two.

    The subunits of a core or flank label are found among its monomers of the composite reference
    (mc_ref_direct_monomer), else, when it has none there (every motif of the label is longer than
    CompositeArgs.monomer_max_length, e.g. CBFB, CAMTA, FERD3L, NFKB), among its monomers of mc_ref_direct_monomer_all:
    the same reference without the length filter.
    """
    # Identify composites
    mc_encode_composite = mc_crosscluster[(mc_crosscluster[MetadataCols.composite_col] == True)]

    # Drop: Errors, Filter flag names
    mc_encode_composite = mc_encode_composite[~((mc_encode_composite[LabelCols.label_core_col].str.startswith(CompositeArgs.composite_error_prefix)) | (mc_encode_composite[LabelCols.label_flank_col].str.startswith(CompositeArgs.composite_error_prefix)))]

    # Label: Family
    annotate_family(mc_encode_composite.metadata, tf_col=LabelCols.label_core_col, family_col=LabelCols.label_core_readable_col)
    annotate_family(mc_encode_composite.metadata, tf_col=LabelCols.label_flank_col, family_col=LabelCols.label_flank_readable_col)

    # Assign references
    composite_ref_col = CompositeArgs.composite_ref_col
    mc_encode_composite_metadata_updates = []
    mc_encode_composite_images_updated = pd.DataFrame(index=mc_encode_composite[MetadataCols.unique_id_col])
    mc_encode_composite_images_existing = set(mc_encode_composite.images())
    # Motifs and images are positional arrays.  Retain this positional key while
    # metadata is temporarily indexed by MCID, then restore it before assignment.
    mc_encode_composite.metadata["_motif_row"] = np.arange(len(mc_encode_composite))

    mc_encode_composite_len = mc_encode_composite.motifs.shape[1]

    for motif_row, (_, row) in enumerate(mc_encode_composite.metadata.iterrows()):
        # MotifCompendium: Subset
        mc_encode_composite_i = mc_encode_composite[np.array([motif_row], dtype=int)]
        name_i = mc_encode_composite_i[MetadataCols.name_col].values[0]
        label_core_i = mc_encode_composite_i[LabelCols.label_core_col].values[0]
        label_core_family_i = mc_encode_composite_i[LabelCols.label_core_readable_col].values[0]
        label_flank_i = mc_encode_composite_i[LabelCols.label_flank_col].values[0]
        label_flank_family_i = mc_encode_composite_i[LabelCols.label_flank_readable_col].values[0]

        # Reference: Core, flank monomers
        mc_ref_core_names_i = _subunit_reference_names(mc_ref_direct_monomer, mc_ref_direct_monomer_all, label_core_i, label_core_family_i)
        if len(mc_ref_core_names_i) == 0:
            raise ValueError(f"Core label '{label_core_i}' ({label_core_family_i}) not found in reference motifs.")

        mc_ref_flank_names_i = _subunit_reference_names(mc_ref_direct_monomer, mc_ref_direct_monomer_all, label_flank_i, label_flank_family_i)
        if len(mc_ref_flank_names_i) == 0:
            raise ValueError(f"Flank label '{label_flank_i}' ({label_flank_family_i}) not found in reference motifs.")

        # Reference: Subset (the composite reference is mc_ref_direct_monomer_all's rows in order: same subsets)
        mc_ref_direct_monomer_i = mc_ref_direct_monomer_all[mc_ref_direct_monomer_all[MetadataCols.name_col].isin(mc_ref_core_names_i | mc_ref_flank_names_i)]
        if len(mc_ref_direct_monomer_i) == 0:
            raise ValueError(f"No reference motifs found for core '{label_core_i}' and flank '{label_flank_i}'.")

        # Assign Score, labels, alignments
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_encode_composite_i,
            assign_from_mc=mc_ref_direct_monomer_i,
            from_label_col=MetadataCols.name_col,
            save_col_prefix=composite_ref_col,
            min_score=0,
            max_submotifs=CompositeArgs.max_submotifs_composite,
            label_unsigned=True,
            save_images=True,
            logo_trimming=0,
        )

        # Assign: Start, End positions
        for j in range(CompositeArgs.max_submotifs_composite):
            # Assign: Reference cluster
            mc_encode_composite_i[f'{composite_ref_col}_cluster{j}'] = mc_encode_composite_i[f'{composite_ref_col}_name{j}'].map(mc_ref_qc_cluster_name_dict)

            # Reference motif
            mc_ref_direct_name_j = mc_encode_composite_i[f'{composite_ref_col}_name{j}'].values[0]
            mc_ref_direct_j = mc_ref_direct_monomer_i[mc_ref_direct_monomer_i[MetadataCols.name_col] == mc_ref_direct_name_j]
            mc_ref_direct_motif_j = mc_ref_direct_j.motifs[0]
            mc_ref_direct_string_fwd_j = mc_ref_direct_j[f'{VisualizeArgs.motif_string_col}_fwd'].values[0]
            mc_ref_direct_string_rev_j = mc_ref_direct_j[f'{VisualizeArgs.motif_string_col}_rev'].values[0]

            # Trim motif: Zero pad
            mc_ref_direct_motif_unpad_j = utils_motif.trim_motif(
                motif=mc_ref_direct_motif_j,
                importance=0,
            )
            mc_ref_direct_motif_len_j = mc_ref_direct_motif_unpad_j.shape[0]

            # Trim motif: Full
            mc_ref_direct_motif_abs_j = np.abs(mc_ref_direct_motif_j)
            per_position_totals = np.sum(mc_ref_direct_motif_abs_j, axis=1)
            included_positions = per_position_totals > (0.5 / mc_ref_direct_motif_len_j) * np.sum(per_position_totals)
            if np.sum(included_positions) == 0:
                raise ValueError(f"All positions trimmed for motif '{mc_ref_direct_name_j}' in composite '{name_i}'.")
            min_index = np.argmax(included_positions)
            max_index = mc_ref_direct_motif_j.shape[0] - np.argmax(included_positions[::-1])
            mc_ref_direct_motif_trimmed_j = mc_ref_direct_motif_j[min_index:max_index]
            mc_ref_direct_motif_trimmed_len_j = mc_ref_direct_motif_trimmed_j.shape[0]

            alignment_rc_j = mc_encode_composite_i[f'{composite_ref_col}_alignment_rc{j}'].values[0]
            alignment_h_j = mc_encode_composite_i[f'{composite_ref_col}_alignment_h{j}'].values[0]

            # Reverse complement
            if alignment_rc_j:
                end_j = mc_encode_composite_len + alignment_h_j - min_index if alignment_h_j is not None else None
                start_j = end_j - mc_ref_direct_motif_trimmed_len_j if end_j is not None else None
                mc_ref_direct_string_trimmed_j = mc_ref_direct_string_rev_j[mc_ref_direct_motif_len_j - max_index : mc_ref_direct_motif_len_j - min_index]
            # Forward
            else:
                start_j = alignment_h_j + min_index if alignment_h_j is not None else None
                end_j = start_j + mc_ref_direct_motif_trimmed_len_j if start_j is not None else None
                mc_ref_direct_string_trimmed_j = mc_ref_direct_string_fwd_j[min_index:max_index]

            mc_encode_composite_i[f'{composite_ref_col}_start{j}'] = start_j
            mc_encode_composite_i[f'{composite_ref_col}_end{j}'] = end_j
            mc_encode_composite_i[f'{composite_ref_col}_{VisualizeArgs.motif_string_col}{j}'] = mc_ref_direct_string_trimmed_j

        # Spacing
        mc_encode_composite_spacing = np.maximum(
            mc_encode_composite_i[f'{composite_ref_col}_start1']
            - mc_encode_composite_i[f'{composite_ref_col}_end0'],
            mc_encode_composite_i[f'{composite_ref_col}_start0']
            - mc_encode_composite_i[f'{composite_ref_col}_end1'],
        )
        mc_encode_composite_i[f'{composite_ref_col}_spacing'] = mc_encode_composite_spacing

        # Update: Metadata, Images
        mc_encode_composite_metadata_updates.append(mc_encode_composite_i.metadata.set_index(MetadataCols.unique_id_col))
        new_image_cols = [image_col for image_col in mc_encode_composite_i.images() if image_col not in mc_encode_composite_images_existing]
        mc_encode_composite_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_encode_composite_i.get_images(image_col),
                index=mc_encode_composite_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_encode_composite_images_updated = mc_encode_composite_images_updated.combine_first(mc_encode_composite_i_new_images)
        mc_encode_composite_images_updated.update(mc_encode_composite_i_new_images)

    # Apply updates: Metadata, Images
    mc_encode_composite_metadata_updates_df = pd.concat(mc_encode_composite_metadata_updates)
    mc_encode_composite.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_encode_composite.metadata = mc_encode_composite.metadata.combine_first(mc_encode_composite_metadata_updates_df)
    mc_encode_composite.metadata.update(mc_encode_composite_metadata_updates_df)
    mc_encode_composite.metadata = (
        mc_encode_composite.metadata.reset_index()
        .sort_values("_motif_row", kind="stable")
        .drop(columns="_motif_row")
        .reset_index(drop=True)
    )
    del mc_encode_composite_metadata_updates_df, mc_encode_composite_metadata_updates

    for image_col in mc_encode_composite_images_updated.columns:
        mc_encode_composite.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None for x in mc_encode_composite_images_updated
                    .reindex(mc_encode_composite[MetadataCols.unique_id_col].to_numpy())[image_col]]
        )
    del mc_encode_composite_images_updated, mc_encode_composite_images_existing

    return mc_encode_composite


def _reference_label_mask(mc_ref, label_i, label_family_i):
    """Reference motifs of a core or flank label: by family, else TF, of the label, else of its family, else of each part."""
    if (label_i in mc_ref[ReferenceMatchCols.ref_family_col].unique()):
        return mc_ref[ReferenceMatchCols.ref_family_col] == label_i
    elif (label_i in mc_ref[ReferenceMatchCols.ref_tf_col].unique()):
        return mc_ref[ReferenceMatchCols.ref_tf_col] == label_i
    elif (label_family_i in mc_ref[ReferenceMatchCols.ref_family_col].unique()):
        return mc_ref[ReferenceMatchCols.ref_family_col] == label_family_i
    elif (label_family_i in mc_ref[ReferenceMatchCols.ref_tf_col].unique()):
        return mc_ref[ReferenceMatchCols.ref_tf_col] == label_family_i
    mask = pd.Series([False] * len(mc_ref), index=mc_ref.metadata.index)
    for label_j in label_i.split("-"):
        if label_j in mc_ref[ReferenceMatchCols.ref_family_col].unique():
            mask |= mc_ref[ReferenceMatchCols.ref_family_col] == label_j
        elif label_j in mc_ref[ReferenceMatchCols.ref_tf_col].unique():
            mask |= mc_ref[ReferenceMatchCols.ref_tf_col] == label_j
    if pd.notna(label_family_i):
        for label_family_j in label_family_i.split("-"):
            if label_family_j in mc_ref[ReferenceMatchCols.ref_family_col].unique():
                mask |= mc_ref[ReferenceMatchCols.ref_family_col] == label_family_j
            elif label_family_j in mc_ref[ReferenceMatchCols.ref_tf_col].unique():
                mask |= mc_ref[ReferenceMatchCols.ref_tf_col] == label_family_j
    return mask


def _subunit_reference_names(mc_ref_direct_monomer, mc_ref_direct_monomer_all, label_i, label_family_i):
    """Names of the subunit candidates of a core or flank label: its composite-reference monomers, else all of its monomers."""
    for mc_ref in (mc_ref_direct_monomer, mc_ref_direct_monomer_all):
        mask = _reference_label_mask(mc_ref, label_i, label_family_i)
        if mask.sum() > 0:
            return set(mc_ref[MetadataCols.name_col][mask.to_numpy()])
    return set()


def export_composites(mc_crosscluster, mc_encode_composite, composite_paths_ns, all_paths_tsv, html_column_desc_dict):
    """
    The inputs of the ISM cooperativity runs (pipeline_encode_composite_ism.py), in <composite-dir>: the composites'
    metadata (per motif and per model), .mc and HTML, and the HTML of the decomposition of every motif (mc_crosscluster).
    """
    composite_ref_col = CompositeArgs.composite_ref_col

    # HTML: Modified Smith-Waterman
    html_columns_crosscluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_crosscluster.columns() + mc_crosscluster.images())]
    html_column_desc_dict_crosscluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_crosscluster}

    os.makedirs(os.path.dirname(composite_paths_ns.html_msw_path), exist_ok=True)
    mc_crosscluster.summary_table_html(
        html_out=composite_paths_ns.html_msw_path,
        columns=[
            MetadataCols.name_col, MetadataCols.composite_col, MetadataCols.num_subunit_col,
            CompositeArgs.bellman_score_col, CompositeArgs.bellman_sim_col, CompositeArgs.bellman_align_col,
            CompositeArgs.bellman_name_col, CompositeArgs.bellman_label_col, CompositeArgs.bellman_logo_col,
            f'{LabelCols.best_label_ref_col}_score0', f'{LabelCols.best_label_ref_col}_name0', f'{LabelCols.best_label_ref_col}_logo0',
        ],
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_crosscluster,
    )

    # Save metadata
    mc_encode_composite.metadata = mc_encode_composite.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    os.makedirs(os.path.dirname(composite_paths_ns.metadata_composite_path), exist_ok=True)
    mc_encode_composite.metadata.to_csv(composite_paths_ns.metadata_composite_path, sep="\t", index=False)

    # Metadata (row = per model): the models each composite is tested in
    mc_encode_composite_metadata_explode = mc_encode_composite.metadata.copy()
    mc_encode_composite_metadata_explode[MetadataCols.id_col] = mc_encode_composite_metadata_explode[MetadataCols.id_col].str.split(",")
    mc_encode_composite_metadata_explode = mc_encode_composite_metadata_explode.explode(MetadataCols.id_col)

    mc_encode_composite_metadata_model = mc_encode_composite_metadata_explode.groupby(MetadataCols.id_col, as_index=False)[MetadataCols.name_col].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str)))
    )
    mc_encode_composite_metadata_model[MetadataCols.num_constituents_col] = mc_encode_composite_metadata_model[MetadataCols.name_col].str.split(",").str.len()
    if all_paths_tsv:
        # Model metadata: one row per (model, head); the model-level columns are the same for both heads
        all_paths_df = pd.read_csv(all_paths_tsv, sep="\t", header=0)
        all_paths_model_df = all_paths_df.drop_duplicates('experiment').set_index('experiment')
        mc_encode_composite_metadata_model[MetadataCols.model_col] = mc_encode_composite_metadata_model[MetadataCols.id_col].map(all_paths_model_df[MetadataCols.model_col])  # bpnet, chrombpnet, procapnet
        mc_encode_composite_metadata_model[MetadataCols.assay_col] = mc_encode_composite_metadata_model[MetadataCols.id_col].map(all_paths_model_df[MetadataCols.assay_col])
        mc_encode_composite_metadata_model[VisualizeArgs.atlas_col] = mc_encode_composite_metadata_model[MetadataCols.id_col].map(all_paths_model_df[VisualizeArgs.atlas_col])  # TF, Chrom, ProCap
    else:
        logging.info("No --all-paths-tsv: the per-model composite metadata has no model, assay or atlas columns.")

    os.makedirs(os.path.dirname(composite_paths_ns.metadata_composite_model_path), exist_ok=True)
    mc_encode_composite_metadata_model.to_csv(composite_paths_ns.metadata_composite_model_path, sep="\t", index=False)

    # Save summary table HTML
    html_columns_crosscluster = [
            col
            for i in range(CompositeArgs.max_submotifs_composite)
            for col in [
                f"{composite_ref_col}_name{i}",
                f'{composite_ref_col}_cluster{i}',
                f"{composite_ref_col}_score{i}",
                f"{composite_ref_col}_logo{i}",
                f"{composite_ref_col}_alignment_rc{i}",
                f"{composite_ref_col}_alignment_h{i}",
                f"{composite_ref_col}_start{i}",
                f"{composite_ref_col}_end{i}",
                f"{composite_ref_col}_{VisualizeArgs.motif_string_col}{i}",
            ]
        ] + [
        html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_encode_composite.columns() + mc_encode_composite.images())
    ]

    os.makedirs(os.path.dirname(composite_paths_ns.html_composite_path), exist_ok=True)
    mc_encode_composite.summary_table_html(
        html_out=composite_paths_ns.html_composite_path,
        columns=html_columns_crosscluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict,
    )

    # Save MotifCompendium: ENCODE composites
    os.makedirs(os.path.dirname(composite_paths_ns.mc_composite_path), exist_ok=True)
    save_replace(mc_encode_composite, composite_paths_ns.mc_composite_path)
    logging.info(f"Composites: {len(mc_encode_composite)} of {len(mc_crosscluster)} filtered crossclusters, in "
                 f"{len(mc_encode_composite_metadata_model)} models: {composite_paths_ns.composite_dir}")


#### EXPORT: FUNCTIONS ---------------------------------------------------------
def motif_category_types(motif_index_df, name_col, flag_col, filtered_names):
    """Each motif's category (DirectTypeArgs.category_types): from its atlas, direct type, filter flag (flag_col) and
    cluster of the level (name_col; filtered_names: those of the level's filtered clusters). The motif index's
    category (export_crosscluster_constituents(): the crossclusters), or the encodecluster level's."""
    return np.select(
        [
            # TF Atlas: Direct
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (motif_index_df[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (motif_index_df[flag_col] == False)
             & (~motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Indirect
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (~motif_index_df[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (motif_index_df[flag_col] == False)
             & (~motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Unvalidated
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & (~motif_index_df[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4]))
             & (motif_index_df[flag_col] == False)
             & (motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Non-canonical
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id)
             & ((motif_index_df[flag_col] == True)
             | (~motif_index_df[name_col].isin(filtered_names)
               ))),
            # Chromatin Atlas: Canonical
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & (motif_index_df[flag_col] == False)
             & (~motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # TF Atlas: Unvalidated
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & (motif_index_df[flag_col] == False)
             & (motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # Chromatin Atlas: Non-canonical
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.chromatlas_id)
             & ((motif_index_df[flag_col] == True)
             | (~motif_index_df[name_col].isin(filtered_names)
               ))),
            # ProCap Atlas: Canonical
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & (motif_index_df[flag_col] == False)
             & (~motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # ProCap Atlas: Unvalidated
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & (motif_index_df[flag_col] == False)
             & (motif_index_df[name_col].str.contains(MetadataCols.name_unknown, na=False))),
            # ProCap Atlas: Non-canonical
            ((motif_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.procapatlas_id)
             & ((motif_index_df[flag_col] == True)
             | (~motif_index_df[name_col].isin(filtered_names)
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


def export_crosscluster_level(args, mc_crosscluster, mc_modelcluster_combined, html_column_desc_dict):
    """
    Write the ENCODE compendium out: mc_crosscluster is the whole compendium (filtered and removed, by filter_flag_col). In
    <output-dir>: the ENCODE compendia (all, filtered, removed: .mc, HTML, MEME, metadata TSV) and the combined
    modelclusters. The levels below: export_crosscluster_constituents().
    """
    paths = crosscluster_paths(args)

    #### Aggregate: Metadata ----------------------------------
    logging.info("Recompiling metadata from the atlases' initial metadata...")

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

    crosscluster_metadata_df = pd.concat(
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
        col for col in required_metadata_cols if col not in crosscluster_metadata_df.columns
    ]
    if missing_metadata_cols:
        raise ValueError(
            "Initial atlas metadata is missing required columns: "
            + ", ".join(missing_metadata_cols)
        )

    # Clean up metadata
    crosscluster_metadata_df[metadata_recompile_cols] = (
        crosscluster_metadata_df[metadata_recompile_cols]
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

    def recompile_metadata(mc_crosscluster_i, crosscluster_metadata_df, recompile_cols):
        rows_to_ids = (
            mc_crosscluster_i.metadata[[MetadataCols.id_col]]
            .assign(_row=mc_crosscluster_i.metadata.index)
            .assign(_encid=lambda d: d[MetadataCols.id_col].astype(str).str.split(","))
            .explode("_encid")
        )

        merged = rows_to_ids.merge(
            crosscluster_metadata_df,
            left_on="_encid",
            right_on=MetadataCols.id_col,
            how="left",
            suffixes=("", "_source"),
        )

        compiled = (
            merged.groupby("_row")[recompile_cols]
            .agg(join_unique)
            .reindex(mc_crosscluster_i.metadata.index)
            .fillna("")
        )

        mc_crosscluster_i.metadata.loc[:, recompile_cols] = compiled

    # Recompile metadata: the whole compendium, before its filtered and removed crossclusters are exported
    recompile_metadata(mc_crosscluster, crosscluster_metadata_df, metadata_recompile_cols)

    # #### Visualize HTML
    # # Add: Notes
    # mc_crosscluster.metadata[notes_col] = None
    # mc_crosscluster.metadata[checked_col] = None

    # Create HTML
    html_columns_crosscluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_crosscluster.columns() + mc_crosscluster.images())]
    html_column_desc_dict_crosscluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_crosscluster}

    os.makedirs(os.path.dirname(paths.html_crosscluster_path), exist_ok=True)
    mc_crosscluster.summary_table_html(
        html_out=paths.html_crosscluster_path,
        columns=html_columns_crosscluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_crosscluster,
    )

    os.makedirs(os.path.dirname(paths.html_crosscluster_filtered_path), exist_ok=True)
    mc_crosscluster_filtered = mc_crosscluster[~mc_crosscluster[FilterArgs.filter_flag_col]]
    mc_crosscluster_filtered.summary_table_html(
        html_out=paths.html_crosscluster_filtered_path,
        columns=html_columns_crosscluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_crosscluster,
    )

    mc_crosscluster_removed = mc_crosscluster[mc_crosscluster[FilterArgs.filter_flag_col]]
    if getattr(args, "save_all", False):
        os.makedirs(os.path.dirname(paths.html_crosscluster_removed_path), exist_ok=True)
        mc_crosscluster_removed.summary_table_html(
            html_out=paths.html_crosscluster_removed_path,
            columns=html_columns_crosscluster,
            logo_trimming=VisualizeArgs.html_logo_trim,
            editable=True,
            column_desc_dict=html_column_desc_dict_crosscluster,
        )

    # # Create HTML: Edited
    # html_columns_crosscluster = [html_col for html_col in html_columns if html_col in (mc_crosscluster.columns() + mc_crosscluster.images())]
    # html_column_desc_dict_crosscluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_crosscluster}

    # os.makedirs(os.path.dirname(paths.html_crosscluster_edited_path), exist_ok=True)
    # mc_crosscluster.summary_table_html(
    #     html_out=paths.html_crosscluster_edited_path,
    #     columns=html_columns_crosscluster,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_crosscluster,
    # )

    # os.makedirs(os.path.dirname(paths.html_crosscluster_edited_filtered_path), exist_ok=True)
    # mc_crosscluster_filtered = mc_crosscluster[~mc_crosscluster[filter_flag_col]]
    # mc_crosscluster_filtered.summary_table_html(
    #     html_out=paths.html_crosscluster_edited_filtered_path,
    #     columns=html_columns_crosscluster+['checked', f'{qc_col}_name0', f'{qc_col}_score0'],
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_crosscluster,
    # )

    # os.makedirs(os.path.dirname(paths.html_crosscluster_edited_removed_path), exist_ok=True)
    # mc_crosscluster_removed = mc_crosscluster[mc_crosscluster[filter_flag_col]]
    # mc_crosscluster_removed.summary_table_html(
    #     html_out=paths.html_crosscluster_edited_removed_path,
    #     columns=html_columns_crosscluster + [filter_flag_col, filter_flag_type_col, f'{qc_col}_score0'],
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict_crosscluster,
    # )

    # Clean up
    mc_crosscluster.metadata = mc_crosscluster.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_crosscluster_filtered.metadata = mc_crosscluster_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_crosscluster_removed.metadata = mc_crosscluster_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')
    mc_modelcluster_combined.metadata = mc_modelcluster_combined.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore')

    # Save MotifCompendium: ENCODE
    os.makedirs(os.path.dirname(paths.mc_crosscluster_path), exist_ok=True)
    mc_crosscluster.save(paths.mc_crosscluster_path)

    os.makedirs(os.path.dirname(paths.mc_crosscluster_filtered_path), exist_ok=True)
    mc_crosscluster_filtered.save(paths.mc_crosscluster_filtered_path)

    if getattr(args, "save_all", False):
        os.makedirs(os.path.dirname(paths.mc_crosscluster_removed_path), exist_ok=True)
        mc_crosscluster_removed.save(paths.mc_crosscluster_removed_path)

    os.makedirs(os.path.dirname(paths.mc_modelcluster_combined_path), exist_ok=True)
    mc_modelcluster_combined.save(paths.mc_modelcluster_combined_path)

    # Load MotifCompendium: ENCODE
    # mc_modelcluster_combined = MotifCompendium.load(paths.mc_modelcluster_combined_path, safe=safe)


    # ### Export
    # Export as MEME txt
    os.makedirs(os.path.dirname(paths.meme_crosscluster_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_crosscluster,
        name_col=MetadataCols.name_col,
        save_loc=paths.meme_crosscluster_path,
        trim_importance=(1/(2*mc_crosscluster.motifs.shape[1])),
    )

    os.makedirs(os.path.dirname(paths.meme_crosscluster_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_crosscluster_filtered,
        name_col=MetadataCols.name_col,
        save_loc=paths.meme_crosscluster_filtered_path,
        trim_importance=(1/(2*mc_crosscluster_filtered.motifs.shape[1])),
    )

    if getattr(args, "save_all", False):
        os.makedirs(os.path.dirname(paths.meme_crosscluster_removed_path), exist_ok=True)
        utils_analysis.export_compendium_meme(
            mc=mc_crosscluster_removed,
            name_col=MetadataCols.name_col,
            save_loc=paths.meme_crosscluster_removed_path,
            trim_importance=(1/(2*mc_crosscluster_removed.motifs.shape[1])),
        )

    # Metadata TSV
    os.makedirs(os.path.dirname(paths.metadata_modelcluster_combined_path), exist_ok=True)
    mc_modelcluster_combined.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_modelcluster_combined_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(paths.metadata_crosscluster_path), exist_ok=True)
    mc_crosscluster.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_crosscluster_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(paths.metadata_crosscluster_path), exist_ok=True)
    mc_crosscluster_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_crosscluster_filtered_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(paths.metadata_crosscluster_path), exist_ok=True)
    if getattr(args, "save_all", False):
        mc_crosscluster_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_crosscluster_removed_path, sep="\t", index=False)



def export_crosscluster_constituents(args, mc_crosscluster, mc_modelcluster_combined, modelcluster2crosscluster_name_map,
                            modelcluster2crosscluster_simple_map, crosscluster_cluster_col, html_column_desc_dict):
    """
    The crossclusters, propagated to every level below (after export_crosscluster_level(); --update-constituents only: on their own,
    from the combine's saved outputs, run_crosscluster_constituents()). modelcluster2crosscluster_*_map map each crosscluster of
    mc_modelcluster_combined (crosscluster_cluster_col) to its name and simple name in mc_crosscluster. In the atlas
    pipelines' directories: every level's compendia (all, filtered, removed: .mc, metadata TSV, MEME, HTML but at the
    motif level) and the motif indexes, with their crosscluster, final filter flags and confidence, over the atlas
    pipelines' own files (each level's modelcluster: the atlas pipelines' update of their levels); in <output-dir>, the
    ENCODE motif index. The FiNeMo hits are annotated by pipeline_encode_hits.py, from these outputs.
    """
    paths = crosscluster_paths(args)
    mc_crosscluster_filtered = mc_crosscluster[~mc_crosscluster[FilterArgs.filter_flag_col].astype(bool).to_numpy()]
    # The levels' HTML column descriptions: the crosscluster level's (export_crosscluster_level())
    html_columns_crosscluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_crosscluster.columns() + mc_crosscluster.images())]
    html_column_desc_dict_crosscluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_crosscluster}

    # ### Motif index
    # Load MotifCompendium
    mc_counts_tf = MotifCompendium.load(paths.mc_counts_tf_path, safe=args.safe)
    mc_cluster_counts_tf = MotifCompendium.load(paths.mc_cluster_counts_tf_path, safe=args.safe)
    mc_contribcluster_counts_tf = MotifCompendium.load(paths.mc_contribcluster_counts_tf_path, safe=args.safe)

    mc_profile_tf = MotifCompendium.load(paths.mc_profile_tf_path, safe=args.safe)
    mc_cluster_profile_tf = MotifCompendium.load(paths.mc_cluster_profile_tf_path, safe=args.safe)
    mc_contribcluster_profile_tf = MotifCompendium.load(paths.mc_contribcluster_profile_tf_path, safe=args.safe)

    mc_modelcluster_tf = MotifCompendium.load(paths.mc_modelcluster_tf_path, safe=args.safe)


    mc_counts_chrom = MotifCompendium.load(paths.mc_counts_chrom_path, safe=args.safe)
    mc_cluster_counts_chrom = MotifCompendium.load(paths.mc_cluster_counts_chrom_path, safe=args.safe)
    mc_contribcluster_counts_chrom = MotifCompendium.load(paths.mc_contribcluster_counts_chrom_path, safe=args.safe)

    mc_profile_chrom = MotifCompendium.load(paths.mc_profile_chrom_path, safe=args.safe)
    mc_cluster_profile_chrom = MotifCompendium.load(paths.mc_cluster_profile_chrom_path, safe=args.safe)
    mc_contribcluster_profile_chrom = MotifCompendium.load(paths.mc_contribcluster_profile_chrom_path, safe=args.safe)

    mc_modelcluster_chrom = MotifCompendium.load(paths.mc_modelcluster_chrom_path, safe=args.safe)
    mc_counts_procap = MotifCompendium.load(paths.mc_counts_procap_path, safe=args.safe)
    mc_cluster_counts_procap = MotifCompendium.load(paths.mc_cluster_counts_procap_path, safe=args.safe)
    mc_contribcluster_counts_procap = MotifCompendium.load(paths.mc_contribcluster_counts_procap_path, safe=args.safe)

    mc_profile_procap = MotifCompendium.load(paths.mc_profile_procap_path, safe=args.safe)
    mc_cluster_profile_procap = MotifCompendium.load(paths.mc_cluster_profile_procap_path, safe=args.safe)
    mc_contribcluster_profile_procap = MotifCompendium.load(paths.mc_contribcluster_profile_procap_path, safe=args.safe)

    mc_modelcluster_procap = MotifCompendium.load(paths.mc_modelcluster_procap_path, safe=args.safe)

    # The legacy Chrom profile indexed compendium predates the modelcluster
    # annotation that the corresponding TF profile already carries. Its
    # motifs still retain their contribcluster assignment, so restore the
    # derived modelcluster name before applying the shared ENCODE mapping.
    if VisualizeArgs.modelcluster_name_col not in mc_profile_chrom.columns():
        contribcluster_to_modelcluster = dict(zip(
            mc_contribcluster_profile_chrom[MetadataCols.name_col],
            mc_contribcluster_profile_chrom[VisualizeArgs.modelcluster_name_col],
        ))
        mc_profile_chrom[VisualizeArgs.modelcluster_name_col] = mc_profile_chrom[
            VisualizeArgs.contribcluster_name_col
        ].map(contribcluster_to_modelcluster)
        if mc_profile_chrom[VisualizeArgs.modelcluster_name_col].notna().sum() == 0:
            raise ValueError(
                "Could not recover modelcluster_name for the legacy Chrom "
                "profile compendium from its contribcluster assignments."
            )

    # Update Modelclusters
    for mc_modelcluster_i in [mc_modelcluster_tf, mc_modelcluster_chrom, mc_modelcluster_procap]:
        for metadata_col in mc_modelcluster_combined.columns():
            if metadata_col not in mc_modelcluster_i.columns():
                mc_modelcluster_i.metadata[metadata_col] = None
        mc_modelcluster_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_modelcluster_i.metadata.update(mc_modelcluster_combined.metadata.set_index(MetadataCols.unique_id_col))
        mc_modelcluster_i.metadata.reset_index(inplace=True)

    ## Map: Modelcluster to crosscluster: Name, Simple name
    mc_modelcluster_tf[VisualizeArgs.crosscluster_name_col] = mc_modelcluster_tf[crosscluster_cluster_col].map(modelcluster2crosscluster_name_map)
    mc_modelcluster_tf[MetadataCols.simple_name_col] = mc_modelcluster_tf[crosscluster_cluster_col].map(modelcluster2crosscluster_simple_map).fillna(mc_modelcluster_tf[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)

    mc_modelcluster_chrom[VisualizeArgs.crosscluster_name_col] = mc_modelcluster_chrom[crosscluster_cluster_col].map(modelcluster2crosscluster_name_map)
    mc_modelcluster_chrom[MetadataCols.simple_name_col] = mc_modelcluster_chrom[crosscluster_cluster_col].map(modelcluster2crosscluster_simple_map).fillna(mc_modelcluster_chrom[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)
    mc_modelcluster_procap[VisualizeArgs.crosscluster_name_col] = mc_modelcluster_procap[crosscluster_cluster_col].map(modelcluster2crosscluster_name_map)
    mc_modelcluster_procap[MetadataCols.simple_name_col] = mc_modelcluster_procap[crosscluster_cluster_col].map(modelcluster2crosscluster_simple_map).fillna(mc_modelcluster_procap[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)

    ## Map: Motif, Cluster, Contribcluster to crosscluster: Name, Simple name
    contribcluster2modelcluster_crosscluster_map = dict(zip(mc_modelcluster_combined[MetadataCols.name_col], mc_modelcluster_combined[VisualizeArgs.crosscluster_name_col]))
    contribcluster2modelcluster_simple_map = dict(zip(mc_modelcluster_combined[MetadataCols.name_col], mc_modelcluster_combined[MetadataCols.simple_name_col]))

    ##  Apply map: Motif, Cluster, Contribcluster to crosscluster: Name, Simple name
    mc_contribcluster_counts_tf[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_counts_tf[MetadataCols.simple_name_col] = mc_contribcluster_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_contribcluster_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_tf[VisualizeArgs.crosscluster_name_col] = mc_cluster_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_counts_tf[MetadataCols.simple_name_col] = mc_cluster_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_cluster_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_tf[VisualizeArgs.crosscluster_name_col] = mc_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_counts_tf[MetadataCols.simple_name_col] = mc_counts_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_counts_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_counts_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_contribcluster_profile_tf[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_profile_tf[MetadataCols.simple_name_col] = mc_contribcluster_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_contribcluster_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_tf[VisualizeArgs.crosscluster_name_col] = mc_cluster_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_profile_tf[MetadataCols.simple_name_col] = mc_cluster_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_cluster_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_tf[VisualizeArgs.crosscluster_name_col] = mc_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_profile_tf[MetadataCols.simple_name_col] = mc_profile_tf[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_profile_tf[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_profile_tf.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')


    mc_contribcluster_counts_chrom[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_counts_chrom[MetadataCols.simple_name_col] = mc_contribcluster_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_contribcluster_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_chrom[VisualizeArgs.crosscluster_name_col] = mc_cluster_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_counts_chrom[MetadataCols.simple_name_col] = mc_cluster_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_cluster_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_chrom[VisualizeArgs.crosscluster_name_col] = mc_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_counts_chrom[MetadataCols.simple_name_col] = mc_counts_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_counts_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_counts_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_contribcluster_profile_chrom[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_profile_chrom[MetadataCols.simple_name_col] = mc_contribcluster_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_contribcluster_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_chrom[VisualizeArgs.crosscluster_name_col] = mc_cluster_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_profile_chrom[MetadataCols.simple_name_col] = mc_cluster_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_cluster_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_chrom[VisualizeArgs.crosscluster_name_col] = mc_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_profile_chrom[MetadataCols.simple_name_col] = mc_profile_chrom[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_profile_chrom[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_profile_chrom.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_contribcluster_counts_procap[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_counts_procap[MetadataCols.simple_name_col] = mc_contribcluster_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_contribcluster_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_counts_procap[VisualizeArgs.crosscluster_name_col] = mc_cluster_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_counts_procap[MetadataCols.simple_name_col] = mc_cluster_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_cluster_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_counts_procap[VisualizeArgs.crosscluster_name_col] = mc_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_counts_procap[MetadataCols.simple_name_col] = mc_counts_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_counts_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_counts_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_counts_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    mc_contribcluster_profile_procap[VisualizeArgs.crosscluster_name_col] = mc_contribcluster_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_contribcluster_profile_procap[MetadataCols.simple_name_col] = mc_contribcluster_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_contribcluster_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_contribcluster_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_contribcluster_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_cluster_profile_procap[VisualizeArgs.crosscluster_name_col] = mc_cluster_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_cluster_profile_procap[MetadataCols.simple_name_col] = mc_cluster_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_cluster_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_cluster_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_cluster_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')
    mc_profile_procap[VisualizeArgs.crosscluster_name_col] = mc_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_crosscluster_map)
    mc_profile_procap[MetadataCols.simple_name_col] = mc_profile_procap[VisualizeArgs.modelcluster_name_col].map(contribcluster2modelcluster_simple_map).fillna(mc_profile_procap[FilterArgs.filter_flag_type_col]).fillna(MetadataCols.name_unknown)
    mc_profile_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id
    mc_profile_procap.metadata.drop(columns=EncodeCombineArgs.remove_cols, inplace=True, errors='ignore')

    ## Filter flag: If no crosscluster_name_col, then True
    mc_counts_tf[FilterArgs.filter_flag_col] = ~mc_counts_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_counts_tf[FilterArgs.filter_flag_col] = ~mc_cluster_counts_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_counts_tf[FilterArgs.filter_flag_col] = ~mc_contribcluster_counts_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_profile_tf[FilterArgs.filter_flag_col] = ~mc_profile_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_profile_tf[FilterArgs.filter_flag_col] = ~mc_cluster_profile_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_profile_tf[FilterArgs.filter_flag_col] = ~mc_contribcluster_profile_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_modelcluster_tf[FilterArgs.filter_flag_col] = ~mc_modelcluster_tf[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])


    mc_counts_chrom[FilterArgs.filter_flag_col] = ~mc_counts_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_counts_chrom[FilterArgs.filter_flag_col] = ~mc_cluster_counts_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_counts_chrom[FilterArgs.filter_flag_col] = ~mc_contribcluster_counts_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_profile_chrom[FilterArgs.filter_flag_col] = ~mc_profile_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_profile_chrom[FilterArgs.filter_flag_col] = ~mc_cluster_profile_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_profile_chrom[FilterArgs.filter_flag_col] = ~mc_contribcluster_profile_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_modelcluster_chrom[FilterArgs.filter_flag_col] = ~mc_modelcluster_chrom[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_counts_procap[FilterArgs.filter_flag_col] = ~mc_counts_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_counts_procap[FilterArgs.filter_flag_col] = ~mc_cluster_counts_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_counts_procap[FilterArgs.filter_flag_col] = ~mc_contribcluster_counts_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_profile_procap[FilterArgs.filter_flag_col] = ~mc_profile_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_cluster_profile_procap[FilterArgs.filter_flag_col] = ~mc_cluster_profile_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])
    mc_contribcluster_profile_procap[FilterArgs.filter_flag_col] = ~mc_contribcluster_profile_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    mc_modelcluster_procap[FilterArgs.filter_flag_col] = ~mc_modelcluster_procap[VisualizeArgs.crosscluster_name_col].isin(mc_crosscluster_filtered[MetadataCols.name_col])

    ## Confidence: If no crosscluster_name_col, then Flag
    mc_crosscluster_filtered_metadata = mc_crosscluster_filtered.metadata.set_index(MetadataCols.name_col)

    mc_counts_tf.metadata[LabelCols.label_conf_col] = mc_counts_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_tf.metadata[LabelCols.label_conf_col] = mc_cluster_counts_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_counts_tf.metadata[LabelCols.label_conf_col] = mc_contribcluster_counts_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_tf.metadata[LabelCols.label_conf_col] = mc_profile_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_tf.metadata[LabelCols.label_conf_col] = mc_cluster_profile_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_profile_tf.metadata[LabelCols.label_conf_col] = mc_contribcluster_profile_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_modelcluster_tf.metadata[LabelCols.label_conf_col] = mc_modelcluster_tf.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])


    mc_counts_chrom.metadata[LabelCols.label_conf_col] = mc_counts_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_chrom.metadata[LabelCols.label_conf_col] = mc_cluster_counts_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_counts_chrom.metadata[LabelCols.label_conf_col] = mc_contribcluster_counts_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_chrom.metadata[LabelCols.label_conf_col] = mc_profile_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_chrom.metadata[LabelCols.label_conf_col] = mc_cluster_profile_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_profile_chrom.metadata[LabelCols.label_conf_col] = mc_contribcluster_profile_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_modelcluster_chrom.metadata[LabelCols.label_conf_col] = mc_modelcluster_chrom.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_counts_procap.metadata[LabelCols.label_conf_col] = mc_counts_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_counts_procap.metadata[LabelCols.label_conf_col] = mc_cluster_counts_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_counts_procap.metadata[LabelCols.label_conf_col] = mc_contribcluster_counts_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_profile_procap.metadata[LabelCols.label_conf_col] = mc_profile_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_cluster_profile_procap.metadata[LabelCols.label_conf_col] = mc_cluster_profile_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    mc_contribcluster_profile_procap.metadata[LabelCols.label_conf_col] = mc_contribcluster_profile_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    mc_modelcluster_procap.metadata[LabelCols.label_conf_col] = mc_modelcluster_procap.metadata[VisualizeArgs.crosscluster_name_col].map(mc_crosscluster_filtered_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])

    # Motif index: every atlas's motifs, per head
    motif_index_parts = [
        (EncodeCombineArgs.tfatlas_id, 'counts', mc_counts_tf),
        (EncodeCombineArgs.tfatlas_id, 'profile', mc_profile_tf),
        (EncodeCombineArgs.chromatlas_id, 'counts', mc_counts_chrom),
        (EncodeCombineArgs.chromatlas_id, 'profile', mc_profile_chrom),
        (EncodeCombineArgs.procapatlas_id, 'counts', mc_counts_procap),
        (EncodeCombineArgs.procapatlas_id, 'profile', mc_profile_procap),
    ]
    mc_motif_index = pd.concat([mc.metadata for _, _, mc in motif_index_parts], axis=0).reset_index(drop=True)

    # Add motif category type: "direct-tf", "indirect-tf", "tf", "non-canonical"
    mc_motif_index[DirectTypeArgs.category_type_col] = motif_category_types(
        mc_motif_index, VisualizeArgs.crosscluster_name_col, FilterArgs.filter_flag_col, mc_crosscluster_filtered[MetadataCols.name_col])

    # Export: Motif index
    mc_motif_index[VisualizeArgs.motif_index_cols].to_csv(paths.metadata_motif_index_path, sep="\t", index=False)
    # mc_motif_index[motif_cateogory_cols].to_csv(mc_motif_cateogory_tsv, sep="\t", index=False)

    # Export: the atlas pipelines' motif indexes (each head's motifs: <head>/metadata/motifcompendium_motif_index.tsv;
    # both heads': <combined>/metadata/motifcompendium_modelcluster_motif_index.tsv), over their files: their own
    # columns, then the ENCODE motif index's they lack (category, confidence, filter flag, crosscluster)
    def export_atlas_motif_index(motif_index_df, path):
        own_cols = list(pd.read_csv(path, sep="\t", nrows=0).columns) if os.path.exists(path) else []
        missing_cols = [col for col in own_cols if col not in motif_index_df.columns]
        if missing_cols:
            logging.warning(f"Motif index {path}: columns no longer in the metadata, not written: {missing_cols}")
        cols = [col for col in own_cols if col in motif_index_df.columns]
        cols += [col for col in VisualizeArgs.motif_index_cols if col not in cols]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        motif_index_df[cols].to_csv(path, sep="\t", index=False)
        if args.verbose:
            logging.info(f'Saved at: {path}')

    motif_index_atlas = {}
    motif_index_ends = np.cumsum([len(mc) for _, _, mc in motif_index_parts])
    for (atlas, head, _), end, start in zip(motif_index_parts, motif_index_ends, np.r_[0, motif_index_ends[:-1]]):
        motif_index_head = mc_motif_index.iloc[start:end]
        motif_index_atlas.setdefault(atlas, []).append(motif_index_head)
        export_atlas_motif_index(motif_index_head, os.path.join(paths.mc_dirs[atlas][head], "metadata", "motifcompendium_motif_index.tsv"))
    for atlas, motif_index_heads in motif_index_atlas.items():
        export_atlas_motif_index(
            pd.concat(motif_index_heads, axis=0).reset_index(drop=True),
            os.path.join(paths.mc_dirs[atlas]['combined'], "metadata", "motifcompendium_modelcluster_motif_index.tsv"),
        )

    # MotifCompendium objects
    mc_alls = {
        'TF': {
            'combined': mc_modelcluster_tf,
            'counts': {
                'contribcluster': mc_contribcluster_counts_tf,
                'cluster': mc_cluster_counts_tf,
                'motif': mc_counts_tf,
            },
            'profile': {
                'contribcluster': mc_contribcluster_profile_tf,
                'cluster': mc_cluster_profile_tf,
                'motif': mc_profile_tf,
            },
        },
        'Chrom': {
            'combined': mc_modelcluster_chrom,
            'counts': {
                'contribcluster': mc_contribcluster_counts_chrom,
                'cluster': mc_cluster_counts_chrom,
                'motif': mc_counts_chrom,
            },
            'profile': {
                'contribcluster': mc_contribcluster_profile_chrom,
                'cluster': mc_cluster_profile_chrom,
                'motif': mc_profile_chrom,
            },
        },
        'ProCap': {
            'combined': mc_modelcluster_procap,
            'counts': {
                'contribcluster': mc_contribcluster_counts_procap,
                'cluster': mc_cluster_counts_procap,
                'motif': mc_counts_procap,
            },
            'profile': {
                'contribcluster': mc_contribcluster_profile_procap,
                'cluster': mc_cluster_profile_procap,
                'motif': mc_profile_procap,
            },
        },
    }

    level_suffixs = {
        'contribcluster': EncodeCombineArgs.suffix_contribcluster,
        'cluster': EncodeCombineArgs.suffix_cluster,
        'motif': EncodeCombineArgs.suffix_motif,
    }
    mc_suffixs = {
        atlas: {'combined': EncodeCombineArgs.suffix_modelcluster, 'counts': level_suffixs, 'profile': level_suffixs}
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



    # Export per-atlas compendia, back into the upstream atlas pipeline directories.
    # One pass over every (atlas, head, level, flag) combination writes all four
    # artefacts, so each flag slice is taken once instead of once per artefact.
    # Flags: _filtered, _removed and _all (every row: the level's motifcompendium_*_all.*).
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
                column_desc_dict=html_column_desc_dict_crosscluster,
            )
            if args.verbose:
                logging.info(f'Saved at: {path}')

        # MotifCompendium object
        path = out_path('mc', '', '.mc')
        save_replace(mc_level_flag, path)
        if args.verbose:
            logging.info(f'Saved at: {path}')


    for atlas, atlas_dict in mc_alls.items():
        # Combined (modelcluster) level
        level = 'combined'
        mc_level = atlas_dict[level]
        for suffix_flag, flag_bool in EncodeCombineArgs.flags:
            if flag_bool is True and not getattr(args, "save_all", False):
                continue  # the removed ones: --save-all
            export_level_flag(
                mc_level_flag=mc_level if flag_bool is None else mc_level[mc_level[FilterArgs.filter_flag_col] == flag_bool],
                mc_level_dir=paths.mc_dirs[atlas][level],
                mc_level_suffix=mc_suffixs[atlas][level],
                suffix_flag=suffix_flag,
                level=level,
            )

        # Head: counts, profile
        for head in ['counts', 'profile']:
            for level, mc_level in atlas_dict[head].items():
                for suffix_flag, flag_bool in EncodeCombineArgs.flags:
                    if flag_bool is True and not getattr(args, "save_all", False):
                        continue  # the removed ones: --save-all
                    export_level_flag(
                        mc_level_flag=mc_level if flag_bool is None else mc_level[mc_level[FilterArgs.filter_flag_col] == flag_bool],
                        mc_level_dir=paths.mc_dirs[atlas][head],
                        mc_level_suffix=mc_suffixs[atlas][head][level],
                        suffix_flag=suffix_flag,
                        level=level,
                    )


def export_encodecluster_level(args, cpaths, mc_encodecluster, membership, html_column_desc_dict):
    """
    The encodecluster level (C5: pipeline_encode_composite.py's singles and merged composites) as an crosscluster level, in
    <composite-dir> (cpaths): motifcompendium_encode (.mc, HTML, metadata TSV, MEME; nothing is removed at
    this level: the crossclusters not in it are the combine's, C4), and the membership of the crossclusters (one row
    per C4 crosscluster: its encodecluster, or why it has none). The levels below:
    export_encodecluster_constituents().
    """
    name_col, flag_col = MetadataCols.name_col, FilterArgs.filter_flag_col
    mc_encodecluster[flag_col] = mc_encodecluster[flag_col].astype(bool)

    # Logos of the rows without one (singles and composites combined with and without them): summary_table_html() draws
    # them only for a compendium without the image column
    for image_name, reverse in (("logo (fwd)", False), ("logo (rev)", True)):
        if image_name not in mc_encodecluster.images():
            continue
        images = list(mc_encodecluster.get_images(image_name))
        missing = [i for i, image in enumerate(images) if image is None]
        if missing:
            mc_missing = mc_encodecluster[missing]
            mc_missing.delete_images(image_name)
            motifs = mc_missing.get_standard_motif_stack()
            mc_missing.add_logos(utils_motif.reverse_complement(motifs) if reverse else motifs, image_name, VisualizeArgs.html_logo_trim)
            for i, image in zip(missing, mc_missing.get_images(image_name)):
                images[i] = image
            mc_encodecluster.set_images(image_name, images)
            logging.info(f"{image_name}: drawn for the {len(missing)} encodeclusters without one")

    #### The encodecluster level --------------------------------------------------------------------
    logging.info(f"Encodecluster level: {len(mc_encodecluster)} encodeclusters -> {cpaths.composite_dir}")
    html_columns_level = [col for col in VisualizeArgs.html_columns if col in (mc_encodecluster.columns() + mc_encodecluster.images())]
    html_column_desc_dict_level = {col: html_column_desc_dict.get(col, col) for col in html_columns_level}
    for path in (cpaths.mc_encodecluster_path, cpaths.html_encodecluster_path,
                 cpaths.metadata_encodecluster_path, cpaths.meme_encodecluster_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    mc_encodecluster.summary_table_html(
        html_out=cpaths.html_encodecluster_path,
        columns=html_columns_level,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_level,
    )
    utils_analysis.export_compendium_meme(
        mc=mc_encodecluster,
        name_col=name_col,
        save_loc=cpaths.meme_encodecluster_path,
        trim_importance=(1/(2*mc_encodecluster.motifs.shape[1])),
    )
    mc_encodecluster.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(
        cpaths.metadata_encodecluster_path, sep="\t", index=False)
    save_replace(mc_encodecluster, cpaths.mc_encodecluster_path)

    # Membership: each crosscluster (C4), with its encodecluster (or why it has none)
    os.makedirs(os.path.dirname(cpaths.metadata_encodecluster_membership_path), exist_ok=True)
    membership.to_csv(cpaths.metadata_encodecluster_membership_path, sep="\t", index=False)


def export_encodecluster_constituents(args, cpaths, mc_encodecluster, membership):
    """
    The encodecluster level (export_encodecluster_level()) added to every level below, as
    export_crosscluster_constituents() adds the crossclusters, but keeping theirs (--update-constituents only: on its own,
    from the level's saved outputs, run_encodecluster_constituents()): the crossclusters (C4, <output-dir>) get the
    number, name and filter flag (True when in no encodecluster) of their encodecluster
    (VisualizeArgs.encodecluster_{cluster,name,flag}_col); every atlas level (by its crosscluster) and the motif
    indexes its name and filter flag (each level's crosscluster: export_crosscluster_constituents()). Their names, MCIDs,
    flags and confidence are those of the ENCODE stage; .mc and metadata TSV rewritten (their HTML and MEME do not show
    the new columns). Then the motif index of the encodecluster level (the hits annotation's).
    membership: one row per crosscluster (its MCID and name; export_encodecluster_level()).
    """
    paths = crosscluster_paths(args)
    name_col, id_col, flag_col = MetadataCols.name_col, MetadataCols.unique_id_col, FilterArgs.filter_flag_col
    flag_type_col = FilterArgs.filter_flag_type_col
    encodecluster_name_col = VisualizeArgs.encodecluster_name_col
    encodecluster_flag_col = VisualizeArgs.encodecluster_flag_col
    encodecluster_cluster_col = VisualizeArgs.encodecluster_cluster_col
    encodecluster_flag_type_col = VisualizeArgs.encodecluster_flag_type_col
    in_encodecluster = membership[encodecluster_cluster_col].notna()
    membership_by_id = membership.set_index(id_col)
    encodecluster_name_by_crosscluster_name = dict(zip(membership.loc[in_encodecluster, name_col],
                                                      membership.loc[in_encodecluster, encodecluster_name_col]))
    # Why a crosscluster of the combine is in no encodecluster, when the combine kept it (left out of C5)
    left_out = ~in_encodecluster & membership[encodecluster_flag_type_col].isin(
        [CompositeArgs.flag_type_no_synergy, CompositeArgs.flag_type_unannotated])
    left_out_type_by_crosscluster_name = dict(zip(membership.loc[left_out, name_col],
                                            membership.loc[left_out, encodecluster_flag_type_col]))

    def add_encodecluster_cols(metadata_df, by_crosscluster_name_col):
        """Its encodecluster's name and filter flag (True: none), by its crosscluster's name."""
        metadata_df[encodecluster_name_col] = metadata_df[by_crosscluster_name_col].map(encodecluster_name_by_crosscluster_name)
        metadata_df[encodecluster_flag_col] = metadata_df[encodecluster_name_col].isna()
        return metadata_df

    #### The crossclusters (C4): the number, name and flag of their encodecluster -------
    logging.info(f"crossclusters (C4), with their encodecluster: {args.output_dir}")
    for mc_path, metadata_path in ((paths.mc_crosscluster_path, paths.metadata_crosscluster_path),
                                   (paths.mc_crosscluster_filtered_path, paths.metadata_crosscluster_filtered_path),
                                   (paths.mc_crosscluster_removed_path, paths.metadata_crosscluster_removed_path)):
        if not os.path.exists(mc_path):  # the removed ones: --save-all
            continue
        mc_level = MotifCompendium.load(mc_path, safe=args.safe)
        crosscluster_ids = mc_level[id_col].astype(str)
        if not crosscluster_ids.isin(membership_by_id.index).all():
            raise ValueError(f"{mc_path}: crossclusters not in the membership, e.g. "
                             f"{crosscluster_ids[~crosscluster_ids.isin(membership_by_id.index)].tolist()[:3]}")
        numbers = crosscluster_ids.map(membership_by_id[encodecluster_cluster_col])
        mc_level[encodecluster_cluster_col] = np.array([None if pd.isna(n) else int(n) for n in numbers], dtype=object)
        mc_level[encodecluster_name_col] = crosscluster_ids.map(membership_by_id[encodecluster_name_col]).to_numpy()
        mc_level[encodecluster_flag_col] = numbers.isna().to_numpy()
        mc_level.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(metadata_path, sep="\t", index=False)
        save_replace(mc_level, mc_path)
    # The modelclusters the crossclusters were combined from
    mc_level = MotifCompendium.load(paths.mc_modelcluster_combined_path, safe=args.safe)
    add_encodecluster_cols(mc_level.metadata, VisualizeArgs.crosscluster_name_col)
    mc_level.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_modelcluster_combined_path, sep="\t", index=False)
    save_replace(mc_level, paths.mc_modelcluster_combined_path)

    #### Every atlas level, by its crosscluster: the name and flag of its encodecluster -
    #   File names as export_crosscluster_constituents(): {file_general}{level_suffix}{flag_suffix}{suffix}{filetype}
    logging.info("Atlas levels, with their encodecluster...")
    level_suffixs = {'contribcluster': EncodeCombineArgs.suffix_contribcluster, 'cluster': EncodeCombineArgs.suffix_cluster,
                     'motif': EncodeCombineArgs.suffix_motif}
    for atlas, atlas_dirs in paths.mc_dirs.items():
        levels = [(atlas_dirs['combined'], EncodeCombineArgs.suffix_modelcluster)] + [
            (atlas_dirs[head], level_suffix) for head in ('counts', 'profile') for level_suffix in level_suffixs.values()]
        for level_dir, level_suffix in levels:
            for suffix_flag, flag_bool in EncodeCombineArgs.flags:
                file_name = EncodeCombineArgs.file_general + level_suffix + suffix_flag
                mc_path = os.path.join(level_dir, "mc", file_name + ".mc")
                metadata_path = os.path.join(level_dir, "metadata", file_name + EncodeCombineArgs.suffix_metadata + ".tsv")
                if not os.path.exists(mc_path):
                    if not (flag_bool is True and not getattr(args, "save_all", False)):  # the removed ones: --save-all
                        logging.warning(f"Not found (skipped): {mc_path}")
                    continue
                mc_level = MotifCompendium.load(mc_path, safe=args.safe)
                add_encodecluster_cols(mc_level.metadata, VisualizeArgs.crosscluster_name_col)
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                mc_level.metadata.to_csv(metadata_path, sep="\t", index=False)
                save_replace(mc_level, mc_path)
                if args.verbose:
                    logging.info(f'Saved at: {mc_path}')

    #### Motif indexes: the ENCODE one and the atlas pipelines', with their encodecluster --
    motif_index_df = pd.read_csv(paths.metadata_motif_index_path, sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
    add_encodecluster_cols(motif_index_df, VisualizeArgs.crosscluster_name_col).to_csv(paths.metadata_motif_index_path, sep="\t", index=False)
    for atlas, atlas_dirs in paths.mc_dirs.items():
        for atlas_index_path in [os.path.join(atlas_dirs[head], "metadata", "motifcompendium_motif_index.tsv") for head in ('counts', 'profile')] + [
                os.path.join(atlas_dirs['combined'], "metadata", "motifcompendium_modelcluster_motif_index.tsv")]:
            if os.path.exists(atlas_index_path):
                atlas_index_df = pd.read_csv(atlas_index_path, sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
                add_encodecluster_cols(atlas_index_df, VisualizeArgs.crosscluster_name_col).to_csv(atlas_index_path, sep="\t", index=False)

    # The encodecluster level's motif index: each motif with its encodecluster (encodecluster_name; by
    # it the hits annotation, pipeline_encode_hits.py) and the encodecluster's filter flag, confidence, simple name,
    # flag type and category in the motif index's columns (crosscluster_name: its crosscluster, as in the ENCODE motif
    # index). A motif in no encodecluster keeps its crosscluster's simple name and flag type (the combine's
    # removed ones), or gets why its crosscluster was left out (no-synergy, composite-unannotated): its hits' name
    encodecluster_metadata = mc_encodecluster.metadata.set_index(name_col)
    encodecluster_names = motif_index_df[encodecluster_name_col]
    motif_index_encodecluster_df = motif_index_df.copy()
    motif_index_encodecluster_df[flag_col] = motif_index_df[encodecluster_flag_col]
    motif_index_encodecluster_df[LabelCols.label_conf_col] = encodecluster_names.map(
        encodecluster_metadata[LabelCols.label_conf_col]).fillna(LabelCols.label_confs[-1])
    left_out_type = motif_index_df[VisualizeArgs.crosscluster_name_col].map(left_out_type_by_crosscluster_name)
    motif_index_encodecluster_df[MetadataCols.simple_name_col] = encodecluster_names.map(
        encodecluster_metadata[MetadataCols.simple_name_col]).fillna(left_out_type).fillna(motif_index_df[MetadataCols.simple_name_col])
    motif_index_encodecluster_df[flag_type_col] = left_out_type.fillna(motif_index_df[flag_type_col])
    motif_index_encodecluster_df[DirectTypeArgs.category_type_col] = motif_category_types(
        motif_index_encodecluster_df, encodecluster_name_col, flag_col, mc_encodecluster[name_col])
    index_cols = list(VisualizeArgs.motif_index_cols)
    index_cols.insert(index_cols.index(VisualizeArgs.crosscluster_name_col) + 1, encodecluster_name_col)
    motif_index_encodecluster_df[index_cols].to_csv(cpaths.metadata_encodecluster_motif_index_path, sep="\t", index=False)
    logging.info(f"Motif index (encodecluster): {cpaths.metadata_encodecluster_motif_index_path}")


def run_crosscluster_constituents(args):
    """--update-constituents only: the crossclusters propagated to every level below (export_crosscluster_constituents()), from the
    combine's saved outputs (<output-dir>/mc: motifcompendium_crosscluster_all.mc, motifcompendium_modelcluster_combined.mc)."""
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_encode_constituents_{os.path.basename(os.path.normpath(args.output_dir))}"
        f"_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.out",
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    )
    logging.info(f"Run started (the levels below the crossclusters only): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logging.info(f"Writing run log to: {log_path}")
    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        ic_scale=args.ic,
        fast_plotting=args.fast_plot,
    )
    paths = crosscluster_paths(args)
    mc_crosscluster = MotifCompendium.load(paths.mc_crosscluster_path, safe=args.safe)
    mc_crosscluster[FilterArgs.filter_flag_col] = mc_crosscluster[FilterArgs.filter_flag_col].astype(bool)
    mc_modelcluster_combined = MotifCompendium.load(paths.mc_modelcluster_combined_path, safe=args.safe)
    html_column_desc_dict = pd.read_csv(args.column_desc_tsv, sep="\t", header=0).set_index('column_name')['column_desc'].to_dict()
    export_crosscluster_constituents(
        args,
        mc_crosscluster,
        mc_modelcluster_combined,
        dict(zip(mc_crosscluster[ClusterArgs.source_cluster_col], mc_crosscluster[MetadataCols.name_col])),
        dict(zip(mc_crosscluster[ClusterArgs.source_cluster_col], mc_crosscluster[MetadataCols.simple_name_col])),
        crosscluster_cluster_col_name(),
        html_column_desc_dict,
    )
    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def run_encodecluster_constituents(args):
    """--update-constituents only (pipeline_encode_composite.py): the encodecluster level added to every level below
    (export_encodecluster_constituents()), from its saved outputs (<composite-dir>: the level's .mc and the
    membership of the crossclusters)."""
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_encodecluster_constituents_{os.path.basename(os.path.normpath(args.output_dir))}"
        f"_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.out",
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    )
    logging.info(f"Run started (the levels below the encodeclusters only): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logging.info(f"Writing run log to: {log_path}")
    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        ic_scale=args.ic,
        fast_plotting=args.fast_plot,
    )
    cpaths = composite_paths(getattr(args, "composite_dir", None) or default_composite_dir(args.output_dir))
    mc_encodecluster = MotifCompendium.load(cpaths.mc_encodecluster_path, safe=args.safe)
    membership = pd.read_csv(cpaths.metadata_encodecluster_membership_path, sep="\t", keep_default_na=False, na_values=[""])
    export_encodecluster_constituents(args, cpaths, mc_encodecluster, membership)
    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def run_encode_combine(args):
    """ENCODE: combine the TF, Chromatin and ProCap modelclusters into crossclusters"""


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
    crosscluster_cluster_col = crosscluster_cluster_col_name()

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()


    ## Inputs and outputs (crosscluster_paths())
    paths = crosscluster_paths(args)
    mc_modelcluster_filtered_tf_path = paths.mc_modelcluster_filtered_tf_path
    mc_modelcluster_filtered_chrom_path = paths.mc_modelcluster_filtered_chrom_path
    mc_modelcluster_filtered_procap_path = paths.mc_modelcluster_filtered_procap_path
    args_json_path = paths.args_json_path

    # ## Combine modelcluster
    logging.info("Loading TF, Chromatin and ProCap modelcluster compendia...")

    # ### Load MotifCompendium
    # Load MotifCompendium
    mc_modelcluster_filtered_tf = MotifCompendium.load(mc_modelcluster_filtered_tf_path, safe=args.safe)
    mc_modelcluster_filtered_chrom = MotifCompendium.load(mc_modelcluster_filtered_chrom_path, safe=args.safe)
    mc_modelcluster_filtered_procap = MotifCompendium.load(mc_modelcluster_filtered_procap_path, safe=args.safe)

    # The atlases' own selection: once an ENCODE combine has written its final flags back,
    # motifcompendium_modelcluster_filtered.mc holds those instead, and this stage would recluster an already-filtered selection
    for mc_input, mc_input_path in (
        (mc_modelcluster_filtered_tf, mc_modelcluster_filtered_tf_path),
        (mc_modelcluster_filtered_chrom, mc_modelcluster_filtered_chrom_path),
        (mc_modelcluster_filtered_procap, mc_modelcluster_filtered_procap_path),
    ):
        if VisualizeArgs.crosscluster_name_col in mc_input.columns():
            raise ValueError(
                f"{mc_input_path} holds the final flags of an earlier ENCODE combine ({VisualizeArgs.crosscluster_name_col}), "
                "not its atlas's own selection: rerun that atlas (its per-head and combine stages) before this stage."
            )

    # Add TF, Chrom, ProCap to source, names, constituents
    mc_modelcluster_filtered_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_modelcluster_filtered_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id
    mc_modelcluster_filtered_procap[VisualizeArgs.atlas_col] = EncodeCombineArgs.procapatlas_id

    # Preserve all columns, None if not specified
    all_metadata_cols = set(mc_modelcluster_filtered_tf.columns()) | set(mc_modelcluster_filtered_chrom.columns()) | set(mc_modelcluster_filtered_procap.columns())

    for mc in [mc_modelcluster_filtered_tf, mc_modelcluster_filtered_chrom, mc_modelcluster_filtered_procap]:
        for col in sorted(all_metadata_cols):
            if col not in mc.columns():
                mc[col] = None

    # Number of FiNeMo hits: NaN, not None, for an atlas without FiNeMo paths (e.g. ProCap), so that the
    # crosscluster sums stay numeric (see the crosscluster num_hits below).
    if MetadataCols.num_hits_col in all_metadata_cols:
        for mc in [mc_modelcluster_filtered_tf, mc_modelcluster_filtered_chrom, mc_modelcluster_filtered_procap]:
            mc[MetadataCols.num_hits_col] = pd.to_numeric(mc[MetadataCols.num_hits_col], errors="coerce")

    # Delete all images, except logos
    keep_images = ['logo (fwd)', 'logo (rev)']

    for mc in [mc_modelcluster_filtered_tf, mc_modelcluster_filtered_chrom, mc_modelcluster_filtered_procap]:
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
    mc_modelcluster_combined = MotifCompendium.combine(
        [mc_modelcluster_filtered_tf, mc_modelcluster_filtered_chrom, mc_modelcluster_filtered_procap],
        safe=args.safe,
    )


    # #### Cluster: crossclusters
    logging.info("Force-clustering TF, Chromatin and ProCap modelclusters into crossclusters...")
    # Force-cluster
    mc_modelcluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_force,
        similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
        save_name=crosscluster_cluster_col,
        largest_clusters_first=True,
        density=ClusterArgs.cluster_force_density,
        seed=ClusterArgs.seeds[0],
    )
    # Reassign: K-means
    mc_modelcluster_combined.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=crosscluster_cluster_col,
        largest_clusters_first=True,
        init_clustering_col=crosscluster_cluster_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_crosscluster = mc_modelcluster_combined[crosscluster_cluster_col]
    logging.info(f"Unique clusters: {len(set(membership_crosscluster))}")

    # Cluster: Recursively
    for i in range(ClusterArgs.cluster_max_iter_force):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Force-cluster: DCC
        mc_modelcluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_on=crosscluster_cluster_col,
            cluster_on_weight=ClusterArgs.average_weight_col,
            save_name=crosscluster_cluster_col,
            largest_clusters_first=True,
            density=ClusterArgs.cluster_force_density,
            seed=ClusterArgs.seeds[0],
        )
        # Reassign: K-means
        mc_modelcluster_combined.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=crosscluster_cluster_col,
            largest_clusters_first=True,
            init_clustering_col=crosscluster_cluster_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_crosscluster = mc_modelcluster_combined[crosscluster_cluster_col]
        if args.verbose:
            logging.info(f"  Unique clusters: {len(set(new_membership_crosscluster))}")
        if np.array_equal(membership_crosscluster, new_membership_crosscluster):
            logging.info(f"Final unique clusters: {len(set(new_membership_crosscluster))}")
            break
        else:
            membership_crosscluster = new_membership_crosscluster


    # # ENCODE Atlas

    # #### Average
    # Average MotifCompendium
    aggregations_crosscluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.aggregations:
        if agg_col_in in mc_modelcluster_combined.columns():
            aggregations_crosscluster.append((agg_col_in, agg, agg_col_out))

    mc_crosscluster = mc_modelcluster_combined.cluster_averages(
        clustering=crosscluster_cluster_col,
        aggregations=aggregations_crosscluster,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Number of FiNeMo hits: NaN, not 0, for a crosscluster without hits information (only modelclusters of
    # an atlas without FiNeMo paths, e.g. ProCap; cluster_averages() sums the non-NaN values, so 0), so that
    # 1_minhits does not flag it. A mixed cluster sums the hits of its modelclusters that have them.
    if MetadataCols.num_hits_col in mc_crosscluster.columns():
        num_hits_crosscluster = mc_modelcluster_combined.metadata.groupby(crosscluster_cluster_col)[MetadataCols.num_hits_col].sum(min_count=1)
        mc_crosscluster[MetadataCols.num_hits_col] = mc_crosscluster[ClusterArgs.source_cluster_col].map(num_hits_crosscluster)

    # Weighted aggregations
    weighted_aggregations_crosscluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.weighted_aggregations:
        if agg_col_in in mc_modelcluster_combined.columns():
            weighted_aggregations_crosscluster.append((agg_col_in, agg, agg_col_out))

    for (agg_col_old, _, agg_col_new) in weighted_aggregations_crosscluster:
        aggregate_weighted_avg(
            mc=mc_modelcluster_combined,
            mc_cluster=mc_crosscluster,
            cluster_col=crosscluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )

    # Merge aggregations
    merge_aggregations_crosscluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_crosscluster.columns():
            merge_aggregations_crosscluster.append(aggregate)
        if agg_col_new not in mc_crosscluster.columns():
            mc_crosscluster.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_crosscluster:
        mc_crosscluster.metadata[agg_col_new] = mc_crosscluster.metadata[[agg_col_old, agg_col_new]].apply(
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
    mc_crosscluster[MetadataCols.total_contrib_score_ratio_col] = mc_crosscluster[MetadataCols.total_contrib_score_col] / mc_crosscluster[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_crosscluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_crosscluster.get_standard_motif_stack())  # Positive / negative
    mc_crosscluster[MetadataCols.composite_col] = mc_crosscluster[MetadataCols.composite_col].astype(bool)

    mc_crosscluster.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlet_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets
    # mc_crosscluster.sort(by=[direct_type_col, composite_col], ascending=True, inplace=True)  # Sort by direct type, composite
    mc_crosscluster.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_crosscluster[MetadataCols.unique_id_col] = unique_id_initial + f".C4.TF_Chrom_ProCap.counts_profile." + mc_crosscluster.metadata.index.map('{:08d}'.format)

    ### Aggregate: Annotations
    mc_tfatlas_metadata_updates = []
    mc_tfatlas_images_updated = pd.DataFrame(index=mc_crosscluster[MetadataCols.unique_id_col])
    mc_tfatlas_images_existing = set(mc_crosscluster.images())

    for i, row in mc_crosscluster.metadata.iterrows():
        # MotifCompendium: Subset
        mc_tfatlas_i = mc_crosscluster[[i]]
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
    mc_crosscluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_crosscluster.metadata = mc_crosscluster.metadata.combine_first(mc_tfatlas_metadata_updates_df)
    mc_crosscluster.metadata.update(mc_tfatlas_metadata_updates_df)
    mc_crosscluster.metadata.reset_index(inplace=True)
    del mc_tfatlas_metadata_updates, mc_tfatlas_metadata_updates_df

    for image_col in mc_tfatlas_images_updated.columns:
        mc_crosscluster.set_images(
            image_name=image_col,
            images=mc_tfatlas_images_updated[image_col].where(
                pd.notna(mc_tfatlas_images_updated[image_col]), None).to_list()
        )
    del mc_tfatlas_images_updated, mc_tfatlas_images_existing


    # #### Annotate
    # ### Annotate: Contribclusters
    # Initialize
    # Experiment: Determined by motif; Composite: Determined by cluster
    mc_crosscluster[LabelCols.label_col] = MetadataCols.name_unknown  # Initial readable name: Unknown
    # mc_crosscluster[direct_label_col] = None  # Inherit TF Atlas
    mc_crosscluster[LabelCols.label_core_col] = None
    mc_crosscluster[LabelCols.direct_label_core_col] = None
    mc_crosscluster[LabelCols.label_flank_col] = None
    # mc_crosscluster[ref_readable_name_col] = None

    ## [General] Label contribclusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_crosscluster,
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
        mc_crosscluster[ref_readable_name_col] = mc_crosscluster[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # cateogory: Mask
    mask_direct_crosscluster = mc_crosscluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
    mask_nondirect_crosscluster = ~mask_direct_crosscluster
    mask_monomer_crosscluster_nondirect = (
        mask_nondirect_crosscluster
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_crosscluster_nondirect = (
        mask_nondirect_crosscluster
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_crosscluster_nondirect = (
        mask_nondirect_crosscluster
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_crosscluster[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_crosscluster_nondirect = (
        mask_dimer_crosscluster_nondirect | mask_trimer_crosscluster_nondirect
    )
    mask_matched_modelcluster_nondirect = (
        mask_monomer_crosscluster_nondirect | mask_dimer_crosscluster_nondirect | mask_trimer_crosscluster_nondirect
    )

    # Dimer-style notation
    mc_crosscluster.metadata.loc[mask_matched_modelcluster_nondirect, LabelCols.label_core_col] = mc_crosscluster.metadata.loc[mask_matched_modelcluster_nondirect, ref_readable_name_cols[0]]
    mc_crosscluster.metadata.loc[mask_matched_modelcluster_nondirect, LabelCols.label_core_readable_col] = mc_crosscluster.metadata.loc[mask_matched_modelcluster_nondirect, ref_readable_name_cols[0]]
    mc_crosscluster.metadata.loc[mask_dimer_crosscluster_nondirect, LabelCols.label_flank_col] = mc_crosscluster.metadata.loc[mask_dimer_crosscluster_nondirect, ref_readable_name_cols[1]]
    mc_crosscluster.metadata.loc[mask_trimer_crosscluster_nondirect, LabelCols.label_flank_col] = mc_crosscluster.metadata.loc[mask_trimer_crosscluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_crosscluster.metadata.loc[mask_trimer_crosscluster_nondirect, ref_readable_name_cols[2]]
    # mc_crosscluster.metadata.loc[mask_dimer_crosscluster_nondirect, num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_crosscluster.metadata.loc[mask_trimer_crosscluster_nondirect, num_subunit_col] = 3
    # mc_crosscluster.metadata[composite_col] = mc_crosscluster.metadata[num_subunit_col] > min_composite_subunit

    # # Trimer-style notation
    # mc_crosscluster[ref_readable_name_col] = mc_crosscluster.metadata[
    #     [
    #         f'{ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

    # Concatenate all unique annotations
    mc_crosscluster[ReferenceMatchCols.ref_name_col] = mc_crosscluster.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### Filter
    # Calculate filters
    mc_crosscluster[FilterArgs.filter_flag_col] = False
    utils_analysis.calculate_filters(
        mc=mc_crosscluster,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )

    # Apply filters
    apply_filter_clusters(
        mc=mc_crosscluster,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # ENCODE filter: Unvalidated, Indirect, from only one experiment with under 250 seqlets
    crosscluster_filter_mask = (
        (mc_crosscluster[LabelCols.label_col] == MetadataCols.name_unknown)  # Unvalidated
        & ~(mc_crosscluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:2]), na=False))  # Keep all Direct TFs A, B-NZ, regardless of QC score
        & ~(mc_crosscluster[MetadataCols.model_col].str.contains(","))  # 
        & (mc_crosscluster[MetadataCols.num_seqlet_col] < 250)
    )
    mc_crosscluster.metadata.loc[crosscluster_filter_mask, FilterArgs.filter_flag_col] = True


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
            assign_to_mc=mc_crosscluster,
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
        mask_qc = (mc_crosscluster[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_removed)
        mc_crosscluster.metadata.loc[mask_qc, LabelCols.label_readable_col] = mc_crosscluster.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_crosscluster.metadata.loc[mask_qc, LabelCols.label_col] = mc_crosscluster.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_crosscluster.metadata.loc[mask_qc, FilterArgs.filter_flag_type_col] = mc_crosscluster.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_crosscluster.metadata.loc[mask_qc, MetadataCols.checked_col] = True

        # Filter:
        mask_filter = (
            mc_crosscluster[FilterArgs.filter_flag_col] | 
            ((mc_crosscluster[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_removed) &  # Flag if matches to removed motifs
            ~(mc_crosscluster[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:2]), na=False)))  # Keep all Direct TFs A, B-NZ, regardless of QC score
        )
        mc_crosscluster.metadata.loc[mask_filter, FilterArgs.filter_flag_col] = True

        # Update: Filter type, Core (Direct), Core, Flank, Label confidence
        mc_crosscluster.metadata.loc[mask_filter, FilterArgs.filter_flag_type_col] = mc_crosscluster.metadata.loc[mask_filter, LabelCols.label_col]
        mc_crosscluster.metadata.loc[mask_filter, LabelCols.direct_label_core_col] = None
        mc_crosscluster.metadata.loc[mask_filter, LabelCols.label_core_col] = None
        mc_crosscluster.metadata.loc[mask_filter, LabelCols.label_flank_col] = None
        mc_crosscluster.metadata.loc[mask_filter, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

    # Remove flagged clusters
    mc_crosscluster_filtered = mc_crosscluster[~mc_crosscluster[FilterArgs.filter_flag_col]]
    mc_crosscluster_removed = mc_crosscluster[mc_crosscluster[FilterArgs.filter_flag_col]]


    # #### Clustering (for sorting only)
    ## Cluster: For sorting only
    # Cluster: Leiden CPM
    logging.info("Clustering crossclusters (for sorting only)...")
    mc_crosscluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_crosscluster_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        init_clustering_col=VisualizeArgs.sort_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_crosscluster_filtered[VisualizeArgs.sort_col]
    logging.info(f"Unique clusters: {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_crosscluster_filtered.cluster(
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
        mc_crosscluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_crosscluster_filtered[VisualizeArgs.sort_col]
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
        mc_crosscluster_filtered.cluster(
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
        mc_crosscluster_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
    
        # Check convergence
        new_membership_cluster = mc_crosscluster_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters: {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # #### Final name: ENCODE Atlas
    # Count occurrence: Core TF family (Direct)
    mc_modelcluster_combined[LabelCols.direct_label_core_readable_col_set] = mc_modelcluster_combined[LabelCols.direct_label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_modelcluster_explode_core_readable_direct = mc_modelcluster_combined.metadata.explode(LabelCols.direct_label_core_readable_col_set)

    # Count occurrence: Core TF family
    mc_modelcluster_combined[LabelCols.label_core_readable_col_set] = mc_modelcluster_combined[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_modelcluster_explode_core_readable = mc_modelcluster_combined.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_modelcluster_combined[LabelCols.label_flank_col_set] = mc_modelcluster_combined[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_modelcluster_explode_flank = mc_modelcluster_combined.metadata.explode(LabelCols.label_flank_col_set)

    ## Count occurrences: Create pivot table
    # Core: Direct: Monomer
    mc_modelcluster_core_readable_direct_monomer_pivot = pd.pivot_table(
        mc_modelcluster_explode_core_readable_direct.loc[~mc_modelcluster_explode_core_readable_direct[MetadataCols.composite_col]],
        index=crosscluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_modelcluster_core_readable_direct_monomer_pivot = mc_modelcluster_core_readable_direct_monomer_pivot.div(
        mc_modelcluster_core_readable_direct_monomer_pivot.sum(axis=1), axis=0)

    # Core: Direct: All
    mc_modelcluster_core_readable_direct_pivot = pd.pivot_table(
        mc_modelcluster_explode_core_readable_direct,
        index=crosscluster_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_modelcluster_core_readable_direct_pivot = mc_modelcluster_core_readable_direct_pivot.div(
        mc_modelcluster_core_readable_direct_pivot.sum(axis=1), axis=0)

    # Core: All
    mc_modelcluster_core_readable_pivot = pd.pivot_table(
        mc_modelcluster_explode_core_readable,
        index=crosscluster_cluster_col,
        columns=LabelCols.label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_modelcluster_core_readable_pivot = mc_modelcluster_core_readable_pivot.div(
        mc_modelcluster_core_readable_pivot.sum(axis=1), axis=0)

    # Composite
    mc_modelcluster_flank_pivot = pd.pivot_table(
        mc_modelcluster_explode_flank,
        index=crosscluster_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_modelcluster_flank_pivot = mc_modelcluster_flank_pivot.div(
        mc_modelcluster_flank_pivot.sum(axis=1), axis=0)

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
            and (source_cluster in mc_modelcluster_core_readable_direct_monomer_pivot.index)
        ):
            core_readable_direct_monomer_frac = mc_modelcluster_core_readable_direct_monomer_pivot.loc[source_cluster]

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
            and (source_cluster in mc_modelcluster_core_readable_direct_pivot.index)
        ):
            core_readable_direct_frac = mc_modelcluster_core_readable_direct_pivot.loc[source_cluster]

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
            and (source_cluster in mc_modelcluster_core_readable_pivot.index)
        ):
            label_core_readable_frac = mc_modelcluster_core_readable_pivot.loc[source_cluster]
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
            and (source_cluster in mc_modelcluster_flank_pivot.index)
        ):
            composite_family_frac = mc_modelcluster_flank_pivot.loc[source_cluster]
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
    mc_crosscluster_filtered.metadata[label_cols] = pd.DataFrame(
        mc_crosscluster_filtered.metadata.apply(assign_labels, axis=1).tolist(),
        index=mc_crosscluster_filtered.metadata.index,
        columns=label_cols,
    )
    # Name: Contribcluster - Removed
    mc_crosscluster_removed[LabelCols.label_col] = mc_crosscluster_removed[FilterArgs.filter_flag_type_col]


    # #### Name
    ## Label readable
    mc_crosscluster_filtered[LabelCols.label_readable_col] = mc_crosscluster_filtered[LabelCols.label_col]
    mc_crosscluster_removed[LabelCols.label_readable_col] = mc_crosscluster_removed[LabelCols.label_col]


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
            assign_to_mc=mc_crosscluster_filtered,
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
        mask_qc = mc_crosscluster_filtered[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_filtered
        mc_crosscluster_filtered.metadata.loc[mask_qc, LabelCols.label_readable_col] = mc_crosscluster_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_crosscluster_filtered.metadata.loc[mask_qc, LabelCols.label_col] = mc_crosscluster_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0']
        mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace("-like", "", regex=True)

        mc_crosscluster_filtered.metadata.loc[mask_qc, MetadataCols.composite_col] = mc_crosscluster_filtered.metadata.loc[mask_qc, f'{MetadataCols.qc_col}_name0'].str.contains("::")
        mask_qc_composite = (mask_qc) & (mc_crosscluster_filtered[MetadataCols.composite_col])
        mc_crosscluster_filtered.metadata.loc[mask_qc_composite, MetadataCols.num_subunit_col] = 2

        mc_crosscluster_filtered.metadata.loc[mask_qc, MetadataCols.checked_col] = True


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
        mc_crosscluster_metadata_edited = pd.read_csv(args.metadata_edited, index_col=0)

        # ``label`` is the canonical editable QC field.  Accept
        # ``label_readable`` only for legacy curated metadata exports, such
        # as the v23 sheet used for this run.
        if LabelCols.label_col in mc_crosscluster_metadata_edited.columns:
            edited_label_col = LabelCols.label_col
        elif LabelCols.label_readable_col in mc_crosscluster_metadata_edited.columns:
            edited_label_col = LabelCols.label_readable_col
            mc_crosscluster_metadata_edited[LabelCols.label_col] = mc_crosscluster_metadata_edited[edited_label_col]
        else:
            raise ValueError(
                "Edited metadata must contain the canonical "
                f"'{LabelCols.label_col}' column."
            )

        # Apply Filter: Removed
        mc_crosscluster_metadata_edited[FilterArgs.filter_flag_col] = mc_crosscluster_metadata_edited[edited_label_col].isin(FilterArgs.filter_flag_types[:-1])

        ## Update metadata: From flag_remove, label
        # Filtered: Filter type, Core, Flank, Composite, num_subunit
        mask_filtered = ~mc_crosscluster_metadata_edited[FilterArgs.filter_flag_col]
        mc_crosscluster_metadata_edited.loc[mask_filtered, FilterArgs.filter_flag_type_col] = None
        mc_crosscluster_metadata_edited.loc[mask_filtered, LabelCols.label_core_col] = mc_crosscluster_metadata_edited.loc[mask_filtered, LabelCols.label_col].str.split("::").str[0]

        # Composite
        mc_crosscluster_metadata_edited[MetadataCols.num_subunit_col] = mc_crosscluster_metadata_edited[LabelCols.label_col].str.split("::").str.len()
        mc_crosscluster_metadata_edited[MetadataCols.composite_col] = (mc_crosscluster_metadata_edited[MetadataCols.num_subunit_col] > 1)
        masK_composite = mc_crosscluster_metadata_edited[MetadataCols.composite_col] & mask_filtered
        mc_crosscluster_metadata_edited.loc[masK_composite, LabelCols.label_flank_col] = mc_crosscluster_metadata_edited.loc[masK_composite, LabelCols.label_col].str.split("::").str[1]

        # Removed: Filter type, Core (Direct), Core, Flank, Label confidence
        mask_removed = mc_crosscluster_metadata_edited[FilterArgs.filter_flag_col]
        mc_crosscluster_metadata_edited.loc[mask_removed, FilterArgs.filter_flag_type_col] = mc_crosscluster_metadata_edited.loc[mask_removed, LabelCols.label_col]
        mc_crosscluster_metadata_edited.loc[mask_removed, LabelCols.direct_label_core_col] = None
        mc_crosscluster_metadata_edited.loc[mask_removed, LabelCols.label_core_col] = None
        mc_crosscluster_metadata_edited.loc[mask_removed, LabelCols.label_flank_col] = None
        mc_crosscluster_metadata_edited.loc[mask_removed, LabelCols.label_conf_col] = LabelCols.label_confs[-1]

        # Keep a legacy display column synchronized when it exists; ``label``
        # remains the canonical field for all QC updates.
        if LabelCols.label_readable_col in mc_crosscluster_metadata_edited.columns:
            mc_crosscluster_metadata_edited[LabelCols.label_readable_col] = mc_crosscluster_metadata_edited[LabelCols.label_col]

        # Update MotifCompendium metadata
        mc_crosscluster_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_crosscluster_metadata_edited.set_index(MetadataCols.unique_id_col, inplace=True)

        mc_crosscluster_filtered.metadata.update(mc_crosscluster_metadata_edited)

        mc_crosscluster_filtered.metadata.reset_index(inplace=True)
        mc_crosscluster_metadata_edited.reset_index(inplace=True)


    # #### Manual: Rules
    ## Manage: Prename, simple name, label readable
    # Swap "high-GC/AT", "ETV-NFAT"
    mc_crosscluster_filtered[LabelCols.label_readable_col] = mc_crosscluster_filtered[LabelCols.label_readable_col].str.replace("GC/AT-bias", "high-GC/AT", regex=True)
    mc_crosscluster_filtered[LabelCols.label_readable_col] = mc_crosscluster_filtered[LabelCols.label_readable_col].str.replace("ETV-NFAT", "ETV-ETS", regex=True)

    ## Label: TF names only
    # Remove "{TF}-like" from readable label
    mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace("-like", "", regex=True)
    # Substitute: AP1 to JUN-FOS
    mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace("AP1", "JUN-FOS", regex=True)
    # Remove: HD/
    mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace("HD/", "", regex=True)
    # Remove: Non-canonical
    for filter_flag_type in FilterArgs.filter_flag_types:
        mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace(filter_flag_type, "", regex=True)

    # Remove: Delimiters
    mc_crosscluster_filtered[LabelCols.label_col] = mc_crosscluster_filtered[LabelCols.label_col].str.replace(r'(::|/|-|_)$', '', regex=True)

    # CpG
    mask_cpg = mc_crosscluster_filtered[LabelCols.label_readable_col].str.contains("/CpG", na=False)
    mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.label_readable_col] = np.where(
        mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col].notna()
        & (mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] != ""),
        mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] + "/CpG",
        "CpG"
    )

    mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.label_col] = np.where(
        mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col].notna()
        & (mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col] != ""),
        mc_crosscluster_filtered.metadata.loc[mask_cpg, LabelCols.direct_label_col],
        "CpG"
    )

    # Filter:
    mask_filter = (
        mc_crosscluster_filtered[FilterArgs.filter_flag_col] | 
        # mc_crosscluster_filtered[direct_type_col].isin(filter_flag_types[:-1]) |
        mc_crosscluster_filtered[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types[:-1])
    )
    mc_crosscluster_filtered.metadata.loc[mask_filter, FilterArgs.filter_flag_col] = True
    mc_crosscluster_filtered.metadata.loc[mask_filter, FilterArgs.filter_flag_type_col] = mc_crosscluster_filtered.metadata.loc[mask_filter, LabelCols.label_readable_col]

    # Remove flagged clusters
    mc_crosscluster_filtered_removed = mc_crosscluster_filtered[mc_crosscluster_filtered[FilterArgs.filter_flag_col]]
    mc_crosscluster_filtered = mc_crosscluster_filtered[~mc_crosscluster_filtered[FilterArgs.filter_flag_col]]

    mc_crosscluster_filtered_removed.metadata = mc_crosscluster_filtered_removed.metadata[mc_crosscluster_removed.columns()]
    for img in mc_crosscluster_filtered_removed.images():
        if img not in mc_crosscluster_removed.images():
            mc_crosscluster_filtered_removed.delete_images(img)

    # Removed: Combine
    mc_crosscluster_removed = MotifCompendium.combine(
        [mc_crosscluster_removed, mc_crosscluster_filtered_removed],
        safe=args.safe,
    )


    # #### Annotate: Best label
    ### Annotate: Best label
    mc_crosscluster_filtered_metadata_updated = mc_crosscluster_filtered.metadata.set_index(MetadataCols.unique_id_col)
    mc_crosscluster_filtered_images_updated = pd.DataFrame(index=mc_crosscluster_filtered_metadata_updated.index)
    mc_crosscluster_filtered_images_existing = set(mc_crosscluster_filtered.images())

    # Update Core, Core (Direct), Flank
    mc_crosscluster_filtered[LabelCols.label_core_col] = np.where(
        mc_crosscluster_filtered[LabelCols.label_col].str.contains("::", na=False),
        mc_crosscluster_filtered[LabelCols.label_col].str.split("::").str[0],
        mc_crosscluster_filtered[LabelCols.label_col]
    )
    mc_crosscluster_filtered[LabelCols.direct_label_core_col] = np.where(
        mc_crosscluster_filtered[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False),
        np.where(
            mc_crosscluster_filtered[LabelCols.label_col].str.contains("::", na=False),
            mc_crosscluster_filtered[LabelCols.label_col].str.split("::").str[0],
            mc_crosscluster_filtered[LabelCols.label_col]
        ),
        None
    )
    mc_crosscluster_filtered[LabelCols.label_flank_col] = np.where(
        mc_crosscluster_filtered[LabelCols.label_col].str.contains("::", na=False),
        mc_crosscluster_filtered[LabelCols.label_col].str.split("::").str[1],
        None
    )

    for i, row in mc_crosscluster_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_crosscluster_filtered_i = mc_crosscluster_filtered[[i]]
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
                target_mc=mc_crosscluster_filtered_i,
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
        mc_crosscluster_filtered_metadata_updated = mc_crosscluster_filtered_metadata_updated.combine_first(mc_crosscluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_crosscluster_filtered_metadata_updated.update(mc_crosscluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
        new_image_cols = [image_col for image_col in mc_crosscluster_filtered_i.images() if image_col.startswith(LabelCols.best_label_ref_col)]
        mc_crosscluster_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_crosscluster_filtered_i.get_images(image_col),
                index=mc_crosscluster_filtered_i.metadata.set_index(MetadataCols.unique_id_col).index,
            )
            for image_col in new_image_cols
        })
        mc_crosscluster_filtered_images_updated = mc_crosscluster_filtered_images_updated.combine_first(mc_crosscluster_filtered_i_new_images)
        mc_crosscluster_filtered_images_updated.update(mc_crosscluster_filtered_i_new_images)

    # Apply updates: Metadata, Images
    mc_crosscluster_filtered.metadata = mc_crosscluster_filtered_metadata_updated.reset_index()
    for image_col in mc_crosscluster_filtered_images_updated.columns:
        mc_crosscluster_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_crosscluster_filtered_images_updated[image_col]],
        )
    del mc_crosscluster_filtered_metadata_updated, mc_crosscluster_filtered_images_updated, mc_crosscluster_filtered_images_existing


    # #### Label confidence
    # Final rules
    mc_crosscluster_filtered[MetadataCols.composite_col] = mc_crosscluster_filtered[LabelCols.label_readable_col].str.contains("::")

    ## Set label readable type
    # Filtered
    label_conf_conditions = [
        mc_crosscluster_filtered[DirectTypeArgs.direct_type_col].str.contains(DirectTypeArgs.direct_types[0], na=False),  # High: A
        mc_crosscluster_filtered[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[1:3]), na=False),  # Medium: B-NZ, B-Z
        ~mc_crosscluster_filtered[FilterArgs.filter_flag_col], # Low: Not flagged
        mc_crosscluster_filtered[FilterArgs.filter_flag_col],   # Flagged: Flagged
    ]
    mc_crosscluster_filtered[LabelCols.label_conf_col] = np.select(
        label_conf_conditions,  # High, Medium, low, Flagged
        LabelCols.label_confs, # High, Medium, low, Flagged
        default=LabelCols.label_confs[-1]  # Remaining: Flagged
    )

    # Removed
    mc_crosscluster_removed[LabelCols.label_conf_col] = LabelCols.label_confs[-1]


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
            assign_to_mc=mc_crosscluster_filtered,
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
        mask_qc = mc_crosscluster_filtered[f'{MetadataCols.qc_col}_score0'] > FilterArgs.qc_min_score_filtered
        mc_crosscluster_filtered.metadata.loc[mask_qc, MetadataCols.checked_col] = True
        mc_crosscluster_filtered.metadata.loc[mask_qc, LabelCols.label_conf_col] = LabelCols.label_confs[2]



    # #### Naming
    ## Simple name: ({+_-})_{prename}_{H_M_L}
    sign = np.where(mc_crosscluster_filtered[MetadataCols.posneg_col] == "pos", "pos", "neg")
    conf = np.select([
        mc_crosscluster_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[0],
        mc_crosscluster_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[1],
        mc_crosscluster_filtered[LabelCols.label_conf_col] == LabelCols.label_confs[2],
        ],
        LabelCols.label_confs_simple[:-1],
        default=LabelCols.label_confs_simple[2],
    )

    mc_crosscluster_filtered[MetadataCols.prename_col] = np.where(
        ~mc_crosscluster_filtered[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types),
        (mc_crosscluster_filtered[LabelCols.label_readable_col]
            + "_" + sign
            + "_" + conf),
        mc_crosscluster_filtered[LabelCols.label_readable_col],
    )

    mc_crosscluster_removed[LabelCols.label_readable_col] = mc_crosscluster_removed[FilterArgs.filter_flag_type_col]
    mc_crosscluster_removed[MetadataCols.prename_col] = mc_crosscluster_removed[FilterArgs.filter_flag_type_col]

    #### Update original MotifCompendium
    logging.info("Propagating crosscluster labels to all levels...")
    ### (5) crosscluster:
    # Combine: Update columns
    mc_crossclusters = [mc_crosscluster_filtered, mc_crosscluster_removed]

    mc_crosscluster.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_crosscluster_images_updated = pd.DataFrame(index=mc_crosscluster.metadata.index)
    mc_crosscluster_images_existing = set(mc_crosscluster.images())

    # Build updates: Metadata, Images
    for mc_crosscluster_i in mc_crossclusters:
        # Metadata
        mc_crosscluster.metadata = mc_crosscluster.metadata.combine_first(mc_crosscluster_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_crosscluster.metadata.update(mc_crosscluster_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
        new_image_cols = [image_col for image_col in mc_crosscluster_i.images() if image_col not in mc_crosscluster_images_existing]
        mc_crosscluster_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_crosscluster_i.get_images(image_col),
                index=mc_crosscluster_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_crosscluster_images_updated = mc_crosscluster_images_updated.combine_first(mc_crosscluster_i_new_images)
        mc_crosscluster_images_updated.update(mc_crosscluster_i_new_images)

    # Apply updates: Metadata, Images
    mc_crosscluster.metadata.reset_index(inplace=True)
    for image_col in mc_crosscluster_images_updated.columns:
        mc_crosscluster.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_crosscluster_images_updated[image_col]],
        )
    del mc_crosscluster_images_updated, mc_crosscluster_images_existing

    # # Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_crosscluster_i in mc_crossclusters:
    #     metadata_cols = list(set(metadata_cols + mc_crosscluster_i.columns()))
    #     image_cols = list(set(image_cols + mc_crosscluster_i.images()))
    # for mc_crosscluster_i in mc_crossclusters:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_crosscluster_i.columns():
    #             mc_crosscluster_i.metadata[metadata_col] = None
    #     for logo_i in ['logo (fwd)', 'logo (rev)']:
    #         mc_crosscluster_i.add_logos(
    #             motifs=mc_crosscluster_i.motifs,
    #             image_name=logo_i,
    #             trim=0,
    #         )
    #     for image_col in image_cols:
    #         if image_col not in mc_crosscluster_i.images():
    #             mc_crosscluster_i.add_logos(
    #                 motifs=np.zeros_like(mc_crosscluster_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Update original MotifCompendium: Contribcluster
    # mc_crosscluster = MotifCompendium.combine(mc_crossclusters, safe=safe)


    # #### Assign name
    ## Rename: {prename_col}_{counter}
    mc_crosscluster.metadata[MetadataCols.counter_col] = mc_crosscluster.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_crosscluster.metadata[MetadataCols.name_new_col] = mc_crosscluster.metadata[MetadataCols.prename_col] + "_" + mc_crosscluster.metadata[MetadataCols.counter_col].astype(str)
    mc_crosscluster.metadata[MetadataCols.name_col] = mc_crosscluster.metadata[MetadataCols.name_new_col]
    mc_crosscluster.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    # Simple name = Prename
    mc_crosscluster[MetadataCols.simple_name_col] = mc_crosscluster[MetadataCols.prename_col]

    # Map: Modelcluster to crosscluster
    modelcluster2crosscluster_name_map = dict(zip(mc_crosscluster[ClusterArgs.source_cluster_col], mc_crosscluster[MetadataCols.name_col]))
    modelcluster2crosscluster_simple_map = dict(zip(mc_crosscluster[ClusterArgs.source_cluster_col], mc_crosscluster[MetadataCols.simple_name_col]))

    mc_modelcluster_combined[VisualizeArgs.crosscluster_name_col] = mc_modelcluster_combined[crosscluster_cluster_col].map(modelcluster2crosscluster_name_map)
    mc_modelcluster_combined[MetadataCols.simple_name_col] = mc_modelcluster_combined[crosscluster_cluster_col].map(modelcluster2crosscluster_simple_map).fillna(mc_modelcluster_combined[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)


    # #### Composites: Modified Smith-Waterman
    # The last step of the filtered crossclusters, before they are exported to every level below: decompose each into
    # direct reference monomers. Below medium confidence, the decomposition decides whether it is a composite, with
    # its core and flank labels (composite/identify_composite_motifs.ipynb). Written back into mc_crosscluster as is.
    logging.info("Composites: decomposing the filtered crossclusters into reference monomers (modified Smith-Waterman)...")
    mc_ref_composite, mc_ref_qc_cluster_name_dict, _ = prepare_composite_reference(mc_ref_direct, args.ref_clustered_hr_metadata)
    filtered_rows = np.flatnonzero(~mc_crosscluster[FilterArgs.filter_flag_col].to_numpy(dtype=bool))
    mc_crosscluster_filtered = mc_crosscluster[filtered_rows.tolist()]
    assign_msw_composites(mc_crosscluster_filtered, mc_ref_composite, ic_scale=args.ic, verbose=args.verbose)
    write_back_rows(
        mc_crosscluster, mc_crosscluster_filtered, filtered_rows,
        columns=[
            CompositeArgs.bellman_score_col, CompositeArgs.bellman_sim_col, CompositeArgs.bellman_orient_col,
            CompositeArgs.bellman_align_col, CompositeArgs.bellman_name_col, CompositeArgs.bellman_label_col,
            MetadataCols.composite_col, MetadataCols.num_subunit_col,
            LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
        ],
        image_names=[CompositeArgs.bellman_logo_col],
    )

    # #### Composites: Identify
    # The composites of the filtered crossclusters, with their reference subunits: the inputs of the ISM
    # cooperativity runs (pipeline_encode_composite_ism.py), in --composite-dir. A core or flank label without a short
    # enough monomer in the composite reference takes its long ones (mc_ref_composite_all)
    logging.info("Composites: identifying the composite crossclusters and their reference subunits...")
    composite_paths_ns = composite_paths(getattr(args, "composite_dir", None) or default_composite_dir(args.output_dir))
    mc_ref_composite_all, _, _ = prepare_composite_reference(mc_ref_direct, args.ref_clustered_hr_metadata, length_filter=False)
    mc_encode_composite = identify_composites(mc_crosscluster_filtered, mc_ref_composite, mc_ref_qc_cluster_name_dict, mc_ref_composite_all)
    del mc_ref_composite_all
    export_composites(mc_crosscluster_filtered, mc_encode_composite, composite_paths_ns, getattr(args, "all_paths_tsv", None), html_column_desc_dict)
    del mc_encode_composite


    # #### Export: the ENCODE compendia; every level below and the motif indexes (--update-constituents now)
    export_crosscluster_level(args, mc_crosscluster, mc_modelcluster_combined, html_column_desc_dict)
    if args.update_constituents == "now":
        export_crosscluster_constituents(
            args,
            mc_crosscluster,
            mc_modelcluster_combined,
            modelcluster2crosscluster_name_map,
            modelcluster2crosscluster_simple_map,
            crosscluster_cluster_col,
            html_column_desc_dict,
        )
    else:
        logging.info("The levels below the crossclusters: not updated (--update-constituents defer; later: --update-constituents only)")

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
        "crosscluster_cluster_col": crosscluster_cluster_col,
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
        "contribcluster_name_col": VisualizeArgs.contribcluster_name_col,
        "modelcluster_name_col": VisualizeArgs.modelcluster_name_col,
        "crosscluster_name_col": VisualizeArgs.crosscluster_name_col,
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

        # Composites
        "composite_dir": composite_paths_ns.composite_dir,

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
    if args.hg38_fasta or args.gencode_gtf:
        logging.info("--hg38-fasta, --gencode-gtf: not used by this stage; the hits are annotated by pipeline_encode_hits.py.")

    # Inputs: the atlas pipelines' stage directories
    stage_dir_args = atlas_stage_dir_args(args.tfatlas_dir, args.chromatlas_dir, args.procapatlas_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.time:
        start_time = time.time()
    if args.update_constituents == "only":
        run_crosscluster_constituents(stage_args(args, **stage_dir_args))
    else:
        run_encode_combine(stage_args(args, **stage_dir_args))
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    logging.info(f'ENCODE pipeline: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
