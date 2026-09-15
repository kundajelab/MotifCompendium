#!/usr/bin/env python
# coding: utf-8
"""
ENCODE GRAMMAR: end-to-end MotifCompendium pipeline.

Runs the full atlas hierarchy from a single CLI, in order:

  1. TF atlas       - counts head    : motif -> cluster -> meta-cluster
  2. TF atlas       - profile head   : motif -> cluster -> meta-cluster
  3. TF atlas       - combine        : counts + profile meta-clusters -> atlascluster
  4. Chromatin atlas- counts head    : motif -> cluster -> meta-cluster
  5. Chromatin atlas- profile head   : motif -> cluster -> meta-cluster
  6. Chromatin atlas- combine        : counts + profile meta-clusters -> atlascluster
  7. ENCODE         - combine        : TF + Chromatin atlasclusters -> ENCODE cluster

Every stage directory is derived from a single --output-dir root, and each stage's
outputs are wired directly into the next stage's inputs:

  <root>/tfatlas/counts       <root>/tfatlas/profile      <root>/tfatlas/combined
  <root>/chromatlas/counts    <root>/chromatlas/profile   <root>/chromatlas/combined
  <root>/encode

Per-atlas configuration lives in configs.PROFILES; set_default_configs(atlas)
binds the correct set before each stage runs.
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

import numpy as np
import pandas as pd
import scipy
import pysam
import h5py

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.composite as utils_composite
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config

import pipeline.encode.configs as configs

#### STAGES ------------------------------------------------------------------
# Stage key -> (label, atlas config profile)
STAGES = {
    "tf-counts":      ("TF atlas: counts head",              "TF"),
    "tf-profile":     ("TF atlas: profile head",             "TF"),
    "tf-combine":     ("TF atlas: combine counts + profile", "TF"),
    "chrom-counts":   ("Chromatin atlas: counts head",       "Chrom"),
    "chrom-profile":  ("Chromatin atlas: profile head",      "Chrom"),
    "chrom-combine":  ("Chromatin atlas: combine counts + profile", "Chrom"),
    "encode-combine": ("ENCODE: combine TF + Chromatin atlasclusters", "ENCODE"),
}
STAGE_ORDER = list(STAGES)


#### ARGUMENTS ---------------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="Run a full, end-to-end ENCODE MotifCompendium pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline control
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER,
                        help="Subset of stages to run, in pipeline order.")

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Root output directory; every stage directory is derived from it.")
    parser.add_argument("--mc-name", type=str, default="mc/motifcompendium.mc",
                        help="Filename (relative to each per-head stage dir) for its initial MotifCompendium object.")

    # Unique identifier versions (kept per atlas, to match the standalone scripts)
    parser.add_argument("--mcid-version-tf", type=str, default="V1",
                        help="MCID version for the TF atlas stages.")
    parser.add_argument("--mcid-version-chrom", type=str, default="V0",
                        help="MCID version for the Chromatin atlas stages.")
    parser.add_argument("--mcid-version-encode", type=str, default="V1",
                        help="MCID version for the ENCODE combine stage.")

    # Input: TF atlas, per head
    parser.add_argument("--tf-counts-input-mc", type=str, default=None,
                        help="TF counts head: input MotifCompendium object (.mc), instead of building from H5s.")
    parser.add_argument("--tf-counts-modisco-h5s", nargs="+", type=str, default=None,
                        help="TF counts head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--tf-counts-input-names", nargs="+", type=str, default=None,
                        help="TF counts head: nickname(s) for the input file(s).")
    parser.add_argument("--tf-counts-metadata", type=str, default=None,
                        help="TF counts head: metadata CSV/TSV, per input file or motif.")

    parser.add_argument("--tf-profile-input-mc", type=str, default=None,
                        help="TF profile head: input MotifCompendium object (.mc), instead of building from H5s.")
    parser.add_argument("--tf-profile-modisco-h5s", nargs="+", type=str, default=None,
                        help="TF profile head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--tf-profile-input-names", nargs="+", type=str, default=None,
                        help="TF profile head: nickname(s) for the input file(s).")
    parser.add_argument("--tf-profile-metadata", type=str, default=None,
                        help="TF profile head: metadata CSV/TSV, per input file or motif.")

    # Input: Chromatin atlas, per head
    parser.add_argument("--chrom-counts-input-mc", type=str, default=None,
                        help="Chromatin counts head: input MotifCompendium object (.mc).")
    parser.add_argument("--chrom-counts-modisco-h5s", nargs="+", type=str, default=None,
                        help="Chromatin counts head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--chrom-counts-input-names", nargs="+", type=str, default=None,
                        help="Chromatin counts head: nickname(s) for the input file(s).")
    parser.add_argument("--chrom-counts-metadata", type=str, default=None,
                        help="Chromatin counts head: metadata CSV/TSV.")

    parser.add_argument("--chrom-profile-input-mc", type=str, default=None,
                        help="Chromatin profile head: input MotifCompendium object (.mc).")
    parser.add_argument("--chrom-profile-modisco-h5s", nargs="+", type=str, default=None,
                        help="Chromatin profile head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--chrom-profile-input-names", nargs="+", type=str, default=None,
                        help="Chromatin profile head: nickname(s) for the input file(s).")
    parser.add_argument("--chrom-profile-metadata", type=str, default=None,
                        help="Chromatin profile head: metadata CSV/TSV.")

    # Input: shared build options
    parser.add_argument("-sh", "--input-subpatterns", action="store_true",
                        help="Use subpatterns (instead of main patterns) when loading TF-MoDISco H5 file(s).")
    parser.add_argument("--modisco-region-width", type=int, default=400,
                        help="[SUPERSEDED - no longer read] Width of region used during the TF-MoDISco run.")

    # Input: Reference databases (shared by every stage)
    parser.add_argument("--mc-ref-qc", type=str, required=True,
                        help="ENCODE QC reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-direct", type=str, required=True,
                        help="ENCODE QC + in vitro reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-clustered-hr", type=str, required=True,
                        help="Human-readable clustered reference MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-clustered-hr-metadata", type=str, required=True,
                        help="Metadata TSV for --mc-ref-clustered-hr.")
    parser.add_argument("--mc-b1h", type=str, required=True,
                        help="B1H zinc-finger reference MotifCompendium object (.mc), for the TF Smith-Waterman ZNF step.")
    # Reference databases required by the confirmed per-atlas pipelines
    parser.add_argument("--ref-invitro", type=str, required=True,
                        help="Reference in-vitro PFM/MotifCompendium (used by the TF and ENCODE stages).")
    parser.add_argument("--tf-class", type=str, default=None,
                        help="TFClass TSV for TF-family annotation (optional; TF stages).")
    parser.add_argument("--human-tf", type=str, default=None,
                        help="Human TF CSV for DBD annotation (optional; TF stages).")

    parser.add_argument("--column-desc-tsv", type=str, required=True,
                        help="TSV describing HTML summary table column names.")

    # Input: ENCODE combine stage only
    parser.add_argument("--tf-init-metadata", type=str, default=None,
                        help="ENCODE stage: initial TF-MoDISco metadata TSV for the TF atlas.")
    parser.add_argument("--chrom-init-metadata", type=str, default=None,
                        help="ENCODE stage: initial TF-MoDISco metadata TSV for the Chromatin atlas.")
    parser.add_argument("--mc-qc-remove-removed", type=str, default=None,
                        help="ENCODE stage: removed MotifCompendium object (.mc) from the prior QC-remove run.")
    parser.add_argument("--mc-qc-pass-filtered", type=str, default=None,
                        help="ENCODE stage: filtered MotifCompendium object (.mc) from the prior QC-pass run.")
    parser.add_argument("--mc-qc-pass-lowq", type=str, default=None,
                        help="ENCODE stage: low-quality MotifCompendium object (.mc) from the prior QC-pass run.")
    parser.add_argument("--metadata-edited", type=str, default=None,
                        help="ENCODE stage: manually edited filtered-metadata CSV.")
    parser.add_argument("--hg38-fasta", type=str, default=None,
                        help="ENCODE stage: hg38 genome FASTA, for hits annotation.")
    parser.add_argument("--gencode-gtf", type=str, default=None,
                        help="ENCODE stage: GENCODE annotation GTF (.gz), for hits annotation.")

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
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ---------------------------------------------------------
def set_default_configs(atlas):
    """Bind the config classes for `atlas` ('TF' | 'Chrom' | 'ENCODE') to module globals."""
    global MetadataCols, TFAnnotationCols, CellTypeCols, ReferenceMatchCols, LabelCols
    global MotifMatchArgs, DirectTypeArgs, SmithWatermanArgs, FilterArgs, MotifFilterArgs
    global ClusterArgs, VisualizeArgs, HitsCols, EncodeCombineArgs, MotifCompendiumArgs
    global AtlasFilterArgs, TFMoDIScoCols

    profile = configs.PROFILES[atlas]
    for name, cls in profile.items():
        globals()[name] = cls()


#### ARG ALIASES -------------------------------------------------------------
# The confirmed per-atlas pipelines read args.<ref_name>; this CLI exposes some of the same
# inputs under different flag names. stage_args() mirrors each one onto the reference name so
# the spliced stage bodies stay byte-identical to their sources.
ARG_ALIASES = {
    "ref_qc":                     "mc_ref_qc",
    "ref_direct":                 "mc_ref_direct",
    "ref_clustered_hr_metadata":  "mc_ref_clustered_hr_metadata",
    "ref_znf_b1h":                "mc_b1h",
    "hg38":                       "hg38_fasta",
    "gencode":                    "gencode_gtf",
    "mc_qc_removed":              "mc_qc_remove_removed",
    "mc_qc_passed":               "mc_qc_pass_filtered",
    "mc_qc_lowq":                 "mc_qc_pass_lowq",
}

#### PATHS -------------------------------------------------------------------
def stage_dirs(root):
    """Fixed per-stage directory layout, derived from the --output-dir root."""
    return {
        "tf-counts":      os.path.join(root, "tfatlas", "counts"),
        "tf-profile":     os.path.join(root, "tfatlas", "profile"),
        "tf-combine":     os.path.join(root, "tfatlas", "combined"),
        "chrom-counts":   os.path.join(root, "chromatlas", "counts"),
        "chrom-profile":  os.path.join(root, "chromatlas", "profile"),
        "chrom-combine":  os.path.join(root, "chromatlas", "combined"),
        "encode-combine": os.path.join(root, "encode"),
    }


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




#### STAGE BODIES -----------------------------------------------------------

def run_tf_sequential(args):
    """TF atlas, one head: motif -> cluster -> meta-cluster"""



    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

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
    mc_motif_direct_path = os.path.join(mc_dir, "motifcompendium_motif_direct.mc")
    mc_motif_nondirect_path = os.path.join(mc_dir, "motifcompendium_motif_nondirect.mc")

    # Clusters
    mc_cluster_path = os.path.join(mc_dir, "motifcompendium_cluster_all.mc")
    mc_cluster_filtered_path = os.path.join(mc_dir, "motifcompendium_cluster_filtered.mc")
    mc_cluster_removed_path = os.path.join(mc_dir, "motifcompendium_cluster_removed.mc")
    mc_cluster_direct_path = os.path.join(mc_dir, "motifcompendium_cluster_direct.mc")
    mc_cluster_nondirect_path = os.path.join(mc_dir, "motifcompendium_cluster_nondirect.mc")

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


    # # Combine MotifCompendium: MC
    # mcs = [mc, mc_znf]

    # # Add metadata cols
    # for mc_i in mcs:
    #     for col in list(set(mc.columns() + mc_znf.columns())):
    #         if col not in mc_i.metadata.columns:
    #             mc_i.metadata[col] = None

    # # Remove image cols
    # for mc_i in mcs:
    #     for image_col in mc_i.images():
    #         if image_col not in list(set(mc.images()) & set(mc_znf.images())):
    #             mc_i.delete_images(image_col)

    # mc = MotifCompendium.combine(
    #     [mc, mc_znf,],
    #     safe=safe,
    # )

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
    # NOTE: Disabled on purpose (matches run_sequential_cluster-ZNF-composite.py, where this
    # block is commented out). The reference MotifCompendium objects already carry
    # `TFClass_family_name` / `HumanTF_DBD` / `family` columns baked in at build time, and those
    # stored values are what the reference-matching steps below are calibrated against.
    # Re-deriving them here overwrites the stored annotations with the current `_FAMILIES`
    # heuristic, which shrinks reference family membership (e.g. `family == 'HOX'` drops from 135
    # to 125 motifs, because SHOX / SHOX2 / RHOXF1 / RHOXF2 no longer match), changing the
    # Direct BETA (non-ZNF) reference set and hence the cluster / meta-cluster counts.
    # mc_refs = [mc_ref_qc, mc_ref_invitro]
    # for mc_ref_i in mc_refs:
    #     annotate_tfclass(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)
    #     annotate_humantf(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)
    #     annotate_family(mc_ref_i.metadata, tf_col=TFAnnotationCols.ref_tf_col)

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

    # # Trimer-style notation
    # mc_qc[ref_readable_name_col] = mc_qc.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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
    aligned_motifs = np.zeros((mc_qc.motifs.shape[0], mc_qc.motifs.shape[1] + 2, mc_qc.motifs.shape[2]))

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

                (msw_score, sim_score, msw_aligned_b1h, msw_orient, msw_align_b1h, msw_align_cwm, msw_scale_f, msw_action, _, _,) = utils_composite.run_modified_smith_waterman_znf(
                    cwm=cwm_ij,
                    b1h_ppm_ic=b1h_ppm_ic_i,
                    sim_threshold=SmithWatermanArgs.msw_sim_threshold,
                    overlap_penalty_f1=SmithWatermanArgs.overlap_penalty_f1, 
                    overlap_penalty_f2=SmithWatermanArgs.overlap_penalty_f2,
                    skip_penalty_f=SmithWatermanArgs.skip_penalty_f, 
                    skip_penalty_l1=SmithWatermanArgs.skip_penalty_l1, 
                    skip_penalty_l2=SmithWatermanArgs.skip_penalty_l2,
                    finger_length=3,
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
                
                # # Switch subunit order, if score1 > score0
                # swap_mask = mc_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score1"] > mc_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score0"]
                # if swap_mask.any():
                #     # Swap score
                #     temp_score = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"].copy()
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"] = temp_score

                #     # Swap name
                #     temp_name = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"].copy()
                #     temp_readable_name = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"].copy()
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"] = temp_name
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"] = temp_readable_name

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
                
                # # Switch subunit order, if score1 > score0
                # swap_mask = mc_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score1"] > mc_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score0"]
                # if swap_mask.any():
                #     # Swap score
                #     temp_score = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"].copy()
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"] = temp_score

                #     # Swap name
                #     temp_name = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"].copy()
                #     temp_readable_name = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"].copy()
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"] = temp_name
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"] = mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"]
                #     mc_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"] = temp_readable_name

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

            # mc_znf_i = mc_i[~(
            #     (mc_i[f'{ReferenceMatchCols.nonznf_ref_col}_score0'] > MotifMatchArgs.min_single_score) |
            #     (mc_i[f'{ReferenceMatchCols.nonznf_ref_col}_score1'] > MotifMatchArgs.min_composite_score) |
            #     (mc_i[f'{ReferenceMatchCols.nonznf_ref_col}_score2'] > MotifMatchArgs.min_composite_score)
            # )]
            
            ## Match: Outside of family
            # utils_analysis.assign_label_from_other_compendium(
            #     assign_to_mc=mc_i,
            #     assign_from_mc=mc_ref_beta_zf_i,
            #     from_label_col=MetadataCols.name_col,
            #     save_col_prefix=ref_oofamily_col,
            #     min_score=MotifMatchArgs.min_composite_score,
            #     max_submotifs=MotifMatchArgs.max_submotifs,
            #     label_unsigned=True,
            #     save_images=False,
            #     logo_trimming=0,
            # )
            
            # # Remove matched motifs
            # mc_znf_i = mc_i[~(
            #     (mc_i[f'{ref_oofamily_col}_score0'] > MotifMatchArgs.min_single_score) |
            #     (mc_i[f'{ref_oofamily_col}_score1'] > MotifMatchArgs.min_composite_score) |
            #     (mc_i[f'{ref_oofamily_col}_score2'] > MotifMatchArgs.min_composite_score)
            # )]
            
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

    # mc_direct_alpha = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[0]]
    # mc_direct_beta_nzf = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[1]]
    # mc_direct_beta_zf = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[2]]
    # mc_direct_gamma = mc_filtered[mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_types[3]]
    # mc_nondirect = mc_filtered[~mc_filtered[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])]

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

    # # Trimer-style notation
    # mc_cluster[ref_readable_name_col] = mc_cluster.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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
                
                # # Switch subunit order, if score1 > score0
                # swap_mask = mc_cluster_direct_alpha_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score1"] > mc_cluster_direct_alpha_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score0"]
                # if swap_mask.any():
                #     # Swap score
                #     temp_score = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"].copy()
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"] = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"]
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"] = temp_score

                #     # Swap name
                #     temp_name = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"].copy()
                #     temp_readable_name = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"].copy()
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"] = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"]
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"] = temp_name
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"] = mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"]
                #     mc_cluster_direct_alpha_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"] = temp_readable_name

                # # Combine: Monomer + Multimer
                # mc_cluster_direct_alpha_i = MotifCompendium.combine(
                #     [mc_cluster_direct_alpha_i_monomer, mc_cluster_direct_alpha_i_multimer],
                #     safe=safe,
                # )

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
            num_subunit = mask_subunit_df.sum(axis=1)
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
                
                # # Switch subunit order, if score1 > score0
                # swap_mask = mc_cluster_direct_beta_nzf_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score1"] > mc_cluster_direct_beta_nzf_i_monomer[f"{ReferenceMatchCols.direct_ref_col}_score0"]
                # if swap_mask.any():
                #     # Swap score
                #     temp_score = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"].copy()
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score0"] = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"]
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_score1"] = temp_score

                #     # Swap name
                #     temp_name = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"].copy()
                #     temp_readable_name = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"].copy()
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name0"] = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"]
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_col}_name1"] = temp_name
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name0"] = mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"]
                #     mc_cluster_direct_beta_nzf_i_monomer.metadata.loc[swap_mask, f"{ReferenceMatchCols.direct_ref_readable_col}_name1"] = temp_readable_name
                
                # # Combine: Monomer + Multimer
                # mc_cluster_direct_beta_nzf_i = MotifCompendium.combine(
                #     [mc_cluster_direct_beta_nzf_i_monomer, mc_cluster_direct_beta_nzf_i_multimer],
                #     safe=safe,
                # )

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
            num_subunit = mask_subunit_df.sum(axis=1)
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

    #     else:
    #         logging.info(f"  No Beta ref found")
    #         for iter in range(MotifMatchArgs.max_submotifs):
    #             mc_cluster_direct_beta_nzf_i.metadata[[
    #                 f"{ReferenceMatchCols.direct_ref_col}_score{iter}", 
    #                 f"{ReferenceMatchCols.direct_ref_col}_name{iter}",
    #                 f"{ReferenceMatchCols.direct_ref_col}_target{iter}",
    #             ]] = None
    #             mc_cluster_direct_beta_nzf_i.add_logos(
    #                 motifs=np.zeros_like(mc_cluster_direct_beta_nzf_i.get_standard_motif_stack()),
    #                 image_name=f"{ReferenceMatchCols.direct_ref_col}_logo{iter}",
    #                 trim=False,
    #             )

    #     # Store in list
    #     mc_cluster_direct_beta_nzf_is.append(mc_cluster_direct_beta_nzf_i)

    # # Recombine
    # mc_cluster_direct_beta_nzf = MotifCompendium.combine(mc_cluster_direct_beta_nzf_is, safe=safe)
    # del mc_cluster_direct_beta_nzf_is

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
        mc_cluster_direct_beta_zf.motifs.shape[1] + 2, 
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
                
                (msw_score, sim_score, msw_aligned_b1h, msw_orient, msw_align_b1h, msw_align_cwm, msw_scale_f, msw_action, _, _) = utils_composite.run_modified_smith_waterman_znf(
                    cwm=cwm_ij,
                    b1h_ppm_ic=b1h_ppm_ic_i,
                    sim_threshold=SmithWatermanArgs.msw_sim_threshold,
                    overlap_penalty_f1=SmithWatermanArgs.overlap_penalty_f1, 
                    overlap_penalty_f2=SmithWatermanArgs.overlap_penalty_f2,
                    skip_penalty_f=SmithWatermanArgs.skip_penalty_f, 
                    skip_penalty_l1=SmithWatermanArgs.skip_penalty_l1, 
                    skip_penalty_l2=SmithWatermanArgs.skip_penalty_l2,
                    finger_length=3,
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

    # # Combine: Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_cluster_filtered_i in mc_cluster_filtereds:
    #     metadata_cols = list(set(metadata_cols + mc_cluster_filtered_i.columns()))
    #     image_cols = list(set(image_cols + mc_cluster_filtered_i.images()))

    # for mc_cluster_filtered_i in mc_cluster_filtereds:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_cluster_filtered_i.columns():
    #             mc_cluster_filtered_i.metadata[metadata_col] = None
    #     for image_col in image_cols:
    #         if image_col not in mc_cluster_filtered_i.images():
    #             mc_cluster_filtered_i.add_logos(
    #                 motifs=np.zeros_like(mc_cluster_filtered_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Clusters: Combine
    # mc_cluster_filtered = MotifCompendium.combine(mc_clusters, safe=safe)

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

    # # Trimer-style notation
    # mc_metacluster[ref_readable_name_col] = mc_metacluster.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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


    def assign_direct_label(row):
        # Initialize
        label_core = None  # Default: None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])

        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]
        row_label_flank_constituents = row[LabelCols.label_flank_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if is_valid(direct_type) and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4]):
            label_core, label_core_readable = assign_direct_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)

        return label_core, label_core_readable, label_flank, label_combined


    def assign_label(row):
        # Initialize
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])
        
        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_core_readable = row[LabelCols.label_core_readable_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(direct_type)
            and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4])
        ):
            label_core, label_core_readable = assign_direct_label_core(row)

            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)
        
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
                label_flank = row_label_flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
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
            elif is_valid(label_core):
                label_combined = str(label_core)
        
        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown
        
        return label_core, label_core_readable, label_flank, label_combined


    ### Label: Metacluster - Filtered
    # Add human-readable label (Direct only)
    mc_metacluster_filtered.metadata[[LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.label_flank_col, LabelCols.direct_label_col]] = pd.DataFrame(
        mc_metacluster_filtered.metadata.apply(assign_direct_label, axis=1).tolist(),
        index=mc_metacluster_filtered.metadata.index
    )
    # Add human-readable label
    mc_metacluster_filtered.metadata[[LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col]] = pd.DataFrame(
        mc_metacluster_filtered.metadata.apply(assign_label, axis=1).tolist(),
        index=mc_metacluster_filtered.metadata.index
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
    mc_removed.save(mc_motif_removed_path)
    mc_direct.save(mc_motif_direct_path)
    mc_nondirect.save(mc_motif_nondirect_path)

    # (2) Clusters
    mc_cluster.save(mc_cluster_path)
    mc_cluster_filtered.save(mc_cluster_filtered_path)
    mc_cluster_removed.save(mc_cluster_removed_path)
    mc_cluster_direct.save(mc_cluster_direct_path)
    mc_cluster_nondirect_filtered.save(mc_cluster_nondirect_path)

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
        trim_importance=(1/(2*mc.motifs.shape[1])),
    )
    utils_analysis.export_compendium_meme(
        mc=mc_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_filtered_path,
        trim_importance=(1/(2*mc_filtered.motifs.shape[1])),
    )
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
        "skip_penalty_f": SmithWatermanArgs.skip_penalty_f,
        "skip_penalty_l1": SmithWatermanArgs.skip_penalty_l1,
        "skip_penalty_l2": SmithWatermanArgs.skip_penalty_l2,
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


def run_tf_combine(args):
    """TF atlas: combine the counts and profile meta-clusters into atlasclusters"""


    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Run started.")

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

    ## Written back into the per-head pipeline dirs, alongside the inputs
    mc_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    mc_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

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


    def assign_direct_label(row):
        # Initialize
        label_core = None  # Default: None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])

        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]
        row_label_flank_constituents = row[LabelCols.label_flank_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if is_valid(direct_type) and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4]):
            label_core, label_core_readable = assign_direct_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)

        return label_core, label_core_readable, label_flank, label_combined


    def assign_label(row):
        # Initialize
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])
    
        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_core_readable = row[LabelCols.label_core_readable_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(direct_type)
            and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4])
        ):
            label_core, label_core_readable = assign_direct_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)
    
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
                label_flank = row_label_flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
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
            elif is_valid(label_core):
                label_combined = str(label_core)
    
        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown
    
        return label_core, label_core_readable, label_flank, label_combined

    # Label: Metacluster - Filtered
    mc_tfatlas_filtered.metadata[[LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.label_flank_col, LabelCols.direct_label_col]] = pd.DataFrame(
        mc_tfatlas_filtered.metadata.apply(assign_direct_label, axis=1).tolist(),
        index=mc_tfatlas_filtered.metadata.index
    )
    # Add human-readable label
    mc_tfatlas_filtered.metadata[[LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col]] = pd.DataFrame(
        mc_tfatlas_filtered.metadata.apply(assign_label, axis=1).tolist(),
        index=mc_tfatlas_filtered.metadata.index
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

    os.makedirs(os.path.dirname(html_atlascluster_removed_path), exist_ok=True)
    mc_tfatlas_removed = mc_tfatlas[mc_tfatlas[FilterArgs.filter_flag_col]]
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
    )

    os.makedirs(os.path.dirname(meme_atlascluster_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_tfatlas_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_filtered_path,
    )

    os.makedirs(os.path.dirname(meme_atlascluster_removed_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_tfatlas_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_removed_path,
    )

    ## EXPORT: MoDISco HDF5
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
    utils_analysis.export_compendium_modisco(
        mc=mc_tfatlas_removed,
        name_col=MetadataCols.name_col,
        save_loc=modisco_atlascluster_removed_path,
    )

    # Metadata TSV
    os.makedirs(metadata_dir, exist_ok=True)
    mc_tfatlas.metadata.to_csv(metadata_atlascluster_path, sep="\t", index=False)


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

    # Save MotifCompendium
    mc_metacluster_counts.save(mc_metacluster_counts_indexed_path)
    mc_metacluster_profile.save(mc_metacluster_profile_indexed_path)

    mc_cluster_counts.save(mc_cluster_counts_indexed_path)
    mc_cluster_profile.save(mc_cluster_profile_indexed_path)

    mc_counts.save(mc_counts_indexed_path)
    mc_profile.save(mc_profile_indexed_path)

    # Motif index
    mc_motif_index = pd.concat([
        mc_counts.metadata,
        mc_profile.metadata,
    ], axis=0).reset_index(drop=True)

    # Export: Motif index
    mc_motif_index[VisualizeArgs.motif_index_cols_atlascluster].to_csv(metadata_motif_index_path, sep="\t", index=False)

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
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def run_chrom_sequential(args):
    """Chromatin atlas, one head: motif -> cluster -> meta-cluster"""



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

    # # From existing name
    # mc[MetadataCols.name_new_col] = (
    #     mc[MetadataCols.posneg_col]
    #     + "_" + "patterns."
    #     + mc[MetadataCols.name_col].str.split(".").str[1:].str.join(".")
    # )

    # # From existing name
    # mc[MetadataCols.name_col] = (
    #     mc[MetadataCols.posneg_col] + "_patterns." + ".".join(
    #         "-".join(
    #             mc[MetadataCols.name_col].str.split("-").str[1:]
    #         ).str.split(".").str[1:]
    #     )
    # )

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

    # # Trimer-style notation
    # mc_qc[ref_readable_name_col] = mc_qc.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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

    # # Trimer-style notation
    # mc_cluster[ref_readable_name_col] = mc_cluster.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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

    # # Trimer-style notation
    # mc_metacluster[ref_readable_name_col] = mc_metacluster.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

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
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])
        
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
                label_flank = row_label_flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
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

    # # Update original MotifCompendium: Metacluster
    # mc_metacluster = MotifCompendium.combine(mc_metaclusters, safe=safe)


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

    # # Share columns
    # mc_clusters = [mc_cluster_filtered, mc_cluster_removed]
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


def run_chrom_combine(args):
    """Chromatin atlas: combine the counts and profile meta-clusters into atlasclusters"""


    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Run started.")

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

    ## Written back into the per-head pipeline dirs, alongside the inputs
    mc_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_counts_indexed_path = os.path.join(args.counts_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    mc_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_profile_indexed_path = os.path.join(args.profile_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

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
    logging.info("Loading per-head (counts, profile) MotifCompendium objects...")
    # Load MotifCompendium
    mc_metacluster_filtered_counts = MotifCompendium.load(mc_metacluster_filtered_counts_path, safe=args.safe)
    mc_metacluster_filtered_profile = MotifCompendium.load(mc_metacluster_filtered_profile_path, safe=args.safe)


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


    # ### Combine: Meta-clusters
    logging.info("Combining counts + profile meta-clusters...")

    # #### Pairwise similarity
    # Combine MotifCompendium
    mc_metacluster_combined = MotifCompendium.combine(
        [mc_metacluster_filtered_counts, mc_metacluster_filtered_profile],
        safe=args.safe,
    )


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
    # ## Cluster: Manual
    logging.info("Merging meta-clusters 1:1 by pairwise similarity...")
    # # Apply threshold
    # sim_threshold = np.where(mc_metacluster_combined.similarity < cluster_sim_threshold_force, 0, mc_metacluster_combined.similarity)
    # np.fill_diagonal(sim_threshold, 0) # Set diagonal to 0
    # sim_threshold_sparse = scipy.sparse.coo_matrix(sim_threshold) # Sparse matrix
    # # Sort by similarity score (descending), then by row index, then by column index
    # cluster_candidates = sorted([(sim_threshold_sparse.data[i], sim_threshold_sparse.row[i], sim_threshold_sparse.col[i]) for i in range(len(sim_threshold_sparse.data))], reverse=True)

    # # Create clusters
    # clustered = [0] * len(mc_metacluster_combined)
    # cluster_num = 0
    # for cluster_candidate in cluster_candidates:
    #     sim, i, j = cluster_candidate
    #     if clustered[i] == 0 and clustered[j] == 0:
    #         # Create new cluster
    #         mc_metacluster_combined.metadata.loc[i, atlascluster_cluster_col] = cluster_num
    #         mc_metacluster_combined.metadata.loc[j, atlascluster_cluster_col] = cluster_num
    #         cluster_num += 1
        
    #         # Record clustered
    #         clustered[i] += 1 
    #         clustered[j] += 1

    # # Assign singleton clusters to unclustered motifs
    # mc_metacluster_combined.metadata.loc[mc_metacluster_combined.metadata[atlascluster_cluster_col].isna(), atlascluster_cluster_col] = range(cluster_num, cluster_num + mc_metacluster_combined.metadata[atlascluster_cluster_col].isna().sum())

    # # Average MotifCompendium
    # mc_chromatlas = mc_metacluster_combined.cluster_averages(
    #     clustering=atlascluster_cluster_col,
    #     aggregations=aggregations,
    #     weight_col=average_weight_col,
    #     compute_quality_stats=False,
    # )

    # min_len = mc_chromatlas[atlascluster_cluster_col].nunique()
    # print(f"Unique clusters (Filtered): {min_len}")
    # ## Cluster: Manual (recursively)
    # for i in range(cluster_max_iter_force):
    #     # Apply threshold
    #     sim_threshold = np.where(mc_metacluster_combined.similarity < cluster_sim_threshold_force, 0, mc_metacluster_combined.similarity)
    #     np.fill_diagonal(sim_threshold, 0) # Set diagonal to 0
    #     sim_threshold_sparse = scipy.sparse.coo_matrix(sim_threshold) # Sparse matrix
    
    #     # Sort by similarity score (descending), then by row index, then by column index
    #     cluster_candidates = sorted([(sim_threshold_sparse.data[i], sim_threshold_sparse.row[i], sim_threshold_sparse.col[i]) for i in range(len(sim_threshold_sparse.data))], reverse=True)

    #     # Create clusters
    #     clustered = [0] * len(mc_metacluster_combined)
    #     cluster_num = 0
    #     for cluster_candidate in cluster_candidates:
    #         sim, i, j = cluster_candidate
    #         if clustered[i] == 0 and clustered[j] == 0:
    #             # Create new cluster
    #             mc_metacluster_combined.metadata.loc[i, atlascluster_cluster_col] = cluster_num
    #             mc_metacluster_combined.metadata.loc[j, atlascluster_cluster_col] = cluster_num
    #             cluster_num += 1
            
    #             # Record clustered
    #             clustered[i] += 1 
    #             clustered[j] += 1

    #     # Assign singleton clusters to unclustered motifs
    #     mc_metacluster_combined.metadata.loc[mc_metacluster_combined.metadata[atlascluster_cluster_col].isna(), atlascluster_cluster_col] = range(cluster_num, cluster_num + mc_metacluster_combined.metadata[atlascluster_cluster_col].isna().sum())

    #     ## Average MotifCompendium
    #     mc_chromatlas = mc_metacluster_combined.cluster_averages(
    #         clustering=atlascluster_cluster_col,
    #         aggregations=aggregations,
    #         weight_col=average_weight_col,
    #         compute_quality_stats=False,
    #     )

    #     new_min_len = mc_chromatlas[atlascluster_cluster_col].nunique()
    #     print(f"Unique clusters (Filtered): {new_min_len}")
    #     if min_len == new_min_len:
    #         break
    #     else:
    #         min_len = new_min_len


    # ## Final: Chromatin Atlas

    # #### Average
    logging.info("Creating atlasclusters, by averaging...")
    # Average MotifCompendium
    aggregations_atlascluster = []
    for (agg_col_in, agg, agg_col_out) in ClusterArgs.aggregations:
        if agg_col_in in mc_metacluster_combined.columns():
            aggregations_atlascluster.append((agg_col_in, agg, agg_col_out))

    mc_chromatlas = mc_metacluster_combined.cluster_averages(
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
            mc_cluster=mc_chromatlas,
            cluster_col=atlascluster_cluster_col,
            agg_col=agg_col_old,
            new_col=agg_col_new,
            weight_col=ClusterArgs.average_weight_col,
        )
    # Merge aggregations
    merge_aggregations_atlascluster = []
    for i, aggregate in enumerate(ClusterArgs.merge_aggregations):
        (agg_col_old, action, agg_col_new) = aggregate
        if agg_col_old in mc_chromatlas.columns():
            merge_aggregations_atlascluster.append(aggregate)
        if agg_col_new not in mc_chromatlas.columns():
            mc_chromatlas.metadata[agg_col_new] = None

    for (agg_col_old, _, agg_col_new) in merge_aggregations_atlascluster:
        mc_chromatlas.metadata[agg_col_new] = mc_chromatlas.metadata[[agg_col_old, agg_col_new]].apply(
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
    # Calculations
    mc_chromatlas[MetadataCols.total_contrib_score_ratio_col] = mc_chromatlas[MetadataCols.total_contrib_score_col] / mc_chromatlas[MetadataCols.group_total_contrib_score_col]  # Contribution score
    mc_chromatlas[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_chromatlas.get_standard_motif_stack())  # Positive / negative
    mc_chromatlas[MetadataCols.composite_col] = mc_chromatlas[MetadataCols.composite_col].astype(bool)
    mc_chromatlas.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)  # Sort by pos/neg, num_seqlets

    # mc_chromatlas.sort(by=[composite_col], ascending=True, inplace=True)  # Sort by composite
    mc_chromatlas.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity
    )  # Add strings

    ## Add: Unique ID
    mc_chromatlas[MetadataCols.unique_id_col] = unique_id_initial + f".C3.{MetadataCols.atlas_type}.counts_profile." + mc_chromatlas.metadata.index.map('{:08d}'.format)
    ### Aggregate: Annotations
    mc_chromatlas_metadata_updates = []
    mc_chromatlas_images_updated = pd.DataFrame(index=mc_chromatlas[MetadataCols.unique_id_col])
    mc_chromatlas_images_existing = set(mc_chromatlas.images())

    for i, row in mc_chromatlas.metadata.iterrows():
        # MotifCompendium: Subset
        mc_chromatlas_i = mc_chromatlas[[i]]
        ref_name_constituents_i = row[ReferenceMatchCols.ref_name_constit_all_col].split(",") if pd.notna(row[ReferenceMatchCols.ref_name_constit_all_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

        ## [General] Reference annotations
        if ref_name_constituents_i:
            mc_ref_qc_i = mc_ref_qc[mc_ref_qc[MetadataCols.name_col].isin(ref_name_constituents_i)]

            # Annotate with best reference constituents
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_chromatlas_i,
                assign_from_mc=mc_ref_qc_i,
                save_col_prefix=ReferenceMatchCols.best_ref_constit_col,
                min_score=0,
                max_submotifs=max_submotifs_i,
                label_unsigned=True,
                save_images=True,
                logo_trimming=0,
            )
    
        ## Update: Metadata, Images
        mc_chromatlas_metadata_updates.append(mc_chromatlas_i.metadata.set_index(MetadataCols.unique_id_col))

        new_image_cols = [image_col for image_col in mc_chromatlas_i.images() if image_col not in mc_chromatlas_images_existing]
        mc_chromatlas_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_chromatlas_i.get_images(image_col),
                index=mc_chromatlas_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_chromatlas_images_updated = mc_chromatlas_images_updated.combine_first(mc_chromatlas_i_new_images)
        mc_chromatlas_images_updated.update(mc_chromatlas_i_new_images)
    # Apply updates: Metadata, Images
    mc_chromatlas_metadata_updates_df = pd.concat(mc_chromatlas_metadata_updates)
    mc_chromatlas.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_chromatlas.metadata = mc_chromatlas.metadata.combine_first(mc_chromatlas_metadata_updates_df)
    mc_chromatlas.metadata.update(mc_chromatlas_metadata_updates_df)
    mc_chromatlas.metadata.reset_index(inplace=True)
    del mc_chromatlas_metadata_updates, mc_chromatlas_metadata_updates_df

    for image_col in mc_chromatlas_images_updated.columns:
        mc_chromatlas.set_images(
            image_name=image_col,
            images=mc_chromatlas_images_updated[image_col].where(
                pd.notna(mc_chromatlas_images_updated[image_col]), None).to_list()
        )
    del mc_chromatlas_images_updated, mc_chromatlas_images_existing


    # #### Annotate
    # ### Annotate: Meta-clusters
    logging.info("Annotating atlasclusters against the reference...")
    # Initialize
    mc_chromatlas[LabelCols.label_col] = MetadataCols.name_unknown  # Initial readable name: Unknown
    mc_chromatlas[LabelCols.label_core_col] = None
    mc_chromatlas[LabelCols.label_core_readable_col] = None
    mc_chromatlas[LabelCols.label_flank_col] = None
    # mc_chromatlas[ref_readable_name_col] = None
    ## [General] Label meta-clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_chromatlas,
        assign_from_mc=mc_ref_qc,
        from_label_col=MetadataCols.name_col,
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
        mc_chromatlas[ref_readable_name_col] = mc_chromatlas[f'{ReferenceMatchCols.ref_col}_name{i}'].map(mc_ref_qc_hr_dict)

    # Classification: Mask
    mask_monomer_metacluster = (
        (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_metacluster = (
        (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_metacluster = (
        (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_chromatlas[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_metacluster = (
        mask_dimer_metacluster | mask_trimer_metacluster
    )
    mask_matched_metacluster = (
        mask_monomer_metacluster | mask_dimer_metacluster | mask_trimer_metacluster
    )

    # Dimer-style notation
    mc_chromatlas.metadata.loc[mask_matched_metacluster, LabelCols.label_core_col] = mc_chromatlas.metadata.loc[mask_matched_metacluster, ref_readable_name_cols[0]]
    mc_chromatlas.metadata.loc[mask_matched_metacluster, LabelCols.label_core_readable_col] = mc_chromatlas.metadata.loc[mask_matched_metacluster, ref_readable_name_cols[0]]
    mc_chromatlas.metadata.loc[mask_dimer_metacluster, LabelCols.label_flank_col] = mc_chromatlas.metadata.loc[mask_dimer_metacluster, ref_readable_name_cols[1]]
    mc_chromatlas.metadata.loc[mask_trimer_metacluster, LabelCols.label_flank_col] = mc_chromatlas.metadata.loc[mask_trimer_metacluster, ref_readable_name_cols[1]] + "::" + mc_chromatlas.metadata.loc[mask_trimer_metacluster, ref_readable_name_cols[2]]
    # mc_chromatlas.metadata.loc[mask_dimer_metacluster, num_subunit_col] = 2  # Composite: Determined with clusters
    # mc_chromatlas.metadata.loc[mask_trimer_metacluster, num_subunit_col] = 3
    # mc_chromatlas.metadata[composite_col] = mc_chromatlas.metadata[num_subunit_col] > min_composite_subunit

    # # Trimer-style notation
    # mc_chromatlas[ref_readable_name_col] = mc_chromatlas.metadata[
    #     [
    #         f'{ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

    # Concatenate all unique annotations
    mc_chromatlas[ReferenceMatchCols.ref_name_col] = mc_chromatlas.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )


    # #### Filter
    logging.info("Filtering atlasclusters...")
    # Calculate filters
    mc_chromatlas[FilterArgs.filter_flag_col] = False

    utils_analysis.calculate_filters(
        mc=mc_chromatlas,
        metric_list=AtlasFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )
    # Apply filters
    apply_filter_clusters(
        mc=mc_chromatlas,
        motif_filter_args=AtlasFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )
    # Remove flagged clusters
    mc_chromatlas_removed = mc_chromatlas[mc_chromatlas[FilterArgs.filter_flag_col]]
    mc_chromatlas_filtered = mc_chromatlas[~(mc_chromatlas[FilterArgs.filter_flag_col])]


    # #### Clustering (for sorting only)
    ## Cluster: For sorting only
    logging.info("Clustering atlasclusters (for sorting only)...")
    # Cluster: Leiden CPM
    mc_chromatlas_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
    )

    # Reassign: K-means
    mc_chromatlas_filtered.cluster(
        algorithm=ClusterArgs.cluster_algorithm_kmeans,
        save_name=VisualizeArgs.sort_col,
        largest_clusters_first=True,
        init_clustering_col=VisualizeArgs.sort_col,
        weight_col=ClusterArgs.average_weight_col,
        assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
        n_iterations=ClusterArgs.n_iterations,
    )
    membership_cluster = mc_chromatlas_filtered[VisualizeArgs.sort_col]
    if args.verbose:
        logging.info(f"Unique clusters (for sorting only): {len(set(membership_cluster))}")


    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter_main):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        # Cluster: Leiden CPM
        mc_chromatlas_filtered.cluster(
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
        mc_chromatlas_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )

        # Check convergence
        new_membership_cluster = mc_chromatlas_filtered[VisualizeArgs.sort_col]
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
        mc_chromatlas_filtered.cluster(
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
        mc_chromatlas_filtered.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=VisualizeArgs.sort_col,
            largest_clusters_first=True,
            init_clustering_col=VisualizeArgs.sort_col,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
        )
    
        # Check convergence
        new_membership_cluster = mc_chromatlas_filtered[VisualizeArgs.sort_col]
        if args.verbose:
            logging.info(f"Unique clusters (for sorting only): {len(set(new_membership_cluster))}")
        if np.array_equal(membership_cluster, new_membership_cluster):
            break
        else:
            membership_cluster = new_membership_cluster


    # #### Final name: Chromatin Atlas
    # Count occurrence: Core TF family
    mc_metacluster_combined[LabelCols.label_core_readable_col_set] = mc_metacluster_combined[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_metacluster_explode_readable_core = mc_metacluster_combined.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_metacluster_combined[LabelCols.label_flank_col_set] = mc_metacluster_combined[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_metacluster_explode_readable_composite = mc_metacluster_combined.metadata.explode(LabelCols.label_flank_col_set)
    ## Count occurrences: Create pivot table
    # Core: All
    mc_metacluster_core_readable_pivot = pd.pivot_table(
        mc_metacluster_explode_readable_core,
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
        mc_metacluster_explode_readable_composite,
        index=atlascluster_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_metacluster_flank_pivot = mc_metacluster_flank_pivot.div(
        mc_metacluster_flank_pivot.sum(axis=1), axis=0)

    # Match order of metaclusters
    mc_metacluster_core_readable_pivot = mc_metacluster_core_readable_pivot.reindex(
        mc_chromatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
    mc_metacluster_flank_pivot = mc_metacluster_flank_pivot.reindex(
        mc_chromatlas_filtered[ClusterArgs.source_cluster_col]).fillna(0).drop([''], axis=1, errors='ignore')
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

    def assign_label(row):
        # Initialize
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])
    
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
                label_flank = row_label_flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
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
            elif is_valid(label_core):
                label_combined = str(label_core)
    
        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown
    
        return label_core, label_core_readable, label_flank, label_combined
    # Label: Chromatlas - Filtered
    mc_chromatlas_filtered.metadata[[LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col]] = pd.DataFrame(
        mc_chromatlas_filtered.metadata.apply(assign_label, axis=1).tolist(),
        index=mc_chromatlas_filtered.metadata.index
    )
    # Label: Chromatlas - Removed
    mc_chromatlas_removed[LabelCols.label_col] = mc_chromatlas_removed[FilterArgs.filter_flag_type_col]
    ## Prename
    mc_chromatlas_filtered[MetadataCols.prename_col] = label_initial + "_" + mc_chromatlas_filtered[LabelCols.label_col]
    mc_chromatlas_removed[MetadataCols.prename_col] = label_initial + "_" + mc_chromatlas_removed[LabelCols.label_col]

    ## Simple name
    mc_chromatlas_filtered[MetadataCols.simple_name_col] = mc_chromatlas_filtered[LabelCols.label_col]
    mc_chromatlas_removed[MetadataCols.simple_name_col] = mc_chromatlas_removed[LabelCols.label_col]
    ### Annotate: Best label
    mc_chromatlas_filtered_metadata_updates = []
    mc_chromatlas_filtered_images_updated = pd.DataFrame(index=mc_chromatlas_filtered[MetadataCols.unique_id_col])
    mc_chromatlas_filtered_images_existing = set(mc_chromatlas_filtered.images())

    for i, row in mc_chromatlas_filtered.metadata.iterrows():
        # MotifCompendium: Subset
        mc_chromatlas_filtered_i = mc_chromatlas_filtered[[i]]
        label_core_i = row[LabelCols.label_core_col].split("-") if pd.notna(row[LabelCols.label_core_col]) else []
        label_flank_i = row[LabelCols.label_flank_col].split("-") if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = row[MetadataCols.composite_col]
        max_submotifs_i = 2 if composite_i else 1
        direct_type_i = row[DirectTypeArgs.direct_type_col]

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
                target_mc=mc_chromatlas_filtered_i,
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
        mc_chromatlas_filtered_metadata_updates.append(mc_chromatlas_filtered_i.metadata.set_index(MetadataCols.unique_id_col))
    
        new_image_cols = [image_col for image_col in mc_chromatlas_filtered_i.images() if image_col not in mc_chromatlas_filtered_images_existing]
        mc_chromatlas_filtered_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_chromatlas_filtered_i.get_images(image_col),
                index=mc_chromatlas_filtered_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_chromatlas_filtered_images_updated = mc_chromatlas_filtered_images_updated.combine_first(mc_chromatlas_filtered_i_new_images)
        mc_chromatlas_filtered_images_updated.update(mc_chromatlas_filtered_i_new_images)
    # Apply updates: Metadata, Images
    mc_chromatlas_filtered_metadata_updates_df = pd.concat(mc_chromatlas_filtered_metadata_updates)
    mc_chromatlas_filtered.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_chromatlas_filtered.metadata = mc_chromatlas_filtered.metadata.combine_first(mc_chromatlas_filtered_metadata_updates_df)
    mc_chromatlas_filtered.metadata.update(mc_chromatlas_filtered_metadata_updates_df)
    mc_chromatlas_filtered.metadata.reset_index(inplace=True)
    del mc_chromatlas_filtered_metadata_updates, mc_chromatlas_filtered_metadata_updates_df

    for image_col in mc_chromatlas_filtered_images_updated.columns:
        mc_chromatlas_filtered.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_chromatlas_filtered_images_updated[image_col]],
        )
    del mc_chromatlas_filtered_images_updated, mc_chromatlas_filtered_images_existing


    # #### Update MotifCompendium
    #### Update original MotifCompendium
    ### (4) Atlas-cluster:
    # Combine: Update columns
    mc_chromatlases = [mc_chromatlas_filtered, mc_chromatlas_removed]

    mc_chromatlas.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
    mc_chromatlas_images_updated = pd.DataFrame(index=mc_chromatlas.metadata.index)
    mc_chromatlas_images_existing = set(mc_chromatlas.images())

    # Build updates: Metadata, Images
    for mc_chromatlas_i in mc_chromatlases:
        # Metadata
        mc_chromatlas.metadata = mc_chromatlas.metadata.combine_first(mc_chromatlas_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_chromatlas.metadata.update(mc_chromatlas_i.metadata.set_index(MetadataCols.unique_id_col))
        # Images
        new_image_cols = [image_col for image_col in mc_chromatlas_i.images() if image_col not in mc_chromatlas_images_existing]
        mc_chromatlas_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_chromatlas_i.get_images(image_col),
                index=mc_chromatlas_i[MetadataCols.unique_id_col],
            )
            for image_col in new_image_cols
        })
        mc_chromatlas_images_updated = mc_chromatlas_images_updated.combine_first(mc_chromatlas_i_new_images)
        mc_chromatlas_images_updated.update(mc_chromatlas_i_new_images)
    # Apply updates: Metadata, Images
    mc_chromatlas.metadata.reset_index(inplace=True)
    for image_col in mc_chromatlas_images_updated.columns:
        mc_chromatlas.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None 
                for x in mc_chromatlas_images_updated[image_col]],
        )
    del mc_chromatlas_images_updated, mc_chromatlas_images_existing
    # # Share columns
    # metadata_cols = []
    # image_cols = []
    # for mc_chromatlas_i in mc_chromatlases:
    #     metadata_cols = list(set(metadata_cols + mc_chromatlas_i.columns()))
    #     image_cols = list(set(image_cols + mc_chromatlas_i.images()))
    # for mc_chromatlas_i in mc_chromatlases:
    #     for metadata_col in metadata_cols:
    #         if metadata_col not in mc_chromatlas_i.columns():
    #             mc_chromatlas_i.metadata[metadata_col] = None
    #     for logo_i in ['logo (fwd)', 'logo (rev)']:
    #         mc_chromatlas_i.add_logos(
    #             motifs=mc_chromatlas_i.motifs,
    #             image_name=logo_i,
    #             trim=0,
    #         )
    #     for image_col in image_cols:
    #         if image_col not in mc_chromatlas_i.images():
    #             mc_chromatlas_i.add_logos(
    #                 motifs=np.zeros_like(mc_chromatlas_i.get_standard_motif_stack()),
    #                 image_name=image_col,
    #                 trim=False,
    #             )

    # # Update original MotifCompendium: Metacluster
    # mc_chromatlas = MotifCompendium.combine(mc_chromatlases, safe=safe)
    # Readable label: Type
    mask_indirect = ~mc_chromatlas[LabelCols.label_col].isin(FilterArgs.filter_flag_types)
    mc_chromatlas.metadata.loc[mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[2]
    mc_chromatlas.metadata.loc[~mask_indirect, LabelCols.label_conf_col] = LabelCols.label_confs[-1]
    ## Rename: {readable_name}_{counter}
    mc_chromatlas.metadata[MetadataCols.counter_col] = mc_chromatlas.metadata.dropna(subset=[MetadataCols.prename_col]).groupby(MetadataCols.prename_col).cumcount().fillna(0).astype(int)
    mc_chromatlas.metadata[MetadataCols.name_new_col] = mc_chromatlas.metadata[MetadataCols.prename_col] + "_" + mc_chromatlas.metadata[MetadataCols.counter_col].astype(str)
    mc_chromatlas.metadata[MetadataCols.name_col] = mc_chromatlas.metadata[MetadataCols.name_new_col]
    mc_chromatlas.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)
    ## Map: Metacluster to Atlas-cluster: Name, Simple name
    metacluster2atlas_name_map = dict(zip(mc_chromatlas[ClusterArgs.source_cluster_col], mc_chromatlas[MetadataCols.name_col]))
    metacluster2atlas_simple_map = dict(zip(mc_chromatlas[ClusterArgs.source_cluster_col], mc_chromatlas[MetadataCols.simple_name_col]))

    mc_metacluster_combined[VisualizeArgs.atlascluster_name_col] = mc_metacluster_combined[atlascluster_cluster_col].map(metacluster2atlas_name_map)
    mc_metacluster_combined[MetadataCols.simple_name_col] = mc_metacluster_combined[atlascluster_cluster_col].map(metacluster2atlas_simple_map)


    # #### Visualize HTML
    ## Create HTML
    # HTML columns
    html_columns_atlascluster = [html_col for html_col in VisualizeArgs.html_columns if html_col in (mc_chromatlas.columns() + mc_chromatlas.images())]
    html_column_desc_dict_atlascluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_atlascluster}

    os.makedirs(os.path.dirname(html_atlascluster_path), exist_ok=True)
    mc_chromatlas.summary_table_html(
        html_out=html_atlascluster_path,
        columns=[FilterArgs.filter_flag_col]+html_columns_atlascluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_atlascluster,
    )

    os.makedirs(os.path.dirname(html_atlascluster_filtered_path), exist_ok=True)
    mc_chromatlas_filtered = mc_chromatlas[~(mc_chromatlas[FilterArgs.filter_flag_col])]
    mc_chromatlas_filtered.summary_table_html(
        html_out=html_atlascluster_filtered_path,
        columns=html_columns_atlascluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_atlascluster,
    )

    os.makedirs(os.path.dirname(html_atlascluster_path), exist_ok=True)
    mc_chromatlas_removed = mc_chromatlas[mc_chromatlas[FilterArgs.filter_flag_col]]
    mc_chromatlas_removed.summary_table_html(
        html_out=html_atlascluster_removed_path,
        columns=[col for col in html_columns_atlascluster if col in mc_chromatlas_removed.columns()],
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_atlascluster,
    )

    # os.makedirs(os.path.dirname(html_chromatlas_monomer_path), exist_ok=True)
    # mc_chromatlas_monomer = mc_chromatlas_filtered[~mc_chromatlas_filtered[composite_col]]
    # mc_chromatlas_monomer.summary_table_html(
    #     html_out=html_chromatlas_monomer_path,
    #     columns=html_columns,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict,
    # )

    # os.makedirs(os.path.dirname(html_chromatlas_multimer_path), exist_ok=True)
    # mc_chromatlas_multimer = mc_chromatlas_filtered[mc_chromatlas_filtered[composite_col]]
    # mc_chromatlas_multimer.summary_table_html(
    #     html_out=html_chromatlas_multimer_path,
    #     columns=html_columns,
    #     logo_trimming=html_logo_trim,
    #     editable=True,
    #     column_desc_dict=html_column_desc_dict,
    # )


    # #### Save MotifCompendium
    # Save MotifCompendium: Chromatlas Atlas
    os.makedirs(os.path.dirname(mc_atlascluster_path), exist_ok=True)
    mc_chromatlas.save(mc_atlascluster_path)

    os.makedirs(os.path.dirname(mc_atlascluster_filtered_path), exist_ok=True)
    mc_chromatlas_filtered.save(mc_atlascluster_filtered_path)

    os.makedirs(os.path.dirname(mc_atlascluster_removed_path), exist_ok=True)
    mc_chromatlas_removed.save(mc_atlascluster_removed_path)

    # os.makedirs(os.path.dirname(mc_chromatlas_monomer_path), exist_ok=True)
    # mc_chromatlas_monomer.save(mc_chromatlas_monomer_path)

    # os.makedirs(os.path.dirname(mc_chromatlas_multimer_path), exist_ok=True)
    # mc_chromatlas_multimer.save(mc_chromatlas_multimer_path)

    os.makedirs(os.path.dirname(mc_metacluster_combined_path), exist_ok=True)
    mc_metacluster_combined.save(mc_metacluster_combined_path)


    # ### Export
    # Export as MEME txt
    os.makedirs(os.path.dirname(meme_atlascluster_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_chromatlas,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_path,
    )

    os.makedirs(os.path.dirname(meme_atlascluster_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_chromatlas_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_filtered_path,
    )

    os.makedirs(os.path.dirname(meme_atlascluster_removed_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_chromatlas_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_atlascluster_removed_path,
    )

    ## EXPORT: MoDISco HDF5
    logging.info("Saving MoDISco HDF5 files...")
    os.makedirs(os.path.dirname(modisco_atlascluster_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_chromatlas,
        name_col=MetadataCols.name_col,
        save_loc=modisco_atlascluster_path,
    )
    utils_analysis.export_compendium_modisco(
        mc=mc_chromatlas_filtered,
        name_col=MetadataCols.name_col,
        save_loc=modisco_atlascluster_filtered_path,
    )
    utils_analysis.export_compendium_modisco(
        mc=mc_chromatlas_removed,
        name_col=MetadataCols.name_col,
        save_loc=modisco_atlascluster_removed_path,
    )
    # Metadata TSV
    os.makedirs(metadata_dir, exist_ok=True)
    mc_chromatlas.metadata.to_csv(metadata_atlascluster_path, sep="\t", index=False)


    # #### Motif index
    # Load MotifCompendium
    mc_metacluster_counts = MotifCompendium.load(mc_metacluster_counts_path, safe=args.safe)
    mc_metacluster_profile = MotifCompendium.load(mc_metacluster_profile_path, safe=args.safe)

    mc_cluster_counts = MotifCompendium.load(mc_cluster_counts_path, safe=args.safe)
    mc_cluster_profile = MotifCompendium.load(mc_cluster_profile_path, safe=args.safe)

    mc_counts = MotifCompendium.load(mc_counts_path, safe=args.safe)
    mc_profile = MotifCompendium.load(mc_profile_path, safe=args.safe)
    # Update Metaclusters
    for mc_metacluster_i in [mc_metacluster_counts, mc_metacluster_profile]:
        for metadata_col in mc_metacluster_combined.columns():
            if metadata_col not in mc_metacluster_i.columns():
                mc_metacluster_i.metadata[metadata_col] = None
        mc_metacluster_i.metadata.set_index(MetadataCols.unique_id_col, inplace=True)
        mc_metacluster_i.metadata.update(mc_metacluster_combined.metadata.set_index(MetadataCols.unique_id_col))
        mc_metacluster_i.metadata.reset_index(inplace=True)
    # Map: Metacluster to Atlas-cluster: Name, Simple name
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
    # Save MotifCompendium
    mc_metacluster_counts.save(mc_metacluster_counts_indexed_path)
    mc_metacluster_profile.save(mc_metacluster_profile_indexed_path)

    mc_cluster_counts.save(mc_cluster_counts_indexed_path)
    mc_cluster_profile.save(mc_cluster_profile_indexed_path)

    mc_counts.save(mc_counts_indexed_path)
    mc_profile.save(mc_profile_indexed_path)
    # Motif index
    mc_motif_index = pd.concat([
        mc_counts.metadata,
        mc_profile.metadata,
    ], axis=0)
    # Export: Motif index
    mc_motif_index[VisualizeArgs.motif_index_cols_atlascluster].to_csv(metadata_motif_index_path, sep="\t", index=False)
    # Arguments JSON
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
        "counter_col": MetadataCols.counter_col,
        "name_unknown": MetadataCols.name_unknown,
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
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def run_encode_combine(args):
    """ENCODE: combine the TF and Chromatin atlasclusters into ENCODE clusters"""


    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Run started.")

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
    encodecluster_cluster_col = f"{ClusterArgs.cluster_algorithm_merge}_{ClusterArgs.cluster_sim_threshold_force}"

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ## Input: TF atlas
    mc_atlascluster_filtered_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_atlascluster_all.mc")

    mc_counts_tf_indexed_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_counts_tf_indexed_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_counts_tf_indexed_path = os.path.join(args.tf_counts_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    mc_profile_tf_indexed_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_profile_tf_indexed_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_profile_tf_indexed_path = os.path.join(args.tf_profile_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    # Written back into the TF atlas dir, alongside the inputs
    mc_atlascluster_indexed_tf_path = os.path.join(args.tf_combined_dir, "mc", "motifcompendium_atlascluster_all-indexed.mc")

    ## Input: Chromatin atlas
    mc_atlascluster_filtered_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_atlascluster_filtered.mc")
    mc_atlascluster_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_atlascluster_all.mc")

    mc_counts_chrom_indexed_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_counts_chrom_indexed_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_counts_chrom_indexed_path = os.path.join(args.chrom_counts_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    mc_profile_chrom_indexed_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_motif_all-indexed.mc")
    mc_cluster_profile_chrom_indexed_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_cluster_all-indexed.mc")
    mc_metacluster_profile_chrom_indexed_path = os.path.join(args.chrom_profile_dir, "mc", "motifcompendium_metacluster_all-indexed.mc")

    # Written back into the Chromatin atlas dir, alongside the inputs
    mc_atlascluster_indexed_chrom_path = os.path.join(args.chrom_combined_dir, "mc", "motifcompendium_atlascluster_all-indexed.mc")

    ## Output: MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")

    mc_atlascluster_combined_path = os.path.join(mc_dir, "motifcompendium_atlascluster_combined.mc")
    mc_encode_path = os.path.join(mc_dir, "motifcompendium_encode_all.mc")
    mc_encode_filtered_path = os.path.join(mc_dir, "motifcompendium_encode_filtered.mc")
    mc_encode_removed_path = os.path.join(mc_dir, "motifcompendium_encode_removed.mc")
    mc_encode_filtered_vf_path = os.path.join(mc_dir, "motifcompendium_encode_filtered_annotated.mc")
    mc_encode_removed_vf_path = os.path.join(mc_dir, "motifcompendium_encode_removed_annotated.mc")

    ## Output: HTML summary table
    html_dir = os.path.join(args.output_dir, "html")

    html_encode_path = os.path.join(html_dir, "motifcompendium_encode_all.html")
    html_encode_filtered_path = os.path.join(html_dir, "motifcompendium_encode_filtered.html")
    html_encode_removed_path = os.path.join(html_dir, "motifcompendium_encode_removed.html")
    html_encode_filtered_vf_path = os.path.join(html_dir, "motifcompendium_encode_filtered_annotated.html")
    html_encode_removed_vf_path = os.path.join(html_dir, "motifcompendium_encode_removed_annotated.html")
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
    encid_metadata_path = os.path.join(args.output_dir, "raw", "encid_metadata.tsv")

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
    }

    # ## Combine atlascluster
    logging.info("Loading TF and Chromatin atlascluster compendia...")

    # ### Load MotifCompendium
    # Load MotifCompendium
    mc_atlascluster_filtered_tf = MotifCompendium.load(mc_atlascluster_filtered_tf_path, safe=args.safe)
    mc_atlascluster_filtered_chrom = MotifCompendium.load(mc_atlascluster_filtered_chrom_path, safe=args.safe)

    # Add TF, Chrom to source, names, constituents
    mc_atlascluster_filtered_tf[VisualizeArgs.atlas_col] = EncodeCombineArgs.tfatlas_id
    mc_atlascluster_filtered_chrom[VisualizeArgs.atlas_col] = EncodeCombineArgs.chromatlas_id

    # Preserve all columns, None if not specified
    all_metadata_cols = set(mc_atlascluster_filtered_tf.columns()) | set(mc_atlascluster_filtered_chrom.columns())

    for mc in [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom]:
        for col in sorted(all_metadata_cols):
            if col not in mc.columns():
                mc[col] = None

    # Delete all images, except logos
    keep_images = ['logo (fwd)', 'logo (rev)']

    for mc in [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom]:
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
        [mc_atlascluster_filtered_tf, mc_atlascluster_filtered_chrom],
        safe=args.safe,
    )


    # #### Cluster: Meta-clusters
    logging.info("Merging TF and Chromatin atlasclusters into ENCODE clusters...")
    ## Cluster: Manual
    # Mask:
    mask_source = ~np.equal.outer(
        mc_atlascluster_combined.metadata[VisualizeArgs.atlas_col].to_numpy(), 
        mc_atlascluster_combined.metadata[VisualizeArgs.atlas_col].to_numpy()
    )  # Source (TF vs. Chromatin)
    mask_composite = np.equal.outer(
        mc_atlascluster_combined.metadata[MetadataCols.composite_col].to_numpy(), 
        mc_atlascluster_combined.metadata[MetadataCols.composite_col].to_numpy()
    )  # Composite
    mask_sim = mc_atlascluster_combined.similarity > ClusterArgs.cluster_sim_threshold_force  # Similarity threshold

    # Apply mask
    sim_threshold = mc_atlascluster_combined.similarity * mask_source * mask_composite * mask_sim
    np.fill_diagonal(sim_threshold, 0) # Set diagonal to 0

    sim_threshold_sparse = scipy.sparse.coo_matrix(sim_threshold) # Sparse matrix
    # Sort by similarity score (descending), then by row index, then by column index
    cluster_candidates = sorted([(sim_threshold_sparse.data[i], sim_threshold_sparse.row[i], sim_threshold_sparse.col[i]) for i in range(len(sim_threshold_sparse.data))], reverse=True)

    # Create clusters
    clustered = [0] * len(mc_atlascluster_combined)
    cluster_num = 0
    for cluster_candidate in cluster_candidates:
        sim, i, j = cluster_candidate
        if clustered[i] == 0 and clustered[j] == 0:
            # Create new cluster
            mc_atlascluster_combined.metadata.loc[i, encodecluster_cluster_col] = cluster_num
            mc_atlascluster_combined.metadata.loc[j, encodecluster_cluster_col] = cluster_num
            cluster_num += 1
        
            # Record clustered
            clustered[i] += 1 
            clustered[j] += 1

    # Assign singleton clusters to unclustered motifs
    mc_atlascluster_combined.metadata.loc[mc_atlascluster_combined.metadata[encodecluster_cluster_col].isna(), encodecluster_cluster_col] = range(cluster_num, cluster_num + mc_atlascluster_combined.metadata[encodecluster_cluster_col].isna().sum())

    # # Force-cluster
    # mc_atlascluster_combined.cluster(
    #     algorithm=cluster_algorithm_force,
    #     similarity_threshold=cluster_sim_threshold_force,
    #     save_name=encodecluster_cluster_col,
    #     largest_clusters_first=True,
    # )
    # min_len = mc_atlascluster_combined[encodecluster_cluster_col].nunique()

    # for i in range(cluster_max_iter):
    #     print(f"Recursively clustering: {i}")
    #     mc_atlascluster_combined.cluster(
    #         algorithm=cluster_algorithm_force,
    #         similarity_threshold=cluster_sim_threshold_force,
    #         cluster_on=encodecluster_cluster_col,
    #         cluster_on_weight=cluster_weight_col,
    #         save_name=encodecluster_cluster_col,
    #         largest_clusters_first=True,
    #     )

    #     new_min_len = mc_atlascluster_combined[encodecluster_cluster_col].nunique()
    #     print(f"Unique clusters: {new_min_len}")
    #     if min_len == new_min_len:
    #         break
    #     else:
    #         min_len = new_min_len


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
    mc_encode[MetadataCols.unique_id_col] = unique_id_initial + f".C4.TF_Chrom.counts_profile." + mc_encode.metadata.index.map('{:08d}'.format)

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
    logging.info(f"Unique clusters (Indirect): {len(set(membership_cluster))}")


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
            logging.info(f"Unique clusters (Indirect): {len(set(new_membership_cluster))}")
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


    def assign_direct_label(row):
        # Initialize
        label_core = None  # Default: None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])

        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]
        row_label_flank_constituents = row[LabelCols.label_flank_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if is_valid(direct_type) and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4]):
            label_core, label_core_readable = assign_direct_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)

        return label_core, label_core_readable, label_flank, label_combined


    def assign_label(row):
        # Initialize
        label_core = None
        label_core_readable = None
        label_flank = None
        label_combined = None

        # Split: Monomer, Composite
        mask_composite = bool(row[MetadataCols.composite_col])
        mask_has_composite = is_valid(row[LabelCols.label_flank_col])
    
        # Row annotations
        row_label_core = row[LabelCols.label_core_col]
        row_label_core_readable = row[LabelCols.label_core_readable_col]
        row_label_flank = row[LabelCols.label_flank_col]
        row_label_core_constituents = row[LabelCols.label_core_constit_all_col]

        ## Rule #1: Direct readable name
        direct_type = row[DirectTypeArgs.direct_type_col]
        if (
            pd.isna(label_core)
            and pd.isna(label_combined)
            and is_valid(direct_type)
            and any(dt in str(direct_type) for dt in DirectTypeArgs.direct_types[:4])
        ):
            label_core, label_core_readable = assign_direct_label_core(row)
            if (
                is_valid(label_core)
                and mask_composite
                and mask_has_composite
            ):
                label_flank = assign_label_flank(row)
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
            elif is_valid(label_core):
                label_combined = str(label_core)
    
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
                label_flank = row_label_flank
                if is_valid(label_flank):
                    label_combined = str(label_core) + "::" + str(label_flank)
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
            elif is_valid(label_core):
                label_combined = str(label_core)
    
        ## Rule 4: Unknown
        if pd.isna(label_combined):
            label_core = None
            label_core_readable = None
            label_flank = None
            label_combined = MetadataCols.name_unknown
    
        return label_core, label_core_readable, label_flank, label_combined

    # Add human-readable name (Direct)
    mc_encode_filtered.metadata[[LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.label_flank_col, LabelCols.direct_label_col]] = pd.DataFrame(
        mc_encode_filtered.metadata.apply(assign_direct_label, axis=1).tolist(),
        index=mc_encode_filtered.metadata.index
    )
    # Add human-readable name
    mc_encode_filtered.metadata[[LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col]] = pd.DataFrame(
        mc_encode_filtered.metadata.apply(assign_label, axis=1).tolist(),
        index=mc_encode_filtered.metadata.index
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

        # Apply Filter: Removed
        mc_encode_metadata_edited[FilterArgs.filter_flag_col] = mc_encode_metadata_edited[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types[:-1])

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

        # Label
        mc_encode_metadata_edited[LabelCols.label_col] = mc_encode_metadata_edited[LabelCols.label_readable_col]

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
    )

    os.makedirs(os.path.dirname(meme_encode_filtered_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_encode_filtered,
        name_col=MetadataCols.name_col,
        save_loc=meme_encode_filtered_path,
    )

    os.makedirs(os.path.dirname(meme_encode_removed_path), exist_ok=True)
    utils_analysis.export_compendium_meme(
        mc=mc_encode_removed,
        name_col=MetadataCols.name_col,
        save_loc=meme_encode_removed_path,
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
    mc_counts_tf = MotifCompendium.load(mc_counts_tf_indexed_path, safe=args.safe)
    mc_cluster_counts_tf = MotifCompendium.load(mc_cluster_counts_tf_indexed_path, safe=args.safe)
    mc_metacluster_counts_tf = MotifCompendium.load(mc_metacluster_counts_tf_indexed_path, safe=args.safe)

    mc_profile_tf = MotifCompendium.load(mc_profile_tf_indexed_path, safe=args.safe)
    mc_cluster_profile_tf = MotifCompendium.load(mc_cluster_profile_tf_indexed_path, safe=args.safe)
    mc_metacluster_profile_tf = MotifCompendium.load(mc_metacluster_profile_tf_indexed_path, safe=args.safe)

    mc_atlascluster_tf = MotifCompendium.load(mc_atlascluster_tf_path, safe=args.safe)


    mc_counts_chrom = MotifCompendium.load(mc_counts_chrom_indexed_path, safe=args.safe)
    mc_cluster_counts_chrom = MotifCompendium.load(mc_cluster_counts_chrom_indexed_path, safe=args.safe)
    mc_metacluster_counts_chrom = MotifCompendium.load(mc_metacluster_counts_chrom_indexed_path, safe=args.safe)

    mc_profile_chrom = MotifCompendium.load(mc_profile_chrom_indexed_path, safe=args.safe)
    mc_cluster_profile_chrom = MotifCompendium.load(mc_cluster_profile_chrom_indexed_path, safe=args.safe)
    mc_metacluster_profile_chrom = MotifCompendium.load(mc_metacluster_profile_chrom_indexed_path, safe=args.safe)

    mc_atlascluster_chrom = MotifCompendium.load(mc_atlascluster_chrom_path, safe=args.safe)

    # Update Atlasclusters
    for mc_atlascluster_i in [mc_atlascluster_tf, mc_atlascluster_chrom]:
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

    # Motif index
    mc_motif_index = pd.concat([
        mc_counts_tf.metadata,
        mc_profile_tf.metadata,
        mc_counts_chrom.metadata,
        mc_profile_chrom.metadata,
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
        ],
        [
            DirectTypeArgs.category_types[0],
            DirectTypeArgs.category_types[1],
            DirectTypeArgs.category_types[3],
            DirectTypeArgs.category_types[4],
            DirectTypeArgs.category_types[2],
            DirectTypeArgs.category_types[3],
            DirectTypeArgs.category_types[4]
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
    }

    mc_suffixs = {
        'TF': {
            'combined': '_tfatlas',
            'counts': {
                'metacluster': '_metacluster',
                'cluster': '_cluster',
                'motif': '',
            },
            'profile': {
                'metacluster': '_metacluster',
                'cluster': '_cluster',
                'motif': '',
            },
        },
        'Chrom': {
            'combined': '_chromatlas',
            'counts': {
                'metacluster': '_metacluster',
                'cluster': '_cluster',
                'motif': '',
            },
            'profile': {
                'metacluster': '_metacluster',
                'cluster': '_cluster',
                'motif': '',
            },
        },
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
    #   File name: {file_general}{level_suffix}{flag_suffix}{general_suffix}{filetype}
    logging.info("Exporting per-atlas compendia, back into the atlas pipeline directories...")


    def export_level_flag(mc_level_flag, mc_level_dir, mc_level_suffix, suffix_flag, level):
        """Write MEME / metadata TSV / HTML / MoDISco H5 / .mc for one flag slice."""

        def out_path(subdir, suffix_general, suffix_filetype):
            file_name = (EncodeCombineArgs.file_general + mc_level_suffix + suffix_flag
                         + suffix_general + suffix_filetype)
            path = os.path.join(mc_level_dir, subdir, file_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path

        # MEME txt
        path = out_path('export', EncodeCombineArgs.suffix_general, '.meme.txt')
        utils_analysis.export_compendium_meme(
            mc=mc_level_flag,
            name_col=MetadataCols.name_col,
            save_loc=path,
        )
        if args.verbose:
            logging.info(f'Saved at: {path}')

        # MoDISco HDF5
        path = out_path('export', EncodeCombineArgs.suffix_general, '.h5')
        utils_analysis.export_compendium_modisco(
            mc=mc_level_flag,
            name_col=MetadataCols.name_col,
            save_loc=path,
        )
        if args.verbose:
            logging.info(f'Saved at: {path}')

        # Metadata TSV
        path = out_path('export', EncodeCombineArgs.suffix_general_metadata, '.tsv')
        mc_level_flag.metadata.to_csv(path, sep="\t", index=False)
        if args.verbose:
            logging.info(f'Saved at: {path}')

        # HTML summary table: Clusters only, motifs have no logos to summarise
        if level != 'motif':
            path = out_path('html', EncodeCombineArgs.suffix_general, '.html')
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
        path = out_path('mc', EncodeCombineArgs.suffix_general, '.mc')
        mc_level_flag.save(path)
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

    # Save MotifCompendium
    mc_counts_tf.save(mc_counts_tf_indexed_path)
    mc_cluster_counts_tf.save(mc_cluster_counts_tf_indexed_path)
    mc_metacluster_counts_tf.save(mc_metacluster_counts_tf_indexed_path)

    mc_profile_tf.save(mc_profile_tf_indexed_path)
    mc_cluster_profile_tf.save(mc_cluster_profile_tf_indexed_path)
    mc_metacluster_profile_tf.save(mc_metacluster_profile_tf_indexed_path)

    mc_atlascluster_tf.save(mc_atlascluster_indexed_tf_path)


    mc_counts_chrom.save(mc_counts_chrom_indexed_path)
    mc_cluster_counts_chrom.save(mc_cluster_counts_chrom_indexed_path)
    mc_metacluster_counts_chrom.save(mc_metacluster_counts_chrom_indexed_path)

    mc_profile_chrom.save(mc_profile_chrom_indexed_path)
    mc_cluster_profile_chrom.save(mc_cluster_profile_chrom_indexed_path)
    mc_metacluster_profile_chrom.save(mc_metacluster_profile_chrom_indexed_path)

    mc_atlascluster_chrom.save(mc_atlascluster_indexed_chrom_path)


    #### Aggregate: Metadata, hits ----------------------------------
    logging.info("Aggregating metadata and motif hits...")

    # Load metadata
    if args.metadata.endswith(".csv"):
        encode_metadata_df = pd.read_csv(args.metadata, sep=",", index_col=0, dtype="string")
    elif args.metadata.endswith(".tsv"):
        encode_metadata_df = pd.read_csv(args.metadata, sep="\t", index_col=0, dtype="string")
    else:
        logging.error("Metadata file must be a CSV or TSV file.")
        raise ValueError("Metadata file must be a CSV or TSV file.")

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

    # Load: Hits metadata
    metadata_hits_df = pd.read_csv(metadata_hits_path, sep="\t", index_col=0)

    # De-duplicate: Use counts
    metadata_hits_df = metadata_hits_df[metadata_hits_df[MetadataCols.head_col] == 'counts']

    # Peaks bed lookup: ENCODE experiment ID -> regions bed10
    encid_metadata_df = pd.read_csv(encid_metadata_path, sep="\t", index_col="experiment")

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

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _worker_fasta = None

    def _init_worker():
        # each worker opens its own handle -- pysam.FastaFile isn't safe to share
        # across processes (the underlying fd's read position would race)
        global _worker_fasta
        import pysam
        _worker_fasta = pysam.FastaFile(args.hg38)

    def annotate_hits_row(hits_tsv, peaks_bed, hits_tsv_annotated):
        hits_df = pd.read_csv(
            hits_tsv,
            sep="\t",
            header=None,
            names=NARROWPEAKS_CUSTOM_COLS,
            keep_default_na=False,
        )
        hits_anno_df = annotate_hits(hits_df, peaks_bed, _worker_fasta)
        hits_anno_df.to_csv(hits_tsv_annotated, sep="\t", index=False)
        return len(hits_anno_df)


    # Annotate hits, per motif -- parallelized across experiments (rows); each row
    # is independent (its own hits/peaks files), so this parallelizes cleanly.
    os.makedirs(hits_annotated_dir, exist_ok=True)
    n_workers = min(8, len(os.sched_getaffinity(0)))
    mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, mp_context=mp_context) as executor:
        futures = {}
        for i, row in metadata_hits_df.iterrows():
            atlas_i = row[VisualizeArgs.atlas_col]
            if atlas_i == EncodeCombineArgs.tfatlas_id:
                hits_tsv = row['finemo_indexed_combined_0p7']
            else:
                hits_tsv = row['finemo_indexed_combined_0.8']

            # Peaks .bed used to call this experiment's hits
            peaks_bed = encid_metadata_df.loc[i, "regions_bed10"]
            hits_tsv_annotated = os.path.join(hits_annotated_dir, f"{i}.hits.annotated.tsv.gz")

            future = executor.submit(annotate_hits_row, hits_tsv, peaks_bed, hits_tsv_annotated)
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
    def summarize_hits_file(hits_tsv_annotated):
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
        futures = {executor.submit(summarize_hits_file, p): p for p in hits_tsv_annotated_paths}
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
        MetadataCols.posneg_col,
        HitsCols.hits_total_col,
        HitsCols.hits_exonic_frac_col,
        HitsCols.hits_promoter_frac_col,
        HitsCols.hits_intronic_frac_col,
        HitsCols.hits_distal_frac_col,
        HitsCols.region_type_barplot_col,
        HitsCols.distToTSS_boxplot_col,
    ]  # Add new figure columns here
    html_column_desc_dict_encodecluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_encodecluster}

    os.makedirs(os.path.dirname(html_encode_filtered_path), exist_ok=True)
    mc_encode_filtered.summary_table_html(
        html_out=html_encode_filtered_vf_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    os.makedirs(os.path.dirname(html_encode_removed_path), exist_ok=True)
    mc_encode_removed.summary_table_html(
        html_out=html_encode_removed_vf_path,
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

    # Save MotifCompendium: ENCODE
    os.makedirs(os.path.dirname(mc_encode_filtered_vf_path), exist_ok=True)
    mc_encode_filtered.save(mc_encode_filtered_vf_path)
    os.makedirs(os.path.dirname(mc_encode_removed_vf_path), exist_ok=True)
    mc_encode_removed.save(mc_encode_removed_vf_path)

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
        "cluster_algorithm_merge": ClusterArgs.cluster_algorithm_merge,
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
    }

    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    args = setup_parser()

    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("ENCODE GRAMMAR pipeline: run started.")

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

    os.makedirs(args.output_dir, exist_ok=True)
    DIRS = stage_dirs(args.output_dir)

    # Stages requested, kept in pipeline order
    selected = [s for s in STAGE_ORDER if s in set(args.stages)]
    logging.info(f"Output root: {args.output_dir}")
    logging.info(f"Stages to run ({len(selected)}/{len(STAGE_ORDER)}): {', '.join(selected)}")


    ### RUN --------------------------------------------------------------------
    for stage in selected:
        label, atlas = STAGES[stage]
        stage_dir = DIRS[stage]
        logging.info("=" * 78)
        logging.info(f"STAGE {STAGE_ORDER.index(stage) + 1}/{len(STAGE_ORDER)}: {label}")
        logging.info(f"  config profile: {atlas}   output: {stage_dir}")
        logging.info("=" * 78)
        if args.time:
            stage_start = time.time()

        # Bind this stage's config profile before its body runs
        set_default_configs(atlas)
        os.makedirs(stage_dir, exist_ok=True)

        # --- 1/2 & 4/5: per-head sequential clustering -----------------------
        if stage in ("tf-counts", "tf-profile", "chrom-counts", "chrom-profile"):
            atlas_key, head = stage.split("-")            # ('tf'|'chrom', 'counts'|'profile')
            prefix = f"{atlas_key}_{head}"
            sa = stage_args(
                args,
                output_dir=stage_dir,
                head_type=head,
                mcid_version=(args.mcid_version_tf if atlas_key == "tf" else args.mcid_version_chrom),
                input_mc=getattr(args, f"{prefix}_input_mc"),
                input_modisco_h5s=getattr(args, f"{prefix}_modisco_h5s"),
                input_names=getattr(args, f"{prefix}_input_names"),
                input_pfms=None,
                metadata=getattr(args, f"{prefix}_metadata"),
            )
            if atlas_key == "tf":
                run_tf_sequential(sa)
            else:
                run_chrom_sequential(sa)

        # --- 3 & 6: combine the two heads into atlasclusters ------------------
        elif stage in ("tf-combine", "chrom-combine"):
            atlas_key = stage.split("-")[0]
            sa = stage_args(
                args,
                output_dir=stage_dir,
                mcid_version=(args.mcid_version_tf if atlas_key == "tf" else args.mcid_version_chrom),
                counts_dir=DIRS[f"{atlas_key}-counts"],
                profile_dir=DIRS[f"{atlas_key}-profile"],
            )
            if atlas_key == "tf":
                run_tf_combine(sa)
            else:
                run_chrom_combine(sa)

        # --- 7: combine the TF and Chromatin atlases --------------------------
        else:
            missing = [n for n in ("tf_init_metadata", "chrom_init_metadata",
                                   "mc_qc_remove_removed", "mc_qc_pass_filtered",
                                   "mc_qc_pass_lowq", "metadata_edited",
                                   "hg38_fasta", "gencode_gtf")
                       if getattr(args, n) is None]
            if missing:
                logging.error("Stage 'encode-combine' requires: "
                              + ", ".join("--" + m.replace("_", "-") for m in missing))
                raise ValueError(f"Missing required arguments for encode-combine: {missing}")
            sa = stage_args(
                args,
                output_dir=stage_dir,
                mcid_version=args.mcid_version_encode,
                tf_dir=DIRS["tf-combine"],
                tf_counts_dir=DIRS["tf-counts"],
                tf_profile_dir=DIRS["tf-profile"],
                chrom_dir=DIRS["chrom-combine"],
                chrom_counts_dir=DIRS["chrom-counts"],
                chrom_profile_dir=DIRS["chrom-profile"],
                tf_combined_dir=DIRS["tf-combine"],
                chrom_combined_dir=DIRS["chrom-combine"],
            )
            run_encode_combine(sa)

        if args.time:
            logging.info(f"Stage '{stage}' time taken: {time.time() - stage_start:.2f}s")

    logging.info("=" * 78)
    logging.info(f'ENCODE GRAMMAR pipeline: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
