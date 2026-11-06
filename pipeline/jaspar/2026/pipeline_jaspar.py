#!/usr/bin/env python
# coding: utf-8
"""
JASPAR-2026: DL - MotifCompendium Pipeline.

Label TF ChIP-seq (BPNet) motifs as direct (matching the target's JASPAR / in vitro
reference) or indirect, filter, cluster within target, meta-cluster, and export
MotifCompendium / HTML / MEME (and optionally MoDISco) summaries. Follows pipeline_tfatlas.py
where the two overlap.
"""
import os
import sys
import re
import json
import time
import argparse
import logging
import warnings
import dataclasses

import h5py
import numpy as np
import pandas as pd

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif

import configs


#### ARGUMENTS ------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(description="Run a MotifCompendium pipeline for JASPAR-2026 DL collection.")

    # Run type
    parser.add_argument("-vr", "--version", type=str, required=True, help="The version to use in the unique identifier (e.g., 2026).")

    # Input
    parser.add_argument("-im", "--input-mc", type=str, default=None, help="Path to the input MotifCompendium object.")
    parser.add_argument("-ih", "--input-modisco-h5s", nargs="+", type=str, default=None, help="Path(s) to the input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-sh", "--input-subpatterns", action="store_true", help="Use subpatterns (instead of main patterns) when loading input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-ip", "--input-pfms", nargs="+", type=str, default=None, help="Path(s) to the input motif PFM files, in PFM or MEME format.")
    parser.add_argument("-in", "--input-names", nargs="+", type=str, default=None, help="Nickname(s) of the input file(s): TF-MoDISco (lite) H5, PFMs, MEME.")
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, per model: CSV, TSV format.")

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")

    # Reference databases: set these to your own local paths
    parser.add_argument("--jaspar-core", type=str, required=True, help="Path to the JASPAR CORE (homo sapiens) reference PFM, as MotifCompendium object (.mc) or MEME file (.txt).")
    parser.add_argument("--jaspar-unvalid", type=str, required=True, help="Path to the JASPAR UNVALIDATED (homo sapiens) reference PFM, as MotifCompendium object (.mc) or MEME file (.txt).")
    parser.add_argument("--ref", type=str, required=True, help="Path to the core reference PFM, as MotifCompendium object (.mc) or MEME file (.txt).")
    parser.add_argument("--ref-invitro", type=str, required=True, help="Path to the in vitro reference PFM, as MotifCompendium object (.mc) or MEME file (.txt).")

    # Compute
    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("--var-chunk", action="store_true", help="Starting from max_chunk, dynamically reduce chunk size, based on available CPU/GPU memory.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Do not check the validity of the MotifCompendium objects.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not scale motifs by information content.")
    parser.add_argument("--fast-plot", action="store_true", help="Reduce the resolution of logo images for faster plotting.")
    parser.add_argument("--save-all", action="store_true", help="Save all intermediate MotifCompendium objects.")
    parser.add_argument("--export-modisco", action="store_true", help="Also export TF-MoDISco HDF5 files (default: off).")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ------------------------------------------------------------
def set_default_configs():
    global TFMoDIScoCols, MetadataCols, TFAnnotationCols, ReferenceMatchCols, LabelCols, MotifMatchArgs
    global DirectTypeArgs, FilterArgs, MotifFilterArgs, ClusterArgs, VisualizeArgs, MotifCompendiumArgs, JasparPipelineArgs

    TFMoDIScoCols = configs.TFMoDIScoCols()
    MetadataCols = configs.MetadataCols()
    TFAnnotationCols = configs.TFAnnotationCols()
    ReferenceMatchCols = configs.ReferenceMatchCols()
    LabelCols = configs.LabelCols()
    MotifMatchArgs = configs.MotifMatchArgs()
    DirectTypeArgs = configs.DirectTypeArgs()
    FilterArgs = configs.FilterArgs()
    MotifFilterArgs = configs.MotifFilterArgs()
    ClusterArgs = configs.ClusterArgs()
    VisualizeArgs = configs.VisualizeArgs()
    MotifCompendiumArgs = configs.MotifCompendiumArgs()
    JasparPipelineArgs = configs.JasparPipelineArgs()


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
    """Load the args.json of each upstream run so provenance chains across pipeline runs.

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
def cuda_device_available() -> tuple[bool, str | None]:
    """Return whether CuPy can access at least one CUDA device.

    ``--use-gpu`` is also used by the batch launcher on CPU-only nodes. CuPy imports there, but
    raises ``cudaErrorNoDevice`` only at the first similarity calculation, so check up front.
    """
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return True, None
        return False, "CuPy reported zero CUDA devices"
    except Exception as error:
        return False, str(error)


def log_time(start: float, label: str) -> None:
    logging.info(f"Time taken ({label}): {time.time() - start:.2f}s")


def load_compendium(path: str, safe: bool) -> MotifCompendium.MotifCompendium:
    """Load a MotifCompendium object (.mc), or build one from a PFM / MEME file."""
    if path.endswith(".mc"):
        return MotifCompendium.load(path, safe=safe)
    return MotifCompendium.build_from_pfm(
        pfm_dict={os.path.splitext(os.path.basename(path))[0]: path},
        safe=safe,
    )


def make_unique_ids(n: int, unique_id_initial: str, level: int) -> list[str]:
    """Unique IDs {unique_id_initial}.C{level}.{index}: level 0 = motifs, 1 = clusters, 2 = meta-clusters."""
    return [f"{unique_id_initial}.C{level}.{i:08d}" for i in range(n)]


def load_hits(model_metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read the FiNeMo hit counts and hit PPMs of every model.

    Row i of a model's motif report is row i of its PPM array (N, 4, L). Motifs without hits have
    NaN PPMs, which become zeros.

    Returns:
        The hits table (motif name, number of hits) and a dict {motif name: (L, 4) PPM}.
    """
    for col in (MetadataCols.hits_report_tsv_col, MetadataCols.hits_ppm_npy_col):
        if col not in model_metadata.columns:
            raise KeyError(f"Metadata column required for FiNeMo hits is missing: {col}")

    hits_dfs = []
    ppm_by_name = {}
    for _, row in model_metadata.iterrows():
        report_df = pd.read_csv(row[MetadataCols.hits_report_tsv_col], sep="\t")
        ppms = np.load(row[MetadataCols.hits_ppm_npy_col], allow_pickle=True).astype(np.float64)
        ppms = np.transpose(np.nan_to_num(ppms, nan=0.0), (0, 2, 1))  # (N, L, 4)
        if len(report_df) != len(ppms):
            raise ValueError(
                f"FiNeMo report and PPMs differ in length for {row[MetadataCols.model_col]}: "
                f"{len(report_df)} vs {len(ppms)}"
            )

        # Older reports name motifs without the experiment string
        motif_names = report_df[MetadataCols.hits_motif_name_col]
        encid = row[MetadataCols.id_col]
        if not motif_names.str.contains(encid, regex=False).all():
            motif_names = motif_names.str.replace(".", f".{encid}_{row[MetadataCols.head_col]}_", regex=False)

        hits_dfs.append(pd.DataFrame({
            MetadataCols.hits_motif_name_col: motif_names.to_numpy(),
            MetadataCols.num_hits_col: report_df[MetadataCols.hits_num_col].to_numpy(),
        }))
        ppm_by_name.update(zip(motif_names, ppms))

    hits_df = pd.concat(hits_dfs, ignore_index=True)
    if hits_df[MetadataCols.hits_motif_name_col].duplicated().any():
        raise ValueError("Motif names in the FiNeMo reports are not unique.")
    return hits_df, ppm_by_name


def stack_ppms(ppm_by_name: dict[str, np.ndarray], names: pd.Series, length: int) -> np.ndarray:
    """Stack the hit PPMs of the given motifs, zero-padded at the end to the CWM length."""
    ppms = np.zeros((len(names), length, 4))
    for i, name in enumerate(names):
        ppm = ppm_by_name[name]
        if ppm.shape[0] > length:
            raise ValueError(f"Hit PPM is longer than the CWM for {name}: {ppm.shape[0]} > {length}")
        ppms[i, :ppm.shape[0], :] = ppm
    return ppms


def l1_normalize(motifs: np.ndarray) -> np.ndarray:
    """L1-normalize each position; all-zero positions stay zero."""
    totals = motifs.sum(axis=2, keepdims=True)
    return np.divide(motifs, totals, out=np.zeros_like(motifs), where=totals != 0)


def _restore_dtypes(metadata: pd.DataFrame, dtypes: pd.Series) -> None:
    """Undo lossless upcasts from combine_first: int columns back to int, bool-only object columns to bool."""
    for col in metadata.columns:
        values = metadata[col]
        if len(values) == 0 or values.isna().any():
            continue
        original = dtypes.get(col)
        if (
            original is not None
            and pd.api.types.is_integer_dtype(original)
            and pd.api.types.is_float_dtype(values)
            and (values == values.round()).all()
        ):
            metadata[col] = values.astype(original)
        elif values.dtype == object and values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            metadata[col] = values.astype(bool)


def update_from_slices(
    mc_target: MotifCompendium.MotifCompendium,
    mc_slices: list[MotifCompendium.MotifCompendium],
    key_col: str,
) -> None:
    """Write the metadata and images of annotated slices back into mc_target, matched on key_col.

    Metadata is merged with combine_first + update: new columns are added and non-NaN slice values
    overwrite. Every image column a slice carries is written for that slice's rows. mc_target keeps
    its row order, its column order (new columns appended), and a RangeIndex.
    """
    mc_slices = [mc_slice for mc_slice in mc_slices if len(mc_slice) > 0]
    if not mc_slices:
        return

    target_metadata = mc_target.metadata
    target_keys = pd.Index(target_metadata[key_col], name=key_col)
    if target_keys.has_duplicates:
        raise ValueError(f"{key_col} is not unique in the target MotifCompendium.")

    # Metadata
    updates = pd.concat([mc_slice.metadata.set_index(key_col) for mc_slice in mc_slices])
    updates = updates[~updates.index.duplicated(keep="last")]
    if not updates.index.isin(target_keys).all():
        raise ValueError(f"Slice {key_col} values are missing from the target MotifCompendium.")

    metadata = target_metadata.set_index(key_col)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # pandas: incompatible-dtype setitem in update()
        metadata = metadata.combine_first(updates)
        metadata.update(updates)
    metadata = metadata.reindex(target_keys).reset_index()
    columns = list(target_metadata.columns)
    metadata = metadata[columns + [col for col in metadata.columns if col not in columns]]
    _restore_dtypes(metadata, target_metadata.dtypes)
    mc_target.metadata = metadata

    # Images
    for image_col in dict.fromkeys(col for mc_slice in mc_slices for col in mc_slice.images()):
        images = pd.Series(
            mc_target.get_images(image_col) if image_col in mc_target.images() else [None] * len(mc_target),
            index=target_keys,
            dtype=object,
        )
        for mc_slice in mc_slices:
            if image_col in mc_slice.images():
                images.loc[mc_slice.metadata[key_col].to_numpy()] = mc_slice.get_images(image_col)
        mc_target.set_images(image_col, [x if isinstance(x, str) else None for x in images.tolist()])


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
        operation (str): The operation to use for the filter. One of: <, <=, >, >=, ==, !=, isna, notna, contains, isin
        threshold (int | float | bool | str | list | set | tuple): The threshold to use for the filter.
        invert (bool): Whether to invert the filter or not.
        override (bool): Whether to override the existing flag or not.
        flag_type (str): The type of flag to apply.
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


def _apply_filters(mc, filters, level, filter_flag_col, filter_flag_type_col, label):
    """Apply each filter that is enabled for this level ('motif' or 'cluster'), in order."""
    for filter_args in filters:
        if getattr(filter_args, f"apply_{level}"):
            logging.info(f"Applying {label}: {filter_args.name}")
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


def _filter_cascade(motif_filter_args: configs.MotifFilterArgs) -> list[tuple[tuple, str]]:
    """Soft, soft override (reference match), moderate, moderate override (direct), hard."""
    return [
        (motif_filter_args.soft_filters, "filter"),
        (motif_filter_args.soft_override_filters, "override filter"),
        (motif_filter_args.moderate_filters, "moderate filter"),
        (motif_filter_args.moderate_override_filters, "moderate override filter"),
        (motif_filter_args.hard_filters, "hard filter"),
    ]


def apply_filter_initial(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """Apply the hard filters to motifs, before annotation."""
    _apply_filters(mc, motif_filter_args.hard_filters, "motif", filter_flag_col, filter_flag_type_col, "hard filter")


def apply_filter_motifs(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """Apply the filter cascade to annotated motifs: direct motifs are only removed by hard filters."""
    for filters, label in _filter_cascade(motif_filter_args):
        _apply_filters(mc, filters, "motif", filter_flag_col, filter_flag_type_col, label)


def apply_filter_clusters(
    mc: MotifCompendium,
    motif_filter_args: configs.MotifFilterArgs,
    filter_flag_col: str,
    filter_flag_type_col: str,
) -> None:
    """Apply the filter cascade to clusters or meta-clusters."""
    for filters, label in _filter_cascade(motif_filter_args):
        _apply_filters(mc, filters, "cluster", filter_flag_col, filter_flag_type_col, f"cluster {label}")


def export_compendium_meme(
    mc: MotifCompendium,
    name_col: str,
    save_loc: str,
) -> None:
    """Export only finite, non-empty motifs in MEME format.

    ``utils_motif.trim_motif`` returns ``None`` for an all-zero motif, which the MotifCompendium
    MEME exporter does not handle. The complete MotifCompendium remains available as ``.mc``.
    """
    motifs = mc.get_standard_motif_stack()
    exportable = np.isfinite(motifs).all(axis=(1, 2)) & np.any(motifs != 0, axis=(1, 2))

    if not np.all(exportable):
        skipped_names = mc[~exportable][name_col].tolist()
        logging.warning(
            "Skipping %d empty or non-finite motif(s) in MEME export to %s: %s",
            len(skipped_names),
            save_loc,
            ", ".join(map(str, skipped_names)),
        )
        mc = mc[exportable]

    utils_analysis.export_compendium_meme(
        mc=mc,
        name_col=name_col,
        save_loc=save_loc,
    )


def cluster_iterative(
    mc: MotifCompendium,
    save_name: str,
    cluster_within: str | None = None,
    verbose: bool = False,
) -> None:
    """Cluster as pipeline_tfatlas.py.

    CPM-Leiden, then up to cluster_max_iter_main recursive CPM-Leiden passes and up to
    cluster_max_iter_force forced DCC passes on the cluster averages, until the membership stops
    changing. Every step is followed by a k-centroids reassignment of the motifs.
    """
    if len(mc) == 0:
        return

    within = {} if cluster_within is None else {"cluster_within": cluster_within}
    on = {"cluster_on": save_name} if cluster_within is None else {"cluster_within_on": (cluster_within, save_name)}

    def reassign():
        mc.cluster(
            algorithm=ClusterArgs.cluster_algorithm_kmeans,
            save_name=save_name,
            largest_clusters_first=True,
            init_clustering_col=save_name,
            weight_col=ClusterArgs.average_weight_col,
            assignment_threshold=ClusterArgs.cluster_sim_assignment_threshold_kmeans,
            n_iterations=ClusterArgs.n_iterations,
            **within,
        )

    # Initial: Leiden CPM
    mc.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=save_name,
        largest_clusters_first=True,
        n_iterations=ClusterArgs.n_iterations,
        seeds=ClusterArgs.seeds,
        **within,
    )
    reassign()
    membership = mc[save_name].to_numpy()
    if verbose:
        logging.info(f"  Unique clusters ({save_name}): {len(set(membership))}")

    # Recursive: Leiden CPM; Force: DCC
    passes = [
        (ClusterArgs.cluster_algorithm_main, ClusterArgs.cluster_sim_threshold, ClusterArgs.cluster_max_iter_main,
         {"n_iterations": ClusterArgs.n_iterations, "seeds": ClusterArgs.seeds}),
        (ClusterArgs.cluster_algorithm_force, ClusterArgs.cluster_sim_threshold_force, ClusterArgs.cluster_max_iter_force,
         {"density": ClusterArgs.cluster_force_density, "seed": ClusterArgs.seeds[0]}),
    ]
    for algorithm, similarity_threshold, max_iter, algorithm_kwargs in passes:
        for i in range(max_iter):
            mc.cluster(
                algorithm=algorithm,
                similarity_threshold=similarity_threshold,
                save_name=save_name,
                weight_col=ClusterArgs.average_weight_col,
                largest_clusters_first=True,
                **on,
                **algorithm_kwargs,
            )
            reassign()

            # Check convergence
            new_membership = mc[save_name].to_numpy()
            if verbose:
                logging.info(f"  Recursively clustering ({algorithm}) {i}: {len(set(new_membership))} clusters")
            if np.array_equal(membership, new_membership):
                break
            membership = new_membership

    logging.info(f"Final unique clusters ({save_name}): {mc[save_name].nunique()} from {len(mc)}")


def aggregate_weighted_avg(metadata: pd.DataFrame, cluster_col: str, agg_col: str, weight_col: str) -> pd.Series:
    """Weighted average of agg_col per cluster, ignoring NaN values; indexed by cluster."""
    valid = metadata[agg_col].notna()
    values = metadata.loc[valid, agg_col].astype(float)
    weights = metadata.loc[valid, weight_col].astype(float)
    clusters = metadata.loc[valid, cluster_col]
    return (values * weights).groupby(clusters).sum() / weights.groupby(clusters).sum()


def merge_constituents(row: pd.Series) -> str:
    """Merge comma-separated constituent lists, keeping first-seen order."""
    return ",".join(dict.fromkeys(
        part
        for agg in row
        if pd.notna(agg) and agg != ""
        for part in agg.split(",")
        if part not in ("", "None")
    ))


def average_clusters(mc: MotifCompendium, clustering_col: str) -> MotifCompendium:
    """Cluster averages (weighted by average_weight_col) with the configured aggregations."""
    columns = mc.columns()
    aggregations = [aggregation for aggregation in ClusterArgs.aggregations if aggregation[0] in columns]
    mc_average = mc.cluster_averages(
        clustering=clustering_col,
        aggregations=aggregations,
        weight_col=ClusterArgs.average_weight_col,
        compute_quality_stats=False,
    )

    # Weighted averages: NaN-safe
    for agg_col, _, new_col in ClusterArgs.weighted_aggregations:
        if agg_col in columns:
            mc_average[new_col] = aggregate_weighted_avg(
                mc.metadata, clustering_col, agg_col, ClusterArgs.average_weight_col,
            ).reindex(mc_average[ClusterArgs.source_cluster_col]).to_numpy()

    # Merge: constituents of the previous level into all constituents
    for prev_col, _, all_col in ClusterArgs.merge_aggregations:
        if all_col not in mc_average.columns():
            mc_average[all_col] = None
        if prev_col in mc_average.columns():
            mc_average[all_col] = mc_average.metadata[[prev_col, all_col]].apply(merge_constituents, axis=1)

    return mc_average


def average_ppms(mc_ppm: MotifCompendium, mc_level: MotifCompendium, clustering_col: str) -> MotifCompendium:
    """Average hit PPMs per cluster (weighted by ppm_weight_col), in the row order and with the metadata of mc_level."""
    mc_level_ppm = mc_ppm.cluster_averages(
        clustering=clustering_col,
        aggregations=[],
        weight_col=ClusterArgs.ppm_weight_col,
        compute_quality_stats=False,
    )
    order = pd.Index(mc_level_ppm[ClusterArgs.source_cluster_col]).get_indexer(mc_level[ClusterArgs.source_cluster_col])
    if len(order) != len(mc_level_ppm) or (order < 0).any():
        raise ValueError(f"PPM and CWM {clustering_col} averages do not cover the same clusters.")
    mc_level_ppm = mc_level_ppm[order.tolist()]
    mc_level_ppm.metadata = mc_level.metadata.copy()
    mc_level_ppm.motifs = l1_normalize(mc_level_ppm.motifs)
    return mc_level_ppm


def to_pfm(mc_ppm: MotifCompendium) -> MotifCompendium:
    """Counts: probability x number of hits."""
    mc_pfm = mc_ppm.copy()
    mc_pfm.motifs = mc_pfm.motifs * mc_pfm[MetadataCols.num_hits_col].to_numpy(dtype=float)[:, np.newaxis, np.newaxis]
    return mc_pfm


def type_tokens(direct_type) -> set[str]:
    """The types in a (possibly comma-joined) direct type value.

    Compared as whole tokens, never substrings: 'INDIRECT' contains 'DIRECT'.
    """
    return set(str(direct_type).split(",")) if is_valid(direct_type) else set()


def is_direct(direct_types: pd.Series) -> pd.Series:
    """Whether a (possibly comma-joined) direct type contains a direct type."""
    return direct_types.map(lambda x: bool(type_tokens(x) & set(DirectTypeArgs.direct_types))).astype(bool)


#### LABELS ---------------------------------------------------------------------
def annotate_reference_labels(
    mc: MotifCompendium,
    readable_map: dict[str, str],
    rows: pd.Series | np.ndarray | None = None,
    set_composite: bool = True,
) -> None:
    """Core / flank labels and subunit counts from the general reference match (pipeline_tfatlas.py).

    A match is a monomer (score0 > min_single_score, no later subunit above min_composite_score),
    a dimer, or a trimer (every subunit above min_composite_score). Only `rows` are labelled.
    """
    name_cols = [f"{ReferenceMatchCols.ref_col}_name{i}" for i in range(MotifMatchArgs.max_submotifs)]
    score_cols = [f"{ReferenceMatchCols.ref_col}_score{i}" for i in range(MotifMatchArgs.max_submotifs)]
    readable_cols = [f"{ReferenceMatchCols.ref_readable_col}_name{i}" for i in range(MotifMatchArgs.max_submotifs)]
    for name_col, readable_col in zip(name_cols, readable_cols):
        mc[readable_col] = mc[name_col].map(readable_map)

    metadata = mc.metadata
    rows = np.ones(len(mc), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    above = [(metadata[score_col] > MotifMatchArgs.min_composite_score).to_numpy() for score_col in score_cols]
    monomer = rows & (metadata[score_cols[0]] > MotifMatchArgs.min_single_score).to_numpy() & ~above[1] & ~above[2]
    dimer = rows & above[0] & above[1] & ~above[2]
    trimer = rows & above[0] & above[1] & above[2]
    matched = monomer | dimer | trimer

    # Dimer-style notation
    metadata.loc[matched, LabelCols.label_core_col] = metadata.loc[matched, readable_cols[0]]
    metadata.loc[matched, LabelCols.label_core_readable_col] = metadata.loc[matched, readable_cols[0]]
    metadata.loc[dimer, LabelCols.label_flank_col] = metadata.loc[dimer, readable_cols[1]]
    metadata.loc[trimer, LabelCols.label_flank_col] = metadata.loc[trimer, readable_cols[1]] + "::" + metadata.loc[trimer, readable_cols[2]]
    if set_composite:
        metadata.loc[dimer, MetadataCols.num_subunit_col] = 2
        metadata.loc[trimer, MetadataCols.num_subunit_col] = 3
        metadata[MetadataCols.composite_col] = metadata[MetadataCols.num_subunit_col] > MotifMatchArgs.min_composite_subunit

    # Concatenate all unique annotations
    metadata[ReferenceMatchCols.ref_name_col] = metadata[name_cols].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1,
    )


def set_label_fields(mc: MotifCompendium, direct_mask: pd.Series | np.ndarray, readable_map: dict[str, str]) -> None:
    """Label columns of pipeline_tfatlas.py.

    Direct rows take their target (and its family) as the core label, as in TF-Atlas. Other rows
    take the general reference match. Composites take the flank from the reference subunits.
    """
    direct_mask = np.asarray(direct_mask, dtype=bool)
    mc.metadata = mc.metadata.assign(**{
        DirectTypeArgs.tf_direct_col:                  None,
        DirectTypeArgs.family_direct_col:              None,
        MetadataCols.num_subunit_col:                  1,
        MetadataCols.num_subunit_direct_col:           None,
        MetadataCols.composite_col:                    False,
        LabelCols.label_col:                           None,
        LabelCols.direct_label_col:                    None,
        LabelCols.label_core_col:                      None,
        LabelCols.direct_label_core_col:               None,
        LabelCols.label_core_readable_col:             None,
        LabelCols.direct_label_core_readable_col:      None,
        LabelCols.label_flank_col:                     None,
    }).copy()
    annotate_reference_labels(mc, readable_map)

    # Direct: target, target family
    metadata = mc.metadata
    target = metadata.loc[direct_mask, MetadataCols.tf_col]
    target_family = metadata.loc[direct_mask, TFAnnotationCols.family_col]
    metadata.loc[direct_mask, DirectTypeArgs.tf_direct_col] = target
    metadata.loc[direct_mask, DirectTypeArgs.family_direct_col] = target_family
    metadata.loc[direct_mask, LabelCols.label_core_col] = target
    metadata.loc[direct_mask, LabelCols.direct_label_core_col] = target
    metadata.loc[direct_mask, LabelCols.label_core_readable_col] = target_family
    metadata.loc[direct_mask, LabelCols.direct_label_core_readable_col] = target_family
    metadata.loc[direct_mask, MetadataCols.num_subunit_direct_col] = metadata.loc[direct_mask, MetadataCols.num_subunit_col]

    # Label: {core}::{flank}
    labelled = metadata[LabelCols.label_core_col].notna().to_numpy()
    has_flank = (metadata[MetadataCols.composite_col].astype(bool) & metadata[LabelCols.label_flank_col].notna()).to_numpy()
    metadata[LabelCols.label_col] = (
        metadata[LabelCols.label_core_col].fillna("").astype(str)
        + np.where(labelled & has_flank, "::" + metadata[LabelCols.label_flank_col].fillna("").astype(str), "")
    )
    metadata.loc[direct_mask, LabelCols.direct_label_col] = metadata.loc[direct_mask, LabelCols.label_col]


def add_counter_names(mc: MotifCompendium) -> None:
    """name = {prename}_{n}, numbering the rows that share a prename in their current order."""
    metadata = mc.metadata
    metadata[MetadataCols.counter_col] = metadata.groupby(MetadataCols.prename_col).cumcount()
    metadata[MetadataCols.name_col] = metadata[MetadataCols.prename_col] + "_" + metadata[MetadataCols.counter_col].astype(str)
    metadata.drop(columns=[MetadataCols.counter_col], inplace=True)


def is_valid(x) -> bool:
    """Pandas-safe validity check."""
    return pd.notna(x) and x != ""


def major_families(frac: pd.Series) -> str | None:
    """Families holding at least the major-family fraction of total contribution, relaxing the
    fraction stepwise; up to max_major_family_count, joined by '-'."""
    if frac.sum() > 0:
        for major_family_frac in np.arange(VisualizeArgs.max_major_family_frac, VisualizeArgs.major_family_increment, VisualizeArgs.major_family_increment):
            major_family_mask = frac >= major_family_frac
            if major_family_mask.any():
                return "-".join(
                    frac[major_family_mask]
                    .sort_values(ascending=False)
                    .index[:VisualizeArgs.max_major_family_count]
                )
    return None


def build_label_pivots(mc_cluster: MotifCompendium, metacluster_col: str, metacluster_ids: pd.Series) -> dict[str, pd.DataFrame]:
    """Per meta-cluster, the fraction of total contribution of each readable family among its clusters."""
    metadata = mc_cluster.metadata.copy()
    split_families = lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x
    metadata[LabelCols.direct_label_core_readable_col_set] = metadata[LabelCols.direct_label_core_readable_col].apply(split_families)
    metadata[LabelCols.label_core_readable_col_set] = metadata[LabelCols.label_core_readable_col].apply(split_families)
    metadata[LabelCols.label_flank_col_set] = metadata[LabelCols.label_flank_col].apply(split_families)

    def pivot(df, col):
        df = df.explode(col).dropna(subset=[col])
        if len(df) == 0:
            return pd.DataFrame(index=metacluster_ids)
        frac = pd.pivot_table(
            df,
            index=metacluster_col,
            columns=col,
            values=MetadataCols.total_contrib_score_col,
            aggfunc="sum",
            fill_value=0,
        )
        frac = frac.div(frac.sum(axis=1), axis=0)
        return frac.reindex(metacluster_ids).fillna(0).drop([""], axis=1, errors="ignore")

    monomer = ~metadata[MetadataCols.composite_col].astype(bool)
    return {
        "direct_monomer": pivot(metadata.loc[monomer], LabelCols.direct_label_core_readable_col_set),
        "direct": pivot(metadata, LabelCols.direct_label_core_readable_col_set),
        "core": pivot(metadata, LabelCols.label_core_readable_col_set),
        "flank": pivot(metadata, LabelCols.label_flank_col_set),
    }


def assign_direct_label_core(row: pd.Series, pivots: dict[str, pd.DataFrame]) -> tuple[str | None, str | None]:
    """Direct core label from constituent clusters (pipeline_tfatlas.py)."""
    direct_label_core = None
    direct_label_core_readable = None
    source_cluster = row[ClusterArgs.source_cluster_col]

    # 1: Single direct target
    tf_direct = row[DirectTypeArgs.tf_direct_col]
    tf_family_direct = row[DirectTypeArgs.family_direct_col]
    if is_valid(tf_direct) and ("," not in str(tf_direct)):
        direct_label_core = str(tf_direct).strip()
        direct_label_core_readable = str(tf_family_direct).strip() if is_valid(tf_family_direct) else direct_label_core

    # 2: Single direct core label among constituents
    direct_label_core_constituents = row[LabelCols.direct_label_core_constit_all_col]
    if (
        direct_label_core is None
        and is_valid(direct_label_core_constituents)
        and ("," not in str(direct_label_core_constituents))
    ):
        direct_label_core = str(direct_label_core_constituents).strip()
        direct_label_core_readable = direct_label_core

    # 3: Major families, direct monomers; 4: Major families, all direct
    for pivot in (pivots["direct_monomer"], pivots["direct"]):
        if direct_label_core is None and source_cluster in pivot.index:
            direct_label_core = major_families(pivot.loc[source_cluster])
            direct_label_core_readable = direct_label_core

    return direct_label_core, direct_label_core_readable


def assign_label_core(row: pd.Series, pivots: dict[str, pd.DataFrame]) -> tuple[str | None, str | None]:
    """Core label from constituent clusters (pipeline_tfatlas.py)."""
    label_core = None
    source_cluster = row[ClusterArgs.source_cluster_col]

    # 1: Single core label among constituents
    label_core_constituents = row[LabelCols.label_core_constit_all_col]
    if is_valid(label_core_constituents) and ("," not in str(label_core_constituents)):
        label_core = str(label_core_constituents).strip()

    # 2: Major families
    if label_core is None and source_cluster in pivots["core"].index:
        label_core = major_families(pivots["core"].loc[source_cluster])

    return label_core, label_core


def assign_label_flank(row: pd.Series, pivots: dict[str, pd.DataFrame]) -> str | None:
    """Flank label from constituent clusters (pipeline_tfatlas.py)."""
    label_flank = None
    source_cluster = row[ClusterArgs.source_cluster_col]

    # 1: Single flank label among constituents
    label_flank_constituents = row[LabelCols.label_flank_constit_all_col]
    if is_valid(label_flank_constituents) and ("," not in str(label_flank_constituents)):
        label_flank = str(label_flank_constituents).strip()

    # 2: Major families
    if label_flank is None and source_cluster in pivots["flank"].index:
        label_flank = major_families(pivots["flank"].loc[source_cluster])

    return label_flank


def _combine_label(row: pd.Series, pivots: dict[str, pd.DataFrame], label_core: str | None, label_flank: str | None = None) -> tuple[str | None, str | None]:
    """{core}::{flank} for composites with a resolvable flank; otherwise {core}."""
    if not is_valid(label_core):
        return label_flank, None
    composite = bool(row[MetadataCols.composite_col])
    has_flank = is_valid(row[LabelCols.label_flank_col]) or is_valid(row[LabelCols.label_flank_constit_all_col])
    if composite and has_flank:
        if not is_valid(label_flank):
            label_flank = assign_label_flank(row, pivots)
        if is_valid(label_flank):
            return label_flank, f"{label_core}::{label_flank}"
    return label_flank, str(label_core)


def assign_labels(row: pd.Series, pivots: dict[str, pd.DataFrame]) -> dict:
    """Direct label (Rule 1) and label (Rules 1-4) of a meta-cluster, by column (pipeline_tfatlas.py): one pass, so every
    rule reads the row as annotated."""
    label_core, label_core_readable, label_flank, label_combined = None, None, None, None

    ## Rule 1: Direct constituents
    if type_tokens(row[DirectTypeArgs.direct_type_col]) & set(DirectTypeArgs.direct_types):
        label_core, label_core_readable = assign_direct_label_core(row, pivots)
        label_flank, label_combined = _combine_label(row, pivots, label_core)
    direct_labels = {
        LabelCols.direct_label_core_col: label_core,
        LabelCols.direct_label_core_readable_col: label_core_readable,
        LabelCols.direct_label_col: label_combined,
    }

    ## Rule 2: Annotated centroid: its own core (and flank, else the constituents' flank)
    if (
        label_combined is None
        and is_valid(row[LabelCols.label_core_col])
        and is_valid(row[LabelCols.label_core_readable_col])
    ):
        label_core = str(row[LabelCols.label_core_col])
        label_core_readable = str(row[LabelCols.label_core_readable_col])
        centroid_flank = row[LabelCols.label_flank_col] if is_valid(row[LabelCols.label_flank_col]) else None
        label_flank, label_combined = _combine_label(row, pivots, label_core, centroid_flank)

    ## Rule 3: Annotated constituents
    if label_combined is None and is_valid(row[LabelCols.label_core_constit_all_col]):
        label_core, label_core_readable = assign_label_core(row, pivots)
        label_flank, label_combined = _combine_label(row, pivots, label_core)

    ## Rule 4: Unknown
    if label_combined is None:
        label_core, label_core_readable, label_flank = None, None, None
        label_combined = MetadataCols.name_unknown

    return {
        **direct_labels,
        LabelCols.label_core_col: label_core,
        LabelCols.label_core_readable_col: label_core_readable,
        LabelCols.label_flank_col: label_flank,
        LabelCols.label_col: label_combined,
    }


#### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()
    set_default_configs()

    ### OPTIONS -----------------------------------------------------------------
    # Set up logging: High-level actions always logged; details gated on --verbose
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if args.use_gpu:
        has_cuda_device, cuda_error = cuda_device_available()
        if not has_cuda_device:
            logging.warning(
                "GPU mode was requested, but no CUDA device is available; "
                f"falling back to CPU mode ({cuda_error})."
            )
            args.use_gpu = False

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

    ### PATHS --------------------------------------------------------------------
    ## Derived identifiers
    unique_id_col = MetadataCols.unique_id_col
    unique_id_initial = f"{unique_id_col}.{args.version}"
    flag_col = FilterArgs.filter_flag_col
    flag_type_col = FilterArgs.filter_flag_type_col
    cluster_col = ClusterArgs.cluster_col
    metacluster_col = ClusterArgs.metacluster_col
    source_cluster_col = ClusterArgs.source_cluster_col

    ## Output directories
    mc_dir = os.path.join(args.output_dir, "mc")
    html_dir = os.path.join(args.output_dir, "html")
    export_dir = os.path.join(args.output_dir, "export")
    metadata_dir = os.path.join(args.output_dir, "metadata")
    for output_dir in (args.output_dir, mc_dir, html_dir, export_dir, metadata_dir):
        os.makedirs(output_dir, exist_ok=True)

    mc_init_path = os.path.join(mc_dir, "motifcompendium_motif_init.mc")
    args_json_path = os.path.join(metadata_dir, "args.json")


    ### BUILD --------------------------------------------------------------------
    # Load the input MotifCompendium object
    if args.input_mc:
        if not os.path.exists(args.input_mc):
            logging.error(f"Input MotifCompendium object not found: {args.input_mc}")
            raise FileNotFoundError(f"Input MotifCompendium object not found: {args.input_mc}")

        logging.info(f"Loading input MotifCompendium object: {args.input_mc}...")
        start_time = time.time()
        mc = MotifCompendium.load(args.input_mc, safe=args.safe)
        if args.verbose:
            logging.info(
                f"Completed loading MotifCompendium object:\n"
                f"  Total number of motifs: {len(mc)}\n"
                f"  Metadata columns: {mc.columns()}"
            )
        if args.time:
            log_time(start_time, "load")

    # Load the input Modisco H5 files
    elif args.input_modisco_h5s:
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
        while True:
            try:
                logging.info("Building MotifCompendium object from input H5 files...")
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
                    log_time(start_time, "build")
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

        logging.info("Loading input PFM files...")
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
            log_time(start_time, "build")

    else:
        logging.error("Input MotifCompendium object or H5 files must be provided.")
        raise ValueError("Input MotifCompendium object or H5 files must be provided.")

    # Load metadata
    if args.metadata:
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

    # Columns left by the previous version of this pipeline in a reused motif_init.mc
    mc.metadata = mc.metadata.drop(columns=JasparPipelineArgs.legacy_cols, errors="ignore").reset_index(drop=True)


    #### PREPROCESS -----------------------------------------------------------------
    logging.info("Preprocessing motifs...")

    # TF-MoDISco pattern name, as used by FiNeMo. Taken from `name` rather than re-assembled from
    # metadata fields: `biosample` is not always the string TF-MoDISco recorded (e.g. "SK-N-SH").
    mc[MetadataCols.name_tfmodisco_col] = (
        mc[MetadataCols.posneg_col]
        + "_" + "patterns."
        + mc[MetadataCols.name_col].str.split(".", n=1).str[1]
    )

    ## Add: Unique ID
    mc[unique_id_col] = make_unique_ids(len(mc), unique_id_initial, 0)


    #### HITS -------------------------------------------------------------------------
    logging.info("Loading FiNeMo hits and hit PPMs...")
    start_time = time.time()
    hits_df, ppm_by_name = load_hits(mc.metadata.drop_duplicates(subset=[MetadataCols.model_col]))
    hits_num = hits_df.set_index(MetadataCols.hits_motif_name_col)[MetadataCols.num_hits_col]
    in_report = mc[MetadataCols.name_tfmodisco_col].isin(hits_num.index)
    mc[MetadataCols.num_hits_col] = mc[MetadataCols.name_tfmodisco_col].map(hits_num).fillna(0).astype(int)

    # Save before any motif is removed: run_pipeline_jaspar.sh reads this file back as --input-mc
    logging.info(f"Saving initial MotifCompendium object: {mc_init_path}...")
    mc.save(mc_init_path)

    # Remove motifs without hits
    has_hits = mc[MetadataCols.num_hits_col] > 0
    logging.info(
        f"Removing motifs without hits: {(~has_hits).sum()} of {len(mc)} "
        f"({(~in_report).sum()} missing from the FiNeMo reports)"
    )
    if args.verbose and (~has_hits).any():
        logging.info(f"Motifs without hits, per target:\n{mc.metadata.loc[~has_hits, MetadataCols.tf_col].value_counts().to_string()}")
    mc = mc[has_hits]

    # Drop path columns
    mc.metadata.drop(columns=[col for col in mc.columns() if col.endswith(('_path', '_paths', '_file', '_files'))], inplace=True)
    if args.time:
        log_time(start_time, "hits")


    #### METRICS ----------------------------------------------------------------------
    logging.info("Calculating motif metrics and filter metrics...")
    # Contribution score
    mc[MetadataCols.avgcontrib_score_col] = mc.motifs.sum(axis=(1, 2))
    mc[MetadataCols.total_contrib_score_col] = mc[MetadataCols.num_seqlets_col] * mc[MetadataCols.avgcontrib_score_col]
    mc[MetadataCols.group_total_contrib_score_col] = mc.metadata.groupby(MetadataCols.id_col)[MetadataCols.total_contrib_score_col].transform('sum')
    mc[MetadataCols.total_contrib_score_ratio_col] = mc[MetadataCols.total_contrib_score_col] / mc[MetadataCols.group_total_contrib_score_col]

    # Filter metrics
    mc[flag_col] = False
    mc[flag_type_col] = None
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )
    # Filter metrics: No scaling, trim zeros
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=['posneg_inverted', 'truncated'],
        trim_importance=0,
    )


    #### REFERENCE --------------------------------------------------------------------
    logging.info("Loading reference motif databases...")
    logging.info(f"  JASPAR CORE: {args.jaspar_core}")
    logging.info(f"  JASPAR UNVALID: {args.jaspar_unvalid}")
    logging.info(f"  Reference: {args.ref}")
    logging.info(f"  Reference (in vitro): {args.ref_invitro}")
    start_time = time.time()
    mc_jaspar_core = load_compendium(args.jaspar_core, args.safe)
    mc_jaspar_unvalid = load_compendium(args.jaspar_unvalid, args.safe)
    mc_ref = load_compendium(args.ref, args.safe)
    mc_ref_invitro = load_compendium(args.ref_invitro, args.safe)

    # Remove multimers: general and in vitro references only. JASPAR references are sliced per
    # target by exact TF, so a composite entry only matches the composite target it belongs to.
    mc_ref = mc_ref[~mc_ref[MetadataCols.name_col].str.contains("::")]
    mc_ref_invitro = mc_ref_invitro[~mc_ref_invitro[MetadataCols.name_col].str.contains("::")]

    # IC scale, once per reference (as pipeline_tfatlas.py)
    for mc_ref_i in (mc_jaspar_core, mc_jaspar_unvalid, mc_ref, mc_ref_invitro):
        mc_ref_i.motifs = utils_motif.ic_scale(mc_ref_i.motifs)

    # JASPAR: CORE + UNVALIDATED
    mc_jaspar = MotifCompendium.combine(
        [mc_jaspar_core, mc_jaspar_unvalid],
        ['JASPAR_CORE', 'JASPAR_UNVALID'],
        safe=args.safe,
    )

    # In vitro reference: JASPAR + in vitro, on their shared metadata columns
    shared_cols = [col for col in mc_ref_invitro.columns() if col in mc_jaspar.columns()]
    mc_jaspar.metadata = mc_jaspar.metadata[shared_cols]
    mc_ref_invitro.metadata = mc_ref_invitro.metadata[shared_cols]
    mc_ref_invitro_jaspar = MotifCompendium.combine(
        [mc_jaspar, mc_ref_invitro],
        ['JASP', 'Reference-invitro'],
        safe=args.safe,
    )

    # Readable name of a general reference match: its TF family
    ref_readable_map = dict(zip(mc_ref[MetadataCols.name_col], mc_ref[TFAnnotationCols.ref_family_col]))

    targets_without_jaspar = sorted(set(mc[MetadataCols.tf_col]) - set(mc_jaspar[TFAnnotationCols.ref_tf_col]))
    logging.info(f"Targets without a JASPAR model: {targets_without_jaspar}")
    if args.time:
        log_time(start_time, "references")


    #### MOTIFS -----------------------------------------------------------------------
    start_time = time.time()
    ## Filter: Hard filters, before annotation
    logging.info("Applying initial motif filters...")
    apply_filter_initial(
        mc=mc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=flag_col,
        filter_flag_type_col=flag_type_col,
    )
    mc_qc = mc[~mc[flag_col].astype(bool)]
    logging.info(f"Motifs passing initial filters: {len(mc_qc)} of {len(mc)}")

    ## Annotate: Direct, against the target's JASPAR + in vitro references
    logging.info("Annotating motifs against each target's JASPAR and in vitro references...")
    mc_qc_slices = []
    targets_without_ref = []
    for i, target in enumerate(mc_qc[MetadataCols.tf_col].unique()):
        mc_i = mc_qc[mc_qc[MetadataCols.tf_col].str.upper() == target.upper()]
        mc_ref_i = mc_ref_invitro_jaspar[mc_ref_invitro_jaspar[TFAnnotationCols.ref_tf_col] == target]
        if args.verbose:
            logging.info(f"Processing {i+1}: {target} ({len(mc_i)} motifs, {len(mc_ref_i)} references)")
        if len(mc_ref_i) == 0:
            targets_without_ref.append(target)
            continue

        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_i,
            assign_from_mc=mc_ref_i,
            save_col_prefix=ReferenceMatchCols.ref_invitro_col,
            min_score=MotifMatchArgs.min_composite_score,
            max_submotifs=MotifMatchArgs.max_submotifs_jasp,
            save_images=False,
            logo_trimming=0,
        )
        mc_qc_slices.append(mc_i)

    update_from_slices(mc_qc, mc_qc_slices, unique_id_col)
    del mc_qc_slices
    logging.info(f"Targets without a JASPAR or in vitro reference: {targets_without_ref}")

    invitro_score_col = f"{ReferenceMatchCols.ref_invitro_col}_score0"
    if invitro_score_col not in mc_qc.columns():
        mc_qc[invitro_score_col] = np.nan
    mc_qc_direct = (mc_qc[invitro_score_col] > MotifMatchArgs.min_single_score_jasp).to_numpy()

    ## Annotate: General reference
    logging.info("Annotating motifs against the general reference...")
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_qc,
        assign_from_mc=mc_ref,
        from_label_col=MetadataCols.name_col,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=False,
        logo_trimming=0,
    )

    ## Labels: Direct, indirect
    mc_qc[DirectTypeArgs.direct_type_col] = np.where(mc_qc_direct, DirectTypeArgs.direct_type, DirectTypeArgs.indirect_type)
    set_label_fields(mc_qc, mc_qc_direct, ref_readable_map)

    ## Filter: Soft, moderate, hard (direct motifs: hard only)
    logging.info("Applying motif filters...")
    apply_filter_motifs(
        mc=mc_qc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=flag_col,
        filter_flag_type_col=flag_type_col,
    )

    # Update: Original MotifCompendium; split
    update_from_slices(mc, [mc_qc], unique_id_col)
    del mc_qc
    motif_removed = mc[flag_col].astype(bool)
    mc_filtered = mc[~motif_removed]
    motif_direct = mc_filtered[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_type
    mc_direct = mc_filtered[motif_direct]
    mc_indirect = mc_filtered[~motif_direct]
    logging.info(
        f"Motifs: total={len(mc)}, direct={len(mc_direct)}, "
        f"indirect={len(mc_indirect)}, removed={motif_removed.sum()}"
    )
    if len(mc_filtered) == 0:
        raise ValueError("No motifs left after filtering.")
    if args.time:
        log_time(start_time, "motifs")


    #### CLUSTERS ---------------------------------------------------------------------
    start_time = time.time()
    ## Cluster: Direct, indirect; within target
    logging.info("Clustering motifs, within target...")
    cluster_iterative(mc_direct, cluster_col, ClusterArgs.cluster_within, args.verbose)
    cluster_iterative(mc_indirect, cluster_col, ClusterArgs.cluster_within, args.verbose)
    if len(mc_direct) > 0 and len(mc_indirect) > 0:
        mc_indirect[cluster_col] = mc_indirect[cluster_col] + mc_direct[cluster_col].max() + 1
    update_from_slices(mc_filtered, [mc_direct, mc_indirect], unique_id_col)
    mc_filtered[cluster_col] = mc_filtered[cluster_col].astype(int)
    del mc_direct, mc_indirect

    ## Average
    logging.info("Creating clusters, by averaging...")
    mc_cluster = average_clusters(mc_filtered, cluster_col)
    mc_cluster[MetadataCols.total_contrib_score_ratio_col] = mc_cluster[MetadataCols.total_contrib_score_col] / mc_cluster[MetadataCols.group_total_contrib_score_col]
    mc_cluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_cluster.get_standard_motif_stack())
    mc_cluster.sort(by=MetadataCols.num_hits_col, ascending=False, inplace=True)  # Order in which a target's clusters are numbered
    mc_cluster.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity,
    )
    mc_cluster[unique_id_col] = make_unique_ids(len(mc_cluster), unique_id_initial, 1)

    ## Annotate: General reference, labels, filter metrics
    logging.info("Annotating clusters against the general reference...")
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_cluster,
        assign_from_mc=mc_ref,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=True,
        logo_trimming=0,
    )
    cluster_direct = (mc_cluster[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_type).to_numpy()
    set_label_fields(mc_cluster, cluster_direct, ref_readable_map)

    mc_cluster[flag_col] = False
    mc_cluster[flag_type_col] = None
    utils_analysis.calculate_filters(
        mc=mc_cluster,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )
    utils_analysis.calculate_filters(
        mc=mc_cluster,
        metric_list=['posneg_inverted', 'truncated'],
        trim_importance=0,
    )

    ## Annotate: Direct clusters, against each target's references
    logging.info("Annotating direct clusters against each target's JASPAR and in vitro references...")
    mc_cluster_slices = []
    mc_cluster_direct = mc_cluster[cluster_direct]
    for i, target in enumerate(mc_cluster_direct[MetadataCols.tf_col].unique()):
        mc_cluster_i = mc_cluster_direct[mc_cluster_direct[MetadataCols.tf_col] == target]
        target_family = mc_cluster_i[TFAnnotationCols.family_col].iloc[0]
        if args.verbose:
            logging.info(f"Processing {i+1}: {target} ({target_family})")

        # In vitro (JASPAR + in vitro), JASPAR CORE, JASPAR UNVALIDATED: target; other: target family
        references = [
            (ReferenceMatchCols.ref_invitro_col, mc_ref_invitro_jaspar[mc_ref_invitro_jaspar[TFAnnotationCols.ref_tf_col] == target]),
            (ReferenceMatchCols.ref_jaspar_core_col, mc_jaspar_core[mc_jaspar_core[TFAnnotationCols.ref_tf_col] == target]),
            (ReferenceMatchCols.ref_jaspar_unvalid_col, mc_jaspar_unvalid[mc_jaspar_unvalid[TFAnnotationCols.ref_tf_col] == target]),
            (ReferenceMatchCols.ref_invitro_other_col, mc_jaspar[mc_jaspar[TFAnnotationCols.ref_family_col] == target_family]),
        ]
        for save_col_prefix, mc_ref_i in references:
            if len(mc_ref_i) > 0:
                utils_analysis.assign_label_from_other_compendium(
                    assign_to_mc=mc_cluster_i,
                    assign_from_mc=mc_ref_i,
                    save_col_prefix=save_col_prefix,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                    save_images=True,
                    logo_trimming=0,
                )
        mc_cluster_slices.append(mc_cluster_i)
    del mc_cluster_direct

    ## Annotate: Indirect clusters, against all in vitro and JASPAR references; filter
    logging.info("Annotating indirect clusters...")
    mc_cluster_indirect = mc_cluster[~cluster_direct]
    if len(mc_cluster_indirect) > 0:
        for save_col_prefix, mc_ref_i in [
            (ReferenceMatchCols.ref_invitro_col, mc_ref_invitro),
            (ReferenceMatchCols.ref_jaspar_col, mc_jaspar),
        ]:
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_indirect,
                assign_from_mc=mc_ref_i,
                save_col_prefix=save_col_prefix,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                save_images=True,
                logo_trimming=0,
            )
        logging.info("Applying cluster filters to indirect clusters...")
        apply_filter_clusters(
            mc=mc_cluster_indirect,
            motif_filter_args=MotifFilterArgs,
            filter_flag_col=flag_col,
            filter_flag_type_col=flag_type_col,
        )
        mc_cluster_slices.append(mc_cluster_indirect)
    del mc_cluster_indirect

    update_from_slices(mc_cluster, mc_cluster_slices, unique_id_col)
    del mc_cluster_slices

    ## Direct clusters: JASPAR CORE, then JASPAR UNVALIDATED, otherwise DIRECT (not confirmed by JASPAR)
    def match_mask(save_col_prefix):
        score_col = f"{save_col_prefix}_score0"
        if score_col not in mc_cluster.columns():
            return np.zeros(len(mc_cluster), dtype=bool)
        return (mc_cluster[score_col] > MotifMatchArgs.min_composite_score).to_numpy()

    mc_cluster[DirectTypeArgs.direct_type_col] = np.select(
        [
            cluster_direct & match_mask(ReferenceMatchCols.ref_jaspar_core_col),
            cluster_direct & match_mask(ReferenceMatchCols.ref_jaspar_unvalid_col),
            cluster_direct,
        ],
        [DirectTypeArgs.jaspar_core_type, DirectTypeArgs.jaspar_unvalid_type, DirectTypeArgs.direct_type],
        default=DirectTypeArgs.indirect_type,
    )

    ## Labels, confidence, names: {target}_{JASPCORE|JASPUNVALID|DIRECT|INDIRECT|flag type}_{n}
    cluster_flag = mc_cluster[flag_col].astype(bool).to_numpy()
    cluster_metadata = mc_cluster.metadata
    cluster_metadata.loc[cluster_flag, LabelCols.label_col] = cluster_metadata.loc[cluster_flag, flag_type_col].fillna(MetadataCols.name_unknown)
    cluster_direct_type = cluster_metadata[DirectTypeArgs.direct_type_col]
    cluster_metadata[LabelCols.label_conf_col] = np.select(
        [
            cluster_flag,
            cluster_direct_type == DirectTypeArgs.jaspar_core_type,
            cluster_direct_type == DirectTypeArgs.jaspar_unvalid_type,
            cluster_direct_type == DirectTypeArgs.direct_type,
            cluster_metadata[LabelCols.label_core_col].notna(),
        ],
        [LabelCols.label_confs[-1], LabelCols.label_confs[0], LabelCols.label_confs[1], LabelCols.label_confs[2], LabelCols.label_confs[2]],
        default=LabelCols.label_confs[-1],
    )
    cluster_metadata[MetadataCols.prename_col] = (
        cluster_metadata[MetadataCols.tf_col]
        + "_" + np.where(cluster_flag, cluster_metadata[LabelCols.label_col], cluster_direct_type)
    )
    cluster_metadata[MetadataCols.simple_name_col] = cluster_metadata[LabelCols.label_col]
    add_counter_names(mc_cluster)
    cluster_kept = ~cluster_flag
    cluster_direct = is_direct(mc_cluster[DirectTypeArgs.direct_type_col]).to_numpy()
    logging.info(
        f"Clusters: total={len(mc_cluster)}, direct={(cluster_kept & cluster_direct).sum()}, "
        f"indirect={(cluster_kept & ~cluster_direct).sum()}, removed={cluster_flag.sum()}"
    )
    if args.time:
        log_time(start_time, "clusters")


    #### METACLUSTERS -----------------------------------------------------------------
    start_time = time.time()
    ## Cluster: Direct, indirect; then combine
    logging.info("Clustering clusters into meta-clusters (direct, indirect)...")
    mc_cluster_direct = mc_cluster[cluster_kept & cluster_direct]
    mc_cluster_indirect = mc_cluster[cluster_kept & ~cluster_direct]
    cluster_iterative(mc_cluster_direct, metacluster_col, ClusterArgs.metacluster_within, args.verbose)
    cluster_iterative(mc_cluster_indirect, metacluster_col, ClusterArgs.metacluster_within, args.verbose)
    if len(mc_cluster_direct) > 0 and len(mc_cluster_indirect) > 0:
        mc_cluster_indirect[metacluster_col] = mc_cluster_indirect[metacluster_col] + mc_cluster_direct[metacluster_col].max() + 1
    update_from_slices(mc_cluster, [mc_cluster_direct, mc_cluster_indirect], unique_id_col)  # Removed clusters: no meta-cluster (NaN)
    del mc_cluster_direct, mc_cluster_indirect

    mc_cluster_kept = mc_cluster[cluster_kept]
    mc_cluster_kept[metacluster_col] = mc_cluster_kept[metacluster_col].astype(int)

    ## Average
    logging.info("Creating meta-clusters, by averaging...")
    mc_metacluster = average_clusters(mc_cluster_kept, metacluster_col)
    mc_metacluster[MetadataCols.total_contrib_score_ratio_col] = mc_metacluster[MetadataCols.total_contrib_score_col] / mc_metacluster[MetadataCols.group_total_contrib_score_col]
    mc_metacluster[MetadataCols.posneg_col] = utils_motif.motif_posneg_sum(mc_metacluster.get_standard_motif_stack())
    mc_metacluster[MetadataCols.composite_col] = mc_metacluster[MetadataCols.composite_col].astype(bool)
    mc_metacluster.sort(by=[MetadataCols.posneg_col, MetadataCols.num_seqlets_col], ascending=False, inplace=True)
    mc_metacluster.add_motif_strings(
        name=VisualizeArgs.motif_string_col,
        importance=VisualizeArgs.motif_string_importance,
        specificity=VisualizeArgs.motif_string_specificity,
    )
    mc_metacluster[unique_id_col] = make_unique_ids(len(mc_metacluster), unique_id_initial, 2)
    metacluster_direct = is_direct(mc_metacluster[DirectTypeArgs.direct_type_col]).to_numpy()

    ## Annotate: General reference, JASPAR
    logging.info("Annotating meta-clusters against the references...")
    mc_metacluster.metadata = mc_metacluster.metadata.assign(**{
        LabelCols.label_col:                      MetadataCols.name_unknown,
        LabelCols.direct_label_col:               None,
        LabelCols.label_core_col:                 None,
        LabelCols.direct_label_core_col:          None,
        LabelCols.label_core_readable_col:        None,
        LabelCols.direct_label_core_readable_col: None,
        LabelCols.label_flank_col:                None,
    }).copy()
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_metacluster,
        assign_from_mc=mc_ref,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=True,
        logo_trimming=0,
    )
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_metacluster,
        assign_from_mc=mc_jaspar,
        save_col_prefix=ReferenceMatchCols.ref_jaspar_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs_jasp,
        save_images=True,
        logo_trimming=0,
    )
    # Centroid labels (indirect); composite is determined by the clusters
    annotate_reference_labels(mc_metacluster, ref_readable_map, rows=~metacluster_direct, set_composite=False)

    ## Filter: Indirect meta-clusters only
    mc_metacluster[flag_col] = False
    mc_metacluster[flag_type_col] = None
    utils_analysis.calculate_filters(
        mc=mc_metacluster,
        metric_list=MotifFilterArgs.motif_metrics,
        trim_importance=FilterArgs.filter_trim_importance,
        trim_length=FilterArgs.filter_trim_length,
    )
    utils_analysis.calculate_filters(
        mc=mc_metacluster,
        metric_list=['posneg_inverted', 'truncated'],
        trim_importance=0,
    )
    apply_filter_clusters(
        mc=mc_metacluster,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=flag_col,
        filter_flag_type_col=flag_type_col,
    )
    mc_metacluster[flag_col] = mc_metacluster[flag_col].astype(bool).to_numpy() & ~metacluster_direct
    metacluster_flag = mc_metacluster[flag_col].to_numpy()
    mc_metacluster_filtered = mc_metacluster[~metacluster_flag]
    mc_metacluster_removed = mc_metacluster[metacluster_flag]

    ## Sort column: Cluster the meta-clusters (for sorting only)
    logging.info("Clustering meta-clusters (for sorting only)...")
    cluster_iterative(mc_metacluster_filtered, VisualizeArgs.sort_col, None, args.verbose)

    ## Labels (pipeline_tfatlas.py), names: {label}_{DIRECT|INDIRECT}_{n}
    logging.info("Assigning meta-cluster labels...")
    pivots = build_label_pivots(mc_cluster_kept, metacluster_col, mc_metacluster_filtered[source_cluster_col])
    if len(mc_metacluster_filtered) > 0:
        # Labels: direct (Rule 1) and label (Rules 1-4), in one pass over the same rows
        label_cols = [
            LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.direct_label_col,
            LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
        ]
        mc_metacluster_filtered.metadata[label_cols] = pd.DataFrame(
            mc_metacluster_filtered.metadata.apply(lambda row: assign_labels(row, pivots), axis=1).tolist(),
            index=mc_metacluster_filtered.metadata.index,
            columns=label_cols,
        )
    mc_metacluster_removed[LabelCols.label_col] = mc_metacluster_removed[flag_type_col].fillna(MetadataCols.name_unknown)

    for mc_metacluster_i in (mc_metacluster_filtered, mc_metacluster_removed):
        group = np.where(
            is_direct(mc_metacluster_i[DirectTypeArgs.direct_type_col]),
            DirectTypeArgs.direct_type,
            DirectTypeArgs.indirect_type,
        )
        mc_metacluster_i[MetadataCols.prename_col] = mc_metacluster_i[LabelCols.label_col].astype(str) + "_" + group
        mc_metacluster_i[MetadataCols.simple_name_col] = mc_metacluster_i[LabelCols.label_col]
    update_from_slices(mc_metacluster, [mc_metacluster_filtered, mc_metacluster_removed], unique_id_col)
    del mc_metacluster_filtered, mc_metacluster_removed

    # Label confidence: JASPAR CORE high; JASPAR UNVALIDATED medium; other direct / indirect low; removed flag
    metacluster_types = mc_metacluster[DirectTypeArgs.direct_type_col].map(type_tokens)
    mc_metacluster[LabelCols.label_conf_col] = np.select(
        [
            metacluster_flag,
            metacluster_types.map(lambda types: DirectTypeArgs.jaspar_core_type in types).to_numpy(dtype=bool),
            metacluster_types.map(lambda types: DirectTypeArgs.jaspar_unvalid_type in types).to_numpy(dtype=bool),
            metacluster_types.map(lambda types: bool(types & {DirectTypeArgs.direct_type, DirectTypeArgs.indirect_type})).to_numpy(dtype=bool),
        ],
        [LabelCols.label_confs[-1], LabelCols.label_confs[0], LabelCols.label_confs[1], LabelCols.label_confs[2]],
        default=LabelCols.label_confs[-1],
    )
    add_counter_names(mc_metacluster)
    logging.info(
        f"Meta-clusters: total={len(mc_metacluster)}, direct={(~metacluster_flag & metacluster_direct).sum()}, "
        f"indirect={(~metacluster_flag & ~metacluster_direct).sum()}, removed={metacluster_flag.sum()}"
    )
    if args.time:
        log_time(start_time, "metaclusters")


    #### PROPAGATE --------------------------------------------------------------------
    logging.info("Propagating cluster / meta-cluster names to all levels...")
    ## Meta-cluster -> Cluster: Name, simple name
    metacluster_name_map = dict(zip(mc_metacluster[source_cluster_col], mc_metacluster[MetadataCols.name_col]))
    metacluster_simple_map = dict(zip(mc_metacluster[source_cluster_col], mc_metacluster[MetadataCols.simple_name_col]))
    mc_cluster[VisualizeArgs.metacluster_name_col] = mc_cluster[metacluster_col].map(metacluster_name_map)
    mc_cluster.metadata.loc[cluster_kept, MetadataCols.simple_name_col] = mc_cluster.metadata.loc[cluster_kept, metacluster_col].map(metacluster_simple_map)
    mc_cluster[VisualizeArgs.sort_col] = mc_cluster[metacluster_col]

    ## Cluster -> Motif: Cluster name, meta-cluster name, simple name
    for col_from, col_to in [
        (MetadataCols.name_col, VisualizeArgs.cluster_name_col),
        (VisualizeArgs.metacluster_name_col, VisualizeArgs.metacluster_name_col),
        (MetadataCols.simple_name_col, MetadataCols.simple_name_col),
    ]:
        mc_filtered[col_to] = mc_filtered[cluster_col].map(dict(zip(mc_cluster[source_cluster_col], mc_cluster[col_from])))
    update_from_slices(mc, [mc_filtered], unique_id_col)  # Removed motifs: no cluster (NaN)
    mc[VisualizeArgs.sort_col] = mc[cluster_col]
    del mc_filtered, mc_cluster_kept

    ## Clean up
    for mc_level in (mc, mc_cluster, mc_metacluster):
        mc_level.metadata.drop(columns=JasparPipelineArgs.drop_cols, errors="ignore", inplace=True)
    # Motif images are not exported (as pipeline_tfatlas.py, to save memory)
    for image_col in mc.images():
        mc.delete_images(image_col)


    #### PPM / PFM --------------------------------------------------------------------
    logging.info("Building hit PPMs and PFMs...")
    start_time = time.time()
    ## Motifs: FiNeMo hit PPMs, matched by TF-MoDISco name
    mc_ppm = mc.copy()
    mc_ppm.motifs = stack_ppms(ppm_by_name, mc[MetadataCols.name_tfmodisco_col], mc.motifs.shape[1]).astype(mc.motifs.dtype)
    mc_ppm.motifs[mc_ppm[flag_col].to_numpy(dtype=bool)] = 0.0  # Removed motifs: no PPM
    mc_ppm.motifs = l1_normalize(mc_ppm.motifs)
    del ppm_by_name

    ## Clusters, meta-clusters: Averages weighted by hits
    mc_ppm_kept = mc_ppm[~mc_ppm[flag_col].to_numpy(dtype=bool)]
    mc_ppm_kept[cluster_col] = mc_ppm_kept[cluster_col].astype(int)
    mc_cluster_ppm = average_ppms(mc_ppm_kept, mc_cluster, cluster_col)
    del mc_ppm_kept
    mc_cluster_ppm_kept = mc_cluster_ppm[~mc_cluster_ppm[flag_col].to_numpy(dtype=bool)]
    mc_cluster_ppm_kept[metacluster_col] = mc_cluster_ppm_kept[metacluster_col].astype(int)
    mc_metacluster_ppm = average_ppms(mc_cluster_ppm_kept, mc_metacluster, metacluster_col)
    del mc_cluster_ppm_kept

    ## PFM: Probability x hits
    mc_pfm = to_pfm(mc_ppm)
    mc_cluster_pfm = to_pfm(mc_cluster_ppm)
    mc_metacluster_pfm = to_pfm(mc_metacluster_ppm)
    if args.time:
        log_time(start_time, "ppm/pfm")


    #### EXPORT -----------------------------------------------------------------------
    start_time = time.time()
    logging.info(f"Saving outputs to: {args.output_dir}...")
    # Saved without --save-all
    core_exports = {("motif", "all"), ("motif", "direct"), ("cluster", "direct")}

    levels = {
        "motif": (mc, mc_ppm, mc_pfm, (mc[DirectTypeArgs.direct_type_col] == DirectTypeArgs.direct_type).to_numpy()),
        "cluster": (mc_cluster, mc_cluster_ppm, mc_cluster_pfm, is_direct(mc_cluster[DirectTypeArgs.direct_type_col]).to_numpy()),
        "metacluster": (mc_metacluster, mc_metacluster_ppm, mc_metacluster_pfm, is_direct(mc_metacluster[DirectTypeArgs.direct_type_col]).to_numpy()),
    }
    counts = {}
    for level, (mc_level, mc_level_ppm, mc_level_pfm, level_direct) in levels.items():
        level_flag = mc_level[flag_col].to_numpy(dtype=bool)
        groups = {
            "all": np.ones(len(mc_level), dtype=bool),
            "direct": ~level_flag & level_direct,
            "indirect": ~level_flag & ~level_direct,
            "removed": level_flag,
        }
        html_columns = [col for col in VisualizeArgs.html_columns if col in (mc_level.columns() + mc_level.images())]

        for group, group_mask in groups.items():
            counts[f"{level}_{group}"] = int(group_mask.sum())
            if not (args.save_all or (level, group) in core_exports):
                continue
            stem = f"motifcompendium_{level}_{group}"
            if group_mask.sum() == 0:
                logging.warning(f"Skipping empty output: {stem}")
                continue
            objects = {"": mc_level[group_mask], "_ppm": mc_level_ppm[group_mask], "_pfm": mc_level_pfm[group_mask]}

            # MotifCompendium objects
            for suffix, mc_i in objects.items():
                mc_i.save(os.path.join(mc_dir, f"{stem}{suffix}.mc"))

            # MEME, MoDISco HDF5 (--export-modisco): Removed motifs / clusters are not exported
            if group != "removed":
                for suffix, mc_i in objects.items():
                    if args.export_modisco:
                        utils_analysis.export_compendium_modisco(
                            mc=mc_i,
                            name_col=MetadataCols.name_col,
                            save_loc=os.path.join(export_dir, f"{stem}{suffix}.h5"),
                        )
                    if level == "motif" and suffix:
                        mc_i = mc_i[~mc_i[flag_col].to_numpy(dtype=bool)]  # Removed motifs have no PPM
                    export_compendium_meme(
                        mc=mc_i,
                        name_col=MetadataCols.name_col,
                        save_loc=os.path.join(export_dir, f"{stem}{suffix}.meme.txt"),
                    )

            # HTML summary table: Clusters, meta-clusters
            if level != "motif":
                objects[""].summary_table_html(
                    html_out=os.path.join(html_dir, f"{stem}.html"),
                    columns=([flag_col] if group == "all" else []) + html_columns,
                    logo_trimming=VisualizeArgs.html_logo_trim,
                    editable=True,
                )

            # Metadata
            objects[""].metadata.to_csv(os.path.join(metadata_dir, f"{stem}_metadata.tsv"), sep="\t", index=False)

    ## Arguments JSON
    args_record = {
        # Every parser argument, verbatim
        "parser_args": vars(args),
        # Upstream provenance: the previous run in this output dir, and the run that produced --input-mc
        "previous_args": collect_previous_args([
            ("previous_run_this_output_dir", args.output_dir),
            ("input_mc_run", os.path.dirname(os.path.dirname(args.input_mc)) if args.input_mc else None),
        ]),
        # Every dataclass in configs.py as used by this run
        "configs": collect_configs_snapshot(),
        "unique_id_initial": unique_id_initial,
        "targets_without_jaspar": targets_without_jaspar,
        "targets_without_reference": targets_without_ref,
        "counts": counts,
    }
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)
    if args.time:
        log_time(start_time, "export")

    ## Overview
    for level in levels:
        logging.info(
            f"{level.capitalize()}s: {counts[f'{level}_all']} "
            f"(direct {counts[f'{level}_direct']}, indirect {counts[f'{level}_indirect']}, removed {counts[f'{level}_removed']})"
        )
    logging.info(f"Clusters, by type:\n{mc_cluster.metadata.loc[~cluster_flag, DirectTypeArgs.direct_type_col].value_counts().to_string()}")
    logging.info(f'Run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
