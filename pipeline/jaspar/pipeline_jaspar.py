#!/usr/bin/env python
# coding: utf-8
"""
JASPAR-2026: DL - MotifCompendium Pipeline.
"""
import os
import sys
import time
import argparse
import logging
import h5py

import numpy as np
import pandas as pd

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif

import pipeline.jaspar.configs as configs


#### ARGUMENTS ------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(description="Run a MotifCompendium pipeline for JASPAR-2026 DL collection.")

    # Input
    parser.add_argument("-im", "--input-mc", type=str, default=None, help="Path to the input MotifCompendium object.")
    parser.add_argument("-ih", "--input-modisco-h5s", nargs="+", type=str, default=None, help="Path(s) to the input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-sh", "--input-subpatterns", action="store_true", help="Use subpatterns (instead of main patterns) when loading input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-ip", "--input-pfms", nargs="+", type=str, default=None, help="Path(s) to the input motif PFM files, in PFM or MEME format.")
    parser.add_argument("-in", "--input-names", nargs="+", type=str, default=None, help="Nickname(s) of the input file(s): TF-MoDISco (lite) H5, PFMs, MEME.")
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, per model: CSV, TSV format.")

    # Output
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")

    # Upstream pipeline directory and reference databases: set these to your own local paths
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
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ------------------------------------------------------------
def set_default_configs():
    global TFMoDIScoCols, MetadataCols, ReferenceMatchCols, MotifMatchArgs
    global FilterArgs, MotifFilterArgs, ClusterArgs, MotifCompendiumArgs, JasparPipelineArgs

    TFMoDIScoCols = configs.TFMoDIScoCols()
    MetadataCols = configs.MetadataCols()
    ReferenceMatchCols = configs.ReferenceMatchCols()
    MotifMatchArgs = configs.MotifMatchArgs()
    FilterArgs = configs.FilterArgs()
    MotifFilterArgs = configs.MotifFilterArgs()
    ClusterArgs = configs.ClusterArgs()
    MotifCompendiumArgs = configs.MotifCompendiumArgs()
    JasparPipelineArgs = configs.JasparPipelineArgs()


#### FUNCTIONS -------------------------------------------------------------
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


def export_compendium_meme(
    mc: MotifCompendium,
    name_col: str,
    save_loc: str,
) -> None:
    """Export only finite, non-empty motifs in MEME format.

    ``utils_motif.trim_motif`` returns ``None`` for an all-zero motif. The
    MotifCompendium MEME exporter trims every motif before writing it, so pass
    it only motifs that can produce a valid MEME matrix. This is non-mutating:
    the complete MotifCompendium remains available in its ``.mc`` output.
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


def modify_hits_tsv(hits_tsv_in, hits_tsv_out, 
    modisco_h5=None,
    experiment_encid=None,
    target=None,
    biosample=None,
    bpnet_encid=None,
    head_type=None,
):
    hits_df = pd.read_csv(hits_tsv_in, sep="\t")

    # Extract motif name: Old, New
    motif_name_old = sorted(hits_df['motif_name'].unique())
    motif_name_new = []
    
    # From Modisco
    if modisco_h5:
        with h5py.File(modisco_h5, 'r') as f:
            for metapattern in ['pos_patterns', 'neg_patterns']:
                if metapattern in f:
                    motif_name_new.extend(f[metapattern].keys())
        motif_name_new = sorted(motif_name_new)

        # Extract unique identifier
        motif_name_format = "_".join(motif_name_new[0].split(".")[-1].split("_")[:-2])
    
    else:
        motif_name_format = f"{experiment_encid}_ChIP_{target}_{biosample}_{bpnet_encid}_{head_type}"

    # Create new names
    motif_name_new = []
    for old_name in motif_name_old:
        new_name = old_name.split(".")
        new_name = new_name[0] + "." + motif_name_format + "_" + new_name[-1]
        motif_name_new.append(new_name)

    # Map old motif names to new motif names
    motif_name_map = dict(zip(motif_name_old, motif_name_new))

    # Update hits_df with new motif names
    hits_df['motif_name'] = hits_df['motif_name'].map(motif_name_map)

    # Write hits tsv
    hits_df.to_csv(hits_tsv_out, sep="\t", index=False)

def rename_modisco_h5(modisco_h5_path: str, 
                      exp_encid: str, 
                      exp_type: str,
                      target: str,
                      biosample_name: str,
                      bpnetmodel_encid: str, 
                      output_head: str,
    ) -> None:
    """Rename patterns and subpatterns in a modisco .h5 file.
    New name format: {experiment_ENCID}_{experiment_type}_{biosample_ENCID}
        _{biosample_common_name}_{bpnetmodel_ENCID}_{output_head}_{pattern_#}
    """
    # Open modisco h5 file in read-write mode
    with h5py.File(modisco_h5_path, 'r+') as f:
        for key1 in f.keys():
            assert key1 in ['pos_patterns', 'neg_patterns'], f"Unexpected key: {key1}"
            for key2 in f[key1].keys():
                # Old key format
                old_key2_formats = ['pattern_']
                old_key2_formats.append(f"{exp_encid}_"
                        f"{exp_type}_"
                        f"{target}_"
                        f"{biosample_name}_"
                        f"{bpnetmodel_encid}_"
                        f"{output_head}_")

                # Get new key name
                num_key2 = key2.split('_')[-1]
                new_key2 = (f"{exp_encid}_"
                f"{exp_type}_"
                f"{target}_"
                f"{biosample_name}_"
                f"{bpnetmodel_encid}_"
                f"{output_head}_pattern_{num_key2}")

                # Rename key
                if any(key2.startswith(fmt) for fmt in old_key2_formats):
                    f[key1].move(key2, new_key2)
                else:
                    logging.info(f"Skipping key: {key2}")

def extract_hits(report_tsv, encid, head_type):
    report_df = pd.read_csv(report_tsv, sep="\t")
    hits_df = report_df[['motif_name', 'num_hits_total']].copy()
    if not all(hits_df['motif_name'].str.contains(encid)):
        hits_df['motif_name'] = hits_df['motif_name'].str.replace(".", f".{encid}_{head_type}_")
    return hits_df

def extract_motifs(motif_npy):
    motifs = np.load(motif_npy, allow_pickle=True)
    motifs = np.nan_to_num(motifs, nan=0.0)
    motifs = np.transpose(motifs, (0, 2, 1))
    return motifs


def cuda_device_available() -> tuple[bool, str | None]:
    """Return whether CuPy can access at least one CUDA device.

    ``--use-gpu`` is also used by the batch launcher on CPU-only nodes.  CuPy
    imports successfully on those nodes, but raises ``cudaErrorNoDevice`` only
    when MotifCompendium starts the first similarity calculation.  Check it up
    front so the pipeline can safely fall back to its CPU implementation.
    """
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return True, None
        return False, "CuPy reported zero CUDA devices"
    except Exception as error:
        return False, str(error)

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

    # Check output
    if args.verbose:
        logging.info("Checking output directory...")
        logging.info(f"Output directory: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        logging.info(f"Creating output directory: {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)


    ### PATHS --------------------------------------------------------------------
    ## MotifCompendium object
    mc_dir = os.path.join(args.output_dir, "mc")

    mc_init_path = os.path.join(mc_dir,"motifcompendium_motif_init.mc")
    mc_path = os.path.join(mc_dir,"motifcompendium_motif_all.mc")
    mc_ppm_path = os.path.join(mc_dir, "motifcompendium_motif_all_ppm.mc")
    mc_pfm_path = os.path.join(mc_dir, "motifcompendium_motif_all_pfm.mc")
    
    mc_invitro_path = os.path.join(mc_dir,"motifcompendium_motif_invitro.mc")
    mc_invitro_ppm_path = os.path.join(mc_dir,"motifcompendium_motif_invitro_ppm.mc")
    mc_invitro_pfm_path = os.path.join(mc_dir,"motifcompendium_motif_invitro_pfm.mc")

    mc_noninvitro_path = os.path.join(mc_dir,"motifcompendium_motif_noninvitro.mc")
    mc_noninvitro_ppm_path = os.path.join(mc_dir,"motifcompendium_motif_noninvitro_ppm.mc")
    mc_noninvitro_pfm_path = os.path.join(mc_dir,"motifcompendium_motif_noninvitro_pfm.mc")

    mc_remove_path = os.path.join(mc_dir,"motifcompendium_motif_removed.mc")
    mc_remove_ppm_path = os.path.join(mc_dir,"motifcompendium_motif_removed_ppm.mc")
    mc_remove_pfm_path = os.path.join(mc_dir,"motifcompendium_motif_removed_pfm.mc")

    # Clusters
    mc_cluster_path = os.path.join(mc_dir, "motifcompendium_cluster_all.mc")
    mc_cluster_ppm_path = os.path.join(mc_dir, "motifcompendium_cluster_all_ppm.mc")
    mc_cluster_pfm_path = os.path.join(mc_dir, "motifcompendium_cluster_all_pfm.mc")

    mc_cluster_invitro_path = os.path.join(mc_dir,"motifcompendium_cluster_invitro.mc")
    mc_cluster_invitro_ppm_path = os.path.join(mc_dir,"motifcompendium_cluster_invitro_ppm.mc")
    mc_cluster_invitro_pfm_path = os.path.join(mc_dir,"motifcompendium_cluster_invitro_pfm.mc")

    mc_cluster_noninvitro_path = os.path.join(mc_dir,"motifcompendium_cluster_noninvitro.mc")
    mc_cluster_noninvitro_ppm_path = os.path.join(mc_dir,"motifcompendium_cluster_noninvitro_ppm.mc")
    mc_cluster_noninvitro_pfm_path = os.path.join(mc_dir,"motifcompendium_cluster_noninvitro_pfm.mc")

    # Metaclusters
    mc_metacluster_invitro_path = os.path.join(mc_dir,"motifcompendium_metacluster_invitro.mc")
    mc_metacluster_invitro_ppm_path = os.path.join(mc_dir,"motifcompendium_metacluster_invitro_ppm.mc")
    mc_metacluster_invitro_pfm_path = os.path.join(mc_dir,"motifcompendium_metacluster_invitro_pfm.mc")


    ## HTML summary table
    html_dir = os.path.join(args.output_dir, "html")
    
    html_cluster_path = os.path.join(html_dir, "motifcompendium_cluster_all.html")
    html_cluster_invitro_path = os.path.join(html_dir, "motifcompendium_cluster_invitro.html")
    html_cluster_noninvitro_path = os.path.join(html_dir, "motifcompendium_cluster_noninvitro.html")

    html_metacluster_invitro_path = os.path.join(html_dir, "motifcompendium_metacluster_invitro.html")


    ## Export: Modisco H5
    export_dir = os.path.join(args.output_dir, "export")

    # Motifs
    modisco_motif_path = os.path.join(export_dir, "motifcompendium_motif_all.h5")
    modisco_motif_ppm_path = os.path.join(export_dir, "motifcompendium_motif_all_ppm.h5")
    modisco_motif_pfm_path = os.path.join(export_dir, "motifcompendium_motif_all_pfm.h5")

    modisco_motif_invitro_path = os.path.join(export_dir, "motifcompendium_motif_invitro.h5")
    modisco_motif_invitro_ppm_path = os.path.join(export_dir, "motifcompendium_motif_invitro_ppm.h5")
    modisco_motif_invitro_pfm_path = os.path.join(export_dir, "motifcompendium_motif_invitro_pfm.h5")

    modisco_motif_noninvitro_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro.h5")
    modisco_motif_noninvitro_ppm_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro_ppm.h5")
    modisco_motif_noninvitro_pfm_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro_pfm.h5")

    modisco_motif_removed_path = os.path.join(export_dir, "motifcompendium_motif_removed.h5")
    modisco_motif_removed_ppm_path = os.path.join(export_dir, "motifcompendium_motif_removed_ppm.h5")
    modisco_motif_removed_pfm_path = os.path.join(export_dir, "motifcompendium_motif_removed_pfm.h5")

    # Clusters
    modisco_cluster_path = os.path.join(export_dir, "motifcompendium_cluster_all.h5")
    modisco_cluster_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_ppm_all.h5")
    modisco_cluster_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_pfm_all.h5")

    modisco_cluster_invitro_path = os.path.join(export_dir, "motifcompendium_cluster_invitro.h5")
    modisco_cluster_invitro_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_invitro_ppm.h5")
    modisco_cluster_invitro_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_invitro_pfm.h5")

    modisco_cluster_noninvitro_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro.h5")
    modisco_cluster_noninvitro_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro_ppm.h5")
    modisco_cluster_noninvitro_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro_pfm.h5")

    # Metaclusters
    modisco_metacluster_invitro_path = os.path.join(export_dir, "motifcompendium_metacluster_invitro.h5")
    modisco_metacluster_invitro_ppm_path = os.path.join(export_dir, "motifcompendium_metacluster_invitro_ppm.h5")
    modisco_metacluster_invitro_pfm_path = os.path.join(export_dir, "motifcompendium_metacluster_invitro_pfm.h5")


    ## Export: MEME
    # Motifs
    meme_motif_path = os.path.join(export_dir, "motifcompendium_motif_all.meme.txt")
    meme_motif_ppm_path = os.path.join(export_dir, "motifcompendium_motif_all_ppm.meme.txt")
    meme_motif_pfm_path = os.path.join(export_dir, "motifcompendium_motif_all_pfm.meme.txt")

    meme_motif_invitro_path = os.path.join(export_dir, "motifcompendium_motif_invitro.meme.txt")
    meme_motif_invitro_ppm_path = os.path.join(export_dir, "motifcompendium_motif_invitro_ppm.meme.txt")
    meme_motif_invitro_pfm_path = os.path.join(export_dir, "motifcompendium_motif_invitro_pfm.meme.txt")

    meme_motif_noninvitro_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro.meme.txt")
    meme_motif_noninvitro_ppm_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro_ppm.meme.txt")
    meme_motif_noninvitro_pfm_path = os.path.join(export_dir, "motifcompendium_motif_noninvitro_pfm.meme.txt")

    meme_motif_removed_path = os.path.join(export_dir, "motifcompendium_motif_removed.meme.txt")
    meme_motif_removed_ppm_path = os.path.join(export_dir, "motifcompendium_motif_removed_ppm.meme.txt")
    meme_motif_removed_pfm_path = os.path.join(export_dir, "motifcompendium_motif_removed_pfm.meme.txt")

    # Clusters
    meme_cluster_path = os.path.join(export_dir, "motifcompendium_cluster_all.meme.txt")
    meme_cluster_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_all_ppm.meme.txt")
    meme_cluster_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_all_pfm.meme.txt")

    meme_cluster_invitro_path = os.path.join(export_dir, "motifcompendium_cluster_invitro.meme.txt")
    meme_cluster_invitro_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_invitro_ppm.meme.txt")
    meme_cluster_invitro_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_invitro_pfm.meme.txt")

    meme_cluster_noninvitro_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro.meme.txt")
    meme_cluster_noninvitro_ppm_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro_ppm.meme.txt")
    meme_cluster_noninvitro_pfm_path = os.path.join(export_dir, "motifcompendium_cluster_noninvitro_pfm.meme.txt")

    # Metaclusters
    meme_metacluster_invitro_path = os.path.join(export_dir, "motifcompendium_metacluster_invitro.meme.txt")


    ## Export: Metadata
    metadata_dir = os.path.join(args.output_dir, "metadata")

    metadata_motif_path = os.path.join(metadata_dir, "motifcompendium_motif_all_metadata.tsv")
    metadata_motif_invitro_path = os.path.join(metadata_dir, "motifcompendium_motif_invitro_metadata.tsv")
    metadata_motif_noninvitro_path = os.path.join(metadata_dir, "motifcompendium_motif_noninvitro_metadata.tsv")
    metadata_motif_removed_path = os.path.join(metadata_dir, "motifcompendium_motif_removed_metadata.tsv")

    metadata_cluster_path = os.path.join(metadata_dir, "motifcompendium_cluster_all_metadata.tsv")
    metadata_cluster_invitro_path = os.path.join(metadata_dir, "motifcompendium_cluster_invitro_metadata.tsv")
    metadata_cluster_noninvitro_path = os.path.join(metadata_dir, "motifcompendium_cluster_noninvitro_metadata.tsv")

    metadata_metacluster_invitro_path = os.path.join(metadata_dir, "motifcompendium_metacluster_invitro_metadata.tsv")


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


    #### METADATA -------------------------------------------------------
    logging.info("Processing metadata...")

    # Update motif names, to match TF-MoDISco
    # Taken from `name`, which already carries the original TF-MoDISco pattern name after
    # the first ".", rather than re-assembled from the individual metadata fields. The
    # metadata `biosample` is not always the string TF-MoDISco/FiNeMo recorded (e.g.
    # "SK-N-SH" is stored as "SH"), so re-assembling produced keys that silently missed
    # the FiNeMo hits report and zeroed `num_hits` for those experiments, which then
    # dropped them from the candidate pool.
    mc.metadata[MetadataCols.name_tfmodisco_col] = (
        mc[MetadataCols.posneg_col]
        + "_" + "patterns."
        + mc[MetadataCols.name_col].str.split(".", n=1).str[1]
    )

    # Calculate metrics
    mc[MetadataCols.avg_contrib_col] = mc.motifs.sum(axis=(1,2))

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


    #### Hits -------------------------------------------------------------------------
    logging.info("Extracting hits and PPMs from Finemo hits metadata...")

    # Extract metadata, PPMs
    model_metadata = mc.metadata.drop_duplicates(subset=[MetadataCols.model_col])

    hits_dfs = []
    ppms = []
    for _, row in model_metadata.iterrows():
        encid = row[MetadataCols.id_col]
        head_type = row[MetadataCols.head_col]
        motif_npy = row[MetadataCols.hits_ppm_npy_col]
        report_tsv = row[MetadataCols.hits_report_tsv_col]

        hits_dfs.append(extract_hits(report_tsv, encid, head_type))
        ppms.append(extract_motifs(motif_npy))

    # Combine metadata
    hits_df = pd.concat(hits_dfs, axis=0, ignore_index=True)

    # Combine ppms
    max_len = max(ppm.shape[1] for ppm in ppms)
    for i, ppm in enumerate(ppms):
        if ppm.shape[1] < max_len:
            ppm_padded = np.zeros((ppm.shape[0], max_len, ppm.shape[2]))
            ppm_padded[:ppm.shape[0], :ppm.shape[1], :ppm.shape[2]] = ppm
            ppms[i] = ppm_padded
    ppms = np.concatenate(ppms, axis=0)

    # Update metadata: Add num_hits
    mc.metadata[MetadataCols.num_hits_col] = mc.metadata[MetadataCols.name_tfmodisco_col].map(
        hits_df.set_index("motif_name")["num_hits_total"]
    ).fillna(0).astype(int)

    # Save MotifCompendium
    logging.info("Saving initial MotifCompendium object...")
    os.makedirs(os.path.dirname(mc_path), exist_ok=True)
    mc.save(mc_init_path)

    # Clean up metadata: Drop all '_path' or '_paths' or '_file' or '_files' columns
    mc.metadata.drop(columns=[col for col in mc.metadata.columns if col.endswith(('_path', '_paths', '_file', '_files'))], inplace=True)


    #### REFERENCE --------------------------------------------------------------------
    logging.info("Loading reference motif databases...")
    logging.info(f"  JASPAR CORE: {args.jaspar_core}")
    logging.info(f"  JASPAR UNVALID: {args.jaspar_unvalid}")
    logging.info(f"  Reference: {args.ref}")
    # Load Ref MotifCompendium
    if args.jaspar_core.endswith(".mc"):
        mc_jaspar_core = MotifCompendium.load(args.jaspar_core, safe=args.safe)
    else:
        mc_jaspar_core = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.jaspar_core))[0]: args.jaspar_core},
            safe=args.safe,
        )
    if args.jaspar_unvalid.endswith(".mc"):
        mc_jaspar_unvalid = MotifCompendium.load(args.jaspar_unvalid, safe=args.safe)
    else:
        mc_jaspar_unvalid = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.jaspar_unvalid))[0]: args.jaspar_unvalid},
            safe=args.safe,
        )
    if args.ref.endswith(".mc"):
        mc_ref = MotifCompendium.load(args.ref, safe=args.safe)
    else:
        mc_ref = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref))[0]: args.ref},
            safe=args.safe,
        )
    if args.ref_invitro.endswith(".mc"):
        mc_ref_invitro = MotifCompendium.load(args.ref_invitro, safe=args.safe)
    else:
        mc_ref_invitro = MotifCompendium.build_from_pfm(
            pfm_dict={os.path.splitext(os.path.basename(args.ref_invitro))[0]: args.ref_invitro},
            safe=args.safe,
        )

    # Reference: Exclude composites
    # Only the in vitro reference drops composites. The JASPAR CORE / UNVALIDATED
    # references keep theirs: they are matched per target by an exact `TF` equality, so a
    # composite entry can only ever be picked up by the composite target it belongs to.
    mc_ref = mc_ref[~mc_ref[MetadataCols.name_col].str.contains("::")]
    mc_ref_invitro = mc_ref_invitro[~mc_ref_invitro[MetadataCols.name_col].str.contains("::")]

    # Combine: JASPAR CORE + Unvalidated
    mc_jaspar = MotifCompendium.combine(
        [mc_jaspar_core, mc_jaspar_unvalid],
        ['JASPAR_CORE', 'JASPAR_UNVALID'],
        safe=args.safe,
    )

    # ## Select JASPAR models only
    # Select ENCIDs with known TF JASPAR motif
    if args.verbose:
        logging.info("Selecting motifs with known JASPAR motif...")
    mc_metadata_updates = []
    missing_TF = []
    missing_family = []
    missing_encids = []

    for i, target in enumerate(mc[MetadataCols.target_col].unique()):
        logging.info(f"Processing {i+1}: {target}")

        # MC: Slice by target
        mc_i = mc[mc[MetadataCols.target_col].str.upper() == target.upper()]
        target_family = mc_i[MetadataCols.family_col].unique()[0]
        encid = mc_i[MetadataCols.id_col].unique().tolist()
        
        # Ref: Slice by target
        logging.info(f"Slicing Ref by target: {target} ({target_family})")
        mc_jaspar_i = mc_jaspar[mc_jaspar['TF'] == target]

        # Assign labels
        if len(mc_jaspar_i) > 0:
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_i,
                assign_from_mc=mc_jaspar_i,
                save_col_prefix=ReferenceMatchCols.ref_jaspar_select_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                save_images=False,
                logo_trimming=0,
            )

            ## Update: Metadata, Images
            mc_metadata_updates.append(mc_i.metadata.set_index(MetadataCols.name_col))

        else:
            logging.info(f"No JASPAR motif for target: {target}")
            missing_TF.append(target)
            missing_family.append(target_family)
            missing_encids.extend(encid)

    logging.info(f"TFs without JASPAR motif: {missing_TF}")
    logging.info(f"TF families without JASPAR motifs: {set(missing_family)}")
    logging.info(f"ENCID without JASPAR motif: {set(missing_encids)}")

    # Apply updates: Metadata, Images
    mc_metadata_updates_df = pd.concat(mc_metadata_updates)
    mc.metadata.set_index(MetadataCols.name_col, inplace=True)
    mc_metadata_row_order = mc.metadata.index  # combine_first sorts the index: restore the original row order
    mc.metadata = mc.metadata.combine_first(mc_metadata_updates_df)
    mc.metadata.update(mc_metadata_updates_df)
    mc.metadata = mc.metadata.reindex(mc_metadata_row_order)
    mc.metadata.reset_index(inplace=True)
    del mc_metadata_updates_df, mc_metadata_updates, mc_metadata_row_order

    # Filter: Flag motifs (min_seqlets, min_hits, no JASPAR match), for bookkeeping / reporting
    apply_filter_initial(
        mc=mc,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # In vitro candidate pool: min_seqlets and min_hits ONLY.
    # NOTE: this is deliberately NOT gated on the per-motif `jasp_match` hard filter.
    # A JASPAR match selects which *experiments* are kept; every motif belonging to a
    # kept experiment stays an in vitro candidate, even if that individual motif does
    # not itself resemble a JASPAR model.
    mc_filtered = mc[
        (mc[MetadataCols.num_seqlets_col] > JasparPipelineArgs.min_seqlets)
        & (mc[MetadataCols.num_hits_col] > JasparPipelineArgs.min_hits)
    ]
    mc_removed = mc[~mc[MetadataCols.name_col].isin(mc_filtered[MetadataCols.name_col])]


    # ## In vitro: Motifs
    # Match metadata columns
    metadata_cols = mc_ref_invitro.columns()
    for ref_mc in [mc_jaspar, mc_ref_invitro]:
        for metadata_col in metadata_cols:
            if metadata_col not in ref_mc.columns():
                metadata_cols.remove(metadata_col)

    for ref_mc in [mc_jaspar, mc_ref_invitro]:
        ref_mc.metadata = ref_mc.metadata[metadata_cols]

    # Use reference: In vitro + JASPAR (CORE+Unvalidated)
    mc_ref_invitro_jaspar = MotifCompendium.combine(
        [mc_jaspar, mc_ref_invitro],
        ['JASP', 'Reference-invitro'],
        safe=args.safe,
    )

    # Find motifs that match with its target/surrogate in vitro motif
    # Save MotifCompendium
    if args.verbose:
        logging.info("Identifying motifs that match with known JASPAR & in vitro motifs...")

    mc_filtered_metadata_updates = []
    mc_invitro_is = []
    missing_TF = []
    missing_family = []

    for i, target in enumerate(mc_filtered.metadata[MetadataCols.target_col].unique()):
        logging.info(f"Processing {i+1}: {target}")

        # MC: Slice by target
        mc_filtered_i = mc_filtered[mc_filtered[MetadataCols.target_col].str.upper() == target.upper()]
        target_family = mc_filtered_i.metadata[MetadataCols.family_col].unique()[0]
        
        # Ref: Slice by target, family
        logging.info(f"Slicing Ref by target: {target}")
        mc_ref_i = mc_ref_invitro_jaspar[mc_ref_invitro_jaspar['TF'] == target]

        # Assign labels
        try:
            if len(mc_ref_i) > 0:
                utils_analysis.assign_label_from_other_compendium(
                    assign_to_mc=mc_filtered_i,
                    assign_from_mc=mc_ref_i,
                    save_col_prefix=ReferenceMatchCols.ref_invitro_col,
                    min_score=MotifMatchArgs.min_composite_score,
                    max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                    save_images=False,
                    logo_trimming=0,
                )

                ## Update: Metadata, Images
                mc_filtered_metadata_updates.append(mc_filtered_i.metadata.set_index(MetadataCols.name_col))

                # Keep the motifs of this target that match the target's own reference.
                # Collected per target (rather than filtered once at the end) so that the
                # scores are the ones computed against this target's reference slice.
                mc_invitro_is.append(mc_filtered_i[
                    mc_filtered_i[f'{ReferenceMatchCols.ref_invitro_col}_score0'] > MotifMatchArgs.min_single_score_jasp
                ])

        except Exception as e:
            logging.info(f"Error processing {target} in {target_family}: {e}")
            missing_TF.append(target)
            missing_family.append(target_family)
            continue

    logging.info(f"TFs with missing known invitro motif: {missing_TF}")
    logging.info(f"TF families with missing known invitro motifs: {set(missing_family)}")

    # Apply updates: Metadata, Images
    mc_filtered_metadata_updates_df = pd.concat(mc_filtered_metadata_updates)
    mc_filtered.metadata.set_index(MetadataCols.name_col, inplace=True)
    mc_filtered_row_order = mc_filtered.metadata.index  # combine_first sorts the index: restore the original row order
    mc_filtered.metadata = mc_filtered.metadata.combine_first(mc_filtered_metadata_updates_df)
    mc_filtered.metadata.update(mc_filtered_metadata_updates_df)
    mc_filtered.metadata = mc_filtered.metadata.reindex(mc_filtered_row_order)
    mc_filtered.metadata.reset_index(inplace=True)
    del mc_filtered_metadata_updates_df, mc_filtered_metadata_updates, mc_filtered_row_order


    # FILTER ---
    if args.verbose:
            logging.info("Annotating motifs with reference database matches...")

    ## [General] Label motifs
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_filtered,
        assign_from_mc=mc_ref,
        save_col_prefix=ReferenceMatchCols.ref_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs,
        label_unsigned=True,
        save_images=False,
        logo_trimming=0,
    )

    # Apply filters
    if args.verbose:
            logging.info("Applying motif filters...")

    # Re-evaluate the flag on the candidate pool: only the motif-quality filters applied
    # below decide whether a candidate becomes non-in-vitro. The min_seqlets / min_hits /
    # jasp_match flags raised by apply_filter_initial() already did their job when the pool
    # was selected, and carrying them forward here would flag every motif whose own CWM
    # does not resemble a JASPAR model, which empties out mc_noninvitro entirely.
    mc_filtered[FilterArgs.filter_flag_col] = False
    mc_filtered[FilterArgs.filter_flag_type_col] = None

    apply_filter_motifs(
        mc=mc_filtered,
        motif_filter_args=MotifFilterArgs,
        filter_flag_col=FilterArgs.filter_flag_col,
        filter_flag_type_col=FilterArgs.filter_flag_type_col,
    )

    # Invitro, Non-invitro, Removed
    # Combine the per-target in vitro matches collected above. Targets are not contiguous
    # in `mc_filtered`, so concatenating the per-target slices is NOT the same row order as
    # filtering `mc_filtered` once; the per-target order is the one used downstream.
    mc_invitro = MotifCompendium.combine(mc_invitro_is, safe=args.safe)
    del mc_invitro_is
    # Non-in vitro: every candidate that is not in vitro. The candidate pool already
    # encodes the min_seqlets / min_hits criteria, so nothing further is excluded here.
    # In particular the motif-quality flag is NOT applied: the pool now contains the
    # low-quality motifs that the JASPAR gate used to hide, and flagging them here
    # removes every remaining candidate, leaving mc_noninvitro empty (which then fails
    # in cluster_averages with "need at least one array to stack").
    mc_noninvitro = mc_filtered[
        ~mc_filtered[MetadataCols.name_col].isin(mc_invitro[MetadataCols.name_col])
    ]
    mc_removed = mc[~(
        (mc[MetadataCols.name_col].isin(mc_invitro[MetadataCols.name_col])) | \
        (mc[MetadataCols.name_col].isin(mc_noninvitro[MetadataCols.name_col]))
    )]

    logging.info(
        f"Motif counts: total={len(mc)}, candidate pool={len(mc_filtered)}, "
        f"in vitro={len(mc_invitro)}, non-in vitro={len(mc_noninvitro)}, removed={len(mc_removed)}"
    )


    ## CLUSTERING ---
    logging.info("Clustering motifs...")

    mc_invitro.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        cluster_within=ClusterArgs.cluster_within,
        save_name=MetadataCols.cluster_col,
    )

    mc_noninvitro.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        cluster_within=ClusterArgs.cluster_within,
        save_name=MetadataCols.cluster_col,
    )

    min_len_invitro = mc_invitro[MetadataCols.cluster_col].nunique()
    min_len_noninvitro = mc_noninvitro[MetadataCols.cluster_col].nunique()
    if args.verbose:
        logging.info(f"Unique clusters (In vitro): {min_len_invitro}")
        logging.info(f"Unique clusters (Non-in vitro): {min_len_noninvitro}")

    # Recursively cluster:
    for i in range(ClusterArgs.cluster_max_iter):
        logging.info(f"Recursively clustering: {i}")
        mc_invitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within_on=(ClusterArgs.cluster_within, MetadataCols.cluster_col),
            save_name=MetadataCols.cluster_col,
        )
        
        new_min_len_invitro = mc_invitro[MetadataCols.cluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique clusters (In vitro): {new_min_len_invitro}")
        if min_len_invitro == new_min_len_invitro:
            break
        else:
            min_len_invitro = new_min_len_invitro


    for i in range(ClusterArgs.cluster_max_iter):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        mc_noninvitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_within_on=(ClusterArgs.cluster_within, MetadataCols.cluster_col),
            save_name=MetadataCols.cluster_col,
        )
        
        new_min_len_noninvitro = mc_noninvitro[MetadataCols.cluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique clusters (Non-in vitro): {new_min_len_noninvitro}")
        if min_len_noninvitro == new_min_len_noninvitro:
            break
        else:
            min_len_noninvitro = new_min_len_noninvitro

    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        mc_invitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.cluster_within, MetadataCols.cluster_col),
            save_name=MetadataCols.cluster_col,
        )
        
        new_min_len_invitro = mc_invitro[MetadataCols.cluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique clusters (In vitro): {new_min_len_invitro}")
        if min_len_invitro == new_min_len_invitro:
            break
        else:
            min_len_invitro = new_min_len_invitro

    for i in range(ClusterArgs.cluster_max_iter):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        mc_noninvitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_within_on=(ClusterArgs.cluster_within, MetadataCols.cluster_col),
            save_name=MetadataCols.cluster_col,
        )

        new_min_len_noninvitro = mc_noninvitro[MetadataCols.cluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique clusters (Non-in vitro): {new_min_len_noninvitro}")
        if min_len_noninvitro == new_min_len_noninvitro:
            break
        else:
            min_len_noninvitro = new_min_len_noninvitro

    # AVERAGES ---
    logging.info("Building cluster averages...")
    mc_cluster_invitro = mc_invitro.cluster_averages(
        clustering=MetadataCols.cluster_col,
        aggregations=JasparPipelineArgs.aggregations,
        compute_quality_stats=False,
    )

    mc_cluster_noninvitro = mc_noninvitro.cluster_averages(
        clustering=MetadataCols.cluster_col,
        aggregations=JasparPipelineArgs.aggregations,
        compute_quality_stats=False,
    )

    # Sort by num_hits: fixes the order in which clusters of a target are numbered below
    mc_cluster_invitro.sort(by=MetadataCols.num_hits_col, ascending=False, inplace=True)

    # CLASSIFY ---
    logging.info("Identifying clusters that match with known JASPAR & in vitro motifs...")

    mc_cluster_invitro_metadata_updates = []
    mc_cluster_invitro_images_updated = pd.DataFrame(index=mc_cluster_invitro[MetadataCols.name_col])
    mc_cluster_invitro_images_existing = set(mc_cluster_invitro.images())

    for i, target in enumerate(mc_cluster_invitro.metadata[MetadataCols.target_col].unique()):
        logging.info(f"Processing {i+1}: {target}")
        
        # MC: Slice by TF
        mc_cluster_invitro_i = mc_cluster_invitro[mc_cluster_invitro[MetadataCols.target_col] == target]
        target_family = mc_cluster_invitro_i.metadata[MetadataCols.family_col].unique()[0]

        # In vitro: Slice by TF
        logging.info(f"Slicing Ref by target: {target} ({target_family})")
        mc_ref_i = mc_ref_invitro_jaspar[mc_ref_invitro_jaspar['TF'] == target]

        # JASPAR: Slice by TF (if none, family)
        mc_jaspar_core_i = mc_jaspar_core[mc_jaspar_core['TF'] == target]
        mc_jaspar_unvalid_i = mc_jaspar_unvalid[mc_jaspar_unvalid['TF'] == target]
        mc_jaspar_family_i = mc_jaspar[mc_jaspar[MetadataCols.family_col] == target_family]

        ## Assign labels: In vitro
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_cluster_invitro_i,
            assign_from_mc=mc_ref_i,
            save_col_prefix=ReferenceMatchCols.ref_invitro_col,
            min_score=MotifMatchArgs.min_composite_score,
            max_submotifs=MotifMatchArgs.max_submotifs_jasp,
            save_images=True,
            logo_trimming=0,
        )

        # JASPAR-CORE
        if len(mc_jaspar_core_i) > 0:
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_invitro_i,
                assign_from_mc=mc_jaspar_core_i,
                save_col_prefix=ReferenceMatchCols.ref_jaspar_core_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                save_images=True,
                logo_trimming=0,
            )

        # JASPAR-UNVALID
        if (len(mc_jaspar_unvalid_i) > 0) and (len(mc_cluster_invitro_i) > 0):
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_invitro_i,
                assign_from_mc=mc_jaspar_unvalid_i,
                save_col_prefix=ReferenceMatchCols.ref_jaspar_unvalid_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                save_images=True,
                logo_trimming=0,
            )

        # JASPAR-OTHER: If remaining
        if len(mc_cluster_invitro_i) > 0:
            utils_analysis.assign_label_from_other_compendium(
                assign_to_mc=mc_cluster_invitro_i,
                assign_from_mc=mc_jaspar_family_i,
                save_col_prefix=ReferenceMatchCols.ref_invitro_other_col,
                min_score=MotifMatchArgs.min_composite_score,
                max_submotifs=MotifMatchArgs.max_submotifs_jasp,
                save_images=True,
                logo_trimming=0,
            )

        ## Update: Metadata, Images
        mc_cluster_invitro_metadata_updates.append(mc_cluster_invitro_i.metadata.set_index(MetadataCols.name_col))

        new_image_cols = [image_col for image_col in mc_cluster_invitro_i.images() if image_col not in mc_cluster_invitro_images_existing]
        mc_cluster_invitro_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_cluster_invitro_i.get_images(image_col),
                index=mc_cluster_invitro_i[MetadataCols.name_col],
            )
            for image_col in new_image_cols
        })
        mc_cluster_invitro_images_updated = mc_cluster_invitro_images_updated.combine_first(mc_cluster_invitro_i_new_images)
        mc_cluster_invitro_images_updated.update(mc_cluster_invitro_i_new_images)


    # Apply updates: Metadata, Images
    mc_cluster_invitro_metadata_updates_df = pd.concat(mc_cluster_invitro_metadata_updates)
    mc_cluster_invitro.metadata.set_index(MetadataCols.name_col, inplace=True)
    mc_cluster_invitro_row_order = mc_cluster_invitro.metadata.index  # combine_first sorts the index: restore the original row order
    mc_cluster_invitro.metadata = mc_cluster_invitro.metadata.combine_first(mc_cluster_invitro_metadata_updates_df)
    mc_cluster_invitro.metadata.update(mc_cluster_invitro_metadata_updates_df)
    mc_cluster_invitro.metadata = mc_cluster_invitro.metadata.reindex(mc_cluster_invitro_row_order)
    mc_cluster_invitro.metadata.reset_index(inplace=True)
    del mc_cluster_invitro_metadata_updates_df, mc_cluster_invitro_metadata_updates, mc_cluster_invitro_row_order

    for image_col in mc_cluster_invitro_images_updated.columns:
        mc_cluster_invitro.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None
                for x in mc_cluster_invitro_images_updated[image_col]]
        )
    del mc_cluster_invitro_images_updated, mc_cluster_invitro_images_existing


    # ANNOTATE ---
    logging.info("Annotating clusters...")
    utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc_cluster_noninvitro,
            assign_from_mc=mc_ref_invitro,
            save_col_prefix=ReferenceMatchCols.ref_invitro_col,
            min_score=MotifMatchArgs.min_composite_score,
            max_submotifs=MotifMatchArgs.max_submotifs_jasp,
            save_images=True,
            logo_trimming=0,
        )
    
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_cluster_noninvitro,
        assign_from_mc=mc_jaspar,
        save_col_prefix=ReferenceMatchCols.ref_jaspar_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs_jasp,
        save_images=True,
        logo_trimming=0,
    )


    # Resolve each cluster to a single group: JASPAR CORE, JASPAR UNVALID, UNCONFIRMED
    def _match_score_mask(prefix):
        col = f'{prefix}_score0'
        if col not in mc_cluster_invitro.columns():
            return pd.Series(False, index=mc_cluster_invitro.metadata.index)
        return mc_cluster_invitro[col] > MotifMatchArgs.min_composite_score

    mc_cluster_invitro_jasparcore_mask = _match_score_mask(ReferenceMatchCols.ref_jaspar_core_col)
    mc_cluster_invitro_jasparunvalid_mask = _match_score_mask(ReferenceMatchCols.ref_jaspar_unvalid_col)

    # CORE wins over UNVALID; every cluster that matches neither falls through to
    # UNCONFIRMED. There is no score threshold on the UNCONFIRMED group: a cluster that
    # matched nothing is still an in vitro cluster and must keep a name and a group.
    group_suffix = pd.Series(np.select(
        [mc_cluster_invitro_jasparcore_mask, mc_cluster_invitro_jasparunvalid_mask],
        ["_JASPCORE_", "_JASPUNVALID_"],
        default="_UNCONFIRM_",
    ), index=mc_cluster_invitro.metadata.index)

    ## Rename: applied once across the whole in-vitro-matched compendium, before slicing into groups
    mc_cluster_invitro.metadata[MetadataCols.cluster_name_col] = mc_cluster_invitro.metadata[MetadataCols.name_col]
    mc_cluster_invitro.metadata[MetadataCols.counter_col] = mc_cluster_invitro.metadata.groupby([MetadataCols.target_col, group_suffix]).cumcount()
    mc_cluster_invitro.metadata[MetadataCols.name_col] = (
        mc_cluster_invitro.metadata[MetadataCols.target_col] + group_suffix + mc_cluster_invitro.metadata[MetadataCols.counter_col].astype(str)
    )
    mc_cluster_invitro.metadata.drop(columns=[MetadataCols.counter_col], axis=1, inplace=True)

    # Rename cluster: Non-invitro: TF_UNANNOT_{num}
    mc_cluster_noninvitro.metadata[MetadataCols.counter_col] = mc_cluster_noninvitro.metadata.groupby(MetadataCols.target_col).cumcount()
    mc_cluster_noninvitro.metadata[MetadataCols.name_new_col] = mc_cluster_noninvitro.metadata[MetadataCols.target_col] + MetadataCols.name_unannotated + mc_cluster_noninvitro.metadata[MetadataCols.counter_col].astype(str)
    mc_cluster_noninvitro.metadata[MetadataCols.cluster_name_col] = mc_cluster_noninvitro.metadata[MetadataCols.name_col]

    mc_cluster_noninvitro.metadata[MetadataCols.name_col] = mc_cluster_noninvitro.metadata[MetadataCols.name_new_col]
    mc_cluster_noninvitro.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    # Sort by name
    mc_cluster_invitro.sort(by=MetadataCols.name_col, ascending=True, inplace=True)
    mc_cluster_noninvitro.sort(by=MetadataCols.name_col, ascending=True, inplace=True)

    # Motifs: Map cluster name to individual motif MotifCompendium
    invitro_cluster_map = dict(zip(
        mc_cluster_invitro[MetadataCols.source_cluster_col],
        mc_cluster_invitro[MetadataCols.name_col],
    ))
    mc_invitro[MetadataCols.cluster_name_col] = mc_invitro[MetadataCols.cluster_col].map(invitro_cluster_map)

    noninvitro_cluster_map = dict(zip(mc_cluster_noninvitro[MetadataCols.source_cluster_col], mc_cluster_noninvitro[MetadataCols.name_col]))
    mc_noninvitro[MetadataCols.cluster_name_col] = mc_noninvitro[MetadataCols.cluster_col].map(noninvitro_cluster_map)

    # Sort by name, alphabetically
    mc_cluster_invitro.sort(by=MetadataCols.name_col, ascending=True, inplace=True)
    mc_cluster_noninvitro.sort(by=MetadataCols.name_col, ascending=True, inplace=True)

    # ## All: Clusters
    # Update: Metadata, Images
    mc_cluster_metadata_cols = set(mc_cluster_invitro.columns() + mc_cluster_noninvitro.columns())
    mc_cluster_image_cols = set(mc_cluster_invitro.images() + mc_cluster_noninvitro.images())
    for mc_cluster_i in [mc_cluster_invitro, mc_cluster_noninvitro]:
        for metadata_col in mc_cluster_metadata_cols:
            if metadata_col not in mc_cluster_i.columns():
                mc_cluster_i.metadata[metadata_col] = None
        for image_col in mc_cluster_image_cols:
            if image_col not in mc_cluster_i.images():
                mc_cluster_i.set_images(image_name=image_col, images=[None] * len(mc_cluster_i))

    # Update: Source cluster number
    mc_cluster_noninvitro[MetadataCols.source_cluster_col] = (
        mc_cluster_invitro[MetadataCols.source_cluster_col].max()
        + mc_cluster_noninvitro[MetadataCols.source_cluster_col]
        + 1
    )

    # Combine
    mc_cluster = MotifCompendium.combine(
        [mc_cluster_invitro, mc_cluster_noninvitro],
        safe=args.safe,
    )


    #### METACLUSTER -------------------------------------------------------
    # CLUSTER ---
    logging.info("Clustering clusters...")

    mc_cluster_invitro.cluster(
        algorithm=ClusterArgs.cluster_algorithm_main,
        similarity_threshold=ClusterArgs.cluster_sim_threshold,
        save_name=MetadataCols.metacluster_col,
    )
    min_len = mc_cluster_invitro[MetadataCols.metacluster_col].nunique()
    if args.verbose:
        logging.info(f"Unique metaclusters: {min_len}")

    # Recursively cluster
    for i in range(ClusterArgs.cluster_max_iter):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        mc_cluster_invitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_main,
            similarity_threshold=ClusterArgs.cluster_sim_threshold,
            cluster_on=MetadataCols.metacluster_col,
            save_name=MetadataCols.metacluster_col,
        )
        
        new_min_len = mc_cluster_invitro[MetadataCols.metacluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique metaclusters: {new_min_len}")
        if min_len == new_min_len:
            break
        else:
            min_len = new_min_len

    # Force-cluster
    for i in range(ClusterArgs.cluster_max_iter):
        if args.verbose:
            logging.info(f"Recursively clustering: {i}")
        mc_cluster_invitro.cluster(
            algorithm=ClusterArgs.cluster_algorithm_force,
            similarity_threshold=ClusterArgs.cluster_sim_threshold_force,
            cluster_on=MetadataCols.metacluster_col,
            save_name=MetadataCols.metacluster_col,
        )
        
        new_min_len = mc_cluster_invitro[MetadataCols.metacluster_col].nunique()
        if args.verbose:
            logging.info(f"Unique metaclusters: {new_min_len}")
        if min_len == new_min_len:
            break
        else:
            min_len = new_min_len

    # AVERAGE ---
    logging.info("Building metacluster averages...")
    mc_metacluster_invitro = mc_cluster_invitro.cluster_averages(
        clustering=MetadataCols.metacluster_col,
        aggregations=JasparPipelineArgs.aggregations,
        compute_quality_stats=False,
    )
    
    # ANNOTATE ---
    logging.info("Annotating metaclusters...")
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_metacluster_invitro,
        assign_from_mc=mc_jaspar,
        save_col_prefix=ReferenceMatchCols.ref_jaspar_col,
        min_score=MotifMatchArgs.min_composite_score,
        max_submotifs=MotifMatchArgs.max_submotifs_jasp,
        save_images=True,
        logo_trimming=0,
    )

    # Rename metacluster: new_name, cluster_name = name
    mc_metacluster_invitro.metadata[MetadataCols.counter_col] = mc_metacluster_invitro.metadata.groupby(MetadataCols.target_col).cumcount()
    mc_metacluster_invitro.metadata[MetadataCols.name_new_col] = mc_metacluster_invitro.metadata[MetadataCols.target_col] + "_" + mc_metacluster_invitro.metadata[MetadataCols.counter_col].astype(str)
    mc_metacluster_invitro.metadata[MetadataCols.metacluster_name_col] = mc_metacluster_invitro.metadata[MetadataCols.name_col]

    mc_metacluster_invitro.metadata[MetadataCols.name_col] = mc_metacluster_invitro.metadata[MetadataCols.name_new_col]
    mc_metacluster_invitro.metadata.drop(columns=[MetadataCols.name_new_col, MetadataCols.counter_col], axis=1, inplace=True)

    # Map cluster name to individual motif MotifCompendium
    invitro_metacluster_map = dict(zip(mc_metacluster_invitro[MetadataCols.source_cluster_col], mc_metacluster_invitro[MetadataCols.name_col]))
    mc_cluster_invitro[MetadataCols.cluster_name_col] = mc_cluster_invitro[MetadataCols.metacluster_col].map(invitro_metacluster_map)

    # Sort by num_seqlets, pos/neg
    mc_metacluster_invitro.metadata['posneg'] = utils_motif.motif_posneg_sum(mc_metacluster_invitro.get_standard_motif_stack())
    mc_metacluster_invitro.sort(by=["posneg","num_motifs"], ascending=False, inplace=True) 


    #### VERSION: FINAL -------------------------------------------------------------------------
    ## CORE: mc, mc_invitro; mc_cluster_invitro
    # Clean up
    mc.metadata.drop(columns=JasparPipelineArgs.drop_cols, errors="ignore", inplace=True)
    for drop_image_i in JasparPipelineArgs.drop_images:
        if drop_image_i in mc.images():
            mc.delete_images(drop_image_i)
    mc_cluster.metadata.drop(columns=JasparPipelineArgs.drop_cols, errors="ignore", inplace=True)
    for drop_image_i in JasparPipelineArgs.drop_images:
        if drop_image_i in mc_cluster.images():
            mc_cluster.delete_images(drop_image_i)
    mc_metacluster_invitro.metadata.drop(columns=JasparPipelineArgs.drop_cols, errors="ignore", inplace=True)
    for drop_image_i in JasparPipelineArgs.drop_images:
        if drop_image_i in mc_metacluster_invitro.images():
            mc_metacluster_invitro.delete_images(drop_image_i)

    # ## Motif: All
    # Flag for removal
    mc[FilterArgs.filter_flag_col] = (mc[MetadataCols.name_col].isin(mc_removed[MetadataCols.name_col]) & \
        ~mc[MetadataCols.name_col].isin(mc_invitro[MetadataCols.name_col]) & \
        ~mc[MetadataCols.name_col].isin(mc_noninvitro[MetadataCols.name_col]))

    # Add to original cluster numbering
    mc_noninvitro.metadata[MetadataCols.cluster_col] = (
        mc_noninvitro.metadata[MetadataCols.cluster_col] 
        + mc_invitro.metadata[MetadataCols.cluster_col].max() 
        + 1
    )

    # Map clustering number to main MotifCompendium
    mc.metadata[MetadataCols.cluster_col] = mc.metadata[MetadataCols.name_col].map(
        dict(zip(mc_invitro.metadata[MetadataCols.name_col], mc_invitro.metadata[MetadataCols.cluster_col])) |
        dict(zip(mc_noninvitro.metadata[MetadataCols.name_col], mc_noninvitro.metadata[MetadataCols.cluster_col]))
    )

    # Map clustering name to main MotifCompendium
    mc.metadata[MetadataCols.cluster_name_col] = mc.metadata[MetadataCols.name_col].map(
        dict(zip(mc_invitro.metadata[MetadataCols.name_col], mc_invitro.metadata[MetadataCols.cluster_name_col])) |
        dict(zip(mc_noninvitro.metadata[MetadataCols.name_col], mc_noninvitro.metadata[MetadataCols.cluster_name_col]))
    )

    ## Re-slice
    mc_filtered = mc[~mc[FilterArgs.filter_flag_col]]
    mc_removed = mc[mc[FilterArgs.filter_flag_col]]
    # mc_invitro = mc_filtered[mc_filtered[f'{ReferenceMatchCols.ref_invitro_col}_score0'] > MotifMatchArgs.min_single_score_jasp]
    # mc_noninvitro = mc_filtered[~mc_filtered[f'{ReferenceMatchCols.ref_invitro_col}_score0'] > MotifMatchArgs.min_single_score_jasp]


    #### PPM -------------------------------------------------------------------------
    ## MOTIFS:
    mc_ppm = mc.copy()

    # Move the FiNeMo hit-call PPMs into mc_ppm, leaving `mc` on its CWMs.
    # `mc` is the CWM compendium and is saved as such; the hit-call PPMs are a separate
    # output (mc_ppm, and mc_pfm derived from it). Assigning them onto `mc` instead
    # overwrote the CWMs, so the saved "motif_all" compendium held PPMs (per-motif sums
    # ~= motif length) rather than contribution scores.
    # Keep index
    mc_ppm.metadata[MetadataCols.index_col] = mc_ppm.metadata.index

    # Sort both sides by the TF-MoDISco name so row i of mc_ppm lines up with row i of ppms
    mc_ppm.sort(MetadataCols.name_tfmodisco_col, inplace=True)
    hits_df.sort_values('motif_name', inplace=True)
    sorted_idx = list(hits_df.index)
    ppms = ppms[sorted_idx, :, :]

    # Replace CWMs into PPMs
    mc_ppm.motifs = ppms

    # Resort
    mc_ppm.sort(MetadataCols.index_col, inplace=True)

    # For flag_remove: Set PPM to zero
    mc_ppm.motifs[mc_ppm[FilterArgs.filter_flag_col]] = 0.0

    # L1 normalize, per position
    mc_ppm.motifs = np.where(mc_ppm.motifs == 0, 0, mc_ppm.motifs / mc_ppm.motifs.sum(axis=(2), keepdims=True))

    # Re-slice
    mc_filtered_ppm = mc_ppm[~mc_ppm[FilterArgs.filter_flag_col]]
    mc_removed_ppm = mc_ppm[mc_ppm[FilterArgs.filter_flag_col]]
    mc_invitro_ppm = mc_filtered_ppm[mc_filtered_ppm[MetadataCols.name_col].isin(mc_invitro[MetadataCols.name_col])]
    mc_noninvitro_ppm = mc_filtered_ppm[mc_filtered_ppm[MetadataCols.name_col].isin(mc_noninvitro[MetadataCols.name_col])]

    ## Update metadata: Invitro
    mc_invitro.metadata[MetadataCols.index_col] = mc_invitro.metadata.index
    mc_invitro.sort(MetadataCols.name_col, inplace=True)
    mc_invitro_ppm.sort(MetadataCols.name_col, inplace=True)
    mc_invitro_ppm.metadata = mc_invitro.metadata
    mc_invitro.sort(MetadataCols.index_col, inplace=True)
    mc_invitro_ppm.sort(MetadataCols.index_col, inplace=True)

    if args.verbose:
        logging.info(f"Total motifs: {len(mc)}")
        logging.info(f"Filtered motifs: {len(mc_removed)}")
        logging.info(f"In vitro motifs: {len(mc_invitro)}")
        logging.info(f"Non-in vitro motifs: {len(mc_noninvitro)}")
        logging.info(f"Removed motifs: {len(mc_removed)}")


    ## CLUSTERS:
    # Average MotifCompendium
    mc_cluster_ppm = mc_filtered_ppm.cluster_averages(
        clustering=MetadataCols.cluster_col,
        aggregations=JasparPipelineArgs.aggregations,
        weight_col=JasparPipelineArgs.ppm_weight_col,
        compute_quality_stats=False,
    )

    # L1 normalize, per position
    mc_cluster_ppm.motifs = np.where(mc_cluster_ppm.motifs == 0, 0, mc_cluster_ppm.motifs / mc_cluster_ppm.motifs.sum(axis=(2), keepdims=True))

    ## Update metadata: From MotifCompendium cluster
    mc_cluster.metadata[MetadataCols.index_col] = mc_cluster.metadata.index
    mc_cluster.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_cluster_ppm.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_cluster_ppm.metadata = mc_cluster.metadata
    mc_cluster.sort(MetadataCols.index_col, inplace=True)
    mc_cluster_ppm.sort(MetadataCols.index_col, inplace=True)

    # Re-slice
    mc_cluster_invitro_ppm = mc_cluster_ppm[mc_cluster_ppm[MetadataCols.name_col].isin(mc_cluster_invitro[MetadataCols.name_col])]
    mc_cluster_noninvitro_ppm = mc_cluster_ppm[mc_cluster_ppm[MetadataCols.name_col].isin(mc_cluster_noninvitro[MetadataCols.name_col])]

    ## Update metadata: Invitro
    mc_cluster_invitro.metadata[MetadataCols.index_col] = mc_cluster_invitro.metadata.index
    mc_cluster_invitro.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_cluster_invitro_ppm.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_cluster_invitro_ppm.metadata = mc_cluster_invitro.metadata
    mc_cluster_invitro.sort(MetadataCols.index_col, inplace=True)
    mc_cluster_invitro_ppm.sort(MetadataCols.index_col, inplace=True)

    if args.verbose:
        logging.info(f"Total clusters: {len(mc_cluster_ppm)}")
        logging.info(f"In vitro clusters: {len(mc_cluster_invitro_ppm)}")
        logging.info(f"Non-in vitro clusters: {len(mc_cluster_noninvitro_ppm)}")


    # METACLUSTERS:
    # Average MotifCompendium
    mc_metacluster_invitro_ppm = mc_cluster_invitro_ppm.cluster_averages(
        clustering=MetadataCols.metacluster_col,
        aggregations=JasparPipelineArgs.aggregations,
        weight_col=JasparPipelineArgs.ppm_weight_col,
        compute_quality_stats=False,
    )

    # L1 normalize, per position
    mc_metacluster_invitro_ppm.motifs = np.where(mc_metacluster_invitro_ppm.motifs == 0, 0, mc_metacluster_invitro_ppm.motifs / mc_metacluster_invitro_ppm.motifs.sum(axis=(2), keepdims=True))

    ## Update metadata: From MotifCompendium cluster
    mc_metacluster_invitro.metadata[MetadataCols.index_col] = mc_metacluster_invitro.metadata.index
    mc_metacluster_invitro.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_metacluster_invitro_ppm.sort(MetadataCols.source_cluster_col, inplace=True)
    mc_metacluster_invitro_ppm.metadata = mc_metacluster_invitro.metadata
    mc_metacluster_invitro.sort(MetadataCols.index_col, inplace=True)
    mc_metacluster_invitro_ppm.sort(MetadataCols.index_col, inplace=True)

    if args.verbose:
        logging.info(f"In vitro metaclusters: {len(mc_metacluster_invitro_ppm)}")


    #### PFM -------------------------------------------------------------------------
    # Multiply probability x num_hits = counts
    ## MOTIFS:
    mc_pfm = mc_ppm.copy()
    mc_pfm.motifs = mc_pfm.motifs * mc_pfm[MetadataCols.num_hits_col].to_numpy()[:, np.newaxis, np.newaxis]

    # Re-slice
    mc_filtered_pfm = mc_pfm[~mc_pfm[FilterArgs.filter_flag_col]]
    mc_removed_pfm = mc_pfm[mc_pfm[FilterArgs.filter_flag_col]]
    mc_invitro_pfm = mc_filtered_pfm[mc_filtered_pfm[MetadataCols.name_col].isin(mc_invitro[MetadataCols.name_col])]
    mc_noninvitro_pfm = mc_filtered_pfm[mc_filtered_pfm[MetadataCols.name_col].isin(mc_noninvitro[MetadataCols.name_col])]


    ## CLUSTERS:
    mc_cluster_pfm = mc_cluster_ppm.copy()
    mc_cluster_pfm.motifs = mc_cluster_pfm.motifs * mc_cluster_pfm[MetadataCols.num_hits_col].to_numpy()[:, np.newaxis, np.newaxis]

    # Re-slice
    mc_cluster_invitro_pfm = mc_cluster_pfm[mc_cluster_pfm[MetadataCols.name_col].isin(mc_cluster_invitro[MetadataCols.name_col])]
    mc_cluster_noninvitro_pfm = mc_cluster_pfm[mc_cluster_pfm[MetadataCols.name_col].isin(mc_cluster_noninvitro[MetadataCols.name_col])]

    ## METACLUSTERS:
    mc_metacluster_invitro_pfm = mc_metacluster_invitro_ppm.copy()
    mc_metacluster_invitro_pfm.motifs = mc_metacluster_invitro_pfm.motifs * mc_metacluster_invitro_pfm[MetadataCols.num_hits_col].to_numpy()[:, np.newaxis, np.newaxis]


    #### EXPORT ---------------------------------------------------------------------
    ## CORE: mc, mc_invitro; mc_cluster_invitro
    logging.info(f"Saving MotifCompendium objects to: {args.output_dir}...")
    ## SAVE: MOTIFCOMPENDIUM OBJECT
    # Motifs: All
    os.makedirs(os.path.dirname(mc_path), exist_ok=True)
    mc.save(mc_path)
    os.makedirs(os.path.dirname(mc_ppm_path), exist_ok=True)
    mc_ppm.save(mc_ppm_path)
    os.makedirs(os.path.dirname(mc_pfm_path), exist_ok=True)
    mc_pfm.save(mc_pfm_path)

    # Motifs: Invitro
    os.makedirs(os.path.dirname(mc_invitro_path), exist_ok=True)
    mc_invitro.save(mc_invitro_path)
    os.makedirs(os.path.dirname(mc_invitro_ppm_path), exist_ok=True)
    mc_invitro_ppm.save(mc_invitro_ppm_path)
    os.makedirs(os.path.dirname(mc_invitro_pfm_path), exist_ok=True)
    mc_invitro_pfm.save(mc_invitro_pfm_path)

    # Clusters: In vitro
    os.makedirs(os.path.dirname(mc_cluster_invitro_path), exist_ok=True)
    mc_cluster_invitro.save(mc_cluster_invitro_path)
    os.makedirs(os.path.dirname(mc_cluster_invitro_ppm_path), exist_ok=True)
    mc_cluster_invitro_ppm.save(mc_cluster_invitro_ppm_path)
    os.makedirs(os.path.dirname(mc_cluster_invitro_pfm_path), exist_ok=True)
    mc_cluster_invitro_pfm.save(mc_cluster_invitro_pfm_path)

    if args.save_all:
        # Motifs
        os.makedirs(os.path.dirname(mc_noninvitro_path), exist_ok=True)
        mc_noninvitro.save(mc_noninvitro_path)
        os.makedirs(os.path.dirname(mc_noninvitro_ppm_path), exist_ok=True)
        mc_noninvitro_ppm.save(mc_noninvitro_ppm_path)
        os.makedirs(os.path.dirname(mc_noninvitro_pfm_path), exist_ok=True)
        mc_noninvitro_pfm.save(mc_noninvitro_pfm_path)

        os.makedirs(os.path.dirname(mc_remove_path), exist_ok=True)
        mc_removed.save(mc_remove_path)
        os.makedirs(os.path.dirname(mc_remove_ppm_path), exist_ok=True)
        mc_removed_ppm.save(mc_remove_ppm_path)
        os.makedirs(os.path.dirname(mc_remove_pfm_path), exist_ok=True)
        mc_removed_pfm.save(mc_remove_pfm_path)

        # Clusters
        os.makedirs(os.path.dirname(mc_cluster_path), exist_ok=True)
        mc_cluster.save(mc_cluster_path)
        os.makedirs(os.path.dirname(mc_cluster_ppm_path), exist_ok=True)
        mc_cluster_ppm.save(mc_cluster_ppm_path)
        os.makedirs(os.path.dirname(mc_cluster_pfm_path), exist_ok=True)
        mc_cluster_pfm.save(mc_cluster_pfm_path)

        os.makedirs(os.path.dirname(mc_cluster_noninvitro_path), exist_ok=True)
        mc_cluster_noninvitro.save(mc_cluster_noninvitro_path)
        os.makedirs(os.path.dirname(mc_cluster_noninvitro_ppm_path), exist_ok=True)
        mc_cluster_noninvitro_ppm.save(mc_cluster_noninvitro_ppm_path)
        os.makedirs(os.path.dirname(mc_cluster_noninvitro_pfm_path), exist_ok=True)
        mc_cluster_noninvitro_pfm.save(mc_cluster_noninvitro_pfm_path)

        # Metaclusters
        os.makedirs(os.path.dirname(mc_metacluster_invitro_path), exist_ok=True)
        mc_metacluster_invitro.save(mc_metacluster_invitro_path)


    ## HTML SUMMARY TABLE
    logging.info(f"Saving HTML summary tables to: {args.output_dir}...")
    # Clusters
    html_columns = [
        MetadataCols.name_col,
        f"{ReferenceMatchCols.ref_jaspar_col}_logo0", f"{ReferenceMatchCols.ref_jaspar_col}_name0", f"{ReferenceMatchCols.ref_jaspar_col}_score0",
        f"{ReferenceMatchCols.ref_invitro_col}_logo0", f"{ReferenceMatchCols.ref_invitro_col}_name0", f"{ReferenceMatchCols.ref_invitro_col}_score0",
    ] + JasparPipelineArgs.html_metadata_columns

    os.makedirs(os.path.dirname(html_cluster_invitro_path), exist_ok=True)
    mc_cluster_invitro.summary_table_html(
        html_out=html_cluster_invitro_path,
        columns=html_columns,
        logo_trimming=0,
        editable=True,
    )

    if args.save_all:
        # Clusters
        os.makedirs(os.path.dirname(html_cluster_noninvitro_path), exist_ok=True)
        mc_cluster_noninvitro.summary_table_html(
            html_out=html_cluster_noninvitro_path,
            columns=html_columns,
            logo_trimming=0,
            editable=True,
        )

        # Metaclusters
        html_columns = [
            MetadataCols.name_col,
            f"{ReferenceMatchCols.ref_jaspar_col}_logo0", f"{ReferenceMatchCols.ref_jaspar_col}_name0", f"{ReferenceMatchCols.ref_jaspar_col}_score0",
        ] + JasparPipelineArgs.html_metadata_columns

        os.makedirs(os.path.dirname(html_cluster_invitro_path), exist_ok=True)
        mc_metacluster_invitro.summary_table_html(
            html_out=html_metacluster_invitro_path,
            columns=html_columns,
            logo_trimming=0,
            editable=True,
        )  


    ## EXPORT: MODISCO HDF5
    logging.info(f"Saving as MoDISco HDF5 files to: {args.output_dir}...")
    # Motifs: All
    os.makedirs(os.path.dirname(modisco_motif_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_path,
    )
    os.makedirs(os.path.dirname(modisco_motif_ppm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_ppm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_ppm_path,
    )
    os.makedirs(os.path.dirname(modisco_motif_pfm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_pfm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_pfm_path,
    )

    # Motifs: Invitro
    os.makedirs(os.path.dirname(modisco_motif_invitro_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_invitro,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_invitro_path,
    )
    os.makedirs(os.path.dirname(modisco_motif_invitro_ppm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_invitro_ppm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_invitro_ppm_path,
    )
    os.makedirs(os.path.dirname(modisco_motif_invitro_pfm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_invitro_pfm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_motif_invitro_pfm_path,
    )

    # Cluster: Invitro
    os.makedirs(os.path.dirname(modisco_cluster_invitro_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_cluster_invitro,
        name_col=MetadataCols.name_col,
        save_loc=modisco_cluster_invitro_path,
    )
    os.makedirs(os.path.dirname(modisco_cluster_invitro_ppm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_cluster_invitro_ppm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_cluster_invitro_ppm_path,
    )
    os.makedirs(os.path.dirname(modisco_cluster_invitro_pfm_path), exist_ok=True)
    utils_analysis.export_compendium_modisco(
        mc=mc_cluster_invitro_pfm,
        name_col=MetadataCols.name_col,
        save_loc=modisco_cluster_invitro_pfm_path,
    )

    if args.save_all:
        # Motifs: Noninvitro
        os.makedirs(os.path.dirname(modisco_motif_noninvitro_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_noninvitro,
            name_col=MetadataCols.name_col,
            save_loc=modisco_motif_noninvitro_path,
        )
        os.makedirs(os.path.dirname(modisco_motif_noninvitro_ppm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_noninvitro_ppm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_motif_noninvitro_ppm_path,
        )
        os.makedirs(os.path.dirname(modisco_motif_noninvitro_pfm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_noninvitro_pfm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_motif_noninvitro_pfm_path,
        )

        # Cluster: Noninvitro
        os.makedirs(os.path.dirname(modisco_cluster_noninvitro_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_cluster_noninvitro,
            name_col=MetadataCols.name_col,
            save_loc=modisco_cluster_noninvitro_path,
        )
        os.makedirs(os.path.dirname(modisco_cluster_noninvitro_ppm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_cluster_noninvitro_ppm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_cluster_noninvitro_ppm_path,
        )
        os.makedirs(os.path.dirname(modisco_cluster_noninvitro_pfm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_cluster_noninvitro_pfm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_cluster_noninvitro_pfm_path,
        )

        # Metacluster: Invitro
        os.makedirs(os.path.dirname(modisco_metacluster_invitro_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_metacluster_invitro,
            name_col=MetadataCols.name_col,
            save_loc=modisco_metacluster_invitro_path,
        )
        os.makedirs(os.path.dirname(modisco_metacluster_invitro_ppm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_metacluster_invitro_ppm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_metacluster_invitro_ppm_path,
        )
        os.makedirs(os.path.dirname(modisco_metacluster_invitro_pfm_path), exist_ok=True)
        utils_analysis.export_compendium_modisco(
            mc=mc_metacluster_invitro_pfm,
            name_col=MetadataCols.name_col,
            save_loc=modisco_metacluster_invitro_pfm_path,
        )

    
    ## EXPORT: MEME TXT
    logging.info(f"Saving as MEME txt files to: {args.output_dir}...")
    # Motif: All
    os.makedirs(os.path.dirname(meme_motif_path), exist_ok=True)
    export_compendium_meme(
        mc=mc,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_path,
    )
    os.makedirs(os.path.dirname(meme_motif_ppm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_ppm[~mc_ppm['flag_remove']],
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_ppm_path,
    )
    os.makedirs(os.path.dirname(meme_motif_pfm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_pfm[~mc_pfm['flag_remove']],
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_pfm_path,
    )

    # Motif: Invitro
    os.makedirs(os.path.dirname(meme_motif_invitro_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_invitro,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_invitro_path,
    )
    os.makedirs(os.path.dirname(meme_motif_invitro_ppm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_invitro_ppm,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_invitro_ppm_path,
    )
    os.makedirs(os.path.dirname(meme_motif_invitro_pfm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_invitro_pfm,
        name_col=MetadataCols.name_col,
        save_loc=meme_motif_invitro_pfm_path,
    )

    # Cluster: Invitro
    os.makedirs(os.path.dirname(meme_cluster_invitro_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_cluster_invitro,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_invitro_path,
    )
    os.makedirs(os.path.dirname(meme_cluster_invitro_ppm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_cluster_invitro_ppm,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_invitro_ppm_path,
    )
    os.makedirs(os.path.dirname(meme_cluster_invitro_pfm_path), exist_ok=True)
    export_compendium_meme(
        mc=mc_cluster_invitro_pfm,
        name_col=MetadataCols.name_col,
        save_loc=meme_cluster_invitro_pfm_path,
    )

    if args.save_all:
        # Motif: Noninvitro
        os.makedirs(os.path.dirname(meme_motif_noninvitro_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_noninvitro,
            name_col=MetadataCols.name_col,
            save_loc=meme_motif_noninvitro_path,
        )
        os.makedirs(os.path.dirname(meme_motif_noninvitro_ppm_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_noninvitro_ppm,
            name_col=MetadataCols.name_col,
            save_loc=meme_motif_noninvitro_ppm_path,
        )
        os.makedirs(os.path.dirname(meme_motif_noninvitro_pfm_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_noninvitro_pfm,
            name_col=MetadataCols.name_col,
            save_loc=meme_motif_noninvitro_pfm_path,
        )

        # Cluster: Noninvitro
        os.makedirs(os.path.dirname(meme_cluster_noninvitro_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_cluster_noninvitro,
            name_col=MetadataCols.name_col,
            save_loc=meme_cluster_noninvitro_path,
        )
        os.makedirs(os.path.dirname(meme_cluster_noninvitro_ppm_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_cluster_noninvitro_ppm,
            name_col=MetadataCols.name_col,
            save_loc=meme_cluster_noninvitro_ppm_path,
        )
        os.makedirs(os.path.dirname(meme_cluster_noninvitro_pfm_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_cluster_noninvitro_pfm,
            name_col=MetadataCols.name_col,
            save_loc=meme_cluster_noninvitro_pfm_path,
        )

        # Metacluster: Invitro
        os.makedirs(os.path.dirname(meme_metacluster_invitro_path), exist_ok=True)
        export_compendium_meme(
            mc=mc_metacluster_invitro,
            name_col=MetadataCols.name_col,
            save_loc=meme_metacluster_invitro_path,
        )


    ## Export metadata
    logging.info(f"Saving metadata to: {args.output_dir}...")
    os.makedirs(os.path.dirname(metadata_motif_path), exist_ok=True)
    mc.metadata.to_csv(metadata_motif_path, sep="\t", index=False)
    os.makedirs(os.path.dirname(metadata_cluster_path), exist_ok=True)
    mc_invitro.metadata.to_csv(metadata_motif_invitro_path, sep="\t", index=False)
    os.makedirs(os.path.dirname(metadata_cluster_invitro_path), exist_ok=True)
    mc_cluster_invitro.metadata.to_csv(metadata_cluster_invitro_path, sep="\t", index=False)

    if args.save_all:
        os.makedirs(os.path.dirname(metadata_motif_noninvitro_path), exist_ok=True)
        mc_noninvitro.metadata.to_csv(metadata_motif_noninvitro_path, sep="\t", index=False)
        os.makedirs(os.path.dirname(metadata_motif_removed_path), exist_ok=True)
        mc_removed.metadata.to_csv(metadata_motif_removed_path, sep="\t", index=False)

        os.makedirs(os.path.dirname(metadata_cluster_noninvitro_path), exist_ok=True)
        mc_cluster.metadata.to_csv(metadata_cluster_path, sep="\t", index=False)
        os.makedirs(os.path.dirname(metadata_cluster_noninvitro_path), exist_ok=True)
        mc_cluster_noninvitro.metadata.to_csv(metadata_cluster_noninvitro_path, sep="\t", index=False)

        os.makedirs(os.path.dirname(metadata_metacluster_invitro_path), exist_ok=True)
        mc_metacluster_invitro.metadata.to_csv(metadata_metacluster_invitro_path, sep="\t", index=False)


    logging.info(f'Run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
