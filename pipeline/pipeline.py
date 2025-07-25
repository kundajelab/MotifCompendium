import os
import sys
import time
import argparse
import logging
import json
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MotifCompendium
import MotifCompendium.utils.config as utils_config
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.clustering as utils_clustering
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.plotting as utils_plotting
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.visualization as utils_visualization

import configs


def setup_parser():
    parser = argparse.ArgumentParser(description="Run the MotifCompendium pipeline.")

    parser.add_argument("-im", "--input-mc", type=str, default=None, help="Path to the input MotifCompendium object.")
    parser.add_argument("-io", "--input-old-mc", type=str, default=None, help="Path to the input old MotifCompendium object.")
    parser.add_argument("-ih", "--input-modisco-h5s", nargs="+", type=str, default=None, help="Path(s) to the input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-sh", "--input-subpatterns", action="store_true", help="Use subpatterns (instead of main patterns) when loading input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("-ip", "--input-pfms", nargs="+", type=str, default=None, help="Path(s) to the input motif PFM files, in PFM or MEME format.")
    parser.add_argument("-in", "--input-names", nargs="+", type=str, default=None, help="Nickname(s) of the input file(s): TF-MoDISco (lite) H5, PFMs, MEME.")

    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, per input file or motif: CSV, TSV format.")
    parser.add_argument("-r", "--reference", type=str, default=None, help="Path to the main reference motif file: MotifCompendium object, or PFM, MEME .txt format.")
    parser.add_argument("--add-reference", nargs="+", type=str, default=None, help="Path(s) to additional reference motif files for final labeling: MotifCompendium object, or PFM, MEME .txt format.")
    parser.add_argument("--modisco-region-width", type=int, default=400, help="Width of region used during TF-Modisco run. (Default: 400 bp)")
    
    parser.add_argument("--no-filter", action="store_false", dest="filter", help="Do not apply ANY filters to motifs, clusters, meta-clusters, sub-clusters. This will override all other filter options.")
    parser.add_argument("--strict-filter", action="store_true", help="Apply a strict filter, that does not add back motifs even when matched with a database.")

    parser.add_argument("--sim-scan", nargs="+", type=float, default=None, help="List of similarity thresholds to scan during clustering.")
    parser.add_argument("--sim-threshold", type=float, default=0.88, help="Similarity threshold to apply during clustering.")
    parser.add_argument("--sim-threshold-meta", type=float, default=None, help="Similarity threshold to apply during clustering ON TOP of clusters, to create meta-clusters (meta-cluster > cluster). If not provided, will not create meta-clusters.")
    parser.add_argument("--sim-threshold-sub", type=float, default=None, help="Similarity threshold to apply during clustering WITHIN clusters, to create sub-clusters (sub-cluster < cluster) If not provided, will not create sub-clusters.")
    parser.add_argument("--sim-threshold-force", type=float, default=None, help="Maximum similarity ensured between clusters, meta-clusters, sub-clusters, by recursive connected components clustering. If not provided, will not enforce minimum similarity.")
    parser.add_argument("--cluster-on", type=str, default=None, help="Prior categorization to cluster on, provided as a column in the metadata. If not provided, will cluster each motif individually. CANNOT have cluster-on and cluster-within at the same time.")
    parser.add_argument("--cluster-within", type=str, default=None, help="Prior categorization to cluster within, provided as a column in the metadata. If not provided, will cluster each motif individually. CANNOT have cluster-on and cluster-within at the same time.")
    parser.add_argument("--cluster-recursive", action="store_true", help="Recursively cluster meta-clusters, and sub-clusters.")
    parser.add_argument("--quality", action="store_true", help="Generate quality metrics and plots to check clustering quality.")
    
    parser.add_argument("--html-motif-collection", action="store_true", help="Generate HTML collection of motif constituents per cluster.")
    parser.add_argument("--html-motif-table", action="store_true", help="Generate HTML summary table of individual motifs.")
    parser.add_argument("--html-motif-removed", action="store_true", help="Generate HTML summary table of removed motifs.")
    parser.add_argument("--html-cluster-table", action="store_true", help="Generate HTML summary table of clusters, meta-clusters, sub-clusters.")
    parser.add_argument("--html-cluster-removed", action="store_true", help="Generate HTML summary table of removed clusters, meta-clusters, sub-clusters.")
    parser.add_argument("--html-max-rows", type=int, default=None, help="Maximum number of rows to display in HTML tables.")
    parser.add_argument("--logo-trim", type=bool_or_float, default=True, help="Trim flanks of logo images: True: Standard trimming; False: No trimming; [0, 1]: Trim flanks with contribution less than <logo-trim> * max contribution.")
    parser.add_argument("--fast-plot", action="store_true", help="Reduce the resolution of logo images for faster plotting.")

    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("--var-chunk", action="store_true", help="Starting from max_chunk, dynamically reduce chunk size, based on available CPU/GPU memory.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Do not check the validity of the MotifCompendium objects.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not scale motifs by information content.")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### DEFAULT OPTIONS ------------------------------------------------------------
def set_default_options():
    global OutputPaths
    global MetadataCols
    global MotifMatchArgs
    global ClusterArgs
    global VisualizeArgs
    global MotifFilterArgs
    global MotifFilterArgs

    OutputPaths = configs.OutputPaths()
    MetadataCols = configs.MetadataCols()
    MotifMatchArgs = configs.MotifMatchArgs()
    ClusterArgs = configs.ClusterArgs()
    VisualizeArgs = configs.VisualizeArgs()
    MotifFilterArgs = configs.MotifFilterArgs()


#### FUNCTIONS -------------------------------------------------------------------
def bool_or_float(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        try:
            return float(value)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Invalid value for bool_or_float: {value}")


def build_inputs_dict(input_list: list, name_list: list | None) -> dict:
    """
    Build a dictionary of input files.
    
    Args:
        input_list (list): List of input files: TF-Modisco H5, PFM, MEME.
        name_list (list | None): List of names for the input files. If None, use the base name of the input files.

    Returns:
        dict: Dictionary of input files: key: name, value: path.
    """
    if name_list is None:
        name_list = [os.path.basename(h5) for h5 in input_list]
    inputs_dict = dict(zip(name_list, input_list))
    return inputs_dict

def label_motifs(
    mc: MotifCompendium,
    reference: str,
    label_col: str,
    max_submotifs: int,
    min_score: float,
    args: argparse.Namespace,
) -> None:
    """
    Label motifs in the MotifCompendium object based on a reference file.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        reference (str): Path to the reference file.
        label_col (str): Column name to save the labels.
        args (argparse.Namespace): The command line arguments.
        
    Returns:
        None
    """
    # Match reference motifs
    if reference.endswith(".mc"):
        ref_mc = MotifCompendium.load(reference, safe=args.safe)
        utils_analysis.assign_label_from_other_compendium(
            assign_to_mc=mc,
            assign_from_mc=ref_mc,
            from_label_col="name",
            save_col_prefix=label_col,
            min_score=min_score,
            max_submotifs=max_submotifs,
            save_images=True,
            logo_trimming=args.logo_trimming,
        )

    elif reference.endswith("pfm.txt") or reference.endswith("pfm") or reference.endswith("meme.txt") or reference.endswith(".meme"):
        utils_analysis.assign_label_from_pfms(
            mc=mc,
            pfm_file=reference,
            ic=False,
            save_col_prefix=label_col,
            min_score=min_score,
            max_submotifs=max_submotifs,
            save_images=True,
            logo_trimming=args.logo_trimming,
        )

    else:
        logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
        raise ValueError(
            "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
        )


def filter_motifs(
    mc: MotifCompendium,
    MotifFilterArgs: configs.MotifFilterArgs,
    MetadataCols: configs.MetadataCols,
    args: argparse.Namespace,
) -> None:
    """
    Apply motif filters to the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        MotifFilterArgs (configs.MotifFilterArgs): The filter arguments.
        args (argparse.Namespace): The command line arguments.
        
    Returns:
        None
    """
    # Filter #1: Calculate and apply filter metrics
    if args.verbose:
        logging.info(f"Calculating filter metrics:\n"
                f"  {MotifFilterArgs.motif_metrics}\n"
        )
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        logging.info(f"Applying filters as flag:\n"
                f"  {MotifFilterArgs.motif_filters}\n"
        )
    if args.time:
        start_time = time.time()
    for filter_args in MotifFilterArgs.motif_filters:
        if filter_args.apply_motif:
            apply_filter_threshold(
                mc=mc,
                flag_col=MetadataCols.filter_col_flag,
                metric=filter_args.metric,
                operation=filter_args.operation,
                threshold=filter_args.threshold,
                override=filter_args.override,
            )
    if args.verbose:
        logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #3: Override flag: Good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for override_filter_args in MotifFilterArgs.override_filters:
        if override_filter_args.apply_motif:
            apply_filter_threshold(
                mc=mc,
                flag_col=MetadataCols.filter_col_flag,
                metric=override_filter_args.metric,
                operation=override_filter_args.operation,
                threshold=override_filter_args.threshold,
                override=override_filter_args.override,
            )
    if args.verbose:
        logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # # Filter #4: Override flag: First motif
    # if args.first_posmotif:
    #     if args.verbose:
    #         logging.info(f"Overriding flags for first positive motif...")
    #     mc[MetadataCols.filter_col_flag] = (mc[MetadataCols.filter_col_flag] & 
    #                                                     ((~mc["name"].str.contains("pattern_0")) |
    #                                                     (mc["posneg"] != "pos")))
    # if args.verbose:
    #     logging.info(f"Total number of first positive motifs: {len(mc[mc['name'].str.contains('pattern_0') & (mc['posneg'] == 'pos')])}")

    # Filter #5: Apply strict filters
    if args.strict_filter:
        if args.verbose:
            logging.info(
                f"Applying strict filters:\n"
                f"  {MotifFilterArgs.strict_filters}"
            )
        if args.time:
            start_time = time.time()
        for strict_filter_args in MotifFilterArgs.strict_filters:
            if strict_filter_args.apply_motif:
                apply_filter_threshold(
                    mc=mc,
                    flag_col=MetadataCols.filter_col_flag,
                    metric=strict_filter_args.metric,
                    operation=strict_filter_args.operation,
                    threshold=strict_filter_args.threshold,
                    override=strict_filter_args.override,
                )
        if args.verbose:
            logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # # Filter #6: Select first positive motif only
    # if args.first_posmotif_only:
    #     if args.verbose:
    #         logging.info(f"Selecting first positive motif only...")
    #     if args.time:
    #         start_time = time.time()
    #     mc[MetadataCols.filter_col_flag] = True
    #     mc[MetadataCols.filter_col_flag] = (mc[MetadataCols.filter_col_flag] & 
    #                                                     ((~mc["name"].str.contains("pattern_0")) |
    #                                                     (mc["posneg"] != "pos")))
    #     if args.verbose:
    #         logging.info(f"Total number of first positive motifs: {len(mc[mc['name'].str.contains('pattern_0') & (mc['posneg'] == 'pos')])}")
    #     if args.time:
    #         logging.info(f"Time taken: {time.time() - start_time:.2f}s")


def filter_clusters(
    mc: MotifCompendium,
    MotifFilterArgs: configs.MotifFilterArgs,
    MetadataCols: configs.MetadataCols,
    args: argparse.Namespace,
) -> None:
    """
    Apply cluster filters to the MotifCompendium object.

    Args:
        mc (MotifCompendium): The MotifCompendium object.
        args (argparse.Namespace): The command line arguments.

    Returns:
        None
    """
    # Filter #1: Calculate and flag filters
    if args.verbose:
        logging.info(f"Calculating filter metrics:\n"
            f"  {MotifFilterArgs.motif_metrics}"
        )
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        logging.info(f"Applying filters as flag:\n"
            f"  {MotifFilterArgs.motif_filters}"
        )
    if args.time:
        start_time = time.time()
    for filter_args in MotifFilterArgs.motif_filters:
        if filter_args.apply_cluster:
            apply_filter_threshold(
                mc=mc,
                flag_col=MetadataCols.filter_col_flag,
                metric=filter_args.metric,
                operation=filter_args.operation,
                threshold=filter_args.threshold,
                override=filter_args.override,
            )
    if args.verbose:
        logging.info(f"Number of flagged motifs: {len(mc[mc[MetadataCols.filter_col_flag]])}")
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    # # Filter #2: Flag singleton clusters
    # if args.rm_singletons:
    #     if args.verbose:
    #         logging.info(f"Flagging singleton clusters...")
    #     if args.time:
    #         start_time = time.time()
    #     mc[MetadataCols.filter_col_flag] = mc["num_motifs"] == 1
    #     if args.verbose:
    #         logging.info(f"Number of singleton clusters: {len(mc[mc['num_motifs'] == 1])}")
    #     if args.time:
    #         logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    # Filter #3: Override flags, for good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for override_filter_args in MotifFilterArgs.override_filters:
        if override_filter_args.apply_cluster:
            apply_filter_threshold(
                mc=mc,
                flag_col=MetadataCols.filter_col_flag,
                metric=override_filter_args.metric,
                operation=override_filter_args.operation,
                threshold=override_filter_args.threshold,
                override=override_filter_args.override,
            )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #4: Apply strict filters
    if args.strict_filter:
        if args.verbose:
            logging.info(
                f"Applying strict filters:\n"
                f"  {MotifFilterArgs.strict_filters}"
            )
        if args.time:
            start_time = time.time()
        for strict_filter_args in MotifFilterArgs.strict_filters:
            if strict_filter_args.apply_cluster:
                apply_filter_threshold(
                    mc=mc,
                    flag_col=MetadataCols.filter_col_flag,
                    metric=strict_filter_args.metric,
                    operation=strict_filter_args.operation,
                    threshold=strict_filter_args.threshold,
                    override=strict_filter_args.override,
                )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

def apply_filter_threshold(
    mc: MotifCompendium,
    flag_col: str,
    metric: str,
    operation: str,
    threshold: float,
    override: bool,
) -> None:
    """
    Apply a filter threshold to the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        flag_col (str): The column name to use for the filter flag.
        metric (str): The metric to use for the filter.
        operation (str): The operation to use for the filter. One of: <, <=, >, >=, ==, !=
        threshold (float): The threshold to use for the filter.
        override (bool): Whether to override the existing flag or not.
    
    Returns:
        None
    """
    if override is True:
        if operation == "<":
            mc[flag_col] = mc[flag_col] & (mc[metric] < threshold)
        elif operation == "<=":
            mc[flag_col] = mc[flag_col] & (mc[metric] <= threshold)
        elif operation == ">":
            mc[flag_col] = mc[flag_col] & (mc[metric] > threshold)
        elif operation == ">=":
            mc[flag_col] = mc[flag_col] & (mc[metric] >= threshold)
        elif operation == "==":
            mc[flag_col] = mc[flag_col] & (mc[metric] == threshold)
        elif operation == "!=":
            mc[flag_col] = mc[flag_col] & (mc[metric] != threshold)
        elif operation == "isna":
            mc[flag_col] = mc[flag_col] & (mc[metric].isna())
        elif operation == "notna":
            mc[flag_col] = mc[flag_col] & (mc[metric].notna())
        else:
            raise ValueError("Invalid operation for filter threshold.")
    elif override is False:
        if operation == "<":
            mc[flag_col] = mc[flag_col] | (mc[metric] < threshold)
        elif operation == "<=":
            mc[flag_col] = mc[flag_col] | (mc[metric] <= threshold)
        elif operation == ">":
            mc[flag_col] = mc[flag_col] | (mc[metric] > threshold)
        elif operation == ">=":
            mc[flag_col] = mc[flag_col] | (mc[metric] >= threshold)
        elif operation == "==":
            mc[flag_col] = mc[flag_col] | (mc[metric] == threshold)
        elif operation == "!=":
            mc[flag_col] = mc[flag_col] | (mc[metric] != threshold)
        elif operation == "isna":
            mc[flag_col] = mc[flag_col] | (mc[metric].isna())
        elif operation == "notna":
            mc[flag_col] = mc[flag_col] | (mc[metric].notna())
        else:
            raise ValueError("Invalid operation for filter threshold.")

def generate_quality_plots(
    mc: MotifCompendium,
    cluster_col_name: str,
    quality_dir: str,
    args: argparse.Namespace,
) -> None:
    """
    Generate quality plots for the MotifCompendium object.
    
    Args:
        mc (MotifCompendium): The MotifCompendium object.
        args (argparse.Namespace): The command line arguments.
        
    Returns:
        None
    """
    # Quality: Histogram
    histogram_path = os.path.join(quality_dir, f"histogram_{cluster_col_name}.png")
    if args.verbose:
        logging.info(f"Summarizing cluster quality (Histogram): {histogram_path}...")
    if args.time:
        start_time = time.time()
    utils_analysis.judge_clustering(
        mc=mc,
        cluster_col=cluster_col_name,
        save_loc=histogram_path,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Quality: Heatmap
    # heatmap_path = os.path.join(quality_dir, f"heatmap_{cluster_col_name}.png")
    # if args.verbose:
    #     logging.info(f"Summarizing cluster quality (Heatmap): {heatmap_path}...")
    # if args.time:
    #     start_time = time.time()
    # mc.heatmap(
    #     sort_by=cluster_col_name,
    #     save_loc=heatmap_path,
    # )
    # plt.savefig(heatmap_path)
    # if args.time:
    #     logging.info(f"Time taken: {time.time() - start_time:.2f}s")

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
    if agg_col not in mc.columns():
        raise ValueError(f"Column {agg_col} not found in metadata.")

    if weight_col not in mc.columns():
        raise ValueError(f"Column {weight_col} not found in metadata.")

    weighted_avgs = []
    for cluster_id in mc_cluster["source_cluster"]:
        mc_subset = mc[mc[cluster_col] == cluster_id]
        if len(mc_subset) == 0:
            weighted_avgs.append(np.nan)
            continue
        weighted_avg = (mc_subset[agg_col] * mc_subset[weight_col]).sum() / mc_subset[weight_col].sum()
        weighted_avgs.append(weighted_avg)
    mc_cluster[new_col] = weighted_avgs

#### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()

    ### OPTIONS -----------------------------------------------------------------
    # Set default options
    set_default_options()

    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # Set compute options
    if args.verbose:
        logging.info(
            f"Setting compute options:\n"
            f"  max_chunk={args.max_chunk},\n"
            f"  max_cpus={args.max_cpus},\n"
            f"  use_gpu={args.use_gpu},\n"
            f"  fast_plot={args.fast_plot}"
        )

    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        fast_plotting=args.fast_plot,
    )

    # Check output
    if args.verbose:
        logging.info("Checking output directory...")
        logging.info(f"Output directory: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        logging.info(f"Creating output directory: {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)


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

    # Load the input old MotifCompendium object
    elif args.input_old_mc:
        # Check input
        if args.verbose:
            logging.info("Checking input paths...")
        if not os.path.exists(args.input_old_mc):
            logging.error(f"Input old MotifCompendium object not found: {args.input_old_mc}")
            raise FileNotFoundError(f"Input old MotifCompendium object not found: {args.input_old_mc}")

        # Load old MotifCompendium object
        if args.verbose:
            logging.info(f"Loading input old MotifCompendium object: {args.input_old_mc}...")
        if args.time:
            start_time = time.time()
        mc = MotifCompendium.load_old_compendium(
            args.input_old_mc,
        )
        if args.verbose:
            logging.info(
                f"Completed loading and rebuilding old MotifCompendium object:\n"
                f"  Total number of motifs: {len(mc)}\n"
                f"  Metadata columns: {mc.columns()}"
            )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Load the input H5 files
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
            elif os.path.getsize(h5) <= 888: # 888 bytes is the minimum size for a valid HDF5 file
                logging.error(f"Input H5 file is empty: {h5}")
                failed_h5s.append(h5)
            
            # Check if H5 file is a valid Modisco file
            else:
                with h5py.File(h5, "r") as f:
                    if "pos_patterns" not in f and "neg_patterns" not in f:
                        logging.error(f"Input H5 file is not a valid Modisco file: {h5}")

        if failed_h5s:
            logging.error(f"Failed to load the following H5 files: {', '.join(failed_h5s)}")
            raise ValueError(f"Failed to load the following H5 files: {', '.join(failed_h5s)}")

        # Build MotifCompendium object
        modisco_dict = build_inputs_dict(args.input_modisco_h5s, args.input_names)
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
                    modisco_region_width=args.modisco_region_width,
                    ic=args.ic,
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
        pfm_dict = build_inputs_dict(args.input_pfms, args.input_names)
        if args.verbose:
            logging.info("Loading input PFM files...")
        if args.time:
            start_time = time.time()
        mc = MotifCompendium.build_from_pfm(
            pfm_dict=pfm_dict,
            ic=args.ic,
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

        # Add metadata aggregation
        aggregate_cols = [agg[0] for agg in ClusterArgs.aggregate_metadata]
        for col in metadata_df.columns:
            if col not in aggregate_cols:
                ClusterArgs.aggregate_metadata.append((col, "concat", f"{col}"))

        # Update HTML table columns
        VisualizeArgs.html_table_cols_base.extend([f"{col}" for col in metadata_df.columns])

    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_full)
    if args.verbose:
        logging.info(f"Saving MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ### LABEL: MODISCO -----------------------------------------------------------
    if args.reference is None:
        args.reference = MotifMatchArgs.reference_default

    ## Apply labels
    if args.verbose:
        logging.info(f"Matching motifs to reference file: {args.reference}...\n"
                f"Label column: {MetadataCols.label_column_prefix}"
        )
    if args.time:
        start_time = time.time()
    label_motifs(
        mc=mc,
        reference=args.reference,
        label_col=MetadataCols.label_column_prefix,
        max_submotifs=MotifMatchArgs.max_submotifs,
        min_score=MotifMatchArgs.min_score,
        args=args,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    ## Additional labels
    if args.add_reference:
        html_addlabel_cols = []
        for reference in args.add_reference:
            label_col = os.path.splitext(os.path.basename(reference))[0]

            ## Label: Cluster
            if args.verbose:
                logging.info(f"Adding labels to clusters, from reference file: {reference}...")
            if args.time:
                start_time = time.time()
            label_motifs(
                mc=mc,
                reference=reference,
                label_col=label_col,
                max_submotifs=1,
                min_score=MotifMatchArgs.min_score,
                args=args,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            # Update HTML table columns
            html_addlabel_cols.extend([f"{label_col}_logo0", f"{label_col}_name0", f"{label_col}_score0"])
            if args.verbose:
                logging.info(f"Adding the following columns to HTML table: {html_addlabel_cols}")
            VisualizeArgs.html_table_cols_label.extend(html_addlabel_cols)

    ## Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_labeled)
    if args.verbose:
        logging.info(f"Saving labeled MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ### FILTER: MODISCO ----------------------------------------------------------
    if args.filter:
        if args.reference is None:
            args.reference = MotifMatchArgs.reference_default

        # Apply filters
        if args.verbose:
            logging.info(f"Applying filters to motifs...")
        mc[MetadataCols.filter_col_flag] = False
        filter_motifs(
            mc=mc,
            MotifFilterArgs=MotifFilterArgs,
            MetadataCols=MetadataCols,
            args=args,
        )

        # Remove flagged motifs
        if args.verbose:
            logging.info(f"Removing flagged motifs...")
        mc_removed = mc[mc[MetadataCols.filter_col_flag]]
        mc = mc[~mc[MetadataCols.filter_col_flag]]
        if args.verbose:
            logging.info(f"Number of motifs after removing flagged motifs: {len(mc)}\n"
                        f"Number of motifs removed: {len(mc_removed)}")

        # Save MotifCompendium objects
        mc_path = os.path.join(args.output_dir, OutputPaths.mc_filtered)
        mc_removed_path = os.path.join(args.output_dir, OutputPaths.mc_removed)
        if args.verbose:
            logging.info(
                f"Saving filtered MotifCompendium objects:\n"
                f"  Filtered: {mc_path}\n"
                f"  Removed: {mc_removed_path}"
            )
        if args.time:
            start_time = time.time()
        mc.save(mc_path)
        mc_removed.save(mc_removed_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ### CLUSTERING ---------------------------------------------------------------
    # Set options
    if args.sim_scan:
        sim_thresholds = args.sim_scan
        if args.sim_threshold not in sim_thresholds:
            sim_thresholds.append(args.sim_threshold)
    else:
        sim_thresholds = [args.sim_threshold]

    # Column names
    cluster_on_name = None
    cluster_within_name = None
    recursive_name = None
    force_name = None
    if args.cluster_within:
        cluster_within_name = f"Within-{args.cluster_within}-"
    if args.cluster_on:
        cluster_on_name = f"On-{args.cluster_on}-"
    if args.cluster_recursive:
        recursive_name = f"-rec"
    if args.sim_threshold_force:
        force_name = f"-force{args.sim_threshold_force}"

    # Weight column
    if ClusterArgs.weight_col not in mc.columns():
        logging.warning(f"Weight column '{ClusterArgs.weight_col}' not found in metadata. Not performing weighted average during cluster aggregation.")
        ClusterArgs.weight_col = None

    # Cluster: Cluster, Meta-cluster, Sub-cluster
    for sim_threshold in sim_thresholds:
        while True:
            try:
                ## Cluster: Cluster motifs
                cluster_col_name = f"{cluster_within_name or ''}{cluster_on_name or ''}{ClusterArgs.algorithm}_{sim_threshold}{recursive_name or ''}{force_name or ''}"
                if args.verbose:
                    logging.info(f"Clustering motifs using: {cluster_col_name}...")
                if args.time:
                    start_time = time.time()
                if args.cluster_on and args.cluster_within: # Cluster ON and WITHIN
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm,
                        similarity_threshold=sim_threshold,
                        cluster_within_on=(args.cluster_within, args.cluster_on),
                        cluster_on_weight=ClusterArgs.weight_col,
                        save_name=cluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                    )
                elif args.cluster_on and not args.cluster_within: # Cluster ON only
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm,
                        similarity_threshold=sim_threshold,
                        cluster_on=args.cluster_on,
                        cluster_on_weight=ClusterArgs.weight_col,
                        save_name=cluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                    )
                elif not args.cluster_on and args.cluster_within: # Cluster WITHIN only
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm,
                        similarity_threshold=sim_threshold,
                        cluster_within=args.cluster_within,
                        save_name=cluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                    )
                else: # Cluster STANDARD
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm,
                        similarity_threshold=sim_threshold,
                        save_name=cluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                    )
                # Recursive cluster
                if args.cluster_recursive:
                    min_len = mc[cluster_col_name].nunique()
                    for i in range(ClusterArgs.max_iter):
                        if args.verbose:
                            logging.info(f"Recursively clustering motifs: {i+1}...")
                        if args.cluster_within:
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm,
                                similarity_threshold=sim_threshold,
                                cluster_within_on=(args.cluster_within, cluster_col_name),
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=cluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                            )
                        else:
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm,
                                similarity_threshold=sim_threshold,
                                cluster_on=cluster_col_name,
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=cluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
                            )
                        new_min_len = mc[cluster_col_name].nunique()
                        if min_len == new_min_len:
                            break
                        else:
                            min_len = new_min_len                    
                # Force-cluster
                if args.sim_threshold_force:
                    if args.verbose:
                        logging.info(f"Force-clustering motifs using: {ClusterArgs.algorithm_force}_{args.sim_threshold_force}{recursive_name or ''}{force_name or ''}...")
                    for i in range(ClusterArgs.max_iter):
                        if args.cluster_within:
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_force,
                                similarity_threshold=args.sim_threshold_force,
                                cluster_within_on=(args.cluster_within, cluster_col_name),
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=cluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_force],
                            )
                        else:
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_force,
                                similarity_threshold=args.sim_threshold_force,
                                cluster_on=cluster_col_name,
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=cluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_force],
                            )
                        new_min_len = mc[cluster_col_name].nunique()
                        if not args.cluster_recursive:
                            break
                        if args.verbose:
                            logging.info(f"Recursively clustering motifs: {i+1}...")
                        if min_len == new_min_len:
                            break
                        else:
                            min_len = new_min_len
                if args.verbose:
                    logging.info(f"Total number of clusters ({cluster_col_name}): {mc[cluster_col_name].nunique()}")
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")
                # Add to aggregate metadata
                ClusterArgs.aggregate_metadata.append((cluster_col_name, "concat", cluster_col_name))


                ## Meta-cluster: Cluster on top of clusters
                if args.sim_threshold_meta:
                    metacluster_col_name = f"{cluster_col_name}-meta{ClusterArgs.algorithm_meta}_{args.sim_threshold_meta}{recursive_name or ''}{force_name or ''}"
                    if args.verbose:
                        logging.info(f"Meta-clustering motifs using: {ClusterArgs.algorithm_meta}_{args.sim_threshold_meta}{recursive_name or ''}{force_name or ''}...")
                    if args.time:
                        start_time = time.time()
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm_meta,
                        similarity_threshold=args.sim_threshold_meta,
                        cluster_on=cluster_col_name,
                        cluster_on_weight=ClusterArgs.weight_col,
                        save_name=metacluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_meta],
                    )
                    # Recursive cluster:
                    if args.cluster_recursive:
                        min_len = mc[metacluster_col_name].nunique()
                        for i in range(ClusterArgs.max_iter):
                            if args.verbose:
                                logging.info(f"Recursively clustering motifs: {i+1}...")
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_meta,
                                similarity_threshold=args.sim_threshold_meta,
                                cluster_on=metacluster_col_name,
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=metacluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_meta],
                            )
                            new_min_len = mc[metacluster_col_name].nunique()
                            if min_len == new_min_len:
                                break
                            else:
                                min_len = new_min_len
                    # Force-cluster:
                    if args.sim_threshold_force:
                        if args.verbose:
                            logging.info(f"Force-clustering motifs using: {ClusterArgs.algorithm_force}_{args.sim_threshold_force}{recursive_name or ''}{force_name or ''}...")
                        for i in range(ClusterArgs.max_iter):
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_force,
                                similarity_threshold=args.sim_threshold_force,
                                cluster_on=metacluster_col_name,
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=metacluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_force],
                            )
                            new_min_len = mc[metacluster_col_name].nunique()
                            if not args.cluster_recursive:
                                break
                            if args.verbose:
                                logging.info(f"Recursively clustering motifs: {i+1}...")
                            if min_len == new_min_len:
                                break
                            else:
                                min_len = new_min_len
                    if args.verbose:
                        logging.info(f"Total number of meta-clusters ({metacluster_col_name}): {mc[metacluster_col_name].nunique()}")
                    if args.time:
                        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
                    # Add to aggregate metadata
                    ClusterArgs.aggregate_metadata.append((metacluster_col_name, "concat", metacluster_col_name))


                ## Sub-cluster: Cluster motifs within clusters
                if args.sim_threshold_sub:
                    subcluster_col_name = f"{cluster_col_name}-sub{ClusterArgs.algorithm_sub}_{args.sim_threshold_sub}{recursive_name or ''}{force_name or ''}"
                    if args.verbose:
                        logging.info(f"Sub-clustering motifs within {cluster_col_name} using: {ClusterArgs.algorithm_sub}_{args.sim_threshold_sub}{recursive_name or ''}{force_name or ''}...")
                    if args.time:
                        start_time = time.time()
                    mc.cluster(
                        algorithm=ClusterArgs.algorithm_sub,
                        similarity_threshold=args.sim_threshold_sub,
                        cluster_within=cluster_col_name,
                        save_name=subcluster_col_name,
                        largest_clusters_first=True,
                        **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_sub],
                    )
                    # Recursive cluster:
                    if args.cluster_recursive:
                        min_len = mc[subcluster_col_name].nunique()
                        for i in range(ClusterArgs.max_iter):
                            if args.verbose:
                                logging.info(f"Recursively clustering motifs: {i+1}...")
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_sub,
                                similarity_threshold=args.sim_threshold_sub,
                                cluster_within_on=(cluster_col_name, subcluster_col_name),
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=subcluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_sub],
                            )
                            new_min_len = mc[subcluster_col_name].nunique()
                            if min_len == new_min_len:
                                break
                            else:
                                min_len = new_min_len
                    # Force-cluster:
                    if args.sim_threshold_force:
                        if args.verbose:
                            logging.info(f"Force-clustering motifs using: {ClusterArgs.algorithm_force}_{args.sim_threshold_force}{recursive_name or ''}{force_name or ''}...")
                        for i in range(ClusterArgs.max_iter):
                            mc.cluster(
                                algorithm=ClusterArgs.algorithm_force,
                                similarity_threshold=args.sim_threshold_force,
                                cluster_within_on=(cluster_col_name, subcluster_col_name),
                                cluster_on_weight=ClusterArgs.weight_col,
                                save_name=subcluster_col_name,
                                largest_clusters_first=True,
                                **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm_force],
                            )
                            new_min_len = mc[subcluster_col_name].nunique()
                            if not args.cluster_recursive:
                                break
                            if args.verbose:
                                logging.info(f"Recursively clustering motifs: {i+1}...")
                            if min_len == new_min_len:
                                break
                            else:
                                min_len = new_min_len
                    if args.verbose:
                        logging.info(f"Total number of sub-clusters ({subcluster_col_name}): {mc[subcluster_col_name].nunique()}")
                    if args.time:
                        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
                    # Add to aggregate metadata
                    ClusterArgs.aggregate_metadata.append((subcluster_col_name, "concat", subcluster_col_name))


                # Quality: Inspect cluster quality
                if args.quality:
                    # Make directory
                    quality_dir = os.path.join(args.output_dir, "quality")
                    if not os.path.exists(quality_dir):
                        if args.verbose:
                            logging.info(f"Creating quality directory: {quality_dir}...")
                        os.makedirs(quality_dir, exist_ok=True)

                    # Cluster: Quality plots
                    if args.verbose:
                        logging.info(f"Generating cluster quality plots...")
                    generate_quality_plots(
                        mc=mc,
                        cluster_col_name=cluster_col_name,
                        quality_dir=quality_dir,
                        args=args,
                    )

                    # Meta-cluster: Quality plots
                    if args.sim_threshold_meta:
                        if args.verbose:
                            logging.info(f"Generating meta-cluster quality plots...")
                        generate_quality_plots(
                            mc=mc,
                            cluster_col_name=metacluster_col_name,
                            quality_dir=quality_dir,
                            args=args,
                        )

                    # Sub-cluster: Quality plots
                    if args.sim_threshold_sub:
                        if args.verbose:
                            logging.info(f"Generating sub-cluster quality plots...")
                        generate_quality_plots(
                            mc=mc,
                            cluster_col_name=subcluster_col_name,
                            quality_dir=quality_dir,
                            args=args,
                        )
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                if not args.var_chunk:
                    raise ValueError("Error building MotifCompendium object.")

                args.max_chunk = args.max_chunk - 100
                MotifCompendium.set_compute_options(max_chunk=args.max_chunk)
                logging.info(f"Retrying with max_chunk={args.max_chunk}...")
                if args.max_chunk <= 0:
                    logging.critical("Error clustering motifs. max_chunk has reached 0.")
                    raise ValueError("Error clustering motifs.")
    
    # Sort MotifCompendium object
    if args.sim_threshold_meta:
        mc.metadata["sort_cluster"] = mc.metadata[metacluster_col_name]
    elif args.sim_threshold:
        mc.metadata["sort_cluster"] = mc.metadata[cluster_col_name]
    else:
        # Cluster: Using same conditions
        if args.verbose:
            logging.info(f"Sorting averages by clusters...")
        mc.cluster(
            algorithm=ClusterArgs.algorithm,
            similarity_threshold=MotifMatchArgs.sort_threshold,
            save_name="sort_cluster",
            largest_clusters_first=True,
            **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
        )

    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_clustered)
    if args.verbose:
        logging.info(f"Saving clustered MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ### AVERAGE ------------------------------------------------------------------
    # Clusters: Average motifs
    if args.verbose:
        logging.info(f"Averaging motifs per cluster: {cluster_col_name}...")
    # Aggregate metadata columns
    aggregate_metadata = [
        (col, agg, new_col)
        for col, agg, new_col in ClusterArgs.aggregate_metadata
        if col in mc.columns()
    ]
    ClusterArgs.aggregate_metadata = aggregate_metadata
    if args.verbose:
        logging.info(f"Aggregating the following metadata columns: {ClusterArgs.aggregate_metadata}")
    while True:
        try:
            if args.time:
                start_time = time.time()
            mc_cluster = mc.cluster_averages(
                clustering=cluster_col_name,
                aggregations=ClusterArgs.aggregate_metadata,
                weight_col=ClusterArgs.weight_col,
                compute_quality_stats=args.quality,
            )
            if args.verbose:
                logging.info(
                    f"Completed averaging motifs per cluster:\n"
                    f"  Total number of clusters: {len(mc_cluster)}\n"
                    f"  Metadata columns: {mc_cluster.columns()}"
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
                logging.critical("Error averaging motifs per cluster. max_chunk has reached 0.")
                raise ValueError("Error averaging motifs per cluster.")

    # Metadata:
    # Add positive/negative column
    mc_cluster["posneg"] = utils_motif.motif_posneg_sum(mc_cluster.get_standard_motif_stack())
    # Add manual aggregation columns: avg_dist_from_summit, avg_contrib
    if "avg_dist_from_summit" in mc.columns():
        aggregate_weighted_avg(
            mc=mc,
            mc_cluster=mc_cluster,
            cluster_col=cluster_col_name,
            agg_col="avg_dist_from_summit",
            new_col="avg_dist_from_summit",
            weight_col=ClusterArgs.weight_col,
        )
    if "avg_contrib" in mc.columns():
        aggregate_weighted_avg(
            mc=mc,
            mc_cluster=mc_cluster,
            cluster_col=cluster_col_name,
            agg_col="avg_contrib",
            new_col="avg_contrib",
            weight_col=ClusterArgs.weight_col,
        )

    # Sort by cluster
    if args.sim_threshold_meta:
        mc_cluster.metadata.rename(
            columns={metacluster_col_name: "sort_cluster"},
            inplace=True,
        )
    else:
        # Cluster: Using same conditions
        if args.verbose:
            logging.info(f"Sorting averages by clusters...")
        mc_cluster.cluster(
            algorithm=ClusterArgs.algorithm,
            similarity_threshold=MotifMatchArgs.sort_threshold,
            save_name="sort_cluster",
            largest_clusters_first=True,
            **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
        )

    # Sort by positive, then negative clusters
    mc_cluster.sort(by=["posneg", "num_motifs"], ascending=False, inplace=True)

    # Save Average MotifCompendium object
    mc_cluster_path = os.path.join(args.output_dir, OutputPaths.mc_cluster)
    if args.verbose:
        logging.info(f"Saving average MotifCompendium object: {mc_cluster_path}...")
    if args.time:
        start_time = time.time()
    mc_cluster.save(mc_cluster_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Meta-clusters: Average clusters
    if args.sim_threshold_meta:
        metacluster_col_name = f"{cluster_col_name}-meta{ClusterArgs.algorithm_meta}_{args.sim_threshold_meta}{recursive_name or ''}{force_name or ''}"
        if args.verbose:
            logging.info(f"Averaging motifs per meta-cluster: {metacluster_col_name}...")
        aggregate_metadata_meta = [
            (col, agg, new_col)
            for col, agg, new_col in ClusterArgs.aggregate_metadata
            if col in mc.columns()
        ]
        if args.verbose:
            logging.info(f"Aggregating the following metadata columns: {aggregate_metadata_meta}")
        mc_metacluster = mc.cluster_averages(
            clustering=metacluster_col_name,
            aggregations=aggregate_metadata_meta,
            weight_col=ClusterArgs.weight_col,
            compute_quality_stats=args.quality,
        )
        if args.verbose:
            logging.info(
                f"Completed averaging motifs per meta-cluster:\n"
                f"  Total number of clusters: {len(mc_metacluster)}\n"
                f"  Metadata columns: {mc_metacluster.columns()}"
            )
        if args.time:
            start_time = time.time()

        # Metadata: Add positive/negative column
        mc_metacluster["posneg"] = utils_motif.motif_posneg_sum(mc_metacluster.get_standard_motif_stack())
        # Add manual aggregation columns: avg_dist_from_summit, avg_contrib
        if "avg_dist_from_summit" in mc.columns():
            aggregate_weighted_avg(
                mc=mc,
                mc_cluster=mc_metacluster,
                cluster_col=metacluster_col_name,
                agg_col="avg_dist_from_summit",
                new_col="avg_dist_from_summit",
                weight_col=ClusterArgs.weight_col,
            )
        if "avg_contrib" in mc.columns():
            aggregate_weighted_avg(
                mc=mc,
                mc_cluster=mc_metacluster,
                cluster_col=metacluster_col_name,
                agg_col="avg_contrib",
                new_col="avg_contrib",
                weight_col=ClusterArgs.weight_col,
            )

        # Sort by cluster
        if args.verbose:
            logging.info(f"Sorting averages by clusters...")
        mc_metacluster.cluster(
            algorithm=ClusterArgs.algorithm,
            similarity_threshold=MotifMatchArgs.sort_threshold,
            save_name="sort_cluster",
            largest_clusters_first=True,
            **ClusterArgs.algorithm_kwargs[ClusterArgs.algorithm],
        )

        # Sort by positive, then negative clusters
        mc_metacluster.sort(by=["posneg", "num_motifs"], ascending=False, inplace=True)

        # Save Average MotifCompendium object (meta-cluster)
        mc_metacluster_path = os.path.join(args.output_dir, OutputPaths.mc_metacluster)
        if args.verbose:
            logging.info(f"Saving average MotifCompendium object (meta-cluster): {mc_metacluster_path}...")
        if args.time:
            start_time = time.time()
        mc_metacluster.save(mc_metacluster_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    # Cluster: Sub-clusters
    if args.sim_threshold_sub:
        subcluster_col_name = f"{cluster_col_name}-sub{ClusterArgs.algorithm_sub}_{args.sim_threshold_sub}{recursive_name or ''}{force_name or ''}"
        if args.verbose:
            logging.info(f"Averaging motifs per sub-cluster: {subcluster_col_name}...")
        aggregate_metadata_sub = [
            (col, agg, new_col)
            for col, agg, new_col in ClusterArgs.aggregate_metadata
            if col in mc.columns()
        ]
        if args.verbose:
            logging.info(f"Aggregating the following metadata columns: {aggregate_metadata_sub}")
        mc_subcluster = mc.cluster_averages(
            clustering=subcluster_col_name,
            aggregations=aggregate_metadata_sub,
            weight_col=ClusterArgs.weight_col,
            compute_quality_stats=args.quality,
        )
        if args.verbose:
            logging.info(
                f"Completed averaging motifs per sub-cluster:\n"
                f"  Total number of clusters: {len(mc_subcluster)}\n"
                f"  Metadata columns: {mc_subcluster.columns()}"
            )
        if args.time:
            start_time = time.time()

        # Add positive/negative column
        mc_subcluster["posneg"] = utils_motif.motif_posneg_sum(mc_subcluster.get_standard_motif_stack())
        # Add manual aggregation columns: avg_dist_from_summit, avg_contrib
        if "avg_dist_from_summit" in mc.columns():
            aggregate_weighted_avg(
                mc=mc,
                mc_cluster=mc_subcluster,
                cluster_col=subcluster_col_name,
                agg_col="avg_dist_from_summit",
                new_col="avg_dist_from_summit",
                weight_col=ClusterArgs.weight_col,
            )
        if "avg_contrib" in mc.columns():
            aggregate_weighted_avg(
                mc=mc,
                mc_cluster=mc_subcluster,
                cluster_col=subcluster_col_name,
                agg_col="avg_contrib",
                new_col="avg_contrib",
                weight_col=ClusterArgs.weight_col,
            )

        # Sort Sub MotifCompendium object
        if args.sort_cluster:
            # Sort by cluster
            mc_subcluster.metadata.rename(
                columns={cluster_col_name: "sort_cluster"},
                inplace=True,
            )
            mc_subcluster.sort(by=["num_motifs"], ascending=False, inplace=True)
            mc_subcluster.sort(by=["sort_cluster"], ascending=True, inplace=True)
        else:
            # Sort by positive, then negative clusters
            mc_metacluster.sort(by=["posneg", "num_motifs"], ascending=False, inplace=True)

        # Save Average MotifCompendium object (sub-cluster)
        mc_subcluster_path = os.path.join(args.output_dir, OutputPaths.mc_subcluster)
        if args.verbose:
            logging.info(f"Saving average MotifCompendium object (sub-cluster): {mc_subcluster_path}...")
        if args.time:
            start_time = time.time()
        mc_subcluster.save(mc_subcluster_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    ### LABEL: CLUSTERS ---------------------------------------------------------------------
    ## Clusters: Apply labels
    if args.verbose:
        logging.info(f"Matching clusters to reference file: {args.reference}...\n"
                f"Label column: {MetadataCols.label_column_prefix}"
        )
    if args.time:
        start_time = time.time()
    label_motifs(
        mc=mc_cluster,
        reference=args.reference,
        label_col=MetadataCols.label_column_prefix,
        max_submotifs=MotifMatchArgs.max_submotifs,
        min_score=MotifMatchArgs.min_score,
        args=args,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    ## Meta-cluster: Apply labels
    if args.sim_threshold_meta:
        if args.verbose:
            logging.info(f"Matching meta-clusters to reference file: {args.reference}...\n"
                    f"Label column: {MetadataCols.label_column_prefix}"
            )
        if args.time:
            start_time = time.time()
        label_motifs(
            mc=mc_metacluster,
            reference=args.reference,
            label_col=MetadataCols.label_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
            args=args,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ## Sub-cluster: Apply filters
    if args.sim_threshold_sub:
        if args.verbose:
            logging.info(f"Matching sub-clusters to reference file: {args.reference}...\n"
                    f"Label column: {MetadataCols.label_column_prefix}"
            )
        if args.time:
            start_time = time.time()
        label_motifs(
            mc=mc_subcluster,
            reference=args.reference,
            label_col=MetadataCols.label_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
            args=args,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ## Additional labels
    if args.add_reference:
        html_addlabel_cols = []
        for reference in args.add_reference:
            label_col = os.path.splitext(os.path.basename(reference))[0]

            ## Label: Cluster
            if args.verbose:
                logging.info(f"Adding labels to clusters, from reference file: {reference}...")
            if args.time:
                start_time = time.time()
            label_motifs(
                mc=mc_cluster,
                reference=reference,
                label_col=label_col,
                max_submotifs=1,
                min_score=MotifMatchArgs.min_score,
                args=args,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            ## Label: Meta-cluster
            if args.sim_threshold_meta:
                if args.verbose:
                    logging.info(f"Adding labels to meta-clusters, from reference file: {reference}...")
                if args.time:
                    start_time = time.time()
                label_motifs(
                    mc=mc_metacluster,
                    reference=reference,
                    label_col=label_col,
                    max_submotifs=1,
                    min_score=MotifMatchArgs.min_score,
                    args=args,
                )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            # Label: Sub-cluster
            if args.sim_threshold_sub:
                if args.verbose:
                    logging.info(f"Adding labels to sub-clusters, from reference file: {reference}...")
                if args.time:
                    start_time = time.time()
                label_motifs(
                    mc=mc_subcluster,
                    reference=reference,
                    label_col=label_col,
                    max_submotifs=1,
                    min_score=MotifMatchArgs.min_score,
                    args=args,
                )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s") 

    ## Save MotifCompendium objects
    # Clusters: Save
    mc_cluster_path = os.path.join(args.output_dir, OutputPaths.mc_cluster_labeled)
    if args.verbose:
        logging.info(f"Saving labeled cluster MotifCompendium object: {mc_cluster_path}...")
    if args.time:
        start_time = time.time()
    mc_cluster.save(mc_cluster_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Meta-clusters: Save
    if args.sim_threshold_meta:
        mc_metacluster_path = os.path.join(args.output_dir, OutputPaths.mc_metacluster_labeled)
        if args.verbose:
            logging.info(f"Saving labeled meta-cluster MotifCompendium object: {mc_metacluster_path}...")
        if args.time:
            start_time = time.time()
        mc_metacluster.save(mc_metacluster_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Sub-clusters: Save
    if args.sim_threshold_sub:
        mc_subcluster_path = os.path.join(args.output_dir, OutputPaths.mc_subcluster_labeled)
        if args.verbose:
            logging.info(f"Saving labeled sub-cluster MotifCompendium object: {mc_subcluster_path}...")
        if args.time:
            start_time = time.time()
        mc_subcluster.save(mc_subcluster_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ### FILTER: CLUSTERS --------------------------------------------------------------------
    if args.filter:
        ## Cluster: Apply filters
        if args.verbose:
            logging.info(f"Applying filters to clusters...")
        mc_cluster[MetadataCols.filter_col_flag] = False
        filter_clusters(
            mc=mc_cluster,
            MotifFilterArgs=MotifFilterArgs,
            MetadataCols=MetadataCols,
            args=args,
        )

        # Remove flagged clusters
        if args.verbose:
            logging.info(f"Removing flagged clusters...")
        mc_cluster_removed = mc_cluster[mc_cluster[MetadataCols.filter_col_flag]]
        mc_cluster = mc_cluster[~mc_cluster[MetadataCols.filter_col_flag]]
        if args.verbose:
            logging.info(f"Number of clusters after removing flagged: {len(mc_cluster)}\n"
                        f"Number of clusters removed: {len(mc_cluster_removed)}")

        # Save MotifCompendium objects
        mc_cluster_path = os.path.join(args.output_dir, OutputPaths.mc_cluster_filtered)
        mc_cluster_removed_path = os.path.join(args.output_dir, OutputPaths.mc_cluster_removed)
        if args.verbose:
            logging.info(
                f"Saving filtered clusters:\n"
                f"  Filtered: {mc_cluster_path}\n"
                f"  Removed: {mc_cluster_removed_path}"
            )
        if args.time:
            start_time = time.time()
        mc_cluster.save(mc_cluster_path)
        mc_cluster_removed.save(mc_cluster_removed_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        ## Meta-cluster: Apply filters
        if args.sim_threshold_meta:
            if args.verbose:
                logging.info(f"Applying filters to meta-clusters...")
            mc_metacluster[MetadataCols.filter_col_flag] = False
            filter_clusters(
                mc=mc_metacluster,
                MotifFilterArgs=MotifFilterArgs,
                MetadataCols=MetadataCols,
                args=args,
            )
            # Remove flagged clusters
            if args.verbose:
                logging.info(f"Removing flagged meta-clusters...")
            mc_metacluster_removed = mc_metacluster[mc_metacluster[MetadataCols.filter_col_flag]]
            mc_metacluster = mc_metacluster[~mc_metacluster[MetadataCols.filter_col_flag]]
            if args.verbose:
                logging.info(f"Number of meta-clusters after removing flagged: {len(mc_metacluster)}\n"
                            f"Number of meta-clusters removed: {len(mc_metacluster_removed)}")
            # Save MotifCompendium objects
            mc_metacluster_path = os.path.join(args.output_dir, OutputPaths.mc_metacluster_filtered)
            mc_metacluster_removed_path = os.path.join(args.output_dir, OutputPaths.mc_metacluster_removed)
            if args.verbose:
                logging.info(
                    f"Saving filtered meta-clusters:\n"
                    f"  Filtered: {mc_metacluster_path}\n"
                    f"  Removed: {mc_metacluster_removed_path}"
                )
            if args.time:
                start_time = time.time()
            mc_metacluster.save(mc_metacluster_path)
            mc_metacluster_removed.save(mc_metacluster_removed_path)
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        ## Sub-cluster: Apply filters
        if args.sim_threshold_sub:
            if args.verbose:
                logging.info(f"Applying filters to sub-clusters...")
            mc_subcluster[MetadataCols.filter_col_flag] = False
            filter_clusters(
                mc=mc_subcluster,
                MotifFilterArgs=MotifFilterArgs,
                MetadataCols=MetadataCols,
                args=args,
            )

            # Remove flagged motifs
            if args.verbose:
                logging.info(f"Removing flagged sub-clusters...")
            mc_subcluster_removed = mc_subcluster[mc_subcluster[MetadataCols.filter_col_flag]]
            mc_subcluster = mc_subcluster[~mc_subcluster[MetadataCols.filter_col_flag]]
            if args.verbose:
                logging.info(f"Number of sub-clusters after removing flagged: {len(mc_subcluster)}\n"
                            f"Number of sub-clusters removed: {len(mc_subcluster_removed)}")
                
            # Save MotifCompendium objects
            mc_subcluster_path = os.path.join(args.output_dir, OutputPaths.mc_subcluster_filtered)
            mc_subcluster_removed_path = os.path.join(args.output_dir, OutputPaths.mc_subcluster_removed)
            if args.verbose:
                logging.info(
                    f"Saving filtered sub-clusters:\n"
                    f"  Filtered: {mc_subcluster_path}\n"
                    f"  Removed: {mc_subcluster_removed_path}"
                )
            if args.time:
                start_time = time.time()
            mc_subcluster.save(mc_subcluster_path)
            mc_subcluster_removed.save(mc_subcluster_removed_path)
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## VISUALIZE ----------------------------------------------------------------
    # Create HTML directory
    if args.html_motif_collection or args.html_motif_table or args.html_motif_removed or args.html_cluster_table or args.html_cluster_removed:
        # Check if HTML directory exists
        html_dir = os.path.join(args.output_dir, "html")
        if not os.path.exists(html_dir):
            if args.verbose:
                logging.info(f"Creating HTML directory: {html_dir}...")
            os.makedirs(html_dir, exist_ok=True)

    # Select HTML table columns
    if args.verbose:
        logging.info(f"Setting HTML table columns...")
    if args.reference:
        VisualizeArgs.html_table_cols_base.extend(VisualizeArgs.html_table_cols_label)
    if args.quality:
        VisualizeArgs.html_table_cols_base.extend(VisualizeArgs.html_table_cols_quality)

    ## MOTIFS
    # Visualize: Motif collection
    if args.html_motif_collection:
        # Create HTML collection
        html_motif_collection_path = os.path.join(html_dir, OutputPaths.html_motif_collection)
        if args.verbose:
            logging.info(f"Visualizing motif collection: {html_motif_collection_path}...\n"
                        f"Number of motifs: {len(mc)}")
        if args.time:
            start_time = time.time()
        mc.motif_collection_html(
            html_out=html_motif_collection_path,
            group_by=cluster_col_name,
            average_motif=True,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Visualize: Motif table
    if args.html_motif_table:
        # Check html table columns
        VisualizeArgs.html_table_cols_motif = [
            col for col in VisualizeArgs.html_table_cols_base
            if col in mc.columns() or col in mc.images()
        ]
        if args.verbose:
            logging.info(f"Visualizing the following columns for motif table: {VisualizeArgs.html_table_cols_motif}")

        # Set max_rows
        max_rows_series = pd.Series([True] * len(mc))
        if args.html_max_rows:
            html_max_rows = min(args.html_max_rows, len(mc))
            max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc) - html_max_rows), index=mc.metadata.index)
        mc_html = mc[max_rows_series]

        # Add logos
        if "logo (fwd)" not in mc_html.images():
            mc_html.add_logos(
                motifs=mc_html.get_standard_motif_stack(),
                image_name="logo (fwd)",
                trim=args.logo_trimming,
            )
        if "logo (rev)" not in mc_html.images():
            mc_html.add_logos(
                motifs=utils_motif.reverse_complement(mc_html.get_standard_motif_stack()),
                image_name="logo (rev)",
                trim=args.logo_trimming,
            )

        # Create HTML table
        html_motif_table_path = os.path.join(html_dir, OutputPaths.html_motif_table)
        if args.verbose:
            logging.info(f"Visualizing motif table: {html_motif_table_path}...\n"
                        f"Number of motifs: {len(mc)}")
        if args.time:
            start_time = time.time()
        mc_html.summary_table_html(
            html_out=html_motif_table_path,
            columns=VisualizeArgs.html_table_cols_motif,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Visualize: Motifs removed
    if args.html_motif_removed and args.filter:
        # Visualize all metadata columns
        VisualizeArgs.html_table_cols_motif_removed = mc_removed.columns() + mc_removed.images()
        if args.verbose:
            logging.info(f"Visualizing the following columns for removed motif table: {VisualizeArgs.html_table_cols_motif_removed}")

        # Set max_rows
        max_rows_series = pd.Series([True] * len(mc_removed))
        if args.html_max_rows:
            html_max_rows = min(args.html_max_rows, len(mc_removed))
            max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_removed) - html_max_rows), index=mc_removed.metadata.index)
        mc_removed_html = mc_removed[max_rows_series]

        # Add logos
        if "logo (fwd)" not in mc_removed_html.images():
            mc_removed_html.add_logos(
                motifs=mc_removed_html.get_standard_motif_stack(),
                image_name="logo (fwd)",
                trim=args.logo_trimming,
            )
        if "logo (rev)" not in mc_removed_html.images():
            mc_removed_html.add_logos(
                motifs=utils_motif.reverse_complement(mc_removed_html.get_standard_motif_stack()),
                image_name="logo (rev)",
                trim=args.logo_trimming,
            )

        # Create HTML table
        html_motif_removed_path = os.path.join(html_dir, OutputPaths.html_motif_removed)
        if args.verbose:
            logging.info(f"Visualizing removed motifs: {html_motif_removed_path}...\n"
                        f"Number of motifs removed: {len(mc_removed)}")
        if args.time:
            start_time = time.time()
        mc_removed_html.summary_table_html(
            html_out=html_motif_removed_path,
            columns=VisualizeArgs.html_table_cols_motif_removed,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ## CLUSTERS
    # Visualize: Cluster table
    if args.html_cluster_table:
        # Check html table columns
        VisualizeArgs.html_table_cols_cluster = [
            col for col in VisualizeArgs.html_table_cols_base
            if col in mc_cluster.columns() or col in mc_cluster.images()
        ]
        if args.verbose:
            logging.info(f"Visualizing the following columns for cluster table: {VisualizeArgs.html_table_cols_cluster}")

        # Set max_rows
        max_rows_series = pd.Series([True] * len(mc_cluster))
        if args.html_max_rows:
            html_max_rows = min(args.html_max_rows, len(mc_cluster))
            max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_cluster) - html_max_rows), index=mc_cluster.metadata.index)
        mc_cluster_html = mc_cluster[max_rows_series]

        # Add logos
        if "logo (fwd)" not in mc_cluster_html.images():
            mc_cluster_html.add_logos(
                motifs=mc_cluster_html.get_standard_motif_stack(),
                image_name="logo (fwd)",
                trim=args.logo_trimming,
            )
        if "logo (rev)" not in mc_cluster_html.images():
            mc_cluster_html.add_logos(
                motifs=utils_motif.reverse_complement(mc_cluster_html.get_standard_motif_stack()),
                image_name="logo (rev)",
                trim=args.logo_trimming,
            )

        # Cluster: Create HTML table
        html_cluster_table_path = os.path.join(html_dir, OutputPaths.html_cluster_table)
        if args.verbose:
            logging.info(f"Visualizing cluster table: {html_cluster_table_path}...\n"
                        f"Number of clusters: {len(mc_cluster)}")
        if args.time:
            start_time = time.time()
        mc_cluster_html.summary_table_html(
            html_out=html_cluster_table_path,
            columns=VisualizeArgs.html_table_cols_cluster,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        
        # Meta-cluster: Create HTML table
        if args.sim_threshold_meta:
            # Check html table columns
            VisualizeArgs.html_table_cols_metacluster = [
                col for col in VisualizeArgs.html_table_cols_base
                if col in mc_metacluster.columns() or col in mc_metacluster.images()
            ]
            if args.verbose:
                logging.info(f"Visualizing the following columns for meta-cluster table: {VisualizeArgs.html_table_cols_metacluster}")

            # Set max_rows
            max_rows_series = pd.Series([True] * len(mc_metacluster))
            if args.html_max_rows:
                html_max_rows = min(args.html_max_rows, len(mc_metacluster))
                max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_metacluster) - html_max_rows), index=mc_metacluster.metadata.index)
            mc_metacluster_html = mc_metacluster[max_rows_series]

            # Add logos
            if "logo (fwd)" not in mc_metacluster_html.images():
                mc_metacluster_html.add_logos(
                    motifs=mc_metacluster_html.get_standard_motif_stack(),
                    image_name="logo (fwd)",
                    trim=args.logo_trimming,
                )
            if "logo (rev)" not in mc_metacluster_html.images():
                mc_metacluster_html.add_logos(
                    motifs=utils_motif.reverse_complement(mc_metacluster_html.get_standard_motif_stack()),
                    image_name="logo (rev)",
                    trim=args.logo_trimming,
                )

            # Create HTML table
            html_metacluster_table_path = os.path.join(html_dir, OutputPaths.html_metacluster_table)
            if args.verbose:
                logging.info(f"Visualizing meta-cluster table: {html_metacluster_table_path}...\n"
                             f"Number of meta-clusters: {len(mc_metacluster)}")
            if args.time:
                start_time = time.time()
            mc_metacluster_html.summary_table_html(
                html_out=html_metacluster_table_path,
                columns=VisualizeArgs.html_table_cols_metacluster,
                editable=VisualizeArgs.editable,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        # Sub-cluster: Create HTML table
        if args.sim_threshold_sub:
            # Check html table columns
            VisualizeArgs.html_table_cols_subcluster = [
                col for col in VisualizeArgs.html_table_cols_base
                if col in mc_subcluster.columns() or col in mc_subcluster.images()
            ]
            if args.verbose:
                logging.info(f"Visualizing the following columns for sub-cluster table: {VisualizeArgs.html_table_cols_subcluster}")
            
            # Set max_rows
            max_rows_series = pd.Series([True] * len(mc_subcluster))
            if args.html_max_rows:
                html_max_rows = min(args.html_max_rows, len(mc_subcluster))
                max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_subcluster) - html_max_rows), index=mc_subcluster.metadata.index)
            mc_subcluster_html = mc_subcluster[max_rows_series]

            # Add logos
            if "logo (fwd)" not in mc_subcluster_html.images():
                mc_subcluster_html.add_logos(
                    motifs=mc_subcluster_html.get_standard_motif_stack(),
                    image_name="logo (fwd)",
                    trim=args.logo_trimming,
                )
            if "logo (rev)" not in mc_subcluster_html.images():
                mc_subcluster_html.add_logos(
                    motifs=utils_motif.reverse_complement(mc_subcluster_html.get_standard_motif_stack()),
                    image_name="logo (rev)",
                    trim=args.logo_trimming,
                )

            # Create HTML table
            html_subcluster_table_path = os.path.join(html_dir, OutputPaths.html_subcluster_table)
            if args.verbose:
                logging.info(f"Visualizing sub-cluster table: {html_subcluster_table_path}...\n"
                             f"Number of sub-clusters: {len(mc_subcluster)}")
            if args.time:
                start_time = time.time()
            mc_subcluster_html.summary_table_html(
                html_out=html_subcluster_table_path,
                columns=VisualizeArgs.html_table_cols_subcluster,
                editable=VisualizeArgs.editable,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Visualize: Clusters removed
    if args.html_cluster_removed and args.filter:
        # Visualize all metadata columns
        VisualizeArgs.html_table_cols_cluster_removed = mc_cluster_removed.columns() + mc_cluster_removed.images()
        if args.verbose:
            logging.info(f"Visualizing the following columns for removed cluster table: {VisualizeArgs.html_table_cols_cluster_removed}")

        # Set max_rows
        max_rows_series = pd.Series([True] * len(mc_cluster_removed))
        if args.html_max_rows:
            html_max_rows = min(args.html_max_rows, len(mc_cluster_removed))
            max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_cluster_removed) - html_max_rows), index=mc_cluster_removed.metadata.index)
        mc_cluster_removed_html = mc_cluster_removed[max_rows_series]

        # Add logos
        if "logo (fwd)" not in mc_cluster_removed_html.images():
            mc_cluster_removed_html.add_logos(
                motifs=mc_cluster_removed_html.get_standard_motif_stack(),
                image_name="logo (fwd)",
                trim=args.logo_trimming,
            )
        if "logo (rev)" not in mc_cluster_removed_html.images():
            mc_cluster_removed_html.add_logos(
                motifs=utils_motif.reverse_complement(mc_cluster_removed_html.get_standard_motif_stack()),
                image_name="logo (rev)",
                trim=args.logo_trimming,
            )

        # Cluster: Create HTML table
        html_cluster_removed_path = os.path.join(html_dir, OutputPaths.html_cluster_removed)
        if args.verbose:
            logging.info(f"Visualizing removed clusters: {html_cluster_removed_path}...\n"
                            f"Number of clusters removed: {len(mc_cluster_removed)}")
        if args.time:
            start_time = time.time()
        mc_cluster_removed_html.summary_table_html(
            html_out=html_cluster_removed_path,
            columns=VisualizeArgs.html_table_cols_cluster_removed,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        # Meta-cluster: Create HTML table
        if args.sim_threshold_meta:
            # Visualize all metadata columns
            VisualizeArgs.html_table_cols_metacluster_removed = mc_metacluster_removed.columns() + mc_metacluster_removed.images()
            if args.verbose:
                logging.info(f"Visualizing the following columns for removed meta-cluster table: {VisualizeArgs.html_table_cols_metacluster_removed}")

            # Set max_rows
            max_rows_series = pd.Series([True] * len(mc_metacluster_removed))
            if args.html_max_rows:
                html_max_rows = min(args.html_max_rows, len(mc_metacluster_removed))
                max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_metacluster_removed) - html_max_rows), index=mc_metacluster_removed.metadata.index)
            mc_metacluster_removed_html = mc_metacluster_removed[max_rows_series]

            # Add logos
            if "logo (fwd)" not in mc_metacluster_removed_html.images():
                mc_metacluster_removed_html.add_logos(
                    motifs=mc_metacluster_removed_html.get_standard_motif_stack(),
                    image_name="logo (fwd)",
                    trim=args.logo_trimming,
                )
            if "logo (rev)" not in mc_metacluster_removed_html.images():
                mc_metacluster_removed_html.add_logos(
                    motifs=utils_motif.reverse_complement(mc_metacluster_removed_html.get_standard_motif_stack()),
                    image_name="logo (rev)",
                    trim=args.logo_trimming,
                )

            # Create HTML table
            html_metacluster_removed_path = os.path.join(html_dir, OutputPaths.html_metacluster_removed)
            if args.verbose:
                logging.info(f"Visualizing removed meta-clusters: {html_metacluster_removed_path}...\n"
                             f"Number of meta-clusters removed: {len(mc_metacluster_removed)}")
            if args.time:
                start_time = time.time()
            mc_metacluster_removed_html.summary_table_html(
                html_out=html_metacluster_removed_path,
                columns=VisualizeArgs.html_table_cols_metacluster_removed,
                editable=VisualizeArgs.editable,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
            
        # Sub-cluster: Create HTML table
        if args.sim_threshold_sub:
            # Visualize all metadata columns
            VisualizeArgs.html_table_cols_subcluster_removed = mc_subcluster_removed.columns() + mc_subcluster_removed.images()
            if args.verbose:
                logging.info(f"Visualizing the following columns for removed cluster table: {VisualizeArgs.html_table_cols_subcluster_removed}")

            # Set max_rows
            max_rows_series = pd.Series([True] * len(mc_subcluster_removed))
            if args.html_max_rows:
                html_max_rows = min(args.html_max_rows, len(mc_subcluster_removed))
                max_rows_series = pd.Series([True] * html_max_rows + [False] * (len(mc_subcluster_removed) - html_max_rows), index=mc_subcluster_removed.metadata.index)
            mc_subcluster_removed_html = mc_subcluster_removed[max_rows_series]

            # Add logos
            if "logo (fwd)" not in mc_subcluster_removed_html.images():
                mc_subcluster_removed_html.add_logos(
                    motifs=mc_subcluster_removed_html.get_standard_motif_stack(),
                    image_name="logo (fwd)",
                    trim=args.logo_trimming,
                )
            if "logo (rev)" not in mc_subcluster_removed_html.images():
                mc_subcluster_removed_html.add_logos(
                    motifs=utils_motif.reverse_complement(mc_subcluster_removed_html.get_standard_motif_stack()),
                    image_name="logo (rev)",
                    trim=args.logo_trimming,
                )

            # Create HTML table
            html_subcluster_removed_path = os.path.join(html_dir, OutputPaths.html_subcluster_removed)
            if args.verbose:
                logging.info(f"Visualizing removed sub-clusters: {html_subcluster_removed_path}...\n"
                             f"Number of sub-clusters removed: {len(mc_subcluster_removed)}")
            if args.time:
                start_time = time.time()
            mc_subcluster_removed_html.summary_table_html(
                html_out=html_subcluster_removed_path,
                columns=VisualizeArgs.html_table_cols_subcluster_removed,
                editable=VisualizeArgs.editable,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## ARGUMENTS ---------------------------------------------------------------
    # Save arguments
    arg_json = os.path.join(args.output_dir, "args.json")
    if args.verbose:
        logging.info(f"Saving arguments to: {arg_json}...")
    all_args = [
        {"args": vars(args)},
        {"OutputPaths": vars(OutputPaths)},
        {"MotifMatchArgs": vars(MotifMatchArgs)},
        {"ClusterArgs": vars(ClusterArgs)},
        {"MetadataCols": vars(MetadataCols)},
        {"MotifFilterArgs": MotifFilterArgs.to_dict()},
        {"VisualizeArgs": vars(VisualizeArgs)},
    ]
    with open(arg_json, "w") as f:
        json.dump(all_args, f, indent=4)

    ## END ----------------------------------------------------------------
    print(f"MotifCompendium pipeline completed successfully. Output saved to {args.output_dir}.")