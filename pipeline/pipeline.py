import os
import sys
import time
import argparse
import logging

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


def setup_parser():
    parser = argparse.ArgumentParser(description="Run the MotifCompendium pipeline.")

    parser.add_argument("-im", "--input-mc", type=str, default=None, help="Path to the input MotifCompendium object.")
    parser.add_argument("-io", "--input-old-mc", type=str, default=None, help="Path to the input old MotifCompendium object.")
    parser.add_argument("-ih", "--input-h5s", nargs="+", type=str, default=None, help="Path to the input Modisco H5 file(s).")
    parser.add_argument("-nh", "--input-names", nargs="+", type=str, default=None, help="Nickname of the input Modisco H5 file(s).")

    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, per h5 or motif: CSV, TSV format.")
    parser.add_argument("-r", "--reference", type=str, default=None, help="Path to the main reference motif file: MotifCompendium object, or PFM, MEME .txt format.")
    parser.add_argument("--add-reference", nargs="+", type=str, default=None, help="Path to additional reference motif files for final labeling: MotifCompendium object, or PFM, MEME .txt format.")
    
    parser.add_argument("--min-seqlets", type=int, default=100, help="Minimum number of seqlets to consider a motif.")
    parser.add_argument("--first-posmotif", action="store_true", help="Guarantee the first positive motif is always included in the MotifCompendium object.")
    parser.add_argument("--first-posmotif-only", action="store_true", help="Only include the first positive motif in the MotifCompendium object.")
    parser.add_argument("--strict-filter", action="store_true", help="Apply a strict filter, that does not add back motifs even when matched with a database.")
    parser.add_argument("--rm-singletons", action="store_true", help="Remove singletons from the averaged MotifCompendium object.")

    parser.add_argument("--sim-scan", nargs="+", type=float, default=None, help="List of similarity thresholds to scan during clustering. MUST INCLUDE SIM_THRESHOLD.")
    parser.add_argument("--sim-threshold", type=float, default=0.9, help="Similarity threshold to apply during clustering.")
    parser.add_argument("--sim-threshold2", type=float, default=None, help="Second similarity threshold to apply during sub-clustering. If not provided, will not create sub-clusters.")
    parser.add_argument("--cluster-by-composite", action="store_true", help="Cluster base motifs and composite motifs separately.")
    parser.add_argument("--quality", action="store_true", help="Calculate quality metrics and plots for clustering.")
    
    parser.add_argument("--html-motif-collection", action="store_true", help="Generate HTML collection of motif constituents per cluster.")
    parser.add_argument("--html-motif-table", action="store_true", help="Generate HTML summary table of individual motifs.")
    parser.add_argument("--html-motif-removed", action="store_true", help="Generate HTML summary table of removed motifs.")
    parser.add_argument("--html-cluster-table", action="store_true", help="Generate HTML summary table of clusters.")
    parser.add_argument("--html-cluster-removed", action="store_true", help="Generate HTML summary table of removed clusters.")
    parser.add_argument("--html-subcluster-table", action="store_true", help="Generate HTML summary table of sub-clusters.")
    parser.add_argument("--html-subcluster-removed", action="store_true", help="Generate HTML summary table of removed sub-clusters.")
    
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not compute information content.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Disable safety checks.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("--fast-plot", action="store_true", help="Use fast plotting.")
    parser.add_argument("--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


## DEFAULT OPTIONS ------------------------------------------------------------
def set_default_options():
    import configs

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


## FUNCTIONS -------------------------------------------------------------------
def build_modisco_dict(h5_list: list, name_list: list | None) -> dict:
    if name_list is None:
        name_list = [os.path.basename(h5) for h5 in h5_list]
    modisco_dict = dict(zip(name_list, h5_list))
    return modisco_dict

def apply_filter_threshold(
    mc: MotifCompendium,
    flag_col: str,
    metric: str,
    operation: str,
    threshold: float,
    override: bool,
) -> None:
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
        else:
            raise ValueError("Invalid operation for filter threshold.")


### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()

    ## OPTIONS -----------------------------------------------------------------
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


    ## BUILD --------------------------------------------------------------------
    # Check input
    if args.verbose:
        logging.info("Checking input paths...")

    # Load the input MotifCompendium object
    if args.input_mc:
        if not os.path.exists(args.input_mc):
            logging.error(f"Input MotifCompendium object not found: {args.input_mc}")
            raise FileNotFoundError(f"Input MotifCompendium object not found: {args.input_mc}")
        if args.verbose:
            logging.info(f"Loading input MotifCompendium object: {args.input_mc}...")
        if args.time:
            start_time = time.time()
        mc = MotifCompendium.load(args.input_mc, safe=args.safe)
        if args.verbose:
            logging.info(
                f"Completed loading MotifCompendium object:\n"
                f"  Total number of motifs: {len(mc)}\n"
                f"  Metadata columns: {mc.metadata.columns.tolist()}"
            )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Load the input old MotifCompendium object
    elif args.input_old_mc:
        if not os.path.exists(args.input_old_mc):
            logging.error(f"Input old MotifCompendium object not found: {args.input_old_mc}")
            raise FileNotFoundError(f"Input old MotifCompendium object not found: {args.input_old_mc}")
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
                f"  Metadata columns: {mc.metadata.columns.tolist()}"
            )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Load the input H5 files
    elif args.input_h5s:
        for h5 in args.input_h5s:
            if not os.path.exists(h5):
                logging.error(f"Input H5 file not found: {h5}")
                raise FileNotFoundError(f"Input H5 file not found: {h5}")
    
        modisco_dict = build_modisco_dict(args.input_h5s, args.input_names)
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
                    ic=args.ic,
                    safe=args.safe,
                )
                if args.verbose:
                    logging.info(
                        f"Completed building MotifCompendium object:\n"
                        f"  Total number of motifs: {len(mc)}\n"
                        f"  Metadata columns: {mc.metadata.columns.tolist()}"
                    )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")
                break

            except Exception as e:
                logging.error(f"Error: {e}")
                args.max_chunk = args.max_chunk - 100
                MotifCompendium.set_compute_options(max_chunk=args.max_chunk)
                logging.info(f"Retrying with max_chunk={args.max_chunk}...")
                if args.max_chunk <= 0:
                    logging.critical("Error building MotifCompendium object. max_chunk has reached 0.")
                    raise ValueError("Error building MotifCompendium object.")
                
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

        # Update metadata aggregation
        new_aggregate_metadata = []
        for col in metadata_df.columns:
            if col not in mc.metadata.columns:
                new_aggregate_metadata.append((col, "concat", f"{col}"))
        ClusterArgs.aggregate_metadata.extend(new_aggregate_metadata)

        # Update HTML table columns
        VisualizeArgs.html_motif_table_cols.extend([f"{col}" for col in metadata_df.columns])
        VisualizeArgs.html_cluster_table_cols.extend([f"{col}" for col in metadata_df.columns])

    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_full)
    if args.verbose:
        logging.info(f"Saving MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## FILTER: MODISCO ----------------------------------------------------------
    mc[MetadataCols.filter_col_flag] = False

    # Filter #1: Reference matching
    if args.reference is None:
        args.reference = MotifMatchArgs.reference_default

    if args.reference.endswith(".mc"):
        if args.verbose:
            logging.info(f"Matching motifs to reference MotifCompendium object: {args.reference}...")
        if args.time:
            start_time = time.time()
        ref_mc = MotifCompendium.load(args.reference, safe=args.safe)
        mc.assign_label_from_other(
            other=ref_mc,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    elif args.reference.endswith("pfm.txt") or args.reference.endswith("meme.txt") or args.reference.endswith(".meme"):
        if args.verbose:
            logging.info(f"Matching motifs to reference file: {args.reference}...")
        if args.time:
            start_time = time.time()
        utils_analysis.assign_label_from_pfm(
            mc=mc,
            pfm_file=args.reference,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    else:
        logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
        raise ValueError(
            "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
        )

    # Filter #2: Calculate and apply filter metrics
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
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    if args.verbose:
        logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")

    # Filter #3: Flag motifs with less than min_seqlets
    if args.verbose:
        logging.info(f"Flagging motifs with less than {args.min_seqlets} seqlets...")
    apply_filter_threshold(
        mc=mc,
        flag_col=MetadataCols.filter_col_flag,
        metric="num_seqlets",
        operation="<",
        threshold=args.min_seqlets,
        override=False,
    )
    if args.verbose:
        logging.info(f"Total number of motifs less than {args.min_seqlets} seqlets: {len(mc[mc['num_seqlets'] < args.min_seqlets])}")

    # Filter #4: Override flag: Good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for addback_filter_args in MotifFilterArgs.override_filters:
        if addback_filter_args.apply_motif:
            apply_filter_threshold(
                mc=mc,
                flag_col=MetadataCols.filter_col_flag,
                metric=addback_filter_args.metric,
                operation=addback_filter_args.operation,
                threshold=addback_filter_args.threshold,
                override=addback_filter_args.override,
            )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    if args.verbose:
        logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")

    # Filter #5: Override flag: First motif
    if args.first_posmotif:
        if args.verbose:
            logging.info(f"Overriding flags for first positive motif...")
        mc[MetadataCols.filter_col_flag] = (mc[MetadataCols.filter_col_flag] & 
                                                        ((~mc["name"].str.contains("pattern_0")) |
                                                        (mc["posneg"] != "pos")))
    if args.verbose:
        logging.info(f"Total number of first positive motifs: {len(mc[mc['name'].str.contains('pattern_0') & (mc['posneg'] == 'pos')])}")

    # Filter #6: Apply strict filters
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
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        if args.verbose:
            logging.info(f"Total number of motifs flagged: {len(mc[mc[MetadataCols.filter_col_flag]])}")

    # Filter #7: Select first positive motif only
    if args.first_posmotif_only:
        if args.verbose:
            logging.info(f"Selecting first positive motif only...")
        mc[MetadataCols.filter_col_flag] = True
        mc[MetadataCols.filter_col_flag] = (mc[MetadataCols.filter_col_flag] & 
                                                        ((~mc["name"].str.contains("pattern_0")) |
                                                        (mc["posneg"] != "pos")))

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


    ## CLUSTERING ---------------------------------------------------------------
    # Set options
    if args.sim_scan:
        sim_thresholds = args.sim_scan
    else:
        sim_thresholds = [args.sim_threshold]
    
    if args.cluster_by_composite:
        # Split by # composite motifs, with best score
        mc["num_composites"] = 0
        mc["num_composites"] = mc[
            [f"{MetadataCols.match_column_prefix}_score{i}" 
             for i in range(MotifMatchArgs.max_submotifs)]].apply(
                 lambda row: max(
                     [i for i, val in enumerate(row) 
                      if val > MotifMatchArgs.composite_threshold], default=0), axis=1)
            
        mc_list = []
        for i in range(MotifMatchArgs.max_submotifs):
            mc_iter = mc[mc["num_composites"] == i]
            if len(mc_iter) > 0:
                mc_list.append(mc_iter)

    # Cluster motifs
    for sim_threshold in sim_thresholds:
        cluster_col_name = f"{ClusterArgs.algorithm}_{sim_threshold}"
        mc[cluster_col_name] = 0
        last_cluster = 0
        if args.verbose:
            logging.info(f"Clustering motifs using: {ClusterArgs.algorithm} {sim_threshold}...")
        if args.time:
            start_time = time.time()

        if args.cluster_by_composite:
            for mc_iter in mc_list:
                # Cluster motifs
                mc_iter.cluster(
                    algorithm=ClusterArgs.algorithm,
                    similarity_threshold=sim_threshold,
                    save_name=cluster_col_name,
                )

                # Add last_cluster, to keep membership unique
                mc_iter[cluster_col_name] += last_cluster

                # Update cluster membership back to full MotifCompendium object
                mc.metadata = mc.metadata.merge(
                    mc_iter[["name", cluster_col_name]],
                    on="name",
                    how="left",
                    suffixes=("", "_drop"),
                )
                mc[f"{cluster_col_name}_drop"] = mc[f"{cluster_col_name}_drop"].fillna(0)
                mc[cluster_col_name] += mc[f"{cluster_col_name}_drop"]
                mc.metadata = mc.metadata.drop(columns=[f"{cluster_col_name}_drop"])

                last_cluster = mc[cluster_col_name].max() + 1
        
        else:
            # Cluster motifs
            mc.cluster(
                algorithm=ClusterArgs.algorithm,
                similarity_threshold=sim_threshold,
                save_name=cluster_col_name,
            )
        
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        if args.verbose:
            logging.info(f"Total number of clusters ({cluster_col_name}): {len(mc[cluster_col_name].unique())}")

        # Sub-cluster motifs
        if args.sim_threshold2:
            subcluster_col_name = f"{ClusterArgs.algorithm}_{sim_threshold}_{args.sim_threshold2}"
            if args.verbose:
                logging.info(f"Sub-clustering motifs within {cluster_col_name} clusters using: {ClusterArgs.algorithm} {args.sim_threshold2}...")
            if args.time:
                start_time = time.time()
            mc.cluster(
                algorithm=ClusterArgs.algorithm,
                similarity_threshold=args.sim_threshold2,
                cluster_within=cluster_col_name,
                save_name=subcluster_col_name,
            )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
            
            if args.verbose:
                logging.info(f"Total number of sub-clusters ({subcluster_col_name}): {len(mc[subcluster_col_name].unique())}")


        # Summarize cluster quality
        if args.quality:
            # Make directory
            quality_dir = os.path.join(args.output_dir, "quality")
            if not os.path.exists(quality_dir):
                if args.verbose:
                    logging.info(f"Creating quality directory: {quality_dir}...")
                os.makedirs(quality_dir, exist_ok=True)

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
            heatmap_path = os.path.join(quality_dir, f"heatmap_{cluster_col_name}.png")
            if args.verbose:
                logging.info(f"Summarizing cluster quality (Heatmap): {heatmap_path}...")
            if args.time:
                start_time = time.time()
            mc.heatmap(
                sort_by=cluster_col_name,
                save_loc=heatmap_path,
            )
            plt.savefig(heatmap_path)
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
            
            # Repeat for sub-clusters
            if args.sim_threshold2:
                # Quality: Histogram (sub-cluster)
                histogram_path = os.path.join(quality_dir, f"histogram_{subcluster_col_name}.png")
                if args.verbose:
                    logging.info(f"Summarizing sub-cluster quality (Histogram): {histogram_path}...")
                if args.time:
                    start_time = time.time()
                utils_analysis.judge_clustering(
                    mc=mc,
                    cluster_col=subcluster_col_name,
                    save_loc=histogram_path,
                )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

                # Quality: Heatmap (sub-cluster)
                heatmap_path = os.path.join(quality_dir, f"heatmap_{subcluster_col_name}.png")
                if args.verbose:
                    logging.info(f"Summarizing sub-cluster quality (Heatmap): {heatmap_path}...")
                if args.time:
                    start_time = time.time()
                mc.heatmap(
                    sort_by=subcluster_col_name,
                    save_loc=heatmap_path,
                )
                plt.savefig(heatmap_path)
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            # Quality: Add HTML columns
            if args.verbose:
                logging.info(f"Adding HTML columns for cluster quality...")
            VisualizeArgs.html_motif_table_cols = [
                "max_external_similarity", "max_external_sim_logo",
                "min_internal_similarity", "min_internal_sim_logo1",
                "min_internal_sim_logo2"] + VisualizeArgs.html_motif_table_cols


    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_clustered)
    if args.verbose:
        logging.info(f"Saving clustered MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## AVERAGE ------------------------------------------------------------------
    cluster_col_name = f"{ClusterArgs.algorithm}_{args.sim_threshold}"

    # Check aggregation metadata columns
    ClusterArgs.aggregate_metadata = [
        (col, agg, new_col)
        for col, agg, new_col in ClusterArgs.aggregate_metadata
        if col in mc.metadata.columns
    ]
    if args.verbose:
        logging.info(f"Aggregating the following metadata columns: {ClusterArgs.aggregate_metadata}")

    # Average: Cluster motifs    
    if args.verbose:
        logging.info(f"Averaging motifs per cluster...")
    while True:
        try:
            if args.time:
                start_time = time.time()
            mc_avg = mc.cluster_averages(
                cluster_col=cluster_col_name,
                aggregations=ClusterArgs.aggregate_metadata,
                weight_col=ClusterArgs.weight_col,
                compute_quality_stats=args.quality,
            )
            if args.verbose:
                logging.info(
                    f"Completed averaging motifs per cluster:\n"
                    f"  Total number of clusters: {len(mc_avg)}\n"
                    f"  Metadata columns: {mc_avg.metadata.columns.tolist()}"
                )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
            break
        
        except Exception as e:
            logging.error(f"Error: {e}")
            args.max_chunk = args.max_chunk - 100
            MotifCompendium.set_compute_options(max_chunk=args.max_chunk)
            logging.info(f"Retrying with max_chunk={args.max_chunk}...")
            if args.max_chunk <= 0:
                logging.critical("Error averaging motifs per cluster. max_chunk has reached 0.")
                raise ValueError("Error averaging motifs per cluster.")
    
    # Add positive/negative column
    mc_avg["posneg"] = utils_motif.motif_posneg_sum(mc_avg.get_standard_motif_stack())
    
    # Sort by positive, then negative clusters
    mc_avg.sort(by=["posneg","num_motifs"], ascending=False, inplace=True)

    # Save Average MotifCompendium object
    mc_avg_path = os.path.join(args.output_dir, OutputPaths.mc_avg)
    if args.verbose:
        logging.info(f"Saving average MotifCompendium object: {mc_avg_path}...")
    if args.time:
        start_time = time.time()
    mc_avg.save(mc_avg_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Repeat for sub-clusters
    if args.sim_threshold2:
        # Average: Cluster motifs (sub-cluster)
        subcluster_col_name = f"{ClusterArgs.algorithm}_{args.sim_threshold}_{args.sim_threshold2}"
        if args.verbose:
            logging.info(f"Averaging motifs per sub-cluster...")
        mc_subavg = mc.cluster_averages(
            cluster_col=subcluster_col_name,
            aggregations=ClusterArgs.aggregate_metadata,
            weight_col=ClusterArgs.weight_col,
            compute_quality_stats=args.quality,
        )
        if args.verbose:
            logging.info(
                f"Completed averaging motifs per sub-cluster:\n"
                f"  Total number of clusters: {len(mc_subavg)}\n"
                f"  Metadata columns: {mc_subavg.metadata.columns.tolist()}"
            )
        if args.time:
            start_time = time.time()

        # Add positive/negative column
        mc_subavg["posneg"] = utils_motif.motif_posneg_sum(mc_subavg.get_standard_motif_stack())
        
        # Save Average MotifCompendium object (sub-cluster)
        mc_subavg_path = os.path.join(args.output_dir, OutputPaths.mc_subavg)
        if args.verbose:
            logging.info(f"Saving average MotifCompendium object (sub-cluster): {mc_subavg_path}...")
        mc_subavg.save(mc_subavg_path)


    ## FILTER: CLUSTERS --------------------------------------------------------------------
    mc_avg[MetadataCols.filter_col_flag] = False

    # Filter #1: Reference matching
    if args.reference.endswith(".mc"):
        if args.verbose:
            logging.info(f"Matching motifs to reference MotifCompendium object: {args.reference}...")
        mc_avg.assign_label_from_other(
            other=ref_mc,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    elif args.reference.endswith("pfm.txt") or args.reference.endswith("meme.txt") or args.reference.endswith(".meme"):
        if args.verbose:
            logging.info(f"Matching motifs to reference file: {args.reference}...")
        utils_analysis.assign_label_from_pfm(
            mc=mc_avg,
            pfm_file=args.reference,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    else:
        logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
        raise ValueError(
            "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
        )

    # Filter #2: Calculate and flag filters
    if args.verbose:
        logging.info(f"Calculating filter metrics:\n"
            f"  {MotifFilterArgs.motif_metrics}"
        )
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc_avg,
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
                mc=mc_avg,
                flag_col=MetadataCols.filter_col_flag,
                metric=filter_args.metric,
                operation=filter_args.operation,
                threshold=filter_args.threshold,
                override=filter_args.override,
            )
    if args.verbose:
        logging.info(f"Number of flagged motifs: {len(mc_avg[mc_avg[MetadataCols.filter_col_flag]])}")
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #3: Flag singleton clusters
    if args.rm_singletons:
        if args.verbose:
            logging.info(f"Flagging singleton clusters...")
        mc_avg[MetadataCols.filter_col_flag] = mc_avg["num_motifs"] == 1
        if args.verbose:
            logging.info(f"Number of singleton clusters: {len(mc_avg[mc_avg['num_motifs'] == 1])}")

    # Filter #4: Override flags, for good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for addback_filter_args in MotifFilterArgs.override_filters:
        if addback_filter_args.apply_cluster:
            apply_filter_threshold(
                mc=mc_avg,
                flag_col=MetadataCols.filter_col_flag,
                metric=addback_filter_args.metric,
                operation=addback_filter_args.operation,
                threshold=addback_filter_args.threshold,
                override=addback_filter_args.override,
            )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

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
            if strict_filter_args.apply_cluster:
                apply_filter_threshold(
                    mc=mc_avg,
                    flag_col=MetadataCols.filter_col_flag,
                    metric=strict_filter_args.metric,
                    operation=strict_filter_args.operation,
                    threshold=strict_filter_args.threshold,
                    override=strict_filter_args.override,
                )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Remove flagged motifs
    if args.verbose:
        logging.info(f"Removing flagged motifs...")
    mc_avg_removed = mc_avg[mc_avg[MetadataCols.filter_col_flag]]
    mc_avg = mc_avg[~mc_avg[MetadataCols.filter_col_flag]]
    if args.verbose:
        logging.info(f"Number of motifs after removing flagged motifs: {len(mc_avg)}\n"
                     f"Number of motifs removed: {len(mc_avg_removed)}")

    # Save MotifCompendium objects
    mc_avg_path = os.path.join(args.output_dir, OutputPaths.mc_avg_filtered)
    mc_avg_removed_path = os.path.join(args.output_dir, OutputPaths.mc_avg_removed)
    if args.verbose:
        logging.info(
            f"Saving filtered MotifCompendium objects:\n"
            f"  Filtered: {mc_avg_path}\n"
            f"  Removed: {mc_avg_removed_path}"
        )
    if args.time:
        start_time = time.time()
    mc_avg.save(mc_avg_path)
    mc_avg_removed.save(mc_avg_removed_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ## FILTER: SUB-CLUSTERS ---------------------------------------------------------------
    if args.sim_threshold2:
        mc_subavg[MetadataCols.filter_col_flag] = False
        
        # Filter #1: Reference matching
        if args.reference.endswith(".mc"):
            if args.verbose:
                logging.info(f"Matching motifs to reference MotifCompendium object: {args.reference}...")
            mc_subavg.assign_label_from_other(
                other=ref_mc,
                save_column_prefix=MetadataCols.match_column_prefix,
                max_submotifs=MotifMatchArgs.max_submotifs,
                min_score=MotifMatchArgs.min_score,
            )
        elif args.reference.endswith("pfm.txt") or args.reference.endswith("meme.txt") or args.reference.endswith(".meme"):
            if args.verbose:
                logging.info(f"Matching motifs to reference file: {args.reference}...")
            utils_analysis.assign_label_from_pfm(
                mc=mc_subavg,
                pfm_file=args.reference,
                save_column_prefix=MetadataCols.match_column_prefix,
                max_submotifs=MotifMatchArgs.max_submotifs,
                min_score=MotifMatchArgs.min_score,
            )
        else:
            logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
            raise ValueError(
                "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
            )
        
        # Filter #2: Calculate and flag filters
        if args.verbose:
            logging.info(f"Calculating filter metrics:\n"
                f"  {MotifFilterArgs.motif_metrics}"
            )
        if args.time:
            start_time = time.time()
        utils_analysis.calculate_filters(
            mc=mc_subavg,
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
                    mc=mc_subavg,
                    flag_col=MetadataCols.filter_col_flag,
                    metric=filter_args.metric,
                    operation=filter_args.operation,
                    threshold=filter_args.threshold,
                    override=filter_args.override,
                )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        if args.verbose:
            logging.info(f"Number of flagged motifs: {len(mc_subavg[mc_subavg[MetadataCols.filter_col_flag]])}")

        # Filter #3: Flag singleton clusters
        if args.rm_singletons:
            if args.verbose:
                logging.info(f"Flagging singleton clusters...")
            mc_subavg[MetadataCols.filter_col_flag] = mc_subavg["num_motifs"] == 1
            if args.verbose:
                logging.info(f"Number of singleton clusters: {len(mc_subavg[mc_subavg['num_motifs'] == 1])}")
        
        # Filter #4: Override flags, for good matches
        if args.verbose:
            logging.info(
                f"Overriding flags for matches above threshold:\n"
                f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
            )
        if args.time:
            start_time = time.time()
        for addback_filter_args in MotifFilterArgs.override_filters:
            if addback_filter_args.apply_cluster:
                apply_filter_threshold(
                    mc=mc_subavg,
                    flag_col=MetadataCols.filter_col_flag,
                    metric=addback_filter_args.metric,
                    operation=addback_filter_args.operation,
                    threshold=addback_filter_args.threshold,
                    override=addback_filter_args.override,
                )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        
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
                if strict_filter_args.apply_cluster:
                    apply_filter_threshold(
                        mc=mc_subavg,
                        flag_col=MetadataCols.filter_col_flag,
                        metric=strict_filter_args.metric,
                        operation=strict_filter_args.operation,
                        threshold=strict_filter_args.threshold,
                        override=strict_filter_args.override,
                    )
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")
        
        # Remove flagged motifs
        if args.verbose:
            logging.info(f"Removing flagged motifs...")
        mc_subavg_removed = mc_subavg[mc_subavg[MetadataCols.filter_col_flag]]
        mc_subavg = mc_subavg[~mc_subavg[MetadataCols.filter_col_flag]]
        if args.verbose:
            logging.info(f"Number of motifs after removing flagged motifs: {len(mc_subavg)}\n"
                        f"Number of motifs removed: {len(mc_subavg_removed)}")
            
        # Save MotifCompendium objects
        mc_subavg_path = os.path.join(args.output_dir, OutputPaths.mc_subavg_filtered)
        mc_subavg_removed_path = os.path.join(args.output_dir, OutputPaths.mc_subavg_removed)
        if args.verbose:
            logging.info(
                f"Saving filtered MotifCompendium objects:\n"
                f"  Filtered: {mc_subavg_path}\n"
                f"  Removed: {mc_subavg_removed_path}"
            )
        if args.time:
            start_time = time.time()
        mc_subavg.save(mc_subavg_path)
        mc_subavg_removed.save(mc_subavg_removed_path)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## LABEL -------------------------------------------------------------------
    if args.add_reference:
        # Label: Average 
        if args.verbose:
            logging.info(f"Adding labels from additional reference files: {args.add_reference}...")
        html_label_cols = []
        for reference in args.add_reference:
            label_col = os.path.splitext(os.path.basename(reference))[0]
            if reference.endswith(".mc"):
                if args.verbose:
                    logging.info(f"Matching motifs to reference MotifCompendium object: {reference}...")
                    logging.info(f"Label column: {label_col}")
                if args.time:
                    start_time = time.time()
                ref_mc = MotifCompendium.load(reference, safe=args.safe)
                mc_avg.assign_label_from_other(
                    other=ref_mc,
                    save_column_prefix=label_col,
                    max_submotifs=1,
                    min_score=MotifMatchArgs.min_score,
                )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            elif reference.endswith("pfm.txt") or reference.endswith("meme.txt") or reference.endswith(".meme"):
                if args.verbose:
                    logging.info(f"Matching motifs to reference file: {reference}...")
                    logging.info(f"Label column: {label_col}")
                if args.time:
                    start_time = time.time()
                utils_analysis.assign_label_from_pfm(
                    mc=mc_avg,
                    pfm_file=reference,
                    save_column_prefix=label_col,
                    max_submotifs=1,
                    min_score=MotifMatchArgs.min_score,
                )
                if args.time:
                    logging.info(f"Time taken: {time.time() - start_time:.2f}s")

            else:
                logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
                raise ValueError(
                    "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
                )
            
            # Add label column to list
            html_label_cols.extend([f"{label_col}_logo0", f"{label_col}_name0", f"{label_col}_score0"])
        
        # Update HTML table columns
        if args.verbose:
            logging.info(f"Adding the following columns to HTML table: {html_label_cols}")
        VisualizeArgs.html_cluster_table_cols = html_label_cols + VisualizeArgs.html_cluster_table_cols
        
        # Label: Sub-average
        if args.sim_threshold2:
            if args.verbose:
                logging.info(f"Adding labels from additional reference files: {args.add_reference}...")
            for reference in args.add_reference:
                label_col = os.path.splitext(os.path.basename(reference))[0]
                if reference.endswith(".mc"):
                    if args.verbose:
                        logging.info(f"Matching motifs to reference MotifCompendium object: {reference}...")
                        logging.info(f"Label column: {label_col}")
                    if args.time:
                        start_time = time.time()
                    ref_mc = MotifCompendium.load(reference, safe=args.safe)
                    mc_subavg.assign_label_from_other(
                        other=ref_mc,
                        save_column_prefix=label_col,
                        max_submotifs=1,
                        min_score=MotifMatchArgs.min_score,
                    )
                    if args.time:
                        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

                elif reference.endswith("pfm.txt") or reference.endswith("meme.txt") or reference.endswith(".meme"):
                    if args.verbose:
                        logging.info(f"Matching motifs to reference file: {reference}...")
                        logging.info(f"Label column: {label_col}")
                    if args.time:
                        start_time = time.time()
                    utils_analysis.assign_label_from_pfm(
                        mc=mc_subavg,
                        pfm_file=reference,
                        save_column_prefix=label_col,
                        max_submotifs=1,
                        min_score=MotifMatchArgs.min_score,
                    )
                    if args.time:
                        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

                else:
                    logging.error("Reference file must be a MotifCompendium object or PFM, MEME .txt file.")
                    raise ValueError(
                        "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
                    )


    ## VISUALIZE ----------------------------------------------------------------
    # Create HTML directory
    if args.html_motif_collection or args.html_motif_table or args.html_motif_removed or args.html_cluster_table or args.html_cluster_removed:
        # Check if HTML directory exists
        html_dir = os.path.join(args.output_dir, "html")
        if not os.path.exists(html_dir):
            if args.verbose:
                logging.info(f"Creating HTML directory: {html_dir}...")
            os.makedirs(html_dir, exist_ok=True)
    
    ## MOTIFS
    # Visualize: Motif collection
    if args.html_motif_collection:
        # Create HTML collection
        html_motif_collection_path = os.path.join(html_dir, OutputPaths.html_motif_collection)
        if args.verbose:
            logging.info(f"Visualizing motif collection: {html_motif_collection_path}...")
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
        VisualizeArgs.html_motif_table_cols = [
            col for col in VisualizeArgs.html_motif_table_cols
            if col in mc.metadata.columns or col in mc.get_images_columns()
        ]
        if args.verbose:
            logging.info(f"Visualizing the following columns for motif table: {VisualizeArgs.html_motif_table_cols}")

        # Create HTML table
        html_motif_table_path = os.path.join(html_dir, OutputPaths.html_motif_table)
        if args.verbose:
            logging.info(f"Visualizing motif table: {html_motif_table_path}...")
        if args.time:
            start_time = time.time()        
        mc.summary_table_html(
            html_out=html_motif_table_path,
            columns=VisualizeArgs.html_motif_table_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Visualize: Motifs removed
    if args.html_motif_removed:
        # Visualize all metadata columns
        html_motif_removed_cols = mc_removed.metadata.columns.tolist() + mc_removed.get_images_columns()
        if args.verbose:
            logging.info(f"Visualizing the following columns for removed motif table: {html_motif_removed_cols}")

        # Create HTML table
        html_motif_removed_path = os.path.join(html_dir, OutputPaths.html_motif_removed)
        if args.verbose:
            logging.info(f"Visualizing removed motifs: {html_motif_removed_path}...")
        if args.time:
            start_time = time.time()        
        mc_removed.summary_table_html(
            html_out=html_motif_removed_path,
            columns=html_motif_removed_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    ## CLUSTERS
    # Visualize: Cluster table
    if args.html_cluster_table:
        # Check html table columns
        VisualizeArgs.html_cluster_table_cols = [
            col for col in VisualizeArgs.html_cluster_table_cols
            if col in mc_avg.metadata.columns or col in mc_avg.get_images_columns()
        ]
        if args.verbose:
            logging.info(f"Visualizing the following columns for cluster table: {VisualizeArgs.html_cluster_table_cols}")

        # Create HTML table
        html_cluster_table_path = os.path.join(html_dir, OutputPaths.html_cluster_table)
        if args.verbose:
            logging.info(f"Visualizing cluster table: {html_cluster_table_path}...")
        if args.time:
            start_time = time.time()        
        mc_avg.summary_table_html(
            html_out=html_cluster_table_path,
            columns=VisualizeArgs.html_cluster_table_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Visualize: Clusters removed
    if args.html_cluster_removed:
        # Visualize all metadata columns
        html_cluster_removed_cols = mc_avg_removed.metadata.columns.tolist() + mc_avg_removed.get_images_columns()
        if args.verbose:
            logging.info(f"Visualizing the following columns for removed cluster table: {html_cluster_removed_cols}")

        # Create HTML table
        html_cluster_removed_path = os.path.join(html_dir, OutputPaths.html_cluster_removed)
        if args.verbose:
            logging.info(f"Visualizing removed clusters: {html_cluster_removed_path}...")
        if args.time:
            start_time = time.time()        
        mc_avg_removed.summary_table_html(
            html_out=html_cluster_removed_path,
            columns=html_cluster_removed_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    ## SUB-CLUSTERS
    # Visualize: Sub-cluster table
    if args.html_subcluster_table:
        # Check html table columns
        VisualizeArgs.html_cluster_table_cols = [
            col for col in VisualizeArgs.html_cluster_table_cols
            if col in mc_subavg.metadata.columns or col in mc_subavg.get_images_columns()
        ]
        if args.verbose:
            logging.info(f"Visualizing the following columns for sub-cluster table: {VisualizeArgs.html_cluster_table_cols}")

        # Create HTML table
        html_subcluster_table_path = os.path.join(html_dir, OutputPaths.html_subcluster_table)
        if args.verbose:
            logging.info(f"Visualizing sub-cluster table: {html_subcluster_table_path}...")
        if args.time:
            start_time = time.time()        
        mc_subavg.summary_table_html(
            html_out=html_subcluster_table_path,
            columns=VisualizeArgs.html_cluster_table_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    
    # Visualize: Sub-clusters removed
    if args.html_subcluster_removed:
        # Visualize all metadata columns
        html_subcluster_removed_cols = mc_subavg_removed.metadata.columns.tolist() + mc_subavg_removed.get_images_columns()
        if args.verbose:
            logging.info(f"Visualizing the following columns for removed sub-cluster table: {html_subcluster_removed_cols}")

        # Create HTML table
        html_subcluster_removed_path = os.path.join(html_dir, OutputPaths.html_subcluster_removed)
        if args.verbose:
            logging.info(f"Visualizing removed sub-clusters: {html_subcluster_removed_path}...")
        if args.time:
            start_time = time.time()        
        mc_subavg_removed.summary_table_html(
            html_out=html_subcluster_removed_path,
            columns=html_subcluster_removed_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    

    ## END ----------------------------------------------------------------
    print(f"MotifCompendium pipeline completed successfully. Output saved to {args.output_dir}.")