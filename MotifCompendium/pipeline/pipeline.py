import os
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument("-m", "--metadata", type=str, default=None, help="Path to the metadata file, by h5, h5_name, or motif: CSV, TSV format.")
    parser.add_argument("-r", "--reference", type=str, default=None, help="Path to a reference motif file: MotifCompendium object, or PFM, MEME .txt format.")
    
    parser.add_argument("--sim-threshold", type=float, default=0.9, help="Similarity threshold to apply during clustering.")
    parser.add_argument("--sim-scan", nargs="+", type=float, default=None, help="List of similarity thresholds to scan during clustering. MUST INCLUDE SIM_THRESHOLD.")
    parser.add_argument("--min-seqlets", type=int, default=100, help="Minimum number of seqlets to consider a motif.")
    parser.add_argument("--cluster-by-composite", action="store_true", help="Cluster by composite motifs")
    parser.add_argument("--quality", action="store_true", help="Generate quality plots for clustering.")
    parser.add_argument("--html-collection", action="store_true", help="Generate HTML collection of motif constituents per cluster.")
    parser.add_argument("--html-table", action="store_true", help="Generate HTML summary table of clustered motifs.")
    
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000, help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("--unsafe", action="store_false", dest="safe", help="Disable safety checks.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("--no-ic", action="store_false", dest="ic", help="Do not compute information content.")
    parser.add_argument("--fast-plot", action="store_true", help="Use fast plotting.")
    parser.add_argument("--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


## DEFAULT OPTIONS ------------------------------------------------------------
def set_default_options():
    import MotifCompendium.pipeline.configs as configs

    global OutputPaths
    global MetadataCols
    global MotifMatchArgs
    global ClusterArgs
    global VisualizeArgs
    global MotifFilterArgs
    global ClusterFilterArgs

    OutputPaths = configs.OutputPaths
    MetadataCols = configs.MetadataCols
    MotifMatchArgs = configs.MotifMatchArgs
    ClusterArgs = configs.ClusterArgs
    VisualizeArgs = configs.VisualizeArgs
    MotifFilterArgs = configs.MotifFilterArgs
    ClusterFilterArgs = configs.ClusterFilterArgs


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
    if override:
        if operation == "<":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] < threshold
            )
        elif operation == "<=":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] <= threshold
            )
        elif operation == ">":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] > threshold
            )
        elif operation == ">=":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] >= threshold
            )
        elif operation == "==":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] == threshold
            )
        elif operation == "!=":
            mc.metadata[flag_col] = mc.metadata[flag_col] & (
                mc.metadata[metric] != threshold
            )
        else:
            raise ValueError("Invalid operation for filter threshold.")
    elif override == False:
        if operation == "<":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] < threshold
            )
        elif operation == "<=":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] <= threshold
            )
        elif operation == ">":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] > threshold
            )
        elif operation == ">=":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] >= threshold
            )
        elif operation == "==":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] == threshold
            )
        elif operation == "!=":
            mc.metadata[flag_col] = mc.metadata[flag_col] | (
                mc.metadata[metric] != threshold
            )
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
        mc = MotifCompendium.load_old_compendium(args.input_old_mc)
        if args.verbose:
            logging.info(
                f"Completed loading old MotifCompendium object:\n"
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
        mc.metadata = mc.metadata.merge(
            metadata_df,
            left_on="model",
            right_on=metadata_df.columns[0],
            how="left",
        )

        # Update metadata aggregation
        new_aggregate_metadata = []
        for col in metadata_df.columns:
            if col not in mc.metadata.columns:
                new_aggregate_metadata.append((col, "concat", f"{col}s"))
        ClusterArgs.aggregate_metadata.extend(new_aggregate_metadata)

        # Update HTML table columns
        VisualizeArgs.html_table_cols.extend([f"{col}s" for col in metadata_df.columns])

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
    # Filter #1: Remove motifs with less than min_seqlets
    if args.verbose:
        logging.info(f"Filtering motifs with less than {args.min_seqlets} seqlets...")
    mc = mc[mc.metadata["num_seqlets"] >= args.min_seqlets]
    if args.verbose:
        logging.info(f"Number of motifs after filtering: {len(mc)}")

    # Filter #2: Reference matching
    if args.reference is None:
        args.reference = MotifMatchArgs.reference_default

    if args.reference.endswith(".mc"):
        if args.verbose:
            logging.info(f"Loading reference MotifCompendium object: {args.reference}...")
        if args.time:
            start_time = time.time()
        ref_mc = MotifCompendium.load(args.reference, safe=args.safe)
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

        if args.verbose:
            logging.info(f"Matching motifs to reference MotifCompendium object...")
        if args.time:
            start_time = time.time()
        mc.assign_label_from_other(
            other=ref_mc,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    elif args.reference.endswith(".txt") or args.reference.endswith(".meme"):
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

    # Filter #3: Calculate and apply filter metrics
    if args.verbose:
        logging.info(f"Calculating filter metrics...")
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        logging.info(f"Applying filter thresholds as flags...")
    mc.metadata[MetadataCols.filter_col_flag] = False
    if args.time:
        start_time = time.time()
    for filter_args in MotifFilterArgs.motif_filters:
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

    # Filter #4: Override flags, for good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.override_filters[0].threshold}, Composite: {MotifFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for override_filter_args in MotifFilterArgs.override_filters:
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

    # Filter #5: Remove flagged motifs
    if args.verbose:
        logging.info(f"Removing flagged motifs...")
    mc_removed = mc[mc.metadata[MetadataCols.filter_col_flag]]
    mc = mc[~mc.metadata[MetadataCols.filter_col_flag]]
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
        score_columns = [f"{MetadataCols.match_column_prefix}_score{i}" for i in range(MotifMatchArgs.max_submotifs)]
        mc.metadata["num_composites"] = mc.metadata[score_columns].values.argmax(axis=1)
        
        mc_list = []
        for i in range(MotifMatchArgs.max_submotifs):
            mc_list.append(mc[mc.metadata["num_composites"] == i])

    # Cluster motifs
    for sim_threshold in sim_thresholds:
        cluster_col_name = f"{ClusterArgs.algorithm}_{sim_threshold}"
        mc.metadata[cluster_col_name] = 0
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
                mc_iter.metadata[cluster_col_name] += last_cluster

                # Update cluster membership back to full MotifCompendium object
                mc.metadata.update(mc_iter.metadata[cluster_col_name])
                last_cluster = mc.metadata[cluster_col_name].max() + 1
        
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
            logging.info(f"Total number of clusters ({cluster_col_name}): {len(mc.metadata[cluster_col_name].unique())}")

        # Summarize cluster quality
        if args.quality:
            # Make directory
            quality_dir = os.path.join(args.output_dir, "quality")
            if not os.path.exists(quality_dir):
                if args.verbose:
                    logging.info(f"Creating quality directory: {quality_dir}...")
                os.makedirs(quality_dir, exist_ok=True)

            # Quality: Histogram
            histogram_path = os.path.join(
                quality_dir, f"histogram_{cluster_col_name}.png"
            )
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
            sns.heatmap(
                mc.clustering_quality(cluster_col=cluster_col_name,),
                cbar=True,
            )
            plt.savefig(heatmap_path)
            if args.time:
                logging.info(f"Time taken: {time.time() - start_time:.2f}s")

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

    # Average: Cluster motifs
    if args.verbose:
        logging.info(f"Averaging motifs per cluster...")
    if args.time:
        start_time = time.time()

    # Check aggregation metadata columns
    for col, agg, new_col in ClusterArgs.aggregate_metadata:
        if col not in mc.metadata.columns:
            ClusterArgs.aggregate_metadata.remove((col, agg, new_col))
    
    mc_avg = mc.cluster_averages(
        cluster_col=cluster_col_name,
        aggregations=list(ClusterArgs.aggregate_metadata),
        weight_col=ClusterArgs.weight_col,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Save Average MotifCompendium object
    mc_avg_path = os.path.join(args.output_dir, OutputPaths.mc_avg)
    if args.verbose:
        logging.info(f"Saving average MotifCompendium object: {mc_avg_path}...")
    if args.time:
        start_time = time.time()
    mc_avg.save(mc_avg_path)
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")


    ## FILTER: CLUSTERS --------------------------------------------------------------------
    # Filter #1: Reference matching
    if args.reference.endswith(".mc"):
        if args.verbose:
            logging.info(f"Matching motifs to reference MotifCompendium object...")
        mc_avg.assign_label_from_other(
            other=ref_mc,
            save_column_prefix=MetadataCols.match_column_prefix,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    elif args.reference.endswith(".txt") or args.reference.endswith(".meme"):
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

    # Filter #2: Calculate and apply filter metrics
    if args.verbose:
        logging.info(f"Calculating filter metrics...")
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc_avg,
        metric_list=ClusterFilterArgs.motif_metrics,
    )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        logging.info(f"Applying filter thresholds as flags...")
    mc_avg.metadata[MetadataCols.filter_col_flag] = False
    if args.time:
        start_time = time.time()
    for filter_args in ClusterFilterArgs.motif_filters:
        apply_filter_threshold(
            mc=mc_avg,
            flag_col=MetadataCols.filter_col_flag,
            metric=filter_args.metric,
            operation=filter_args.operation,
            threshold=filter_args.threshold,
            override=filter_args.override,
        )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #4: Override flags, for good matches
    if args.verbose:
        logging.info(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {ClusterFilterArgs.override_filters[0].threshold}, Composite: {ClusterFilterArgs.override_filters[1].threshold}'
        )
    if args.time:
        start_time = time.time()
    for override_filter_args in ClusterFilterArgs.override_filters:
        apply_filter_threshold(
            mc=mc_avg,
            flag_col=MetadataCols.filter_col_flag,
            metric=override_filter_args.metric,
            operation=override_filter_args.operation,
            threshold=override_filter_args.threshold,
            override=override_filter_args.override,
        )
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #5: Remove flagged motifs
    if args.verbose:
        logging.info(f"Removing flagged motifs...")
    mc_avg_removed = mc_avg[mc_avg.metadata[MetadataCols.filter_col_flag]]
    mc_avg = mc_avg[~mc_avg.metadata[MetadataCols.filter_col_flag]]
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


    ## VISUALIZE ----------------------------------------------------------------
    # Visualize: Cluster collection
    if args.html_collection or args.html_table:
        html_dir = os.path.join(args.output_dir, "html")
        if not os.path.exists(html_dir):
            if args.verbose:
                logging.info(f"Creating HTML directory: {html_dir}...")
            os.makedirs(html_dir, exist_ok=True)

    if args.html_collection:
        html_collection_path = os.path.join(html_dir, OutputPaths.html_collection)
        if args.verbose:
            logging.info(f"Visualizing cluster collection: {html_collection_path}...")
        if args.time:
            start_time = time.time()
        mc.motif_collection_html(
            html_out=html_collection_path,
            group_by=cluster_col_name,
            average_motif=True,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")

    # Visualize: Cluster table
    if args.html_table:
        html_table_path = os.path.join(html_dir, OutputPaths.html_table)
        if args.verbose:
            logging.info(f"Visualizing cluster table: {html_table_path}...")
        if args.time:
            start_time = time.time()

        # Check html table columns
        for col in VisualizeArgs.html_table_cols:
            if col not in mc_avg.metadata.columns or col not in mc_avg.get_image_columns():
                VisualizeArgs.html_table_cols.remove(col)
        
        mc_avg.summary_table_html(
            html_out=html_table_path,
            columns=VisualizeArgs.html_table_cols,
            editable=VisualizeArgs.editable,
        )
        if args.time:
            logging.info(f"Time taken: {time.time() - start_time:.2f}s")
