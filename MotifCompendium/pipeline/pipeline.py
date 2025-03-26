import os
import time
import argparse

import numpy as np
import pandas as pd
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

    parser.add_argument(
        "-i",
        "--input-h5s",
        nargs="+",
        type=str,
        required=True,
        help="Path to the input Modisco H5 file(s).",
    )
    parser.add_argument(
        "-n",
        "--input-names",
        nargs="+",
        type=str,
        default=None,
        help="Nickname of the input Modisco H5 file(s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="Path to the metadata file, by h5, h5_name, or motif: CSV, TSV format.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default=None,
        help="Path to a reference motif file: MotifCompendium object, or PFM, MEME .txt format.",
    )

    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold to apply during clustering.",
    )
    parser.add_argument(
        "--sim-scan",
        type=list,
        default=None,
        help="List of similarity thresholds to scan during clustering. MUST INCLUDE SIM_THRESHOLD.",
    )
    parser.add_argument(
        "--min-seqlets",
        type=int,
        default=100,
        help="Minimum number of seqlets to consider a motif.",
    )

    parser.add_argument(
        "--quality", action="store_true", help="Generate quality plots for clustering."
    )
    parser.add_argument(
        "--html-collection",
        action="store_true",
        help="Generate HTML collection of motif constituents per cluster.",
    )
    parser.add_argument(
        "--html-table",
        action="store_true",
        help="Generate HTML summary table of clustered motifs.",
    )

    parser.add_argument(
        "-ch",
        "--max-chunk",
        type=int,
        default=1000,
        help="Maximum number of motifs to process at a time. Set to -1 to use no chunking.",
    )
    parser.add_argument(
        "-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use."
    )
    parser.add_argument(
        "--unsafe", action="store_false", dest="safe", help="Disable safety checks."
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for processing."
    )
    parser.add_argument(
        "--no-ic",
        action="store_false",
        dest="ic",
        help="Do not compute information content.",
    )
    parser.add_argument("--fast-plot", action="store_true", help="Use fast plotting.")

    parser.add_argument(
        "--time", action="store_true", help="Print time taken for each step."
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    args = parser.parse_args()
    return args


## DEFAULT OPTIONS ------------------------------------------------------------
def set_default_options():
    import MotifCompendium.pipeline.configs as configs

    global OutputPaths
    global MetadataCols
    global MotifMatchArgs
    global ClusterArgs
    global MotifFilterArgs

    OutputPaths = configs.OutputPaths
    MetadataCols = configs.MetadataCols
    MotifMatchArgs = configs.MotifMatchArgs
    ClusterArgs = configs.ClusterArgs
    MotifFilterArgs = configs.MotifFilterArgs


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

    # Set compute options
    if args.verbose:
        print(
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

    # Check input, output paths
    if args.verbose:
        print(f"Checking input, output paths...")
        print(f"Number of Input H5 files: {len(args.input_h5s)}")
        print(f"Output directory: {args.output_dir}")
    for h5 in args.input_h5s:
        if not os.path.exists(h5):
            raise FileNotFoundError(f"Input H5 file not found: {h5}")
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)

    ## BUILD --------------------------------------------------------------------
    # Load the input H5 files
    modisco_dict = build_modisco_dict(args.input_h5s, args.input_names)
    if args.verbose:
        print(f"Loading input H5 files...")
    while True:
        try:
            if args.verbose:
                print(f"Building MotifCompendium object from input H5 files...")
            if args.time:
                start_time = time.time()
            mc = MotifCompendium.build_from_modisco(
                modisco_dict=modisco_dict,
                ic=args.ic,
                safe=args.safe,
            )
            if args.verbose:
                print(
                    f"Completed building MotifCompendium object:\n"
                    f"  Total number of motifs: {len(mc.motifs)}\n"
                    f"  Metadata columns: {mc.metadata.columns.tolist()}"
                )
            if args.time:
                print(f"Time taken: {time.time() - start_time:.2f}s")
            break
        except Exception as e:
            print(f"Error: {e}")
            args.max_chunk = args.max_chunk - 100
            MotifCompendium.set_compute_options(max_chunk=args.max_chunk)
            print(f"Retrying with max_chunk={args.max_chunk}...")
            if args.max_chunk <= 0:
                raise ValueError("Error building MotifCompendium object.")

    # Load metadata
    if args.metadata:
        if args.verbose:
            print(f"Loading metadata: {args.metadata}...")
        if args.metadata.endswith(".csv"):
            metadata_df = pd.read_csv(args.metadata)
        elif args.metadata.endswith(".tsv"):
            metadata_df = pd.read_csv(args.metadata, sep="\t")
        else:
            raise ValueError("Metadata file must be a CSV or TSV file.")

        if len(metadata_df) == len(mc.metadata):
            mc.metadata = pd.concat([mc.metadata, metadata_df], axis=1, join="inner")
        elif len(metadata_df) == len(args.input_h5s):
            mc.metadata = mc.metadata.merge(
                metadata_df, left_index=True, right_index=True, how="left"
            )
        elif len(metadata_df) == len(set(args.input_names)):
            mc.metadata = mc.metadata.merge(
                metadata_df,
                left_on="model",
                right_on=metadata_df.columns[0],
                how="left",
            )
        else:
            raise ValueError(
                "Number of rows in metadata file must either match \
                the number of motifs, the number of input H5 files, or the number of unique input names."
            )

    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_full)
    if args.verbose:
        print(f"Saving MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    ## FILTER: MODISCO ----------------------------------------------------------
    # Filter #1: Remove motifs with less than min_seqlets
    if args.verbose:
        print(f"Filtering motifs with less than {args.min_seqlets} seqlets...")
    mc = mc[mc.metadata["num_seqlets"] >= args.min_seqlets]
    if args.verbose:
        print(f"Number of motifs after filtering: {len(mc.motifs)}")

    # Filter #2: Reference matching
    if args.reference is None:
        args.reference = MotifMatchArgs.reference_default

    if args.reference.endswith(".mc"):
        if args.verbose:
            print(f"Loading reference MotifCompendium object: {args.reference}...")
        ref_mc = MotifCompendium.load(args.reference, safe=args.safe)
        if args.verbose:
            print(f"Matching motifs to reference MotifCompendium object...")
        mc.assign_label_from_other(
            other=ref_mc,
            save_col_sim=MetadataCols.match_col_score,
            save_col_name=MetadataCols.match_col_name,
            save_col_logo=MetadataCols.match_col_logo,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    elif args.reference.endswith(".txt") or args.reference.endswith(".meme"):
        if args.verbose:
            print(f"Matching motifs to reference file: {args.reference}...")
        utils_analysis.label_from_pfms(
            mc=mc,
            pfm_file=args.reference,
            save_col_sim=MetadataCols.match_col_score,
            save_col_name=MetadataCols.match_col_name,
            save_col_logo=MetadataCols.match_col_logo,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    else:
        raise ValueError(
            "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
        )

    # Filter #3: Calculate and apply filter metrics
    if args.verbose:
        print(f"Calculating filter metrics...")
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc,
        metric_list=MotifFilterArgs.motif_metrics,
    )
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        print(f"Applying filter thresholds as flags...")
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
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #4: Override flags, for good matches
    if args.verbose:
        print(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.overrides_list[0]["threshold"]}, Composite: {MotifFilterArgs.overrides_list[1]["threshold"]}'
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
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #5: Remove flagged motifs
    if args.verbose:
        print(f"Removing flagged motifs...")
    mc_removed = mc[mc.metadata[MetadataCols.filter_col_flag]]
    mc = mc[~mc.metadata[MetadataCols.filter_col_flag]]

    # Save MotifCompendium objects
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_filtered)
    mc_removed_path = os.path.join(args.output_dir, OutputPaths.mc_removed)
    if args.verbose:
        print(
            f"Saving filtered MotifCompendium objects:\n"
            f"  Filtered: {mc_path}\n"
            f"  Removed: {mc_removed_path}"
        )
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    mc_removed.save(mc_removed_path)
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    ## CLUSTERING ---------------------------------------------------------------
    # Set options
    if args.sim_scan:
        sim_thresholds = args.sim_scan
    else:
        sim_thresholds = [args.sim_threshold]

    # Split by composite motifs
    mc_dict = {}
    for iter in range(1, composite_threholds.max_submotifs):
        mc_dict[f"mc_{iter}"] = mc[
            mc.metadata[f"{MetadataCols.match_col_name}{iter}"]
            > MotifMatchArgs.composite_threshold
        ]

    # Cluster motifs
    for sim_threshold in sim_thresholds:
        cluster_col_name = f"{ClusterArgs.algorithm}_{sim_threshold}"
        mc.metadata[cluster_col_name] = 0
        last_cluster = 0
        for iter, mc_iter in mc_dict.items():
            # Cluster motifs
            if args.verbose:
                print(
                    f"Clustering motifs using: {ClusterArgs.algorithm} {sim_threshold}..."
                )
            if args.time:
                start_time = time.time()
            mc_iter.cluster(
                algorithm=ClusterArgs.algorithm,
                sim_threshold=sim_threshold,
                save_name=cluster_col_name,
            )
            if args.time:
                print(f"Time taken: {time.time() - start_time:.2f}s")
            if args.verbose:
                print(
                    f"Total number of clusters ({cluster_col_name}): {len(mc_iter.metadata[cluster_col_name].unique())}"
                )

            # Add last_cluster, to keep membership unique
            mc_iter.metadata[cluster_col_name] = (
                mc_iter.metadata[cluster_col_name] + last_cluster
            )

            # Update cluster membership back to full MotifCompendium object
            mc.metadata.update(mc_iter.metadata[cluster_col_name])
            last_cluster = mc.metadata[cluster_col_name].max() + 1

        # Summarize cluster quality
        if quality:
            # Make directory
            quality_dir = os.path.join(args.output_dir, "quality")
            if not quality_dir.exists():
                if args.verbose:
                    print(f"Creating quality directory: {quality_dir}...")
                quality_dir.mkdir(parents=True, exist_ok=True)

            # Quality: Histogram
            histogram_path = os.path.join(
                quality_dir, f"histogram_{cluster_col_name}.png"
            )
            if args.verbose:
                print(f"Summarizing cluster quality (Histogram): {histogram_path}...")
            if args.time:
                start_time = time.time()
            utils_analysis.judge_clustering(
                mc=mc,
                clustering=cluster_col_name,
                save_loc=histogram_path,
            )
            if args.time:
                print(f"Time taken: {time.time() - start_time:.2f}s")

            # Quality: Heatmap
            heatmap_path = os.path.join(quality_dir, f"heatmap_{cluster_col_name}.png")
            if args.verbose:
                print(f"Summarizing cluster quality (Heatmap): {heatmap_path}...")
            if args.time:
                start_time = time.time()
            sns.heatmap(
                mc.clustering_quality(
                    cluster_col=cluster_col_name,
                ),
                cbar=True,
            )
            plt.savefig(heatmap_path)
            if args.time:
                print(f"Time taken: {time.time() - start_time:.2f}s")

    # Save MotifCompendium object
    mc_path = os.path.join(args.output_dir, OutputPaths.mc_clustered)
    if args.verbose:
        print(f"Saving clustered MotifCompendium object: {mc_path}...")
    if args.time:
        start_time = time.time()
    mc.save(mc_path)
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    ## AVERAGE ------------------------------------------------------------------
    cluster_col_name = f"{ClusterArgs.algorithm}_{args.sim_threshold}"

    # Average: Cluster motifs
    if args.verbose:
        print(f"Averaging motifs per cluster...")
    if args.time:
        start_time = time.time()
    mc_avg = mc.average_motif(
        cluster_col=cluster_col_name,
        aggregations=ClusterArgs.aggregate_metadata,
        weight_col=ClusterArgs.weight_col,
    )
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Save Average MotifCompendium object
    mc_avg_path = os.path.join(args.output_dir, OutputPaths.mc_avg)
    if args.verbose:
        print(f"Saving average MotifCompendium object: {mc_avg_path}...")
    if args.time:
        start_time = time.time()
    mc_avg.save(mc_avg_path)
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    ## FILTER: CLUSTERS --------------------------------------------------------------------
    # Filter #1: Reference matching
    if args.reference.endswith(".mc"):
        if args.verbose:
            print(f"Matching motifs to reference MotifCompendium object...")
        mc_avg.assign_label_from_other(
            other=ref_mc,
            save_col_sim=MetadataCols.match_col_score,
            save_col_name=MetadataCols.match_col_name,
            save_col_logo=MetadataCols.match_col_logo,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    elif args.reference.endswith(".txt") or args.reference.endswith(".meme"):
        if args.verbose:
            print(f"Matching motifs to reference file: {args.reference}...")
        utils_analysis.label_from_pfms(
            mc=mc_avg,
            pfm_file=args.reference,
            save_col_sim=MetadataCols.match_col_score,
            save_col_name=MetadataCols.match_col_name,
            save_col_logo=MetadataCols.match_col_logo,
            max_submotifs=MotifMatchArgs.max_submotifs,
            min_score=MotifMatchArgs.min_score,
        )
    else:
        raise ValueError(
            "Reference file must be a MotifCompendium object or PFM, MEME .txt file."
        )

    # Filter #2: Calculate and apply filter metrics
    if args.verbose:
        print(f"Calculating filter metrics...")
    if args.time:
        start_time = time.time()
    utils_analysis.calculate_filters(
        mc=mc_avg,
        metric_list=MotifFilterArgs.motif_metrics,
    )
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    if args.verbose:
        print(f"Applying filter thresholds as flags...")
    mc_avg.metadata[MetadataCols.filter_col_flag] = False
    if args.time:
        start_time = time.time()
    for filter_args in MotifFilterArgs.motif_filters:
        apply_filter_threshold(
            mc=mc_avg,
            flag_col=MetadataCols.filter_col_flag,
            metric=filter_args.metric,
            operation=filter_args.operation,
            threshold=filter_args.threshold,
            override=filter_args.override,
        )
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #4: Override flags, for good matches
    if args.verbose:
        print(
            f"Overriding flags for matches above threshold:\n"
            f'Base: {MotifFilterArgs.overrides_list[0]["threshold"]}, Composite: {MotifFilterArgs.overrides_list[1]["threshold"]}'
        )
    if args.time:
        start_time = time.time()
    for override_filter_args in MotifFilterArgs.override_filters:
        apply_filter_threshold(
            mc=mc_avg,
            flag_col=MetadataCols.filter_col_flag,
            metric=override_filter_args.metric,
            operation=override_filter_args.operation,
            threshold=override_filter_args.threshold,
            override=override_filter_args.override,
        )
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Filter #5: Remove flagged motifs
    if args.verbose:
        print(f"Removing flagged motifs...")
    mc_avg_removed = mc_avg[mc_avg.metadata[MetadataCols.filter_col_flag]]
    mc_avg = mc_avg[~mc_avg.metadata[MetadataCols.filter_col_flag]]

    # Save MotifCompendium objects
    mc_avg_path = os.path.join(args.output_dir, OutputPaths.mc_avg_filtered)
    mc_avg_removed_path = os.path.join(args.output_dir, OutputPaths.mc_avg_remove)
    if args.verbose:
        print(
            f"Saving filtered MotifCompendium objects:\n"
            f"  Filtered: {mc_avg_path}\n"
            f"  Removed: {mc_avg_removed_path}"
        )
    if args.time:
        start_time = time.time()
    mc_avg.save(mc_avg_path)
    mc_avg_removed.save(mc_avg_removed_path)
    if args.time:
        print(f"Time taken: {time.time() - start_time:.2f}s")

    ## VISUALIZE ----------------------------------------------------------------
    # Visualize: Cluster collection
    if args.html_collection:
        if args.verbose:
            print(f"Visualizing cluster collection...")
        html_dir = os.path.join(args.output_dir, "html")
        if not os.path.exists(html_dir):
            if args.verbose:
                print(f"Creating HTML directory: {html_dir}...")
            os.makedirs(html_dir, exist_ok=True)

        html_collection_path = os.path.join(html_dir, OutputPaths.html_collection)
        if args.time:
            start_time = time.time()
        mc.motif_collection(
            html_out=html_collection_path,
            group_by=cluster_col_name,
            average_motif=True,
        )

    # Visualize: Cluster table
    if args.html_table:
        if args.verbose:
            print(f"Visualizing cluster table...")
        html_table_path = os.path.join(html_dir, OutputPaths.html_table)
        if args.time:
            start_time = time.time()
        mc.summary_table_html(
            html_out=html_table_path,
            group_by=cluster_col_name,
            average_motif=True,
        )
