#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: merge the composites by their ISM cooperativity, into the final ENCODE MotifCompendium (C5).

After pipeline_encode.py (the ENCODE compendium; its composites and their subunits in --composite-dir) and
pipeline_encode_composite_ism.py (ISM cooperativity per composite and model, in <composite-dir>/ism), as
composite/identify_composite_motifs.ipynb after the ISM runs:

  1. ISM: each composite's ISM contributions and results, per model, compared with the composite (similarity,
     spacing); each (composite, model) classified as hard, soft, hard & soft or no synergy, and each composite by its
     strongest evidence across models. <composite-dir>/{mc,export,html}/ENCODE-motifcompendium_composite_ism*
  2. Merge: duplicate composites (centroids, until no pair is similar enough), then the soft and no-synergy composites
     of the same reference clusters; the merged ones reannotated; the no-synergy composites removed.
  3. The final ENCODE MotifCompendium: the singles and the merged composites (<composite-dir>/mc/ENCODE-motifcompendium_vf.mc).
     The ENCODE clusters left out of it (no-synergy and unannotated composites) are added to the removed ones.
  4. Exported as by pipeline_encode.py: the ENCODE compendia (all, filtered, removed), the motif index and every level
     below, over the ENCODE stage's and the atlas pipelines' files. Then pipeline_encode_hits.py.

Rerunning this stage needs the ENCODE stage's outputs first (pipeline_encode.py), which this stage overwrites.
"""
import os
import sys
import time
import re
import argparse
import logging
import json
import importlib.util

import h5py
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity
import MotifCompendium.utils.config as utils_config


# The ENCODE stage (its configs, paths, export and save_replace): the module pipeline_encode_full.py has loaded, else
# this script's sibling
if "pipeline_encode" in sys.modules:
    pipeline_encode = sys.modules["pipeline_encode"]
else:
    _spec = importlib.util.spec_from_file_location(
        "pipeline_encode", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_encode.py")
    )
    pipeline_encode = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = pipeline_encode
    _spec.loader.exec_module(pipeline_encode)
configs = pipeline_encode.configs
save_replace = pipeline_encode.save_replace


#### ARGUMENTS ---------------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="ENCODE MotifCompendium: merge the composites by their ISM cooperativity, into the final ENCODE MotifCompendium.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output: the ENCODE stage's directory, the composites
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory of the ENCODE stage (pipeline_encode.py --output-dir), overwritten with the final compendium.")
    parser.add_argument("--composite-dir", type=str, default=None,
                        help="Directory of the composites and their ISM runs (default: 'composite', next to --output-dir).")
    parser.add_argument("--mcid-version", type=str, default="V0", help="ENCODE MCID version.")
    parser.add_argument("--tfatlas-dir", type=str, required=True,
                        help="Output root of the TF atlas pipeline (pipeline/bpnet/pipeline_bpnet.py).")
    parser.add_argument("--chromatlas-dir", type=str, required=True,
                        help="Output root of the Chromatin atlas pipeline (pipeline/chrombpnet/pipeline_chrombpnet.py).")
    parser.add_argument("--procapatlas-dir", type=str, required=True,
                        help="Output root of the ProCap atlas pipeline (pipeline/procapnet/pipeline_procapnet.py).")

    # Input: Reference databases
    parser.add_argument("--mc-ref-qc", type=str, required=True,
                        help="ENCODE QC reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-direct", type=str, required=True,
                        help="ENCODE QC + in vitro reference PFM MotifCompendium object (.mc).")
    parser.add_argument("--mc-ref-clustered-hr-metadata", type=str, required=True,
                        help="Metadata TSV of the human-readable clustered reference.")
    parser.add_argument("--column-desc-tsv", type=str, required=True,
                        help="TSV describing HTML summary table column names.")

    # Input: initial metadata of each atlas (recompiled into the final compendium)
    parser.add_argument("--tf-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the TF atlas.")
    parser.add_argument("--chrom-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the Chromatin atlas.")
    parser.add_argument("--procap-init-metadata", type=str, required=True,
                        help="Initial TF-MoDISco metadata TSV of the ProCap atlas.")

    # Compute
    parser.add_argument("-g", "--use-gpu", action="store_true", help="Use GPU for processing.")
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("-ch", "--max-chunk", type=int, default=1000,
                        help="Maximum number of motifs to process at a time. Set to -1 for no chunking.")
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
    """Bind the ENCODE config classes (configs.PROFILES["ENCODE"]) to module globals, here and in pipeline_encode
    (whose functions this stage calls)."""
    pipeline_encode.set_default_configs()
    for name, cls in configs.PROFILES["ENCODE"].items():
        globals()[name] = cls()


#### FUNCTIONS ---------------------------------------------------------------
def load_reference(path, safe):
    """A reference MotifCompendium: .mc, else a PFM file."""
    if path.endswith(".mc"):
        return MotifCompendium.load(path, safe=safe)
    return MotifCompendium.build_from_pfm(pfm_dict={os.path.splitext(os.path.basename(path))[0]: path}, safe=safe)


def align_columns(mcs):
    """Give each MotifCompendium every column and image of the others (None where absent), for MotifCompendium.combine."""
    columns = list(dict.fromkeys(col for mc in mcs for col in mc.columns()))
    images = list(dict.fromkeys(image for mc in mcs for image in mc.images()))
    for mc in mcs:
        for col in columns:
            if col not in mc.columns():
                mc.metadata[col] = [None] * len(mc)
        for image_col in images:
            if image_col not in mc.images():
                mc.set_images(image_name=image_col, images=[None] * len(mc))


def final_encode_compendium(mc_encode, mc_encode_c4_filtered, mc_encode_c4_removed, mc_encode_composite_nosynergy,
                            mc_atlascluster_combined, encodecluster_cluster_col, safe=False):
    """
    The final ENCODE MotifCompendium (filtered and removed, by filter_flag_col), and its map from each ENCODE cluster of
    the ENCODE stage (encodecluster_cluster_col of mc_atlascluster_combined, which is updated) to its final name and
    simple name.

    Filtered: the C5 ENCODE clusters (mc_encode). Removed: the ENCODE stage's removed ones (mc_encode_c4_removed), and
    its filtered ENCODE clusters left out of C5: the no-synergy composites and the composites without subunits (dropped,
    with an error label). Each C5 ENCODE cluster is named as in the ENCODE stage, or, a merged composite, by its
    members' names, comma-joined: so every name of the ENCODE stage maps to its final ENCODE cluster.

    Returns:
        (mc_encode_final, mc_atlascluster_combined, atlascluster2encode_name_map, atlascluster2encode_simple_map,
         ENCODE clusters left out of C5, by flag type)
    """
    assert not pd.concat([mc_encode_c4_filtered[MetadataCols.name_col], mc_encode_c4_removed[MetadataCols.name_col]]).astype(str).str.contains(",").any(), \
        "ENCODE cluster names with a comma: the members of merged composites could not be told apart"
    c4_to_c5_name = {
        c4_name: c5_name
        for c5_name in mc_encode[MetadataCols.name_col].astype(str)
        for c4_name in c5_name.split(",")
    }
    nosynergy_c4_names = {
        c4_name
        for nosynergy_name in mc_encode_composite_nosynergy[MetadataCols.name_col].astype(str)
        for c4_name in nosynergy_name.split(",")
    }
    dropped_mask = ~mc_encode_c4_filtered[MetadataCols.name_col].astype(str).isin(c4_to_c5_name)
    mc_encode_dropped = mc_encode_c4_filtered[dropped_mask]
    flag_type_dropped = np.where(
        mc_encode_dropped[MetadataCols.name_col].astype(str).isin(nosynergy_c4_names),
        CompositeArgs.flag_type_no_synergy,
        CompositeArgs.flag_type_unannotated,
    )
    mc_encode_dropped[FilterArgs.filter_flag_col] = True
    mc_encode_dropped[FilterArgs.filter_flag_type_col] = flag_type_dropped
    mc_encode_dropped[LabelCols.label_conf_col] = LabelCols.label_confs[-1]
    for col in (LabelCols.label_readable_col, MetadataCols.prename_col, MetadataCols.simple_name_col):
        mc_encode_dropped[col] = flag_type_dropped
    logging.info(f"Left out of C5 (removed): {dict(pd.Series(flag_type_dropped).value_counts())}")

    mc_encode[FilterArgs.filter_flag_col] = False
    mc_encode_removed = mc_encode_c4_removed
    if len(mc_encode_dropped):
        align_columns([mc_encode_c4_removed, mc_encode_dropped])
        mc_encode_removed = MotifCompendium.combine([mc_encode_c4_removed, mc_encode_dropped], safe=safe)
    align_columns([mc_encode, mc_encode_removed])
    mc_encode_final = MotifCompendium.combine([mc_encode, mc_encode_removed], safe=safe)
    mc_encode_final[FilterArgs.filter_flag_col] = mc_encode_final[FilterArgs.filter_flag_col].astype(bool)
    logging.info(f"Final ENCODE MotifCompendium: {len(mc_encode)} filtered, {len(mc_encode_removed)} removed ENCODE clusters")

    # Map: Atlascluster to final ENCODE cluster (via its ENCODE cluster in the ENCODE stage)
    c4_to_final_name = {
        **{name: name for name in mc_encode_removed[MetadataCols.name_col].astype(str)},
        **c4_to_c5_name,
    }
    final_simple_map = dict(zip(mc_encode_final[MetadataCols.name_col].astype(str), mc_encode_final[MetadataCols.simple_name_col]))
    atlascluster2encode_c4_map = dict(zip(
        pd.concat([mc_encode_c4_filtered[ClusterArgs.source_cluster_col], mc_encode_c4_removed[ClusterArgs.source_cluster_col]]),
        pd.concat([mc_encode_c4_filtered[MetadataCols.name_col], mc_encode_c4_removed[MetadataCols.name_col]]).astype(str),
    ))
    atlascluster2encode_name_map = {source: c4_to_final_name[name] for source, name in atlascluster2encode_c4_map.items()}
    atlascluster2encode_simple_map = {source: final_simple_map[name] for source, name in atlascluster2encode_name_map.items()}

    mc_atlascluster_combined[VisualizeArgs.encodecluster_name_col] = mc_atlascluster_combined[encodecluster_cluster_col].map(atlascluster2encode_name_map)
    mc_atlascluster_combined[MetadataCols.simple_name_col] = mc_atlascluster_combined[encodecluster_cluster_col].map(atlascluster2encode_simple_map).fillna(mc_atlascluster_combined[MetadataCols.simple_name_col]).fillna(MetadataCols.name_unknown)

    num_left_out = dict(pd.Series(flag_type_dropped, dtype=object).value_counts())
    return mc_encode_final, mc_atlascluster_combined, atlascluster2encode_name_map, atlascluster2encode_simple_map, num_left_out


def run_encode_composite(args):
    """ENCODE: merge the composites by their ISM cooperativity, into the final ENCODE MotifCompendium"""

    ### OPTIONS -----------------------------------------------------------------
    # Run log, in <output-dir>/logs/, as in the ENCODE stage
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_composite_{os.path.basename(os.path.normpath(args.output_dir))}"
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

    MotifCompendium.set_compute_options(
        max_chunk=args.max_chunk,
        max_cpus=args.max_cpus,
        use_gpu=args.use_gpu,
        ic_scale=args.ic,
        fast_plotting=args.fast_plot,
    )

    ### PATHS --------------------------------------------------------------------
    paths = pipeline_encode.encode_paths(args)
    cpaths = pipeline_encode.composite_paths(getattr(args, "composite_dir", None) or pipeline_encode.default_composite_dir(args.output_dir))
    logging.info(f"Composites: {cpaths.composite_dir}")

    ## Derived identifiers
    unique_id_initial = f'{MetadataCols.unique_id_col}.{args.mcid_version}'
    encodecluster_cluster_col = pipeline_encode.encodecluster_cluster_col_name()

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ### INPUTS -------------------------------------------------------------------
    # The ENCODE stage's compendium (C4): filtered (the singles and the composites, with their decomposition) and
    # removed ENCODE clusters, and the combined atlasclusters with their ENCODE cluster
    logging.info("Loading the ENCODE compendia (pipeline_encode.py)...")
    mc_encode = MotifCompendium.load(paths.mc_encode_filtered_path, safe=args.safe)
    mc_encode_removed = MotifCompendium.load(paths.mc_encode_removed_path, safe=args.safe)
    mc_atlascluster_combined = MotifCompendium.load(paths.mc_atlascluster_combined_path, safe=args.safe)
    if CompositeArgs.ism_class_col in mc_encode.columns():
        raise ValueError(
            f"{paths.mc_encode_filtered_path} is already the final compendium of an earlier run of this stage "
            f"({CompositeArgs.ism_class_col}), not the ENCODE stage's: rerun pipeline_encode.py before this stage."
        )
    mc_encode_c4_filtered = mc_encode
    mc_encode_c4_removed = mc_encode_removed

    # References: without multimers, IC-scaled, with their reference cluster and its label
    logging.info("Loading reference motif databases...")
    mc_ref_qc = load_reference(args.mc_ref_qc, args.safe)
    mc_ref_direct = load_reference(args.mc_ref_direct, args.safe)

    mc_ref_qc = mc_ref_qc[~mc_ref_qc[MetadataCols.name_col].str.contains('::')]
    mc_ref_direct = mc_ref_direct[~mc_ref_direct[MetadataCols.name_col].str.contains('::')]

    mc_ref_qc.motifs = utils_motif.ic_scale(mc_ref_qc.motifs)
    mc_ref_direct.motifs = utils_motif.ic_scale(mc_ref_direct.motifs)

    _, mc_ref_qc_cluster_name_dict, mc_ref_qc_cluster_label_dict = pipeline_encode.prepare_composite_reference(
        mc_ref_direct, args.mc_ref_clustered_hr_metadata)
    for mc_ref in (mc_ref_qc, mc_ref_direct):
        mc_ref[CompositeArgs.reference_cluster_col] = mc_ref[MetadataCols.name_col].map(mc_ref_qc_cluster_name_dict)
        mc_ref[LabelCols.label_col] = mc_ref[MetadataCols.name_col].map(mc_ref_qc_cluster_label_dict)

    # Composite, ISM metadata cols
    composite_cols = [
        MetadataCols.name_col,
        MetadataCols.assay_col,
        LabelCols.label_core_col,
        LabelCols.label_core_readable_col,
        LabelCols.label_flank_col,
        LabelCols.label_flank_readable_col,
        f'{CompositeArgs.composite_ref_col}_spacing',
    ] + [
        col
        for i in range(CompositeArgs.max_submotifs_composite)
        for col in [
            f"{CompositeArgs.composite_ref_col}_name{i}",
            f'{CompositeArgs.composite_ref_col}_cluster{i}',
            f"{CompositeArgs.composite_ref_col}_score{i}",
            f"{CompositeArgs.composite_ref_col}_alignment_rc{i}",
            f"{CompositeArgs.composite_ref_col}_alignment_h{i}",
            f"{CompositeArgs.composite_ref_col}_start{i}",
            f"{CompositeArgs.composite_ref_col}_end{i}",
            f"{CompositeArgs.composite_ref_col}_{VisualizeArgs.motif_string_col}{i}",
        ]
    ]

    ### COMPOSITES (composite/identify_composite_motifs.ipynb, after the ISM runs) ------------------------------
    # Load MotifCompendium: ENCODE composite
    mc_encode_composite = MotifCompendium.load(cpaths.mc_composite_path, safe=args.safe)

    # Metadata
    mc_encode_composite[CompositeArgs.num_ids_col] = mc_encode_composite[MetadataCols.id_col].str.split(",").str.len()

    # Load metadata
    mc_encode_composite_metadata_df = pd.read_csv(cpaths.metadata_composite_path, sep="\t", header=0)

    # ## Compare optimal spacing vs. original motif

    # Load ISM cooperativity metadata
    mc_encode_composite_metadata_model_df = pd.read_csv(cpaths.metadata_composite_model_path, sep="\t",)

    # Explode MetadataCols.name_col
    mc_encode_composite_metadata_model_explode_df = mc_encode_composite_metadata_model_df.copy()
    mc_encode_composite_metadata_model_explode_df[MetadataCols.name_col] = mc_encode_composite_metadata_model_explode_df[MetadataCols.name_col].str.split(",")
    mc_encode_composite_metadata_model_explode_df = mc_encode_composite_metadata_model_explode_df.explode(MetadataCols.name_col)

    # Record original composite metadata
    for composite_metadata_col in composite_cols[1:]:
        mc_encode_composite_metadata_model_explode_df[composite_metadata_col] = mc_encode_composite_metadata_model_explode_df[MetadataCols.name_col].map(
            mc_encode_composite_metadata_df.set_index(MetadataCols.name_col)[composite_metadata_col]
        )

    # Record path of contribs_mean.h5, results.tsv: ism_composite_motifs.py writes to
    # cpaths.ism_dir/<name, with "/" -> ".">/<ID>/, so apply the same sanitization here
    mc_encode_composite_metadata_model_explode_df[CompositeArgs.ism_contrib_path_col] = mc_encode_composite_metadata_model_explode_df.apply(
        lambda row: os.path.join(cpaths.ism_dir, row[MetadataCols.name_col].replace("/", "."), row[MetadataCols.id_col], "contribs_mean.h5"),
        axis=1,
    )
    mc_encode_composite_metadata_model_explode_df[CompositeArgs.ism_results_path_col] = mc_encode_composite_metadata_model_explode_df.apply(
        lambda row: os.path.join(cpaths.ism_dir, row[MetadataCols.name_col].replace("/", "."), row[MetadataCols.id_col], "results.tsv"),
        axis=1,
    )

    # Load ISM contributions
    logging.info("Loading the ISM contributions and results...")
    contrib_motifs = []
    contrib_footprints = []
    ism_scores = {
        CompositeArgs.name_id_col: [],
        MetadataCols.name_col: [],
        f'ism_{VisualizeArgs.motif_string_col}_0': [],
        f'ism_{VisualizeArgs.motif_string_col}_1': [],
        CompositeArgs.ism_spacing_col: [],
        MetadataCols.id_col: [],
        CompositeArgs.ism_z_score_col: [],
        CompositeArgs.ism_p_value_col: [],
        CompositeArgs.ism_effect_max_col: [],
        CompositeArgs.ism_effect_medium_col: [],
    }

    for i, row in mc_encode_composite_metadata_model_explode_df.iterrows():
        name_i = row[MetadataCols.name_col]
        id_i = row[MetadataCols.id_col]
        name_id_i = f"{name_i}_{id_i}"
        contrib_path_i = row[CompositeArgs.ism_contrib_path_col]
        results_path_i = row[CompositeArgs.ism_results_path_col]
        if (os.path.exists(contrib_path_i)) and (os.path.exists(results_path_i)):
            ## Contributions
            with h5py.File(contrib_path_i, "r") as f:
                contribs_mean = f["data"][:]                  # (100 seqs, 4, 2114), float16

            # Average across seqs
            contributions_mean = contribs_mean.astype(np.float32).mean(axis=0)   # (4, 2114)

            MIDPOINT = 2114 // 2                                                  # 1057
            motif = contributions_mean[:, MIDPOINT - 50 : MIDPOINT + 50]          # (4, 100)

            ## Results
            results_df = pd.read_csv(results_path_i, sep="\t").iloc[0]
            best_spacing = int(results_df["best_spacing"])
            seqA = results_df["seq1"]
            seqB = results_df["seq2"]
            z_score = results_df["max_Z"]
            p_value = results_df["joint_vs_sum_p_value"]
            effect_max = results_df["joint_vs_ind_max"]
            effect_medium = results_df["joint_vs_ind_med_distance"]

            if best_spacing % 2 == 0:
                end_A   = MIDPOINT - best_spacing / 2 - 0.5
                start_B = MIDPOINT + best_spacing / 2 - 0.5
            else:
                end_A   = MIDPOINT - best_spacing // 2 + 0.5
                start_B = MIDPOINT + best_spacing // 2 + 1.5
            start_A = end_A - len(seqA)
            end_B   = start_B + len(seqB)

            # Identify footprint: From start, end
            footprint = contributions_mean[:, int(start_A + 0.5) : int(end_B + 0.5)]   # (4, len(seqA)+spacing+len(seqB))

            # Store
            contrib_motifs.append(motif)
            contrib_footprints.append(footprint)
        
            ism_scores[CompositeArgs.name_id_col].append(name_id_i)
            ism_scores[MetadataCols.name_col].append(name_i)
            ism_scores[MetadataCols.id_col].append(id_i)

            ism_scores[f'ism_{VisualizeArgs.motif_string_col}_0'].append(seqA)
            ism_scores[f'ism_{VisualizeArgs.motif_string_col}_1'].append(seqB)
            ism_scores[CompositeArgs.ism_spacing_col].append(best_spacing)
        
            ism_scores[CompositeArgs.ism_z_score_col].append(z_score)
            ism_scores[CompositeArgs.ism_p_value_col].append(p_value)
            ism_scores[CompositeArgs.ism_effect_max_col].append(effect_max)
            ism_scores[CompositeArgs.ism_effect_medium_col].append(effect_medium)

    # Pad motifs, footprints to same length
    max_motif_len = max(m.shape[1] for m in contrib_motifs)
    max_footprint_len = max(f.shape[1] for f in contrib_footprints)
    for i in range(len(contrib_motifs)):
        motif = contrib_motifs[i]
        footprint = contrib_footprints[i]

        # Pad motif
        if motif.shape[1] < max_motif_len:
            pad_width = max_motif_len - motif.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            contrib_motifs[i] = np.pad(motif, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

        # Pad footprint
        if footprint.shape[1] < max_footprint_len:
            pad_width = max_footprint_len - footprint.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            contrib_footprints[i] = np.pad(footprint, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Stack motifs, footprints (order: N, L, 4)
    contrib_motifs_stacked = np.stack(contrib_motifs, axis=0).transpose(0, 2, 1)   # (N, L, 4)
    contrib_footprints_stacked = np.stack(contrib_footprints, axis=0).transpose(0, 2, 1)   # (N, L, 4)

    # Build DataFrame: ISM scores
    ism_scores_df = pd.DataFrame(ism_scores)

    # Apply Benjamini-Hochberg correction for multiple testing
    ism_scores_df[CompositeArgs.ism_p_value_adj_col] = false_discovery_control(ism_scores_df[CompositeArgs.ism_p_value_col].values, method="bh")

    # Concatenate ISM scores with metadata, based on MetadataCols.name_col and MetadataCols.id_col
    composite_ism_metadata_df = ism_scores_df.merge(
        mc_encode_composite_metadata_model_explode_df,
        on=[MetadataCols.name_col, MetadataCols.id_col],
        how='left',
    )

    # Build MotifCompendium: ISM contributions
    mc_encode_composite_ism_motifs = MotifCompendium.build(
        motifs=contrib_motifs_stacked,
        metadata=composite_ism_metadata_df,
        safe=args.safe,
    )
    # mc_encode_composite_ism_footprints = MotifCompendium.build(
    #     motifs=contrib_footprints_stacked,
    #     metadata=composite_ism_metadata_df,
    #     safe=args.safe,
    # )

    # Save MotifCompendium: ISM contributions
    save_replace(mc_encode_composite_ism_motifs, cpaths.mc_composite_ism_motifs_init_path)
    # mc_encode_composite_ism_footprints.save(mc_encode_composite_ism_footprints_init_path)

    # Calculate similarity: ENCODE composite vs ISM contributions
    mc_encode_composite_motifs_adj = mc_encode_composite.motifs.copy()
    ism_motifs_adj = mc_encode_composite_ism_motifs.motifs.copy()
    # ism_footprints_adj = mc_encode_composite_ism_footprints.motifs.copy()

    max_length = max(mc_encode_composite_motifs_adj.shape[1], ism_motifs_adj.shape[1])
    if mc_encode_composite_motifs_adj.shape[1] < max_length:
        pad = max_length - mc_encode_composite_motifs_adj.shape[1]
        mc_encode_composite_motifs_adj = np.pad(mc_encode_composite_motifs_adj, ((0, 0), (0, pad), (0, 0)))
    if ism_motifs_adj.shape[1] < max_length:
        pad = max_length - ism_motifs_adj.shape[1]
        ism_motifs_adj = np.pad(ism_motifs_adj, ((0, 0), (0, pad), (0, 0)))

    mc_ism_motifs_similarity, mc_ism_motifs_alignment_rc, mc_ism_motifs_alignment_h = utils_similarity.compute_similarities(
        [ism_motifs_adj, mc_encode_composite_motifs_adj],
        [(0, 1)],
    )[0]
    # mc_ism_footprints_similarity, mc_ism_footprints_alignment_rc, mc_ism_footprints_alignment_h = utils_motif.compute_similarities(
    #     [ism_footprints_adj, mc_encode_composite_motifs_adj],
    #     [(0, 1)],
    # )[0]

    # Store similarity score
    mc_encode_composite_ism_idx = mc_encode_composite_ism_motifs[MetadataCols.name_col].map(
        pd.Series(
            np.arange(len(mc_encode_composite.metadata)),
            index=mc_encode_composite.metadata[MetadataCols.name_col],
        )
    ).to_numpy(dtype=int)
    mc_encode_composite_ism_motifs[CompositeArgs.ism_similarity_col] = (
        mc_ism_motifs_similarity[np.arange(len(mc_encode_composite_ism_idx)), mc_encode_composite_ism_idx]
    )
    # mc_encode_composite_ism_footprints[ism_similarity_footprint_col] = (
    #     mc_ism_footprints_similarity[np.arange(len(mc_encode_composite_ism_idx)), mc_encode_composite_ism_idx]
    # )

    # Calculate spacing difference: ENCODE composite vs. ISM contributions
    mc_encode_composite_ism_motifs[CompositeArgs.ism_spacing_diff_col] = abs(
        mc_encode_composite_ism_motifs[CompositeArgs.ism_spacing_col] - mc_encode_composite_ism_motifs[f'{CompositeArgs.composite_ref_col}_spacing']
    )
    # mc_encode_composite_ism_footprints[CompositeArgs.ism_spacing_diff_col] = abs(
    #     mc_encode_composite_ism_footprints[CompositeArgs.ism_spacing_col] - mc_encode_composite_ism_footprints[f'{CompositeArgs.composite_ref_col}_spacing']
    # )

    # Determine identity: ENCODE composite vs. ISM contributions
    mc_encode_composite_ism_motifs[CompositeArgs.ism_identity_col] = (
        (mc_encode_composite_ism_motifs[CompositeArgs.ism_similarity_col] >= CompositeArgs.ism_similarity_threshold) &
        (mc_encode_composite_ism_motifs[CompositeArgs.ism_spacing_diff_col] <= CompositeArgs.ism_spacing_threshold)
    )
    # mc_encode_composite_ism_footprints[CompositeArgs.ism_identity_col] = (
    #     (mc_encode_composite_ism_footprints[CompositeArgs.ism_similarity_col] >= CompositeArgs.ism_similarity_threshold) &
    #     (mc_encode_composite_ism_footprints[CompositeArgs.ism_spacing_diff_col] <= CompositeArgs.ism_spacing_threshold)
    # )

    # ## Classify: Hard, soft, no cooperativity

    # ### Classify: one classification per motif per ENCID model

    def classify_syntax(row):
        is_synergistic = row[CompositeArgs.ism_p_value_adj_col] < CompositeArgs.p_val_thresh
        is_hard = is_synergistic and row[CompositeArgs.ism_effect_max_col] > CompositeArgs.delta_thresh and row[CompositeArgs.ism_z_score_col] > CompositeArgs.z_thresh
        is_soft = is_synergistic and row[CompositeArgs.ism_effect_medium_col] > CompositeArgs.delta_thresh
        if is_hard and is_soft:
            return "hard & soft"
        elif is_hard:
            return "hard"
        elif is_soft:
            return "soft"
        return "no synergy"

    # Roll up the per-(motif, ENCID) classification into one call per motif: a
    # composite motif is tested against every ENCID/biosample where both TFs are
    # assayed, so we take the strongest evidence across ENCIDs as representative --
    # cooperativity detected in even one relevant biological context is real
    # evidence of the underlying grammar, even if it isn't significant in every
    # context tested (e.g. TF expression/cofactor availability differs by biosample).
    result_priority = list(CompositeArgs.result_priority)
    result_rank = {r: i for i, r in enumerate(result_priority)}

    # Apply classification
    mc_encode_composite_ism_motifs[CompositeArgs.ism_class_col] = mc_encode_composite_ism_motifs.metadata.apply(classify_syntax, axis=1)
    # mc_encode_composite_ism_footprints[CompositeArgs.ism_class_col] = mc_encode_composite_ism_footprints.metadata.apply(classify_syntax, axis=1)

    # Save MotifCompendium: ISM contributions
    save_replace(mc_encode_composite_ism_motifs, cpaths.mc_composite_ism_motifs_path)
    # mc_encode_composite_ism_footprints.save(mc_encode_composite_ism_footprints_path)

    # ### Classify: one classification per motif (aggregate)

    # Apply classification rank
    mc_encode_composite_ism_motifs[CompositeArgs.ism_class_rank_col] = mc_encode_composite_ism_motifs[CompositeArgs.ism_class_col].map(result_rank)
    # mc_encode_composite_ism_footprints[CompositeArgs.ism_class_rank_col] = mc_encode_composite_ism_footprints[CompositeArgs.ism_class_col].map(result_rank)

    # Count classification per composite motif (across models)
    mc_encode_composite_ism_motifs_class_counts = (
        mc_encode_composite_ism_motifs.metadata.groupby(
            [MetadataCols.name_col, CompositeArgs.ism_class_col])[MetadataCols.id_col].nunique()
            .unstack(fill_value=0)
            .reindex(columns=result_priority, fill_value=0)
            .rename(columns=CompositeArgs.result_col_suffix)
            .add_prefix("n_encids_")
    )
    # mc_encode_composite_ism_footprints_class_counts = (
    #     mc_encode_composite_ism_footprints.metadata.groupby(
    #         [MetadataCols.name_col, MetadataCols.id_col]).size()
    #         .unstack(fill_value=0)
    #         .reindex(columns=result_priority, fill_value=0)
    #         .rename(columns=CompositeArgs.result_col_suffix)
    #         .add_prefix("n_encids_")
    # )

    # Metadata
    mc_encode_composite_ism_motifs_class_counts[CompositeArgs.num_ids_col] = mc_encode_composite_ism_motifs_class_counts.sum(axis=1)
    # mc_encode_composite_ism_footprints_class_counts[CompositeArgs.num_ids_col] = mc_encode_composite_ism_footprints_class_counts.sum(axis=1)

    # Best per composite motif: strongest evidence of cooperativity across all models
    mc_encode_composite_ism_motifs_class_best = (
        mc_encode_composite_ism_motifs.metadata.sort_values(
            [MetadataCols.name_col, CompositeArgs.ism_class_rank_col, CompositeArgs.ism_p_value_adj_col, CompositeArgs.ism_z_score_col], kind="stable"
        ).drop_duplicates(MetadataCols.name_col, keep="first").set_index(MetadataCols.name_col)
    )
    # mc_encode_composite_ism_footprints_class_best = (
    #     mc_encode_composite_ism_footprints.metadata.sort_values(
    #         [MetadataCols.name_col, CompositeArgs.ism_class_rank_col, CompositeArgs.ism_p_value_adj_col, CompositeArgs.ism_z_score_col], kind="stable"
    #     ).groupby(MetadataCols.name_col).first()
    # )

    # Best per composite motif: Index
    mc_encode_composite_ism_motifs_best_idx = mc_encode_composite_ism_motifs_class_best[CompositeArgs.name_id_col].map(
        pd.Series(
            np.arange(len(mc_encode_composite_ism_motifs.metadata)),
            index=mc_encode_composite_ism_motifs[CompositeArgs.name_id_col],
        )
    ).to_numpy(dtype=int)

    # Logos: of the best ISM motif of each composite only (the composites' ISM logos)
    mc_encode_composite_ism_motifs_best = mc_encode_composite_ism_motifs[mc_encode_composite_ism_motifs_best_idx.tolist()]
    mc_encode_composite_ism_motifs_best.add_logos(
        motifs=mc_encode_composite_ism_motifs_best.motifs,
        image_name="logo (fwd)",
        trim=VisualizeArgs.html_logo_trim
    )
    mc_encode_composite_ism_motifs_best.add_logos(
        motifs=utils_motif.reverse_complement(mc_encode_composite_ism_motifs_best.motifs),
        image_name="logo (rev)",
        trim=VisualizeArgs.html_logo_trim
    )
    mc_encode_composite_ism_motifs_class_logos = pd.DataFrame({
        CompositeArgs.name_id_col: mc_encode_composite_ism_motifs_class_best[CompositeArgs.name_id_col],
        CompositeArgs.ism_logo_fwd_col: list(mc_encode_composite_ism_motifs_best.get_images("logo (fwd)")),
        CompositeArgs.ism_logo_rev_col: list(mc_encode_composite_ism_motifs_best.get_images("logo (rev)")),
    }).set_index(CompositeArgs.name_id_col)
    del mc_encode_composite_ism_motifs_best

    # Merge counts, best
    mc_encode_composite_ism_motifs_class_metadata = mc_encode_composite_ism_motifs_class_counts.join(
        mc_encode_composite_ism_motifs_class_best[[
            CompositeArgs.name_id_col,
            MetadataCols.id_col,
            f'ism_{VisualizeArgs.motif_string_col}_0',
            f'ism_{VisualizeArgs.motif_string_col}_1',
            CompositeArgs.ism_similarity_col,
            CompositeArgs.ism_spacing_col,
            CompositeArgs.ism_spacing_diff_col,
            CompositeArgs.ism_identity_col,
            CompositeArgs.ism_class_col, 
            CompositeArgs.ism_z_score_col, 
            CompositeArgs.ism_p_value_col, 
            CompositeArgs.ism_p_value_adj_col, 
            CompositeArgs.ism_effect_max_col, 
            CompositeArgs.ism_effect_medium_col
        ]]
    )
    # mc_encode_composite_ism_footprints_class = mc_encode_composite_ism_footprints_class_counts.join(
    #     mc_encode_composite_ism_footprints_class_best[[CompositeArgs.ism_class_col, MetadataCols.id_col, CompositeArgs.ism_z_score_col, CompositeArgs.ism_p_value_col, CompositeArgs.ism_p_value_adj_col, CompositeArgs.ism_effect_max_col, CompositeArgs.ism_effect_medium_col]]
    # )

    # Rename: Best ID
    mc_encode_composite_ism_motifs_class_metadata.rename(columns={
        CompositeArgs.name_id_col: CompositeArgs.ism_name_id_best_col,
        MetadataCols.id_col: CompositeArgs.ism_id_best_col,
    }, inplace=True)
    # mc_encode_composite_ism_footprints_class.rename(columns={MetadataCols.id_col: f'{MetadataCols.id_col}_best'}, inplace=True)

    mc_encode_composite.metadata.rename(columns={
        'ism_best_id': CompositeArgs.ism_id_best_col,
    }, inplace=True)

    # Combine with metadata (use motifs)
    mc_encode_composite.metadata["_motif_row"] = np.arange(len(mc_encode_composite))
    mc_encode_composite.metadata = (
        mc_encode_composite.metadata
        .merge(
            mc_encode_composite_ism_motifs_class_metadata,
            on=MetadataCols.name_col,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        .sort_values("_motif_row", kind="stable")
        .drop(columns="_motif_row")
        .reset_index(drop=True)
    )

    for ism_logo_col in [CompositeArgs.ism_logo_fwd_col, CompositeArgs.ism_logo_rev_col]:
        mc_encode_composite.set_images(
            image_name=ism_logo_col,
            images=mc_encode_composite_ism_motifs_class_logos.loc[mc_encode_composite[CompositeArgs.ism_name_id_best_col]][ism_logo_col].to_list()
        )

    # Sort by name
    mc_encode_composite.sort(
        by=[MetadataCols.name_col],
        ascending=True,
        inplace=True,
    )

    # Save ISM cooperativity metadata
    os.makedirs(os.path.dirname(cpaths.metadata_composite_ism_motifs_path), exist_ok=True)
    mc_encode_composite_ism_motifs.metadata.to_csv(cpaths.metadata_composite_ism_motifs_path, sep="\t", index=False)
    os.makedirs(os.path.dirname(cpaths.metadata_composite_ism_motifs_agg_path), exist_ok=True)
    mc_encode_composite_ism_motifs_class_metadata.to_csv(cpaths.metadata_composite_ism_motifs_agg_path, sep="\t", index=True)

    # os.makedirs(os.path.dirname(mc_encode_composite_ism_footprints_metadata_path), exist_ok=True)
    # mc_encode_composite_ism_footprints.metadata.to_csv(mc_encode_composite_ism_footprints_metadata_path, sep="\t", index=False)
    # os.makedirs(os.path.dirname(mc_encode_composite_ism_footprints_metadata_agg_path), exist_ok=True)
    # mc_encode_composite_ism_footprints_class.to_csv(mc_encode_composite_ism_footprints_metadata_agg_path, sep="\t", index=True)

    # HTML summary table: ISM cooperativity
    os.makedirs(os.path.dirname(cpaths.html_composite_ism_path), exist_ok=True)
    mc_encode_composite.summary_table_html(
        html_out=cpaths.html_composite_ism_path,
        columns=[
            html_col for html_col in CompositeArgs.html_columns_composite if html_col in (mc_encode_composite.columns() + mc_encode_composite.images())
        ],
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict,
    )

    # Save MotifCompendium: ENCODE composite with ISM metadata
    save_replace(mc_encode_composite, cpaths.mc_composite_ism_path)

    # ### Clean up composites

    # Keep copy
    mc_encode_composite_all = mc_encode_composite.copy()

    # #### Merge: Duplicate motifs

    # Initialize duplicate provenance. Rerun the cleanup load cell to start a new merge.
    if MetadataCols.unique_id_col not in mc_encode_composite.metadata:
        raise KeyError(f"{MetadataCols.unique_id_col} is required for duplicate provenance")
    mc_encode_composite.metadata.drop(columns=["duplicate_name", "duplicate_members"], errors="ignore", inplace=True)
    mc_encode_composite[CompositeArgs.duplicate_members_col] = mc_encode_composite[MetadataCols.unique_id_col].astype(str)
    mc_encode_composite[CompositeArgs.duplicate_num_members_col] = 1
    mc_encode_composite[CompositeArgs.duplicate_score_col] = 0.0

    def comma_union(values):
        """Flatten existing comma lists, retaining each token once in input order."""
        tokens = dict.fromkeys(
            token.strip() for value in values.dropna().astype(str)
            for token in value.split(",") if token.strip()
        )
        return ",".join(tokens) if tokens else None

    def aggregate_composite_memberships(mc, cluster_col, centroid):
        """One representative for composite/ISM fields; union other strings, sum counts."""
        metadata = mc.metadata.reset_index(drop=True).copy()
        metadata[CompositeArgs.ism_class_rank_col] = metadata[CompositeArgs.ism_class_col].map(result_rank)
        metadata["_representative_position"] = np.arange(len(metadata))
        ranked = metadata.sort_values(
            [cluster_col, CompositeArgs.ism_class_rank_col, CompositeArgs.ism_p_value_adj_col, CompositeArgs.ism_z_score_col,
             "_representative_position"],
            ascending=[True, True, True, False, True], na_position="last", kind="stable",
        )
        best = ranked.drop_duplicates(cluster_col).set_index(cluster_col, drop=False)
        groups = metadata.groupby(cluster_col, sort=True)
        keep = {cluster_col, CompositeArgs.soft_nosynergy_membership_col, MetadataCols.composite_col}
        sums = {CompositeArgs.num_ids_col, MetadataCols.num_seqlet_col, MetadataCols.total_contrib_score_col,
                MetadataCols.group_total_contrib_score_col, MetadataCols.num_constituents_col}
        for col in metadata:
            if col in keep or col.startswith("ism_") or col.startswith("composite_"):
                continue
            nonnull = metadata[col].dropna()
            if col.startswith("n_encids_") or col in sums:
                best[col] = groups[col].sum(min_count=1)
            elif len(nonnull) and nonnull.map(lambda value: isinstance(value, str)).all():
                best[col] = groups[col].agg(comma_union)
        best[CompositeArgs.duplicate_members_col] = groups[CompositeArgs.duplicate_members_col].agg(comma_union)
        best[CompositeArgs.duplicate_num_members_col] = best[CompositeArgs.duplicate_members_col].str.split(",").str.len()
        best[CompositeArgs.duplicate_score_col] = groups[CompositeArgs.duplicate_score_col].max()
        if CompositeArgs.soft_nosynergy_num_members_col in metadata:
            best[CompositeArgs.soft_nosynergy_num_members_col] = groups[CompositeArgs.soft_nosynergy_num_members_col].max()
        if centroid:
            working = mc.copy()
            working.metadata = metadata
            weight_col = CompositeArgs.num_ids_col if CompositeArgs.num_ids_col in metadata else None
            if weight_col is not None:
                weights = pd.to_numeric(metadata[weight_col], errors="coerce")
                if not (np.isfinite(weights).all() and (weights > 0).all()):
                    weight_col = None
            merged = working.cluster_averages(
                clustering=cluster_col, aggregations=[], weight_col=weight_col,
                compute_quality_stats=False,
            )
            best = best.loc[merged.metadata["source_cluster"].to_numpy()]
        positions = best["_representative_position"].to_numpy(dtype=int).tolist()
        if not centroid:
            merged = mc[positions]
        merged.metadata = best.drop(columns="_representative_position").reset_index(drop=True)
        merged.metadata[ClusterArgs.source_cluster_col] = best[cluster_col].to_numpy()
        if centroid:
            for image_name in mc.images():
                if image_name not in {"logo (fwd)", "logo (rev)"}:
                    images = mc.get_images(image_name)
                    merged.set_images(image_name, [images[i] for i in positions])
        return merged

    def assign_duplicate_memberships(mc):
        """Greedy disjoint pairs, descending similarity, then ascending row positions."""
        i, j = np.where(np.triu(mc.similarity > CompositeArgs.duplicate_sim_threshold, k=1))
        candidates = sorted(zip(mc.similarity[i, j], i, j), key=lambda x: (-x[0], x[1], x[2]))
        membership = np.full(len(mc), -1, dtype=int)
        scores = mc[CompositeArgs.duplicate_score_col].to_numpy(dtype=float).copy()
        next_cluster = 0
        for score, i, j in candidates:
            if membership[i] == -1 and membership[j] == -1:
                membership[[i, j]] = next_cluster
                scores[[i, j]] = np.maximum(scores[[i, j]], score)
                next_cluster += 1
        pair_count = next_cluster
        singletons = np.flatnonzero(membership < 0)
        membership[singletons] = np.arange(next_cluster, next_cluster + len(singletons))
        mc[CompositeArgs.duplicate_cluster_col] = membership
        mc[CompositeArgs.duplicate_score_col] = scores
        return pair_count

    # Steps 1–2: assign duplicate memberships, then aggregate motifs/metadata.
    # Centroids must be recomputed between rounds to decide the next memberships.
    for duplicate_iteration in range(len(mc_encode_composite)):
        pair_count = assign_duplicate_memberships(mc_encode_composite)
        if not pair_count:
            break
        mc_encode_composite = aggregate_composite_memberships(
            mc_encode_composite, CompositeArgs.duplicate_cluster_col, centroid=True
        )
        logging.info(f"Duplicate round {duplicate_iteration + 1}: {pair_count} pairs, "
                     f"{len(mc_encode_composite)} remaining")

    # #### Merge: Soft + No synergy

    # Step 1: assign memberships from the exact ordered reference-cluster tuple.
    mc_encode_composite_all = mc_encode_composite.copy()
    soft_nosynergy_ref_cluster_cols = [
        f"{CompositeArgs.composite_ref_col}_cluster0", f"{CompositeArgs.composite_ref_col}_cluster1",
    ]
    soft_nosynergy_membership = mc_encode_composite_all.metadata.copy()
    eligible = (
        soft_nosynergy_membership[CompositeArgs.ism_class_col].isin(["soft", "no synergy"])
        & soft_nosynergy_membership[soft_nosynergy_ref_cluster_cols].notna().all(axis=1)
    )
    pairs = pd.MultiIndex.from_frame(
        soft_nosynergy_membership.loc[eligible, soft_nosynergy_ref_cluster_cols]
    )
    codes, unique_pairs = (
        pd.factorize(pairs, sort=True)
        if len(pairs)
        else (np.array([], dtype=int), [])
    )
    memberships = np.full(len(soft_nosynergy_membership), -1, dtype=int)
    memberships[np.flatnonzero(eligible.to_numpy())] = codes
    singletons = np.flatnonzero(~eligible.to_numpy())
    memberships[singletons] = np.arange(len(unique_pairs), len(unique_pairs) + len(singletons))
    soft_nosynergy_membership[CompositeArgs.soft_nosynergy_membership_col] = pd.Series(
        [None] * len(soft_nosynergy_membership), index=soft_nosynergy_membership.index, dtype=object
    )
    soft_nosynergy_membership.loc[eligible, CompositeArgs.soft_nosynergy_membership_col] = [
        repr(pair) for pair in pairs
    ]
    soft_nosynergy_membership[CompositeArgs.soft_nosynergy_merge_cluster_col] = memberships
    soft_nosynergy_membership[CompositeArgs.soft_nosynergy_num_members_col] = (
        soft_nosynergy_membership.groupby(CompositeArgs.soft_nosynergy_merge_cluster_col)[MetadataCols.name_col].transform("size")
        .where(eligible, 0)
    )
    mc_encode_composite_all.metadata = soft_nosynergy_membership

    # Step 2: retain each group's best observed motif and aggregate its metadata.
    mc_encode_composite = aggregate_composite_memberships(
        mc_encode_composite_all, CompositeArgs.soft_nosynergy_merge_cluster_col, centroid=False
    )
    logging.info(f"Soft/no-synergy merge: {len(mc_encode_composite_all)} -> {len(mc_encode_composite)}")

    # Step 3: reannotate only groups actually merged by the soft/no-synergy step.
    # The complete result is restored after assigning labels to this subset.
    mc_encode_composite_merged = mc_encode_composite.copy()
    soft_reannotation_idx = np.flatnonzero(
        mc_encode_composite[CompositeArgs.soft_nosynergy_num_members_col].to_numpy() > 1
    ).astype(int)
    mc_encode_composite = mc_encode_composite[soft_reannotation_idx.astype(int).tolist()]
    mc_encode_composite[LabelCols.label_col] = MetadataCols.name_unknown
    # Preserve the inherited flank for direct-label rules; indirect labels are reassigned below.

    ## [General] Label meta-clusters
    utils_analysis.assign_label_from_other_compendium(
        assign_to_mc=mc_encode_composite,
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
    # The reference-matching call is intentionally skipped for an empty
    # reannotation subset. Give that empty frame the expected typed columns so
    # the rest of this cell remains a args.safe no-op.
    for i in range(MotifMatchArgs.max_submotifs):
        reference_name_col = f'{ReferenceMatchCols.ref_col}_name{i}'
        reference_score_col = f'{ReferenceMatchCols.ref_col}_score{i}'
        if reference_name_col not in mc_encode_composite.metadata:
            mc_encode_composite[reference_name_col] = pd.Series(
                index=mc_encode_composite.metadata.index, dtype=object
            )
        if reference_score_col not in mc_encode_composite.metadata:
            mc_encode_composite[reference_score_col] = pd.Series(
                index=mc_encode_composite.metadata.index, dtype=float
            )
    reference_name_map = mc_ref_qc_cluster_label_dict if len(mc_encode_composite) else {}
    if DirectTypeArgs.direct_type_col not in mc_encode_composite.metadata:
        mc_encode_composite[DirectTypeArgs.direct_type_col] = pd.Series(
            index=mc_encode_composite.metadata.index, dtype=object
        )
    for i, (ref_readable_name_col, ref_score_col) in enumerate(zip(ref_readable_name_cols, ref_score_cols)):    
        mc_encode_composite[ref_readable_name_col] = mc_encode_composite[f'{ReferenceMatchCols.ref_col}_name{i}'].map(reference_name_map)

    # cateogory: Mask
    mask_direct_encodecluster = mc_encode_composite[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False)
    mask_nondirect_encodecluster = ~mask_direct_encodecluster
    mask_monomer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_single_score)
        & ~(mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_dimer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & ~(mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_trimer_encodecluster_nondirect = (
        mask_nondirect_encodecluster
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score0'] > MotifMatchArgs.min_composite_score)
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score1'] > MotifMatchArgs.min_composite_score)
        & (mc_encode_composite[f'{ReferenceMatchCols.ref_col}_score2'] > MotifMatchArgs.min_composite_score)
    )
    mask_multimer_encodecluster_nondirect = (
        mask_dimer_encodecluster_nondirect | mask_trimer_encodecluster_nondirect
    )
    mask_matched_atlascluster_nondirect = (
        mask_monomer_encodecluster_nondirect | mask_dimer_encodecluster_nondirect | mask_trimer_encodecluster_nondirect
    )

    # Dimer-style notation
    mc_encode_composite.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_col] = mc_encode_composite.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_encode_composite.metadata.loc[mask_matched_atlascluster_nondirect, LabelCols.label_core_readable_col] = mc_encode_composite.metadata.loc[mask_matched_atlascluster_nondirect, ref_readable_name_cols[0]]
    mc_encode_composite.metadata.loc[mask_dimer_encodecluster_nondirect, LabelCols.label_flank_col] = mc_encode_composite.metadata.loc[mask_dimer_encodecluster_nondirect, ref_readable_name_cols[1]]
    if mask_trimer_encodecluster_nondirect.any():
        mc_encode_composite.metadata.loc[mask_trimer_encodecluster_nondirect, LabelCols.label_flank_col] = mc_encode_composite.metadata.loc[mask_trimer_encodecluster_nondirect, ref_readable_name_cols[1]] + "::" + mc_encode_composite.metadata.loc[mask_trimer_encodecluster_nondirect, ref_readable_name_cols[2]]
    # mc_encode_composite.metadata.loc[mask_dimer_encodecluster_nondirect, num_subunit_col] = 2
    # mc_encode_composite.metadata.loc[mask_trimer_encodecluster_nondirect, num_subunit_col] = 3
    # mc_encode_composite.metadata[MetadataCols.composite_col] = mc_encode_composite.metadata[num_subunit_col] > min_composite_subunit

    # # Trimer-style notation
    # mc_encode_composite[ref_readable_name_col] = mc_encode_composite.metadata[
    #     [
    #         f'{ReferenceMatchCols.ref_readable_col}_name{submotif_i}' 
    #         for submotif_i in range(MotifMatchArgs.max_submotifs)
    #     ]
    # ].agg(
    #     lambda x: "::".join(filter(pd.notna, x)), axis=1
    # )

    # Concatenate all unique annotations
    mc_encode_composite[ReferenceMatchCols.ref_name_col] = mc_encode_composite.metadata[
        [f'{ReferenceMatchCols.ref_col}_name{submotif_i}' for submotif_i in range(MotifMatchArgs.max_submotifs)]
    ].agg(
        lambda x: ",".join(pd.unique(x.dropna().astype(str))),
        axis=1
    )

    # Assign new labels
    # Count occurrence: Core TF family (Direct)
    mc_encode_composite_all[LabelCols.direct_label_core_readable_col_set] = mc_encode_composite_all[LabelCols.direct_label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_encode_composite_all_explode_core_readable_direct = mc_encode_composite_all.metadata.explode(LabelCols.direct_label_core_readable_col_set)

    # Count occurrence: Core TF family
    mc_encode_composite_all[LabelCols.label_core_readable_col_set] = mc_encode_composite_all[LabelCols.label_core_readable_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_encode_composite_all_explode_core_readable = mc_encode_composite_all.metadata.explode(LabelCols.label_core_readable_col_set)

    # Count occurrence: Composite TF family
    mc_encode_composite_all[LabelCols.label_flank_col_set] = mc_encode_composite_all[LabelCols.label_flank_col].apply(lambda x: set(re.split(r",|-", x)) if pd.notna(x) else x)
    mc_encode_composite_all_explode_flank = mc_encode_composite_all.metadata.explode(LabelCols.label_flank_col_set)

    ## Count occurrences: Create pivot table
    # Core: Direct: Monomer. Metadata concatenation can serialize the
    # composite flag as a string, so normalize it before boolean masking.
    mc_encode_composite_all_is_composite = (
        mc_encode_composite_all_explode_core_readable_direct[MetadataCols.composite_col]
        .astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    )
    mc_encode_composite_core_readable_direct_monomer_pivot = pd.pivot_table(
        mc_encode_composite_all_explode_core_readable_direct.loc[~mc_encode_composite_all_is_composite],
        index=CompositeArgs.soft_nosynergy_merge_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_encode_composite_core_readable_direct_monomer_pivot = mc_encode_composite_core_readable_direct_monomer_pivot.div(
        mc_encode_composite_core_readable_direct_monomer_pivot.sum(axis=1), axis=0)

    # Core: Direct: All
    mc_encode_composite_core_readable_direct_pivot = pd.pivot_table(
        mc_encode_composite_all_explode_core_readable_direct,
        index=CompositeArgs.soft_nosynergy_merge_cluster_col,
        columns=LabelCols.direct_label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_encode_composite_core_readable_direct_pivot = mc_encode_composite_core_readable_direct_pivot.div(
        mc_encode_composite_core_readable_direct_pivot.sum(axis=1), axis=0)

    # Core: All
    mc_encode_composite_core_readable_pivot = pd.pivot_table(
        mc_encode_composite_all_explode_core_readable,
        index=CompositeArgs.soft_nosynergy_merge_cluster_col,
        columns=LabelCols.label_core_readable_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_encode_composite_core_readable_pivot = mc_encode_composite_core_readable_pivot.div(
        mc_encode_composite_core_readable_pivot.sum(axis=1), axis=0)

    # Composite
    mc_encode_composite_flank_pivot = pd.pivot_table(
        mc_encode_composite_all_explode_flank,
        index=CompositeArgs.soft_nosynergy_merge_cluster_col,
        columns=LabelCols.label_flank_col_set,
        values=MetadataCols.total_contrib_score_col,
        aggfunc='sum',
        fill_value=0
    )
    mc_encode_composite_flank_pivot = mc_encode_composite_flank_pivot.div(
        mc_encode_composite_flank_pivot.sum(axis=1), axis=0)

    # Assign labels: Functions
    def is_valid(x):
        """Pandas-args.safe validity check"""
        return pd.notna(x) and x != ""


    def assign_direct_label_core(row):
        """Assign label using weighted occurrence from constituents"""
        # Initialize
        direct_label_core = None
        direct_label_core_readable = None
        source_cluster = row[CompositeArgs.soft_nosynergy_merge_cluster_col]

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
            and (source_cluster in mc_encode_composite_core_readable_direct_monomer_pivot.index)
        ):
            core_readable_direct_monomer_frac = mc_encode_composite_core_readable_direct_monomer_pivot.loc[source_cluster]

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
            and (source_cluster in mc_encode_composite_core_readable_direct_pivot.index)
        ):
            core_readable_direct_frac = mc_encode_composite_core_readable_direct_pivot.loc[source_cluster]

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
        source_cluster = row[CompositeArgs.soft_nosynergy_merge_cluster_col]

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
            and (source_cluster in mc_encode_composite_core_readable_pivot.index)
        ):
            label_core_readable_frac = mc_encode_composite_core_readable_pivot.loc[source_cluster]
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
        source_cluster = row[CompositeArgs.soft_nosynergy_merge_cluster_col]

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
            and (source_cluster in mc_encode_composite_flank_pivot.index)
        ):
            composite_family_frac = mc_encode_composite_flank_pivot.loc[source_cluster]
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

    # Add human-readable labels: Direct (Rule 1) and label (Rules 1-4), in one pass over the same annotated rows
    label_cols = [
        LabelCols.direct_label_core_col, LabelCols.direct_label_core_readable_col, LabelCols.direct_label_col,
        LabelCols.label_core_col, LabelCols.label_core_readable_col, LabelCols.label_flank_col, LabelCols.label_col,
    ]
    labels = pd.DataFrame(
        mc_encode_composite.metadata.apply(assign_labels, axis=1).tolist(),
        index=mc_encode_composite.metadata.index,
        columns=label_cols,
    )
    for col in label_cols:
        mc_encode_composite[col] = labels[col]

    # Restore full result, preserving the rows outside this reannotation subset.
    for col in mc_encode_composite.metadata:
        if col not in mc_encode_composite_merged.metadata:
            mc_encode_composite_merged.metadata[col] = None
        if len(soft_reannotation_idx):
            mc_encode_composite_merged.metadata.loc[soft_reannotation_idx, col] = (
                mc_encode_composite.metadata[col].to_numpy()
            )
    for image_name in mc_encode_composite.images():
        images = (list(mc_encode_composite_merged.get_images(image_name))
                  if image_name in mc_encode_composite_merged.images()
                  else [None] * len(mc_encode_composite_merged))
        for idx, value in zip(soft_reannotation_idx, mc_encode_composite.get_images(image_name)):
            images[idx] = value
        mc_encode_composite_merged.set_images(image_name, images)
    mc_encode_composite = mc_encode_composite_merged

    def assign_direct_composite(
        target_mc,
        direct_reference_mc,
        indirect_reference_mc,
        min_score,
        max_submotifs=1,
        label_unsigned=True,
        save_images=True,
        logo_trimming=True,
        save_col_prefix="match",
    ):
        """Assign a direct core first, then optional residual submotif matches.

        This is local rather than imported from ``pipeline_encode.py`` because that
        pipeline binds a different set of global configuration classes.  The
        implementation uses this notebook's column names and MotifCompendium APIs.
        """
        if not len(target_mc) or not len(direct_reference_mc) or MotifMatchArgs.max_submotifs < 1:
            return

        target = target_mc.motifs.copy()
        direct = direct_reference_mc.motifs.copy()
        indirect = indirect_reference_mc.motifs.copy()
        max_length = max(target.shape[1], direct.shape[1], indirect.shape[1])

        def pad(motifs):
            return np.pad(motifs, ((0, 0), (0, max_length - motifs.shape[1]), (0, 0)))

        target, direct, indirect = pad(target), pad(direct), pad(indirect)
        direct_raw, indirect_raw = direct.copy(), indirect.copy()
        if label_unsigned:
            target, direct, indirect = np.abs(target), np.abs(direct), np.abs(indirect)

        ic_scale = utils_config.get_ic_scale()
        if ic_scale:
            target, direct, indirect = (
                utils_motif.ic_scale(target),
                utils_motif.ic_scale(direct),
                utils_motif.ic_scale(indirect),
            )
            utils_config.set_ic_scale(False)

        def normalize(motifs):
            norms = np.linalg.norm(motifs, axis=(1, 2), keepdims=True)
            return np.divide(motifs, norms, out=np.zeros_like(motifs), where=norms != 0)

        try:
            working = normalize(target)
            direct_norm, indirect_norm = normalize(direct), normalize(indirect)
            reference_norm = np.concatenate([direct_norm, indirect_norm], axis=0)
            reference_raw = np.concatenate([direct_raw, indirect_raw], axis=0)
            direct_names = direct_reference_mc[MetadataCols.name_col].tolist()
            reference_names = direct_names + indirect_reference_mc[MetadataCols.name_col].tolist()
            active = np.ones(len(working), dtype=bool)

            for submotif_i in range(MotifMatchArgs.max_submotifs):
                candidates = direct_norm if submotif_i == 0 else reference_norm
                candidate_raw = direct_raw if submotif_i == 0 else reference_raw
                candidate_names = direct_names if submotif_i == 0 else reference_names
                if not len(candidates) or not active.any():
                    break
                similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
                    [working, candidates], [(0, 1)]
                )[0]
                target_norm = np.linalg.norm(working, axis=(1, 2))[:, None]
                candidate_norm = np.linalg.norm(candidates, axis=(1, 2))[None, :]
                similarity *= target_norm * candidate_norm
                match_idx = similarity.argmax(axis=1)
                match_score = similarity[np.arange(len(working)), match_idx] * np.sqrt(submotif_i + 1)
                active &= match_score >= min_score
                match_score = np.where(active, match_score, 0.0)
                match_motifs = candidate_raw[match_idx].copy()
                match_motifs[~active] = 0
                match_labels = [candidate_names[idx] if keep else None for idx, keep in zip(match_idx, active)]
                target_mc[f"{save_col_prefix}_score{submotif_i}"] = match_score
                target_mc[f"{save_col_prefix}_name{submotif_i}"] = match_labels
                if save_images:
                    target_mc.add_logos(match_motifs, f"{save_col_prefix}_logo{submotif_i}", logo_trimming)
                if not active.any():
                    break
                working = utils_motif.remove_motif_component(
                    working,
                    match_motifs,
                    alignment_rc[np.arange(len(working)), match_idx],
                    alignment_h[np.arange(len(working)), match_idx],
                )
                working[~active] = 0
        finally:
            if ic_scale:
                utils_config.set_ic_scale(True)


    ### Annotate: Best label
    # Rebuild all best-label fields on each run.  This prevents old annotations or
    # image columns from surviving a notebook rerun for rows without a valid label.
    mc_encode_composite.metadata.drop(
        columns=[
            col for col in mc_encode_composite.metadata
            if col.startswith(f"{LabelCols.best_label_ref_col}_")
        ],
        inplace=True,
    )
    existing_best_label_images = [
        image_name for image_name in mc_encode_composite.images()
        if image_name.startswith(f"{LabelCols.best_label_ref_col}_")
    ]
    if existing_best_label_images:
        mc_encode_composite.delete_images(existing_best_label_images)
    for submotif_i in range(CompositeArgs.max_submotifs_composite):
        mc_encode_composite[f"{LabelCols.best_label_ref_col}_name{submotif_i}"] = None
        mc_encode_composite[f"{LabelCols.best_label_ref_col}_score{submotif_i}"] = 0.0

    mc_encode_composite.metadata["_motif_row"] = np.arange(len(mc_encode_composite))
    mc_encode_composite_metadata_updated = mc_encode_composite.metadata.set_index(MetadataCols.unique_id_col)
    mc_encode_composite_images_updated = pd.DataFrame(index=mc_encode_composite_metadata_updated.index)

    # Update Core, Core (Direct), Flank
    mc_encode_composite[LabelCols.label_readable_col] = mc_encode_composite[LabelCols.label_col]
    mc_encode_composite[LabelCols.label_core_col] = np.where(
        mc_encode_composite[LabelCols.label_col].str.contains("::", na=False),
        mc_encode_composite[LabelCols.label_col].str.split("::").str[0],
        mc_encode_composite[LabelCols.label_col]
    )
    mc_encode_composite[LabelCols.direct_label_core_col] = np.where(
        mc_encode_composite[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[:4]), na=False),
        np.where(
            mc_encode_composite[LabelCols.label_col].str.contains("::", na=False),
            mc_encode_composite[LabelCols.label_col].str.split("::").str[0],
            mc_encode_composite[LabelCols.label_col]
        ),
        None
    )
    mc_encode_composite[LabelCols.label_flank_col] = np.where(
        mc_encode_composite[LabelCols.label_col].str.contains("::", na=False),
        mc_encode_composite[LabelCols.label_col].str.split("::").str[1],
        None
    )

    for motif_row, (_, row) in enumerate(mc_encode_composite.metadata.iterrows()):
        # MotifCompendium: Subset
        mc_encode_composite_i = mc_encode_composite[[motif_row]]
        label_core_i = re.split(r'[-/]', row[LabelCols.label_core_col]) if pd.notna(row[LabelCols.label_core_col]) else []
        direct_label_core_i = re.split(r'[-/]', row[LabelCols.direct_label_core_col]) if pd.notna(row[LabelCols.direct_label_core_col]) else []
        label_flank_i = re.split(r'[-/]', row[LabelCols.label_flank_col]) if pd.notna(row[LabelCols.label_flank_col]) else []
        composite_i = str(row[MetadataCols.composite_col]).strip().lower() in {"true", "1", "1.0"}
        max_submotifs_i = 2 if composite_i else 1

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
                target_mc=mc_encode_composite_i,
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
        mc_encode_composite_metadata_updated = mc_encode_composite_metadata_updated.combine_first(mc_encode_composite_i.metadata.set_index(MetadataCols.unique_id_col))
        mc_encode_composite_metadata_updated.update(mc_encode_composite_i.metadata.set_index(MetadataCols.unique_id_col))
        new_image_cols = [image_col for image_col in mc_encode_composite_i.images() if image_col.startswith(LabelCols.best_label_ref_col)]
        mc_encode_composite_i_new_images = pd.DataFrame({
            image_col: pd.Series(
                mc_encode_composite_i.get_images(image_col),
                index=mc_encode_composite_i.metadata.set_index(MetadataCols.unique_id_col).index,
            )
            for image_col in new_image_cols
        })
        mc_encode_composite_images_updated = mc_encode_composite_images_updated.combine_first(mc_encode_composite_i_new_images)
        mc_encode_composite_images_updated.update(mc_encode_composite_i_new_images)

    # Apply updates: Metadata, Images
    mc_encode_composite.metadata = (
        mc_encode_composite_metadata_updated.reset_index()
        .sort_values("_motif_row", kind="stable")
        .drop(columns="_motif_row")
        .reset_index(drop=True)
    )
    for image_col in mc_encode_composite_images_updated.columns:
        mc_encode_composite.set_images(
            image_name=image_col,
            images=[x if pd.notna(x) else None for x in mc_encode_composite_images_updated
                    .reindex(mc_encode_composite[MetadataCols.unique_id_col].to_numpy())[image_col]],
        )
    del mc_encode_composite_metadata_updated, mc_encode_composite_images_updated


    ## Set label readable type
    # Filtered
    label_conf_conditions = [
        mc_encode_composite[DirectTypeArgs.direct_type_col].str.contains(DirectTypeArgs.direct_types[0], na=False),  # High: A
        mc_encode_composite[DirectTypeArgs.direct_type_col].str.contains("|".join(DirectTypeArgs.direct_types[1:3]), na=False),  # Medium: B-NZ, B-Z
        ~mc_encode_composite[FilterArgs.filter_flag_col].astype(str).str.lower().isin(["true", "1", "1.0"]), # Low: Not flagged
        mc_encode_composite[FilterArgs.filter_flag_col].astype(str).str.lower().isin(["true", "1", "1.0"]),  # Flagged
    ]
    mc_encode_composite[LabelCols.label_conf_col] = np.select(
        label_conf_conditions,  # High, Medium, low, Flagged
        LabelCols.label_confs, # High, Medium, low, Flagged
        default=LabelCols.label_confs[-1]  # Remaining: Flagged
    )


    ## Simple name: ({+_-})_{prename}_{H_M_L}
    sign = np.where(mc_encode_composite[MetadataCols.posneg_col] == "pos", "pos", "neg")
    conf = np.select([
        mc_encode_composite[LabelCols.label_conf_col] == LabelCols.label_confs[0],
        mc_encode_composite[LabelCols.label_conf_col] == LabelCols.label_confs[1],
        mc_encode_composite[LabelCols.label_conf_col] == LabelCols.label_confs[2],
        ],
        LabelCols.label_confs_simple[:-1],
        default=LabelCols.label_confs_simple[2],
    )

    mc_encode_composite[MetadataCols.prename_col] = np.where(
        ~mc_encode_composite[LabelCols.label_readable_col].isin(FilterArgs.filter_flag_types),
        (mc_encode_composite[LabelCols.label_readable_col]
            + "_" + sign
            + "_" + conf),
        mc_encode_composite[LabelCols.label_readable_col],
    )

    # Simple name = Prename
    mc_encode_composite[MetadataCols.simple_name_col] = mc_encode_composite[MetadataCols.prename_col]

    # Show the final consensus motif in the duplicate-logo column for merged rows.
    duplicate_logo_motifs = np.where(
        (mc_encode_composite.metadata[CompositeArgs.duplicate_num_members_col].to_numpy() > 1)[:, None, None],
        mc_encode_composite.motifs,
        0,
    )
    mc_encode_composite.add_logos(
        motifs=duplicate_logo_motifs,
        image_name=CompositeArgs.duplicate_logo_col,
        trim=VisualizeArgs.html_logo_trim,
    )

    # #### Remove: No synergy

    # Remove: No synergy
    mc_encode_composite_nosynergy = mc_encode_composite[mc_encode_composite[CompositeArgs.ism_class_col] == "no synergy"]
    mc_encode_composite = mc_encode_composite[mc_encode_composite[CompositeArgs.ism_class_col] != "no synergy"]

    # #### Plot

    # HTML summary table: ISM cooperativity
    os.makedirs(os.path.dirname(cpaths.html_composite_ism_dedup_path), exist_ok=True)
    mc_encode_composite.summary_table_html(
        html_out=cpaths.html_composite_ism_dedup_path,
        columns=CompositeArgs.html_columns_composite,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict,
    )

    # Save MotifCompendium: ENCODE composite with ISM metadata and merged duplicates
    save_replace(mc_encode_composite, cpaths.mc_composite_ism_dedup_path)

    # ### Re-combine all: ENCODE MotifCompendium (C5)

    # ENCODE: Single
    mc_encode_single = mc_encode[(mc_encode[MetadataCols.composite_col] == False)]

    # Re-Combine: Full ENCODE MotifCompendium
    mc_encode_columns = list(set(mc_encode_single.columns() + mc_encode_composite.columns()))
    mc_encode_images = list(set(mc_encode_single.images() + mc_encode_composite.images()))

    for mc_encode_i in [mc_encode_single, mc_encode_composite]:
        for col in mc_encode_columns:
            if col not in mc_encode_i.columns():
                mc_encode_i.metadata[col] = [None] * len(mc_encode_i)
        for image_col in mc_encode_images:
            if image_col not in mc_encode_i.images():
                mc_encode_i.set_images(image_name=image_col, images=[None] * len(mc_encode_i))

    mc_encode = MotifCompendium.combine(
        [mc_encode_single, mc_encode_composite],
        safe=args.safe,
    )

    ## Name: MCID
    # MCID
    mc_encode[MetadataCols.unique_id_col] = unique_id_initial + f".{CompositeArgs.mcid_level}." + mc_encode.metadata.index.map('{:08d}'.format)

    # HTML report
    mc_encode.summary_table_html(
        html_out=cpaths.html_encode_vf_path,
        columns=[col for col in VisualizeArgs.html_columns if col in (mc_encode.columns() + mc_encode.images())],
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict,
    )

    # Save: Full ENCODE MotifCompendium
    save_replace(mc_encode, cpaths.mc_encode_vf_path)

    ### FINAL ENCODE MOTIFCOMPENDIUM -----------------------------------------------------------------------------
    logging.info("Final ENCODE MotifCompendium: the C5 ENCODE clusters and the removed ones...")
    (mc_encode_final, mc_atlascluster_combined, atlascluster2encode_name_map, atlascluster2encode_simple_map,
     num_left_out) = final_encode_compendium(
        mc_encode, mc_encode_c4_filtered, mc_encode_c4_removed, mc_encode_composite_nosynergy,
        mc_atlascluster_combined, encodecluster_cluster_col, safe=args.safe,
    )

    # #### Export: ENCODE compendia, motif index and every level below (as the ENCODE stage)
    pipeline_encode.export_encode_levels(
        args,
        mc_encode_final,
        mc_atlascluster_combined,
        atlascluster2encode_name_map,
        atlascluster2encode_simple_map,
        encodecluster_cluster_col,
        html_column_desc_dict,
    )

    # Args JSON
    args_record = {
        "parser_args": vars(args),
        "configs": pipeline_encode.collect_configs_snapshot(),
        "composite_dir": cpaths.composite_dir,
        "num_encode_filtered": int((~mc_encode_final[FilterArgs.filter_flag_col]).sum()),
        "num_encode_removed": int(mc_encode_final[FilterArgs.filter_flag_col].sum()),
        "num_left_out_of_c5": num_left_out,
        "log_path": log_path,
    }
    args_json_path = os.path.join(paths.metadata_dir, "args_composite.json")
    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_record, f, indent=2, default=str)

    logging.info(f"Completed run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


#### MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()
    set_default_configs()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("ENCODE composites: run started.")

    stage_dir_args = pipeline_encode.atlas_stage_dir_args(args.tfatlas_dir, args.chromatlas_dir, args.procapatlas_dir)
    if args.time:
        start_time = time.time()
    run_encode_composite(pipeline_encode.stage_args(args, **stage_dir_args))
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    logging.info(f'ENCODE composites: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
