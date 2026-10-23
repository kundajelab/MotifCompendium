#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: index and annotate the FiNeMo hits with the final ENCODE clusters.

Runs after the ENCODE compendium is final: pipeline_encode.py, then (with the composites merged)
pipeline_encode_composite.py. From their outputs (--output-dir: the ENCODE stage's directory) and the
atlas pipelines' per-head motif compendia (their FiNeMo and peaks paths, per experiment and head):

  1. Index each experiment's FiNeMo hits with the ENCODE cluster of its motif (bgzip + tabix BEDs,
     both heads collapsed): <output-dir>/export/hits_indexed
  2. Annotate each hit: peak type, nearest gene and TSS, GC/N content, distance to the peak summit
     (ArchR-style, from GENCODE and hg38): <output-dir>/export/hits_annotated
  3. Summarize the hits per ENCODE cluster (counts per peak type, distances to the TSS and the peak
     summit, figures) into the filtered and removed ENCODE compendia: .mc, HTML and metadata TSV.

Moved from pipeline_encode.py, unchanged.
"""
import os
import sys
import time
import argparse
import logging
import gzip
import json
import importlib.util

import numpy as np
import pandas as pd
import pysam

import MotifCompendium


# The ENCODE stage (its configs, paths and save_replace): the module pipeline_encode_full.py has loaded, else this
# script's sibling
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
        description="ENCODE MotifCompendium: index and annotate the FiNeMo hits with the final ENCODE clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output: the ENCODE stage's directory
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory of the ENCODE stage (pipeline_encode.py --output-dir).")
    parser.add_argument("--tfatlas-dir", type=str, required=True,
                        help="Output root of the TF atlas pipeline (pipeline/bpnet/pipeline_bpnet.py).")
    parser.add_argument("--chromatlas-dir", type=str, required=True,
                        help="Output root of the Chromatin atlas pipeline (pipeline/chrombpnet/pipeline_chrombpnet.py).")
    parser.add_argument("--procapatlas-dir", type=str, required=True,
                        help="Output root of the ProCap atlas pipeline (pipeline/procapnet/pipeline_procapnet.py).")

    # Input: annotation
    parser.add_argument("--hg38-fasta", type=str, required=True,
                        help="hg38 genome FASTA, for hits annotation.")
    parser.add_argument("--gencode-gtf", type=str, required=True,
                        help="GENCODE annotation GTF (.gz), for hits annotation.")
    parser.add_argument("--column-desc-tsv", type=str, required=True,
                        help="TSV describing HTML summary table column names.")

    # Compute
    parser.add_argument("-cp", "--max-cpus", type=int, default=1, help="Maximum number of CPUs to use.")
    parser.add_argument("--unsafe", action="store_false", dest="safe",
                        help="Do not check the validity of the MotifCompendium objects.")
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


#### HITS INDEXING: WORKERS ---------------------------------------------------
# Each FiNeMo hit with the ENCODE cluster of its motif, as an indexed BED (bgzip + tabix); counts and profile
# hits of an experiment are then collapsed into one BED. From sherlock/runs/encode/finemo/apply_mc/update_hits_bed.py.
HITS_BED_COLS = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
HITS_INDEXED_COLS = HITS_BED_COLS + ['category', 'confidence', 'head', 'name_tfmodisco', 'hits_bed_cluster']
HITS_CONFIDENCES = ['high', 'medium', 'low', 'flag']  # label_confidence, best first
HITS_CONFIDENCES_SIMPLE = ['high', 'medium', 'low', '']
HITS_COLORS = {'high': '0,128,0', 'medium': '255,255,0', 'low': '255,0,0'}  # Green, yellow, red; others grey
HITS_GREY = '128,128,128'  # Flag, non-canonical, filtered
HITS_NAME_UNKNOWN = 'unvalidated'
HITS_NAME_NONE = 'None'
HITS_CATEGORY_NONCANONICAL = 'non-canonical'
HITS_MOTIF_COLS = ['name_tfmodisco', 'hits_bed_name', 'hits_bed_cluster', 'label_confidence', 'category', 'flagged']


def _hits_pattern_key(posneg, names):
    # "pos" / "neg" + "." + pattern number (the trailing "pattern_{#}"), as in add_num_hits() of the atlas pipelines
    return posneg.astype(str).str[:3] + "." + names.astype(str).str.extract(r"(?:^|[^A-Za-z])pattern_(\d+)$", expand=False)


def _write_bed_tabix(bed_df, bed_gz_path):
    # Sort, write, then bgzip + tabix-index: writes bed_gz_path and bed_gz_path.tbi (removes the plain BED)
    os.makedirs(os.path.dirname(bed_gz_path), exist_ok=True)
    bed_plain_path = bed_gz_path.removesuffix(".gz")
    bed_df.sort_values(['chrom', 'start', 'end']).to_csv(bed_plain_path, sep="\t", header=False, index=False)
    pysam.tabix_index(bed_plain_path, preset="bed", force=True)


def _index_hits_bed(hits_bed_in, hits_bed_out, motif_lookup, head):
    """FiNeMo hits of one experiment and head, each with the ENCODE cluster of its motif.

    motif_lookup: pos/neg pattern key -> HITS_MOTIF_COLS values. Hits of a motif not in it are unvalidated,
    without a cluster (None), flagged and grey, as in update_hits_bed.py.
    """
    hits_df = pd.read_csv(hits_bed_in, sep="\t", header=None, usecols=range(6), names=HITS_BED_COLS[:6], keep_default_na=False)
    motifs = pd.DataFrame.from_dict(motif_lookup, orient="index", columns=HITS_MOTIF_COLS)
    matched = motifs.reindex(_hits_pattern_key(hits_df['name'].str.split(".").str[0], hits_df['name']).to_numpy())
    confidence = matched['label_confidence'].fillna(HITS_CONFIDENCES[-1]).to_numpy()
    category = matched['category'].fillna(HITS_CATEGORY_NONCANONICAL).to_numpy()
    flagged = matched['flagged'].fillna(True).astype(bool).to_numpy()

    hits_df['name_tfmodisco'] = matched['name_tfmodisco'].fillna(pd.Series(hits_df['name'].to_numpy(), index=matched.index)).to_numpy()
    hits_df['name'] = matched['hits_bed_name'].fillna(HITS_NAME_UNKNOWN).to_numpy()
    hits_df['hits_bed_cluster'] = matched['hits_bed_cluster'].fillna(HITS_NAME_NONE).to_numpy()
    hits_df['thickStart'] = hits_df['start']
    hits_df['thickEnd'] = hits_df['end']
    hits_df['head'] = head
    hits_df['category'] = category
    hits_df['confidence'] = confidence
    item_rgb = pd.Series(confidence).map(HITS_COLORS).fillna(HITS_GREY).to_numpy()
    item_rgb[(category == HITS_CATEGORY_NONCANONICAL) | flagged] = HITS_GREY
    hits_df['itemRgb'] = item_rgb
    _write_bed_tabix(hits_df[HITS_INDEXED_COLS], hits_bed_out)
    return len(hits_df)


def _collapse_overlapping_hits(df, deduplicate_col):
    """Collapse overlapping hits sharing chrom + deduplicate_col (not for None); keep the most confident label."""
    df = df.sort_values(['chrom', deduplicate_col, 'start', 'end']).reset_index(drop=True)
    confidence_rank = {confidence: i for i, confidence in enumerate(HITS_CONFIDENCES)}

    chroms = df['chrom'].to_numpy()
    starts = df['start'].to_numpy()
    ends = df['end'].to_numpy()
    names = df['name'].to_numpy()
    scores = df['score'].to_numpy()
    strands = df['strand'].to_numpy()
    rgbs = df['itemRgb'].to_numpy()
    categories = df['category'].to_numpy()
    confidences = df['confidence'].to_numpy()
    heads = df['head'].astype(str).to_numpy()
    tfmodiscos = df['name_tfmodisco'].astype(str).to_numpy()
    clusters = df['hits_bed_cluster'].to_numpy()
    conditions = df[deduplicate_col].to_numpy()

    merged_rows = []
    n = len(df)
    i = 0
    while i < n:
        curr_chrom, curr_start, curr_end = chroms[i], starts[i], ends[i]
        curr_name, curr_score, curr_strand, curr_rgb = names[i], scores[i], strands[i], rgbs[i]
        curr_category, curr_confidence = categories[i], confidences[i]
        curr_cluster, curr_condition = clusters[i], conditions[i]
        curr_heads, curr_tfmodisco = [heads[i]], [tfmodiscos[i]]

        j = i + 1
        while (
            j < n
            and chroms[j] == curr_chrom
            and conditions[j] == curr_condition
            and curr_condition != HITS_NAME_NONE
            and starts[j] < curr_end
        ):
            curr_end = max(curr_end, ends[j])
            curr_heads.append(heads[j])
            curr_tfmodisco.append(tfmodiscos[j])
            if confidence_rank.get(curr_confidence, float('inf')) > confidence_rank.get(confidences[j], float('inf')):
                curr_name, curr_score, curr_strand, curr_rgb = names[j], scores[j], strands[j], rgbs[j]
                curr_category, curr_confidence = categories[j], confidences[j]
            j += 1

        merged_rows.append({
            'chrom': curr_chrom, 'start': curr_start, 'end': curr_end,
            'name': curr_name, 'score': curr_score, 'strand': curr_strand,
            'thickStart': curr_start, 'thickEnd': curr_end, 'itemRgb': curr_rgb,
            'category': curr_category, 'confidence': curr_confidence,
            'head': ';'.join(dict.fromkeys(curr_heads)),
            'name_tfmodisco': ';'.join(dict.fromkeys(curr_tfmodisco)),
            'hits_bed_cluster': curr_cluster,
        })
        i = j

    return pd.DataFrame(merged_rows, columns=HITS_INDEXED_COLS)


def _combine_hits_beds(hits_beds_in, hits_bed_out):
    """Indexed hits of an experiment's heads, combined; overlapping hits of an ENCODE cluster collapsed."""
    hits_df = pd.concat(
        [pd.read_csv(path, sep="\t", header=None, names=HITS_INDEXED_COLS, keep_default_na=False) for path in hits_beds_in],
        axis=0, ignore_index=True,
    )
    hits_df = _collapse_overlapping_hits(hits_df, 'hits_bed_cluster')
    _write_bed_tabix(hits_df, hits_bed_out)
    return len(hits_df)


#### HITS ANNOTATION: WORKERS ------------------------------------------------
# ProcessPoolExecutor pickles the function that it runs, so that function must be module-level
# (a function nested in run_encode_hits() cannot be pickled). run_encode_hits() puts its
# annotation closure here before the (forked) pool starts, so the workers inherit it.
_HITS_WORKER = {}


def _hits_worker_init(hg38_fasta_path):
    # Each worker opens its own handle: pysam.FastaFile isn't safe to share across processes
    # (the underlying fd's read position would race)
    import pysam
    _HITS_WORKER["fasta"] = pysam.FastaFile(hg38_fasta_path)


def _hits_worker_run(hits_tsv, peaks_bed, hits_tsv_annotated):
    return _HITS_WORKER["annotate_row"](hits_tsv, peaks_bed, hits_tsv_annotated, _HITS_WORKER["fasta"])


def _summarize_hits_file(hits_tsv_annotated):
    # Per annotated hits file: hits per (cluster, peak type), and distances to the TSS / peak summit per cluster
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


def run_encode_hits(args):
    """ENCODE: index and annotate the FiNeMo hits with the final ENCODE clusters"""

    ### OPTIONS -----------------------------------------------------------------
    # Run log, in <output-dir>/logs/, as in the ENCODE stage
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_pipeline_hits_{os.path.basename(os.path.normpath(args.output_dir))}"
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

    ### PATHS --------------------------------------------------------------------
    paths = pipeline_encode.encode_paths(args)

    ## HTML column descriptions
    html_column_desc_df = pd.read_csv(args.column_desc_tsv, sep="\t", header=0)
    html_column_desc_dict = html_column_desc_df.set_index('column_name')['column_desc'].to_dict()

    ### INPUTS -------------------------------------------------------------------
    # The final ENCODE compendium: filtered and removed ENCODE clusters, and the motif index (every motif with its
    # ENCODE cluster, final filter flag, confidence and category)
    logging.info("Loading the final ENCODE compendia and the motif index...")
    mc_encode_filtered = MotifCompendium.load(paths.mc_encode_filtered_path, safe=args.safe)
    mc_encode_removed = MotifCompendium.load(paths.mc_encode_removed_path, safe=args.safe)
    mc_motif_index = pd.read_csv(paths.metadata_motif_index_path, sep="\t", keep_default_na=False, na_values=[""])

    # FiNeMo and peaks paths, per experiment and head: the per-head motif compendia (metadata only), for the hits
    # indexing and annotation below
    hits_finemo_path_cols = [MetadataCols.id_col, MetadataCols.head_col, MetadataCols.finemo_path_col]
    hits_finemo_paths_dfs = []
    for atlas_id, mc_head_path in (
        (EncodeCombineArgs.tfatlas_id, paths.mc_counts_tf_path), (EncodeCombineArgs.tfatlas_id, paths.mc_profile_tf_path),
        (EncodeCombineArgs.chromatlas_id, paths.mc_counts_chrom_path), (EncodeCombineArgs.chromatlas_id, paths.mc_profile_chrom_path),
        (EncodeCombineArgs.procapatlas_id, paths.mc_counts_procap_path), (EncodeCombineArgs.procapatlas_id, paths.mc_profile_procap_path),
    ):
        mc_head_metadata = pd.read_hdf(mc_head_path, key="metadata")
        if MetadataCols.finemo_path_col in mc_head_metadata.columns:
            hits_finemo_paths_dfs.append(
                mc_head_metadata[[col for col in hits_finemo_path_cols + [MetadataCols.peaks_path_col] if col in mc_head_metadata.columns]]
                .assign(**{VisualizeArgs.atlas_col: atlas_id})
            )
    hits_finemo_paths_df = pd.concat(
        hits_finemo_paths_dfs or [pd.DataFrame(columns=hits_finemo_path_cols + [VisualizeArgs.atlas_col])],
        ignore_index=True,
    )
    if MetadataCols.peaks_path_col not in hits_finemo_paths_df.columns:
        hits_finemo_paths_df[MetadataCols.peaks_path_col] = None


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

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Index hits, per experiment: its FiNeMo hits (both heads), each with the ENCODE cluster of its motif, as
    # indexed BEDs (bgzip + tabix) in <export>/hits_indexed: {ID}.{head}.hits.indexed.bed.gz and, both heads
    # collapsed, {ID}.hits.indexed.combined.bed.gz (from sherlock/runs/encode/finemo/apply_mc/update_hits_bed.py).
    # FiNeMo paths: the motifs' finemo_path (from the atlas pipelines' --{head}-finemo-paths), read above.
    logging.info("Indexing FiNeMo hits with their ENCODE clusters...")
    hits_indexed_dir = os.path.join(paths.export_dir, "hits_indexed")
    hits_index_df = mc_motif_index.copy()

    # Hits BED name and ENCODE cluster, per motif
    flagged = hits_index_df[FilterArgs.filter_flag_col].fillna(True).astype(bool)
    simple_name = hits_index_df[MetadataCols.simple_name_col]
    encode_name = hits_index_df[VisualizeArgs.encodecluster_name_col]
    confidence_simple = pd.Series(np.select(
        [hits_index_df[LabelCols.label_conf_col] == confidence for confidence in HITS_CONFIDENCES],
        HITS_CONFIDENCES_SIMPLE,
        default=HITS_CONFIDENCES_SIMPLE[-1],
    ), index=hits_index_df.index)
    hits_bed_name = simple_name.fillna(HITS_NAME_UNKNOWN)
    hits_bed_cluster = encode_name.fillna(simple_name).fillna(HITS_NAME_NONE)

    # TF atlas: Direct motifs named by their TF; others by their ENCODE simple name
    mask_tf = hits_index_df[VisualizeArgs.atlas_col] == EncodeCombineArgs.tfatlas_id
    mask_direct = hits_index_df[DirectTypeArgs.direct_type_col].isin(DirectTypeArgs.direct_types[:4])
    mask = mask_tf & ~flagged & mask_direct
    hits_bed_name[mask] = (hits_index_df[MetadataCols.tf_col] + "_" + hits_index_df[MetadataCols.posneg_col] + "_" + confidence_simple)[mask]
    mask = mask_tf & (flagged | ~mask_direct) & simple_name.astype(bool)
    hits_bed_name[mask] = simple_name[mask]
    hits_bed_cluster[mask_tf & ~flagged] = encode_name[mask_tf & ~flagged]

    # Chromatin, ProCap atlases: ENCODE simple name
    mask = hits_index_df[VisualizeArgs.atlas_col].isin([EncodeCombineArgs.chromatlas_id, EncodeCombineArgs.procapatlas_id]) & simple_name.astype(bool)
    hits_bed_name[mask] = simple_name[mask]
    hits_bed_cluster[mask] = encode_name[mask]

    # Flagged: Simple name = filter flag type (CpG, tandem-repeats, high-GC/AT, low-complexity, unvalidated)
    hits_bed_name[flagged] = simple_name[flagged]
    hits_bed_cluster[flagged] = simple_name[flagged]
    hits_index_df["hits_bed_name"] = hits_bed_name.fillna(simple_name).fillna(HITS_NAME_UNKNOWN)
    hits_index_df["hits_bed_cluster"] = hits_bed_cluster.fillna(encode_name).fillna(simple_name).fillna(HITS_NAME_NONE)
    hits_index_df["flagged"] = flagged
    hits_index_df["pattern_key"] = _hits_pattern_key(hits_index_df[MetadataCols.posneg_col], hits_index_df[MetadataCols.name_col])

    # Index: per experiment and head with a FiNeMo path
    hits_models_df = (
        hits_finemo_paths_df.dropna(subset=[MetadataCols.finemo_path_col])
        .drop_duplicates(subset=[MetadataCols.id_col, MetadataCols.head_col])
        [[MetadataCols.id_col, MetadataCols.head_col, VisualizeArgs.atlas_col, MetadataCols.finemo_path_col]]
    )
    if hits_models_df.empty:
        logging.warning("No FiNeMo paths (finemo_path) in the atlas compendia: hits are not indexed or annotated.")
    # Peaks used to call each experiment's hits, for the annotation below (the counts head's, else the profile's)
    hits_peaks_paths = (
        hits_finemo_paths_df.dropna(subset=[MetadataCols.peaks_path_col])
        .sort_values(MetadataCols.head_col)
        .drop_duplicates(subset=[MetadataCols.id_col])
        .set_index(MetadataCols.id_col)[MetadataCols.peaks_path_col]
        .to_dict()
    )
    n_workers_index = max(1, min(args.max_cpus, len(os.sched_getaffinity(0))))
    hits_indexed_paths = {}
    with ProcessPoolExecutor(max_workers=n_workers_index, mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {}
        for _, row in hits_models_df.iterrows():
            model_hits_index = hits_index_df[
                (hits_index_df[MetadataCols.id_col] == row[MetadataCols.id_col])
                & (hits_index_df[MetadataCols.head_col] == row[MetadataCols.head_col])
            ].dropna(subset=["pattern_key"]).drop_duplicates(subset=["pattern_key"])
            motif_lookup = dict(zip(model_hits_index["pattern_key"], zip(
                model_hits_index[MetadataCols.name_col], model_hits_index["hits_bed_name"], model_hits_index["hits_bed_cluster"],
                model_hits_index[LabelCols.label_conf_col], model_hits_index[DirectTypeArgs.category_type_col], model_hits_index["flagged"],
            )))
            hits_bed_out = os.path.join(hits_indexed_dir, f"{row[MetadataCols.id_col]}.{row[MetadataCols.head_col]}.hits.indexed.bed.gz")
            future = executor.submit(_index_hits_bed, row[MetadataCols.finemo_path_col], hits_bed_out, motif_lookup, row[MetadataCols.head_col])
            futures[future] = (row[MetadataCols.id_col], row[MetadataCols.head_col], hits_bed_out)
        for future in as_completed(futures):
            encid, head, hits_bed_out = futures[future]
            try:
                future.result()
                hits_indexed_paths.setdefault(encid, {})[head] = hits_bed_out
            except Exception as e:
                logging.warning(f"Indexing hits failed for {encid} ({head}): {e}")

    # Combine: both heads of an experiment, overlapping hits of an ENCODE cluster collapsed
    hits_combined_paths = {}
    with ProcessPoolExecutor(max_workers=n_workers_index, mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {}
        for encid, head_paths in hits_indexed_paths.items():
            if len(head_paths) == 1:
                hits_combined_paths[encid] = next(iter(head_paths.values()))
                continue
            hits_bed_out = os.path.join(hits_indexed_dir, f"{encid}.hits.indexed.combined.bed.gz")
            futures[executor.submit(_combine_hits_beds, [head_paths[head] for head in sorted(head_paths)], hits_bed_out)] = (encid, hits_bed_out)
        for future in as_completed(futures):
            encid, hits_bed_out = futures[future]
            try:
                future.result()
                hits_combined_paths[encid] = hits_bed_out
            except Exception as e:
                logging.warning(f"Combining indexed hits failed for {encid}: {e}")
    logging.info(f"Indexed hits: {len(hits_combined_paths)} experiments")

    # Hits metadata: per experiment, its atlas and indexed hits (for the annotation below)
    metadata_hits_df = (
        hits_models_df.drop_duplicates(subset=[MetadataCols.id_col])
        .set_index(MetadataCols.id_col)[[VisualizeArgs.atlas_col]]
    )
    metadata_hits_df["hits_indexed"] = metadata_hits_df.index.map(hits_combined_paths)
    metadata_hits_df = metadata_hits_df.dropna(subset=["hits_indexed"])
    os.makedirs(os.path.dirname(paths.metadata_hits_path), exist_ok=True)
    metadata_hits_df.to_csv(paths.metadata_hits_path, sep="\t")


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

    def annotate_hits_row(hits_tsv, peaks_bed, hits_tsv_annotated, fasta):
        hits_df = pd.read_csv(
            hits_tsv,
            sep="\t",
            header=None,
            names=NARROWPEAKS_CUSTOM_COLS,
            keep_default_na=False,
        )
        hits_anno_df = annotate_hits(hits_df, peaks_bed, fasta)
        hits_anno_df.to_csv(hits_tsv_annotated, sep="\t", index=False)
        return len(hits_anno_df)

    # The (forked) workers inherit this closure; the pool runs the module-level _hits_worker_run
    _HITS_WORKER["annotate_row"] = annotate_hits_row


    # Annotate hits, per motif -- parallelized across experiments (rows); each row
    # is independent (its own hits/peaks files), so this parallelizes cleanly.
    os.makedirs(paths.hits_annotated_dir, exist_ok=True)
    n_workers = min(8, len(os.sched_getaffinity(0)))
    mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_hits_worker_init, initargs=(args.hg38,), mp_context=mp_context) as executor:
        futures = {}
        for i, row in metadata_hits_df.iterrows():
            hits_tsv = row["hits_indexed"]
            if pd.isna(hits_tsv) or not os.path.exists(hits_tsv):
                logging.warning("Skipping missing hits file for %s: %s", i, hits_tsv)
                continue
            # Indexed hits only (NARROWPEAKS_CUSTOM_COLS: each hit with its ENCODE cluster); a raw FiNeMo
            # hits file has no ENCODE cluster to summarize hits by
            with (gzip.open(hits_tsv, "rt") if hits_tsv.endswith(".gz") else open(hits_tsv)) as hits_handle:
                n_hits_cols = len(hits_handle.readline().rstrip("\n").split("\t"))
            if n_hits_cols != len(NARROWPEAKS_CUSTOM_COLS):
                logging.warning(
                    "Skipping hit annotation for %s: %s has %d columns, not the %d of indexed hits",
                    i, hits_tsv, n_hits_cols, len(NARROWPEAKS_CUSTOM_COLS),
                )
                continue

            # Peaks used to call this experiment's hits (peaks_path: the atlas pipelines' --{head}-peaks-paths)
            peaks_bed = hits_peaks_paths.get(i)
            if peaks_bed is None:
                logging.warning("Skipping hit annotation for %s: no peaks (peaks_path)", i)
                continue
            if pd.isna(peaks_bed) or not os.path.exists(peaks_bed):
                logging.warning(
                    "Skipping hit annotation for %s: missing peaks %s",
                    i, peaks_bed,
                )
                continue
            hits_tsv_annotated = os.path.join(paths.hits_annotated_dir, f"{i}.hits.annotated.tsv.gz")

            future = executor.submit(_hits_worker_run, hits_tsv, peaks_bed, hits_tsv_annotated)
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
    # Compile hits summary per motif -- pool per-hit annotations (peakType,
    # distToTSS, distToPeakSummit) across every experiment that contributes to the
    # same final combined motif cluster (hits_bed_cluster / ENCODE_name),
    # parallelized across annotated files
    from collections import defaultdict

    hits_tsv_annotated_paths = [
        os.path.join(paths.hits_annotated_dir, f"{i}.hits.annotated.tsv.gz")
        for i in metadata_hits_df.index
    ]
    hits_tsv_annotated_paths = [p for p in hits_tsv_annotated_paths if os.path.exists(p)]

    peak_type_counts_total = None
    dist_tss_by_cluster = defaultdict(list)
    dist_summit_by_cluster = defaultdict(list)

    n_workers = min(8, len(os.sched_getaffinity(0)))
    mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(_summarize_hits_file, p): p for p in hits_tsv_annotated_paths}
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

    # Hits summary, per-cluster hit columns and figures: only if any indexed hits were annotated
    hits_summarized = peak_type_counts_total is not None
    if not hits_summarized:
        logging.warning(
            "No indexed hits were annotated: the hits summary and the region-type / distance-to-TSS "
            "figures are skipped (the FiNeMo hits need their ENCODE clusters: NARROWPEAKS_CUSTOM_COLS)."
        )
    else:
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
        html_col for html_col in [
            MetadataCols.posneg_col,
            HitsCols.hits_total_col,
            HitsCols.hits_exonic_frac_col,
            HitsCols.hits_promoter_frac_col,
            HitsCols.hits_intronic_frac_col,
            HitsCols.hits_distal_frac_col,
            HitsCols.region_type_barplot_col,
            HitsCols.distToTSS_boxplot_col,
        ]  # Add new figure columns here
        if html_col in (mc_encode_filtered.columns() + mc_encode_filtered.images())
    ]
    html_column_desc_dict_encodecluster = {col: html_column_desc_dict.get(col, col) for col in html_columns_encodecluster}

    os.makedirs(os.path.dirname(paths.html_encode_filtered_path), exist_ok=True)
    mc_encode_filtered.summary_table_html(
        html_out=paths.html_encode_filtered_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    os.makedirs(os.path.dirname(paths.html_encode_removed_path), exist_ok=True)
    mc_encode_removed.summary_table_html(
        html_out=paths.html_encode_removed_path,
        columns=html_columns_encodecluster,
        logo_trimming=VisualizeArgs.html_logo_trim,
        editable=True,
        column_desc_dict=html_column_desc_dict_encodecluster,
    )

    # Metadata TSV
    os.makedirs(os.path.dirname(paths.metadata_encode_path), exist_ok=True)
    mc_encode_filtered.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_encode_filtered_path, sep="\t", index=False)

    os.makedirs(os.path.dirname(paths.metadata_encode_path), exist_ok=True)
    mc_encode_removed.metadata.drop(columns=EncodeCombineArgs.remove_cols, errors='ignore').to_csv(paths.metadata_encode_removed_path, sep="\t", index=False)

    # Save MotifCompendium: ENCODE, with the hits annotation (over the ones saved before it)
    save_replace(mc_encode_filtered, paths.mc_encode_filtered_path)
    save_replace(mc_encode_removed, paths.mc_encode_removed_path)

    # Args JSON
    args_record = {
        "parser_args": vars(args),
        "configs": pipeline_encode.collect_configs_snapshot(),
        "log_path": log_path,
    }
    args_json_path = os.path.join(paths.metadata_dir, "args_hits.json")
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
    logging.info("ENCODE hits: run started.")

    stage_dir_args = pipeline_encode.atlas_stage_dir_args(args.tfatlas_dir, args.chromatlas_dir, args.procapatlas_dir)
    if args.time:
        start_time = time.time()
    run_encode_hits(pipeline_encode.stage_args(args, **stage_dir_args))
    if args.time:
        logging.info(f"Time taken: {time.time() - start_time:.2f}s")
    logging.info(f'ENCODE hits: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
