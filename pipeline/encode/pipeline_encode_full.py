#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: end-to-end pipeline, all atlases and the ENCODE combine in one process.

Runs the full atlas hierarchy from a single CLI, in order:

  1. TF atlas       - counts head    : motif -> cluster -> meta-cluster
  2. TF atlas       - profile head   : motif -> cluster -> meta-cluster
  3. TF atlas       - combine        : counts + profile meta-clusters -> atlascluster
  4. Chromatin atlas- counts head    : motif -> cell-type cluster -> meta-cluster
  5. Chromatin atlas- profile head   : motif -> cell-type cluster -> meta-cluster
  6. Chromatin atlas- combine        : counts + profile meta-clusters -> atlascluster
  7. ProCap atlas   - counts head    : motif -> cell-type cluster -> meta-cluster
  8. ProCap atlas   - profile head   : motif -> cell-type cluster -> meta-cluster
  9. ProCap atlas   - combine        : counts + profile meta-clusters -> atlascluster
 10. ENCODE         - combine        : TF + Chromatin + ProCap atlasclusters -> ENCODE cluster

The stages are imported from the atlas pipelines, not copied: pipeline/bpnet/pipeline_bpnet.py (TF),
pipeline/chrombpnet/pipeline_chrombpnet.py (Chromatin), pipeline/procapnet/pipeline_procapnet.py
(ProCap) and pipeline/encode/pipeline_encode.py (ENCODE combine). Each binds its own configs.py
(set_default_configs()) before its stages run. Running the atlas pipelines separately (e.g. as a
SLURM array) and then pipeline_encode.py gives the same result.

Every stage directory is derived from a single --output-dir root, and each stage's
outputs are wired directly into the next stage's inputs:

  <root>/bpnet/counts         <root>/bpnet/profile        <root>/bpnet/combined
  <root>/chrombpnet/counts    <root>/chrombpnet/profile   <root>/chrombpnet/combined
  <root>/procapnet/counts     <root>/procapnet/profile    <root>/procapnet/combined
  <root>/encode
"""
import os
import sys
import time
import argparse
import logging
import importlib.util
from argparse import Namespace

import MotifCompendium


#### PIPELINES ---------------------------------------------------------------
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pipeline/


def load_pipeline(name, path):
    """Import a pipeline script as a module (its stage functions and its own configs), without running its main."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline_bpnet = load_pipeline("pipeline_bpnet", os.path.join(PIPELINE_DIR, "bpnet", "pipeline_bpnet.py"))
pipeline_chrombpnet = load_pipeline("pipeline_chrombpnet", os.path.join(PIPELINE_DIR, "chrombpnet", "pipeline_chrombpnet.py"))
pipeline_procapnet = load_pipeline("pipeline_procapnet", os.path.join(PIPELINE_DIR, "procapnet", "pipeline_procapnet.py"))
pipeline_encode = load_pipeline("pipeline_encode", os.path.join(PIPELINE_DIR, "encode", "pipeline_encode.py"))

# Atlas key -> (pipeline module, its per-head stage, its combine stage)
ATLAS_PIPELINES = {
    "tf":     (pipeline_bpnet, pipeline_bpnet.run_tf_sequential, pipeline_bpnet.run_tf_combine),
    "chrom":  (pipeline_chrombpnet, pipeline_chrombpnet.run_chrom_sequential, pipeline_chrombpnet.run_chrom_combine),
    "procap": (pipeline_procapnet, pipeline_procapnet.run_procap_sequential, pipeline_procapnet.run_procap_combine),
}


#### STAGES ------------------------------------------------------------------
# Stage key -> (label, pipeline module)
STAGES = {
    "tf-counts":      ("TF atlas: counts head",              "pipeline_bpnet"),
    "tf-profile":     ("TF atlas: profile head",             "pipeline_bpnet"),
    "tf-combine":     ("TF atlas: combine counts + profile", "pipeline_bpnet"),
    "chrom-counts":   ("Chromatin atlas: counts head",       "pipeline_chrombpnet"),
    "chrom-profile":  ("Chromatin atlas: profile head",      "pipeline_chrombpnet"),
    "chrom-combine":  ("Chromatin atlas: combine counts + profile", "pipeline_chrombpnet"),
    "procap-counts":  ("ProCap atlas: counts head",          "pipeline_procapnet"),
    "procap-profile": ("ProCap atlas: profile head",         "pipeline_procapnet"),
    "procap-combine": ("ProCap atlas: combine counts + profile", "pipeline_procapnet"),
    "encode-combine": ("ENCODE: combine TF + Chromatin + ProCap atlasclusters", "pipeline_encode"),
}
STAGE_ORDER = list(STAGES)


#### ARGUMENTS ---------------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for ENCODE MotifCompendium.",
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
    parser.add_argument("--mcid-version", type=str, default="V0", help="ENCODE MCID version.")

    # Input: TF atlas, per head
    parser.add_argument("--tf-counts-input-mc", type=str, default=None,
                        help="TF counts head: input MotifCompendium object (.mc), instead of building from H5s.")
    parser.add_argument("--tf-counts-modisco-h5s", nargs="+", type=str, default=None,
                        help="TF counts head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--tf-counts-input-names", nargs="+", type=str, default=None,
                        help="TF counts head: nickname(s) for the input file(s).")
    parser.add_argument("--tf-counts-metadata", type=str, default=None,
                        help="TF counts head: metadata CSV/TSV, per input file or motif.")
    parser.add_argument("--tf-counts-finemo-paths", type=str, default=None,
                        help="TF counts head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--tf-counts-peaks-paths", type=str, default=None,
                        help="TF counts head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

    parser.add_argument("--tf-profile-input-mc", type=str, default=None,
                        help="TF profile head: input MotifCompendium object (.mc), instead of building from H5s.")
    parser.add_argument("--tf-profile-modisco-h5s", nargs="+", type=str, default=None,
                        help="TF profile head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--tf-profile-input-names", nargs="+", type=str, default=None,
                        help="TF profile head: nickname(s) for the input file(s).")
    parser.add_argument("--tf-profile-metadata", type=str, default=None,
                        help="TF profile head: metadata CSV/TSV, per input file or motif.")
    parser.add_argument("--tf-profile-finemo-paths", type=str, default=None,
                        help="TF profile head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--tf-profile-peaks-paths", type=str, default=None,
                        help="TF profile head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

    # Input: Chromatin atlas, per head
    parser.add_argument("--chrom-counts-input-mc", type=str, default=None,
                        help="Chromatin counts head: input MotifCompendium object (.mc).")
    parser.add_argument("--chrom-counts-modisco-h5s", nargs="+", type=str, default=None,
                        help="Chromatin counts head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--chrom-counts-input-names", nargs="+", type=str, default=None,
                        help="Chromatin counts head: nickname(s) for the input file(s).")
    parser.add_argument("--chrom-counts-metadata", type=str, default=None,
                        help="Chromatin counts head: metadata CSV/TSV.")
    parser.add_argument("--chrom-counts-finemo-paths", type=str, default=None,
                        help="Chromatin counts head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--chrom-counts-peaks-paths", type=str, default=None,
                        help="Chromatin counts head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

    parser.add_argument("--chrom-profile-input-mc", type=str, default=None,
                        help="Chromatin profile head: input MotifCompendium object (.mc).")
    parser.add_argument("--chrom-profile-modisco-h5s", nargs="+", type=str, default=None,
                        help="Chromatin profile head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--chrom-profile-input-names", nargs="+", type=str, default=None,
                        help="Chromatin profile head: nickname(s) for the input file(s).")
    parser.add_argument("--chrom-profile-metadata", type=str, default=None,
                        help="Chromatin profile head: metadata CSV/TSV.")
    parser.add_argument("--chrom-profile-finemo-paths", type=str, default=None,
                        help="Chromatin profile head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--chrom-profile-peaks-paths", type=str, default=None,
                        help="Chromatin profile head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

    # Input: ProCap atlas, per head
    parser.add_argument("--procap-counts-input-mc", type=str, default=None,
                        help="ProCap counts head: input MotifCompendium object (.mc).")
    parser.add_argument("--procap-counts-modisco-h5s", nargs="+", type=str, default=None,
                        help="ProCap counts head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--procap-counts-input-names", nargs="+", type=str, default=None,
                        help="ProCap counts head: nickname(s) for the input file(s).")
    parser.add_argument("--procap-counts-metadata", type=str, default=None,
                        help="ProCap counts head: metadata CSV/TSV.")
    parser.add_argument("--procap-counts-finemo-paths", type=str, default=None,
                        help="ProCap counts head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--procap-counts-peaks-paths", type=str, default=None,
                        help="ProCap counts head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

    parser.add_argument("--procap-profile-input-mc", type=str, default=None,
                        help="ProCap profile head: input MotifCompendium object (.mc).")
    parser.add_argument("--procap-profile-modisco-h5s", nargs="+", type=str, default=None,
                        help="ProCap profile head: input TF-MoDISco (lite) H5 file(s).")
    parser.add_argument("--procap-profile-input-names", nargs="+", type=str, default=None,
                        help="ProCap profile head: nickname(s) for the input file(s).")
    parser.add_argument("--procap-profile-metadata", type=str, default=None,
                        help="ProCap profile head: metadata CSV/TSV.")
    parser.add_argument("--procap-profile-finemo-paths", type=str, default=None,
                        help="ProCap profile head: FiNeMo paths file (TSV, no header: input name, FiNeMo hits file), for the number of hits per motif (counted in the hits file) and its filter (1_minhits). If not provided, the filter is skipped.")
    parser.add_argument("--procap-profile-peaks-paths", type=str, default=None,
                        help="ProCap profile head: peaks paths file (TSV, no header: input name, peaks BED), for the ENCODE hits annotation.")

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
    parser.add_argument("--procap-init-metadata", type=str, default=None,
                        help="ENCODE stage: initial TF-MoDISco metadata TSV for the ProCap atlas.")
    parser.add_argument("--mc-qc-remove-removed", type=str, default=None,
                        help="ENCODE stage (optional): removed MotifCompendium object (.mc) from the prior QC-remove run.")
    parser.add_argument("--mc-qc-pass-filtered", type=str, default=None,
                        help="ENCODE stage (optional): filtered MotifCompendium object (.mc) from the prior QC-pass run.")
    parser.add_argument("--mc-qc-pass-lowq", type=str, default=None,
                        help="ENCODE stage (optional): low-quality MotifCompendium object (.mc) from the prior QC-pass run.")
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
    parser.add_argument("--export-modisco", action="store_true", help="Also export TF-MoDISco HDF5 files (default: off).")
    parser.add_argument("-t", "--time", action="store_true", help="Print time taken for each step.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()


#### ARG ALIASES -------------------------------------------------------------
# The atlas pipelines' stages read args.<ref_name>; this CLI exposes some of the same inputs under
# different flag names. stage_args() mirrors each one onto the reference name.
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
        "tf-counts":      os.path.join(root, "bpnet", "counts"),
        "tf-profile":     os.path.join(root, "bpnet", "profile"),
        "tf-combine":     os.path.join(root, "bpnet", "combined"),
        "chrom-counts":   os.path.join(root, "chrombpnet", "counts"),
        "chrom-profile":  os.path.join(root, "chrombpnet", "profile"),
        "chrom-combine":  os.path.join(root, "chrombpnet", "combined"),
        "procap-counts":  os.path.join(root, "procapnet", "counts"),
        "procap-profile": os.path.join(root, "procapnet", "profile"),
        "procap-combine": os.path.join(root, "procapnet", "combined"),
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


#### MAIN ----------------------------------------------------------------------
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

    # Stage 'encode-combine': Required inputs are checked here, before the stages that precede it.
    # The QC inputs (--mc-qc-remove-removed, --mc-qc-pass-filtered, --mc-qc-pass-lowq) are optional:
    # each QC step is skipped without its input.
    if "encode-combine" in selected:
        missing = [n for n in ("tf_init_metadata", "chrom_init_metadata", "procap_init_metadata", "metadata_edited",
                               "hg38_fasta", "gencode_gtf")
                   if getattr(args, n) is None]
        if missing:
            logging.error("Stage 'encode-combine' requires: "
                          + ", ".join("--" + m.replace("_", "-") for m in missing))
            raise ValueError(f"Missing required arguments for encode-combine: {missing}")


    ### RUN --------------------------------------------------------------------
    for stage in selected:
        label, module_name = STAGES[stage]
        stage_dir = DIRS[stage]
        logging.info("=" * 78)
        logging.info(f"STAGE {STAGE_ORDER.index(stage) + 1}/{len(STAGE_ORDER)}: {label}")
        logging.info(f"  pipeline: {module_name}   output: {stage_dir}")
        logging.info("=" * 78)
        if args.time:
            stage_start = time.time()

        # Bind this stage's pipeline configs before its body runs
        sys.modules[module_name].set_default_configs()
        os.makedirs(stage_dir, exist_ok=True)

        # --- 1/2, 4/5 & 7/8: per-head sequential clustering -----------------------
        if stage in ("tf-counts", "tf-profile", "chrom-counts", "chrom-profile", "procap-counts", "procap-profile"):
            atlas_key, head = stage.split("-")            # ('tf'|'chrom'|'procap', 'counts'|'profile')
            prefix = f"{atlas_key}_{head}"
            sa = stage_args(
                args,
                output_dir=stage_dir,
                head_type=head,
                mcid_version=args.mcid_version,
                input_mc=getattr(args, f"{prefix}_input_mc"),
                input_modisco_h5s=getattr(args, f"{prefix}_modisco_h5s"),
                input_names=getattr(args, f"{prefix}_input_names"),
                input_pfms=None,
                metadata=getattr(args, f"{prefix}_metadata"),
                finemo_paths=getattr(args, f"{prefix}_finemo_paths"),
                peaks_paths=getattr(args, f"{prefix}_peaks_paths"),
            )
            ATLAS_PIPELINES[atlas_key][1](sa)

        # --- 3, 6 & 9: combine the two heads into atlasclusters ------------------
        elif stage in ("tf-combine", "chrom-combine", "procap-combine"):
            atlas_key = stage.split("-")[0]
            sa = stage_args(
                args,
                output_dir=stage_dir,
                mcid_version=args.mcid_version,
                counts_dir=DIRS[f"{atlas_key}-counts"],
                profile_dir=DIRS[f"{atlas_key}-profile"],
            )
            ATLAS_PIPELINES[atlas_key][2](sa)

        # --- 10: combine the TF, Chromatin and ProCap atlases --------------------------
        else:
            sa = stage_args(
                args,
                output_dir=stage_dir,
                mcid_version=args.mcid_version,
                tf_dir=DIRS["tf-combine"],
                tf_counts_dir=DIRS["tf-counts"],
                tf_profile_dir=DIRS["tf-profile"],
                chrom_dir=DIRS["chrom-combine"],
                chrom_counts_dir=DIRS["chrom-counts"],
                chrom_profile_dir=DIRS["chrom-profile"],
                tf_combined_dir=DIRS["tf-combine"],
                chrom_combined_dir=DIRS["chrom-combine"],
                procap_dir=DIRS["procap-combine"],
                procap_counts_dir=DIRS["procap-counts"],
                procap_profile_dir=DIRS["procap-profile"],
                procap_combined_dir=DIRS["procap-combine"],
            )
            pipeline_encode.run_encode_combine(sa)

        if args.time:
            logging.info(f"Stage '{stage}' time taken: {time.time() - stage_start:.2f}s")

    logging.info("=" * 78)
    logging.info(f'ENCODE GRAMMAR pipeline: run completed: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
