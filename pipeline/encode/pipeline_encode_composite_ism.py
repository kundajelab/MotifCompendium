#!/usr/bin/env python
# coding: utf-8
"""
ENCODE MotifCompendium: ISM cooperativity of the composite motifs, per model (composite/ism_composite_models.py).

For each model, every composite motif it is tested in (--run-metadata: the per-model composite metadata of
pipeline_encode.py): its two subunits are inserted into background sequences of each fold at every spacing and
orientation (spacing ISM) and alone (solo ISM), and the contributions are computed at the best arrangement
(DeepLIFT/SHAP). Per (motif, model), in <output-dir>/<motif>/<ID>/: results.tsv, the mean-reduced predictions and
contributions, and the figures; pipeline_encode_composite.py reads results.tsv and contribs_mean.h5.

Runs in a torch environment with bpnetlite and tangermeme (not MotifCompendium's). The ISM helpers are
utils_composite_ism.py (HDMA's tangermeme_utils, copied). A SLURM array runs one model per task (--model-idx N);
without --model-idx, every model of --run-metadata is run, one after the other.
"""
import argparse
import contextlib
import os
import sys

import datetime
import gc
import logging
import importlib.util

import numpy as np
import pandas as pd

import torch
from pyfaidx import Fasta

from bpnetlite import BPNet
from bpnetlite.bpnet import BasePairNet, CountWrapper
from bpnetlite.attribute import deep_lift_shap

from tangermeme.io import extract_loci
from tangermeme.utils import reverse_complement
from tangermeme.ersatz import multisubstitute, substitute
from tangermeme.predict import predict
from tangermeme.space import space

from scipy.stats import wilcoxon

import matplotlib.figure
import matplotlib.pyplot as plt


def _load_sibling(name, file_name):
    """Import a module of this directory by its path (pipeline/encode has no package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The HDMA ChromBPNet/tangermeme ISM helpers (fold wrapping, tensor summarization, spacing-experiment plotting),
# copied into utils_composite_ism.py
utils_composite_ism = _load_sibling("utils_composite_ism", "utils_composite_ism.py")
from utils_composite_ism import MIDPOINT, CHAR_IGNORE
from utils_composite_ism import BPNetWrapper, ChromBPNetWrapper, ZeroControlWrapper, reshape_folds
from utils_composite_ism import summarize_across_folds, mean_across_sequences
from utils_composite_ism import (
    save_pred_dict_h5, save_array_h5, load_array_h5,
    reduce_pred_dict, save_nested_h5, load_nested_h5,
)
from utils_composite_ism import (
    is_palindrome, calc_effect_Z_score, calc_effect_entropy,
    plot_effect_zscores, plot_effect_spacing, plot_effect_spacing_lines,
    plot_effect_solo, plot_profiles_spacing, plot_profiles_solo,
    plot_mean_contributions, plot_contributions_heatmap, plot_profile_heatmap_z,
    plot_effect_boxplot,
)

# Column names: the ENCODE configs (configs.py)
configs = _load_sibling("configs_encode", "configs.py")
name_col = configs.Encode_MetadataCols.name_col
id_col = configs.Encode_MetadataCols.id_col
composite_ref_col = configs.Encode_CompositeArgs.composite_ref_col
motif_string_col = configs.Encode_VisualizeArgs.motif_string_col


#### ARGUMENTS ------------------------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(description="Run in silico mutagenesis (ISM) for composite motifs in ENCODE GRAMMAR.")

    parser.add_argument("--motif-metadata", type=str, required=True, help="Path to the motif metadata file, per motif: CSV, TSV format (pipeline_encode.py: <composite-dir>/export/ENCODE-motifcompendium_composite_metadata.tsv).")
    parser.add_argument("--model-metadata", type=str, required=True, help="Path to the ENCID/model file metadata (model + background paths per fold), per ENCID: CSV, TSV format, e.g. the pipeline's raw/all_paths.tsv (composite/build_all_paths.py) or a v23-style raw/encid_metadata.tsv.")
    parser.add_argument("--run-metadata", type=str, required=True, help="Path to the run metadata file, per model: one row per model (ID), with name_col holding a comma-separated aggregation of every motif name that model should run. CSV, TSV format (pipeline_encode.py: <composite-dir>/export/ENCODE-motifcompendium_composite_metadata_model.tsv).")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory (<composite-dir>/ism).")
    parser.add_argument("--genome-fa", type=str, required=True, help="Genome FASTA (hg38) of the background sequences.")
    parser.add_argument("--model-idx", type=int, nargs="*", default=None, help="Index(es) of the model(s) to process (for all relevant motifs), rows of --run-metadata. Default: every model.")

    parser.add_argument("--n-spaces", type=int, default=201, help="Number of spaces to test for spacing experiments (default: 201, from 0 to 200).")
    parser.add_argument("--n-background-seqs", type=int, default=100, help="Number of background sequences to sample for ISM (default: 100).")

    parser.add_argument("--seed", type=int, default=100, help="Random seed for reproducibility (default: 100).")
    parser.add_argument("--overwrite-all", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--skip-done", action="store_true", help="With --overwrite-all: still skip motifs whose .done exists, and recompute every step of the others instead of reusing their saved files (reruns of failed runs, whose partial files may be truncated, e.g. by a full disk).")
    parser.add_argument("--save-all", action="store_true", help="Additionally save the full per-fold/per-sequence arrays (contribs.h5, predictions.h5, predictions_solo.h5). The mean-reduced files are always written; these are much larger and are not needed by anything downstream.")

    parser.add_argument("--skip-interpret", dest="run_interpret", action="store_false", help="Skip model interpretation step.")
    parser.add_argument("--overwrite-interpret", action="store_true", help="Overwrite existing interpretation output files only.")

    return parser.parse_args()


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# logging: INFO+ progress narration to stdout, WARNING+ to stderr, for all
# motifs/ENCIDs, plus a per-(motif, ENCID) file handler added/removed inside
# the main loop
ism_logger = logging.getLogger("ism_cooperativity")
ism_logger.setLevel(logging.INFO)
ism_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ism_ch_out = logging.StreamHandler(sys.stdout)
ism_ch_out.setLevel(logging.INFO)
ism_ch_out.addFilter(lambda record: record.levelno < logging.WARNING)
ism_ch_out.setFormatter(ism_log_formatter)
ism_logger.addHandler(ism_ch_out)

ism_ch_err = logging.StreamHandler(sys.stderr)
ism_ch_err.setLevel(logging.WARNING)
ism_ch_err.setFormatter(ism_log_formatter)
ism_logger.addHandler(ism_ch_err)

#### FUNCTIONS --------------------------------------------------------------------
# ISM Cooperativity: Helpers
def load_background_seqs(negatives_bed_path, genome_fa, n_seqs, seed):
    """
    Load a reproducible random subset of background (negative) sequences for one fold,
    skipping windows that contain ambiguous bases (N in hg38, i.e. assembly gaps).

    one_hot_encode maps an N to an all-zero column, which is not a valid one-hot
    input: tangermeme rejects it downstream (substitute/multisubstitute/space and
    deep_lift_shap all validate with allow_N=False, raising "X must be one-hot
    encoded. and cannot have unknown characters."), and even if it were accepted it
    would be out-of-distribution for the model -- deep_lift_shap's default
    dinucleotide_shuffle reference would silently read each all-zero column as an 'A'.
    Since negatives files hold far more candidate regions than we sample, extra
    candidates are drawn so that n_seqs clean sequences are still returned.
    """
    negatives_df = pd.read_csv(negatives_bed_path, sep="\t", header=None)

    # shuffle once (reproducibly), then walk it in chunks keeping only N-free windows
    negatives_df = negatives_df.sample(frac=1, random_state=seed)

    kept, n_kept, n_skipped = [], 0, 0
    for start in range(0, len(negatives_df), n_seqs):
        X_chunk = extract_loci(negatives_df.iloc[start:start + n_seqs], genome_fa, ignore=CHAR_IGNORE)

        # an ambiguous base is an all-zero column along the alphabet axis, so a
        # clean window is one whose every position sums to exactly 1
        is_clean = (X_chunk.sum(dim=1) == 1).all(dim=-1)

        n_skipped += int((~is_clean).sum())
        kept.append(X_chunk[is_clean])
        n_kept += int(is_clean.sum())

        if n_kept >= n_seqs:
            break

    X = torch.cat(kept)[:n_seqs]

    if n_skipped > 0:
        ism_logger.warning(
            f"Skipped {n_skipped} background region(s) containing ambiguous bases (N) "
            f"in {negatives_bed_path}")

    if len(X) < n_seqs:
        ism_logger.warning(
            f"Only {len(X)} of {n_seqs} requested N-free background sequences available "
            f"in {negatives_bed_path}")

    return X.float()

def spacing_offset_for(motifs):
    """
    The spacing value at position 0 of test_spacing_per_fold's swept spacing
    axis for this motif pair: negative enough that motifs[1] can be shaved
    all the way down to a single base (tangermeme's multisubstitute() shaves
    the overlap off the front of motifs[1] for negative spacing).
    """
    return -(len(motifs[1]) - 1)

def test_spacing_per_fold(model_seq_pairs, motifs, key, n_spaces):
    """
    Run the spacing experiment for one motif pair and save the outputs as a
    dataframe. Unlike the HDMA version, each fold uses its own held-out
    (model, background sequences) pair instead of a background shared across folds.

    Spacings are swept from spacing_offset_for(motifs) through
    spacing_offset_for(motifs) + n_spaces - 1: still n_spaces total values /
    n_spaces-1 bp of range, just shifted so the negative end covers motifs[1]
    overlapping the back of motifs[0] rather than always starting at 0 bp.
    """
    spacing_offset = spacing_offset_for(motifs)
    spacings = torch.arange(spacing_offset, spacing_offset + n_spaces)

    y_before_list, y_after_list = [], []
    for model, seqs in model_seq_pairs:
        y_before, y_after = space(model, seqs, motifs, spacing=spacings[:, None])
        y_before_list.append(y_before)
        y_after_list.append(y_after)

    y_before = reshape_folds(y_before_list)
    y_after = reshape_folds(y_after_list)

    y_after_mean, y_after_sd = summarize_across_folds(mean_across_sequences(y_after[1]))

    # y_before has no spacing axis (the baseline doesn't depend on spacing);
    # broadcast it against every spacing in y_after explicitly, since torch's
    # implicit broadcasting would otherwise align y_after's spacing axis
    # against y_before's batch axis and raise a shape-mismatch error
    y_effect = y_after[1] - y_before[1].unsqueeze(2)
    y_effect_mean, y_effect_sd = summarize_across_folds(mean_across_sequences(y_effect))

    df = pd.DataFrame({
        "key": [key] * n_spaces,
        "spacing": spacings.numpy(),
        "counts_after_mean": y_after_mean.squeeze(),
        "counts_after_sd": y_after_sd.squeeze(),
        "effect_mean": y_effect_mean.squeeze(),
        "effect_sd": y_effect_sd.squeeze(),
    })

    return y_before, y_after, df

class ProCapNetWrapper(torch.nn.Module):
    """
    A wrapper class that returns counts and transformed profiles from a ProCapNet
    model (bpnetlite BPNet: two stranded profile outputs, one total-count output,
    no control track), like BPNetWrapper/ChromBPNetWrapper.

    ProCapNet is trained with a single multinomial over both strands and all
    positions (bpnetlite _mixture_loss flattens the strands before the
    log-softmax), so the profile is softmaxed jointly over (strand, position) and
    scaled by the one predicted total count, matching count_scaled_profile in
    ayhe/procap-atlas src/modeling/profile.py. BPNetWrapper's per-strand softmax
    would give each strand the full predicted total.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        y_profile, y_counts = self.model(X)

        y_profile = torch.nn.functional.softmax(y_profile.reshape(y_profile.shape[0], -1), dim=-1).reshape(y_profile.shape)
        y_profile = y_profile * torch.exp(y_counts).unsqueeze(2)  # (N, 1) -> (N, 1, 1)

        return y_profile, y_counts

def model_folds_for(encid_metadata_j):
    """
    The folds of one model: those with a background (negatives) path in its model
    metadata row, i.e. 0-4 for BPNet/ChromBPNet and 0-6 for ProCapNet in
    all_paths.tsv (0-4 in a v23 encid_metadata.tsv).
    """
    return sorted(
        int(col[len("negatives_fold"):]) for col in encid_metadata_j.columns
        if col.startswith("negatives_fold") and pd.notna(encid_metadata_j[col].values[0])
    )

def load_model_seq_pairs(encid_metadata_j, n_background_seqs, seed, genome_fa):
    """
    Load one (wrapped model, background sequences) pair per fold of one model: the
    fold's bias-free model and its own background (negative) sequences.
    """
    model_j = encid_metadata_j['model'].values[0]

    model_seq_pairs = []
    for fold in model_folds_for(encid_metadata_j):
        negatives_bed_k = encid_metadata_j[f'negatives_fold{fold}'].values[0]

        if model_j == "chrombpnet":
            # bias-free submodel; the raw model_fold{N} still includes the Tn5/DNase bias track
            model_path_k = encid_metadata_j[f'nobias_fold{fold}'].values[0]
            model_k = BPNet.from_chrombpnet(filename=model_path_k)
            model_wrapper_k = ChromBPNetWrapper(model_k)
        elif model_j == "bpnet":
            # bpnet ChIP models have no nobias/bias split; model_fold{N} is the only model file
            model_path_k = encid_metadata_j[f'model_fold{fold}'].values[0]
            model_k = BasePairNet.from_bpnet(filename=model_path_k)
            model_k = ZeroControlWrapper(model_k)  # no real control track for synthetic ISM sequences
            model_wrapper_k = BPNetWrapper(model_k)
        elif model_j == "procapnet":
            # PRO-cap BPNet saved whole by bpnetlite's fit ({ID}.fold{N}.torch, the
            # best-validation checkpoint); trained without a control track
            model_path_k = encid_metadata_j[f'model_fold{fold}'].values[0]
            model_k = torch.load(model_path_k, weights_only=False, map_location=torch.device("cpu"))
            model_wrapper_k = ProCapNetWrapper(model_k)
        else:
            raise ValueError(f"Unknown model type '{model_j}' (expected bpnet, chrombpnet or procapnet)")
        seqs_k = load_background_seqs(negatives_bed_k, genome_fa, n_background_seqs, seed)

        model_seq_pairs.append((model_wrapper_k, seqs_k))

    return model_seq_pairs


def fold_summarize_profiles(reduced):
    """
    Replace each stage's sequence-averaged profile, kept per fold (`profile_seqmean`,
    shape (folds, ...)), with its mean and SD across folds (`profile_foldmean`,
    `profile_foldsd`). Those are the only things the plots compute from it, so the
    reduced h5s no longer grow with the number of folds (5 BPNet/ChromBPNet, 7
    ProCapNet). It is the same summarize_across_folds call the plots made on the
    loaded file, so the plotted values are unchanged. Files written before this change
    hold `profile_seqmean` and are converted the same way when loaded.
    """
    for stages in reduced.values():
        for values in stages.values():
            # "_best" holds a tensor, not a stage dict
            if isinstance(values, dict) and "profile_seqmean" in values:
                values["profile_foldmean"], values["profile_foldsd"] = summarize_across_folds(values.pop("profile_seqmean"))
    return reduced

@contextlib.contextmanager
def rasterized_plot_data(zorder=2.5, dpi=150):
    """
    While active, every figure save (Figure.savefig, which plt.savefig and plotnine's
    ggplot.save both call, so it covers the tangermeme_utils plotting helpers with their
    own dpi/bbox settings) first rasterizes each axes' data layers -- artists below
    `zorder`: heatmap cells, lines, error bands, points, bars, logo glyphs -- and keeps
    the spines, ticks, text and legends (zorder >= 2.5) as vector. Otherwise the PDFs
    store every heatmap cell and profile point as its own shape (a 100 x 1000 profile
    heatmap is 100,000 rectangles). `dpi` sets the resolution of the rasterized layers
    in place of the helper's own (150: indistinguishable from vector on screen).

    Used only for the figures it shrinks: the two heatmaps, the profile panels and the
    Z-score dot plot. The sparse line/bar/box plots are smaller as vector, and the
    contribution logo would blur.
    """
    savefig = matplotlib.figure.Figure.savefig

    def savefig_rasterized(fig, *args, **kwargs):
        for ax in fig.axes:
            ax.set_rasterization_zorder(zorder)
        if dpi is not None:
            kwargs["dpi"] = dpi
        return savefig(fig, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig_rasterized
    try:
        yield
    finally:
        matplotlib.figure.Figure.savefig = savefig



#### ISM: ONE MODEL -------------------------------------------------------------
def run_composite_ism_model(args, model_idx, mc_encode_composite_metadata_df, mc_encode_composite_metadata_model_df,
                            encid_metadata_df, genome_fa):
    """ISM cooperativity of every composite motif of one model (row model_idx of the run metadata)."""
    # Select the model [j] to process, then load it once and run every motif
    # experiment it appears in -- rather than indexing per motif and reloading
    # every model each time that motif's ENCID list is walked.
    if model_idx >= len(mc_encode_composite_metadata_model_df):
        # lets an array job span more indices than there are models
        ism_logger.warning(f"Model index {model_idx} is past the last of {len(mc_encode_composite_metadata_model_df)} models in {args.run_metadata}, nothing to run.")
        return

    mc_encode_composite_metadata_model_df_j = mc_encode_composite_metadata_model_df.iloc[[model_idx]]
    encid_j = mc_encode_composite_metadata_model_df_j[id_col].values[0]
    names_j = mc_encode_composite_metadata_model_df_j[name_col].values[0].split(",")

    # all_paths.tsv holds one row per head (counts, profile) of each model; the model
    # and negatives paths are the same in both, so the first row is used
    encid_metadata_j = encid_metadata_df[encid_metadata_df['experiment'] == encid_j].iloc[:1]
    if len(encid_metadata_j) == 0:
        ism_logger.warning(f"No metadata found for {encid_j}, skipping.")
        return

    model_j = encid_metadata_j['model'].values[0]
    ism_logger.info(f"{encid_j}: {model_j} model, folds {model_folds_for(encid_metadata_j)}")

    # Perform forward pass through each fold once for this model: load the fold's
    # bias-free model and its own held-out background (negative) sequences. Every
    # motif experiment below reuses these loaded models/sequences.
    model_seq_pairs = load_model_seq_pairs(encid_metadata_j, args.n_background_seqs, args.seed, genome_fa)

    # Run ISM cooperativity analysis per motif [i] that this model appears in
    for i, name_i in enumerate(names_j):

        mc_encode_composite_metadata_df_i = mc_encode_composite_metadata_df[mc_encode_composite_metadata_df[name_col] == name_i]
        if len(mc_encode_composite_metadata_df_i) == 0:
            ism_logger.warning(f"No motif metadata found for {name_i}, skipping.")
            continue

        name_i_safe = name_i.replace("/", ".")
        seqA = mc_encode_composite_metadata_df_i[f'{composite_ref_col}_{motif_string_col}0'].values[0]
        seqB = mc_encode_composite_metadata_df_i[f'{composite_ref_col}_{motif_string_col}1'].values[0]
        observed_spacing = mc_encode_composite_metadata_df_i[f'{composite_ref_col}_spacing'].values[0]

        # A pair of motifs can be tested in four orientations:
        # - A> B>: A, B, head-to-tail   - A> B<: head-to-head
        # - B> A>: B, A, head-to-tail   - A< B>: tail-to-tail
        input_motifs = {
            "A,B HT": [seqA, seqB],
            "B,A HT": [seqB, seqA],
            "HH": [seqA, reverse_complement(seqB)],
            "TT": [reverse_complement(seqA), seqB],
        }

        out_dir = os.path.join(args.output_dir, name_i_safe, encid_j)
        os.makedirs(out_dir, exist_ok=True)

        datetimenow = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        ism_ch2 = logging.FileHandler(os.path.join(out_dir, f'{encid_j}--{name_i_safe}--{datetimenow}.log'))
        ism_ch2.setLevel(logging.INFO)
        ism_ch2.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        ism_logger.addHandler(ism_ch2)

        try:
            if (args.skip_done or not args.overwrite_all) and os.path.exists(os.path.join(out_dir, ".done")):
                ism_logger.info(f"{name_i} ({encid_j}) already done, skipping")
                ism_logger.removeHandler(ism_ch2)
                continue

            ism_logger.info(f"{name_i} ({encid_j}): seqA={seqA}, seqB={seqB}, observed spacing={observed_spacing}")

            # mean-reduced files are always written and are what everything
            # downstream reads; the full per-fold/per-sequence arrays are only
            # written under --save-all
            path_spacing_mean_hdf5 = os.path.join(out_dir, "predictions_mean.h5")
            path_solo_mean_hdf5 = os.path.join(out_dir, "predictions_solo_mean.h5")
            path_contribs_mean_hdf5 = os.path.join(out_dir, "contribs_mean.h5")
            path_contribs_mean_motif_hdf5 = os.path.join(out_dir, "contribs_mean_motif.h5")

            path_spacing_hdf5 = os.path.join(out_dir, "predictions.h5")
            path_solo_hdf5 = os.path.join(out_dir, "predictions_solo.h5")
            path_contribs_all_hdf5 = os.path.join(out_dir, "contribs.h5")

            path_spacing_df = os.path.join(out_dir, "summarized_effects_per_spacing.tsv")
            path_solo_df = os.path.join(out_dir, "summarized_effects_solo.tsv")

            config_df = pd.DataFrame({
                "motif_name": [name_i],
                "ENCID": [encid_j],
                "n_spaces": [args.n_spaces],
                "seqA": [seqA],
                "seqB": [seqB],
                "observed_spacing": [observed_spacing],
            })
            for key, motif in input_motifs.items():
                config_df[key] = [f"{motif[0]}_{motif[1]}"]

            # Spacing ISM ----------------------------------------------------
            if not args.overwrite_all and os.path.exists(path_spacing_mean_hdf5):
                ism_logger.info("\tLoading precomputed spacing ISM...")
                spacing_reduced = fold_summarize_profiles(load_nested_h5(path_spacing_mean_hdf5))
                spacing_dfs = pd.read_csv(path_spacing_df, sep="\t")
                spacing_preds = None
            else:
                ism_logger.info("\tRunning spacing ISM...")
                spacing_dfs, spacing_preds = [], {}
                for key, motifs in input_motifs.items():
                    ism_logger.info(f"\t\t{key}")
                    y_before, y_after, df = test_spacing_per_fold(model_seq_pairs, motifs, key, n_spaces=args.n_spaces)
                    spacing_preds[key] = {"y_before": y_before, "y_after": y_after}
                    spacing_dfs.append(df)

                # reset index: concatenating dfs would otherwise retain each
                # df's original index, corrupting idxmax() below
                spacing_dfs = pd.concat(spacing_dfs, ignore_index=True)

                # saving is deferred until best_position is known below, since the
                # per-sequence profile is only kept at the winning spacing
                spacing_reduced = fold_summarize_profiles(reduce_pred_dict(spacing_preds))

            # profiles averaged across sequences, then summarized across folds
            profile_preds_summarized = {}
            for key in input_motifs.keys():
                profile_preds_summarized[key] = {
                    "y_before_means": spacing_reduced[key]["y_before"]["profile_foldmean"],
                    "y_before_sd": spacing_reduced[key]["y_before"]["profile_foldsd"],
                    "y_after_means": spacing_reduced[key]["y_after"]["profile_foldmean"],
                    "y_after_sd": spacing_reduced[key]["y_after"]["profile_foldsd"],
                }

            # find the spacing/orientation with the strongest cooperative effect
            max_idx = spacing_dfs.effect_mean.idxmax()
            max_row = spacing_dfs.iloc[max_idx]
            best_spacing = int(max_row.spacing)
            best_orientation = max_row.key

            # spacing_offset converts a true (possibly negative) spacing value into
            # its position along the swept spacing axis, for indexing into predictions
            spacing_offsets = {key: spacing_offset_for(motifs) for key, motifs in input_motifs.items()}
            best_spacing_offset = spacing_offsets[best_orientation]
            best_position = best_spacing - best_spacing_offset

            ism_logger.info(f"Best: {best_orientation} {best_spacing} bp, mean effect: {max_row['effect_mean']}")

            if spacing_preds is not None:
                # per-sequence profile at the winning spacing only: the z-scored
                # profile heatmap is the sole consumer of per-sequence profiles,
                # and it only ever looks at best_position
                spacing_reduced["_best"] = {
                    "profile_foldmean_at_best":
                        spacing_preds[best_orientation]["y_after"][0][:, :, best_position, :].mean(dim=0)
                }

                save_nested_h5(path_spacing_mean_hdf5, spacing_reduced)
                spacing_dfs.to_csv(path_spacing_df, sep="\t", index=False)
                if args.save_all:
                    save_pred_dict_h5(path_spacing_hdf5, spacing_preds)

            # Solo motif ISM ---------------------------------------------------
            input_motifs_solo = {"A": seqA, "B": seqB}

            if not args.overwrite_all and os.path.exists(path_solo_mean_hdf5):
                ism_logger.info("\tLoading precomputed solo ISM...")
                solo_reduced = fold_summarize_profiles(load_nested_h5(path_solo_mean_hdf5))
                solo_dfs = pd.read_csv(path_solo_df, sep="\t")
            else:
                ism_logger.info("\tRunning solo motif ISM...")
                solo_dfs, solo_preds = [], {}

                with torch.no_grad():
                    for key, motif in input_motifs_solo.items():
                        ism_logger.info(f"\t\t{key}")

                        y_before_list, y_after_list = [], []
                        for model, seqs in model_seq_pairs:
                            X_after_solo = substitute(seqs, motif)
                            y_before_fold = predict(model, seqs)
                            y_after_fold = predict(model, X_after_solo)
                            y_before_list.append((y_before_fold[0], y_before_fold[1]))
                            y_after_list.append((y_after_fold[0], y_after_fold[1]))
                            del X_after_solo

                        torch.cuda.empty_cache()

                        y_before = reshape_folds(y_before_list)
                        y_after = reshape_folds(y_after_list)

                        y_after_mean, y_after_sd = summarize_across_folds(mean_across_sequences(y_after[1]))
                        y_effect = y_after[1] - y_before[1]
                        y_effect_mean, y_effect_sd = summarize_across_folds(mean_across_sequences(y_effect))

                        solo_preds[key] = {"y_before": y_before, "y_after": y_after}

                        solo_dfs.append(pd.DataFrame({
                            "key": [key],
                            "counts_after_mean": y_after_mean.squeeze().tolist(),
                            "counts_after_sd": y_after_sd.squeeze().tolist(),
                            "effect_mean": y_effect_mean.squeeze().tolist(),
                            "effect_sd": y_effect_sd.squeeze().tolist(),
                        }))

                # combine independent effects of A and B: sum for profile,
                # log-additive for counts (multiplicative model)
                y_before_profile_mult = solo_preds['A']['y_before'][0]
                y_after_profile_mult = solo_preds['A']['y_after'][0] + solo_preds['B']['y_after'][0]

                y_before_counts_mult = solo_preds['A']['y_before'][1] + solo_preds['B']['y_before'][1]
                y_after_counts_mult = solo_preds['A']['y_after'][1] + solo_preds['B']['y_after'][1]

                y_before_counts_mean, y_before_counts_sd = summarize_across_folds(mean_across_sequences(y_before_counts_mult))
                y_after_counts_mean, y_after_counts_sd = summarize_across_folds(mean_across_sequences(y_after_counts_mult))

                solo_preds['A + B'] = {
                    "y_before": [y_before_profile_mult, y_before_counts_mult],
                    "y_after": [y_after_profile_mult, y_after_counts_mult],
                }

                y_effect_mult = solo_preds['A']['y_after'][1] + solo_preds['B']['y_after'][1] - solo_preds['A']['y_before'][1] - solo_preds['B']['y_before'][1]
                y_effect_mult_mean, y_effect_mult_sd = summarize_across_folds(mean_across_sequences(y_effect_mult))

                solo_dfs.append(pd.DataFrame({
                    "key": ["A + B"],
                    "counts_after_mean": y_after_counts_mean.squeeze().tolist(),
                    "counts_after_sd": y_after_counts_sd.squeeze().tolist(),
                    "effect_mean": y_effect_mult_mean.squeeze().tolist(),
                    "effect_sd": y_effect_mult_sd.squeeze().tolist(),
                }))
                solo_dfs = pd.concat(solo_dfs, ignore_index=True)

                solo_reduced = fold_summarize_profiles(reduce_pred_dict(solo_preds))

                save_nested_h5(path_solo_mean_hdf5, solo_reduced)
                solo_dfs.to_csv(path_solo_df, sep="\t", index=False)
                if args.save_all:
                    save_pred_dict_h5(path_solo_hdf5, solo_preds)

            # solo profiles averaged across sequences, then summarized across folds
            solo_profile_preds_summarized = {}
            for key in solo_reduced.keys():
                solo_profile_preds_summarized[key] = {
                    "y_before_means": solo_reduced[key]["y_before"]["profile_foldmean"],
                    "y_before_sd": solo_reduced[key]["y_before"]["profile_foldsd"],
                    "y_after_means": solo_reduced[key]["y_after"]["profile_foldmean"],
                    "y_after_sd": solo_reduced[key]["y_after"]["profile_foldsd"],
                }

            # DeepLIFT/SHAP contributions at the best spacing/orientation -----
            if args.run_interpret:
                if not args.overwrite_interpret and os.path.exists(path_contribs_mean_hdf5):
                    ism_logger.info("\tLoading precomputed deep lift shap values...")
                    contribs_mean = load_array_h5(path_contribs_mean_hdf5).float()
                else:
                    ism_logger.info("\tRunning deep lift shap...")
                    contributions_list = []
                    for model, seqs in model_seq_pairs:
                        X_best = multisubstitute(seqs, input_motifs[best_orientation], [best_spacing])
                        contributions_fold = deep_lift_shap(CountWrapper(model.model), X_best.float(), random_state=0, batch_size=16)
                        contributions_list.append(contributions_fold)
                    contributions = torch.stack(contributions_list, dim=0)

                    # folds averaged, sequences kept: every consumer averages over
                    # folds first, and the per-sequence heatmap needs the sequences
                    contribs_mean = contributions.mean(dim=0)

                    save_array_h5(path_contribs_mean_hdf5, contribs_mean, dtype=np.float16)
                    if args.save_all:
                        save_array_h5(path_contribs_all_hdf5, contributions, dtype=np.float16)

                    del contributions_list, contributions
                    torch.cuda.empty_cache()

                # fully collapsed mean track (was mean_contribs.h5), now derived
                contributions_mean = contribs_mean.mean(dim=0)

            # effects per sequence, mean across folds, at the best spacing/orientation.
            # y_before has no spacing axis (the baseline doesn't depend on spacing), so
            # selecting a single spacing from y_after already gives a matching shape
            # counts_foldmean is already averaged over folds, so these are the same
            # per-sequence effect vectors the full arrays produced
            y_effect_joint = (
                spacing_reduced[best_orientation]['y_after']['counts_foldmean'][:, best_position, :]
                - spacing_reduced[best_orientation]['y_before']['counts_foldmean']
            )

            y_effect_mult = (
                solo_reduced['A']['y_after']['counts_foldmean'] + solo_reduced['B']['y_after']['counts_foldmean']
                - solo_reduced['A']['y_before']['counts_foldmean'] - solo_reduced['B']['y_before']['counts_foldmean']
            )

            # Z-scores & entropy over spacings/orientations -------------------
            effects = spacing_dfs.effect_mean.values
            Z = calc_effect_Z_score(effects)
            S = calc_effect_entropy(effects)

            df_z = pd.DataFrame({"orientation": spacing_dfs.key, "spacing": spacing_dfs.spacing, "Z": Z})
            df_z.to_csv(os.path.join(out_dir, "Z_scores.tsv"), sep="\t", index=False)

            # Joint-vs-independent effect comparisons, matching HDMA's 04c-plot_cooperativity_results.Rmd
            # ("hard" vs "soft" syntax classification is done downstream from these two quantities).
            sum_effect = solo_dfs.loc[solo_dfs.key == "A + B", "effect_mean"].values[0]

            # how much stronger the joint prediction is than the independent sum, at the global optimum
            joint_vs_ind_max = max_row.effect_mean - sum_effect

            # does synergy persist broadly across a medium-distance spacing window (20-150bp) at the
            # best orientation, rather than only at the single best (possibly very close-range) spacing?
            med_distance_df = spacing_dfs[(spacing_dfs.key == best_orientation) & (spacing_dfs.spacing.between(20, 150))]
            if len(med_distance_df) > 0:
                med_distance_row = med_distance_df.loc[med_distance_df.effect_mean.idxmax()]
                joint_vs_ind_med_distance = med_distance_row.effect_mean - sum_effect
            else:
                joint_vs_ind_med_distance = np.nan

            # Wilcoxon signed-rank test: H1 = joint effect > sum of independent effects
            results_dict = {
                "motif_name": name_i,
                "ENCID": encid_j,
                # NOTE: seq1/seq2 are ordered per the best arrangement, which may not equal seqA/seqB
                "seq1": input_motifs[best_orientation][0],
                "seq2": input_motifs[best_orientation][1],
                "seqA_palindrome": is_palindrome(input_motifs[best_orientation][0]),
                "seqB_palindrome": is_palindrome(input_motifs[best_orientation][1]),
                "best_orientation": best_orientation,
                "best_spacing": best_spacing,
                "best_seq": (
                    f"{input_motifs[best_orientation][0]}{'N' * best_spacing}{input_motifs[best_orientation][1]}"
                    if best_spacing >= 0 else
                    f"{input_motifs[best_orientation][0]}{input_motifs[best_orientation][1][-best_spacing:]}"
                ),
                "joint_effect": max_row.effect_mean,
                "joint_vs_ind_max": joint_vs_ind_max,
                "joint_vs_ind_med_distance": joint_vs_ind_med_distance,
                "joint_vs_sum_p_value": wilcoxon(y_effect_joint, y_effect_mult, alternative="greater").pvalue,
                "max_Z": Z.max(),
                "entropy": S,
            }
            results_dict["dist_centers"] = len(input_motifs[best_orientation][0]) / 2 + best_spacing + len(input_motifs[best_orientation][1]) / 2

            ism_logger.info(results_dict)
            results = pd.DataFrame(results_dict)

            # save outputs -----------------------------------------------------
            if not os.path.exists(os.path.join(out_dir, "config.tsv")):
                config_df.to_csv(os.path.join(out_dir, "config.tsv"), sep="\t", index=False)
            results.to_csv(os.path.join(out_dir, "results.tsv"), sep="\t", index=False)

            # plots --------------------------------------------------------------
            ism_logger.info("\tBeginning plotting...")

            # sum over the strand/task axis (e.g. +/- strand bpnet profiles) rather than
            # squeeze, since squeeze(1) is a no-op whenever that axis has size > 1
            spacing_preds_mean_across_folds = spacing_reduced["_best"]["profile_foldmean_at_best"].sum(1)

            # z-score the predicted profile per background sequence
            spacing_preds_mean_across_folds_z = []
            for k in range(spacing_preds_mean_across_folds.shape[0]):
                spacing_preds_mean_across_folds_z.append(
                    (spacing_preds_mean_across_folds[k] - spacing_preds_mean_across_folds[k].mean(0)) / spacing_preds_mean_across_folds[k].std(0)
                )
            spacing_preds_mean_across_folds_z = torch.stack(spacing_preds_mean_across_folds_z)

            with rasterized_plot_data():
                plot_effect_zscores(df_z, input_motifs, fig_path=os.path.join(out_dir, "Z_scores_dotplot.pdf"))
            plot_effect_spacing(spacing_dfs, input_motifs, fig_path=os.path.join(out_dir, "effect_spacing.pdf"), n_spaces=args.n_spaces)
            plot_effect_spacing_lines(spacing_dfs, solo_dfs, name_i, fig_path=os.path.join(out_dir, "effect_spacing_lines.pdf"))
            plot_effect_solo(solo_dfs, best_orientation, best_spacing, fig_path=os.path.join(out_dir, "effect_solo.pdf"))
            # profiles are predicted counts per bp, whose absolute scale varies by orders of
            # magnitude across models (BPNet TF ChIP over nonpeak background sits far below
            # ChromBPNet ATAC), so let each figure auto-scale to one clean tick above its own
            # largest displayed value rather than pinning a model-dependent floor
            with rasterized_plot_data():
                plot_profiles_spacing(profile_preds_summarized, input_motifs, fig_path=os.path.join(out_dir, "profiles_per_spacing.pdf"), spacing_offsets=spacing_offsets, y_max_sig_figs=1)
            with rasterized_plot_data():
                plot_profiles_solo(solo_profile_preds_summarized, profile_preds_summarized, best_orientation, best_spacing, fig_path=os.path.join(out_dir, "profiles_solo.pdf"), spacing_offset=best_spacing_offset, y_max_sig_figs=1)

            # windows for highlighting the two motifs, relative to the midpoint of the 2114 bp input window
            coords = {
                "zoom_start": MIDPOINT - 50,
                "zoom_end": MIDPOINT + 50,
            }
            if best_spacing % 2 == 0:
                coords["end_A"] = MIDPOINT - best_spacing / 2 - 0.5
                coords["start_B"] = MIDPOINT + best_spacing / 2 - 0.5
            else:
                coords["end_A"] = MIDPOINT - best_spacing // 2 + 0.5
                coords["start_B"] = MIDPOINT + best_spacing // 2 + 1.5
            coords["start_A"] = coords["end_A"] - len(input_motifs[best_orientation][0])
            coords["end_B"] = coords["start_B"] + len(input_motifs[best_orientation][1])

            if args.run_interpret:
                # the contribution matrix actually drawn as the logo (third row) of
                # contributions_mean.pdf: fold- and sequence-averaged scores over the
                # +/-50 bp zoom window, at the best orientation/spacing. Saved with the
                # motif coordinates so the matrix can be interpreted standalone.
                save_array_h5(
                    path_contribs_mean_motif_hdf5,
                    contributions_mean[:, coords['zoom_start']:coords['zoom_end']],
                    attrs={
                        "best_orientation": best_orientation,
                        "best_spacing": best_spacing,
                        "seqA": input_motifs[best_orientation][0],
                        "seqB": input_motifs[best_orientation][1],
                        # inserted-motif bounds, relative to the start of this matrix
                        "start_A": coords['start_A'] - coords['zoom_start'],
                        "end_A": coords['end_A'] - coords['zoom_start'],
                        "start_B": coords['start_B'] - coords['zoom_start'],
                        "end_B": coords['end_B'] - coords['zoom_start'],
                        # and the window this was sliced from, in the 2114 bp input frame
                        "zoom_start": coords['zoom_start'],
                        "zoom_end": coords['zoom_end'],
                    },
                )

                plot_mean_contributions(profile_preds_summarized, contributions_mean, coords, input_motifs, best_spacing, best_orientation,
                    fig_path=os.path.join(out_dir, "contributions_mean.pdf"), spacing_offset=best_spacing_offset)

                contribs_sum_mean_across_folds = contribs_mean.sum(1)
                with rasterized_plot_data():
                    plot_contributions_heatmap(contribs_sum_mean_across_folds, coords, best_orientation, best_spacing, fig_path=os.path.join(out_dir, "contributions_heatmap.pdf"))

            with rasterized_plot_data():
                plot_profile_heatmap_z(spacing_preds_mean_across_folds_z, coords, best_orientation, best_spacing, fig_path=os.path.join(out_dir, "profile_heatmap.pdf"))

            df_pred_effect = pd.DataFrame({
                "example": np.arange(y_effect_joint.shape[0]),
                "Joint": y_effect_joint.squeeze(),
                "A + B": y_effect_mult.squeeze(),
                "A": (solo_reduced['A']['y_after']['counts_foldmean'] - solo_reduced['A']['y_before']['counts_foldmean']).squeeze(),
                "B": (solo_reduced['B']['y_after']['counts_foldmean'] - solo_reduced['B']['y_before']['counts_foldmean']).squeeze(),
            })
            plot_effect_boxplot(df_pred_effect, best_orientation, best_spacing, fig_path=os.path.join(out_dir, "effect_boxplot.pdf"))

            plt.close("all")

            open(os.path.join(out_dir, ".done"), 'a').close()
            ism_logger.info(f"\tDone with {name_i} in {encid_j}.")
            ism_logger.removeHandler(ism_ch2)

        except Exception as e:
            ism_logger.error(f"Error processing {name_i} in {encid_j}: {e}")
            ism_logger.removeHandler(ism_ch2)
            continue

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    del model_seq_pairs
    torch.cuda.empty_cache()
    gc.collect()


#### MAIN -----------------------------------------------------------------------
if __name__ == "__main__":
    args = setup_parser()

    torch.manual_seed(args.seed)
    ism_logger.info(f"GPU available: {torch.cuda.is_available()}")
    genome_fa = Fasta(args.genome_fa)

    ## ISM Cooperativity
    # Load MotifCompendium: ENCODE composite (per motif)
    mc_encode_composite_metadata_df = pd.read_csv(args.motif_metadata, sep="\t", header=0)

    # Load MotifCompendium: ENCODE composite (per model). One row per model (ID),
    # with name_col holding a comma-separated aggregation of every motif name that
    # model should run.
    mc_encode_composite_metadata_model_df = pd.read_csv(args.run_metadata, sep="\t", header=0)

    # Load ENCID metadata
    encid_metadata_df = pd.read_csv(args.model_metadata, sep="\t", header=0)

    model_idxs = args.model_idx if args.model_idx else range(len(mc_encode_composite_metadata_model_df))
    for model_idx in model_idxs:
        run_composite_ism_model(args, model_idx, mc_encode_composite_metadata_df, mc_encode_composite_metadata_model_df,
                                encid_metadata_df, genome_fa)

    print(f"Completed ISM cooperativity: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
