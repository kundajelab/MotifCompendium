# MotifCompendium pipelines

This directory contains three command-line pipelines using MotifCompendium:

1. [`basic/pipeline.py`](basic/pipeline.py) is the general-purpose pipeline for loading, annotating, filtering, clustering, and visualizing a motif set.
2. [`jaspar/pipeline_jaspar.py`](jaspar/pipeline_jaspar.py) reproduces the JASPAR 2026 deep-learning collection workflow from [JASPAR 2026: expansion of transcription factor binding profiles and integration of deep learning models](https://doi.org/10.1093/nar/gkaf1209).
3. [`encode/pipeline_encode.py`](encode/pipeline_encode.py) reproduces the ENCODE MotifCompendium from [A unified lexicon of predictive DNA sequence motifs from ENCODE transcription factor binding and chromatin accessibility assays](https://doi.org/10.5281/zenodo.17123347).

These are advanced workflows whose scientific defaults live in the adjacent
`configs.py` files. Read and, when appropriate, edit those configurations before
using the JASPAR or ENCODE pipelines for a different collection.

## Setup and invocation

Install MotifCompendium as described in the repository-level
[`README.md`](../README.md). From the repository root:

```bash
conda activate motifcompendium       # or motifcompendium-gpu
pip install -e .
```

Each script imports the `configs.py` in its own directory, so it can be run from
any working directory. The examples below use paths relative to the repository
root.

```bash
python pipeline/basic/pipeline.py --help
python pipeline/jaspar/pipeline_jaspar.py --help
python pipeline/encode/pipeline_encode.py --help
```

In the examples below, values such as `/path/to/...` are placeholders. Output
directories are created automatically, but parent locations must be writable.

### Input formats shared by the pipelines

- A MotifCompendium object is a serialized `.mc` file.
- A TF-MoDISco/TF-MoDISco-lite input is an HDF5 file containing
  `pos_patterns` and/or `neg_patterns`.
- PFM inputs may be PFM files or MEME-format motif collections.
- `--input-subpatterns` switches TF-MoDISco loading from main patterns to
  subpatterns.
- Where `--input-names` is supported, supply one unique name for every input
  file, in the same order. If omitted, file basenames are used.

When more than one alternative input type is supplied, the scripts use the
first applicable type rather than combining them. Prefer supplying exactly one
input type per run.

### Compute options shared by the pipelines

The three entry points share these main compute controls:

| Option | Meaning |
| --- | --- |
| `--use-gpu` / `-g` | Request GPU computation. Use the GPU environment. |
| `--max-cpus N` / `-cp N` | Maximum CPU count; default is `1`. |
| `--max-chunk N` / `-ch N` | Maximum motifs processed together; default is `1000`; `-1` disables chunking. |
| `--var-chunk` | Retry a failed build/clustering operation with progressively smaller chunks. |
| `--no-ic` | Disable information-content scaling. |
| `--fast-plot` | Make lower-resolution motif logos more quickly. |
| `--unsafe` | Disable MotifCompendium validity checks. Use only for trusted/problematic legacy data. |
| `--time` / `-t` | Report step timings. |
| `--verbose` / `-v` | Print detailed progress. |

`--var-chunk` reduces the chunk size in increments of 100. It is useful for
memory pressure, but it does not repair malformed inputs.

## 1. Basic pipeline

The basic pipeline is the recommended entry point for a new motif collection.
It builds or loads a MotifCompendium, optionally merges metadata and reference
labels, filters motifs, clusters them, computes cluster averages, optionally
creates contribclusters and sub-clusters, and exports metadata.

### Minimal examples

Cluster motifs from two TF-MoDISco files:

```bash
python pipeline/basic/pipeline.py \
  --input-modisco-h5s /path/to/model_a/modisco.h5 /path/to/model_b/modisco.h5 \
  --input-names model_a model_b \
  --output-dir /path/to/output/basic \
  --max-cpus 8 \
  --verbose --time
```

Cluster a MEME motif collection without the default filters:

```bash
python pipeline/basic/pipeline.py \
  --input-pfms /path/to/motifs.meme.txt \
  --output-dir /path/to/output/basic \
  --no-filter
```

Continue from an existing or legacy object:

```bash
# Current object
python pipeline/basic/pipeline.py \
  --input-mc /path/to/input.mc \
  --output-dir /path/to/output/basic

# Object saved by an older MotifCompendium version
python pipeline/basic/pipeline.py \
  --input-old-mc /path/to/old_input.mc \
  --output-dir /path/to/output/basic
```

### Metadata

Attach optional CSV or TSV metadata with `--metadata`:

```bash
python pipeline/basic/pipeline.py \
  --input-modisco-h5s /path/to/a.h5 /path/to/b.h5 \
  --input-names a b \
  --metadata /path/to/models.tsv \
  --output-dir /path/to/output/basic
```

Metadata is merged in one of two ways:

- If its row count equals the number of motifs, rows are joined to motifs by
  index.
- Otherwise it is treated as per-model metadata. Its first column is joined to
  the MotifCompendium `model` column, or to `source` when `model` is absent.

Do not include the reserved columns `name`, `num_seqlets`, `posneg`, or
`avg_dist_from_summit`; they are produced by the loader. Any metadata column
used by `--cluster-on` or `--cluster-within` must be present after this merge.

### Reference annotation

Use `--reference` to label motifs and cluster averages against one primary
`.mc`, PFM, or MEME database. `--add-reference` accepts additional databases;
their filename stems become output-column prefixes.

```bash
python pipeline/basic/pipeline.py \
  --input-pfms /path/to/query.meme.txt \
  --reference pipeline/data/MotifCompendium-Database-Human.meme.txt \
  --add-reference /path/to/another_reference.mc \
  --output-dir /path/to/output/basic \
  --html-motif-table --html-cluster-table
```

No reference database is selected by default.

### Filtering and clustering

Default motif and cluster filters are defined in
[`basic/configs.py`](basic/configs.py) and are enabled unless `--no-filter` is
passed. With a primary reference, high-scoring matches can override ordinary
filter flags; `--strict-filter` enables strict filters that are not rescued by
such matches.

The primary clustering threshold defaults to `0.88`:

```bash
python pipeline/basic/pipeline.py \
  --input-mc /path/to/input.mc \
  --output-dir /path/to/output/basic \
  --sim-threshold 0.88 \
  --sim-threshold-meta 0.80 \
  --sim-threshold-sub 0.93 \
  --cluster-recursive \
  --quality
```

- `--sim-threshold-meta` groups primary clusters into broader contribclusters.
- `--sim-threshold-sub` subdivides motifs within each primary cluster.
- `--sim-threshold-force` repeatedly applies dense connected-components
  clustering to enforce the requested separation.
- `--sim-threshold-clean` reassigns motifs using the configured centroid
  cleanup algorithm when similarity improves by at least that value.
- `--cluster-within COLUMN` prevents motifs in different values of `COLUMN`
  from clustering together.
- `--cluster-on COLUMN` first aggregates an existing categorization and
  clusters those groups.
- `--sim-scan T1 T2 ...` runs several primary thresholds. The normal
  `--sim-threshold` is appended if absent, and the final threshold processed is
  the one used for the final averaged-cluster outputs. Combine this option with
  `--quality` when comparing thresholds.

The exact algorithms, aggregation columns, seeds, and filter thresholds are in
[`basic/configs.py`](basic/configs.py).

### HTML and output files

HTML is opt-in. Common choices are:

```bash
--html-motif-collection \
--html-motif-table \
--html-motif-removed \
--html-cluster-table \
--html-cluster-removed \
--html-max-rows 1000 \
--logo-trimming 0.1
```

`--logo-trimming` accepts `true`, `false`, or a number from 0 to 1.

The `.mc` objects are saved in `mc/`, for example:

- `mc/motifcompendium.mc`: loaded/built input after preprocessing.
- `mc/motifcompendium_filtered.mc` and `mc/motifcompendium_removed.mc`: retained
  and rejected motifs, when filtering is enabled.
- `mc/motifcompendium_cluster_filtered.mc`: retained cluster averages.
- `mc/motifcompendium_contribcluster_filtered.mc` and
  `mc/motifcompendium_subcluster_filtered.mc`: optional hierarchy levels.
- `mc/motifcompendium_clustered.mc` (`--save-clustered`): motifs with final
  cluster assignments. Not saved by default; the assignments are also in
  `export/motifcompendium_metadata.tsv`.
- `mc/motifcompendium_labeled.mc`, `mc/motifcompendium_{cluster,contribcluster,subcluster}_labeled.mc`
  (`--save-labeled`): reference matches before filtering, when a reference is
  given. Not saved by default.
- `html/`: requested interactive tables and motif collections.
- `quality/`: requested clustering diagnostics.
- `export/args.json`: command-line and configuration provenance.
- `export/*_metadata.tsv`: motif and cluster metadata tables.

## 2. JASPAR 2026 deep-learning pipeline

The JASPAR pipeline implements the specialized workflow associated with
[JASPAR 2026: expansion of transcription factor binding profiles and
integration of deep learning models](https://doi.org/10.1093/nar/gkaf1209).
It is not a parametrized replacement for the basic pipeline: its filtering,
classification, matching, clustering, and export rules are fixed in
[`jaspar/configs.py`](jaspar/configs.py).

### Required inputs

Supply one query source:

- `--input-mc FILE`, or
- `--input-modisco-h5s FILE [FILE ...]`, or
- `--input-pfms FILE [FILE ...]`.

Four reference collections are mandatory:

| Option | Reference |
| --- | --- |
| `--jaspar-core` | Human JASPAR CORE motifs. |
| `--jaspar-unvalid` | Human JASPAR UNVALIDATED motifs. |
| `--ref` | Main/core comparison reference. |
| `--ref-invitro` | In-vitro comparison reference. |

Each reference may be a MotifCompendium `.mc` object or a MEME/PFM collection.
The reference metadata must contain the columns expected by the JASPAR config;
these files are scientific inputs, not interchangeable labels-only motif sets.

### Example

```bash
python pipeline/jaspar/pipeline_jaspar.py \
  --input-modisco-h5s /path/to/model_a.h5 /path/to/model_b.h5 \
  --input-names model_a model_b \
  --metadata /path/to/model_metadata.tsv \
  --jaspar-core /path/to/jaspar_core_human.mc \
  --jaspar-unvalid /path/to/jaspar_unvalidated_human.mc \
  --ref /path/to/core_reference.mc \
  --ref-invitro /path/to/invitro_reference.mc \
  --output-dir /path/to/output/jaspar \
  --max-cpus 8 \
  --verbose --time
```

For HDF5 input, provide exactly as many `--input-names` as HDF5 files. The
optional metadata file is CSV or TSV. If its number of rows equals the motif
count it is joined by row index; otherwise its first column is joined to the
input `model` or `source` column.

The JASPAR script checks for a CUDA device when `--use-gpu` is requested and
falls back to CPU with a warning if none is available.

### Outputs

The output directory has four main subdirectories:

- `mc/`: MotifCompendium objects for all motifs, in-vitro-supported motifs,
  cluster averages, and PPM/PFM representations.
- `html/`: cluster summaries.
- `export/`: TF-MoDISco HDF5 and MEME exports.
- `metadata/`: TSV metadata tables.

The main products are
`mc/motifcompendium_motif_all.mc`,
`mc/motifcompendium_motif_invitro.mc`, and
`mc/motifcompendium_cluster_invitro.mc`, with matching HDF5, MEME, and metadata
exports. Pass `--save-all` to retain additional non-in-vitro, removed,
all-cluster, and contribcluster intermediate objects and exports.

## 3. ENCODE pipeline

The ENCODE entry point (`encode/pipeline_encode_full.py`) constructs three
atlases, each from counts and profile TF-MoDISco heads, then combines them and
finalizes the composite motifs:

```text
TF counts ─┐
           ├─ TF atlas ─────────┐
TF profile ┘                    │
Chromatin counts ─┐             ├─ ENCODE combine ─ composite ISM ─ composite ─────── propagate ─ hits
                  ├─ Chromatin ─┤   (+ composites)    (per model)    (encode-        (the levels  (by encode-
Chromatin profile ┘   atlas     │                                    cluster, C5)    below C5)    cluster)
ProCap counts ─┐                │
               ├─ ProCap atlas ─┘
ProCap profile ┘
```

The fourteen stage names, in execution order, are:

1. `tf-counts`
2. `tf-profile`
3. `tf-combine`
4. `chrom-counts`
5. `chrom-profile`
6. `chrom-combine`
7. `procap-counts`
8. `procap-profile`
9. `procap-combine`
10. `encode-combine`: `encode/pipeline_encode.py`
11. `encode-composite-ism`: `encode/pipeline_encode_composite_ism.py`
12. `encode-composite`: `encode/pipeline_encode_composite.py`
13. `encode-propagate`: each clustering stage's `--update-constituents only`
14. `encode-hits`: `encode/pipeline_encode_hits.py`

With no `--stages`, all fourteen are selected. A supplied subset is always run in
the order above, regardless of the order on the command line.

### Fixed output layout and dependencies

All paths are derived from one `--output-dir`:

```text
OUTPUT/
├── bpnet/        (TF atlas)
│   ├── counts/
│   ├── profile/
│   └── combined/
├── chrombpnet/   (Chromatin atlas)
│   ├── counts/
│   ├── profile/
│   └── combined/
├── procapnet/    (ProCap atlas)
│   ├── counts/
│   ├── profile/
│   └── combined/
├── cross/        (crossclusters, C4: compendium, motif index)
└── encode/       (composites, their ISM runs in ism/, the merged composites; the
                   encodecluster level, C5, its motif index and the hits)
```

The combine stages load their upstream products from these exact locations.
Consequently, selecting `tf-combine`, `chrom-combine`, or `encode-combine`
without its prerequisite stages is a resume operation: the expected outputs
must already exist under the same output root.

Each clustering stage also updates the levels below it (its constituents), over the
files there: an atlas combine stage adds its atlas-cluster names to the per-head
`motifcompendium_*_all.mc` (and writes the modelcluster motif index); the ENCODE
combine its crosscluster names, final filter flags and confidence to every atlas
level (`motifcompendium_{level}_{filtered,removed,all}.mc`, with their MEME, HTML
and metadata TSV) and the motif indexes; the composite stage its encodeclusters
to the crossclusters, every atlas level and the motif indexes. Each script's
`--update-constituents` sets when: `now` (default, the stage itself), `defer` (not: the
levels below stay as they were made) or `only` (that update alone, from the stage's
saved outputs, nothing else run). The ENCODE pipeline defers them all (C3 and C4
stay as their stages wrote them, so they can be redone) until C5 is made:
`encode-propagate` then runs each stage's update, in the order the levels were made
(the atlases', the ENCODE combine's, the composite stage's), so that each level ends
with the last one's. After that, the atlas's own selection is gone, so the combine
stages refuse to rerun on it: rerun the atlas first.

Each level saves only `mc/motifcompendium_{level}_{all,filtered}.mc`
(`{level}`: `motif`, `cluster`, `contribcluster`, `modelcluster`, `crosscluster`), plus
`motifcompendium_contribcluster_combined.mc` (atlas combine),
`motifcompendium_modelcluster_combined.mc` (ENCODE combine) and
`motifcompendium_motif_init.mc` (head stages). Its removed clusters
(`motifcompendium_{level}_removed`: `.mc`, HTML, MEME, metadata TSV, MoDISco H5)
are saved only with `--save-all` (every script of the ENCODE pipeline has it; the
encode/src launchers pass it); the later stages read the `_all` and `_filtered`
ones. The encodecluster level (C5) is `encode/mc/motifcompendium_encode.mc`
(no `_all`/`_filtered`); its removed compendium,
`encode/mc/motifcompendium_encode_removed.mc` (+ HTML, MEME, metadata TSV;
`--save-all`), holds the combine's filtered crossclusters it leaves out.

Each stage writes some combination of `mc/`, `html/`, `export/`, `metadata/`,
and `logs/`. Every stage records arguments/configuration in
`metadata/args.json`. Principal hierarchy files include:

- Head stages: `mc/motifcompendium_motif_all.mc`,
  `motifcompendium_cluster_all.mc`, and `motifcompendium_contribcluster_all.mc`.
- Atlas combine stages: `mc/motifcompendium_modelcluster_all.mc` and filtered
  and removed variants.
- ENCODE combine: `mc/motifcompendium_crosscluster_all.mc` and filtered and removed
  variants.

### Shared references

The parser currently requires all of the following on every invocation,
including a partial-stage invocation:

```bash
--mc-ref-qc /path/to/encode_qc_reference.mc \
--mc-ref-direct /path/to/encode_direct_reference.mc \
--mc-ref-clustered-hr /path/to/human_readable_clustered_reference.mc \
--mc-ref-clustered-hr-metadata /path/to/human_readable_reference_metadata.tsv \
--mc-b1h /path/to/b1h_znf_reference.mc \
--ref-invitro /path/to/invitro_reference.mc \
--column-desc-tsv /path/to/html_column_descriptions.tsv
```

The scientific configuration expects reference metadata fields used for TF,
family, directness, and readable-label annotation. Optional `--tf-class` (TSV)
and `--human-tf` (CSV) add TF-family/DBD annotations to TF stages.

### Running stages 1–6

For each of the four heads, supply either its `.mc` input or one or more
TF-MoDISco HDF5 files. This example builds both atlases but stops before the
final ENCODE merge:

```bash
python pipeline/encode/pipeline_encode.py \
  --stages tf-counts tf-profile tf-combine \
           chrom-counts chrom-profile chrom-combine \
  --output-dir /path/to/output/encode_run \
  --tf-counts-modisco-h5s /path/to/tf_counts_a.h5 /path/to/tf_counts_b.h5 \
  --tf-counts-input-names tf_counts_a tf_counts_b \
  --tf-counts-metadata /path/to/tf_counts_metadata.tsv \
  --tf-profile-modisco-h5s /path/to/tf_profile_a.h5 /path/to/tf_profile_b.h5 \
  --tf-profile-input-names tf_profile_a tf_profile_b \
  --tf-profile-metadata /path/to/tf_profile_metadata.tsv \
  --chrom-counts-input-mc /path/to/chrom_counts.mc \
  --chrom-counts-metadata /path/to/chrom_counts_metadata.tsv \
  --chrom-profile-input-mc /path/to/chrom_profile.mc \
  --chrom-profile-metadata /path/to/chrom_profile_metadata.tsv \
  --mc-ref-qc /path/to/encode_qc_reference.mc \
  --mc-ref-direct /path/to/encode_direct_reference.mc \
  --mc-ref-clustered-hr /path/to/human_readable_clustered_reference.mc \
  --mc-ref-clustered-hr-metadata /path/to/human_readable_reference_metadata.tsv \
  --mc-b1h /path/to/b1h_znf_reference.mc \
  --ref-invitro /path/to/invitro_reference.mc \
  --column-desc-tsv /path/to/html_column_descriptions.tsv \
  --tf-class /path/to/tfclass.tsv \
  --human-tf /path/to/human_tf.csv \
  --max-cpus 8 \
  --verbose --time
```

To resume only the two atlas combines after all four head directories have
already been populated:

```bash
python pipeline/encode/pipeline_encode.py \
  --stages tf-combine chrom-combine \
  --output-dir /path/to/output/encode_run \
  --mc-ref-qc /path/to/encode_qc_reference.mc \
  --mc-ref-direct /path/to/encode_direct_reference.mc \
  --mc-ref-clustered-hr /path/to/human_readable_clustered_reference.mc \
  --mc-ref-clustered-hr-metadata /path/to/human_readable_reference_metadata.tsv \
  --mc-b1h /path/to/b1h_znf_reference.mc \
  --ref-invitro /path/to/invitro_reference.mc \
  --column-desc-tsv /path/to/html_column_descriptions.tsv
```

The default MCID versions are `V1` for TF, `V0` for chromatin, and `V1` for
ENCODE. Override them with `--mcid-version-tf`, `--mcid-version-chrom`, and
`--mcid-version-encode` only when deliberately creating a different identifier
namespace.

### ENCODE stages

Each is also a script of its own, with the atlases' output roots as
`--tfatlas-dir`, `--chromatlas-dir`, `--procapatlas-dir`, the ENCODE stage's
directory as `--output-dir` and the composites' and encodecluster
level's as `--composite-dir` (default: `encode/`, next to `--output-dir`):

10. `encode-combine` (`pipeline_encode.py`): force-clusters the modelclusters
    into crossclusters, annotates and filters them, and decomposes the filtered
    ones into reference monomers (modified Bellman-Ford), before exporting
    them (and, at `encode-propagate`, to every level below). The composites and their subunits are written to
    `encode/` (`export/motifcompendium_encode_composite_metadata{,_model}.tsv`,
    `mc/motifcompendium_encode_composite.mc`). `--all-paths-tsv` (optional)
    adds each model's model, assay and atlas to the per-model table. Its
    update of every level below writes, over the atlas pipelines' files, each
    level's compendia (all, filtered, removed: `.mc`, metadata TSV, MEME, HTML)
    and the motif indexes, with their crossclusters.
11. `encode-composite-ism` (`pipeline_encode_composite_ism.py`): ISM
    cooperativity of every composite in every model it was found in. It needs
    torch, bpnetlite and tangermeme, so `pipeline_encode_full.py` runs it as a
    subprocess of `--ism-python`, model after model; for the full collection,
    run it as a SLURM array instead (`--model-idx`, one model per task), then
    the remaining stages. Its helpers, `utils_composite_ism.py`, are HDMA's
    `tangermeme_utils` (https://github.com/GreenleafLab/HDMA/), copied.
12. `encode-composite` (`pipeline_encode_composite.py`): classifies the
    composites by cooperativity (hard, soft, no synergy), merges the duplicate
    ones and the soft and no-synergy ones of the same reference clusters,
    removes the no-synergy ones, and combines the rest with the singles: the
    encodecluster level (C5), written like a crosscluster level to
    `encode/` (`motifcompendium_encode`: `.mc`, HTML, metadata
    TSV, MEME; the crossclusters in no encodecluster are the combine's
    removed ones and the composites left out). The composites left out (no
    synergy; no subunits) are its removed compendium,
    `motifcompendium_encode_removed` (the same files, `--save-all`): the
    merged no-synergy composites with their ISM evidence, numbered and with
    C5 MCIDs after the encodeclusters, named by flag type as `encode-combine`
    names its removed crossclusters (`no-synergy_0`, ...); the membership
    does not name them. As a clustering, each encodecluster has a number
    (`source_cluster`) and a C5 MCID, and each crosscluster of
    `encode-combine` its encodecluster, or why it has none
    (`encode/metadata/motifcompendium_encode_membership.tsv`:
    the crosscluster's `MCID`, `name`, `source_cluster`, its
    encodecluster's `encodecluster_{cluster,MCID,name,simple_name}`,
    `encodecluster_flag_remove` and `encodecluster_flag_type`);
    they are named as `encode-combine` names its own (`{prename}_{counter}`).
    As at the other levels, a column about another level is named after it
    (`modelcluster_name`, `crosscluster_name`, `encodecluster_*`).
    Its HTML reports (`encode/html/motifcompendium_encode_composite_ism*.html`)
    are always written; the ISM evidence
    (`mc/motifcompendium_encode_composite_ism_motifs{_init,}.mc`, its metadata
    TSVs) and the classified composites (`mc/motifcompendium_encode_composite_ism.mc`)
    only with `--save-all`.
    Its update of the levels below keeps their names, MCIDs and flags and adds
    columns: the crossclusters `encodecluster_cluster`,
    `encodecluster_name` and `encodecluster_flag_remove`; every atlas
    level and the motif indexes `encodecluster_name` and
    `encodecluster_flag_remove`; then the level's motif index (the hits':
    each motif's `encodecluster_name`, with its crosscluster's
    `crosscluster_name`, and the encodecluster's flag, confidence, simple name
    and category).
    It can be rerun on the same `encode-combine` outputs.
13. `encode-propagate`: the levels below C5, once it is checked (`--stages ...
    encode-composite`, then `--stages encode-propagate encode-hits`): each atlas
    pipeline's, `pipeline_encode.py`'s and `pipeline_encode_composite.py`'s
    `--update-constituents only`, in that order.
14. `encode-hits` (`pipeline_encode_hits.py`): indexes each experiment's FiNeMo
    hits with the encodeclusters (`--level encodecluster`, the
    default; `--level crosscluster`: the crossclusters) and annotates them, in
    `encode/export/` (needs an indexed hg38 FASTA, `--hg38-fasta`, and a
    GENCODE GTF, `--gencode-gtf`).

## Reproducibility and troubleshooting

- Start with `--verbose --time`; these pipelines can be long-running and the
  extra progress messages make input and memory failures much easier to locate.
- Use a new output directory for parameter changes. The scripts create files
  incrementally and do not clean stale products from an earlier run.
- Keep each run's `args.json`. It captures command-line values and configured
  defaults; ENCODE stages also record available upstream provenance.
- If an HDF5 file is rejected, confirm it is a nonempty TF-MoDISco file with a
  top-level `pos_patterns` or `neg_patterns` group.
- If memory is exhausted, reduce `--max-chunk`, enable `--var-chunk`, or move to
  the GPU environment and pass `--use-gpu`.
- Configuration changes alter the scientific result. Record changes to
  [`basic/configs.py`](basic/configs.py),
  [`jaspar/configs.py`](jaspar/configs.py), or
  [`encode/configs.py`](encode/configs.py) together with the output provenance.
