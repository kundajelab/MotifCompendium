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

Run the scripts from the repository root. This matters because they import
modules through the top-level `pipeline` package.

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
creates meta-clusters and sub-clusters, and exports metadata.

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

- `--sim-threshold-meta` groups primary clusters into broader meta-clusters.
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

The output root contains the principal `.mc` objects directly, for example:

- `motifcompendium.mc`: loaded/built input after preprocessing.
- `motifcompendium_labeled.mc`: motif-level reference matches, when requested.
- `motifcompendium_filtered.mc` and `motifcompendium_removed.mc`: retained and
  rejected motifs, when filtering is enabled.
- `motifcompendium_clustered.mc`: motifs with final cluster assignments.
- `motifcompendium_cluster_filtered.mc`: retained cluster averages.
- `motifcompendium_metacluster_filtered.mc` and
  `motifcompendium_subcluster_filtered.mc`: optional hierarchy levels.
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
all-cluster, and meta-cluster intermediate objects and exports.

## 3. ENCODE pipeline

The ENCODE entry point constructs two atlases, each from counts and profile
TF-MoDISco heads, then combines the atlases:

```text
TF counts ─┐
           ├─ TF atlas ─────────┐
TF profile ┘                    │
                               ├─ ENCODE collection
Chromatin counts ─┐             │
                  ├─ Chromatin ─┘
Chromatin profile ┘   atlas
```

The seven stage names, in execution order, are:

1. `tf-counts`
2. `tf-profile`
3. `tf-combine`
4. `chrom-counts`
5. `chrom-profile`
6. `chrom-combine`
7. `encode-combine`

With no `--stages`, all seven are selected. A supplied subset is always run in
the order above, regardless of the order on the command line.

### Fixed output layout and dependencies

All paths are derived from one `--output-dir`:

```text
OUTPUT/
├── tfatlas/
│   ├── counts/
│   ├── profile/
│   └── combined/
├── chromatlas/
│   ├── counts/
│   ├── profile/
│   └── combined/
└── encode/
```

The combine stages load their upstream products from these exact locations.
Consequently, selecting `tf-combine`, `chrom-combine`, or `encode-combine`
without its prerequisite stages is a resume operation: the expected outputs
must already exist under the same output root. Combine stages also write
`*-indexed.mc` products back into their upstream stage directories.

Each stage writes some combination of `mc/`, `html/`, `export/`, `metadata/`,
and `logs/`. Every stage records arguments/configuration in
`metadata/args.json`. Principal hierarchy files include:

- Head stages: `mc/motifcompendium_motif_all.mc`,
  `motifcompendium_cluster_all.mc`, and `motifcompendium_metacluster_all.mc`.
- Atlas combine stages: `mc/motifcompendium_atlascluster_all.mc` and filtered
  and removed variants.
- ENCODE combine: `mc/motifcompendium_encode_all.mc` and filtered and removed
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

### Current status of `encode-combine`

In the current source, the final `encode-combine` stage is not runnable end to
end solely through the documented CLI:

- It validates eight options as present: `--tf-init-metadata`,
  `--chrom-init-metadata`, `--mc-qc-remove-removed`,
  `--mc-qc-pass-filtered`, `--mc-qc-pass-lowq`, `--metadata-edited`,
  `--hg38-fasta`, and `--gencode-gtf`.
- The stage body later reads `args.metadata`, but the parser does not define
  `--metadata` and the wrapper does not map either initial-metadata option to
  it. This raises an `AttributeError` before final export.
- Hit annotation also expects pre-existing
  `encode/metadata/motifcompendium_encode_metadata_hits.tsv` and
  `encode/raw/encid_metadata.tsv`; no CLI options populate those paths.
- `--mc-name`, `--modisco-region-width`, and `--mc-ref-clustered-hr` are
  currently accepted (the last is required) but not read; only
  `--mc-ref-clustered-hr-metadata` is used for the readable-name mapping.

Stages 1–6 can be run as shown above. Before using stage 7, fix or supply the
missing metadata/hit-index wiring for the intended dataset. The stage also
needs an indexed hg38 FASTA (the `.fai` must be available) and a GENCODE GTF for
genomic hit annotation. Do not treat a run that stops after stage 6 as a
completed ENCODE collection.

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
