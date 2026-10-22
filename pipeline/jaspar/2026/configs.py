from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Union


## TF-MODISCO -----------------------------------------------------------------------------------------------------
@dataclass
class TFMoDIScoCols:
    region_width: int = 400


## METADATA -------------------------------------------------------------------------------------------------------
@dataclass
class MetadataCols:
    # General naming / ID columns
    name_col: str = 'name'
    prename_col: str = 'prename'
    simple_name_col: str = 'simple_name'
    name_new_col: str = 'new_name'
    name_tfmodisco_col: str = 'name_tfmodisco'
    name_constituents_col: str = 'name_constituents'
    name_unknown: str = 'unvalidated'
    counter_col: str = 'counter'

    # Unique identifier: JASPAR.{version}.C{0: motif, 1: cluster, 2: metacluster}.{index}
    unique_id_col: str = 'JASPAR'

    id_col: str = 'ID'
    tf_col: str = 'target'
    model_col: str = 'model'
    head_col: str = 'head'
    accession_col: str = 'accession_id'
    biosample_col: str = 'biosample'
    assay_col: str = 'assay'

    composite_col: str = 'composite'  # True, False
    num_subunit_col: str = 'num_subunit'
    num_subunit_direct_col: str = 'num_subunit_direct'

    posneg_col: str = 'posneg'
    num_seqlets_col: str = 'num_seqlets'
    num_hits_col: str = 'num_hits'
    num_constituents_col: str = 'num_constituents'
    avgcontrib_score_col: str = "avg_contrib_score"
    total_contrib_score_col: str = "total_contrib_score"
    group_total_contrib_score_col: str = "group_total_contrib_score"
    total_contrib_score_ratio_col: str = "total_contrib_score_ratio"
    avgdist_summit_col: str = "avg_dist_from_summit"

    # FiNeMo hits, per model
    hits_report_tsv_col: str = 'hits_report_tsv_path'
    hits_ppm_npy_col: str = 'hits_ppm_npy_path'
    hits_motif_name_col: str = 'motif_name'
    hits_num_col: str = 'num_hits_total'


## TF ANNOTATION ----------------------------------------------------------------------------------------------------
@dataclass
class TFAnnotationCols:
    family_col: str = 'family'

    # Reference annotation
    ref_tf_col: str = "TF"
    ref_family_col: str = "family"


## REFERENCE MATCH --------------------------------------------------------------------------------------------------
@dataclass
class ReferenceMatchCols:
    # General reference (--ref)
    ref_col: str = "reference"
    ref_name_col: str = 'reference_name'
    ref_name_constit_1_col: str = "reference_name_constit_prev"
    ref_name_constit_all_col: str = "reference_name_constit_all"
    ref_readable_col: str = "reference_readable"

    # In vitro: the target's JASPAR (CORE + UNVALIDATED) + in vitro reference
    ref_invitro_col: str = "invitro"
    ref_invitro_name_constit_1_col: str = "invitro_name_constit_prev"
    ref_invitro_name_constit_all_col: str = "invitro_name_constit_all"

    # Cluster / metacluster annotation
    ref_jaspar_col: str = "JASPAR-2026-HUMAN"
    ref_jaspar_core_col: str = "JASPAR-2026-HUMAN-CORE"
    ref_jaspar_unvalid_col: str = "JASPAR-2026-HUMAN-UNVALID"
    ref_invitro_other_col: str = "INVITRO-OTHER"


## LABELS ------------------------------------------------------------------------------------------------------------
@dataclass
class LabelCols:
    label_col: str = 'label'  # Metacluster: Readable TF family label: {TF_family0}-{TF_family1}-...
    label_constit_1_col: str = 'label_constit_prev'
    label_constit_all_col: str = 'label_constit_all'

    direct_label_col: str = 'direct_label'  # Metacluster: Readable TF family label (direct only)
    direct_label_constit_1_col: str = 'direct_label_constit_prev'
    direct_label_constit_all_col: str = 'direct_label_constit_all'

    label_core_col: str = 'label_core'  # Core: Main TF
    label_core_col_set: str = 'label_core_set'
    label_core_constit_1_col: str = 'label_core_constit_prev'
    label_core_constit_all_col: str = 'label_core_constit_all'

    direct_label_core_col: str = 'direct_label_core'  # (Direct) Core: Main TF
    direct_label_core_col_set: str = 'direct_label_core_set'
    direct_label_core_constit_1_col: str = 'direct_label_core_constit_prev'
    direct_label_core_constit_all_col: str = 'direct_label_core_constit_all'

    label_core_readable_col: str = 'label_core_readable'  # Core: Main TF family
    label_core_readable_col_set: str = 'label_core_readable_col_set'
    label_core_readable_constit_1_col: str = 'label_core_readable_constit_prev'
    label_core_readable_constit_all_col: str = 'label_core_readable_constit_all'

    direct_label_core_readable_col: str = 'direct_label_core_readable'  # (Direct) Core: Main TF family
    direct_label_core_readable_col_set: str = 'direct_label_core_readable_col_set'
    direct_label_core_readable_constit_1_col: str = 'direct_label_core_readable_constit_prev'
    direct_label_core_readable_constit_all_col: str = 'direct_label_core_readable_constit_all'

    label_flank_col: str = 'label_flank'  # Composite: Subunit TF
    label_flank_col_set: str = 'label_flank_set'
    label_flank_constit_1_col: str = 'label_flank_constit_prev'
    label_flank_constit_all_col: str = 'label_flank_constit_all'

    label_conf_col: str = 'label_confidence'
    label_confs: List[str] = field(default_factory=lambda: ['high', 'medium', 'low', 'flag'])


## MOTIF MATCH --------------------------------------------------------------------------------------------------
@dataclass
class MotifMatchArgs:
    min_single_score_jasp: float = 0.88  # Direct: match with the target's own JASPAR / in vitro reference
    min_single_score: float = 0.8
    min_composite_score: float = 0.7
    max_submotifs_jasp: int = 1
    max_submotifs: int = 3
    min_composite_subunit: float = 1.0


## DIRECT TYPE -------------------------------------------------------------------------------------------------
@dataclass
class DirectTypeArgs:
    direct_type_col: str = 'direct_type'
    tf_direct_col: str = 'target_direct'  # Target TF (direct only)
    family_direct_col: str = 'family_direct'  # Target TF family (direct only)

    # Motif level: direct (matches the target's JASPAR / in vitro reference) or indirect.
    # Cluster level: direct clusters confirmed by the target's JASPAR CORE / UNVALIDATED models take that
    # group; the rest stay DIRECT. Values are compared as whole tokens: 'INDIRECT' contains 'DIRECT'.
    direct_type: str = 'DIRECT'
    indirect_type: str = 'INDIRECT'
    jaspar_core_type: str = 'JASPCORE'
    jaspar_unvalid_type: str = 'JASPUNVALID'
    direct_types: Tuple[str, ...] = ('DIRECT', 'JASPCORE', 'JASPUNVALID')


## FILTERING ---------------------------------------------------------------------------------------------------
@dataclass
class FilterArgs:
    filter_flag_col: str = 'flag_remove'
    filter_flag_type_col: str = 'flag_type'
    filter_trim_importance: float = 0
    filter_trim_length: Union[int, None] = None
    filter_ic_scale: bool = False
    filter_flag_types: List[str] = field(default_factory=lambda: [
        'CpG', 'tandem-repeats', 'high-GC-AT', 'low-complexity', MetadataCols.name_unknown,
    ])


## MOTIF FILTERS (thresholds applied during motif/cluster filtering) ------------------------------------------
@dataclass
class _FilterArgs:
    # INTERNAL: Standard filter arguments
    name: str
    metric: str
    operation: str
    threshold: Union[float, bool, tuple]
    invert: bool
    override: bool
    apply_motif: bool
    apply_cluster: bool
    flag_type: str | None = None

    def to_dict(self):
        # Return filter arguments as a dictionary
        return asdict(self)


@dataclass
class MotifFilterArgs:
    # Motif metrics to be calculated
    motif_metrics: tuple = (
        "motif_entropy",
        "weighted_base_entropy",
        "weighted_position_entropy",
        "posbase_entropy_score",
        "copair_entropy_score",
        "copair_composition",
        "dinuc_entropy_score",
        "dinuc_composition",
        "dinuc_score",
        "posneg_inverted",
        "possum_negsum_ratio",
        "posmax_negmax_ratio",
        "maxmean_ratio",
        "truncated",
    )
    # Soft filters (same as pipeline_tfatlas.py)
    soft_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=MetadataCols.num_seqlets_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=False,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="2_truncated",
            metric="truncated",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_inverted",
            metric="posneg_inverted",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_1",
            metric="possum_negsum_ratio",
            operation="<",
            threshold=2.0,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_2",
            metric="posmax_negmax_ratio",
            operation="<",
            threshold=2.5,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="4_singlepeak",
            metric="motif_entropy",
            operation="<",
            threshold=0.35,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="5_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.85,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="6_noisypeaks",
            metric="weighted_base_entropy",
            operation=">",
            threshold=0.7,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="7_gcbias_1",
            metric="copair_entropy_score",
            operation=">",
            threshold=0.3,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="7_gcbias_2",
            metric="copair_composition",
            operation=">",
            threshold=0.4,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="8_broadsingle_1",
            metric="weighted_position_entropy",
            operation=">",
            threshold=0.9,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="8_broadsingle_2",
            metric="posbase_entropy_score",
            operation=">",
            threshold=0.4,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_1",
            metric="dinuc_entropy_score",
            operation=">",
            threshold=0.4,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_2",
            metric="dinuc_composition",
            operation=">",
            threshold=0.75,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_3",
            metric="dinuc_score",
            operation=">",
            threshold=0.4,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
    )
    # Moderate filters (same as pipeline_tfatlas.py)
    moderate_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=MetadataCols.num_seqlets_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=False,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="2_truncated",
            metric="truncated",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_inverted",
            metric="posneg_inverted",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_1",
            metric="possum_negsum_ratio",
            operation="<",
            threshold=2.0,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_2",
            metric="posmax_negmax_ratio",
            operation="<",
            threshold=2.5,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="5_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.9,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="6_noisypeaks",
            metric="weighted_base_entropy",
            operation=">",
            threshold=0.75,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="7_gcbias_1",
            metric="copair_entropy_score",
            operation=">",
            threshold=0.4,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="7_gcbias_2",
            metric="copair_composition",
            operation=">",
            threshold=0.45,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="8_broadsingle_1",
            metric="weighted_position_entropy",
            operation=">",
            threshold=0.925,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="8_broadsingle_2",
            metric="posbase_entropy_score",
            operation=">",
            threshold=0.45,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_1",
            metric="dinuc_entropy_score",
            operation=">",
            threshold=0.5,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_2",
            metric="dinuc_composition",
            operation=">",
            threshold=0.775,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_3",
            metric="dinuc_score",
            operation=">",
            threshold=0.425,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
    )
    # Hard filters (same as pipeline_tfatlas.py)
    hard_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=MetadataCols.num_seqlets_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=False,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="2_truncated",
            metric="truncated",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_inverted",
            metric="posneg_inverted",
            operation="==",
            threshold=True,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_1",
            metric="possum_negsum_ratio",
            operation="<",
            threshold=1.5,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="3_posneg_ratio_2",
            metric="posmax_negmax_ratio",
            operation="<",
            threshold=2.0,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="2_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.925,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="3_noisypeaks",
            metric="weighted_base_entropy",
            operation=">",
            threshold=0.8,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="low-complexity",
        ),
        _FilterArgs(
            name="7_gcbias_1",
            metric="copair_entropy_score",
            operation=">",
            threshold=0.45,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="7_gcbias_2",
            metric="copair_composition",
            operation=">",
            threshold=0.6,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="high-GC-AT",
        ),
        _FilterArgs(
            name="8_broadsingle_1",
            metric="weighted_position_entropy",
            operation=">",
            threshold=0.95,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="8_broadsingle_2",
            metric="posbase_entropy_score",
            operation=">",
            threshold=0.55,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_1",
            metric="dinuc_entropy_score",
            operation=">",
            threshold=0.55,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_2",
            metric="dinuc_composition",
            operation=">",
            threshold=0.8,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
        _FilterArgs(
            name="9_dinucrepeat_3",
            metric="dinuc_score",
            operation=">",
            threshold=0.45,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="tandem-repeats",
        ),
    )
    # Soft override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    soft_override_filters: tuple = (
        _FilterArgs(
            name="base_match",
            metric=f"{ReferenceMatchCols.ref_col}_score0",
            operation=">",
            threshold=MotifMatchArgs.min_single_score,
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
    ) + tuple(
        _FilterArgs(
            name="composite_match",
            metric=f"{ReferenceMatchCols.ref_col}_score{iter}",
            operation=">",
            threshold=MotifMatchArgs.min_composite_score,
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        )
        for iter in range(1, MotifMatchArgs.max_submotifs)
    )
    # Moderate override: direct motifs are only removed by hard filters (exact match: 'INDIRECT' contains 'DIRECT')
    moderate_override_filters: tuple = (
        _FilterArgs(
            name="direct_match",
            metric=DirectTypeArgs.direct_type_col,
            operation="isin",
            threshold=DirectTypeArgs.direct_types,
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
    )

    def to_dict(self):
        # Return filters as a dictionary
        return {
            "motif_metrics": list(self.motif_metrics),
            "soft_filters": [f.to_dict() for f in self.soft_filters],
            "moderate_filters": [f.to_dict() for f in self.moderate_filters],
            "hard_filters": [f.to_dict() for f in self.hard_filters],
            "soft_override_filters": [f.to_dict() for f in self.soft_override_filters],
            "moderate_override_filters": [f.to_dict() for f in self.moderate_override_filters],
        }


## CLUSTERING ---------------------------------------------------------------------------------------------------
@dataclass
class ClusterArgs:
    # Algorithm (same as pipeline_tfatlas.py)
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_algorithm_kmeans: str = "k_centroids"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_force_density: float = 1.0
    cluster_sim_assignment_threshold_kmeans: float = 0.1
    cluster_max_iter_main: int = 0
    cluster_max_iter_force: int = 50
    n_iterations: int = -1  # Run until convergence
    seeds: List[int] = field(default_factory=lambda: [100, 200])

    # Boundaries
    cluster_within: str = MetadataCols.tf_col
    metacluster_within: str = MetadataCols.composite_col

    # Weights: CWMs by seqlets, hit PPMs by hits
    average_weight_col: str = MetadataCols.num_seqlets_col
    ppm_weight_col: str = MetadataCols.num_hits_col

    # Columns
    cluster_col: str = 'cluster'
    metacluster_col: str = 'metacluster'
    source_cluster_col: str = "source_cluster"

    # Aggregation of per-motif metadata into cluster-level metadata (columns missing from the input are skipped)
    aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (MetadataCols.name_col, "count", MetadataCols.num_constituents_col),
        (MetadataCols.name_col, "concat", MetadataCols.name_constituents_col),
        (MetadataCols.assay_col, "concat", MetadataCols.assay_col),
        (MetadataCols.head_col, "concat", MetadataCols.head_col),
        (MetadataCols.id_col, "concat", MetadataCols.id_col),
        (MetadataCols.model_col, "concat", MetadataCols.model_col),
        (MetadataCols.num_seqlets_col, "sum", MetadataCols.num_seqlets_col),
        (MetadataCols.num_hits_col, "sum", MetadataCols.num_hits_col),
        (MetadataCols.total_contrib_score_col, "sum", MetadataCols.total_contrib_score_col),
        (MetadataCols.group_total_contrib_score_col, "sum", MetadataCols.group_total_contrib_score_col),
        (DirectTypeArgs.direct_type_col, "concat", DirectTypeArgs.direct_type_col),
        (DirectTypeArgs.tf_direct_col, "concat", DirectTypeArgs.tf_direct_col),
        (DirectTypeArgs.family_direct_col, "concat", DirectTypeArgs.family_direct_col),
        (LabelCols.label_col, "concat", LabelCols.label_constit_1_col),
        (LabelCols.label_constit_all_col, "concat", LabelCols.label_constit_all_col),
        (LabelCols.direct_label_col, "concat", LabelCols.direct_label_constit_1_col),
        (LabelCols.direct_label_constit_all_col, "concat", LabelCols.direct_label_constit_all_col),
        (LabelCols.label_core_col, "concat", LabelCols.label_core_constit_1_col),
        (LabelCols.label_core_constit_all_col, "concat", LabelCols.label_core_constit_all_col),
        (LabelCols.direct_label_core_col, "concat", LabelCols.direct_label_core_constit_1_col),
        (LabelCols.direct_label_core_constit_all_col, "concat", LabelCols.direct_label_core_constit_all_col),
        (LabelCols.label_core_readable_col, "concat", LabelCols.label_core_readable_constit_1_col),
        (LabelCols.label_core_readable_constit_all_col, "concat", LabelCols.label_core_readable_constit_all_col),
        (LabelCols.direct_label_core_readable_col, "concat", LabelCols.direct_label_core_readable_constit_1_col),
        (LabelCols.direct_label_core_readable_constit_all_col, "concat", LabelCols.direct_label_core_readable_constit_all_col),
        (LabelCols.label_flank_col, "concat", LabelCols.label_flank_constit_1_col),
        (LabelCols.label_flank_constit_all_col, "concat", LabelCols.label_flank_constit_all_col),
        (ReferenceMatchCols.ref_name_col, "concat", ReferenceMatchCols.ref_name_constit_1_col),
        (ReferenceMatchCols.ref_name_constit_all_col, "concat", ReferenceMatchCols.ref_name_constit_all_col),
        (f"{ReferenceMatchCols.ref_invitro_col}_name0", "concat", ReferenceMatchCols.ref_invitro_name_constit_1_col),
        (ReferenceMatchCols.ref_invitro_name_constit_all_col, "concat", ReferenceMatchCols.ref_invitro_name_constit_all_col),
        (MetadataCols.tf_col, "concat", MetadataCols.tf_col),
        (TFAnnotationCols.family_col, "concat", TFAnnotationCols.family_col),
        (MetadataCols.biosample_col, "concat", MetadataCols.biosample_col),
    ])

    # Weighted by average_weight_col; NaN-safe (see aggregate_weighted_avg)
    weighted_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (MetadataCols.composite_col, "average", MetadataCols.composite_col),
        (MetadataCols.avgdist_summit_col, "average", MetadataCols.avgdist_summit_col),
        (MetadataCols.avgcontrib_score_col, "average", MetadataCols.avgcontrib_score_col),
        (MetadataCols.num_subunit_col, "average", MetadataCols.num_subunit_col),
        (MetadataCols.num_subunit_direct_col, "average", MetadataCols.num_subunit_direct_col),
    ])

    # Merge the previous level's constituents into the running set of all constituents
    merge_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (ReferenceMatchCols.ref_name_constit_1_col, "merge", ReferenceMatchCols.ref_name_constit_all_col),
        (ReferenceMatchCols.ref_invitro_name_constit_1_col, "merge", ReferenceMatchCols.ref_invitro_name_constit_all_col),
        (LabelCols.label_constit_1_col, "merge", LabelCols.label_constit_all_col),
        (LabelCols.label_core_constit_1_col, "merge", LabelCols.label_core_constit_all_col),
        (LabelCols.label_core_readable_constit_1_col, "merge", LabelCols.label_core_readable_constit_all_col),
        (LabelCols.label_flank_constit_1_col, "merge", LabelCols.label_flank_constit_all_col),
        (LabelCols.direct_label_constit_1_col, "merge", LabelCols.direct_label_constit_all_col),
        (LabelCols.direct_label_core_constit_1_col, "merge", LabelCols.direct_label_core_constit_all_col),
        (LabelCols.direct_label_core_readable_constit_1_col, "merge", LabelCols.direct_label_core_readable_constit_all_col),
    ])


## VISUALIZATION / HTML -----------------------------------------------------------------------------------------
@dataclass
class VisualizeArgs:
    html_logo_trim: float = 0
    motif_string_col: str = 'motif_string'
    motif_string_importance: float = 1 / 50
    motif_string_specificity: float = 0.6

    cluster_name_col: str = 'cluster_name'
    metacluster_name_col: str = 'metacluster_name'
    sort_col: str = 'sort'

    # Metacluster labels: families above this fraction of total contribution, relaxed stepwise
    max_major_family_frac: float = 0.3
    major_family_increment: float = -0.05
    max_major_family_count: int = 4

    # HTML summary table columns (columns missing from an object are skipped)
    html_columns: List[str] = field(default_factory=lambda: [
        MetadataCols.unique_id_col, MetadataCols.name_col, LabelCols.label_col, LabelCols.direct_label_col,
        LabelCols.label_conf_col, DirectTypeArgs.direct_type_col, MetadataCols.composite_col,
        MetadataCols.num_subunit_col, MetadataCols.num_constituents_col, MetadataCols.num_seqlets_col,
        MetadataCols.num_hits_col, VisualizeArgs.sort_col,
    ] + [
        f"{col}_{field_name}0"
        for col in [
            ReferenceMatchCols.ref_invitro_col,
            ReferenceMatchCols.ref_jaspar_core_col,
            ReferenceMatchCols.ref_jaspar_unvalid_col,
            ReferenceMatchCols.ref_invitro_other_col,
            ReferenceMatchCols.ref_jaspar_col,
        ]
        for field_name in ["logo", "name", "score"]
    ] + [
        f"{ReferenceMatchCols.ref_col}_{field_name}{j}"
        for j in range(MotifMatchArgs.max_submotifs - 1)
        for field_name in ["logo", "name", "score"]
    ] + [
        LabelCols.label_core_col, LabelCols.direct_label_core_col, LabelCols.label_flank_col,
        VisualizeArgs.cluster_name_col, VisualizeArgs.metacluster_name_col,
        VisualizeArgs.motif_string_col,
        MetadataCols.avgdist_summit_col, MetadataCols.avgcontrib_score_col,
        DirectTypeArgs.tf_direct_col, DirectTypeArgs.family_direct_col,
        MetadataCols.tf_col, TFAnnotationCols.family_col, MetadataCols.biosample_col,
    ])


## MOTIFCOMPENDIUM ---------------------------------------------------------------------------------------------------
@dataclass
class MotifCompendiumArgs:
    # Minimum size (bytes) for a file to be considered a non-empty HDF5 file
    min_h5_size_bytes: int = 888

    # Metadata columns that a user-supplied --metadata file may not define (reserved for internal use)
    reserved_metadata_columns: List[str] = field(default_factory=lambda: [
        "name", "num_seqlets", "posneg", "avg_dist_from_summit",
    ])


## JASPAR PIPELINE -----------------------------------------------------------------------------------------------
@dataclass
class JasparPipelineArgs:
    # Columns written by the previous version of this pipeline into motif_init.mc, dropped on load
    legacy_cols: List[str] = field(default_factory=lambda: [
        'avg_contrib',
    ])

    # Columns dropped before export
    drop_cols: List[str] = field(default_factory=lambda: [
        'ic-scaled',
        'dinuc_entropy_ratio',
        'posbase_entropy_ratio',
        'copair_entropy_ratio',
        'motif_entropy',
        'posneg_inverted',
        'truncated',
        'weighted_base_entropy',
    ])
