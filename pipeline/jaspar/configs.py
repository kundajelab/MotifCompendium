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
    name_new_col: str = 'name_new'
    name_tfmodisco_col: str = 'name_tfmodisco'
    name_unannotated: str = '_UNANNOT_'
    counter_col: str = 'counter'

    assay_col: str = 'assay'
    target_col: str = 'target'
    family_col: str = 'family'
    id_col: str = 'ID'
    head_col: str = 'head'
    model_col: str = 'model'
    accession_col: str = 'accession_id'
    biosample_col: str = 'biosample'

    posneg_col: str = 'posneg'
    num_seqlets_col: str = 'num_seqlets'
    num_hits_col: str = 'num_hits'
    avg_dist_from_summit_col: str = 'avg_dist_from_summit'
    avg_contrib_col: str = 'avg_contrib'

    hits_report_tsv_col: str = 'hits_report_tsv_path'
    hits_ppm_npy_col: str = 'hits_ppm_npy_path'

    # Clustering
    cluster_col: str = 'cluster'
    metacluster_col: str = 'metacluster'
    cluster_name_col: str = 'cluster_name'
    metacluster_name_col: str = 'metacluster_name'
    source_cluster_col: str = 'source_cluster'
    index_col: str = 'index'


## REFERENCE MATCH --------------------------------------------------------------------------------------------------
@dataclass
class ReferenceMatchCols:
    ref_col: str = "reference"

    # JASPAR-2026 pipeline: save_col_prefix identifiers used when matching against reference compendia
    ref_jaspar_select_col: str = "JASPAR"  # selects ENCODE motifs that have a matching JASPAR model
    ref_invitro_col: str = "invitro"       # identifies motifs confirmed by the in vitro + JASPAR reference
    ref_jaspar_col: str = "JASPAR-2026-HUMAN"
    ref_jaspar_core_col: str = "JASPAR-2026-HUMAN-CORE"
    ref_jaspar_unvalid_col: str = "JASPAR-2026-HUMAN-UNVALID"
    ref_invitro_other_col: str = "INVITRO-OTHER"


## MOTIF MATCH --------------------------------------------------------------------------------------------------
@dataclass
class MotifMatchArgs:
    min_single_score_jasp: float = 0.88
    min_single_score: float = 0.8
    min_composite_score: float = 0.7
    max_submotifs_jasp: int = 1
    max_submotifs: int = 3


## FILTERING ---------------------------------------------------------------------------------------------------
@dataclass
class FilterArgs:
    filter_flag_col: str = 'flag_remove'
    filter_flag_type_col: str = 'flag_type'
    filter_trim_importance: float = 0
    filter_trim_length: Union[int, None] = None


## INTERNAL: Motif filter thresholds -----------------------------------------------------------------------------
@dataclass
class _FilterArgs:
    # INTERNAL: Standard filter arguments
    name: str
    metric: str
    operation: str
    threshold: Union[float, bool]
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
    # Soft filters: Initial
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
            threshold=0.6,
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
            flag_type="high-GC/AT",
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
            flag_type="high-GC/AT",
        ),
        _FilterArgs(
            name="8_broadsingle_1",
            metric="weighted_position_entropy",
            operation=">",
            threshold=0.85,
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
    # Hard filters: Same as TF moderate filters
    hard_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=MetadataCols.num_seqlets_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="1_minhits",
            metric=MetadataCols.num_hits_col,
            operation="==",
            threshold=0,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="jasp_match",
            metric=f"{ReferenceMatchCols.ref_jaspar_select_col}_score0",
            operation="<",
            threshold=MotifMatchArgs.min_single_score_jasp,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
            flag_type="no-jaspar"
        ),
    )
    # Override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    override_filters: tuple = (
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
    )
    # ) + tuple(
    #     _FilterArgs(
    #         name="composite_match",
    #         metric=f"{ReferenceMatchCols.ref_col}_score{iter}",
    #         operation=">",
    #         threshold=MotifMatchArgs.min_composite_score,
    #         invert=True,
    #         override=True,
    #         apply_motif=True,
    #         apply_cluster=True,
    #     )
    #     for iter in range(1, MotifMatchArgs.max_submotifs)
    # )

    def to_dict(self):
        # Return filters as a dictionary
        return {
            "motif_metrics": list(self.motif_metrics),
            "soft_filters": [f.to_dict() for f in self.soft_filters],
            "hard_filters": [f.to_dict() for f in self.hard_filters],
            "override_filters": [f.to_dict() for f in self.override_filters],
        }


## CLUSTERING ---------------------------------------------------------------------------------------------------
@dataclass
class ClusterArgs:
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_within: str = MetadataCols.target_col
    cluster_max_iter: int = 50


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
    # In vitro candidate pool: a motif is a candidate if it clears BOTH thresholds
    # (strict greater-than, matching the reference notebook's `num_seqlets > min_seqlet`).
    # Deliberately independent of the `jasp_match` hard filter: the JASPAR match selects
    # which experiments to keep, not which individual motifs are in vitro candidates.
    min_seqlets: int = 100
    min_hits: int = 0

    # Cluster-averaging
    ppm_weight_col: str = MetadataCols.num_hits_col
    aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (MetadataCols.name_col, "count", "num_motifs"),
        (MetadataCols.num_seqlets_col, "sum", MetadataCols.num_seqlets_col),
        (MetadataCols.num_hits_col, "sum", MetadataCols.num_hits_col),
        (MetadataCols.model_col, "concat", MetadataCols.model_col),
        (MetadataCols.id_col, "concat", MetadataCols.id_col),
        (MetadataCols.avg_dist_from_summit_col, "average", MetadataCols.avg_dist_from_summit_col),
        (MetadataCols.avg_contrib_col, "average", MetadataCols.avg_contrib_col),
        (MetadataCols.target_col, "concat", MetadataCols.target_col),
        (MetadataCols.family_col, "concat", MetadataCols.family_col),
        (MetadataCols.biosample_col, "concat", MetadataCols.biosample_col),
    ])

    # HTML summary tables: columns shared by every summary table, appended after the match columns
    html_metadata_columns: List[str] = field(default_factory=lambda: [
        MetadataCols.name_col, MetadataCols.num_seqlets_col, MetadataCols.num_hits_col,
        MetadataCols.avg_dist_from_summit_col, MetadataCols.avg_contrib_col,
        MetadataCols.target_col, MetadataCols.family_col, MetadataCols.biosample_col,
    ])

    drop_cols: List[str] = field(default_factory=lambda: [
        # MetadataCols.avg_contrib_col,
        # MetadataCols.avg_dist_from_summit_col,
        'ic-scaled',
        'dinuc_entropy_ratio',
        'posbase_entropy_ratio',
        'copair_entropy_ratio',
        'motif_entropy',
        'posneg_inverted',
        'truncated',
        'weighted_base_entropy',
    ] + [
        # item
        # for i in range(MotifMatchArgs.max_submotifs)
        # for item in (
        #     f'{ReferenceMatchCols.ref_col}_name{i}',
        #     f'{ReferenceMatchCols.ref_col}_score{i}',
        #     f'{ReferenceMatchCols.ref_col}_logo{i}',
        # )
    ])

    drop_images: List[str] = field(default_factory=lambda: [
        # item
        # for i in range(MotifMatchArgs.max_submotifs)
        # for item in (
        #     f'{ReferenceMatchCols.ref_col}_logo{i}',
        # )
    ])
