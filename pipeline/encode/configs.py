from dataclasses import dataclass, field, asdict, replace
from typing import List, Tuple, Union


## SHARED ---------------------------------------------------------------------------------------------------
## MOTIF FILTERS (thresholds applied during motif/cluster filtering) ------------------------------------------
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
## MOTIF MATCH --------------------------------------------------------------------------------------------------
@dataclass
class MotifMatchArgs:
    min_single_score: float = 0.8
    min_composite_score: float = 0.7
    max_submotifs: int = 3
    min_composite_subunit: float = 1.0
## MOTIFCOMPENDIUM ---------------------------------------------------------------------------------------------------
@dataclass
class MotifCompendiumArgs:
    # Minimum size (bytes) for a file to be considered a non-empty HDF5 file
    min_h5_size_bytes: int = 888

    # Metadata columns that a user-supplied --metadata file may not define (reserved for internal use)
    reserved_metadata_columns: List[str] = field(default_factory=lambda: [
        "name", "num_seqlets", "posneg", "avg_dist_from_summit",
    ])
## CELL TYPE ANNOTATIONS ---------------------------------------------------------------------------------------------
@dataclass
class CellTypeCols:
    celltype_col: str = 'celltype'
    organ_col: str = 'organ'
    slim_cell_col: str = 'cell_slims'
    slim_organ_col: str = 'organ_slims'
    slim_dev_col: str = 'developmental_slims'
    slim_system_col: str = 'system_slims'

## TF-MODISCO ---------------------------------------------------------------------------------------------------
# Per-atlas MoDISco region width (was previously the --modisco-region-width CLI arg; the
# confirmed per-atlas pipelines read it from configs instead).
@dataclass
class TF_TFMoDIScoCols:
    region_width: int = 400


@dataclass
class Chrom_TFMoDIScoCols:
    region_width: int = 500


## TF ATLAS CONFIGS ----------------------------------------------------------------------------
## METADATA -------------------------------------------------------------------------------------------------------
@dataclass
class TF_MetadataCols:
    # General naming / ID columns
    name_col: str = 'name'
    prename_col: str = 'prename'
    simple_name_col: str = 'simple_name'
    name_new_col: str = 'new_name'
    name_constituents_col: str = 'name_constituents'
    counter_col: str = 'counter'
    name_unknown: str = 'unvalidated'
    qc_col: str = 'qc_flag'

    atlas_type: str = 'TF'
    id_col: str = 'ID'
    tf_col: str = 'target'
    tf_encode_col: str = 'ENCODE_type'
    model_col: str = 'model'
    head_col: str = 'head'
    accession_col: str = 'bpnet_accession'
    biosample_col: str = 'biosample'
    assay_col: str = 'assay'
    znf_col: str = 'znf'

    unique_id_col: str = 'MCID'

    composite_col: str = 'composite'  # True, False
    num_subunit_col: str = 'num_subunit'
    num_subunit_direct_col: str = 'num_subunit_direct'

    num_seqlets_col: str = 'num_seqlets'
    num_constituents_col: str = 'num_constituents'
    posneg_col: str = 'posneg'
    avgcontrib_score_col: str = "avg_contrib_score"
    total_contrib_score_col: str = "total_contrib_score"
    group_total_contrib_score_col: str = "group_total_contrib_score"
    total_contrib_score_ratio_col: str = "total_contrib_score_ratio"
    avgdist_summit_col: str = "avg_dist_from_summit"
## TF ANNOTATION ----------------------------------------------------------------------------------------------------
@dataclass
class TF_TFAnnotationCols:
    family_col: str = 'family'

    # TF Class
    tfclass_family_name_col: str = 'TFClass_family_name'
    tfclass_family_id_col: str = 'TFClass_family_id'

    # The Human TF
    humantf_dbd_col: str = 'HumanTF_DBD'

    # Reference annotation
    ref_tf_col: str = "TF"
    ref_family_col: str = "family"
## REFERENCE MATCH --------------------------------------------------------------------------------------------------
@dataclass
class TF_ReferenceMatchCols:
    ref_col: str = "reference"
    ref_name_col: str = 'reference_name'
    ref_name_constit_1_col: str = "reference_name_constit_prev"
    ref_name_constit_all_col: str = "reference_name_constit_all"
    ref_readable_col: str = "reference_readable"
    best_ref_constit_col: str = "best_reference_constit_all"

    direct_ref_col: str = "direct_reference"
    direct_ref_name_col: str = "direct_reference_name"
    direct_ref_name_constit_1_col: str = "direct_reference_name_constit_prev"
    direct_ref_name_constit_all_col: str = "direct_reference_name_constit_all"
    direct_ref_readable_col: str = "direct_reference_readable"
    best_direct_ref_constit_col: str = "best_direct_reference_constit_all"

    znf_ref_name_col: str = "znf_reference_name"
    znf_ref_name_constit_1_col: str = "znf_reference_name_constit_prev"
    znf_ref_name_constit_all_col: str = "znf_reference_name_constit_all"

    nonznf_ref_readable_col: str = "nonznf_reference_readable"
    nonznf_ref_col: str = "nonznf_reference"
## LABELS ------------------------------------------------------------------------------------------------------------
@dataclass
class TF_LabelCols:
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

    best_label_ref_col: str = "best_label_reference"
## DIRECT TYPE (Alpha, Beta Non-ZNF) -----------------------------------------------------------------------------
@dataclass
class TF_DirectTypeArgs:
    direct_type_col: str = 'direct_type'  # ["A", "B-NZ", "B-Z", "C", "X"]
    tf_direct_col: str = 'target_direct'  # Target TF
    family_direct_col: str = 'family_direct'  # Target TF family
    direct_types: Tuple[str, ...] = ("A", "B-NZ", "B-Z", "C", "X")  # Non-canonical: None

    direct_alpha_key: str = 'direct_alpha'
    direct_alpha_mononer_key: str = 'direct_alpha_mononer'
    direct_alpha_multimer_key: str = 'direct_alpha_multimer'
    direct_beta_nzf_key: str = 'direct_beta_nzf'
    direct_beta_nzf_monomer_key: str = 'direct_beta_nzf_monomer'
    direct_beta_nzf_multimer_key: str = 'direct_beta_nzf_multimer'
    direct_beta_zf_key: str = 'direct_beta_zf'
    direct_gamma_key: str = 'direct_gamma'
## SMITH-WATERMAN (Direct Beta ZNF, Gamma) -----------------------------------------------------------------------
@dataclass
class SmithWatermanArgs:
    msw_score_col: str = "smith-waterman_score"
    msw_name_col: str = "smith-waterman_name"
    msw_sim_col: str = "smith-waterman_similarity"
    msw_orient_col: str = "smith-waterman_orientation"
    msw_align_col: str = "smith-waterman_alignment"
    msw_action_col: str = "smith-waterman_action"
    msw_logo_col: str = "smith-waterman_logo"

    msw_sim_threshold: float = 0
    overlap_penalty_f1: float = -0.02
    overlap_penalty_f2: float = -0.1
    skip_penalty_f: float = -0.1
    skip_penalty_l1: float = -0.02
    skip_penalty_l2: float = -0.1

    min_msw_score: float = 0.6  # F1-optimized: 0.68
    min_topk_msw_score: float = 0.5  # Select top k
    topk: int = 1

    # Direct: Gamma
    gamma_col: str = TF_MetadataCols.avgcontrib_score_col
## FILTERING ---------------------------------------------------------------------------------------------------
@dataclass
class TF_FilterArgs:
    filter_flag_col: str = 'flag_remove'
    filter_flag_type_col: str = 'flag_type'
    filter_trim_importance: float = 0
    filter_trim_length: Union[int, None] = None
    filter_ic_scale: bool = False
    filter_flag_types: List[str] = field(default_factory=lambda: [
        'CpG', 'tandem-repeats', 'high-GC-AT', 'low-complexity', TF_MetadataCols.name_unknown,
    ])
@dataclass
class TF_MotifFilterArgs:
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
            metric=TF_MetadataCols.num_seqlets_col,
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
    # Moderate filters
    moderate_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=TF_MetadataCols.num_seqlets_col,
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
    # Hard filters
    hard_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=TF_MetadataCols.num_seqlets_col,
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
            metric=f"{TF_ReferenceMatchCols.ref_col}_score0",
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
            metric=f"{TF_ReferenceMatchCols.ref_col}_score{iter}",
            operation=">",
            threshold=MotifMatchArgs.min_composite_score,
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        )
        for iter in range(1, MotifMatchArgs.max_submotifs)
    )
    # Moderate override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    moderate_override_filters: tuple = (
        _FilterArgs(
            name=f"direct_match_{TF_DirectTypeArgs.direct_types[0]}",  # "A"
            metric=TF_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=TF_DirectTypeArgs.direct_types[0],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{TF_DirectTypeArgs.direct_types[1]}",  # "B-NZ"
            metric=TF_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=TF_DirectTypeArgs.direct_types[1],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{TF_DirectTypeArgs.direct_types[2]}",  # "B-Z"
            metric=TF_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=TF_DirectTypeArgs.direct_types[2],
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
class TF_ClusterArgs:
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_algorithm_kmeans: str = "k_centroids"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_force_density: float = 1.0
    cluster_sim_assignment_threshold_kmeans: float = 0.1
    direct_cluster_within: str = TF_DirectTypeArgs.tf_direct_col
    nondirect_cluster_within: str = TF_MetadataCols.tf_col
    metacluster_within: str = TF_MetadataCols.composite_col
    # combine_counts-profile: merge counts + profile meta-clusters into atlasclusters
    atlascluster_within: str = TF_MetadataCols.composite_col
    cluster_max_iter_main: int = 0
    cluster_max_iter_force: int = 50
    average_weight_col: str = TF_MetadataCols.num_seqlets_col
    n_iterations: int = -1  # Run until convergence
    seeds: List[int] = field(default_factory=lambda: [100, 200])
    source_cluster_col: str = "source_cluster"

    # Aggregation of per-motif metadata into cluster-level metadata
    aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (TF_MetadataCols.name_col, "count", TF_MetadataCols.num_constituents_col),
        (TF_MetadataCols.name_col, "concat", TF_MetadataCols.name_constituents_col),
        (TF_MetadataCols.assay_col, "concat", TF_MetadataCols.assay_col),
        (TF_MetadataCols.head_col, "concat", TF_MetadataCols.head_col),
        (TF_MetadataCols.id_col, "concat", TF_MetadataCols.id_col),
        (TF_MetadataCols.model_col, "concat", TF_MetadataCols.model_col),
        (TF_MetadataCols.num_seqlets_col, "sum", TF_MetadataCols.num_seqlets_col),
        (TF_MetadataCols.total_contrib_score_col, "sum", TF_MetadataCols.total_contrib_score_col),
        (TF_MetadataCols.group_total_contrib_score_col, "sum", TF_MetadataCols.group_total_contrib_score_col),
        (TF_DirectTypeArgs.direct_type_col, "concat", TF_DirectTypeArgs.direct_type_col),
        (TF_DirectTypeArgs.tf_direct_col, "concat", TF_DirectTypeArgs.tf_direct_col),
        (TF_DirectTypeArgs.family_direct_col, "concat", TF_DirectTypeArgs.family_direct_col),
        (TF_LabelCols.label_col, "concat", TF_LabelCols.label_constit_1_col),  # [Dimer-style notation] Combined
        (TF_LabelCols.label_constit_1_col, "concat", TF_LabelCols.label_constit_all_col),
        (TF_LabelCols.direct_label_col, "concat", TF_LabelCols.direct_label_constit_1_col),  # [Dimer-style notation] Combined (Direct)
        (TF_LabelCols.direct_label_constit_1_col, "concat", TF_LabelCols.direct_label_constit_all_col),
        (TF_LabelCols.label_core_col, "concat", TF_LabelCols.label_core_constit_1_col),  # [Dimer-style notation] Core TF
        (TF_LabelCols.label_core_constit_all_col, "concat", TF_LabelCols.label_core_constit_all_col),
        (TF_LabelCols.direct_label_core_col, "concat", TF_LabelCols.direct_label_core_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (TF_LabelCols.direct_label_core_constit_all_col, "concat", TF_LabelCols.direct_label_core_constit_all_col),
        (TF_LabelCols.label_core_readable_col, "concat", TF_LabelCols.label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF family
        (TF_LabelCols.label_core_readable_constit_all_col, "concat", TF_LabelCols.label_core_readable_constit_all_col),
        (TF_LabelCols.direct_label_core_readable_col, "concat", TF_LabelCols.direct_label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (TF_LabelCols.direct_label_core_readable_constit_all_col, "concat", TF_LabelCols.direct_label_core_readable_constit_all_col),
        (TF_LabelCols.label_flank_col, "concat", TF_LabelCols.label_flank_constit_1_col),  # [Dimer-style notation] Flank
        (TF_LabelCols.label_flank_constit_all_col, "concat", TF_LabelCols.label_flank_constit_all_col),
        (TF_ReferenceMatchCols.ref_name_col, "concat", TF_ReferenceMatchCols.ref_name_constit_1_col),  # All unique reference motifs
        (TF_ReferenceMatchCols.ref_name_constit_all_col, "concat", TF_ReferenceMatchCols.ref_name_constit_all_col),
        (TF_ReferenceMatchCols.direct_ref_name_col, "concat", TF_ReferenceMatchCols.direct_ref_name_constit_1_col),  # All unique direct reference motifs
        (TF_ReferenceMatchCols.direct_ref_name_constit_all_col, "concat", TF_ReferenceMatchCols.direct_ref_name_constit_all_col),
        (TF_ReferenceMatchCols.znf_ref_name_col, "concat", TF_ReferenceMatchCols.znf_ref_name_constit_1_col),  # All unique direct reference motifs: Beta ZNF
        (TF_ReferenceMatchCols.znf_ref_name_constit_all_col, "concat", TF_ReferenceMatchCols.znf_ref_name_constit_all_col),
        (TF_MetadataCols.tf_col, "concat", TF_MetadataCols.tf_col),
        (TF_TFAnnotationCols.family_col, "concat", TF_TFAnnotationCols.family_col),
        (TF_TFAnnotationCols.tfclass_family_name_col, "concat", TF_TFAnnotationCols.tfclass_family_name_col),
        (TF_TFAnnotationCols.tfclass_family_id_col, "concat", TF_TFAnnotationCols.tfclass_family_id_col),
        (TF_TFAnnotationCols.humantf_dbd_col, "concat", TF_TFAnnotationCols.humantf_dbd_col),
        (TF_MetadataCols.biosample_col, "concat", TF_MetadataCols.biosample_col),
    ] + [
        (f'{col}_name{submotif_i}', 'concat', f'{col}_name{submotif_i}_constit_prev')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [TF_ReferenceMatchCols.direct_ref_col, TF_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ] + [
        (f'{col}_name{submotif_i}_constit_all', 'concat', f'{col}_name{submotif_i}_constit_all')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [TF_ReferenceMatchCols.direct_ref_col, TF_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ])

    weighted_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (TF_MetadataCols.composite_col, "average", TF_MetadataCols.composite_col),
        (TF_MetadataCols.avgdist_summit_col, "average", TF_MetadataCols.avgdist_summit_col),
        (TF_MetadataCols.avgcontrib_score_col, "average", TF_MetadataCols.avgcontrib_score_col),
        (TF_MetadataCols.num_subunit_col, "average", TF_MetadataCols.num_subunit_col),
        (TF_MetadataCols.num_subunit_direct_col, "average", TF_MetadataCols.num_subunit_direct_col),
    ])

    merge_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (TF_ReferenceMatchCols.ref_name_constit_1_col, "merge", TF_ReferenceMatchCols.ref_name_constit_all_col),
        (TF_ReferenceMatchCols.direct_ref_name_constit_1_col, "merge", TF_ReferenceMatchCols.direct_ref_name_constit_all_col),
        (TF_ReferenceMatchCols.znf_ref_name_constit_1_col, "merge", TF_ReferenceMatchCols.znf_ref_name_constit_all_col),
        (TF_LabelCols.label_constit_1_col, "merge", TF_LabelCols.label_constit_all_col),
        (TF_LabelCols.label_core_constit_1_col, "merge", TF_LabelCols.label_core_constit_all_col),
        (TF_LabelCols.label_core_readable_constit_1_col, "merge", TF_LabelCols.label_core_readable_constit_all_col),
        (TF_LabelCols.label_flank_constit_1_col, "merge", TF_LabelCols.label_flank_constit_all_col),
        (TF_LabelCols.direct_label_constit_1_col, "merge", TF_LabelCols.label_constit_all_col),
        (TF_LabelCols.direct_label_core_constit_1_col, "merge", TF_LabelCols.direct_label_core_constit_all_col),
        (TF_LabelCols.direct_label_core_readable_constit_1_col, "merge", TF_LabelCols.direct_label_core_readable_constit_all_col),
    ])
## VISUALIZATION / HTML -----------------------------------------------------------------------------------------
@dataclass
class TF_VisualizeArgs:
    html_logo_trim: float = 1 / 50
    motif_string_col: str = 'motif_string'
    motif_string_importance: float = 1 / 50
    motif_string_specificity: float = 0.6

    cluster_name_col: str = 'cluster_name'
    metacluster_name_col: str = 'metacluster_name'
    atlascluster_name_col: str = 'atlascluster_name'
    sort_col: str = 'sort'

    max_major_family_frac: float = 0.3
    major_family_increment: float = -0.05
    max_major_family_count: int = 4

    # Removed-motif HTML table: number of motifs sampled for the summary table
    html_removed_sample_size: int = 1000

    # Motif index export columns
    motif_index_cols: List[str] = field(default_factory=lambda: [
        TF_MetadataCols.name_col,
        TF_MetadataCols.id_col,
        TF_MetadataCols.head_col,
        TF_MetadataCols.biosample_col,
        TF_MetadataCols.tf_col,
        TF_MetadataCols.tf_encode_col,
        TF_TFAnnotationCols.family_col,
        TF_MetadataCols.num_subunit_direct_col,
        TF_MetadataCols.num_subunit_col,
        TF_TFAnnotationCols.tfclass_family_name_col,
        TF_TFAnnotationCols.tfclass_family_id_col,
        TF_TFAnnotationCols.humantf_dbd_col,
        TF_DirectTypeArgs.direct_type_col,
        TF_DirectTypeArgs.tf_direct_col,
        TF_DirectTypeArgs.family_direct_col,
        TF_VisualizeArgs.cluster_name_col,
        TF_VisualizeArgs.metacluster_name_col,
    ])

    # Motif index export columns: combine_counts-profile (atlascluster level)
    motif_index_cols_atlascluster: List[str] = field(default_factory=lambda: [
        TF_MetadataCols.name_col,
        TF_MetadataCols.id_col,
        TF_MetadataCols.head_col,
        TF_MetadataCols.biosample_col,
        TF_MetadataCols.tf_col,
        TF_MetadataCols.tf_encode_col,
        TF_TFAnnotationCols.family_col,
        TF_DirectTypeArgs.direct_type_col,
        TF_DirectTypeArgs.tf_direct_col,
        TF_DirectTypeArgs.family_direct_col,
        TF_MetadataCols.num_subunit_col,
        TF_TFAnnotationCols.tfclass_family_name_col,
        TF_TFAnnotationCols.tfclass_family_id_col,
        TF_TFAnnotationCols.humantf_dbd_col,
        TF_VisualizeArgs.cluster_name_col,
        TF_VisualizeArgs.metacluster_name_col,
        TF_VisualizeArgs.atlascluster_name_col,
    ])

    # HTML summary table columns
    html_columns: List[str] = field(default_factory=lambda: [
        TF_MetadataCols.unique_id_col, TF_MetadataCols.name_col, TF_LabelCols.label_col, TF_LabelCols.direct_label_col,
        TF_LabelCols.label_conf_col, TF_DirectTypeArgs.direct_type_col, TF_MetadataCols.composite_col,
        TF_MetadataCols.num_subunit_direct_col, TF_MetadataCols.num_subunit_col, TF_MetadataCols.num_seqlets_col,
        TF_VisualizeArgs.sort_col,
    ] + [
        f"{col}_{field_name}{j}"
        for col in [TF_ReferenceMatchCols.direct_ref_col, TF_ReferenceMatchCols.ref_col, TF_ReferenceMatchCols.best_direct_ref_constit_col, TF_ReferenceMatchCols.best_ref_constit_col]
        for j in range(MotifMatchArgs.max_submotifs - 1)
        for field_name in ["logo", "name", "score"]
    ] + [
        f'{TF_LabelCols.best_label_ref_col}_logo0', f'{TF_LabelCols.best_label_ref_col}_name0', f'{TF_LabelCols.best_label_ref_col}_score0',
        TF_LabelCols.label_core_col, TF_LabelCols.direct_label_core_col, TF_LabelCols.label_flank_col,
        SmithWatermanArgs.msw_logo_col, SmithWatermanArgs.msw_name_col, SmithWatermanArgs.msw_score_col,
        TF_VisualizeArgs.motif_string_col,
        TF_MetadataCols.avgdist_summit_col, TF_MetadataCols.avgcontrib_score_col,
        TF_DirectTypeArgs.tf_direct_col, TF_DirectTypeArgs.family_direct_col,
        TF_MetadataCols.tf_col, TF_TFAnnotationCols.family_col, TF_TFAnnotationCols.tfclass_family_name_col,
        TF_TFAnnotationCols.tfclass_family_id_col, TF_TFAnnotationCols.humantf_dbd_col,
    ])

## Chrom ATLAS CONFIGS -------------------------------------------------------------------------
## METADATA -------------------------------------------------------------------------------------------------------
@dataclass
class Chrom_MetadataCols:
    # General naming / ID columns
    name_col: str = 'name'
    prename_col: str = 'prename'
    simple_name_col: str = 'simple_name'
    name_new_col: str = 'new_name'
    name_constituents_col: str = 'name_constituents'
    counter_col: str = 'counter'
    name_unknown: str = 'unvalidated'

    atlas_type: str = 'Chrom'
    id_col: str = 'ID'
    assay_col: str = 'assay'
    model_col: str = 'model'
    head_col: str = 'head'
    file_col: str = 'file'
    accession_col: str = 'chrombpnet_accession'
    biosample_col: str = 'biosample'

    unique_id_col: str = 'MCID'

    composite_col: str = 'composite'  # True, False
    num_subunit_col: str = 'num_subunit'

    num_seqlets_col: str = 'num_seqlets'
    num_constituents_col: str = 'num_constituents'
    posneg_col: str = 'posneg'
    avgcontrib_score_col: str = "avg_contrib_score"
    total_contrib_score_col: str = "total_contrib_score"
    group_total_contrib_score_col: str = "group_total_contrib_score"
    total_contrib_score_ratio_col: str = "total_contrib_score_ratio"
    avgdist_summit_col: str = "avg_dist_from_summit"
## REFERENCE MATCH --------------------------------------------------------------------------------------------------
@dataclass
class Chrom_ReferenceMatchCols:
    ref_tf_col: str = "TF"
    ref_family_col: str = "family"

    ref_col: str = "reference"
    ref_name_col: str = 'reference_name'
    ref_name_constit_1_col: str = "reference_name_constit_prev"
    ref_name_constit_all_col: str = "reference_name_constit_all"
    best_ref_constit_col: str = "best_reference_constit_all"

    ref_readable_col: str = "reference_readable"

    ref_znf_col: str = "reference_znf"
    ref_znf_name_col: str = "reference_znf_name"
    ref_znf_name_constit_1_col: str = "reference_znf_name_constit_prev"
    ref_znf_name_constit_all_col: str = "reference_znf_name_constit_all"

    ref_nonznf_readable_col: str = "reference_nonznf_readable"
    ref_nonznf_col: str = "reference_nonznf"
## LABELS ------------------------------------------------------------------------------------------------------------
@dataclass
class Chrom_LabelCols:
    label_col: str = 'label'  # Metacluster: Readable TF family label: {TF_family0}-{TF_family1}-...
    label_constit_1_col: str = 'label_constit_prev'
    label_constit_all_col: str = 'label_constit_all'

    label_core_col: str = 'label_core'  # Core: Main TF
    label_core_col_set: str = 'label_core_set'
    label_core_constit_1_col: str = 'label_core_constit_prev'
    label_core_constit_all_col: str = 'label_core_constit_all'

    label_core_readable_col: str = 'label_core_readable'  # Core: Main TF family
    label_core_readable_col_set: str = 'label_core_readable_col_set'
    label_core_readable_constit_1_col: str = 'label_core_readable_constit_prev'
    label_core_readable_constit_all_col: str = 'label_core_readable_constit_all'

    label_flank_col: str = 'label_flank'  # Composite: Subunit TF(s)
    label_flank_col_set: str = 'label_flank_set'
    label_flank_constit_1_col: str = 'label_flank_constit_prev'
    label_flank_constit_all_col: str = 'label_flank_constit_all'

    label_conf_col: str = 'label_confidence'
    label_confs: List[str] = field(default_factory=lambda: ['high', 'medium', 'low', 'flag'])

    best_label_ref_col: str = "best_label_reference"
## DIRECT TYPE --------------------------------------------------------------------------------------------------
@dataclass
class Chrom_DirectTypeArgs:
    direct_type_col: str = 'direct_type'  # ["A", "B-NZ", "B-Z", "C", "X"]
    direct_types: Tuple[str, ...] = ("A", "B-NZ", "B-Z", "C", "X")  # Non-canonical: None
## FILTERING ---------------------------------------------------------------------------------------------------
@dataclass
class Chrom_FilterArgs:
    filter_flag_col: str = 'flag_remove'
    filter_flag_type_col: str = 'flag_type'
    filter_trim_importance: float = 0
    filter_trim_length: Union[int, None] = None
    filter_ic_scale: bool = False
    filter_flag_types: List[str] = field(default_factory=lambda: [
        'CpG', 'tandem-repeats', 'high-GC-AT', 'low-complexity', Chrom_MetadataCols.name_unknown,
    ])
@dataclass
class Chrom_MotifFilterArgs:
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
            metric=Chrom_MetadataCols.num_seqlets_col,
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
            metric=Chrom_MetadataCols.num_seqlets_col,
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
    # Override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    override_filters: tuple = (
        _FilterArgs(
            name="base_match",
            metric=f"{Chrom_ReferenceMatchCols.ref_col}_score0",
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
class Chrom_ClusterArgs:
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_algorithm_kmeans: str = "k_centroids"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_force_density: float = 1.0
    cluster_sim_assignment_threshold_kmeans: float = 0.1
    cluster_within: str = CellTypeCols.celltype_col
    metacluster_within: str = Chrom_MetadataCols.composite_col
    # combine_counts-profile: merge counts + profile meta-clusters into atlasclusters
    atlascluster_within: str = Chrom_MetadataCols.composite_col
    cluster_max_iter_main: int = 0
    cluster_max_iter_force: int = 50
    average_weight_col: str = Chrom_MetadataCols.num_seqlets_col
    n_iterations: int = -1  # Run until convergence
    seeds: List[int] = field(default_factory=lambda: [100, 200])
    source_cluster_col: str = "source_cluster"

    # Aggregation of per-motif metadata into cluster-level metadata
    aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Chrom_MetadataCols.name_col, "count", Chrom_MetadataCols.num_constituents_col),
        (Chrom_MetadataCols.name_col, "concat", Chrom_MetadataCols.name_constituents_col),
        (Chrom_MetadataCols.assay_col, "concat", Chrom_MetadataCols.assay_col),
        (Chrom_MetadataCols.head_col, "concat", Chrom_MetadataCols.head_col),
        (Chrom_MetadataCols.id_col, "concat", Chrom_MetadataCols.id_col),
        (Chrom_MetadataCols.model_col, "concat", Chrom_MetadataCols.model_col),
        (Chrom_MetadataCols.num_seqlets_col, "sum", Chrom_MetadataCols.num_seqlets_col),
        (Chrom_MetadataCols.total_contrib_score_col, "sum", Chrom_MetadataCols.total_contrib_score_col),
        (Chrom_MetadataCols.group_total_contrib_score_col, "sum", Chrom_MetadataCols.group_total_contrib_score_col),
        (Chrom_DirectTypeArgs.direct_type_col, "concat", Chrom_DirectTypeArgs.direct_type_col),
        (Chrom_LabelCols.label_col, "concat", Chrom_LabelCols.label_constit_1_col),  # [Dimer-style notation] Combined
        (Chrom_LabelCols.label_constit_1_col, "concat", Chrom_LabelCols.label_constit_all_col),
        (Chrom_LabelCols.label_core_col, "concat", Chrom_LabelCols.label_core_constit_1_col),  # [Dimer-style notation] Core TF
        (Chrom_LabelCols.label_core_constit_all_col, "concat", Chrom_LabelCols.label_core_constit_all_col),
        (Chrom_LabelCols.label_core_readable_col, "concat", Chrom_LabelCols.label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF family
        (Chrom_LabelCols.label_core_readable_constit_all_col, "concat", Chrom_LabelCols.label_core_readable_constit_all_col),
        (Chrom_LabelCols.label_flank_col, "concat", Chrom_LabelCols.label_flank_constit_1_col),  # [Dimer-style notation] Flank
        (Chrom_LabelCols.label_flank_constit_all_col, "concat", Chrom_LabelCols.label_flank_constit_all_col),
        (Chrom_ReferenceMatchCols.ref_name_col, "concat", Chrom_ReferenceMatchCols.ref_name_constit_1_col),  # All unique reference motifs
        (Chrom_ReferenceMatchCols.ref_name_constit_all_col, "concat", Chrom_ReferenceMatchCols.ref_name_constit_all_col),
        (Chrom_ReferenceMatchCols.ref_znf_name_col, "concat", Chrom_ReferenceMatchCols.ref_znf_name_constit_1_col),  # All unique direct reference motifs: Beta ZNF
        (Chrom_ReferenceMatchCols.ref_znf_name_constit_all_col, "concat", Chrom_ReferenceMatchCols.ref_znf_name_constit_all_col),
        (CellTypeCols.celltype_col, "concat", CellTypeCols.celltype_col),
        (CellTypeCols.organ_col, "concat", CellTypeCols.organ_col),
        (CellTypeCols.slim_cell_col, "concat", CellTypeCols.slim_cell_col),
        (CellTypeCols.slim_organ_col, "concat", CellTypeCols.slim_organ_col),
        (CellTypeCols.slim_dev_col, "concat", CellTypeCols.slim_dev_col),
        (CellTypeCols.slim_system_col, "concat", CellTypeCols.slim_system_col),
        (Chrom_MetadataCols.biosample_col, "concat", Chrom_MetadataCols.biosample_col),
    ] + [
        (f'{col}_name{submotif_i}', 'concat', f'{col}_name{submotif_i}_constit_prev')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [Chrom_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ] + [
        (f'{col}_name{submotif_i}_constit_all', 'concat', f'{col}_name{submotif_i}_constit_all')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [Chrom_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ])

    weighted_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Chrom_MetadataCols.composite_col, "average", Chrom_MetadataCols.composite_col),
        (Chrom_MetadataCols.avgdist_summit_col, "average", Chrom_MetadataCols.avgdist_summit_col),
        (Chrom_MetadataCols.avgcontrib_score_col, "average", Chrom_MetadataCols.avgcontrib_score_col),
        (Chrom_MetadataCols.num_subunit_col, "average", Chrom_MetadataCols.num_subunit_col),
    ])

    merge_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Chrom_ReferenceMatchCols.ref_name_constit_1_col, "merge", Chrom_ReferenceMatchCols.ref_name_constit_all_col),
        (Chrom_LabelCols.label_constit_1_col, "merge", Chrom_LabelCols.label_constit_all_col),
        (Chrom_LabelCols.label_core_constit_1_col, "merge", Chrom_LabelCols.label_core_constit_all_col),
        (Chrom_LabelCols.label_core_readable_constit_1_col, "merge", Chrom_LabelCols.label_core_readable_constit_all_col),
        (Chrom_LabelCols.label_flank_constit_1_col, "merge", Chrom_LabelCols.label_flank_constit_all_col),
    ])
## VISUALIZATION / HTML -----------------------------------------------------------------------------------------
@dataclass
class Chrom_VisualizeArgs:
    html_logo_trim: float = 0
    motif_string_col: str = 'motif_string'
    motif_string_importance: float = 1 / 30
    motif_string_specificity: float = 0.6

    cluster_name_col: str = 'cluster_name'
    metacluster_name_col: str = 'metacluster_name'
    atlascluster_name_col: str = 'atlascluster_name'
    metacluster_name: str = 'Chromatin'
    sort_col: str = 'sort'

    max_major_family_frac: float = 0.3
    major_family_increment: float = -0.05
    max_major_family_count: int = 4

    # Removed-motif HTML table: number of motifs sampled for the summary table
    html_removed_sample_size: int = 1000

    # Motif index export columns
    motif_index_cols: List[str] = field(default_factory=lambda: [
        Chrom_MetadataCols.name_col,
        Chrom_MetadataCols.id_col,
        Chrom_MetadataCols.model_col,
        Chrom_MetadataCols.head_col,
        Chrom_MetadataCols.file_col,
        Chrom_MetadataCols.biosample_col,
        Chrom_MetadataCols.num_subunit_col,
        Chrom_VisualizeArgs.cluster_name_col,
        Chrom_VisualizeArgs.metacluster_name_col,
        CellTypeCols.celltype_col,
        CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col,
        CellTypeCols.slim_organ_col,
        CellTypeCols.slim_dev_col,
        CellTypeCols.slim_system_col,
    ])

    # Motif index export columns: combine_counts-profile (atlascluster level)
    motif_index_cols_atlascluster: List[str] = field(default_factory=lambda: [
        Chrom_MetadataCols.name_col,
        Chrom_MetadataCols.id_col,
        Chrom_MetadataCols.model_col,
        Chrom_MetadataCols.head_col,
        Chrom_MetadataCols.file_col,
        CellTypeCols.celltype_col,
        CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col,
        CellTypeCols.slim_organ_col,
        CellTypeCols.slim_dev_col,
        CellTypeCols.slim_system_col,
        Chrom_MetadataCols.biosample_col,
        Chrom_VisualizeArgs.cluster_name_col,
        Chrom_VisualizeArgs.metacluster_name_col,
        Chrom_VisualizeArgs.atlascluster_name_col,
    ])

    # HTML summary table columns
    html_columns: List[str] = field(default_factory=lambda: [
        Chrom_MetadataCols.unique_id_col, Chrom_MetadataCols.name_col, Chrom_LabelCols.label_col, Chrom_LabelCols.label_conf_col,
        Chrom_DirectTypeArgs.direct_type_col, Chrom_MetadataCols.composite_col, Chrom_MetadataCols.num_subunit_col,
        Chrom_MetadataCols.num_seqlets_col, Chrom_VisualizeArgs.sort_col,
    ] + [
        f"{col}_{field_name}{j}"
        for col in [Chrom_ReferenceMatchCols.ref_col, Chrom_ReferenceMatchCols.best_ref_constit_col, Chrom_LabelCols.best_label_ref_col]
        for j in range(MotifMatchArgs.max_submotifs - 1)
        for field_name in ["logo", Chrom_MetadataCols.name_col, "score"]
    ] + [
        Chrom_LabelCols.label_core_col, Chrom_LabelCols.label_flank_col,
        Chrom_VisualizeArgs.motif_string_col,
        Chrom_MetadataCols.avgdist_summit_col, Chrom_MetadataCols.avgcontrib_score_col,
        CellTypeCols.celltype_col, CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col, CellTypeCols.slim_organ_col, CellTypeCols.slim_dev_col, CellTypeCols.slim_system_col,
    ])

## ENCODE ATLAS CONFIGS ------------------------------------------------------------------------
## METADATA -------------------------------------------------------------------------------------------------------
@dataclass
class Encode_MetadataCols:
    # General naming / ID columns
    name_col: str = 'name'
    prename_col: str = 'prename'
    simple_name_col: str = 'simple_name'
    name_new_col: str = 'new_name'
    name_constituents_col: str = 'name_constituents'
    counter_col: str = 'counter'
    name_unknown: str = 'unvalidated'
    qc_col: str = 'qc_flag'

    id_col: str = 'ID'
    tf_col: str = 'target'
    tf_encode_col: str = 'ENCODE_type'
    model_col: str = 'model'
    head_col: str = 'head'
    accession_col: str = 'bpnet_accession'
    biosample_col: str = 'biosample'
    assay_col: str = 'assay'
    znf_col: str = 'znf'

    unique_id_col: str = 'MCID'

    composite_col: str = 'composite'  # True, False
    num_subunit_col: str = 'num_subunit'
    num_subunit_direct_col: str = 'num_subunit_direct'

    num_seqlet_col: str = 'num_seqlets'
    num_constituents_col: str = 'num_constituents'
    posneg_col: str = 'posneg'
    avgcontrib_score_col: str = "avg_contrib_score"
    total_contrib_score_col: str = "total_contrib_score"
    group_total_contrib_score_col: str = "group_total_contrib_score"
    total_contrib_score_ratio_col: str = "total_contrib_score_ratio"
    avgdist_summit_col: str = "avg_dist_from_summit"

    # Curation notes
    notes_col: str = 'notes'
    checked_col: str = 'checked'
    finemo_col: str = 'finemo'
## TF ANNOTATION ----------------------------------------------------------------------------------------------------
@dataclass
class Encode_TFAnnotationCols:
    family_col: str = 'family'

    # TF Class
    tfclass_family_name_col: str = 'TFClass_family_name'
    tfclass_family_id_col: str = 'TFClass_family_id'

    # The Human TF
    humantf_dbd_col: str = 'HumanTF_DBD'
## REFERENCE MATCH --------------------------------------------------------------------------------------------------
@dataclass
class Encode_ReferenceMatchCols:
    ref_tf_col: str = "TF"
    ref_family_col: str = "family"

    ref_col: str = "reference"
    ref_name_col: str = 'reference_name'
    ref_name_constit_1_col: str = "reference_name_constit_prev"
    ref_name_constit_all_col: str = "reference_name_constit_all"
    best_ref_constit_col: str = "best_reference_constit_all"

    ref_readable_col: str = "reference_readable"

    direct_ref_col: str = "direct_reference"
    direct_ref_name_col: str = "direct_reference_name"
    direct_ref_name_constit_1_col: str = "direct_reference_name_constit_prev"
    direct_ref_name_constit_all_col: str = "direct_reference_name_constit_all"
    best_direct_ref_constit_col: str = "best_direct_reference_constit_all"

    direct_ref_readable_col: str = "direct_reference_readable"

    znf_ref_name_col: str = "znf_reference_name"
    znf_ref_name_constit_1_col: str = "znf_reference_name_constit_prev"
    znf_ref_name_constit_all_col: str = "znf_reference_name_constit_all"

    nonznf_ref_readable_col: str = "nonznf_reference_readable"
    nonznf_ref_col: str = "nonznf_reference"
## LABELS ------------------------------------------------------------------------------------------------------------
@dataclass
class Encode_LabelCols:
    label_col: str = 'label'  # Metacluster: Readable TF family label: {TF_family0}-{TF_family1}-...
    label_readable_col: str = 'label_readable'
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
    label_confs: Tuple[str, ...] = ('high', 'medium', 'low', 'flag')
    label_confs_simple: Tuple[str, ...] = ('high', 'med', 'low', '')

    best_label_ref_col: str = "best_label_reference"
## DIRECT TYPE (Alpha, Beta Non-ZNF) -----------------------------------------------------------------------------
@dataclass
class Encode_DirectTypeArgs:
    direct_type_col: str = 'direct_type'  # ["A", "B-NZ", "B-Z", "C", "X"]
    tf_direct_col: str = 'target_direct'  # Target TF
    family_direct_col: str = 'family_direct'  # Target TF family
    direct_types: Tuple[str, ...] = ("A", "B-NZ", "B-Z", "C", "X")

    category_type_col: str = 'type'
    category_types: Tuple[str, ...] = ("direct", "indirect", "validated", "unvalidated", "non-canonical")
## FILTERING ---------------------------------------------------------------------------------------------------
@dataclass
class Encode_FilterArgs:
    filter_flag_col: str = 'flag_remove'
    filter_flag_type_col: str = 'flag_type'
    filter_trim_importance: float = 0
    filter_trim_length: Union[int, None] = None
    filter_ic_scale: bool = False
    filter_flag_types: List[str] = field(default_factory=lambda: [
        'CpG', 'tandem-repeats', 'high-GC-AT', 'low-complexity', Encode_MetadataCols.name_unknown,
    ])
    qc_min_score_removed: float = 0.99
    qc_min_score_filtered: float = 0.99
@dataclass
class Encode_MotifFilterArgs:
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
            metric=Encode_MetadataCols.num_seqlet_col,
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
    # Moderate filters
    moderate_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=Encode_MetadataCols.num_seqlet_col,
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
            threshold=0.5,
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
    # Hard filters
    hard_filters: tuple = (
        _FilterArgs(
            name="1_minseqlets",
            metric=Encode_MetadataCols.num_seqlet_col,
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
            metric=f"{Encode_ReferenceMatchCols.ref_col}_score0",
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
            metric=f"{Encode_ReferenceMatchCols.ref_col}_score{iter}",
            operation=">",
            threshold=MotifMatchArgs.min_composite_score,
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        )
        for iter in range(1, MotifMatchArgs.max_submotifs)
    )
    # Moderate override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    moderate_override_filters: tuple = (
        _FilterArgs(
            name=f"direct_match_{Encode_DirectTypeArgs.direct_types[0]}",  # "A"
            metric=Encode_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=Encode_DirectTypeArgs.direct_types[0],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{Encode_DirectTypeArgs.direct_types[1]}",  # "B-NZ"
            metric=Encode_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=Encode_DirectTypeArgs.direct_types[1],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{Encode_DirectTypeArgs.direct_types[2]}",  # "B-Z"
            metric=Encode_DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=Encode_DirectTypeArgs.direct_types[2],
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
## VISUALIZATION / HTML / HIERARCHY -------------------------------------------------------------------------------
@dataclass
class Encode_VisualizeArgs:
    html_logo_trim: float = 1 / 100
    motif_string_col: str = 'motif_string'
    motif_string_importance: float = 1 / 50
    motif_string_specificity: float = 0.55

    # Hierarchy: motif -> cluster -> metacluster -> atlascluster -> ENCODE cluster
    cluster_name_col: str = 'cluster_name'
    metacluster_name_col: str = 'metacluster_name'
    atlascluster_name_col: str = 'atlascluster_name'
    encodecluster_name_col: str = 'ENCODE_name'
    atlas_col: str = 'atlas'
    sort_col: str = 'sort'

    max_major_family_frac: float = 0.3
    major_family_increment: float = -0.05
    max_major_family_count: int = 4

    # HTML summary table columns
    html_columns: List[str] = field(default_factory=lambda: [
        Encode_MetadataCols.unique_id_col, Encode_MetadataCols.name_col, Encode_LabelCols.label_readable_col,
        Encode_LabelCols.label_conf_col, Encode_DirectTypeArgs.direct_type_col, Encode_MetadataCols.num_seqlet_col,
        Encode_VisualizeArgs.sort_col,
        f'{Encode_LabelCols.best_label_ref_col}_logo0', f'{Encode_LabelCols.best_label_ref_col}_name0', f'{Encode_LabelCols.best_label_ref_col}_score0',
    ] + [
        f"{col}_{field_name}{j}"
        for col in [Encode_ReferenceMatchCols.best_direct_ref_constit_col, Encode_ReferenceMatchCols.best_ref_constit_col,
                    Encode_ReferenceMatchCols.direct_ref_col, Encode_ReferenceMatchCols.ref_col]
        for j in range(MotifMatchArgs.max_submotifs - 1)
        for field_name in ["logo", "name", "score"]
    ] + [
        Encode_LabelCols.direct_label_col, Encode_LabelCols.label_col, Encode_LabelCols.direct_label_core_col,
        Encode_LabelCols.label_core_col, Encode_LabelCols.label_flank_col,
        Encode_VisualizeArgs.motif_string_col,
        Encode_DirectTypeArgs.tf_direct_col, Encode_MetadataCols.tf_col, Encode_DirectTypeArgs.family_direct_col, Encode_TFAnnotationCols.family_col,
        Encode_TFAnnotationCols.tfclass_family_name_col, Encode_TFAnnotationCols.tfclass_family_id_col, Encode_TFAnnotationCols.humantf_dbd_col,
        CellTypeCols.celltype_col, CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col, CellTypeCols.slim_organ_col, CellTypeCols.slim_dev_col, CellTypeCols.slim_system_col,
        Encode_MetadataCols.assay_col, Encode_MetadataCols.id_col, Encode_MetadataCols.head_col,
        Encode_MetadataCols.biosample_col,
    ])

    # Motif index export columns
    motif_index_cols: List[str] = field(default_factory=lambda: [
        Encode_MetadataCols.name_col,
        Encode_VisualizeArgs.atlas_col,
        Encode_MetadataCols.assay_col,
        Encode_MetadataCols.head_col,
        Encode_MetadataCols.posneg_col,
        Encode_MetadataCols.id_col,
        Encode_MetadataCols.biosample_col,
        Encode_MetadataCols.tf_col,
        Encode_MetadataCols.tf_encode_col,
        Encode_TFAnnotationCols.family_col,
        Encode_DirectTypeArgs.direct_type_col,
        Encode_DirectTypeArgs.tf_direct_col,
        Encode_DirectTypeArgs.family_direct_col,
        Encode_MetadataCols.num_subunit_col,
        Encode_TFAnnotationCols.tfclass_family_name_col,
        Encode_TFAnnotationCols.tfclass_family_id_col,
        Encode_TFAnnotationCols.humantf_dbd_col,
        CellTypeCols.celltype_col,
        CellTypeCols.organ_col,
        CellTypeCols.slim_cell_col,
        CellTypeCols.slim_organ_col,
        CellTypeCols.slim_dev_col,
        CellTypeCols.slim_system_col,
        Encode_DirectTypeArgs.category_type_col,
        Encode_LabelCols.label_conf_col,
        Encode_FilterArgs.filter_flag_col,
        Encode_FilterArgs.filter_flag_type_col,
        Encode_VisualizeArgs.cluster_name_col,
        Encode_VisualizeArgs.metacluster_name_col,
        Encode_VisualizeArgs.atlascluster_name_col,
        Encode_VisualizeArgs.encodecluster_name_col,
        Encode_MetadataCols.simple_name_col,
    ])

    # Motif category export columns
    motif_category_cols: List[str] = field(default_factory=lambda: [
        Encode_MetadataCols.name_col,
        Encode_MetadataCols.id_col,
        Encode_MetadataCols.head_col,
        Encode_MetadataCols.biosample_col,
        Encode_DirectTypeArgs.category_type_col,
    ])
## CLUSTERING ---------------------------------------------------------------------------------------------------
@dataclass
class Encode_ClusterArgs:
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_algorithm_kmeans: str = "k_centroids"
    cluster_algorithm_merge: str = "merge_1-1"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_force_density: float = 1.0
    cluster_sim_assignment_threshold_kmeans: float = 0.1
    cluster_max_iter_main: int = 0
    cluster_max_iter_force: int = 50
    average_weight_col: str = Encode_MetadataCols.num_seqlet_col
    n_iterations: int = -1  # Run until convergence
    seeds: List[int] = field(default_factory=lambda: [100, 200])
    source_cluster_col: str = 'source_cluster'

    # Aggregation of per-motif metadata into cluster-level metadata
    aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Encode_MetadataCols.name_col, "count", Encode_MetadataCols.num_constituents_col),
        (Encode_MetadataCols.name_col, "concat", Encode_MetadataCols.name_constituents_col),
        (Encode_MetadataCols.assay_col, "concat", Encode_MetadataCols.assay_col),
        (Encode_VisualizeArgs.atlas_col, "concat", Encode_VisualizeArgs.atlas_col),
        (Encode_MetadataCols.head_col, "concat", Encode_MetadataCols.head_col),
        (Encode_MetadataCols.id_col, "concat", Encode_MetadataCols.id_col),
        (Encode_MetadataCols.model_col, "concat", Encode_MetadataCols.model_col),
        (Encode_MetadataCols.num_seqlet_col, "sum", Encode_MetadataCols.num_seqlet_col),
        (Encode_MetadataCols.total_contrib_score_col, "sum", Encode_MetadataCols.total_contrib_score_col),
        (Encode_MetadataCols.group_total_contrib_score_col, "sum", Encode_MetadataCols.group_total_contrib_score_col),
        (Encode_DirectTypeArgs.direct_type_col, "concat", Encode_DirectTypeArgs.direct_type_col),
        (Encode_DirectTypeArgs.tf_direct_col, "concat", Encode_DirectTypeArgs.tf_direct_col),
        (Encode_DirectTypeArgs.family_direct_col, "concat", Encode_DirectTypeArgs.family_direct_col),
        (Encode_LabelCols.label_col, "concat", Encode_LabelCols.label_constit_1_col),  # [Dimer-style notation] Combined
        (Encode_LabelCols.label_constit_1_col, "concat", Encode_LabelCols.label_constit_all_col),
        (Encode_LabelCols.direct_label_col, "concat", Encode_LabelCols.direct_label_constit_1_col),  # [Dimer-style notation] Combined (Direct)
        (Encode_LabelCols.direct_label_constit_1_col, "concat", Encode_LabelCols.direct_label_constit_all_col),
        (Encode_LabelCols.label_core_col, "concat", Encode_LabelCols.label_core_constit_1_col),  # [Dimer-style notation] Core TF
        (Encode_LabelCols.label_core_constit_all_col, "concat", Encode_LabelCols.label_core_constit_all_col),
        (Encode_LabelCols.direct_label_core_col, "concat", Encode_LabelCols.direct_label_core_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (Encode_LabelCols.direct_label_core_constit_all_col, "concat", Encode_LabelCols.direct_label_core_constit_all_col),
        (Encode_LabelCols.label_core_readable_col, "concat", Encode_LabelCols.label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF family
        (Encode_LabelCols.label_core_readable_constit_all_col, "concat", Encode_LabelCols.label_core_readable_constit_all_col),
        (Encode_LabelCols.direct_label_core_readable_col, "concat", Encode_LabelCols.direct_label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (Encode_LabelCols.direct_label_core_readable_constit_all_col, "concat", Encode_LabelCols.direct_label_core_readable_constit_all_col),
        (Encode_LabelCols.label_flank_col, "concat", Encode_LabelCols.label_flank_constit_1_col),  # [Dimer-style notation] Flank
        (Encode_LabelCols.label_flank_constit_all_col, "concat", Encode_LabelCols.label_flank_constit_all_col),
        (Encode_ReferenceMatchCols.ref_name_col, "concat", Encode_ReferenceMatchCols.ref_name_constit_1_col),  # All unique reference motifs
        (Encode_ReferenceMatchCols.ref_name_constit_all_col, "concat", Encode_ReferenceMatchCols.ref_name_constit_all_col),
        (Encode_ReferenceMatchCols.direct_ref_name_col, "concat", Encode_ReferenceMatchCols.direct_ref_name_constit_1_col),  # All unique direct reference motifs
        (Encode_ReferenceMatchCols.direct_ref_name_constit_all_col, "concat", Encode_ReferenceMatchCols.direct_ref_name_constit_all_col),
        (Encode_ReferenceMatchCols.znf_ref_name_col, "concat", Encode_ReferenceMatchCols.znf_ref_name_constit_1_col),  # All unique direct reference motifs: Beta ZNF
        (Encode_ReferenceMatchCols.znf_ref_name_constit_all_col, "concat", Encode_ReferenceMatchCols.znf_ref_name_constit_all_col),
        (Encode_MetadataCols.tf_col, "concat", Encode_MetadataCols.tf_col),
        (Encode_TFAnnotationCols.family_col, "concat", Encode_TFAnnotationCols.family_col),
        (Encode_TFAnnotationCols.tfclass_family_name_col, "concat", Encode_TFAnnotationCols.tfclass_family_name_col),
        (Encode_TFAnnotationCols.tfclass_family_id_col, "concat", Encode_TFAnnotationCols.tfclass_family_id_col),
        (Encode_TFAnnotationCols.humantf_dbd_col, "concat", Encode_TFAnnotationCols.humantf_dbd_col),
        (CellTypeCols.celltype_col, "concat", CellTypeCols.celltype_col),
        (CellTypeCols.organ_col, "concat", CellTypeCols.organ_col),
        (CellTypeCols.slim_cell_col, "concat", CellTypeCols.slim_cell_col),
        (CellTypeCols.slim_organ_col, "concat", CellTypeCols.slim_organ_col),
        (CellTypeCols.slim_dev_col, "concat", CellTypeCols.slim_dev_col),
        (CellTypeCols.slim_system_col, "concat", CellTypeCols.slim_system_col),
        (Encode_MetadataCols.biosample_col, "concat", Encode_MetadataCols.biosample_col),
    ] + [
        (f'{col}_name{submotif_i}', 'concat', f'{col}_name{submotif_i}_constit_prev')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [Encode_ReferenceMatchCols.direct_ref_col, Encode_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ] + [
        (f'{col}_name{submotif_i}_constit_all', 'concat', f'{col}_name{submotif_i}_constit_all')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [Encode_ReferenceMatchCols.direct_ref_col, Encode_ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ])

    weighted_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Encode_MetadataCols.composite_col, "average", Encode_MetadataCols.composite_col),
        (Encode_MetadataCols.avgdist_summit_col, "average", Encode_MetadataCols.avgdist_summit_col),
        (Encode_MetadataCols.avgcontrib_score_col, "average", Encode_MetadataCols.avgcontrib_score_col),
        (Encode_MetadataCols.num_subunit_col, "average", Encode_MetadataCols.num_subunit_col),
        (Encode_MetadataCols.num_subunit_direct_col, "average", Encode_MetadataCols.num_subunit_direct_col),
    ])

    merge_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (Encode_ReferenceMatchCols.ref_name_constit_1_col, "merge", Encode_ReferenceMatchCols.ref_name_constit_all_col),
        (Encode_ReferenceMatchCols.direct_ref_name_constit_1_col, "merge", Encode_ReferenceMatchCols.direct_ref_name_constit_all_col),
        (Encode_ReferenceMatchCols.znf_ref_name_constit_1_col, "merge", Encode_ReferenceMatchCols.znf_ref_name_constit_all_col),
        (Encode_LabelCols.label_constit_1_col, "merge", Encode_LabelCols.label_constit_all_col),
        (Encode_LabelCols.label_core_constit_1_col, "merge", Encode_LabelCols.label_core_constit_all_col),
        (Encode_LabelCols.label_core_readable_constit_1_col, "merge", Encode_LabelCols.label_core_readable_constit_all_col),
        (Encode_LabelCols.label_flank_constit_1_col, "merge", Encode_LabelCols.label_flank_constit_all_col),
        (Encode_LabelCols.direct_label_constit_1_col, "merge", Encode_LabelCols.label_constit_all_col),
        (Encode_LabelCols.direct_label_core_constit_1_col, "merge", Encode_LabelCols.direct_label_core_constit_all_col),
        (Encode_LabelCols.direct_label_core_readable_constit_1_col, "merge", Encode_LabelCols.direct_label_core_readable_constit_all_col),
    ])
## HITS ANNOTATION ----------------------------------------------------------------------------------------------
@dataclass
class HitsCols:
    hits_total_col: str = 'n_hits_total'
    hits_promoter_col: str = 'n_hits_Promoter'
    hits_promoter_frac_col: str = 'n_hits_Promoter_frac'
    hits_exonic_col: str = 'n_hits_Exonic'
    hits_exonic_frac_col: str = 'n_hits_Exonic_frac'
    hits_intronic_col: str = 'n_hits_Intronic'
    hits_intronic_frac_col: str = 'n_hits_Intronic_frac'
    hits_distal_col: str = 'n_hits_Distal'
    hits_distal_frac_col: str = 'n_hits_Distal_frac'

    region_type_barplot_col: str = 'region_type_barplot'
    distToTSS_values_col: str = 'distToTSS_values'
    distToTSS_boxplot_col: str = 'distToTSS_boxplot'
## ENCODE COMBINE PIPELINE -------------------------------------------------------------------------------------
@dataclass
class EncodeCombineArgs:
    # Atlas identifiers
    tfatlas_id: str = 'TF'
    chromatlas_id: str = 'Chrom'

    # Per-atlas export file naming, when writing back into the upstream atlas pipeline dirs:
    #   {file_general}{level_suffix}{flag_suffix}{general_suffix}{filetype}
    file_general: str = 'motifcompendium'
    suffix_general: str = '_vf'
    suffix_general_metadata: str = '_metadata_vf'

    # Filter-flag value per output flavour: '_filtered' keeps flag_remove == False
    flags: List[Tuple[str, bool]] = field(default_factory=lambda: [
        ('_filtered', False),
        ('_removed', True),
    ])

    # Level suffixes, per atlas / head
    suffix_combined_tf: str = '_tfatlas'
    suffix_combined_chrom: str = '_chromatlas'
    suffix_metacluster: str = '_metacluster'
    suffix_cluster: str = '_cluster'
    suffix_motif: str = ''

    # Metadata columns dropped before export
    remove_cols: List[str] = field(default_factory=lambda: [
        'path',
        Encode_MetadataCols.finemo_col,
        'path_check',
        Encode_MetadataCols.notes_col,
        Encode_MetadataCols.checked_col,
    ])

## ATLAS-CLUSTER (COMBINE) FILTERS: TF -----------------------------------------------------------------
# Filters used by the TF atlas-cluster / combine stage. These come from
# combine_counts-profile.ipynb, which uses different GC/AT-bias thresholds than
# TF_MotifFilterArgs (used by the per-head stage). Kept separate so the two stages cannot drift.
#
# Deltas vs TF_MotifFilterArgs -- metric "copair_composition", operation ">", filter "7_gcbias_2":
#   soft_filters      0.45  (TF_MotifFilterArgs: 0.4)
#   moderate_filters  0.5   (TF_MotifFilterArgs: 0.45)
#   hard_filters      0.6   (same)
# flag_type stays "high-GC-AT" (a "/" breaks the MoDISco HDF5 export). See CHANGELOG 2026-08-31.
_TF_ATLAS_FILTER_THRESHOLD_OVERRIDES = {
    "soft_filters": {"7_gcbias_2": 0.45},      # TF_MotifFilterArgs: 0.4
    "moderate_filters": {"7_gcbias_2": 0.5},   # TF_MotifFilterArgs: 0.45
}


def _tf_atlas_filter_group(group_name: str) -> tuple:
    """Copy a TF_MotifFilterArgs group, applying the TF atlas-stage threshold overrides.

    Uses dataclasses.replace so the _FilterArgs instances are copies: the tuple defaults are
    shared class-level objects, so mutating them would also change TF_MotifFilterArgs.
    """
    overrides = _TF_ATLAS_FILTER_THRESHOLD_OVERRIDES.get(group_name, {})
    filters = []
    for filter_args in getattr(TF_MotifFilterArgs(), group_name):
        filter_copy = replace(filter_args)
        if filter_copy.name in overrides:
            filter_copy = replace(filter_copy, threshold=overrides[filter_copy.name])
        filters.append(filter_copy)
    return tuple(filters)


@dataclass
class TF_AtlasFilterArgs:
    motif_metrics: tuple = field(default_factory=lambda: tuple(TF_MotifFilterArgs().motif_metrics))
    soft_filters: tuple = field(default_factory=lambda: _tf_atlas_filter_group("soft_filters"))
    soft_override_filters: tuple = field(default_factory=lambda: _tf_atlas_filter_group("soft_override_filters"))
    moderate_filters: tuple = field(default_factory=lambda: _tf_atlas_filter_group("moderate_filters"))
    moderate_override_filters: tuple = field(default_factory=lambda: _tf_atlas_filter_group("moderate_override_filters"))
    hard_filters: tuple = field(default_factory=lambda: _tf_atlas_filter_group("hard_filters"))


    def to_dict(self):
        return {
            "motif_metrics": list(self.motif_metrics),
            "soft_filters": [f.to_dict() for f in self.soft_filters],
            "moderate_filters": [f.to_dict() for f in self.moderate_filters],
            "hard_filters": [f.to_dict() for f in self.hard_filters],
            "soft_override_filters": [f.to_dict() for f in self.soft_override_filters],
            "moderate_override_filters": [f.to_dict() for f in self.moderate_override_filters],
        }


## ATLAS-CLUSTER (COMBINE) FILTERS: Chrom -----------------------------------------------------------------
# Filters used by the Chrom atlas-cluster / combine stage. These come from
# combine_counts-profile.ipynb, which uses different GC/AT-bias thresholds than
# Chrom_MotifFilterArgs (used by the per-head stage). Kept separate so the two stages cannot drift.
#
# Deltas vs Chrom_MotifFilterArgs -- metric "copair_composition", operation ">", filter "7_gcbias_2":
#   soft_filters  0.45  (Chrom_MotifFilterArgs: 0.4)
#   hard_filters  0.5   (Chrom_MotifFilterArgs: 0.45)
# flag_type stays "high-GC-AT" (a "/" breaks the MoDISco HDF5 export). See CHANGELOG 2026-08-31.
_Chrom_ATLAS_FILTER_THRESHOLD_OVERRIDES = {
    "soft_filters": {"7_gcbias_2": 0.45},      # Chrom_MotifFilterArgs: 0.4
    "hard_filters": {"7_gcbias_2": 0.5},       # Chrom_MotifFilterArgs: 0.45
}


def _chrom_atlas_filter_group(group_name: str) -> tuple:
    """Copy a Chrom_MotifFilterArgs group, applying the Chrom atlas-stage threshold overrides.

    Uses dataclasses.replace so the _FilterArgs instances are copies: the tuple defaults are
    shared class-level objects, so mutating them would also change Chrom_MotifFilterArgs.
    """
    overrides = _Chrom_ATLAS_FILTER_THRESHOLD_OVERRIDES.get(group_name, {})
    filters = []
    for filter_args in getattr(Chrom_MotifFilterArgs(), group_name):
        filter_copy = replace(filter_args)
        if filter_copy.name in overrides:
            filter_copy = replace(filter_copy, threshold=overrides[filter_copy.name])
        filters.append(filter_copy)
    return tuple(filters)


@dataclass
class Chrom_AtlasFilterArgs:
    motif_metrics: tuple = field(default_factory=lambda: tuple(Chrom_MotifFilterArgs().motif_metrics))
    soft_filters: tuple = field(default_factory=lambda: _chrom_atlas_filter_group("soft_filters"))
    hard_filters: tuple = field(default_factory=lambda: _chrom_atlas_filter_group("hard_filters"))
    override_filters: tuple = field(default_factory=lambda: _chrom_atlas_filter_group("override_filters"))


    def to_dict(self):
        return {
            "motif_metrics": list(self.motif_metrics),
            "soft_filters": [f.to_dict() for f in self.soft_filters],
            "hard_filters": [f.to_dict() for f in self.hard_filters],
            "override_filters": [f.to_dict() for f in self.override_filters],
        }


## PROFILES -------------------------------------------------------------------------------------------------
# Per-atlas config selection. set_default_configs(atlas) binds these to module globals.
PROFILES = {
    "TF": {
        "ClusterArgs": TF_ClusterArgs,
        "DirectTypeArgs": TF_DirectTypeArgs,
        "FilterArgs": TF_FilterArgs,
        "LabelCols": TF_LabelCols,
        "MetadataCols": TF_MetadataCols,
        "MotifFilterArgs": TF_MotifFilterArgs,
        "ReferenceMatchCols": TF_ReferenceMatchCols,
        "TFAnnotationCols": TF_TFAnnotationCols,
        "VisualizeArgs": TF_VisualizeArgs,
        "SmithWatermanArgs": SmithWatermanArgs,
        "AtlasFilterArgs": TF_AtlasFilterArgs,
        "TFMoDIScoCols": TF_TFMoDIScoCols,
        "MotifMatchArgs": MotifMatchArgs,
        "MotifCompendiumArgs": MotifCompendiumArgs,
        "CellTypeCols": CellTypeCols,
    },
    "Chrom": {
        "ClusterArgs": Chrom_ClusterArgs,
        "DirectTypeArgs": Chrom_DirectTypeArgs,
        "FilterArgs": Chrom_FilterArgs,
        "LabelCols": Chrom_LabelCols,
        "MetadataCols": Chrom_MetadataCols,
        "MotifFilterArgs": Chrom_MotifFilterArgs,
        "AtlasFilterArgs": Chrom_AtlasFilterArgs,
        "TFMoDIScoCols": Chrom_TFMoDIScoCols,
        "ReferenceMatchCols": Chrom_ReferenceMatchCols,
        "VisualizeArgs": Chrom_VisualizeArgs,
        "MotifMatchArgs": MotifMatchArgs,
        "MotifCompendiumArgs": MotifCompendiumArgs,
        "CellTypeCols": CellTypeCols,
    },
    "ENCODE": {
        "ClusterArgs": Encode_ClusterArgs,
        "DirectTypeArgs": Encode_DirectTypeArgs,
        "FilterArgs": Encode_FilterArgs,
        "LabelCols": Encode_LabelCols,
        "MetadataCols": Encode_MetadataCols,
        "MotifFilterArgs": Encode_MotifFilterArgs,
        "ReferenceMatchCols": Encode_ReferenceMatchCols,
        "TFAnnotationCols": Encode_TFAnnotationCols,
        "VisualizeArgs": Encode_VisualizeArgs,
        "HitsCols": HitsCols,
        "EncodeCombineArgs": EncodeCombineArgs,
        "MotifMatchArgs": MotifMatchArgs,
        "MotifCompendiumArgs": MotifCompendiumArgs,
        "CellTypeCols": CellTypeCols,
    },
}
