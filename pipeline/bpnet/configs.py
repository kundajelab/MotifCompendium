from dataclasses import dataclass, field, asdict, replace
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
    num_hits_col: str = 'num_hits'
    num_constituents_col: str = 'num_constituents'
    posneg_col: str = 'posneg'
    avgcontrib_score_col: str = "avg_contrib_score"
    total_contrib_score_col: str = "total_contrib_score"
    group_total_contrib_score_col: str = "group_total_contrib_score"
    total_contrib_score_ratio_col: str = "total_contrib_score_ratio"
    avgdist_summit_col: str = "avg_dist_from_summit"


## TF ANNOTATION ----------------------------------------------------------------------------------------------------
@dataclass
class TFAnnotationCols:
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
class ReferenceMatchCols:
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

    best_label_ref_col: str = "best_label_reference"


## MOTIF MATCH --------------------------------------------------------------------------------------------------
@dataclass
class MotifMatchArgs:
    min_single_score: float = 0.8
    min_composite_score: float = 0.7
    max_submotifs: int = 3
    min_composite_subunit: float = 1.0


## DIRECT TYPE (Alpha, Beta Non-ZNF) -----------------------------------------------------------------------------
@dataclass
class DirectTypeArgs:
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
    gamma_col: str = MetadataCols.avgcontrib_score_col


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
            name="1_minhits",
            metric=MetadataCols.num_hits_col,
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
            name="1_minhits",
            metric=MetadataCols.num_hits_col,
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
            name="1_minhits",
            metric=MetadataCols.num_hits_col,
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
    # Moderate override: filter_flag_col AND apply_filter_threshold must both be True, to keep flag True
    moderate_override_filters: tuple = (
        _FilterArgs(
            name=f"direct_match_{DirectTypeArgs.direct_types[0]}",  # "A"
            metric=DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=DirectTypeArgs.direct_types[0],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{DirectTypeArgs.direct_types[1]}",  # "B-NZ"
            metric=DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=DirectTypeArgs.direct_types[1],
            invert=True,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name=f"direct_match_{DirectTypeArgs.direct_types[2]}",  # "B-Z"
            metric=DirectTypeArgs.direct_type_col,
            operation="contains",
            threshold=DirectTypeArgs.direct_types[2],
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


## ATLAS-CLUSTER (COMBINE) FILTERS -------------------------------------------------------------------------------
# Filters used by the atlas-cluster / combine stage (pipeline_tfatlas_combine.py).
#
# These come from combine_counts-profile.ipynb, which uses different GC/AT-bias thresholds
# than MotifFilterArgs (used by the per-head pipeline, pipeline_tfatlas.py). They are kept in a
# separate class so the two stages cannot silently drift into each other.
#
# Deltas vs MotifFilterArgs -- metric "copair_composition", operation ">":
#   soft_filters      "7_gcbias_2": 0.45  (MotifFilterArgs: 0.4)
#   moderate_filters  "7_gcbias_2": 0.5   (MotifFilterArgs: 0.45)
#   hard_filters      "7_gcbias_2": 0.6   (same as MotifFilterArgs)
# Everything else (metrics, operations, invert / override / apply_motif / apply_cluster flags,
# and every other threshold) is identical to MotifFilterArgs, so the groups below are derived
# from it and only the thresholds listed above are overridden.
#
# DELIBERATE DEVIATION from combine_counts-profile.ipynb: flag_type stays "high-GC-AT" rather
# than the notebook's "high-GC/AT". A "/" in flag_type reaches the cluster name / label columns
# and raises "Motif names cannot have slashes (/) in them!" during the MoDISco HDF5 export
# (h5py treats "/" as a path separator). See CHANGELOG 2026-08-31.
_ATLAS_FILTER_THRESHOLD_OVERRIDES = {
    "soft_filters": {"7_gcbias_2": 0.45},      # MotifFilterArgs: 0.4
    "moderate_filters": {"7_gcbias_2": 0.5},   # MotifFilterArgs: 0.45
}


def _atlas_filter_group(group_name: str) -> tuple:
    """Copy a MotifFilterArgs filter group, applying the atlas-stage threshold overrides.

    Uses dataclasses.replace so the _FilterArgs instances are copies: the tuple defaults on
    MotifFilterArgs are shared class-level objects, so mutating them in place would also change
    MotifFilterArgs for the per-head pipeline.
    """
    overrides = _ATLAS_FILTER_THRESHOLD_OVERRIDES.get(group_name, {})
    filters = []
    for filter_args in getattr(MotifFilterArgs(), group_name):
        filter_copy = replace(filter_args)
        if filter_copy.name in overrides:
            filter_copy = replace(filter_copy, threshold=overrides[filter_copy.name])
        filters.append(filter_copy)
    return tuple(filters)


@dataclass
class AtlasFilterArgs:
    # Motif metrics to be calculated (same as MotifFilterArgs)
    motif_metrics: tuple = field(default_factory=lambda: tuple(MotifFilterArgs().motif_metrics))

    soft_filters: tuple = field(default_factory=lambda: _atlas_filter_group("soft_filters"))
    soft_override_filters: tuple = field(default_factory=lambda: _atlas_filter_group("soft_override_filters"))
    moderate_filters: tuple = field(default_factory=lambda: _atlas_filter_group("moderate_filters"))
    moderate_override_filters: tuple = field(default_factory=lambda: _atlas_filter_group("moderate_override_filters"))
    hard_filters: tuple = field(default_factory=lambda: _atlas_filter_group("hard_filters"))

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
    cluster_algorithm_main: str = "cpm_leiden"
    cluster_algorithm_force: str = "dcc"
    cluster_algorithm_kmeans: str = "k_centroids"
    cluster_sim_threshold: float = 0.88
    cluster_sim_threshold_force: float = 0.92
    cluster_force_density: float = 1.0
    cluster_sim_assignment_threshold_kmeans: float = 0.1
    direct_cluster_within: str = DirectTypeArgs.tf_direct_col
    nondirect_cluster_within: str = MetadataCols.tf_col
    metacluster_within: str = MetadataCols.composite_col
    # combine_counts-profile: merge counts + profile meta-clusters into atlasclusters
    atlascluster_within: str = MetadataCols.composite_col
    cluster_max_iter_main: int = 0
    cluster_max_iter_force: int = 50
    average_weight_col: str = MetadataCols.num_seqlets_col
    n_iterations: int = -1  # Run until convergence
    seeds: List[int] = field(default_factory=lambda: [100, 200])
    source_cluster_col: str = "source_cluster"

    # Aggregation of per-motif metadata into cluster-level metadata
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
        (LabelCols.label_col, "concat", LabelCols.label_constit_1_col),  # [Dimer-style notation] Combined
        (LabelCols.label_constit_1_col, "concat", LabelCols.label_constit_all_col),
        (LabelCols.direct_label_col, "concat", LabelCols.direct_label_constit_1_col),  # [Dimer-style notation] Combined (Direct)
        (LabelCols.direct_label_constit_1_col, "concat", LabelCols.direct_label_constit_all_col),
        (LabelCols.label_core_col, "concat", LabelCols.label_core_constit_1_col),  # [Dimer-style notation] Core TF
        (LabelCols.label_core_constit_all_col, "concat", LabelCols.label_core_constit_all_col),
        (LabelCols.direct_label_core_col, "concat", LabelCols.direct_label_core_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (LabelCols.direct_label_core_constit_all_col, "concat", LabelCols.direct_label_core_constit_all_col),
        (LabelCols.label_core_readable_col, "concat", LabelCols.label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF family
        (LabelCols.label_core_readable_constit_all_col, "concat", LabelCols.label_core_readable_constit_all_col),
        (LabelCols.direct_label_core_readable_col, "concat", LabelCols.direct_label_core_readable_constit_1_col),  # [Dimer-style notation] Core TF (Direct)
        (LabelCols.direct_label_core_readable_constit_all_col, "concat", LabelCols.direct_label_core_readable_constit_all_col),
        (LabelCols.label_flank_col, "concat", LabelCols.label_flank_constit_1_col),  # [Dimer-style notation] Flank
        (LabelCols.label_flank_constit_all_col, "concat", LabelCols.label_flank_constit_all_col),
        (ReferenceMatchCols.ref_name_col, "concat", ReferenceMatchCols.ref_name_constit_1_col),  # All unique reference motifs
        (ReferenceMatchCols.ref_name_constit_all_col, "concat", ReferenceMatchCols.ref_name_constit_all_col),
        (ReferenceMatchCols.direct_ref_name_col, "concat", ReferenceMatchCols.direct_ref_name_constit_1_col),  # All unique direct reference motifs
        (ReferenceMatchCols.direct_ref_name_constit_all_col, "concat", ReferenceMatchCols.direct_ref_name_constit_all_col),
        (ReferenceMatchCols.znf_ref_name_col, "concat", ReferenceMatchCols.znf_ref_name_constit_1_col),  # All unique direct reference motifs: Beta ZNF
        (ReferenceMatchCols.znf_ref_name_constit_all_col, "concat", ReferenceMatchCols.znf_ref_name_constit_all_col),
        (MetadataCols.tf_col, "concat", MetadataCols.tf_col),
        (TFAnnotationCols.family_col, "concat", TFAnnotationCols.family_col),
        (TFAnnotationCols.tfclass_family_name_col, "concat", TFAnnotationCols.tfclass_family_name_col),
        (TFAnnotationCols.tfclass_family_id_col, "concat", TFAnnotationCols.tfclass_family_id_col),
        (TFAnnotationCols.humantf_dbd_col, "concat", TFAnnotationCols.humantf_dbd_col),
        (MetadataCols.biosample_col, "concat", MetadataCols.biosample_col),
    ] + [
        (f'{col}_name{submotif_i}', 'concat', f'{col}_name{submotif_i}_constit_prev')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [ReferenceMatchCols.direct_ref_col, ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ] + [
        (f'{col}_name{submotif_i}_constit_all', 'concat', f'{col}_name{submotif_i}_constit_all')
        for submotif_i in range(MotifMatchArgs.max_submotifs)
        for col in [ReferenceMatchCols.direct_ref_col, ReferenceMatchCols.ref_col]  # All unique reference motifs per subunit
    ])

    weighted_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (MetadataCols.composite_col, "average", MetadataCols.composite_col),
        (MetadataCols.avgdist_summit_col, "average", MetadataCols.avgdist_summit_col),
        (MetadataCols.avgcontrib_score_col, "average", MetadataCols.avgcontrib_score_col),
        (MetadataCols.num_subunit_col, "average", MetadataCols.num_subunit_col),
        (MetadataCols.num_subunit_direct_col, "average", MetadataCols.num_subunit_direct_col),
    ])

    merge_aggregations: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        (ReferenceMatchCols.ref_name_constit_1_col, "merge", ReferenceMatchCols.ref_name_constit_all_col),
        (ReferenceMatchCols.direct_ref_name_constit_1_col, "merge", ReferenceMatchCols.direct_ref_name_constit_all_col),
        (ReferenceMatchCols.znf_ref_name_constit_1_col, "merge", ReferenceMatchCols.znf_ref_name_constit_all_col),
        (LabelCols.label_constit_1_col, "merge", LabelCols.label_constit_all_col),
        (LabelCols.label_core_constit_1_col, "merge", LabelCols.label_core_constit_all_col),
        (LabelCols.label_core_readable_constit_1_col, "merge", LabelCols.label_core_readable_constit_all_col),
        (LabelCols.label_flank_constit_1_col, "merge", LabelCols.label_flank_constit_all_col),
        (LabelCols.direct_label_constit_1_col, "merge", LabelCols.label_constit_all_col),
        (LabelCols.direct_label_core_constit_1_col, "merge", LabelCols.direct_label_core_constit_all_col),
        (LabelCols.direct_label_core_readable_constit_1_col, "merge", LabelCols.direct_label_core_readable_constit_all_col),
    ])


## VISUALIZATION / HTML -----------------------------------------------------------------------------------------
@dataclass
class VisualizeArgs:
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
        MetadataCols.name_col,
        MetadataCols.id_col,
        MetadataCols.head_col,
        MetadataCols.biosample_col,
        MetadataCols.tf_col,
        MetadataCols.tf_encode_col,
        TFAnnotationCols.family_col,
        MetadataCols.num_subunit_direct_col,
        MetadataCols.num_subunit_col,
        TFAnnotationCols.tfclass_family_name_col,
        TFAnnotationCols.tfclass_family_id_col,
        TFAnnotationCols.humantf_dbd_col,
        DirectTypeArgs.direct_type_col,
        DirectTypeArgs.tf_direct_col,
        DirectTypeArgs.family_direct_col,
        VisualizeArgs.cluster_name_col,
        VisualizeArgs.metacluster_name_col,
    ])

    # Motif index export columns: combine_counts-profile (atlascluster level)
    motif_index_cols_atlascluster: List[str] = field(default_factory=lambda: [
        MetadataCols.name_col,
        MetadataCols.id_col,
        MetadataCols.head_col,
        MetadataCols.biosample_col,
        MetadataCols.tf_col,
        MetadataCols.tf_encode_col,
        TFAnnotationCols.family_col,
        DirectTypeArgs.direct_type_col,
        DirectTypeArgs.tf_direct_col,
        DirectTypeArgs.family_direct_col,
        MetadataCols.num_subunit_col,
        TFAnnotationCols.tfclass_family_name_col,
        TFAnnotationCols.tfclass_family_id_col,
        TFAnnotationCols.humantf_dbd_col,
        VisualizeArgs.cluster_name_col,
        VisualizeArgs.metacluster_name_col,
        VisualizeArgs.atlascluster_name_col,
    ])

    # HTML summary table columns
    html_columns: List[str] = field(default_factory=lambda: [
        MetadataCols.unique_id_col, MetadataCols.name_col, LabelCols.label_col, LabelCols.direct_label_col,
        LabelCols.label_conf_col, DirectTypeArgs.direct_type_col, MetadataCols.composite_col,
        MetadataCols.num_subunit_direct_col, MetadataCols.num_subunit_col, MetadataCols.num_seqlets_col, MetadataCols.num_hits_col,
        VisualizeArgs.sort_col,
    ] + [
        f"{col}_{field_name}{j}"
        for col in [ReferenceMatchCols.direct_ref_col, ReferenceMatchCols.ref_col, ReferenceMatchCols.best_direct_ref_constit_col, ReferenceMatchCols.best_ref_constit_col]
        for j in range(MotifMatchArgs.max_submotifs - 1)
        for field_name in ["logo", "name", "score"]
    ] + [
        f'{LabelCols.best_label_ref_col}_logo0', f'{LabelCols.best_label_ref_col}_name0', f'{LabelCols.best_label_ref_col}_score0',
        LabelCols.label_core_col, LabelCols.direct_label_core_col, LabelCols.label_flank_col,
        SmithWatermanArgs.msw_logo_col, SmithWatermanArgs.msw_name_col, SmithWatermanArgs.msw_score_col,
        VisualizeArgs.motif_string_col,
        MetadataCols.avgdist_summit_col, MetadataCols.avgcontrib_score_col,
        DirectTypeArgs.tf_direct_col, DirectTypeArgs.family_direct_col,
        MetadataCols.tf_col, TFAnnotationCols.family_col, TFAnnotationCols.tfclass_family_name_col,
        TFAnnotationCols.tfclass_family_id_col, TFAnnotationCols.humantf_dbd_col,
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
