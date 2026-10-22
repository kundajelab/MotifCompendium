from dataclasses import dataclass, field, asdict
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
    num_hits_col: str = 'num_hits'
    # Input paths, per model (from the CLI)
    modisco_path_col: str = 'modisco_path'
    pfm_path_col: str = 'pfm_path'
    finemo_path_col: str = 'finemo_path'
    peaks_path_col: str = 'peaks_path'
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
            apply_cluster=True,
            flag_type="unvalidated",
        ),
        _FilterArgs(
            name="1_minhits",
            metric=Encode_MetadataCols.num_hits_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
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
            name="1_minhits",
            metric=Encode_MetadataCols.num_hits_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
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
            name="1_minhits",
            metric=Encode_MetadataCols.num_hits_col,
            operation="<",
            threshold=100,
            invert=False,
            override=False,
            apply_motif=True,
            apply_cluster=True,
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
        Encode_LabelCols.label_conf_col, Encode_DirectTypeArgs.direct_type_col, Encode_MetadataCols.num_seqlet_col, Encode_MetadataCols.num_hits_col,
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
        (Encode_MetadataCols.num_hits_col, "sum", Encode_MetadataCols.num_hits_col),
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
        (Encode_LabelCols.direct_label_constit_1_col, "merge", Encode_LabelCols.direct_label_constit_all_col),
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
    procapatlas_id: str = 'ProCap'

    # Per-atlas export file naming, when writing back into the upstream atlas pipeline dirs (over their own files):
    #   {file_general}{level_suffix}{flag_suffix}{suffix}{filetype}, suffix: '' (.mc, MEME, HTML) or suffix_metadata (TSV)
    file_general: str = 'motifcompendium'
    suffix_metadata: str = '_metadata'

    # Filter-flag value per output flavour: '_filtered' keeps flag_remove == False
    flags: List[Tuple[str, bool]] = field(default_factory=lambda: [
        ('_filtered', False),
        ('_removed', True),
    ])

    # Level suffixes: combined (atlasclusters), and per head
    suffix_atlascluster: str = '_atlascluster'
    suffix_metacluster: str = '_metacluster'
    suffix_cluster: str = '_cluster'
    suffix_motif: str = '_motif'

    # Metadata columns dropped before export. The motifs' input paths (modisco_path, pfm_path, finemo_path, peaks_path)
    # are kept:
    # a rerun of the ENCODE combine reads the FiNeMo and peaks paths from the per-head compendia it writes back.
    remove_cols: List[str] = field(default_factory=lambda: [
        'path',
        Encode_MetadataCols.finemo_col,
        'path_check',
        Encode_MetadataCols.notes_col,
        Encode_MetadataCols.checked_col,
    ])

## PROFILES -------------------------------------------------------------------------------------------------
# Config selection: pipeline_encode.set_default_configs() binds these to module globals.
# The TF, Chromatin and ProCap atlas configs live in their pipelines' configs.py (pipeline/bpnet,
# pipeline/chrombpnet, pipeline/procapnet).
PROFILES = {
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
