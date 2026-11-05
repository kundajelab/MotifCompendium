from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union


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
    label_col: str = 'label'  # Contribcluster: Readable TF family label: {TF_family0}-{TF_family1}-...
    label_readable_col: str = 'label_readable'
    label_constit_1_col: str = 'label_constit_prev'
    label_constit_all_col: str = 'label_constit_all'

    direct_label_col: str = 'direct_label'  # Contribcluster: Readable TF family label (direct only)
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

    label_flank_readable_col: str = 'label_flank_readable'  # Composite: Subunit TF family

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

    # Hierarchy: motif -> cluster -> contribcluster -> modelcluster -> crosscluster
    cluster_name_col: str = 'cluster_name'
    contribcluster_name_col: str = 'contribcluster_name'
    modelcluster_name_col: str = 'modelcluster_name'
    crosscluster_name_col: str = 'crosscluster_name'
    # Its encodecluster (C5; pipeline_encode_composite.py): name and filter flag (True: in none) at every level below,
    # number at the crossclusters (C4); MCID, simple name and why it has none in the membership of the crossclusters
    encodecluster_name_col: str = 'encodecluster_name'
    encodecluster_flag_col: str = 'encodecluster_flag_remove'
    encodecluster_cluster_col: str = 'encodecluster_cluster'
    encodecluster_mcid_col: str = 'encodecluster_MCID'
    encodecluster_simple_name_col: str = 'encodecluster_simple_name'
    encodecluster_flag_type_col: str = 'encodecluster_flag_type'
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
        Encode_VisualizeArgs.contribcluster_name_col,
        Encode_VisualizeArgs.modelcluster_name_col,
        Encode_VisualizeArgs.crosscluster_name_col,
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

    # Filter-flag value per output flavour: '_filtered' keeps flag_remove == False, '_removed' True, '_all' (None) every row
    flags: List[Tuple[str, Optional[bool]]] = field(default_factory=lambda: [
        ('_filtered', False),
        ('_removed', True),
        ('_all', None),
    ])

    # Level suffixes: combined (modelclusters), and per head
    suffix_modelcluster: str = '_modelcluster'
    suffix_encodecluster: str = '_encodecluster'  # The encodecluster level (C5), in <run>/encode
    suffix_contribcluster: str = '_contribcluster'
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

## COMPOSITE MOTIFS ------------------------------------------------------------------------------------------
# Composite motifs of the crossclusters (composite/identify_composite_motifs.ipynb): the modified Bellman-Ford
# decomposition into reference monomers (pipeline_encode.py), ISM cooperativity (pipeline_encode_composite_ism.py),
# and the merge of duplicate and soft / no-synergy composites into the final ENCODE MotifCompendium
# (pipeline_encode_composite.py).
@dataclass
class Encode_CompositeArgs:
    # Modified Bellman-Ford: decomposition into direct reference monomers
    bellman_score_col: str = "bellman_composite_score"
    bellman_name_col: str = "bellman_composite_name"
    bellman_label_col: str = "bellman_composite_label"
    bellman_sim_col: str = "bellman_composite_similarity"
    bellman_orient_col: str = "bellman_composite_orientation"
    bellman_align_col: str = "bellman_composite_alignment"
    bellman_logo_col: str = "bellman_composite_logo"

    # utils_composite.run_modified_bellman_ford_composite() at its defaults since 3acaa1f (v27; v26 and before:
    # sim threshold 0.5, overlap penalty 1.0, match count penalty 0.1, the old score)
    bellman_sim_min_threshold: float = 0.0
    bellman_subunit_motif_trim: bool = True  # Trim each reference subunit (as logo_trimming: True = 1/M)
    bellman_overlap_penalty_fraction: float = 0.0  # Of the target energy two consecutive subunits both cover (0: no penalty)
    bellman_match_count_max: int = 2
    bellman_match_count_penalty: float = 0.05  # Per match, subtracted from the score (cosine units)
    bellman_match_count_fraction: float = 1
    bellman_target_similarity: float = 0.9
    bellman_collapse: str = "sum_norm"  # The matched subunits' collapse into one motif, scored by its cosine

    # Reference monomers: the direct reference without its composites, long non-ZNF motifs and IRFs
    reference_cluster_col: str = "reference_cluster"
    ref_composites: List[str] = field(default_factory=lambda: [
        'HOCO.IRF3.H14CORE.0.PS.A', 'HOCO.SOX8.H14CORE.0.PSM.A', 'JASP.MA0050.2.IRF1', 'CISBP.M05748_3.00.IRF8',
        'CISBP.M05005_3.00.ETV3L', 'CISBP.M03194_3.00.ETV6', 'CISBP.M05074_3.00.GATA2', 'CISBP.M03668_3.00.RUNX1',
        'CISBP.M05082_3.00.GATA2', 'CISBP.M03670_3.00.RUNX1', 'HOCO.ETV4.H14CORE.0.P.B', 'CISBP.M04921_3.00.ETV3L',
        'CISBP.M05737_3.00.IRF3', 'CISBP.M03078_3.00.PRDM1', 'CISBP.M03188_3.00.ETV3L', 'CISBP.M03137_3.00.HIC2',
        'CISBP.M04534_3.00.JUN', 'CISBP.M05751_3.00.IRF7', 'CISBP.M04200_3.00.TEAD4', 'CISBP.M03961_3.00.GATA2',
        'HOCO.ETV2.H14INVITRO.1.PM.A', 'HOCO.IRF4.H14CORE.0.P.B', 'CISBP.M04943_3.00.ETV3L', 'JASP.MA0101.1.REL',
        'CISBP.M05069_3.00.GATA2', 'HOCO.IRF3.H14INVITRO.0.PS.A', 'HOCO.IRF8.H14CORE.1.SM.B', 'CISBP.M05002_3.00.ETV3L',
        'CISBP.M05078_3.00.GATA2', 'CISBP.M05937_3.00.RUNX1', 'CISBP.M05745_3.00.IRF8', 'JASP.MA0107.1.RELA',
        'HOCO.IRF8.H14CORE.0.P.B', 'JASP.MA0050.1.IRF1', 'CISBP.M05753_3.00.IRF9', 'CISBP.M05752_3.00.IRF7',
        'HOCO.IRF2.H14INVITRO.0.PSM.A', 'HOCO.IRF5.H14INVITRO.0.SM.B', 'HOCO.IRF8.H14CORE.2.SM.B', 'HOCO.IRF8.H14INVITRO.0.P.B',
        'CISBP.M05738_3.00.IRF3', 'JASP.MA0772.1.IRF7', 'JASP.MA0653.1.IRF9', 'HOCO.IRF1.H14CORE.0.P.B',
        'HOCO.IRF3.H14CORE.1.SM.B', 'HOCO.IRF4.H14INVITRO.2.SM.B', 'HOCO.IRF9.H14CORE.0.PSM.A', 'JASP.MA0051.1.IRF2',
        'HOCO.IRF8.H14INVITRO.1.SM.B', 'CISBP.M03543_3.00.IRF9', 'CISBP.M05747_3.00.IRF8', 'HOCO.IRF1.H14INVITRO.0.P.D',
        'CISBP.M05754_3.00.IRF9', 'HOCO.IRF5.H14CORE.0.SM.B', 'JASP.MA0652.1.IRF8', 'HOCO.IRF4.H14INVITRO.0.P.B',
        'CISBP.M03541_3.00.IRF7', 'HOCO.IRF9.H14INVITRO.0.PSM.A', 'JASP.MA1418.1.IRF3', 'CISBP.M05746_3.00.IRF8',
        'HOCO.IRF4.H14INVITRO.1.S.B', 'HOCO.IRF3.H14INVITRO.1.SM.B', 'HOCO.IRF8.H14INVITRO.2.SM.B', 'CISBP.M00876_3.00.IRF2',
        'JASP.MA0051.2.IRF2', 'HOCO.IRF2.H14CORE.0.PSM.A', 'JASP.MA0772.2.IRF7', 'CISBP.M03539_3.00.IRF8',
        'CISBP.M05750_3.00.IRF2', 'CISBP.M02336_3.00.IRF5', 'CISBP.M03540_3.00.IRF8', 'HOCO.IRF4.H14CORE.1.S.B',
        'CISBP.M05744_3.00.IRF8', 'JASP.MA1419.1.IRF4', 'CISBP.M03535_3.00.IRF3', 'CISBP.M05736_3.00.IRF3',
        'HOCO.IRF5.H14CORE.1.S.B', 'CISBP.M05743_3.00.IRF8', 'HOCO.IRF5.H14INVITRO.1.S.B', 'CISBP.M03538_3.00.IRF8',
        'JASP.MA1418.2.IRF3', 'CISBP.M02337_3.00.IRF7', 'JASP.MA0652.2.IRF8', 'CISBP.M05739_3.00.IRF5',
        'CISBP.M05749_3.00.IRF2', 'JASP.MA1419.2.IRF4', 'HOCO.IRF7.H14INVITRO.1.SM.B', 'CISBP.M05741_3.00.IRF5',
        'HOCO.IRF7.H14CORE.1.SM.B', 'CISBP.M05742_3.00.IRF5', 'CISBP.M05740_3.00.IRF5', 'CISBP.M00230_3.00.IRF8',
        'JASP.MA1420.1.IRF5', 'HOCO.IRF7.H14CORE.0.S.B', 'HOCO.IRF6.H14INVITRO.1.S.C', 'HOCO.IRF7.H14INVITRO.0.S.B',
        'CISBP.M03542_3.00.IRF7', 'CISBP.M03536_3.00.IRF5', 'CISBP.M02340_3.00.IRF9', 'HOCO.IRF6.H14CORE.1.S.C',
        'CISBP.M02335_3.00.IRF5', 'CISBP.M05735_3.00.IRF3', 'CISBP.M02339_3.00.IRF9', 'CISBP.M00228_3.00.IRF9',
        'CISBP.M02341_3.00.IRF5', 'CISBP.M03537_3.00.IRF5', 'CISBP.M00232_3.00.IRF5', 'CISBP.M02338_3.00.IRF5',
        'CISBP.M02342_3.00.IRF5', 'CISBP.M00229_3.00.IRF3', 'HOCO.IRF6.H14INVITRO.0.SM.B', 'CISBP.M00231_3.00.IRF5',
        'CISBP.M05733_3.00.IRF5', 'HOCO.IRF6.H14CORE.0.SM.B', 'CISBP.M05734_3.00.IRF5', 'CISBP.M05731_3.00.IRF5',
        'CISBP.M05732_3.00.IRF5', 'CISBP.M05068_3.00.GATA2', 'CISBP.M05070_3.00.GATA2', 'CISBP.M05072_3.00.GATA2',
        'CISBP.M05067_3.00.GATA2', 'CISBP.M03270_3.00.GATA2', 'CISBP.M03269_3.00.GATA2', 'CISBP.M05081_3.00.GATA2',
        'CISBP.M05080_3.00.GATA2', 'CISBP.M05076_3.00.GATA2', 'CISBP.M05084_3.00.GATA2', 'HOCO.GATA3.H14CORE.1.SM.B',
        'HOCO.GATA3.H14INVITRO.1.SM.B', 'JASP.MA0766.1.GATA5', 'JASP.MA1104.1.GATA6', 'HOCO.KLF15.H14CORE.1.P.C',
        'CISBP.M05965_3.00.SOX18', 'CISBP.M05955_3.00.SOX18', 'CISBP.M04998_3.00.ETV3L', 'CISBP.M02935_3.00.PRDM1',
        'CISBP.M03212_3.00.SPIB', 'JASP.MA1643.1.NFIB', 'JASP.MA1643.2.NFIB', 'CISBP.M05951_3.00.NFIA',
        'JASP.MA2510.1.ZBTB41', 'JASP.UN0145.1.ZBTB37', 'HOCO.KLF8.H14INVITRO.1.P.D', 'HOCO.NR1D1.H14CORE.1.SM.B',
        'CISBP.M03672_3.00.RUNX1', 'CISBP.M05938_3.00.RUNX1', 'HOCO.KLF8.H14CORE.1.P.C', 'CISBP.M04941_3.00.ETV3L',
        'CISBP.M05966_3.00.SOX18', 'HOCO.SOX6.H14INVITRO.0.P.D', 'HOCO.SPIB.H14CORE.1.S.C', 'HOCO.SOX6.H14CORE.0.P.B',
        'JASP.MA0867.2.SOX4', 'CISBP.M05969_3.00.SOX18', 'HOCO.SPIB.H14INVITRO.1.S.C', 'HOCO.KLF15.H14INVITRO.1.P.C',
        'HOCO.NR1D2.H14INVITRO.1.SM.B', 'HOCO.KLF12.H14INVITRO.0.P.C', 'JASP.MA0867.3.SOX4', 'CISBP.M05956_3.00.SOX18',
        'JASP.MA0080.4.SPI1', 'CISBP.M05970_3.00.SOX18', 'CISBP.M03169_3.00.SPIB', 'CISBP.M00210_3.00.SPIB',
        'HOCO.NR1D2.H14CORE.1.SM.B', 'CISBP.M04169_3.00.RUNX1', 'HOCO.COT1.H14INVITRO.1.PSM.A', 'CISBP.M04025_3.00.DPRX',
        'HOCO.TEAD1.H14INVITRO.2.S.C', 'CISBP.M03185_3.00.ETV3L', 'CISBP.M03190_3.00.ETV3L',
    ])
    monomer_long_tf_names: str = "SOX|SPI|KLF|RUNX|NR1|ETV|ETS|TEAD|NFKB|REL|HOX|PAX|STAT|POU|ELF|NFAT|HES|USF|FOX"
    monomer_long_families: List[str] = field(default_factory=lambda: ["SP", "MAX"])
    monomer_max_length: int = 10  # Longer (motif string) non-ZNF monomers are removed
    monomer_irf_keep: str = "JASP.MA1509.1.IRF6"  # The only IRF kept

    # Subunits: reference matches per composite
    composite_ref_col: str = "composite_reference"
    max_submotifs_composite: int = 3  # Reference submotifs re-matched to the merged composites (C5); identify_composites(): 2 (core, flank)
    composite_error_prefix: str = "_"  # Composites with a core or flank label starting with it are dropped
    families: List[str] = field(default_factory=lambda: [
        'ARID', 'ARNT', 'BCL', 'BHLH', 'CEBP', 'CREB', 'CDX', 'DMRT', 'ESRR', 'ELS',
        'ETS', 'ETL', 'ETV', 'E2F', 'FOS', 'FOX', 'GABP', 'GATA', 'GFI', 'HMG',
        'HNF', 'HOX', 'JUN', 'KIAA', 'MAF', 'MAX', 'MEF', 'NFAT', 'NFE', 'NFI',
        'NFY', 'NANOG', 'NR', 'NRF', 'NEURO', 'PAX', 'PHO', 'POU', 'PPAR', 'PRDM',
        'REL', 'ROR', 'RXR', 'SP', 'SPI', 'STAT', 'TCF', 'CTCF', 'TFAP', 'TFCP',
        'TSC22', 'TBX', 'YY', 'ZBTB', 'ZFP', 'ZNF', 'ZSCAN',
    ])

    # ISM cooperativity: per (motif, model) results, then one classification per motif
    name_id_col: str = 'name_id'
    num_ids_col: str = 'num_ids'
    ism_contrib_path_col: str = 'ism_contrib_path'
    ism_results_path_col: str = 'ism_results_path'
    ism_logo_fwd_col: str = 'ism_logo_fwd'
    ism_logo_rev_col: str = 'ism_logo_rev'
    ism_similarity_col: str = 'ism_similarity'
    ism_id_best_col: str = 'ism_id_best'
    ism_name_id_best_col: str = 'ism_name_id_best'
    ism_spacing_col: str = 'ism_spacing'
    ism_spacing_diff_col: str = 'ism_spacing_diff'
    ism_z_score_col: str = 'ism_z_score'
    ism_p_value_col: str = 'ism_p_value'
    ism_p_value_adj_col: str = 'ism_p_value_adj'
    ism_effect_max_col: str = 'ism_effect_max'
    ism_effect_medium_col: str = 'ism_effect_medium'
    ism_identity_col: str = 'ism_identity'
    ism_class_col: str = 'ism_class'  # "hard & soft", "hard", "soft", "no synergy"
    ism_class_rank_col: str = 'ism_class_rank'

    ism_similarity_threshold: float = 0.95
    ism_spacing_threshold: int = 1

    # Classification thresholds, matching HDMA's 04c-plot_cooperativity_results.Rmd:
    # "hard" = synergy is sharply spacing-dependent (peaked at one spacing/orientation);
    # "soft" = synergy persists broadly across a medium-distance (20-150bp) spacing window;
    # a pair can be both, or neither ("no synergy").
    p_val_thresh: float = 0.001
    z_thresh: float = 4
    delta_thresh: float = 0.15
    result_priority: Tuple[str, ...] = ("hard & soft", "hard", "soft", "no synergy")
    result_col_suffix: dict = field(default_factory=lambda: {
        "hard & soft": "hard_and_soft", "hard": "hard", "soft": "soft", "no synergy": "no_synergy",
    })

    # Merge: duplicate motifs, then soft + no-synergy composites of the same reference clusters
    duplicate_sim_threshold: float = 0.92
    duplicate_score_col: str = 'duplicate_score'
    duplicate_logo_col: str = 'duplicate_logo'
    duplicate_cluster_col: str = 'duplicate_cluster'
    duplicate_members_col: str = f'duplicate_{Encode_MetadataCols.unique_id_col}'
    duplicate_num_members_col: str = 'num_duplicate_members'
    soft_nosynergy_membership_col: str = 'soft_nosynergy_membership'
    soft_nosynergy_merge_cluster_col: str = 'soft_nosynergy_merge_cluster'
    soft_nosynergy_num_members_col: str = 'num_soft_nosynergy_members'

    # Final ENCODE MotifCompendium (C5): the singles and the merged composites. Crossclusters left out of it
    # are removed, with these flag types: the no-synergy composites, and the composites without subunits (the
    # level's removed compendium, motifcompendium_encode_removed: --save-all).
    mcid_level: str = "C5.encode.TF_Chrom_ProCap.counts_profile"
    # The encodecluster level (C5), in <run>/<composite_dir_name>: as a clustering, each encodecluster has a
    # number (its source_cluster: 0, 1, ...), each crosscluster (C4) the number of its encodecluster; every level
    # below its encodecluster's name and filter flag (VisualizeArgs.encodecluster_*_col)
    composite_dir_name: str = "encode"
    flag_type_no_synergy: str = 'no-synergy'
    flag_type_unannotated: str = 'composite-unannotated'

    # Composite HTML summary tables (ISM cooperativity, merged composites)
    html_columns_composite: List[str] = field(default_factory=lambda: [
        Encode_MetadataCols.simple_name_col,
        Encode_MetadataCols.id_col,
        Encode_CompositeArgs.duplicate_score_col,
        Encode_CompositeArgs.duplicate_num_members_col,
        Encode_CompositeArgs.duplicate_members_col,
        Encode_CompositeArgs.soft_nosynergy_membership_col,
        Encode_CompositeArgs.soft_nosynergy_num_members_col,
        Encode_CompositeArgs.duplicate_logo_col,
    ] + [
        Encode_CompositeArgs.ism_logo_fwd_col,
        Encode_CompositeArgs.ism_logo_rev_col,
        Encode_CompositeArgs.ism_similarity_col,
        Encode_CompositeArgs.ism_z_score_col,
        Encode_CompositeArgs.ism_p_value_col,
        Encode_CompositeArgs.ism_effect_max_col,
        Encode_CompositeArgs.ism_effect_medium_col,
        Encode_CompositeArgs.ism_identity_col,
        Encode_CompositeArgs.ism_class_col,
        Encode_CompositeArgs.ism_spacing_col,
        f'{Encode_CompositeArgs.composite_ref_col}_spacing',
        Encode_CompositeArgs.ism_spacing_diff_col,
    ] + [
        Encode_CompositeArgs.bellman_score_col,
        Encode_CompositeArgs.bellman_sim_col,
        Encode_CompositeArgs.bellman_align_col,
        Encode_CompositeArgs.bellman_name_col,
        Encode_CompositeArgs.bellman_label_col,
        Encode_CompositeArgs.bellman_logo_col,
    ] + [
        col
        for i in range(Encode_CompositeArgs.max_submotifs_composite)
        for col in [
            f"{Encode_CompositeArgs.composite_ref_col}_name{i}",
            f"{Encode_CompositeArgs.composite_ref_col}_score{i}",
            f"{Encode_CompositeArgs.composite_ref_col}_logo{i}",
            f"{Encode_CompositeArgs.composite_ref_col}_{Encode_VisualizeArgs.motif_string_col}{i}",
        ]
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
        "CompositeArgs": Encode_CompositeArgs,
        "MotifMatchArgs": MotifMatchArgs,
        "MotifCompendiumArgs": MotifCompendiumArgs,
        "CellTypeCols": CellTypeCols,
    },
}
