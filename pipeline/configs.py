import os
from dataclasses import dataclass, field, asdict
from typing import Union, List, Tuple


## INTERNAL -------------------------------------------------------------------------------------------------------
@dataclass
class _FilterArgs:
    # INTERNAL: Standard filter arguments
    name: str
    metric: str
    operation: str
    threshold: Union[float, bool]
    override: bool
    apply_motif: bool
    apply_cluster: bool

    def to_dict(self):
        # Return filter arguments as a dictionary
        return asdict(self)


## PARAMETERS --------------------------------------------------------------------------------------------------
@dataclass
class MetadataCols:
    # INTERNAL: Metadata columns for MotifCompendium
    match_column_prefix: str = "reference"
    filter_col_flag: str = "flag_remove"

@dataclass
class OutputPaths:
    # Relative output paths for MotifCompendium objects and HTMLs
    mc_full: str = "motifcompendium.mc"
    mc_filtered: str = "motifcompendium_filtered.mc"
    mc_removed: str = "motifcompendium_removed.mc"
    mc_clustered: str = "motifcompendium_clustered.mc"

    mc_avg: str = "motifcompendium_avg.mc"
    mc_avg_filtered: str = "motifcompendium_avg_filtered.mc"
    mc_avg_removed: str = "motifcompendium_avg_removed.mc"
    
    mc_metaavg: str = "motifcompendium_metaavg.mc"
    mc_metaavg_filtered: str = "motifcompendium_metaavg_filtered.mc"
    mc_metaavg_removed: str = "motifcompendium_metaavg_removed.mc"

    mc_subavg: str = "motifcompendium_subavg.mc"
    mc_subavg_filtered: str = "motifcompendium_subavg_filtered.mc"
    mc_subavg_removed: str = "motifcompendium_subavg_removed.mc"

    html_motif_collection: str = "motifcompendium_motif_collection.html"
    html_motif_table: str = "motifcompendium_motif_table.html"
    html_motif_removed: str = "motifcompendium_motif_removed.html"
    
    html_cluster_table: str = "motifcompendium_cluster_table.html"
    html_cluster_removed: str = "motifcompendium_cluster_removed.html"
    html_metacluster_table: str = "motifcompendium_metacluster_table.html"
    html_metacluster_removed: str = "motifcompendium_metacluster_removed.html"
    html_subcluster_table: str = "motifcompendium_subcluster_table.html"
    html_subcluster_removed: str = "motifcompendium_subcluster_removed.html"


@dataclass
class MotifMatchArgs:
    # Parameters for matching vs. reference database
    reference_default: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "HUMAN-JASPAR2024-HOCOMOCOv13.meme.txt",
    )
    max_submotifs: int = 2
    min_score: float = 0.7
    base_threshold: float = 0.88
    composite_threshold: float = 0.7


@dataclass
class ClusterArgs:
    # Parameters for clustering
    algorithm: str = "cpm_leiden"
    algorithm_meta: str = "cpm_leiden"
    algorithm_sub: str = "cpm_leiden"
    algorithm_force: str = "dcc"
    weight_col: str = "num_seqlets"
    max_iter: int = 100
    aggregate_metadata: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("name", "count", "num_motifs"),
        ("num_seqlets", "sum", "num_seqlets"),
        ("model", "concat", "model"),
        ("avg_dist_summit", "average", "avg_dist_summit"),
        ("avg_contrib", "average", "avg_contrib"),
        ("invitro_cluster", "concat", "invitro_cluster"),
        ("target", "concat", "target"),
        ("family", "concat", "family"),
        ("tissue", "concat", "tissue"),
        ("organ", "concat", "organ"),
        ("biosample", "concat", "biosample"),
    ])
    algorithm_kwargs: dict = field(default_factory=lambda: {
        "leiden": {},
        "leidenalg": {},
        "rb_leiden": {},
        "mod_leiden": {},
        "modularity_leiden": {},
        "cpm": {},
        "cpm_leiden": {},
        "cc": {},
        "connected_components": {},
        "dcc": {"density": 1.0,},
        "dense_cc": {"density": 1.0,},
        "spectral": {"k": 100},
    })

@dataclass
class VisualizeArgs:
    # Allow editable HTML table
    editable: bool = True
    # Specify HTML table columns
    html_table_cols: List[str] = field(default_factory=lambda: ["name",
        "best_match_similarity", "best_match_cluster",
        "highest_external_similarity", "highest_external_similarity_motif", "highest_external_similarity_cluster",
        "lowest_internal_similarity", "lowest_internal_similarity_motif1", "lowest_internal_similarity_motif2"] + 
        [col
        for iter in range(MotifMatchArgs.max_submotifs)
        for col in [
            f"{MetadataCols.match_column_prefix}_logo{iter}",
            f"{MetadataCols.match_column_prefix}_name{iter}",
            f"{MetadataCols.match_column_prefix}_score{iter}",
            ]
        ] +
        ["posneg", "num_motifs", "num_seqlets", "avg_dist_summit", "avg_contrib", 
         "invitro_cluster", "target", "tissue", "organ", "system", # biosample,
         ])

@dataclass
class MotifFilterArgs:
    # Motif metrics to be calculated
    motif_metrics: tuple = (
        "motif_entropy",
        "weighted_base_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
        "posneg_inverted",
        "truncated",
    )
    # Motif filters to be applied
    motif_filters: tuple = (
        _FilterArgs(
            name="1_singlepeak",
            metric="motif_entropy",
            operation="<",
            threshold=0.35,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="2_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.73,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="3_noisypeaks",
            metric="weighted_base_entropy",
            operation=">",
            threshold=0.5,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="4_broadsingle",
            metric="posbase_entropy_ratio",
            operation=">",
            threshold=1.4,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="5_gcbias",
            metric="copair_entropy_ratio",
            operation=">",
            threshold=1.4,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="6_dinucrepeat",
            metric="dinuc_entropy_ratio",
            operation=">",
            threshold=1.6,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="7_posneg_inverted",
            metric="posneg_inverted",
            operation="==",
            threshold=True,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="8_truncated",
            metric="truncated",
            operation="==",
            threshold=True,
            override=False,
            apply_motif=True,
            apply_cluster=False,
        ),
    )
    # Override: filter_col_flag AND apply_filter_threshold must both be True, to keep flag True
    override_filters: tuple = (
        _FilterArgs(
            name="base_match",
            metric=f"{MetadataCols.match_column_prefix}_score0",
            operation="<",
            threshold=MotifMatchArgs.base_threshold,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        ),
    ) + tuple(
        _FilterArgs(
            name="composite_match",
            metric=f"{MetadataCols.match_column_prefix}_score{iter}",
            operation="<",
            threshold=MotifMatchArgs.composite_threshold,
            override=True,
            apply_motif=True,
            apply_cluster=True,
        )
        for iter in range(1, MotifMatchArgs.max_submotifs)
    )
    # Repeat strict filters
    strict_filters: tuple = (
        _FilterArgs(
            name="2_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.75,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="3_noisypeaks",
            metric="weighted_base_entropy",
            operation=">",
            threshold=0.55,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="4_broadsingle",
            metric="posbase_entropy_ratio",
            operation=">",
            threshold=1.5,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="5_gcbias",
            metric="copair_entropy_ratio",
            operation=">",
            threshold=1.5,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="6_dinucrepeat",
            metric="dinuc_entropy_ratio",
            operation=">",
            threshold=2.0,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="7_posneg_inverted",
            metric="posneg_inverted",
            operation="==",
            threshold=True,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
        _FilterArgs(
            name="8_truncated",
            metric="truncated",
            operation="==",
            threshold=True,
            override=False,
            apply_motif=True,
            apply_cluster=True,
        ),
    )

    def to_dict(self):
        # Return filters as a dictionary
        return {
            "motif_metrics": list(self.motif_metrics),
            "motif_filters": [f.to_dict() for f in self.motif_filters],
            "override_filters": [f.to_dict() for f in self.override_filters],
            "strict_filters": [f.to_dict() for f in self.strict_filters],
        }