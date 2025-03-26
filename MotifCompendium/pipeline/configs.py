import os
from dataclasses import dataclass
from typing import Union


@dataclass
class FilterArgs:
    name: str
    metric: str
    operation: str
    threshold: Union[float, bool]
    override: bool

@dataclass
class OutputPaths:
    mc_full: str = "motifcompendium.mc"
    mc_filtered: str = "motifcompendium_filtered.mc"
    mc_removed: str = "motifcompendium_removed.mc"
    mc_clustered: str = "motifcompendium_clustered.mc"

    mc_avg: str = "motifcompendium_avg.mc"
    mc_avg_filtered: str = "motifcompendium_avg_filtered.mc"
    mc_avg_removed: str = "motifcompendium_avg_removed.mc"

    html_collection: str = "motifcompendium_collection.html"
    html_table: str = "motifcompendium_table.html"

@dataclass
class MetadataCols:
    match_col_name: str = "reference_name"
    match_col_score: str = "reference_similarity"
    match_col_logo: str = "reference_logo"
    filter_col_flag: str = "flag_remove"

@dataclass
class MotifMatchArgs:
    reference_default: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "JASPAR2024-HOCOMOCOv13.meme.txt"
    )
    max_submotifs: int = 3
    min_score: float = 0.5
    composite_threshold: float = 0.7

@dataclass
class ClusterArgs:
    algorithm: str = "cpm_leiden"
    weight_col: str = "num_seqlets"
    aggregate_metadata: tuple = (
        ("name", "count", "num_motifs"),
        ("num_seqlets", "sum", "num_seqlets"),
        ("model", "concat", "models"),
    )

@dataclass
class VisualizeArgs:
    html_table_cols: tuple = tuple(
        col
        for iter in range(1, MotifMatchArgs.max_submotifs)
        for col in [
            f"{MetadataCols.match_col_logo}{iter}",
            f"{MetadataCols.match_col_name}{iter}",
            f"{MetadataCols.match_col_score}{iter}",
        ]
    ) + (
        "name", "num_motifs", "num_seqlets", "models"
    )

@dataclass
class MotifFilterArgs:
    motif_metrics: tuple = (
        "negpattern_pospeak",
        "motif_entropy",
        "posbase_entropy_ratio",
        "copair_entropy_ratio",
        "dinuc_entropy_ratio",
    )
    motif_filters: tuple = (
        FilterArgs(
            name="1_singlepeak",
            metric="motif_entropy",
            operation="<",
            threshold=0.45,
            override=False,
        ),
        FilterArgs(
            name="2_noisemix",
            metric="motif_entropy",
            operation=">",
            threshold=0.75,
            override=False,
        ),
        FilterArgs(
            name="3_broadsingle",
            metric="posbase_entropy_ratio",
            operation=">",
            threshold=2.0,
            override=False,
        ),
        FilterArgs(
            name="4_gcbias",
            metric="copair_entropy_ratio",
            operation=">",
            threshold=2.0,
            override=False,
        ),
        FilterArgs(
            name="5_dinucrepeat",
            metric="dinuc_entropy_ratio",
            operation=">",
            threshold=4.0,
            override=False,
        ),
        FilterArgs(
            name="6_negpattern_pospeak",
            metric="negpattern_pospeak",
            operation="==",
            threshold=True,
            override=False,
        ),
    )
    # Override: filter_col_flag AND apply_filter_threshold must both be True, to keep flag True
    override_filters: tuple = (
        (FilterArgs(
            name="base_match",
            metric=MetadataCols.match_col_score,
            operation=">",
            threshold=0.9,
            override=True,
        ),)
        + tuple(
            FilterArgs(
                name="composite_match",
                metric=f"{MetadataCols.match_col_score}{iter}",
                operation="<",
                threshold=MotifMatchArgs.composite_threshold,
                override=True,
            ) for iter in range(1, MotifMatchArgs.max_submotifs)
        )
    )