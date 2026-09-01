import h5py
import pandas as pd

from MotifCompendium import MotifCompendium as MotifCompendiumClass
import MotifCompendium.utils.motif as utils_motif


def export_modisco(
    mc: MotifCompendiumClass,
    name_col: str,
    save_loc: str,
) -> None:
    """Exports a MotifCompendium object in the Modisco file format.

    Exports a MotifCompendium into an h5py file that matches the structure of Modisco
    outputs. Each motif in the MotifCompendium becomes a pattern in the Modisco output.

    Args:
        mc: The MotifCompendium to export.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the Modisco h5py to.

    Note:
        Motif names cannot have slashes (/) in them.
        The resultant h5py file can be fed directly into FiNeMo.
    """
    # Check name cols are unique
    if len(mc) != len(mc[name_col].unique()):
        raise ValueError("Motif names must be unique!")
    pos_neg_series = pd.Series(utils_motif.motif_posneg_sum(mc.motifs))
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = mc.motifs.shape[1]
        for pos_neg in ["pos", "neg"]:
            if pos_neg in pos_neg_series.values:
                # Metapattern: Pos, Neg
                metapattern_group = f.create_group(f"{pos_neg}_patterns")
                mc_posneg = mc[pos_neg_series == pos_neg]
                motifs_posneg = mc_posneg.motifs
                pattern_names = mc_posneg[name_col].tolist()
                # Pattern
                for i in range(len(mc_posneg)):
                    name = f"{pattern_names[i]}"
                    if "/" in name:
                        raise ValueError("Motif names cannot have slashes (/) in them!")
                    motif = motifs_posneg[i]
                    pattern_group = metapattern_group.create_group(name)
                    pattern_group.create_dataset("contrib_scores", data=motif)


def cluster_average_and_export_modisco(
    mc: MotifCompendiumClass,
    cluster_name: str,
    save_loc: str,
    *,
    export_subpatterns: bool = False,
    weight_col: str | None = None,
) -> None:
    """Exports cluster average motifs in the Modisco file format.

    Exports a MotifCompendium into an h5py file that matches the structure of Modisco
    outputs. A clustering is specified, and the cluster averages each become a pattern
    in the Modisco output. Optionally, each motif in the MotifCompendium can become a
    subpattern of the cluster it is a part of.

    Args:
        mc: The MotifCompendium to export.
        cluster_name: The motif clustering to group motifs by.
        save_loc: The location to save the Modisco h5py to.
        export_subpatterns: Whether or not to export the individual motifs as
          subpatterns under the cluster average patterns.
        weight_col: The name of the metadata column to be used to weight motifs when
          computing motif averages. The data in the weight_col should be numeric.

    Note:
        Cluster names cannot have slashes (/) in them! If export_subpatterns is True,
          then motif names (taken from the "name" column) cannot have slashes (/) in
          them!
        The resultant h5py file can be fed directly into FiNeMo.
    """
    if export_subpatterns and "name" not in mc.columns():
        raise KeyError(
            "If export_subpatterns is True, then the MotifCompendium must have a 'name' column."
        )
    mc_avg = mc.cluster_averages(
        clustering=cluster_name,
        aggregations=[],
        weight_col=weight_col,
    )
    mc_avg.sort("source_cluster", inplace=True)
    pos_neg_series = pd.Series(utils_motif.motif_posneg_sum(mc_avg.motifs))
    with h5py.File(save_loc, "w") as f:
        f.attrs["window_size"] = mc_avg.motifs.shape[1]
        for pos_neg in ["pos", "neg"]:
            if pos_neg in pos_neg_series.values:
                # Metapattern: Pos, Neg
                metapattern_group = f.create_group(f"{pos_neg}_patterns")
                mc_avg_posneg = mc_avg[pos_neg_series == pos_neg]
                pattern_names = mc_avg_posneg["source_cluster"].tolist()
                if len(pattern_names) != len(set(pattern_names)):
                    raise ValueError("Cluster names must be unique!")
                # Pattern
                for i in range(len(mc_avg_posneg)):
                    pattern_name = f"{pattern_names[i]}"
                    if "/" in pattern_name:
                        raise ValueError(
                            "Cluster names cannot have slashes (/) in them!"
                        )
                    avg_motif = mc_avg_posneg.motifs[i]
                    pattern_group = metapattern_group.create_group(pattern_name)
                    pattern_group.create_dataset("contrib_scores", data=avg_motif)
                    # Subpatterns
                    if export_subpatterns:
                        mc_i = mc[mc[cluster_name] == pattern_name]
                        subpattern_names_i = mc_i["name"].tolist()
                        if len(subpattern_names_i) != len(set(subpattern_names_i)):
                            raise ValueError("Motif names must be unique!")
                        for j in range(len(mc_i)):
                            subpattern_name = f"{subpattern_names_i[j]}"
                            if "/" in subpattern_name:
                                raise ValueError(
                                    "Motif names cannot have slashes (/) in them!"
                                )
                            subpattern_group = pattern_group.create_group(
                                subpattern_name
                            )
                            motif = mc_i.motifs[j]
                            subpattern_group.create_dataset(
                                "contrib_scores", data=motif
                            )


def export_meme(
    mc: MotifCompendiumClass, name_col: str, save_loc: str
) -> None:
    """Exports a MotifCompendium in the MEME file format.

    Exports a MotifCompendium into a MEME file format with each motif in the
      MotifCompendium becoming a motif in the MEME output.

    Args:
        mc: The MotifCompendium to export.
        name_col: The column in the MotifCompendium to name the motifs by.
        save_loc: The location to save the MEME file to.
    """
    motif_names = mc[name_col].tolist()
    num_seqlets = None
    if "num_seqlets" in mc.columns():
        num_seqlets = mc["num_seqlets"].tolist()
    # Write MEME file
    with open(save_loc, "w") as f:
        f.write("MEME version 4\n")
        f.write(f"ALPHABET= ACGT\n")
        f.write(f"strands: +\n")
        f.write(f"Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")
        for i in range(len(mc)):
            name = motif_names[i]
            motif = mc.motifs[i]
            # Remove empty flanks
            motif = utils_motif.trim_motif(motif, 0)  # Remove zero flanks
            # Write motif
            f.write(f"\nMOTIF {name}\n")
            motif_size_line = f"letter-probability matrix: alength= {motif.shape[1]} w= {motif.shape[0]}"
            if num_seqlets is not None:
                motif_size_line += f" nsites= {num_seqlets[i]}\n"
            else:
                motif_size_line += "\n"
            f.write(motif_size_line)
            for j in range(motif.shape[0]):
                f.write(" ".join([f"{x:.6f}" for x in motif[j, :]]) + "\n")
