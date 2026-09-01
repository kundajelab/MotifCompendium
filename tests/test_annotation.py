import numpy as np
import pandas as pd

import MotifCompendium
import MotifCompendium.analysis.annotation as analysis_annotation


def test_basic_annotation():
    # Define dummy motifs
    gata = dummy_gata_motif()
    fox = dummy_fox_motif()
    # Define compendium
    motifs = np.zeros((3, 30, 4))
    # Single GATA
    motifs[0, 5:10, :] = gata
    # Double GATA
    motifs[1, 5:10, :] = gata
    motifs[1, 15:20, :] = gata
    # GATA + FOX
    motifs[2, 5:10, :] = gata
    motifs[2, 15:22, :] = fox
    # Build MotifCompendium
    metadata = pd.DataFrame(
        {
            "name": ["Single GATA", "Double GATA", "GATA + FOX"],
        }
    )
    mc = MotifCompendium.build(motifs, metadata, safe=False)
    # Define reference
    references = np.zeros((2, 30, 4))
    references[0, 5:10, :] = gata
    references[1, 15:22, :] = fox

    # Annotate motifs
    analysis_annotation.annotate_from_labeled_motifs(
        mc,
        references,
        reference_labels=["GATA", "FOX"],
    )
    # Check correct
    assert(mc["match_motifs_0"].tolist() == ["GATA", "GATA + GATA", "FOX + GATA"])


def test_annotation_multimatch():
    # Define dummy motifs
    gata = dummy_gata_motif()
    fox = dummy_fox_motif()
    distorted_gata = gata.copy() + 0.1 * np.random.rand(*gata.shape)
    distorted_fox = fox.copy() + 0.1 * np.random.rand(*fox.shape)
    # Define compendium
    motifs = np.zeros((3, 30, 4))
    # Single GATA
    motifs[0, 5:10, :] = gata
    # Double GATA
    motifs[1, 5:10, :] = gata
    motifs[1, 15:20, :] = gata
    # GATA + FOX
    motifs[2, 5:10, :] = gata
    motifs[2, 15:22, :] = fox
    # Build MotifCompendium
    metadata = pd.DataFrame(
        {
            "name": ["Single GATA", "Double GATA", "GATA + FOX"],
        }
    )
    mc = MotifCompendium.build(motifs, metadata, safe=False)
    # Define reference
    references = np.zeros((4, 30, 4))
    references[0, 5:10, :] = gata
    references[1, 15:22, :] = fox
    references[2, 5:10, :] = distorted_gata
    references[3, 15:22, :] = distorted_fox
    # Annotate motifs
    analysis_annotation.annotate_from_labeled_motifs(
        mc,
        references,
        reference_labels=["GATA", "FOX", "Distorted GATA", "Distorted FOX"],
        multi_match=2
    )
    # Check correct
    print(mc)
    print(mc["match_motifs_0"])
    print(mc["match_motifs_1"])
    assert(mc["match_motifs_0"].tolist() == ["GATA", "GATA + GATA", "FOX + GATA"])
    assert(mc["match_motifs_1"].tolist() == ["Distorted GATA", "Distorted GATA + Distorted GATA", "Distorted FOX + GATA"])


def test_annotation_direct():
    # Define dummy motifs
    gata = dummy_gata_motif()
    fox = dummy_fox_motif()
    # Define compendium
    motifs = np.zeros((3, 30, 4))
    # Single GATA
    motifs[0, 5:10, :] = gata
    # Double GATA
    motifs[1, 5:10, :] = gata
    motifs[1, 15:20, :] = gata
    # GATA + FOX
    motifs[2, 5:10, :] = gata
    motifs[2, 15:22, :] = fox
    # Build MotifCompendium
    metadata = pd.DataFrame(
        {
            "name": ["Single GATA", "Double GATA", "GATA + FOX"],
        }
    )
    mc = MotifCompendium.build(motifs, metadata, safe=False)
    # Define reference
    gata_reference = np.zeros((1, 30, 4))
    gata_reference[0, 5:10, :] = gata
    fox_reference = np.zeros((1, 30, 4))
    fox_reference[0, 15:22, :] = fox
    # # Annotate motifs with direct GATA
    # analysis_annotation.annotate_with_direct_motifs(
    #     mc,
    #     gata_reference,
    #     direct_labels=["GATA"],
    #     other_motifs=fox_reference,
    #     other_labels=["FOX"],
    #     save_col_prefix="directgata",
    # )
    # # Check correct
    # print(mc["directgata_motifs_0"])
    # assert(mc["directgata_motifs_0"].tolist() == ["GATA", "GATA + GATA", "GATA + FOX"])
    # Annotate motifs with direct FOX
    analysis_annotation.annotate_with_direct_motifs(
        mc,
        fox_reference,
        direct_labels=["FOX"],
        other_motifs=gata_reference,
        other_labels=["GATA"],
        save_col_prefix="directfox",
    )
    # mc.summary_table_html("/oak/stanford/groups/akundaje/salil512/web/temp/direct_annotation_summary.html",
    #                       ["directgata_logo_0", "directgata_motifs_0", "directgata_similarity_0",
    #                        "directfox_logo_0", "directfox_motifs_0", "directfox_similarity_0"],)
    # Check correct
    print(mc["directfox_motifs_0"])
    print(mc["directfox_similarity_0"])
    assert(mc["directfox_motifs_0"].tolist() == ["", "", "FOX + GATA"])


############################
# Testing helper functions #
############################
def dummy_gata_motif():
    gata = np.zeros((5, 4))
    gata[0, 2] = 1
    gata[1, 0] = 1
    gata[2, 3] = 1
    gata[3, 0] = 1
    gata[4, 0] = 1
    return gata


def dummy_fox_motif():
    fox = np.zeros((7, 4))
    fox[0, 3] = 1
    fox[1, 2] = 1
    fox[2, 3] = 1
    fox[3, 3] = 1
    fox[4, 3] = 1
    fox[5, 0] = 1
    fox[6, 1] = 1
    return fox

