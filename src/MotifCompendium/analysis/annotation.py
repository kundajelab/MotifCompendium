from typing import Literal

import numpy as np

from MotifCompendium import MotifCompendium as MotifCompendiumClass
import MotifCompendium.utils.loader as utils_loader
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.similarity as utils_similarity


AnnotationType = Literal["single", "homocomposite", "heterocomposite", "all"]


def annotate_from_compendium(
    mc: MotifCompendiumClass,
    reference_mc: MotifCompendiumClass,
    reference_label_col: str = "name",
    save_col_prefix: str = "match",
    annotation_type: AnnotationType = "all",
    *,  # Keyword-only arguments
    min_match_similarity: float = 0.8,
    sufficient_match_similarity: float = 0.95,
    min_monomeric_match_score: float = 0.6,
    multi_match: int = 1,
    max_composite_size: int = 2,
    label_unsigned: bool = False,
    save_images: bool = True,
    logo_plotting_kws: dict = {"ic_scale": True, "trim": 0},
) -> None:
    """Annotate motifs using a reference MotifCompendium with already labeled motifs.

    Perform auto-annotation of motifs in a MotifCompendium using already labeled motifs
    in a reference MotifCompendium. The annotation is performed by finding similar motifs
    or by building similar composite motifs using motifs in the reference
    MotifCompendium. Annotation relies on motif similarity calculations and benefits from
    GPU acceleration. The provided MotifCompendium is modified in-place to add annotation
    information in new columns.

    Args:
        mc: MotifCompendium to annotate.
        reference_mc: MotifCompendium to use as reference for annotation. Must have a
          column with labels for each motif.
        reference_label_col: The column in reference_mc to use for labels.
        save_col_prefix: The prefix to use for saving annotation information in mc.
          Columns "{save_col_prefix}_motifs_{i}" and "{save_col_prefix}_similarity_{i}"
          and, if save_images is True, "{save_col_prefix}_logo_{i}" will be added for
          each multimatch i. The columns will contain the annotation name, the annotation
          similarity score, and the annotation logo, respectively. If the columns already
          exist, they will be overwritten.
        annotation_type: The type of annotation to perform. "single" will only annotate
          based on the closest single reference motif. "homocomposite" will annotate
          motifs as a homocomposite of multiple copies of the same reference motif.
          "heterocomposite" will annotate motifs as a heterocomposite of multiple motifs
          (this can include multiple copies of the same motif and may capture
          homocomposite motifs). "all" will first look to annotate motifs as single
          reference motifs and will attempt to annotate unlabeled motifs as homocomposite
          and will attempt to annotate unlabeled motifs as heterocomposite.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity at which a match is considered
          sufficient and no further annotation is needed. This is used in composite
          matching to determine when no further motifs need to be added to the composite
          and in "all" annotation to determine if a motif that has been annotated as a
          single/homocomposite match has been sufficiently annotated or needs further
          consideration as a homocomposite/heterocomposite match.
        min_monomeric_match_score: The minimum match score that a monomer must have to be
          able to be added to a composite.
        multi_match: The number of possible motif matches to save for each motif. If
          annotation_type is "homocomposite", each multi_match will be composed of a
          different motif. If annotation_type is "heterocomposite", each multi_match will
          have a different seed motif. If annotation_type is "all", multi_match will be
          applied at each stage and the highest similarity matches across all stages will
          be saved.
        max_composite_size: The maximum number of motifs to include in a composite motif.
        label_unsigned: Whether to label motifs in an unsigned way. If True, the absolute
          value of the motifs and reference motifs are used for annotation. This means
          that a negative motif can be annotated by a positive reference motif.
        save_images: Whether to save images of the matches in addition to the match
          labels. If True, the logos of the matches will be a saved image with the name
          "{save_col_prefix}_logo_{i}" for each multimatch i.
        logo_plotting_kws: A dictionary of keyword arguments to pass to the .add_logos()
          method that specify how motifs should be plotted. This will only be used if
          save_images is True. Please refer to the MotifCompendium.add_logos() method for
          more details.
    """
    # Check arguments
    if not isinstance(mc, MotifCompendiumClass):
        raise ValueError("mc must be a MotifCompendium object")
    if not isinstance(reference_mc, MotifCompendiumClass):
        raise ValueError("reference_mc must be a MotifCompendium object")
    # Extract reference motifs and reference labels
    reference_motifs = reference_mc.motifs
    reference_labels = reference_mc[reference_label_col].tolist()
    # Annotate motifs
    annotate_from_labeled_motifs(
        mc,
        reference_motifs,
        reference_labels,
        save_col_prefix,
        annotation_type,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        min_monomeric_match_score=min_monomeric_match_score,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
        save_images=save_images,
        logo_plotting_kws=logo_plotting_kws,
    )


def annotate_from_pfm_file(
    mc: MotifCompendiumClass,
    pfm_file: str,
    save_col_prefix: str = "match",
    annotation_type: AnnotationType = "all",
    *,  # Keyword-only arguments
    min_match_similarity: float = 0.8,
    sufficient_match_similarity: float = 0.95,
    min_monomeric_match_score: float = 0.6,
    multi_match: int = 1,
    max_composite_size: int = 2,
    label_unsigned: bool = False,
    save_images: bool = True,
    logo_plotting_kws: dict = {"ic_scale": True, "trim": 0},
) -> None:
    """Annotate motifs using a PFM file of reference motifs.

    Perform auto-annotation of motifs in a MotifCompendium using reference motifs
    specified in a PFM file. The annotation is performed by finding similar motifs or by
    building similar composite motifs using motifs in the PFM file. Annotation relies on
    motif similarity calculations and benefits from GPU acceleration. The provided
    MotifCompendium is modified in-place to add annotation information in new columns.

    Args:
        mc: MotifCompendium to annotate.
        pfm_file: The path to the PFM file containing reference motifs. This file must be
          in the PFM or MEME file format.
        save_col_prefix: The prefix to use for saving annotation information in mc.
          Columns "{save_col_prefix}_motifs_{i}" and "{save_col_prefix}_similarity_{i}"
          and, if save_images is True, "{save_col_prefix}_logo_{i}" will be added for
          each multimatch i. The columns will contain the annotation name, the annotation
          similarity score, and the annotation logo, respectively. If the columns already
          exist, they will be overwritten.
        annotation_type: The type of annotation to perform. "single" will only annotate
          based on the closest single reference motif. "homocomposite" will annotate
          motifs as a homocomposite of multiple copies of the same reference motif.
          "heterocomposite" will annotate motifs as a heterocomposite of multiple motifs
          (this can include multiple copies of the same motif and may capture
          homocomposite motifs). "all" will first look to annotate motifs as single
          reference motifs and will attempt to annotate unlabeled motifs as homocomposite
          and will attempt to annotate unlabeled motifs as heterocomposite.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity at which a match is considered
          sufficient and no further annotation is needed. This is used in composite
          matching to determine when no further motifs need to be added to the composite
          and in "all" annotation to determine if a motif that has been annotated as a
          single/homocomposite match has been sufficiently annotated or needs further
          consideration as a homocomposite/heterocomposite match.
        min_monomeric_match_score: The minimum match score that a monomer must have to be
          able to be added to a composite.
        multi_match: The number of possible motif matches to save for each motif. If
          annotation_type is "homocomposite", each multi_match will be composed of a
          different motif. If annotation_type is "heterocomposite", each multi_match will
          have a different seed motif. If annotation_type is "all", multi_match will be
          applied at each stage and the highest similarity matches across all stages will
          be saved.
        max_composite_size: The maximum number of motifs to include in a composite motif.
        label_unsigned: Whether to label motifs in an unsigned way. If True, the absolute
          value of the motifs and reference motifs are used for annotation. This means
          that a negative motif can be annotated by a positive reference motif.
        save_images: Whether to save images of the matches in addition to the match
          labels. If True, the logos of the matches will be a saved image with the name
          "{save_col_prefix}_logo_{i}" for each multimatch i.
        logo_plotting_kws: A dictionary of keyword arguments to pass to the .add_logos()
          method that specify how motifs should be plotted. This will only be used if
          save_images is True. Please refer to the MotifCompendium.add_logos() method for
          more details.
    """
    # Check arguments
    if not isinstance(mc, MotifCompendiumClass):
        raise ValueError("mc must be a MotifCompendium object")
    # Extract reference motifs and reference labels
    try:
        reference_motifs, reference_labels = utils_loader.load_pfm(
            pfm_file, motif_length=mc.motifs.shape[1]
        )
    except Exception as e:
        raise ValueError(f"Error loading PFM file: {e}")
    # Annotate motifs
    annotate_from_labeled_motifs(
        mc,
        reference_motifs,
        reference_labels,
        save_col_prefix,
        annotation_type,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        min_monomeric_match_score=min_monomeric_match_score,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
        save_images=save_images,
        logo_plotting_kws=logo_plotting_kws,
    )


def annotate_from_labeled_motifs(
    mc: MotifCompendiumClass,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    save_col_prefix: str = "match",
    annotation_type: AnnotationType = "all",
    *,  # Keyword-only arguments
    min_match_similarity: float = 0.8,
    sufficient_match_similarity: float = 0.95,
    min_monomeric_match_score: float = 0.6,
    multi_match: int = 1,
    max_composite_size: int = 2,
    label_unsigned: bool = False,
    save_images: bool = True,
    logo_plotting_kws: dict = {"ic_scale": True, "trim": 0},
) -> None:
    """Annotate motifs using a labeled set of motifs.

    Perform auto-annotation of motifs in a MotifCompendium using a provided set of
    reference motifs. The annotation is performed by finding similar motifs or by
    building similar composite motifs using the reference motifs. Annotation relies on
    motif similarity calculations and benefits from GPU acceleration. The provided
    MotifCompendium is modified in-place to add annotation information in new columns.

    Args:
        mc: MotifCompendium to annotate.
        reference_motifs: A (M, L', 4) motif stack of M reference motifs used for
          annotation. The length can be different from the motifs in the provided
          MotifCompendium.
        reference_labels: A list of length M containing the labels for the reference
          motifs.
        save_col_prefix: The prefix to use for saving annotation information in mc.
          Columns "{save_col_prefix}_motifs_{i}" and "{save_col_prefix}_similarity_{i}"
          and, if save_images is True, "{save_col_prefix}_logo_{i}" will be added for
          each multimatch i. The columns will contain the annotation name, the annotation
          similarity score, and the annotation logo, respectively. If the columns already
          exist, they will be overwritten.
        annotation_type: The type of annotation to perform. "single" will only annotate
          based on the closest single reference motif. "homocomposite" will annotate
          motifs as a homocomposite of multiple copies of the same reference motif.
          "heterocomposite" will annotate motifs as a heterocomposite of multiple motifs
          (this can include multiple copies of the same motif and may capture
          homocomposite motifs). "all" will first look to annotate motifs as single
          reference motifs and will attempt to annotate unlabeled motifs as homocomposite
          and will attempt to annotate unlabeled motifs as heterocomposite.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity at which a match is considered
          sufficient and no further annotation is needed. This is used in composite
          matching to determine when no further motifs need to be added to the composite
          and in "all" annotation to determine if a motif that has been annotated as a
          single/homocomposite match has been sufficiently annotated or needs further
          consideration as a homocomposite/heterocomposite match.
        min_monomeric_match_score: The minimum match score that a monomer must have to be
          able to be added to a composite.
        multi_match: The number of possible motif matches to save for each motif. If
          annotation_type is "homocomposite", each multi_match will be composed of a
          different motif. If annotation_type is "heterocomposite", each multi_match will
          have a different seed motif. If annotation_type is "all", multi_match will be
          applied at each stage and the highest similarity matches across all stages will
          be saved.
        max_composite_size: The maximum number of motifs to include in a composite motif.
        label_unsigned: Whether to label motifs in an unsigned way. If True, the absolute
          value of the motifs and reference motifs are used for annotation. This means
          that a negative motif can be annotated by a positive reference motif.
        save_images: Whether to save images of the matches in addition to the match
          labels. If True, the logos of the matches will be a saved image with the name
          "{save_col_prefix}_logo_{i}" for each multimatch i.
        logo_plotting_kws: A dictionary of keyword arguments to pass to the .add_logos()
          method that specify how motifs should be plotted. This will only be used if
          save_images is True. Please refer to the MotifCompendium.add_logos() method for
          more details.
    """
    # Check arguments
    if not isinstance(mc, MotifCompendiumClass):
        raise ValueError("mc must be a MotifCompendium object")
    utils_motif.validate_motif_stack(reference_motifs)
    # Extract motifs
    motifs = mc.motifs
    # Resize motifs
    max_length = max(motifs.shape[1], reference_motifs.shape[1])
    motifs = utils_motif.resize_motif(motifs, max_length)
    reference_motifs = utils_motif.resize_motif(reference_motifs, max_length)
    # Annotate motifs
    match annotation_type:
        case "single":
            match_triples = _annotate_single(
                motifs,
                reference_motifs,
                reference_labels,
                min_match_similarity,
                multi_match,
                label_unsigned,
            )
        case "homocomposite":
            match_triples = _annotate_homocomposite(
                motifs,
                reference_motifs,
                reference_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case "heterocomposite":
            match_triples = _annotate_heterocomposite(
                motifs,
                reference_motifs,
                reference_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case "all":
            match_triples = _annotate_all(
                motifs,
                reference_motifs,
                reference_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case _:
            raise ValueError(f"Invalid annotation_type: {annotation_type}")
    # Save match information
    for i, (match_motifs, match_label, match_similarity) in enumerate(match_triples):
        mc[f"{save_col_prefix}_motifs_{i}"] = match_label
        mc[f"{save_col_prefix}_similarity_{i}"] = match_similarity
        if save_images:
            mc.add_logos(
                match_motifs,
                f"{save_col_prefix}_logo_{i}",
                logo_plotting_kws=logo_plotting_kws,
            )


def annotate_with_direct_motifs(
    mc: MotifCompendiumClass,
    direct_motifs: np.ndarray,
    direct_labels: list[str],
    other_motifs: np.ndarray,
    other_labels: list[str],
    save_col_prefix: str = "match",
    annotation_type: AnnotationType = "all",
    *,  # Keyword-only arguments
    min_match_similarity: float = 0.8,
    sufficient_match_similarity: float = 0.95,
    min_monomeric_match_score: float = 0.6,
    multi_match: int = 1,
    max_composite_size: int = 2,
    label_unsigned: bool = False,
    save_images: bool = True,
    logo_plotting_kws: dict = {"ic_scale": True, "trim": 0},
):
    """Annotate motifs using a labeled set of direct motifs in addition to other motifs.

    Perform auto-annotation of motifs in a MotifCompendium using a provided set of
    reference motifs, split into direct motifs and other motifs. Direct motifs are motifs
    that every match should include some component of. Single matches will only come from
    direct motifs, homocomposite matches will only be made of direct motifs, and
    heterocomposite matches will be made of at least one direct motif but can also
    include any number of additional direct or other motifs. The annotation is performed
    by finding similar motifs or by building similar composite motifs using the reference
    motifs. Annotation relies on motif similarity calculations and benefits from GPU
    acceleration. The provided MotifCompendium is modified in-place to add annotation
    information in new columns.

    Args:
        mc: MotifCompendium to annotate.
        direct_motifs: A (M, L', 4) motif stack of M direct reference motifs used for
          annotation. The length can be different from the motifs in the provided
          MotifCompendium or in the other_motifs set.
        direct_labels: A list of length M containing the labels for the direct
          motifs.
        other_motifs: A (M', L'', 4) motif stack of M' other reference motifs used for
          annotation. The length can be different from the motifs in the provided
          MotifCompendium or in the direct_motifs set.
        other_labels: A list of length M' containing the labels for the other
          motifs.
        save_col_prefix: The prefix to use for saving annotation information in mc.
          Columns "{save_col_prefix}_motifs_{i}" and "{save_col_prefix}_similarity_{i}"
          and, if save_images is True, "{save_col_prefix}_logo_{i}" will be added for
          each multimatch i. The columns will contain the annotation name, the annotation
          similarity score, and the annotation logo, respectively. If the columns already
          exist, they will be overwritten.
        annotation_type: The type of annotation to perform. "single" will only annotate
          based on the closest single direct motif. "homocomposite" will annotate motifs
          as a homocomposite of multiple copies of the same direct motif.
          "heterocomposite" will annotate motifs as a heterocomposite of multiple motifs
          which must include at least one direct motif but can contain other direct or
          other reference motifs. "all" will first look to annotate motifs as single
          reference motifs and will attempt to annotate unlabeled motifs as homocomposite
          and will attempt to annotate unlabeled motifs as heterocomposite.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity at which a match is considered
          sufficient and no further annotation is needed. This is used in composite
          matching to determine when no further motifs need to be added to the composite
          and in "all" annotation to determine if a motif that has been annotated as a
          single/homocomposite match has been sufficiently annotated or needs further
          consideration as a homocomposite/heterocomposite match.
        min_monomeric_match_score: The minimum match score that a monomer must have to be
          able to be added to a composite.
        multi_match: The number of possible motif matches to save for each motif. If
          annotation_type is "homocomposite", each multi_match will be composed of a
          different direct motif. If annotation_type is "heterocomposite", each
          multi_match will have a different seed direct motif. If annotation_type is
          "all", multi_match will be applied at each stage and the highest similarity
          matches across all stages will be saved.
        max_composite_size: The maximum number of motifs to include in a composite motif.
        label_unsigned: Whether to label motifs in an unsigned way. If True, the absolute
          value of the motifs and reference motifs are used for annotation. This means
          that a negative motif can be annotated by a positive reference motif.
        save_images: Whether to save images of the matches in addition to the match
          labels. If True, the logos of the matches will be a saved image with the name
          "{save_col_prefix}_logo_{i}" for each multimatch i.
        logo_plotting_kws: A dictionary of keyword arguments to pass to the .add_logos()
          method that specify how motifs should be plotted. This will only be used if
          save_images is True. Please refer to the MotifCompendium.add_logos() method for
          more details.
    """
    # Check arguments
    if not isinstance(mc, MotifCompendiumClass):
        raise ValueError("mc must be a MotifCompendium object")
    utils_motif.validate_motif_stack(direct_motifs)
    utils_motif.validate_motif_stack(other_motifs)
    # Extract motifs
    motifs = mc.motifs
    # Resize motifs
    max_length = max(motifs.shape[1], direct_motifs.shape[1], other_motifs.shape[1])
    motifs = utils_motif.resize_motif(motifs, max_length)
    direct_motifs = utils_motif.resize_motif(direct_motifs, max_length)
    other_motifs = utils_motif.resize_motif(other_motifs, max_length)
    # Annotate motifs
    match annotation_type:
        case "single":
            match_triples = _annotate_single(
                motifs,
                direct_motifs,
                direct_labels,
                min_match_similarity,
                multi_match,
                label_unsigned,
            )
        case "homocomposite":
            match_triples = _annotate_homocomposite(
                motifs,
                direct_motifs,
                direct_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case "heterocomposite":
            match_triples = _annotate_heterocomposite_direct(
                motifs,
                direct_motifs,
                direct_labels,
                other_motifs,
                other_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case "all":
            match_triples = _annotate_all_direct(
                motifs,
                direct_motifs,
                direct_labels,
                other_motifs,
                other_labels,
                min_match_similarity,
                sufficient_match_similarity,
                multi_match,
                max_composite_size,
                label_unsigned,
            )
        case _:
            raise ValueError(f"Invalid annotation_type: {annotation_type}")
    # Save match information
    for i, (match_motifs, match_label, match_similarity) in enumerate(match_triples):
        mc[f"{save_col_prefix}_motifs_{i}"] = match_label
        mc[f"{save_col_prefix}_similarity_{i}"] = match_similarity
        if save_images:
            mc.add_logos(
                match_motifs,
                f"{save_col_prefix}_logo_{i}",
                logo_plotting_kws=logo_plotting_kws,
            )


#################################
# INTERNAL ANNOTATION FUNCTIONS #
#################################
def _annotate_single(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    multi_match: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs based on similarity to single reference motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        multi_match: The number of top matches to consider.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE SINGLE ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Compute similarity
    similarity, _, _ = utils_similarity.compute_similarities(
        [motifs, reference_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    # NOTE: right now, with the way unsigned is, flipped sign motifs will have flipped sign reconstructions
    # Identify top matches
    match_idxs, match_similarity = _top_k_per_row(
        similarity, multi_match
    )  # (N, multi_match)
    # Prepare return
    match_triples = []
    for i in range(multi_match):
        # Get match i
        match_idxs_i = match_idxs[:, i]  # (N,)
        match_similarity_i = match_similarity[:, i]  # (N,)
        match_motif_i = reference_motifs[match_idxs_i]  # (N, L, 4)
        # Remove matches below min threshold
        match_mask_i = match_similarity_i >= min_match_similarity  # (N,)
        match_idxs_i[~match_mask_i] = -1
        match_similarity_i[~match_mask_i] = 0
        match_motif_i[~match_mask_i] = 0
        # Get labels
        match_label_i = [reference_labels[x] if x >= 0 else "" for x in match_idxs_i]
        # Save match triple
        match_triples.append((match_motif_i, match_label_i, match_similarity_i))
    # Return
    return match_triples




def _annotate_homocomposite(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    min_monomeric_match_score: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs based on similarity to homocomposites of reference motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        min_monomeric_match_score: The minimum match score that a monomer must have to be
          able to be added to a composite.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE HOMOCOMPOSITE ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Identify first round monomers
    N = motifs.shape[0]
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs, reference_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    match_idxs, _ = _top_k_per_row(similarity, multi_match)  # (N, multi_match)
    # Prepare outputs
    match_triples = []
    # Go through monomers
    for i in range(multi_match):
        # Set up variables
        considering = np.arange(N)
        bad_monomer = np.zeros(N, dtype=bool) # All monomer are initially good
        monomer_idxs_i = match_idxs[:, i]  # Best monomer index for each motif
        monomers_i = reference_motifs[
            monomer_idxs_i
        ].copy()  # Best monomer for each motif
        alignment_rc_i_0 = alignment_rc[
            considering, monomer_idxs_i
        ]  # alignment_rc of best monomer for each motif
        alignment_h_i_0 = alignment_h[
            considering, monomer_idxs_i
        ]  # alignment_h of best monomer for each motif
        labels_i = [
            reference_labels[x] for x in monomer_idxs_i
        ]  # Best monomer label for each motif
        # Build composites
        residual, reconstruction = utils_motif.remove_motif_component(
            motifs,
            monomers_i,
            alignment_rc_i_0,
            alignment_h_i_0,
        )  # Residuals + reconstructions will be used for future iterations
        # Go through composite iterations
        for j in range(1, max_composite_size):
            # Compute current reconstruction similarity + check if sufficient
            considering_reconstruction_similarities = (
                utils_similarity.compute_similarity_prealigned(
                    motifs[considering].copy(),
                    reconstruction[considering].copy(),
                    unsigned=label_unsigned,
                )
            )
            met_sufficient_similarity = (
                considering_reconstruction_similarities >= sufficient_match_similarity
            )
            # Update considering + others that map to considering
            keep_considering = ~(met_sufficient_similarity | bad_monomer)
            considering = considering[keep_considering] # Keep motifs that are not sufficient and still have good monomers
            if len(considering) == 0:
                break
            residual = residual[keep_considering].copy()
            monomers_i = monomers_i[keep_considering].copy()
            monomer_labels_i = [labels_i[x] for x in keep_considering]
            # Get next monomer match
            similarity_i_j, alignment_rc_i_j, alignment_h_i_j = (
                utils_similarity.compute_similarities(
                    [residual, monomers_i], [(0, 1)], unsigned=label_unsigned
                )[0]
            )
            alignment_rc_i_j = np.diag(
                alignment_rc_i_j
            )  # alignment_rc of best monomer for each motif, for this iteration
            alignment_h_i_j = np.diag(
                alignment_h_i_j
            )  # alignment_h of best monomer for each motif, for this iteration
            # Extract monomer
            residual, reconstruction_j, labels_j, bad_monomer = _remove_monomer(
                residual, monomers_i, alignment_rc_i_j, alignment_h_i_j, monomer_labels_i, min_monomeric_match_score
            )
            reconstruction[considering] += reconstruction_j # Update reconstruction to include this iteration's monomer
            for i, x in enumerate(considering):
                labels_i[x] += f" + {labels_j[i]}" # Update label to include this iteration's monomer
        # Check final similarity meets min threshold
        final_similarity = utils_similarity.compute_similarity_prealigned(
            motifs,
            reconstruction,
            unsigned=label_unsigned,
        )
        match_mask_i = final_similarity >= min_match_similarity
        labels_i = [labels_i[x] if match_mask_i[x] else "" for x in range(N)]
        reconstruction[~match_mask_i] = 0
        final_similarity[~match_mask_i] = 0
        # Save match triple
        match_triples.append((reconstruction, labels_i, final_similarity))
    # Re-sort match_triples by similarity, across iterations
    match_triples_sorted = _sort_across_tuples(match_triples)
    # Return
    return match_triples_sorted


def _annotate_heterocomposite(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs based on similarity to heterocomposites of reference motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE HETEROCOMPOSITE ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Identify first round monomers
    N = motifs.shape[0]
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs, reference_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    match_idxs, _ = _top_k_per_row(similarity, multi_match)  # (N, multi_match)
    # Prepare outputs
    match_triples = []
    # Go through monomers
    for i in range(multi_match):
        # Set up variables
        considering = np.arange(N)
        bad_monomer = np.zeros(N, dtype=bool) # All monomer are initially good
        monomer_idxs_i_0 = match_idxs[:, i]  # Best monomer index for each motif
        monomers_i_0 = reference_motifs[
            monomer_idxs_i_0
        ].copy()  # Best monomer for each motif
        alignment_rc_i_0 = alignment_rc[
            considering, monomer_idxs_i_0
        ]  # alignment_rc of best monomer for each motif
        alignment_h_i_0 = alignment_h[
            considering, monomer_idxs_i_0
        ]  # alignment_h of best monomer for each motif
        labels_i = [
            reference_labels[x] for x in monomer_idxs_i_0
        ]  # Best monomer label for each motif
        # Build composites
        residual, reconstruction = utils_motif.remove_motif_component(
            motifs,
            monomers_i_0,
            alignment_rc_i_0,
            alignment_h_i_0,
        )  # Residuals + reconstructions will be used for future iterations
        # Go through composite iterations
        for j in range(1, max_composite_size):
            # Compute current reconstruction similarity + update considering + considering residuals
            considering_reconstruction_similarities = (
                utils_similarity.compute_similarity_prealigned(
                    motifs[considering].copy(),
                    reconstruction[considering].copy(),
                    unsigned=label_unsigned,
                )
            )
            met_sufficient_similarity = (
                considering_reconstruction_similarities >= sufficient_match_similarity
            )
            # Update considering + map others that map to considering
            keep_considering = ~(met_sufficient_similarity | bad_monomer)
            considering = considering[keep_considering]
            if len(considering) == 0:
                break
            residual = residual[keep_considering].copy()
            reconstruction = reconstruction[keep_considering].copy()
            monomer_labels_i


            considering = considering[
                ~met_sufficient_similarity
            ]  # Keep matching for motifs that have not yet reached sufficient similarity
            if len(considering) == 0:
                break
            residual = residual[
                ~met_sufficient_similarity
            ].copy()  # Will be working with the residuals of the motifs that have not yet reached sufficient similarity
            # Try monomer match
            similarity_i_j, alignment_rc_i_j, alignment_h_i_j = (
                utils_similarity.compute_similarities(
                    [residual, reference_motifs], [(0, 1)], unsigned=label_unsigned
                )[0]
            )
            monomer_idxs_i_j = np.argmax(
                similarity_i_j, axis=1
            )  # Best monomer index for each motif, for this iteration
            monomers_i_j = reference_motifs[
                monomer_idxs_i_j
            ].copy()  # Best monomer for each motif, for this iteration
            labels_i_j = [
                reference_labels[x] for x in monomer_idxs_i_j
            ]  # Best monomer label for each motif, for this iteration
            alignment_rc_i_j = alignment_rc_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_rc of best monomer for each motif, for this iteration
            alignment_h_i_j = alignment_h_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_h of best monomer for each motif, for this iteration
            # TODO: some criteria for if this monomer is good enough
            # Get residual + reconstruction for this iteration
            residual, reconstruction_j = utils_motif.remove_motif_component(
                residual,
                monomers_i_j,
                alignment_rc_i_j,
                alignment_h_i_j,
            )
            reconstruction[
                considering
            ] += reconstruction_j  # Update reconstruction to include this iteration's monomer
            for idx, x in enumerate(considering):
                labels_i[
                    x
                ] += f" + {labels_i_j[idx]}"  # Update label to include this iteration's monomer
        # Check final similarity meets min threshold
        final_similarity = utils_similarity.compute_similarity_prealigned(
            motifs,
            reconstruction,
            unsigned=label_unsigned,
        )
        match_mask_i = final_similarity >= min_match_similarity
        labels_i = [labels_i[x] if match_mask_i[x] else "" for x in range(N)]
        reconstruction[~match_mask_i] = 0
        final_similarity[~match_mask_i] = 0
        # Save match triple
        match_triples.append((reconstruction, labels_i, final_similarity))
    # Re-sort match_triples by similarity, across iterations
    match_triples_sorted = _sort_across_tuples(match_triples)
    # Return
    return match_triples_sorted






def _annotate_homocomposite(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs based on similarity to homocomposites of reference motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE HOMOCOMPOSITE ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # First iteration - identify monomers
    N = motifs.shape[0]
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs, reference_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    match_idxs, _ = _top_k_per_row(similarity, multi_match)  # (N, multi_match)
    # Prepare outputs
    match_triples = []
    # Go through monomers
    for i in range(multi_match):
        # Complete remainder of first iteration - set up residual/reconstruction/considering
        monomer_idxs_i = match_idxs[:, i]  # Best monomer index for each motif
        monomers_i = reference_motifs[
            monomer_idxs_i
        ].copy()  # Best monomer for each motif
        labels_i = [
            reference_labels[x] for x in monomer_idxs_i
        ]  # Best monomer label for each motif
        considering = np.arange(N)  # All matches that have not reached sufficiency
        alignment_rc_i_0 = alignment_rc[
            considering, monomer_idxs_i
        ]  # alignment_rc of best monomer for each motif
        alignment_h_i_0 = alignment_h[
            considering, monomer_idxs_i
        ]  # alignment_h of best monomer for each motif
        residual, reconstruction = utils_motif.remove_motif_component(
            motifs,
            monomers_i,
            alignment_rc_i_0,
            alignment_h_i_0,
        )  # Residuals + reconstructions will be used for future iterations
        # Go through composite iterations
        for j in range(1, max_composite_size):
            # Compute current reconstruction similarity + update considering + considering residuals
            considering_reconstruction_similarities = (
                utils_similarity.compute_similarity_prealigned(
                    motifs[considering].copy(),
                    reconstruction[considering].copy(),
                    unsigned=label_unsigned,
                )
            )
            met_sufficient_similarity = (
                considering_reconstruction_similarities >= sufficient_match_similarity
            )
            considering = considering[
                ~met_sufficient_similarity
            ]  # Keep matching for motifs that have not yet reached sufficient similarity
            if len(considering) == 0:
                break
            residual = residual[
                ~met_sufficient_similarity
            ].copy()  # Will be working with the residuals of the motifs that have not yet reached sufficient similarity
            monomers_i = monomers_i[
                ~met_sufficient_similarity
            ].copy()  # Will be working with the monomers of the motifs that have not yet reached sufficient similarity
            # Try monomer match
            similarity_i_j, alignment_rc_i_j, alignment_h_i_j = (
                utils_similarity.compute_similarities(
                    [residual, monomers_i], [(0, 1)], unsigned=label_unsigned
                )[0]
            )
            alignment_rc_i_j = np.diag(
                alignment_rc_i_j
            )  # alignment_rc of best monomer for each motif, for this iteration
            alignment_h_i_j = np.diag(
                alignment_h_i_j
            )  # alignment_h of best monomer for each motif, for this iteration
            # TODO: some criteria for if this monomer is good enough
            # Get residual + reconstruction for this iteration
            residual, reconstruction_j = utils_motif.remove_motif_component(
                residual,
                monomers_i,
                alignment_rc_i_j,
                alignment_h_i_j,
            )
            reconstruction[
                considering
            ] += reconstruction_j  # Update reconstruction to include this iteration's monomer
            for x in considering:
                labels_i[
                    x
                ] += f" + {labels_i[x]}"  # Update label to include this iteration's monomer
        # Check final similarity meets min threshold
        final_similarity = utils_similarity.compute_similarity_prealigned(
            motifs,
            reconstruction,
            unsigned=label_unsigned,
        )
        match_mask_i = final_similarity >= min_match_similarity
        labels_i = [labels_i[x] if match_mask_i[x] else "" for x in range(N)]
        reconstruction[~match_mask_i] = 0
        final_similarity[~match_mask_i] = 0
        # Save match triple
        match_triples.append((reconstruction, labels_i, final_similarity))
    # Re-sort match_triples by similarity, across iterations
    match_triples_sorted = _sort_across_tuples(match_triples)
    # Return
    return match_triples_sorted


def _annotate_heterocomposite(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs based on similarity to heterocomposites of reference motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE HETEROCOMPOSITE ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # First iteration - identify monomers
    N = motifs.shape[0]
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs, reference_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    match_idxs, _ = _top_k_per_row(similarity, multi_match)  # (N, multi_match)
    # Prepare outputs
    match_triples = []
    # Go through monomers
    for i in range(multi_match):
        # Complete remainder of first iteration - set up residual/reconstruction/considering
        monomer_idxs_i_0 = match_idxs[:, i]  # Best monomer index for each motif
        monomers_i_0 = reference_motifs[
            monomer_idxs_i_0
        ].copy()  # Best monomer for each motif
        labels_i = [
            reference_labels[x] for x in monomer_idxs_i_0
        ]  # Best monomer label for each motif
        considering = np.arange(N)  # All matches that have not reached sufficiency
        alignment_rc_i_0 = alignment_rc[
            considering, monomer_idxs_i_0
        ]  # alignment_rc of best monomer for each motif
        alignment_h_i_0 = alignment_h[
            considering, monomer_idxs_i_0
        ]  # alignment_h of best monomer for each motif
        residual, reconstruction = utils_motif.remove_motif_component(
            motifs,
            monomers_i_0,
            alignment_rc_i_0,
            alignment_h_i_0,
        )  # Residuals + reconstructions will be used for future iterations
        # Go through composite iterations
        for j in range(1, max_composite_size):
            # Compute current reconstruction similarity + update considering + considering residuals
            considering_reconstruction_similarities = (
                utils_similarity.compute_similarity_prealigned(
                    motifs[considering].copy(),
                    reconstruction[considering].copy(),
                    unsigned=label_unsigned,
                )
            )
            met_sufficient_similarity = (
                considering_reconstruction_similarities >= sufficient_match_similarity
            )
            considering = considering[
                ~met_sufficient_similarity
            ]  # Keep matching for motifs that have not yet reached sufficient similarity
            if len(considering) == 0:
                break
            residual = residual[
                ~met_sufficient_similarity
            ].copy()  # Will be working with the residuals of the motifs that have not yet reached sufficient similarity
            # Try monomer match
            similarity_i_j, alignment_rc_i_j, alignment_h_i_j = (
                utils_similarity.compute_similarities(
                    [residual, reference_motifs], [(0, 1)], unsigned=label_unsigned
                )[0]
            )
            monomer_idxs_i_j = np.argmax(
                similarity_i_j, axis=1
            )  # Best monomer index for each motif, for this iteration
            monomers_i_j = reference_motifs[
                monomer_idxs_i_j
            ].copy()  # Best monomer for each motif, for this iteration
            labels_i_j = [
                reference_labels[x] for x in monomer_idxs_i_j
            ]  # Best monomer label for each motif, for this iteration
            alignment_rc_i_j = alignment_rc_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_rc of best monomer for each motif, for this iteration
            alignment_h_i_j = alignment_h_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_h of best monomer for each motif, for this iteration
            # TODO: some criteria for if this monomer is good enough
            # Get residual + reconstruction for this iteration
            residual, reconstruction_j = utils_motif.remove_motif_component(
                residual,
                monomers_i_j,
                alignment_rc_i_j,
                alignment_h_i_j,
            )
            reconstruction[
                considering
            ] += reconstruction_j  # Update reconstruction to include this iteration's monomer
            for idx, x in enumerate(considering):
                labels_i[
                    x
                ] += f" + {labels_i_j[idx]}"  # Update label to include this iteration's monomer
        # Check final similarity meets min threshold
        final_similarity = utils_similarity.compute_similarity_prealigned(
            motifs,
            reconstruction,
            unsigned=label_unsigned,
        )
        match_mask_i = final_similarity >= min_match_similarity
        labels_i = [labels_i[x] if match_mask_i[x] else "" for x in range(N)]
        reconstruction[~match_mask_i] = 0
        final_similarity[~match_mask_i] = 0
        # Save match triple
        match_triples.append((reconstruction, labels_i, final_similarity))
    # Re-sort match_triples by similarity, across iterations
    match_triples_sorted = _sort_across_tuples(match_triples)
    # Return
    return match_triples_sorted


def _annotate_all(
    motifs: np.ndarray,
    reference_motifs: np.ndarray,
    reference_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs with single motifs, homocomposites, or heterocomposites.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        reference_motifs: A (M, L, 4) motif stack of reference motifs.
        reference_labels: A list of M labels for each of the reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print("--- ANNOTATE ALL ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(reference_motifs)
    if motifs.shape[1] != reference_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(reference_labels) != reference_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Single annotation
    N = motifs.shape[0]
    motif_shape = motifs[0].shape
    single_match_triples = _annotate_single(
        motifs,
        reference_motifs,
        reference_labels,
        min_match_similarity=min_match_similarity,
        multi_match=multi_match,
        label_unsigned=label_unsigned,
    )  # Check which ones meet sufficiency
    sufficient_matches = (
        single_match_triples[0][2] >= sufficient_match_similarity
    )  # only need to check 0 because already sorted
    considering_idxs = np.arange(N)[~sufficient_matches]
    motifs_to_annotate = motifs[~sufficient_matches].copy()
    # Homocomposite annotation
    homocomposite_match_triples = _annotate_homocomposite(
        motifs_to_annotate,
        reference_motifs,
        reference_labels,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
    )
    # Expand triples to full motif sets
    homocomposite_match_triples_expanded_split = []
    considering_idxs_reverse_map = {idx: i for i, idx in enumerate(considering_idxs)}
    for i in range(N):
        # TODO: SEARCH SORTED
        if i in considering_idxs_reverse_map:
            idx = considering_idxs_reverse_map[i]
            homocomposite_match_triples_expanded_split.append(
                [
                    (
                        homocomposite_match_triples[k][0][idx],
                        homocomposite_match_triples[k][1][idx],
                        homocomposite_match_triples[k][2][idx],
                    )
                    for k in range(len(homocomposite_match_triples))
                ]
            )
        else:
            homocomposite_match_triples_expanded_split.append(
                [
                    (np.zeros(motif_shape), "", 0)
                    for k in range(len(homocomposite_match_triples))
                ]
            )
    homocomposite_match_triples_expanded = [
        (
            np.stack(
                [homocomposite_match_triples_expanded_split[i][k][0] for i in range(N)]
            ),
            [homocomposite_match_triples_expanded_split[i][k][1] for i in range(N)],
            np.array(
                [homocomposite_match_triples_expanded_split[i][k][2] for i in range(N)]
            ),
        )
        for k in range(len(homocomposite_match_triples))
    ]
    # Check which ones meet sufficiency
    sufficient_matches = (
        homocomposite_match_triples[0][2] >= sufficient_match_similarity
    )  # only need to check 0 because already sorted
    considering_idxs = considering_idxs[~sufficient_matches]
    motifs_to_annotate = motifs_to_annotate[~sufficient_matches].copy()
    # Heterocomposite annotation
    heterocomposite_match_triples = _annotate_heterocomposite(
        motifs_to_annotate,
        reference_motifs,
        reference_labels,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
    )
    # Expand triples to full motif sets
    heterocomposite_match_triples_expanded_split = []
    considering_idxs_reverse_map = {idx: i for i, idx in enumerate(considering_idxs)}
    for i in range(N):
        if i in considering_idxs_reverse_map:
            idx = considering_idxs_reverse_map[i]
            heterocomposite_match_triples_expanded_split.append(
                [
                    (
                        heterocomposite_match_triples[k][0][idx],
                        heterocomposite_match_triples[k][1][idx],
                        heterocomposite_match_triples[k][2][idx],
                    )
                    for k in range(len(heterocomposite_match_triples))
                ]
            )
        else:
            heterocomposite_match_triples_expanded_split.append(
                [
                    (np.zeros(motif_shape), "", 0)
                    for k in range(len(heterocomposite_match_triples))
                ]
            )
    heterocomposite_match_triples_expanded = [
        (
            np.stack(
                [
                    heterocomposite_match_triples_expanded_split[i][k][0]
                    for i in range(N)
                ]
            ),
            [heterocomposite_match_triples_expanded_split[i][k][1] for i in range(N)],
            np.array(
                [
                    heterocomposite_match_triples_expanded_split[i][k][2]
                    for i in range(N)
                ]
            ),
        )
        for k in range(len(heterocomposite_match_triples))
    ]
    # Combine all triples
    all_triples = (
        single_match_triples
        + homocomposite_match_triples_expanded
        + heterocomposite_match_triples_expanded
    )
    # Re-sort match_triples by similarity, across iterations
    all_triples_sorted = _sort_across_tuples(all_triples)
    # Return top multi_match triples
    return all_triples_sorted[:multi_match]


def _annotate_heterocomposite_direct(
    motifs: np.ndarray,
    direct_motifs: np.ndarray,
    direct_labels: list[str],
    other_motifs: np.ndarray,
    other_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs as heterocompsites with at least one direct motif.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        direct_motifs: A (M, L, 4) motif stack of direct motifs.
        direct_labels: A list of M labels for each of the direct motifs.
        other_motifs: A (M', L, 4) motif stack of other reference motifs.
        other_labels: A list of M' labels for each of the other reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE HETEROCOMPOSITE DIRECT ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(direct_motifs)
    if motifs.shape[1] != direct_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(direct_labels) != direct_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    utils_motif.validate_motif_stack(other_motifs)
    if motifs.shape[1] != other_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(other_labels) != other_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Create all_reference_motifs
    N = motifs.shape[0]
    all_reference_motifs = np.concatenate([direct_motifs, other_motifs], axis=0)
    all_reference_labels = direct_labels + other_labels
    # First iteration - identify monomers from direct motifs
    similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
        [motifs, direct_motifs], [(0, 1)], unsigned=label_unsigned
    )[0]
    direct_match_idxs, _ = _top_k_per_row(similarity, multi_match)  # (N, multi_match)
    # Prepare outputs
    match_triples = []
    # Go through monomers
    for i in range(multi_match):
        # Complete remainder of first iteration - set up residual/reconstruction/considering
        monomer_idxs_i_0 = direct_match_idxs[:, i]  # Best monomer index for each motif
        monomers_i_0 = direct_motifs[
            monomer_idxs_i_0
        ].copy()  # Best monomer for each motif
        labels_i = [
            direct_labels[x] for x in monomer_idxs_i_0
        ]  # Best monomer label for each motif
        considering = np.arange(N)  # All matches that have not reached sufficiency
        alignment_rc_i_0 = alignment_rc[
            considering, monomer_idxs_i_0
        ]  # alignment_rc of best monomer for each motif
        alignment_h_i_0 = alignment_h[
            considering, monomer_idxs_i_0
        ]  # alignment_h of best monomer for each motif
        residual, reconstruction = utils_motif.remove_motif_component(
            motifs,
            monomers_i_0,
            alignment_rc_i_0,
            alignment_h_i_0,
        )  # Residuals + reconstructions will be used for future iterations
        # TODO: PROPERLY USE MONOMER GOODNESS MATCH

        # Go through composite iterations
        for j in range(1, max_composite_size):
            # Compute current reconstruction similarity + update considering + considering residuals
            considering_reconstruction_similarities = (
                utils_similarity.compute_similarity_prealigned(
                    motifs[considering].copy(),
                    reconstruction[considering].copy(),
                    unsigned=label_unsigned,
                )
            )
            met_sufficient_similarity = (
                considering_reconstruction_similarities >= sufficient_match_similarity
            )
            considering = considering[
                ~met_sufficient_similarity
            ]  # Keep matching for motifs that have not yet reached sufficient similarity
            if len(considering) == 0:
                break
            residual = residual[
                ~met_sufficient_similarity
            ].copy()  # Will be working with the residuals of the motifs that have not yet reached sufficient similarity
            # Try monomer match
            similarity_i_j, alignment_rc_i_j, alignment_h_i_j = (
                utils_similarity.compute_similarities(
                    [residual, all_reference_motifs], [(0, 1)], unsigned=label_unsigned
                )[0]
            )
            monomer_idxs_i_j = np.argmax(
                similarity_i_j, axis=1
            )  # Best monomer index for each motif, for this iteration
            monomers_i_j = all_reference_motifs[
                monomer_idxs_i_j
            ].copy()  # Best monomer for each motif, for this iteration
            labels_i_j = [
                all_reference_labels[x] for x in monomer_idxs_i_j
            ]  # Best monomer label for each motif, for this iteration
            alignment_rc_i_j = alignment_rc_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_rc of best monomer for each motif, for this iteration
            alignment_h_i_j = alignment_h_i_j[
                np.arange(len(considering)), monomer_idxs_i_j
            ]  # alignment_h of best monomer for each motif, for this iteration
            # TODO: some criteria for if this monomer is good enough
            # Get residual + reconstruction for this iteration
            residual, reconstruction_j = utils_motif.remove_motif_component(
                residual,
                monomers_i_j,
                alignment_rc_i_j,
                alignment_h_i_j,
            )
            reconstruction[
                considering
            ] += reconstruction_j  # Update reconstruction to include this iteration's monomer
            for idx, x in enumerate(considering):
                labels_i[
                    x
                ] += f" + {labels_i_j[idx]}"  # Update label to include this iteration's monomer
        # Check final similarity meets min threshold
        final_similarity = utils_similarity.compute_similarity_prealigned(
            motifs,
            reconstruction,
            unsigned=label_unsigned,
        )
        match_mask_i = final_similarity >= min_match_similarity
        labels_i = [labels_i[x] if match_mask_i[x] else "" for x in range(N)]
        reconstruction[~match_mask_i] = 0
        final_similarity[~match_mask_i] = 0
        # Save match triple
        match_triples.append((reconstruction, labels_i, final_similarity))
    # Re-sort match_triples by similarity, across iterations
    match_triples_sorted = _sort_across_tuples(match_triples)
    # Return
    return match_triples_sorted


def _annotate_all_direct(
    motifs: np.ndarray,
    direct_motifs: np.ndarray,
    direct_labels: list[str],
    other_motifs: np.ndarray,
    other_labels: list[str],
    min_match_similarity: float,
    sufficient_match_similarity: float,
    multi_match: int,
    max_composite_size: int,
    label_unsigned: bool,
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Annotate motifs with single/homocomposites/heterocomposites with direct motifs.

    Args:
        motifs: A (N, L, 4) motif stack to annotate.
        direct_motifs: A (M, L, 4) motif stack of direct motifs.
        direct_labels: A list of M labels for each of the direct motifs.
        other_motifs: A (M', L, 4) motif stack of other reference motifs.
        other_labels: A list of M' labels for each of the other reference motifs.
        min_match_similarity: The minimum similarity for a match to be considered valid.
        sufficient_match_similarity: The similarity threshold for a composite to be
          considered sufficient. If a composite motif reaches this similarity while
          max_composite_size has not been reached, no further motifs will be added to the
          composite. Conversely, the composite will continue to grow until it reaches
          sufficient_match_similarity or until it reaches max_composite_size, whichever
          comes first.
        multi_match: The number of top matches to consider.
        max_composite_size: The maximum number of reference motifs to stitch together
          into a composite motif.
        label_unsigned: Whether to annotate motifs in an unsigned manner.

    Returns:
        A list of tuples containing the matched motifs, their labels, and similarities.
          The length of the list is equal to multi_match. The matches are sorted by
          match similarity so that match_triples[i][2] >= match_triples[j][2] for i < j.
    """
    print(f"--- ANNOTATE ALL DIRECT ---")
    # Check arguments
    utils_motif.validate_motif_stack(motifs)
    utils_motif.validate_motif_stack(direct_motifs)
    if motifs.shape[1] != direct_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(direct_labels) != direct_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    utils_motif.validate_motif_stack(other_motifs)
    if motifs.shape[1] != other_motifs.shape[1]:
        raise ValueError("Motifs and reference motifs must have the same length")
    if len(other_labels) != other_motifs.shape[0]:
        raise ValueError(
            "Number of reference labels must match number of reference motifs"
        )
    # Single annotation
    N = motifs.shape[0]
    motif_shape = motifs[0].shape
    single_match_triples = _annotate_single(
        motifs,
        direct_motifs,
        direct_labels,
        min_match_similarity=min_match_similarity,
        multi_match=multi_match,
        label_unsigned=label_unsigned,
    )
    # Check which ones meet sufficiency
    sufficient_matches = (
        single_match_triples[0][2] >= sufficient_match_similarity
    )  # only need to check 0 because already sorted
    considering_idxs = np.arange(N)[~sufficient_matches]
    motifs_to_annotate = motifs[~sufficient_matches].copy()
    # Homocomposite annotation
    homocomposite_match_triples = _annotate_homocomposite(
        motifs_to_annotate,
        direct_motifs,
        direct_labels,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
    )
    # Expand triples to full motif sets
    homocomposite_match_triples_expanded_split = []
    considering_idxs_reverse_map = {idx: i for i, idx in enumerate(considering_idxs)}
    for i in range(N):
        # TODO: SEARCH SORTED
        if i in considering_idxs_reverse_map:
            idx = considering_idxs_reverse_map[i]
            homocomposite_match_triples_expanded_split.append(
                [
                    (
                        homocomposite_match_triples[k][0][idx],
                        homocomposite_match_triples[k][1][idx],
                        homocomposite_match_triples[k][2][idx],
                    )
                    for k in range(len(homocomposite_match_triples))
                ]
            )
        else:
            homocomposite_match_triples_expanded_split.append(
                [
                    (np.zeros(motif_shape), "", 0)
                    for k in range(len(homocomposite_match_triples))
                ]
            )
    homocomposite_match_triples_expanded = [
        (
            np.stack(
                [homocomposite_match_triples_expanded_split[i][k][0] for i in range(N)]
            ),
            [homocomposite_match_triples_expanded_split[i][k][1] for i in range(N)],
            np.array(
                [homocomposite_match_triples_expanded_split[i][k][2] for i in range(N)]
            ),
        )
        for k in range(len(homocomposite_match_triples))
    ]
    # Check which ones meet sufficiency
    sufficient_matches = (
        homocomposite_match_triples[0][2] >= sufficient_match_similarity
    )  # only need to check 0 because already sorted
    considering_idxs = considering_idxs[~sufficient_matches]
    motifs_to_annotate = motifs_to_annotate[~sufficient_matches].copy()
    # Heterocomposite annotation
    heterocomposite_match_triples = _annotate_heterocomposite_direct(
        motifs_to_annotate,
        direct_motifs,
        direct_labels,
        other_motifs,
        other_labels,
        min_match_similarity=min_match_similarity,
        sufficient_match_similarity=sufficient_match_similarity,
        multi_match=multi_match,
        max_composite_size=max_composite_size,
        label_unsigned=label_unsigned,
    )
    # Expand triples to full motif sets
    heterocomposite_match_triples_expanded_split = []
    considering_idxs_reverse_map = {idx: i for i, idx in enumerate(considering_idxs)}
    for i in range(N):
        if i in considering_idxs_reverse_map:
            idx = considering_idxs_reverse_map[i]
            heterocomposite_match_triples_expanded_split.append(
                [
                    (
                        heterocomposite_match_triples[k][0][idx],
                        heterocomposite_match_triples[k][1][idx],
                        heterocomposite_match_triples[k][2][idx],
                    )
                    for k in range(len(heterocomposite_match_triples))
                ]
            )
        else:
            heterocomposite_match_triples_expanded_split.append(
                [
                    (np.zeros(motif_shape), "", 0)
                    for k in range(len(heterocomposite_match_triples))
                ]
            )
    heterocomposite_match_triples_expanded = [
        (
            np.stack(
                [
                    heterocomposite_match_triples_expanded_split[i][k][0]
                    for i in range(N)
                ]
            ),
            [heterocomposite_match_triples_expanded_split[i][k][1] for i in range(N)],
            np.array(
                [
                    heterocomposite_match_triples_expanded_split[i][k][2]
                    for i in range(N)
                ]
            ),
        )
        for k in range(len(heterocomposite_match_triples))
    ]
    # Combine all triples
    all_triples = (
        single_match_triples
        + homocomposite_match_triples_expanded
        + heterocomposite_match_triples_expanded
    )
    # Re-sort match_triples by similarity, across iterations
    all_triples_sorted = _sort_across_tuples(all_triples)
    # Return top multi_match triples
    return all_triples_sorted[:multi_match]


def _top_k_per_row(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the top k values and indices for each row in a 2D array.

    Args:
        X: A (N, M) np.ndarray of values
        k: The number of top values to return for each row

    Returns:
        A tuple of (idxs, values) where idxs is a (N, k) np.ndarray of the indices of the
          top k values for each row and values is a (N, k) np.ndarray of the top k values
          for each row.
    """
    idxs_unsorted = np.argpartition(X, -k, axis=1)[:, -k:]
    X_unsorted = np.take_along_axis(X, idxs_unsorted, axis=1)
    order = np.argsort(X_unsorted, axis=1)[:, ::-1]
    idxs_sorted = np.take_along_axis(idxs_unsorted, order, axis=1)  # (N, k)
    X_sorted = np.take_along_axis(X, idxs_sorted, axis=1)  # (N, k)
    return idxs_sorted, X_sorted


def _sort_across_tuples(
    data: list[tuple[np.ndarray, list[str], np.ndarray]],
) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    """Sort instances across tuples of (motifs, labels, similarities) by similarities."""
    K = len(data)
    N = data[0][0].shape[0]

    data_split = [
        [(data[k][0][i], data[k][1][i], data[k][2][i]) for k in range(K)]
        for i in range(N)
    ]
    data_split_sorted = [
        sorted(x, key=lambda t: t[2], reverse=True) for x in data_split
    ]
    data_sorted_joined = [
        (
            np.stack([data_split_sorted[i][k][0] for i in range(N)]),
            [data_split_sorted[i][k][1] for i in range(N)],
            np.array([data_split_sorted[i][k][2] for i in range(N)]),
        )
        for k in range(K)
    ]

    return data_sorted_joined


def _monomer_match_goodness(old_motif: np.ndarray, new_motif: np.ndarray, match: np.ndarray) -> np.ndarray:
    """A score of how good a monomer match is."""
    old_motif_ic = utils_motif.ic_scale(old_motif)  # (N, L, 4)
    old_motif_importance = np.sum(np.square(old_motif_ic), axis=2)  # (N, L)
    new_motif_ic = utils_motif.ic_scale(new_motif)  # (N, L, 4)
    new_motif_importance = np.sum(np.square(new_motif_ic), axis=2)  # (N, L)
    match_ic = utils_motif.ic_scale(match)  # (N, L, 4)
    match_importance = np.sum(np.square(match_ic), axis=2)  # (N, L)
    old_times_match = np.sum(old_motif_importance * match_importance, axis=1)  # (N,)
    new_times_match = np.sum(new_motif_importance * match_importance, axis=1)  # (N,)
    percent_left = (old_times_match - new_times_match) / old_times_match
    return percent_left


def _pretty_print_motif(motif):
    for i in range(motif.shape[0]):
        print(
            f"pos {i}: A={motif[i, 0]:.2f}, C={motif[i, 1]:.2f}, G={motif[i, 2]:.2f}, T={motif[i, 3]:.2f}"
        )


def _remove_monomer(
        previous_residual: np.ndarray,
        monomers: np.ndarray,
        alignment_rc: np.ndarray,
        alignment_h: np.ndarray,
        monomer_labels: list[str],
        min_monomeric_match_score: float,
):
    potential_residual, potential_reconstruction = utils_motif.remove_motif_component(
        previous_residual,
        monomers,
        alignment_rc,
        alignment_h,
    )
    monomeric_match_score = _monomer_match_goodness(previous_residual, potential_reconstruction, monomers)
    bad_monomer = monomeric_match_score < min_monomeric_match_score
    new_residual = np.where(bad_monomer[:, None, None], previous_residual, potential_residual)
    new_reconstruction = np.where(bad_monomer[:, None, None], 0, potential_reconstruction)
    new_labels = ["" if x else monomer_labels[i] for i, x in enumerate(bad_monomer)]
    return new_residual, new_reconstruction, new_labels, bad_monomer
