import multiprocessing

import numpy as np
from tqdm import tqdm

import MotifCompendium.utils.config as utils_config
from MotifCompendium.utils.motif import validate_motif_stack_similarity
from MotifCompendium.utils.similarity_core import compute_similarity_and_align


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarities(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Performs similarity and calculations between sets of motif stacks.

    Given a list of motif stacks and an instruction of which motif stacks to perform
      pairwise similarity calculations between, this function returns a list of results
      of each one of the specified calculations.

    Args:
        motif_stack_list: A list of np.ndarrays representing stacks of motifs.
        calculations: A list where each element specifies a motif similarity calculation
          to perform. Each element of the list is a tuple of two ints where each int is
          the index of a motif stack in the motif_stack_list. For example, the
          calculation (i, j) says to compute the pairwise similarity between all motifs
          in motif stack i and motif stack j.

    Returns:
        A list of motif calculation result tuples. There is one motif calculation result
          tuple for each calculation specification tuple in calculations. Each motif
          calculation result tuple consists of a similarity matrix, an alignemnt_fr
          matrix, and an alignment_h matrix.
        The maximum size of a motif stack to perform similarity calculations
          on. All motif stacks larger than max_chunk will be chunked into smaller motif
          stacks of size <= max_chunk by _chunk_motif_stacks_and_calcs(). Calculations
          will occur on the smaller chunks and then the results will be reassembled with
          _reassemble_results().
        Whether or not to use GPUs to accelerate computing similarity. If True,
          similarity calculations are carried out by the functions in utils file
          .similarity_core_gpu.py. If False, they will be carried out by the function in
          utils file .similarity_core_cpu.py.
    """
    for motif_stack in motif_stack_list:
        validate_motif_stack_similarity(motif_stack)
    if utils_config.get_max_chunk() != -1:
        (
            chunked_motif_stack_list,
            chunked_calculations,
            chunk_map,
        ) = _chunk_motif_stacks_and_calcs(
            motif_stack_list, calculations, utils_config.get_max_chunk()
        )
        chunked_results = _compute_similarity_and_align_parallel(
            chunked_motif_stack_list,
            chunked_calculations,
        )
        return _reassemble_results(
            calculations, chunked_calculations, chunked_results, chunk_map
        )
    else:
        return _compute_similarity_and_align_parallel(
            motif_stack_list,
            calculations,
        )


def find_most_similar_motif(
    motifs_of_interest: np.ndarray, reference_motifs: np.ndarray
) -> tuple[list[float], list[int]]:
    """Finds the most similar motif given a set of reference motifs.

    For a set of motifs of interest and a set of reference motifs, this function
      calls compute_similarities() to compute the similarity between the two sets of
      motifs. Then, for each motif of interest, it finds the most similar reference
      motif.

    Args:
        motifs_of_interest: A np.ndarray representing a stack of motifs of interest.
        reference_motifs: A np.ndarray representing a stack of reference motifs.

    Returns:
        A tuple of two lists. The first list contains the maximum similarity score
          for each motif of interest. The second list contains the index of the most
          similar reference motif for each motif.
    """
    similarity, _, _ = compute_similarities(
        [motifs_of_interest, reference_motifs], [(0, 1)]
    )[0]
    max_similarity = np.max(similarity, axis=1).tolist()
    max_similarity_idx = np.argmax(similarity, axis=1).tolist()
    return max_similarity, max_similarity_idx


#####################
# PRIVATE FUNCTIONS #
#####################
def _chunk_motif_stacks_and_calcs(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    max_chunk: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[int, int]]:
    """Chunks motifs and calculations."""
    # CHUNK MOTIFS
    chunked_motif_stack_list = []
    chunk_map = dict()
    current_idx = 0
    for i, motif_stack in enumerate(motif_stack_list):
        if motif_stack.shape[0] <= max_chunk:
            chunked_motif_stack_list.append(motif_stack)
            chunk_map[i] = [current_idx]
            current_idx += 1
        else:
            motif_stack_chunks = _chunk_axis_0(motif_stack, max_chunk)
            chunk_map_entry = []
            for chunk in motif_stack_chunks:
                chunked_motif_stack_list.append(chunk)
                chunk_map_entry.append(current_idx)
                current_idx += 1
            chunk_map[i] = chunk_map_entry
    # CHUNK CALCULATIONS
    chunked_calculations = []
    for c0, c1 in calculations:
        for c0_chunk in chunk_map[c0]:
            for c1_chunk in chunk_map[c1]:
                chunked_calculations.append((c0_chunk, c1_chunk))
    return chunked_motif_stack_list, chunked_calculations, chunk_map


def _chunk_axis_0(X: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """Chunks along axis 0."""
    N = X.shape[0]
    X_chunks = []
    for i in range(N // chunk_size):
        X_chunks.append(X[i * chunk_size : (i + 1) * chunk_size])
    if N % chunk_size > 0:
        X_chunks.append(X[(i + 1) * chunk_size :])
    assert len(X_chunks) == int(np.ceil(N / chunk_size))
    return X_chunks


def _compute_similarity_and_align_parallel(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute similarities and alignments."""
    if utils_config.get_use_gpu():
        # SINGLE GPU CALCULATIONS
        if utils_config.get_progress_bar():
            results = []
            for c in tqdm(calculations, desc="computing similarities on GPU..."):
                results.append(
                    compute_similarity_and_align(
                        motif_stack_list[c[0]], motif_stack_list[c[1]]
                    )
                )
            return results
        else:
            return [
                compute_similarity_and_align(motif_stack_list[c[0]], motif_stack_list[c[1]])
                for c in calculations
            ]
    else:
        if utils_config.get_max_cpus() == 1 or len(calculations) == 1:
            # SINGLE CPU CALCULATIONS
            return [
                compute_similarity_and_align(
                    motif_stack_list[c[0]], motif_stack_list[c[1]]
                )
                for c in calculations
            ]
        else:
            # MULTI-CPU CALCULATIONS
            inputs = [
                (motif_stack_list[c[0]], motif_stack_list[c[1]]) for c in calculations
            ]
            num_processes = min(
                utils_config.get_max_cpus(), multiprocessing.cpu_count()
            )  # don't use more CPUs than available
            with multiprocessing.Pool(processes=num_processes) as p:
                return p.starmap(compute_similarity_and_align, inputs)


def _reassemble_results(
    calculations: list[tuple[int, int]],
    chunked_calculations: list[tuple[int, int]],
    chunked_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    chunk_map: dict[int, int],
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """Reassemble chunked results."""
    chunked_calculations_revmap = {c: i for i, c in enumerate(chunked_calculations)}
    results = []
    for c0, c1 in calculations:
        sim_block = []
        rc_block = []
        ali_block = []
        for c0_chunk in chunk_map[c0]:
            sim_block_row = []
            rc_block_row = []
            ali_block_row = []
            for c1_chunk in chunk_map[c1]:
                sim, rc, ali = chunked_results[
                    chunked_calculations_revmap[(c0_chunk, c1_chunk)]
                ]
                sim_block_row.append(sim)
                rc_block_row.append(rc)
                ali_block_row.append(ali)
            sim_block.append(sim_block_row)
            rc_block.append(rc_block_row)
            ali_block.append(ali_block_row)
        sim = np.block(sim_block)
        rc = np.block(rc_block)
        ali = np.block(ali_block)
        results.append((sim, rc, ali))
    assert len(results) == len(calculations)
    return results
