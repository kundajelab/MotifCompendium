import multiprocessing

import numpy as np

from .similarity_core_cpu import compute_similarity_and_align


####################
# PUBLIC FUNCTIONS #
####################
def compute_similarities(
    sims_list, calculations, max_chunk, max_cpus, use_gpu, l2=False
):
    """Computes similarities.

    Description

    Args:
        sims_list:
        calculations:
        max_chunk:
        max_cpus:
        use_gpu:
        l2:

    Returns:
        asdf.

    Notes:
        asdf.
    """
    for sims in sims_list:
        validate_sims(sims)
    if max_chunk is not None:
        chunked_sims_list, chunked_calculations, chunk_map = _chunk_sims_and_calcs(
            sims_list, calculations, max_chunk
        )
        chunked_results = _compute_similarity_and_align_parallel(
            chunked_sims_list, chunked_calculations, max_cpus, use_gpu, l2=l2
        )
        return _reassemble_results(
            calculations, chunked_calculations, chunked_results, chunk_map
        )
    else:
        return _compute_similarity_and_align_parallel(
            sims_list, calculations, max_cpus, use_gpu, l2=l2
        )


#####################
# PRIVATE FUNCTIONS #
#####################
def _chunk_sims_and_calcs(sims_list, calculations, max_chunk):
    """Chunks sims and calculations."""
    # CHUNK MOTIFS
    chunked_sims_list = []
    chunk_map = dict()
    current_idx = 0
    for i, sims in enumerate(sims_list):
        if sims.shape[0] <= max_chunk:
            chunked_sims_list.append(sims)
            chunk_map[i] = [current_idx]
            current_idx += 1
        else:
            sims_chunks = _chunk_axis_0(sims, max_chunk)
            chunk_map_entry = []
            for chunk in sims_chunks:
                chunked_sims_list.append(chunk)
                chunk_map_entry.append(current_idx)
                current_idx += 1
            chunk_map[i] = chunk_map_entry
    # CHUNK CALCULATIONS
    chunked_calculations = []
    for c0, c1 in calculations:
        for c0_chunk in chunk_map[c0]:
            for c1_chunk in chunk_map[c1]:
                chunked_calculations.append((c0_chunk, c1_chunk))
    return chunked_sims_list, chunked_calculations, chunk_map


def _chunk_axis_0(X, chunk_size):
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
    sims_list, calculations, max_cpus, use_gpu, l2=False
):
    """Compute similarities and alignments."""
    if max_cpus is None:
        if use_gpu:
            # SINGLE GPU CALCULATIONS
            from .similarity_core_gpu import gpu_compute_similarity_and_align

            return [
                gpu_compute_similarity_and_align(
                    sims_list[c[0]], sims_list[c[1]], l2=l2
                )
                for c in calculations
            ]
        else:
            # SINGLE CPU CALCULATIONS
            return [
                compute_similarity_and_align(sims_list[c[0]], sims_list[c[1]], l2=l2)
                for c in calculations
            ]
    else:
        if use_gpu:
            # MULTI-GPU CALCULATIONS
            print("max_cpu and use_gpu are incompatible options")
            assert False
        else:
            # MULTI-CPU CALCULATIONS
            inputs = [(sims_list[c[0]], sims_list[c[1]], l2) for c in calculations]
            num_processes = min(
                max_cpus, multiprocessing.cpu_count()
            )  # don't use more CPUs than available
            with multiprocessing.Pool(processes=num_processes) as p:
                return p.starmap(compute_similarity_and_align, inputs)


def _reassemble_results(calculations, chunked_calculations, chunked_results, chunk_map):
    """Reassemble chunked results."""
    chunked_calculations_revmap = {c: i for i, c in enumerate(chunked_calculations)}
    results = []
    for c0, c1 in calculations:
        sim_block = []
        fb_block = []
        ali_block = []
        for c0_chunk in chunk_map[c0]:
            sim_block_row = []
            fb_block_row = []
            ali_block_row = []
            for c1_chunk in chunk_map[c1]:
                sim, fb, ali = chunked_results[
                    chunked_calculations_revmap[(c0_chunk, c1_chunk)]
                ]
                sim_block_row.append(sim)
                fb_block_row.append(fb)
                ali_block_row.append(ali)
            sim_block.append(sim_block_row)
            fb_block.append(fb_block_row)
            ali_block.append(ali_block_row)
        sim = np.block(sim_block)
        fb = np.block(fb_block)
        ali = np.block(ali_block)
        results.append((sim, fb, ali))
    assert len(results) == len(calculations)
    return results
