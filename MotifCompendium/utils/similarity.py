import multiprocessing

import numpy as np

from .motif import validate_motif_stack

####################
# PUBLIC FUNCTIONS #
####################
def compute_similarities_and_alignments(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    max_chunk: int | None,
    max_cpus: int | None,
    use_gpu: bool | None,
    sim_type: str | None,
    safe: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Performs similarity and alignment calculations between sets of motif stacks.

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
        max_chunk: The maximum size of a motif stack to perform similarity calculations
          on. All motif stacks larger than max_chunk will be chunked into smaller motif
          stacks of size <= max_chunk by _chunk_motif_stacks_and_calcs(). Calculations
          will occur on the smaller chunks and then the results will be reassembled with
          _reassemble_results().
        max_cpus: The maximum number of CPUs to use for computing similarity.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. If True,
          similarity calculations are carried out by the functions in utils file
          .similarity_core_gpu.py. If False, they will be carried out by the function in
          utils file .similarity_core_cpu.py.
        sim_type: The type of similarity metric to compute: 'l2', 'sqrt', 'jss'

    Returns:
        A list of motif calculation result tuples. There is one motif calculation result
          tuple for each calculation specification tuple in calculations. Each motif
          calculation result tuple consists of a similarity matrix, an alignemnt_fr
          matrix, and an alignment_h matrix.
    """
    if max_chunk is None:
        max_chunk = DEFAULT_MAX_CHUNK
    if max_cpus is None:
        max_cpus = DEFAULT_MAX_CPUS
    if use_gpu is None:
        use_gpu = DEFAULT_USE_GPU
    if sim_type is None:
        sim_type = DEFAULT_SIM_TYPE
    # TODO: FIX EDGE CASE THAT HAS ARISEN DUE TO DEFAULTS
    if use_gpu:
        max_cpus = None
    if safe:
        for motif_stack in motif_stack_list:
            validate_motif_stack(motif_stack)
            
    if max_chunk is not None:
        (
            chunked_motif_stack_list,
            chunked_calculations,
            chunk_map,
        ) = _chunk_motif_stacks_and_calcs(motif_stack_list, calculations, max_chunk)
        chunked_results = _compute_similarity_and_align_parallel(
            chunked_motif_stack_list, chunked_calculations,
            max_cpus, use_gpu, sim_type
        )
        return _reassemble_results(
            calculations, chunked_calculations, chunked_results, chunk_map
        )
    else:
        return _compute_similarity_and_align_parallel(
            motif_stack_list, calculations,
            max_cpus, use_gpu, sim_type
        )


def compute_similarities_known_alignments(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    alignfr_stack_list: list[np.ndarray],
    alignh_stack_list: list[np.ndarray],
    max_chunk: int | None,
    max_cpus: int | None,
    use_gpu: bool | None,
    sim_type: str | None,
) -> np.ndarray:
    """Performs similarity calculations, for known alignments.

    Args:
        motif_stack_list: A list of np.ndarrays representing stacks of motifs.
        calculations: A list of calculation orders, where each element specifies a motif 
          similarity calculation
        alignfr_stack_list: A list of (N, M) np.ndarray representing the forward-reverse alignment
          for each calculation.
        alignh_stack_list: A list of (N, M) np.ndarray representing the horizontal alignment
          for each calculation.
        max_chunk: The maximum size of a motif stack to perform similarity calculations
          on. All motif stacks larger than max_chunk will be chunked into smaller motif
          stacks of size <= max_chunk by _chunk_motif_stacks_and_calcs(). Calculations
          will occur on the smaller chunks and then the results will be reassembled with
          _reassemble_results().
        max_cpus: The maximum number of CPUs to use for computing similarity.
        use_gpu: Whether or not to use GPUs to accelerate computing similarity. If True,
          similarity calculations are carried out by the functions in utils file
          .similarity_core_gpu.py. If False, they will be carried out by the function in
          utils file .similarity_core_cpu.py.
        sim_type: The type of similarity metric to compute: 'l2', 'sqrt', 'jss'

    Returns:
        A (N, N) np.ndarray representing the similarity matrix.
    """
    if max_chunk is None:
        max_chunk = DEFAULT_MAX_CHUNK
    if max_cpus is None:
        max_cpus = DEFAULT_MAX_CPUS
    if use_gpu is None:
        use_gpu = DEFAULT_USE_GPU
    if sim_type is None:
        sim_type = DEFAULT_SIM_TYPE
    # TODO: FIX EDGE CASE THAT HAS ARISEN DUE TO DEFAULTS
    if use_gpu:
        max_cpus = None

    if max_chunk is not None:
        # Chunk motifs, calculations, and alignments
        (
            chunked_motif_stack_list, chunked_calculations,
            chunked_alignfr_stack_list, chunked_alignh_stack_list,
            chunk_map,
        ) = _chunk_motifs_calcs_aligns(
            motif_stack_list, calculations, 
            alignfr_stack_list, alignh_stack_list,
            max_chunk,
        )
        
        chunked_simliarities = _compute_similarity_known_alignment(
            chunked_motif_stack_list, chunked_calculations,
            chunked_alignfr_stack_list, chunked_alignh_stack_list,
            max_cpus, use_gpu, sim_type
        )

        return _reassemble_similarity(
            calculations, chunked_calculations, chunked_simliarities, chunk_map
        )
    else:
        return _compute_similarity_known_alignment(
            motif_stack_list, calculations,
            alignfr_stack_list, alignh_stack_list,
            max_cpus, use_gpu, sim_type
        )


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


def _chunk_motifs_calcs_aligns(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    alignfr_stack_list: list[np.ndarray],
    alignh_stack_list: list[np.ndarray],
    max_chunk: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[list[np.ndarray]], list[list[np.ndarray]], dict[int, int]]:
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
    
    # CHUNK ALIGNMENTS
    chunked_alignfr_stack_list = []
    chunked_alignh_stack_list = []
    for i, (alignfr, alignh) in enumerate(zip(alignfr_stack_list, alignh_stack_list)):
        if alignfr.shape[0] <= max_chunk:
            chunked_alignfr_stack_list.append(alignfr)
            chunked_alignh_stack_list.append(alignh)
        else:
            alignfr_chunks = _chunk_2D(alignfr, max_chunk)
            alignh_chunks = _chunk_2D(alignh, max_chunk)
            for alignfr_chunk, alignh_chunk in zip(alignfr_chunks, alignh_chunks):
                chunked_alignfr_stack_list.append(alignfr_chunk)
                chunked_alignh_stack_list.append(alignh_chunk)

    # CHUNK CALCULATIONS
    chunked_calculations = []
    for c0, c1 in calculations:
        for c0_chunk in chunk_map[c0]:
            for c1_chunk in chunk_map[c1]:
                chunked_calculations.append((c0_chunk, c1_chunk))

    return (chunked_motif_stack_list, chunked_calculations,
        chunked_alignfr_stack_list, chunked_alignh_stack_list,
        chunk_map)


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


def _chunk_2D(X: np.ndarray, chunk_size: int) -> list[list[np.ndarray]]:
    """Chunks along both axes."""
    N = X.shape[0]
    M = X.shape[1]
    X_chunks = []
    for i in range(int(np.ceil(N / chunk_size))):
        Y_chunks = []
        for j in range(int(np.ceil(M / chunk_size))):
            i_end = min(int((i + 1) * chunk_size), N)
            j_end = min(int((j + 1) * chunk_size), M)
            Y_chunks.append(X[i * chunk_size : i_end, j * chunk_size : j_end])
        X_chunks.append(Y_chunks)
    assert len(X_chunks) == int(np.ceil(N / chunk_size))
    return X_chunks
            

def _compute_similarity_and_align_parallel(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    max_cpus: int | None,
    use_gpu: bool,
    sim_type: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute similarities and alignments."""
    if max_cpus is None:
        if use_gpu:
            # SINGLE GPU CALCULATIONS
            from .similarity_core_gpu import gpu_compute_similarity_and_align

            return [
                gpu_compute_similarity_and_align(
                    motif_stack_list[c0], motif_stack_list[c1], sim_type
                )
                for c0, c1 in calculations
            ]
        else:
            # SINGLE CPU CALCULATIONS
            from .similarity_core_cpu import cpu_compute_similarity_and_align

            return [
                cpu_compute_similarity_and_align(
                    motif_stack_list[c0], motif_stack_list[c1], sim_type,
                )
                for c0, c1 in calculations
            ]
    else:
        if use_gpu:
            # MULTI-GPU CALCULATIONS
            print("max_cpu and use_gpu are incompatible options")
            assert False
        else:
            # MULTI-CPU CALCULATIONS
            from .similarity_core_cpu import cpu_compute_similarity_and_align

            inputs = [
                (motif_stack_list[c0], motif_stack_list[c1], sim_type)
                for c0, c1 in calculations
            ]
            num_processes = min(
                max_cpus, multiprocessing.cpu_count()
            )  # don't use more CPUs than available
            with multiprocessing.Pool(processes=num_processes) as p:
                return p.starmap(cpu_compute_similarity_and_align, inputs)


def _compute_similarity_known_alignment(
    motif_stack_list: list[np.ndarray],
    calculations: list[tuple[int, int]],
    alignfr_stack_list: list[list[np.ndarray]],
    alignh_stack_list: list[list[np.ndarray]],
    max_cpus: int | None,
    use_gpu: bool,
    sim_type: str,
) -> list[np.ndarray]:
    """Compute similarities and alignments."""
    if max_cpus is None:
        if use_gpu:
            # SINGLE GPU CALCULATIONS
            from .similarity_core_gpu import gpu_compute_similarity_known_alignment

            return [
                gpu_compute_similarity_known_alignment(
                    motif_stack_list[c0], motif_stack_list[c1], 
                    alignfr_stack_list[c0][c1], alignh_stack_list[c0][c1],
                    sim_type
                )
                for c0, c1 in calculations
            ]
        else:
            # SINGLE CPU CALCULATIONS
            from .similarity_core_cpu import cpu_compute_similarity_known_alignment

            return [
                cpu_compute_similarity_known_alignment(
                    motif_stack_list[c0], motif_stack_list[c1], 
                    alignfr_stack_list[c0][c1], alignh_stack_list[c0][c1],
                    sim_type
                )
                for c0, c1 in calculations
            ]
    else:
        if use_gpu:
            # MULTI-GPU CALCULATIONS
            print("max_cpu and use_gpu are incompatible options")
            assert False
        else:
            # MULTI-CPU CALCULATIONS
            from .similarity_core_cpu import cpu_compute_similarity_and_align

            inputs = [
                (motif_stack_list[c0], motif_stack_list[c1], 
                 alignfr_stack_list[c0][c1], alignh_stack_list[c0][c1],
                 sim_type)
                for c0, c1 in calculations
            ]
            num_processes = min(
                max_cpus, multiprocessing.cpu_count()
            )  # don't use more CPUs than available
            with multiprocessing.Pool(processes=num_processes) as p:
                return p.starmap(cpu_compute_similarity, inputs)


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


def _reassemble_similarity(
    calculations: list[tuple[int, int]],
    chunked_calculations: list[tuple[int, int]],
    chunked_simliarities: list[np.ndarray],
    chunk_map: dict[int, int],
) -> np.ndarray:
    """Reassemble chunked results."""
    chunked_calculations_revmap = {c: i for i, c in enumerate(chunked_calculations)}
    results = []
    for c0, c1 in calculations:
        sim_block = []
        for c0_chunk in chunk_map[c0]:
            sim_block_row = []
            for c1_chunk in chunk_map[c1]:
                sim = chunked_simliarities[
                    chunked_calculations_revmap[(c0_chunk, c1_chunk)]
                ]
                sim_block_row.append(sim)
            sim_block.append(sim_block_row)
        sim = np.block(sim_block)
        results.append(sim)
    assert len(results) == len(calculations)
    return results


############
# SETTINGS #
############
DEFAULT_MAX_CHUNK = None
DEFAULT_MAX_CPUS = None
DEFAULT_USE_GPU = False
DEFAULT_SIM_TYPE = "l2"


def set_default_options(
    max_chunk: int | None = None,
    max_cpus: int | None = None,
    use_gpu: bool | None = None,
    sim_type: str | None = None,
):
    """Set default values for max_chunk, max_cpus, use_gpu, and l2."""
    if max_chunk is not None:
        DEFAULT_MAX_CHUNK = max_chunk
    if max_cpus is not None:
        DEFAULT_MAX_CPUS = max_cpus
    if use_gpu is not None:
        DEFAULT_USE_GPU = use_gpu
    if sim_type is not None:
        DEFAULT_SIM_TYPE = sim_type
