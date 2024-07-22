import numpy as np

import multiprocessing

from .similarity_core_cpu import compute_similarity_and_align


###############################
# SIMILARITY HELPER FUNCTIONS #
###############################
MOTIF_8_TO_4 = np.zeros((8, 4))
MOTIF_8_TO_4[0, 0] = 1; MOTIF_8_TO_4[1, 0] = 1
MOTIF_8_TO_4[2, 1] = 1; MOTIF_8_TO_4[3, 1] = 1
MOTIF_8_TO_4[4, 2] = 1; MOTIF_8_TO_4[2, 2] = 1
MOTIF_8_TO_4[6, 3] = 1; MOTIF_8_TO_4[3, 3] = 1

'''
def reverse_complement(motifs):
	return motifs[:, ::-1, ::-1]
'''

def validate_sims(sims):
	assert(type(sims) == np.ndarray)
	assert(len(sims.shape) == 3)
	assert(sims.shape[1] == 30)
	assert(sims.shape[2] in [4, 8])
	assert((sims >= 0).all())
	assert(np.allclose(sims.sum(axis=(1, 2)), 1))

def sim8_to_sim4(sims):
	return sims@MOTIF_8_TO_4


###############################################
# SIMILARITY CALCULATION MANAGEMENT FUNCTIONS #
###############################################
def _chunk_axis_0(X, chunk_size):
	N = X.shape[0]
	X_chunks = []
	for i in range(N//chunk_size):
		X_chunks.append(X[i*chunk_size:(i+1)*chunk_size])
	if N % chunk_size > 0:
		X_chunks.append(X[(i+1)*chunk_size:])
	assert(len(X_chunks) == int(np.ceil(N/chunk_size)))
	return X_chunks

def _chunk_sims_and_calcs(sims_list, calculations, max_chunk):
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

def _compute_similarity_and_align_parallel(sims_list, calculations, max_parallel, use_gpu, l2=False):
	if max_parallel is None:
		if use_gpu:
			# SINGLE GPU CALCULATIONS
			from .similarity_core_gpu import gpu_compute_similarity_and_align
			return [gpu_compute_similarity_and_align(sims_list[c[0]], sims_list[c[1]], l2=l2) for c in calculations]
		else:
			# SINGLE CPU CALCULATIONS
			return [compute_similarity_and_align(sims_list[c[0]], sims_list[c[1]], l2=l2) for c in calculations]
	else:
		if use_gpu:
			print("not yet implemented"); assert(False)
			# MULTI-GPU CALCULATIONS
			# Define inputs to similarity_core_gpu.gpu_similarity_worker
			gpu_inputs = [[] for _ in range(max_parallel)]
			for idx, c in enumerate(calculations):
				gpu_idx = idx % max_parallel
				gpu_inputs[gpu_idx].append((c, sims_list[c[0]], sims_list[c[1]]))
			worker_inputs = [(i, x) for i, x in enumerate(gpu_inputs)]
			# Start pool with one set of inputs per process --> each process handles a single GPU
			from .similarity_core_gpu import gpu_similarity_worker
			with multiprocessing.Pool(processes=max_parallel) as p:
				results = p.starmap(gpu_similarity_worker, worker_inputs)
			# Correctly re-join results
			results_dict = {}
			for process_results in results:
				for c, r in process_results:
					results_dict[c] = r
			return [results_dict[c] for c in calculations]
		else:
			# MULTI-CPU CALCULATIONS
			inputs = [(sims_list[c[0]], sims_list[c[1]], l2) for c in calculations]
			max_cpus = min(max_parallel, multiprocessing.cpu_count()) # don't use more cpus than available
			with multiprocessing.Pool(processes=max_cpus) as p:
				return p.starmap(compute_similarity_and_align, inputs)

def _reassemble_results(calculations, chunked_calculations, chunked_results, chunk_map):
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
				sim, fb, ali = chunked_results[chunked_calculations_revmap[(c0_chunk, c1_chunk)]]
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
	assert(len(results) == len(calculations))
	return results

def compute_similarities(sims_list, calculations, max_chunk, max_parallel, use_gpu, l2=False):
	for sims in sims_list:
		validate_sims(sims)
	if max_chunk is not None:
		chunked_sims_list, chunked_calculations, chunk_map = _chunk_sims_and_calcs(sims_list, calculations, max_chunk)
		chunked_results = _compute_similarity_and_align_parallel(chunked_sims_list, chunked_calculations, max_parallel, use_gpu, l2=l2)
		return _reassemble_results(calculations, chunked_calculations, chunked_results, chunk_map)
	else:
		return _compute_similarity_and_align_parallel(sims_list, calculations, max_parallel, use_gpu, l2=l2)

