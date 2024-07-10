import numpy as np

import multiprocessing

import utils_similarity_core_cpu


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


#############################
# SIMILARITY CORE FUNCTIONS #
#############################
'''
LEFTTENSOR = np.zeros((59, 88, 30))
for i in range(59):
	LEFTTENSOR[i, i:i+30, :] = np.eye(30)

RIGHTTENSOR = np.zeros((30, 88))
RIGHTTENSOR[:, 29:59] = np.eye(30)

def compute_similarity_left_side(motifs):
	# motifs = (N, 30, K)
	K = motifs.shape[2]
	motif_slices = [motifs[:, :, i] for i in range(K)] # (N, 30)
	motif_slice_pivots = [x.T for x in motif_slices] # (30, N)
	left_side_prepivots = [LEFTTENSOR@x for x in motif_slice_pivots] # (59, 88, N)
	left_side_matrices = [np.transpose(x, axes=(2, 0, 1)) for x in left_side_prepivots] # (N, 59, 88)
	for x in left_side_matrices:
		assert(motifs.shape[0] == x.shape[0])
		assert((x.shape[1] == 59) and (x.shape[2] == 88))
	return left_side_matrices
	
def compute_similarity_right_side(motifs):
	# motifs = (M, 30, K)
	K = motifs.shape[2]
	motifs_pivot = np.transpose(motifs, axes=(0, 2, 1)) # (M, K, 30)
	right_side_prepivot = motifs_pivot@RIGHTTENSOR # (M, K, 88)
	right_side_matrix = np.transpose(right_side_prepivot, axes=(2, 0, 1)) # (88, M, K)
	right_side_matrices = [right_side_matrix[:, :, i] for i in range(K)] # (88, M)
	for x in right_side_matrices:
		assert(motifs.shape[0] == x.shape[1])
		assert(x.shape[0] == 88)
	return right_side_matrices

def compute_similarity(motif_set_1, motif_set_2):
	# TRANSPOSE FOR EFFICIENCY
	N_original = motif_set_1.shape[0]
	M_original = motif_set_2.shape[0]
	if N_original > M_original:
		temp = motif_set_1
		motif_set_1 = motif_set_2
		motif_set_2 = temp
	transpose = N_original > M_original
	N = motif_set_1.shape[0]
	M = motif_set_2.shape[0]
	# COMPUTE LEFT SIDE TENSOR
	left_side_matrices = compute_similarity_left_side(motif_set_1)
	# COMPUTE RIGHT SIDE TENSOR
	right_side_matrices = compute_similarity_right_side(motif_set_2)
	# SUM ACROSS ATCG
	convs = [left_side_matrices[i]@right_side_matrices[i] for i in range(len(left_side_matrices))]
	total_sum = np.sum(convs, axis=0)
	# COMPUTE ALIGNMENT
	best_alignment_scores = np.max(total_sum, axis=1)
	best_alignments = np.argmax(total_sum, axis=1) - 29
	# REALIGN IF NEEDED
	if transpose:
		best_alignment_scores = best_alignment_scores.T
		best_alignments = -best_alignments.T # negative because transposing flips alignment
	assert(best_alignment_scores.shape == (N_original, M_original))
	assert(best_alignments.shape == (N_original, M_original))
	return best_alignment_scores, best_alignments

def compute_similarity_and_align(simsA, simsB):
	validate_sims(simsA)
	validate_sims(simsB)
	samsA = np.sqrt(simsA)
	samsB = np.sqrt(simsB)
	samsB_revcomp = reverse_complement(samsB)
	# forward similarity
	sim_1, sim_1_alignments = compute_similarity(samsA, samsB) # skew-symmetric alignment
	# backward similarity
	sim_2, sim_2_alignments = compute_similarity(samsA, samsB_revcomp) # symmetric alignment
	# aligning
	sim_12 = np.stack([sim_1, sim_2])
	sim = np.max(sim_12, axis=0)
	sim_fb = np.argmax(sim_12, axis=0)
	sim_alignments = np.where(sim_fb == 0, sim_1_alignments, sim_2_alignments)
	sim_alignments[sim == 0] = 0 # alignment erroneously always -29
	return sim, sim_fb, sim_alignments
'''


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

def _compute_similarity_and_align_parallel(sims_list, calculations, max_parallel, use_gpu):
	if max_parallel is None:
		if use_gpu:
			# SINGLE GPU CALCULATIONS
			import utils_similarity_core_gpu
			return [utils_similarity_core_gpu.compute_similarity_and_align(sims_list[c[0]], sims_list[c[1]]) for c in calculations]
		else:
			# SINGLE CPU CALCULATIONS
			return [utils_similarity_core_cpu.compute_similarity_and_align(sims_list[c[0]], sims_list[c[1]]) for c in calculations]
	else:
		if use_gpu:
			print("not yet implemented"); assert(False)
			# MULTI-GPU CALCULATIONS
			# Define inputs to utils_similarity_core_gpu.gpu_similarity_worker
			gpu_inputs = [[] for _ in range(max_parallel)]
			for idx, c in enumerate(calculations):
				gpu_idx = idx % max_parallel
				gpu_inputs[gpu_idx].append((c, sims_list[c[0]], sims_list[c[1]]))
			worker_inputs = [(i, x) for i, x in enumerate(gpu_inputs)]
			# Start pool with one set of inputs per process --> each process handles a single GPU
			import utils_similarity_core_gpu
			with multiprocessing.Pool(processes=max_parallel) as p:
				results = p.starmap(utils_similarity_core_gpu.gpu_similarity_worker, worker_inputs)
			# Correctly re-join results
			results_dict = {}
			for process_results in results:
				for c, r in process_results:
					results_dict[c] = r
			return [results_dict[c] for c in calculations]
		else:
			# MULTI-CPU CALCULATIONS
			inputs = [(sims_list[c[0]], sims_list[c[1]]) for c in calculations]
			max_cpus = min(max_parallel, multiprocessing.cpu_count()) # don't use more cpus than available
			with multiprocessing.Pool(processes=max_cpus) as p:
				return p.starmap(utils_similarity_core_cpu.compute_similarity_and_align, inputs)

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

def compute_similarities(sims_list, calculations, max_chunk, max_parallel, use_gpu):
	for sims in sims_list:
		validate_sims(sims)
	if max_chunk is not None:
		chunked_sims_list, chunked_calculations, chunk_map = _chunk_sims_and_calcs(sims_list, calculations, max_chunk)
		chunked_results = _compute_similarity_and_align_parallel(chunked_sims_list, chunked_calculations, max_parallel, use_gpu)
		return _reassemble_results(calculations, chunked_calculations, chunked_results, chunk_map)
	else:
		return _compute_similarity_and_align_parallel(sims_list, calculations, max_parallel, use_gpu)

