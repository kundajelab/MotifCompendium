import cupy as cp


def reverse_complement(motifs):
	return motifs[:, ::-1, ::-1]

def LEFTTENSOR():
	x = cp.zeros((59, 88, 30))
	for i in range(59):
		x[i, i:i+30, :] = cp.eye(30)
	return x

def RIGHTTENSOR():
	x = cp.zeros((30, 88))
	x[:, 29:59] = cp.eye(30)
	return x

def compute_similarity_left_side(motifs):
	# motifs = (N, 30, K)
	K = motifs.shape[2]
	motif_slices = [motifs[:, :, i] for i in range(K)] # (N, 30)
	motif_slice_pivots = [x.T for x in motif_slices] # (30, N)
	left_side_prepivots = [LEFTTENSOR()@x for x in motif_slice_pivots] # (59, 88, N)
	left_side_matrices = [cp.transpose(x, axes=(2, 0, 1)) for x in left_side_prepivots] # (N, 59, 88)
	for x in left_side_matrices:
		assert(motifs.shape[0] == x.shape[0])
		assert((x.shape[1] == 59) and (x.shape[2] == 88))
	return left_side_matrices
	
def compute_similarity_right_side(motifs):
	# motifs = (M, 30, K)
	K = motifs.shape[2]
	motifs_pivot = cp.transpose(motifs, axes=(0, 2, 1)) # (M, K, 30)
	right_side_prepivot = motifs_pivot@RIGHTTENSOR() # (M, K, 88)
	right_side_matrix = cp.transpose(right_side_prepivot, axes=(2, 0, 1)) # (88, M, K)
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
	convs = cp.stack([left_side_matrices[i]@right_side_matrices[i] for i in range(len(left_side_matrices))]) # need to stack
	total_sum = cp.sum(convs, axis=0)
	# COMPUTE ALIGNMENT
	best_alignment_scores = cp.max(total_sum, axis=1)
	best_alignments = cp.argmax(total_sum, axis=1) - 29
	# REALIGN IF NEEDED
	if transpose:
		best_alignment_scores = best_alignment_scores.T
		best_alignments = -best_alignments.T # negative because transposing flips alignment
	assert(best_alignment_scores.shape == (N_original, M_original))
	assert(best_alignments.shape == (N_original, M_original))
	return best_alignment_scores, best_alignments

def gpu_compute_similarity_and_align(simsA, simsB, l2=False):
	simsA = cp.asarray(simsA)
	simsB = cp.asarray(simsB)
	if l2:
		samsA = simsA/cp.linalg.norm(simsA, axis=(1, 2), keepdims=True)
		samsB = simsB/cp.linalg.norm(simsB, axis=(1, 2), keepdims=True)
	else:
		samsA = cp.sqrt(simsA)
		samsB = cp.sqrt(simsB)
	samsA_revcomp = reverse_complement(samsA)
	# forward similarity
	sim_1, sim_1_alignments = compute_similarity(samsA, samsB) # skew-symmetric alignment
	# backward similarity
	sim_2, sim_2_alignments = compute_similarity(samsA_revcomp, samsB) # symmetric alignment
	# aligning
	sim_12 = cp.stack([sim_1, sim_2])
	sim = cp.max(sim_12, axis=0)
	sim_fb = cp.argmax(sim_12, axis=0)
	sim_alignments = cp.where(sim_fb == 0, sim_1_alignments, sim_2_alignments)
	sim_alignments[sim == 0] = 0 # alignment erroneously always -29
	return sim.get(), sim_fb.get(), sim_alignments.get()

def gpu_similarity_worker(gpu_idx, inputs):
	outputs = []
	with cp.cuda.Device(gpu_idx):
		for calculation, simsA, simsB in inputs:
			result = compute_similarity_and_align(simsA, simsB)
			outputs.append((calculation, result))
	return outputs

