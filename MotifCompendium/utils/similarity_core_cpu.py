import numpy as np


def reverse_complement(motifs):
	return motifs[:, ::-1, ::-1]

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

def compute_similarity_and_align(simsA, simsB, l2=False):
	if l2:
		samsA = simsA/np.linalg.norm(simsA, axis=(1, 2), keepdims=True)
		samsB = simsB/np.linalg.norm(simsB, axis=(1, 2), keepdims=True)
	else:
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
	sim_alignments[sim == 0] = 0 # alignment erroneously always -29 when 0 similarity (argmax artifact)
	return sim, sim_fb, sim_alignments

