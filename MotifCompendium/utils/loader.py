import h5py
import numpy as np

MOTIF_4_TO_8_POS = np.zeros((4, 8))
MOTIF_4_TO_8_POS[0, 0] = 1
MOTIF_4_TO_8_POS[1, 2] = 1
MOTIF_4_TO_8_POS[2, 5] = 1
MOTIF_4_TO_8_POS[3, 7] = 1

MOTIF_4_TO_8_NEG = np.zeros((4, 8))
MOTIF_4_TO_8_NEG[0, 1] = 1
MOTIF_4_TO_8_NEG[1, 3] = 1
MOTIF_4_TO_8_NEG[2, 4] = 1
MOTIF_4_TO_8_NEG[3, 6] = 1

def _4_to_8(x):
	x_pos = np.maximum(x, 0)
	x_neg = np.maximum(-x, 0)
	x_pos_8 = x_pos@MOTIF_4_TO_8_POS
	x_neg_8 = x_neg@MOTIF_4_TO_8_NEG
	x_8 = x_pos_8 + x_neg_8
	return x_8

def _8_to_4(x):
	x_pos_4 = x@MOTIF_4_TO_8_POS.T
	x_neg_4 = x@MOTIF_4_TO_8_NEG.T
	x_4 = x_pos_4 - x_neg_4
	return x_4

def ic_scale(x):
	# INPUT = (30, 4)
	x_abs = np.abs(x)
	x_avg = x_abs/np.sum(x_abs, axis=1, keepdims=True)
	xlogx = x_avg*np.log2(x_avg, where=(x_avg != 0))
	entropy = np.sum(-xlogx, axis=1, keepdims=True)/2
	ic = 1 - entropy
	return x * ic

def sequence_importance_from_seqlets(seqlets):
	# INPUT = (N, 30, 4)
	'''
	seqlets_8 = _4_to_8(seqlets)
	seqlets_importance = seqlets_8/np.sum(seqlets_8, axis=(1, 2), keepdims=True)
	motif_importance = np.mean(seqlets_importance, axis=0)
	'''
	seqlets_avg = np.mean(seqlets, axis=0)
	motif = ic_scale(seqlets_avg)
	motif_8 = _4_to_8(motif)
	motif_importance = motif_8/np.sum(motif_8)
	# TODO: CUTOFF UNDER SOME CERTAIN VALUE AND IGNORE ALL ELSE THEN RENORMALIZE
	# motif_importance[motif_importance < 1/240] = 0
	# motif_importance /= np.sum(motif_importance)
	return motif_importance

def load_modisco(modisco_file):
	sims, cwms, names = [], [], []
	with h5py.File(modisco_file, 'r') as f:
		if "pos_patterns" in f:
			for pattern in list(f["pos_patterns"]):
				seqlets = f["pos_patterns"][pattern]["seqlets"]["contrib_scores"][()]
				motif_sim = sequence_importance_from_seqlets(seqlets)
				sims.append(motif_sim)
				motif_cwm = f["pos_patterns"][pattern]["contrib_scores"][()]
				cwms.append(motif_cwm)
				names.append(f"pos.{pattern}")
		if "neg_patterns" in f:	
			for pattern in list(f["neg_patterns"]):
				seqlets = f["neg_patterns"][pattern]["seqlets"]["contrib_scores"][()]
				motif_sim = sequence_importance_from_seqlets(seqlets)
				sims.append(motif_sim)
				motif_cwm = f["neg_patterns"][pattern]["contrib_scores"][()]
				cwms.append(motif_cwm)
				names.append(f"neg.{pattern}")
	sims = np.stack(sims, axis=0)
	# cwms = np.stack(cwms, axis=0)
	cwms = _8_to_4(sims)
	return sims, cwms, names

def load_modiscos(modisco_dict):
	print("uh loading")
	sims, cwms, names = [], [], []
	for m_name, modisco in modisco_dict.items():
		m_sims, m_cwms, m_names = load_modisco(modisco)
		m_names = [f"{m_name}-{x}" for x in m_names]
		sims.append(m_sims)
		cwms.append(m_cwms)
		names += m_names
	sims = np.concatenate(sims, axis=0)
	cwms = np.concatenate(cwms, axis=0)
	return sims, cwms, names

