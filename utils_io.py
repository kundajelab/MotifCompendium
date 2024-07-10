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

def sequence_importance_from_seqlets(seqlets):
	# INPUT = (N, 30, 4)
	seqlets_pos = np.maximum(seqlets, 0)
	seqlets_neg = np.maximum(-seqlets, 0)
	seqlets_pos_8 = seqlets_pos@MOTIF_4_TO_8_POS
	seqlets_neg_8 = seqlets_neg@MOTIF_4_TO_8_NEG
	seqlets_8 = seqlets_pos_8 + seqlets_neg_8
	seqlets_importance = seqlets_8/np.sum(seqlets_8, axis=(1, 2), keepdims=True)
	motif_importance = np.mean(seqlets_importance, axis=0)
	# TODO: CUTOFF UNDER SOME CERTAIN VALUE AND IGNORE ALL ELSE THEN RENORMALIZE
	# motif_importance[motif_importance < 1/240] = 0
	# motif_importance /= np.sum(motif_importance)
	return motif_importance

def load_modiscos(modisco_dict):
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
	cwms = np.stack(cwms, axis=0)
	return sims, cwms, names

