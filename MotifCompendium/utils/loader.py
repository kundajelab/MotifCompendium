import h5py
import numpy as np
import pandas as pd


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

def sequence_importance_from_seqlets(seqlets, ic=False):
	# INPUT = (N, 30, 4)
	if ic:
		seqlets_avg = np.mean(seqlets, axis=0)
		motif = ic_scale(seqlets_avg)
		motif_8 = _4_to_8(motif)
		motif_importance = motif_8/np.sum(motif_8)
	else:
		seqlets_8 = _4_to_8(seqlets)
		seqlets_importance = seqlets_8/np.sum(seqlets_8, axis=(1, 2), keepdims=True)
		motif_importance = np.mean(seqlets_importance, axis=0)
	# TODO: CUTOFF UNDER SOME CERTAIN VALUE AND IGNORE ALL ELSE THEN RENORMALIZE
	# motif_importance[motif_importance < 1/240] = 0
	# motif_importance /= np.sum(motif_importance)
	return motif_importance

def load_modisco(modisco_file, ic=False):
	sims, cwms, names, counts = [], [], [], []
	with h5py.File(modisco_file, 'r') as f:
		if "pos_patterns" in f:
			for pattern in list(f["pos_patterns"]):
				seqlets = f["pos_patterns"][pattern]["seqlets"]["contrib_scores"][()]
				motif_sim = sequence_importance_from_seqlets(seqlets, ic=ic)
				sims.append(motif_sim)
				motif_cwm = f["pos_patterns"][pattern]["contrib_scores"][()]
				cwms.append(motif_cwm)
				names.append(f"pos.{pattern}")
				counts.append(seqlets.shape[0])
		if "neg_patterns" in f:	
			for pattern in list(f["neg_patterns"]):
				seqlets = f["neg_patterns"][pattern]["seqlets"]["contrib_scores"][()]
				motif_sim = sequence_importance_from_seqlets(seqlets, ic=ic)
				sims.append(motif_sim)
				motif_cwm = f["neg_patterns"][pattern]["contrib_scores"][()]
				cwms.append(motif_cwm)
				names.append(f"neg.{pattern}")
				counts.append(seqlets.shape[0])
	sims = np.stack(sims, axis=0)
	# cwms = np.stack(cwms, axis=0)
	cwms = _8_to_4(sims)
	return sims, cwms, names, counts

def load_modiscos(modisco_dict, ic=False):
	print("uh loading")
	sims, cwms, names, counts = [], [], [], []
	for m_name, modisco in modisco_dict.items():
		m_sims, m_cwms, m_names, m_counts = load_modisco(modisco, ic=ic)
		m_names = [f"{m_name}-{x}" for x in m_names]
		sims.append(m_sims)
		cwms.append(m_cwms)
		names += m_names
		counts += m_counts
	sims = np.concatenate(sims, axis=0)
	cwms = np.concatenate(cwms, axis=0)
	return sims, cwms, names, counts

def squash_motif(motif, squash_to=30):
	if len(motif) < squash_to:
		n_add = squash_to - len(motif)
		zeros_df = pd.DataFrame(np.zeros((n_add, motif.shape[1])), columns=motif.columns)
		motif = pd.concat([motif, zeros_df], ignore_index=True)
		return motif
	elif len(motif) > squash_to:
		motif_np = motif.to_numpy()
		i_sums = []
		for i in range(len(motif) - 29):
			i_sums.append((np.sum(motif_np[i:i+30, :]), i))
		i_sums = sorted(i_sums, reverse=True)
		top_i = i_sums[0][1]
		motif = pd.DataFrame(motif_np[top_i:top_i+30, :], columns=motif.columns)
		return motif
	else:
		return motif

def load_pfm(pfm_file):
	names = []
	pwms = []
	active_pwm = False
	with open(pfm_file, 'r') as f:
	        for line in f:
	                x = line.strip()
	                if active_pwm:
	                        if x.startswith(">"):
	                                # submit
	                                pwms.append(squash_motif(pd.DataFrame(current_pwm)))
	                                names.append(current_pwm_name)
	                                # restart
	                                current_pwm_name = x[1:]
	                                current_pwm = {"A": [], "C": [], "G": [], "T": []}
	                        else:
	                                a, c, g, t = x.split()
	                                a, c, g, t = float(a), float(c), float(g), float(t)
	                                acgt = np.asarray([a, c, g, t])
	                                xlogx = acgt*np.log2(acgt, where=(acgt != 0))
	                                entropy = np.sum(-xlogx)/2
	                                ic = 1 - entropy
	                                acgt_ic = acgt * ic
	                                current_pwm["A"].append(acgt_ic[0])
	                                current_pwm["C"].append(acgt_ic[1])
	                                current_pwm["G"].append(acgt_ic[2])
	                                current_pwm["T"].append(acgt_ic[3])
	                else:
	                        assert(x.startswith(">"))
	                        active_pwm = True
	                        current_pwm_name = x[1:]
	                        current_pwm = {"A": [], "C": [], "G": [], "T": []}

	pwms_mtx = np.stack([x.to_numpy() for x in pwms], axis=0)
	pwms_mtx /= np.sum(pwms_mtx, axis=(1, 2), keepdims=True)
	return pwms_mtx, names

