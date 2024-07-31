import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import logomaker


def average_motifs(motifs_8, alignment_fb, alignment_h):
	motifs_4 = _8_to_4(motifs_8)
	N = motifs_4.shape[0]
	if N == 1:
		return motifs_8[0, :, :]
	shifts = [alignment_h[i, 0] if alignment_fb[i, 0] == 0 else -alignment_h[i, 0] for i in range(N)]
	max_shift = max(shifts)
	min_shift = min(shifts)
	width = 30 + max_shift - min_shift
	motif_sum = np.zeros((width, 4))
	for i in range(N):
		if alignment_fb[i, 0] == 0:
			s = alignment_h[i, 0]
			motif_i = motifs_4[i, :, :]
		else:
			s = -alignment_h[i, 0]
			motif_i = motifs_4[i, ::-1, ::-1]
		motif_sum[np.abs(min_shift)+s:np.abs(min_shift)+s+30, :] += motif_i
	motif_avg = motif_sum/N
	squashed_motif = squash_motif(motif_avg)
	squashed_motif_8 = _4_to_8(squashed_motif)
	squashed_motif_8 /= np.sum(squashed_motif_8)
	return squashed_motif_8

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

def squash_motif(motif, squash_to=30):
	N, c = motif.shape
	i_sums = []
	for i in range(N - squash_to + 1):
		i_sums.append((np.sum(np.abs(motif[i:i+squash_to, :])), i))
	i_sums = sorted(i_sums, reverse=True)
	top_i = i_sums[0][1]
	return motif[top_i:top_i+squash_to, :]

