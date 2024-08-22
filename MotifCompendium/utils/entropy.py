import os

import numpy as np
import pandas as pd

def cwm4_to_cwm8(cwm4: np.array) -> np.array:
	# Expand base pair to A+,C+,G+,T+,A-,C-,G-,T-
	rows, cols = cwm4.shape
	cwm8 = np.zeros((rows, 2*cols))

	cwm8[:,:cols] = np.where(cwm4 > 0, cwm4, 0)
	cwm8[:,cols:] = np.where(cwm4 < 0, -cwm4, 0)

	return cwm8

def cwm8_to_pair28(cwm8: np.array) -> np.array:
	# Expand A+,C+,G+,T+,A-,C-,G-,T- to non-repeating, without order dinucleotide pair combinations
	# Current dimensions: cwm8
	rows, cols = cwm8.shape
	nuc = 2 # Dinucleotide pair
	
	# New dimensions
	new_cols = int(np.math.factorial(cols) / (np.math.factorial(cols-nuc) * np.math.factorial(nuc))) # C(n,r)
	new_cols_2 = int(new_cols/2)
	pair28 = np.zeros((rows, new_cols))

	## Matrix multiplication + mask
	mask = np.tril(np.ones(cols), k=-1).astype(bool) # Mask for bottom left off-diagonal half
	
	for i in range(rows):
		pair_perm = np.outer(cwm8[i,:], cwm8[i,:]) # Permutations: With order
		pair28[i,:] = pair_perm[mask] # Combinations: Without order; Exclude self (e.g., AA,CC,GG,TT)

	return pair28

def cwm8_to_dinuc56(cwm8: np.array) -> np.array:
	# Evaluate two positions at a time, as non-self A+,C+,G+,T+,A-,C-,G-,T- dinucleotide pair permutations
	# Current dimensions: cwm8
	rows, cols = cwm8.shape

	# Dimensions: Dinuc64
	new_rows = rows // 2 # Drop final position if odd length
	new_cols = cols ** 2 - cols # Exclude repeat apirs
	dinuc56 = np.zeros((new_rows, new_cols))

	# Calculate dinucleotide pairs, as product of distributions	
	mask = ~np.eye(cols, dtype=bool)

	for i in range(new_rows):
		dinuc_pair = np.outer(cwm8[2*i,:], cwm8[2*i+1,:])
		dinuc56[i,:] = dinuc_pair[mask]

	return dinuc56

def shannon_entropy(prob_array: np.array, epsilon: float = 1e-10) -> float:
	# Normalize, flatten array
	prob_array = prob_array / np.sum(prob_array)
	prob_array = prob_array.flatten()

	# Replace zeroes with epsilon
	prob_array[prob_array == 0] = epsilon
	length = prob_array.shape[0]

	# Calculate Shannon entropy
	entropy = -np.sum(prob_array * np.log2(prob_array)) / np.log2(length)

	return entropy

def calculate_entropy(cwm4: np.array, length: int = 30) -> tuple:
	if not isinstance(cwm4, np.ndarray):
		raise TypeError("cwm4 must be a NumPy array")
	if len(cwm4.shape) != 2:
		raise ValueError("cwm4 must be a 2D array")
	
	# Find normalized, standard length CWM
	rows, cols = cwm4.shape

	# Create cwm8, pair28, dinuc56, as probability
	cwm8 = cwm4_to_cwm8(cwm4)
	cwm8_prob = cwm8 / np.sum(cwm8)

	pair28 = cwm8_to_pair28(cwm8)
	pair28_prob = pair28 / np.sum(pair28)

	dinuc56 = cwm8_to_dinuc56(cwm8)
	dinuc56_prob = dinuc56 / np.sum(dinuc56)

	# Sum across bases (w,), normalize
	pos_prob = np.sum(cwm8_prob, axis=1) / np.sum(cwm8_prob)
	pair_pos_prob = np.sum(pair28_prob, axis=1) / np.sum(pair28_prob)	
	dinuc_pos_prob = np.sum(dinuc56_prob, axis=1) / np.sum(dinuc56_prob)
	
	# Sum across positions (8 or 64,), normalize
	base_prob = np.sum(cwm8_prob, axis=0) / np.sum(cwm8_prob)
	pair_base_prob = np.sum(pair28_prob, axis=0) / np.sum(pair28_prob)	
	dinuc_base_prob = np.sum(dinuc56_prob, axis=0) / np.sum(dinuc56_prob)

	# Calulcate entropy metrics
	# 1) CWM entropy: Entropy calculated on (30,8)
	# Purpose: High = Archetype #1: Noise/chaos, Low = Archetype #2: Sharp nucleotide peak (e.g., G)
	cwm_entropy = shannon_entropy(cwm8_prob)

	# 2) Entropy ratio:
	# Purpose: High = Archetype #3: Single nucleotide repeats (e.g., AAAAA, GGGGG)
	pos_entropy = shannon_entropy(pos_prob)
	base_entropy = shannon_entropy(base_prob)
	entropy_ratio = pos_entropy / base_entropy # High entropy when: High positional = Broad profile, Low base: Single base

	# 3) Pair ratio:
	# Purpose: High = GC, AT bias
	pair_pos_entropy = shannon_entropy(pair_pos_prob)
	pair_base_entropy = shannon_entropy(pair_base_prob)
	pair_entropy_ratio = pair_pos_entropy / pair_base_entropy

	# 4) Dinucleotide ratio
	# Purpose: High = Dinucleotide repeats (e.g., GCGCGC)
	dinuc_pos_entropy = shannon_entropy(dinuc_pos_prob)
	dinuc_base_entropy = shannon_entropy(dinuc_base_prob)
	dinuc_entropy_ratio = dinuc_pos_entropy / dinuc_base_entropy

	return (cwm_entropy, entropy_ratio, pair_entropy_ratio, dinuc_entropy_ratio)
