import h5py
import numpy as np
import pandas as pd

import time

import utils_clustering
import utils_io
import utils_plotting
import utils_similarity


######################################
# MOTIF COMPENDIUM FACTORY FUNCTIONS #
######################################
def load(file_loc):
	with h5py.File(file_loc, 'r') as f:
		sims = f["sims"][:]
		logos = f["logos"][:]
		similarity = f["similarity"][:]
		alignment_fb = f["alignment_fb"][:]
		alignment_h = f["alignment_h"][:]
	metadata = pd.read_hdf(file_loc, key="metadata")
	return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h, metadata)

def build(sims, logos=None, metadata=None, max_chunk=None, max_parallel=None, use_gpu=False):
	# SIMS
	utils_similarity.validate_sims(sims)
	# LOGOS
	if logos is None:
		logos = sims if sims.shape[2] == 4 else utils_similarity.sim8_to_sim4(sims)
	# METADATA
	if metadata is None:
		metadata = pd.DataFrame()
		metadata["name"] = [f"motif_{i}" for i in range(sims.shape[0])]
	# SIMILARITY
	print("aligning"); start = time.time()
	# similarity, alignment_fb, alignment_h = utils_similarity.compute_similarity_and_align(sims, sims, False)
	similarity, alignment_fb, alignment_h = utils_similarity.compute_similarities([sims], [(0, 0)], max_chunk, max_parallel, use_gpu)[0]
	np.fill_diagonal(similarity, 1)
	print(f"completed {time.time() - start}")
	# CONSTRUCT OBJECT
	return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h, metadata)

def build_from_modisco(modisco_dict, max_chunk=None, max_parallel=None, use_gpu=False):
	sims, cwms, names = utils_io.load_modiscos(modisco_dict)
	metadata = pd.DataFrame(); metadata["name"] = names
	compendium = build(sims, logos=cwms, metadata=metadata, max_chunk=max_chunk, max_parallel=max_parallel, use_gpu=use_gpu)
	return compendium

def combine(compendiums, max_chunk=None, max_parallel=None, use_gpu=False):
	n = len(compendiums)
	# SIMILARITIES
	sims_list = [mc.sims for mc in compendiums]
	calculations = []
	for i in range(n):
		for j in range(i+1, n):
			calculations.append((i, j))
	similarity_results = utils_similarity.compute_similarities(sims_list, calculations, max_chunk, max_parallel, use_gpu)
	similarity_block = [[None for i in range(n)] for i in range(n)]
	alignment_fb_block = [[None for i in range(n)] for i in range(n)]
	alignment_h_block = [[None for i in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(n):
			if i == j:
				similarity_block[i][j] = compendiums[i].similarity
				alignment_fb_block[i][j] = compendiums[i].alignment_fb
				alignment_h_block[i][j] = compendiums[i].alignment_h
			elif i > j:
				similarity_block[i][j], alignment_fb_block[i][j], alignment_h_block[i][j] = results_revcomp[(i, j)]
				
			else:
				similarity_block[i][j] = similarity_block[j][i].T
				alignment_fb_block[i][j] = alignemnt_fb_block[j][i].T
				alignment_h_block[i][j] = alignment_h_block[j][i].T
	similarity = np.block(similarity_block)
	alignment_fb = np.block(alignment_fb_block)
	alignment_h = np.block(alignment_h_block)
	# SIMS + LOGOS
	sims = np.concatenate(sims_list)
	logos = np.concatenate([mc.logos for mc in compendiums])
	# METADATA
	metadata = pd.concat([mc.metadata for mc in compendiums])
	return MotifCompendium(sims, logos, similarity, alignment_fb, alignment_h, metadata)
	print("not yet implemented")


##########################
# MOTIF COMPENDIUM CLASS #
##########################
class MotifCompendium():
	def __init__(self, sims, logos, similarity, alignment_fb, alignment_h, metadata):
		self.sims = sims
		self.logos = logos
		self.similarity = similarity
		self.alignment_fb = alignment_fb
		self.alignment_h = alignment_h
		self.metadata = metadata
		# self.validate()

	def cluster(self, algorithm="leiden", similarity_threshold=0.8, save_name="cluster"):
		self.metadata[save_name] = utils_clustering.cluster(self.similarity, algorithm, similarity_threshold)

	def extend(self, sims, metadata=None, max_chunk=None, max_parallel=None):
		
		print("not yet implemented")

	def assign(self, sims, n=5):
		print("not yet implemented")

	###
	# VISUALIZATION
	###
	def create_html(self, html_out):
		utils_plotting.create_html(self.logos, list(self.metadata["cluster"]), self.alignment_fb, self.alignment_h, list(self.metadata["name"]), html_out)

	def heatmap(self, annot=False, label=False, show=False, save_loc=None):
		if label:
			utils_plotting.plot_heatmap(self.similarity, annot=annot, labels=list(self.metadata["name"]), show=show, save_loc=save_loc)
		else:
			utils_plotting.plot_heatmap(self.similarity, annot=annot, show=show, save_loc=save_loc)

	###
	# GENERAL
	###
	def save(self, save_loc):
		with h5py.File(save_loc, 'w') as f:
			f.create_dataset("sims", data=self.sims)
			f.create_dataset("logos", data=self.logos)
			f.create_dataset("similarity", data=self.similarity)
			f.create_dataset("alignment_fb", data=self.alignment_fb)
			f.create_dataset("alignment_h", data=self.alignment_h)
		self.metadata.to_hdf(save_loc, key="metadata", mode="a")

	def validate(self):
		# SIMS
		assert(type(self.sims) == np.ndarray)
		assert(len(self.sims.shape) == 3)
		assert(self.sims.shape[1] == 30)
		assert(self.sims.shape[2] in [4, 8])
		assert((self.sims >= 0).all())
		assert(np.allclose(self.sims.sum(axis=(1, 2)), 1))
		# LOGOS	
		assert(type(self.logos) == np.ndarray)
		assert(len(self.logos.shape) == 3)
		assert(self.logos.shape[1] == 30)
		assert(self.logos.shape[2] == 4)
		# SIMILARITY
		assert(type(self.similarity) == np.ndarray)
		assert(len(self.similarity.shape) == 2)
		assert(np.allclose(self.similarity, self.similarity.T))
		assert(np.max(self.similarity) == 1)
		assert((self.similarity >= 0).all())
		# ALIGNMENT_FB
		assert(type(self.alignment_fb) == np.ndarray)
		assert(len(self.alignment_fb.shape) == 2)
		assert(np.allclose(self.alignment_fb, self.alignment_fb.T))
		assert(((self.alignment_fb == 0) | (self.alignment_fb == 1)).all())
		# ALIGNMENT_H
		assert(type(self.alignment_h) == np.ndarray)
		assert(len(self.alignment_h.shape) == 2)
		assert(np.allclose(self.alignment_h, np.where(self.alignment_fb == 0, -self.alignment_h.T, self.alignment_h.T)))
		# METADATA
		assert(type(self.metadata) == pd.DataFrame)
		# SHAPE MATCHES
		assert(self.sims.shape[0] == self.logos.shape[0])
		assert(self.sims.shape[0] == self.similarity.shape[0])
		assert(self.sims.shape[0] == self.alignment_fb.shape[0])
		assert(self.sims.shape[0] == self.alignment_h.shape[0])
		assert(self.sims.shape[0] == len(self.metadata))

	def get_similarity_slice(self, slice1, slice2=None, with_names=False):
		assert(isinstance(slice1, pd.Series))
		assert(slice1.dtype == bool)
		keep_idxs_1 = list(self.metadata[slice1].index)
		if slice2 is None:
			similarity_slice = self.similarity[keep_idxs_1, :]
			assert(similarity_slice.shape == (len(keep_idxs_1), len(self)))
			if with_names:
				similarity_slice_df = pd.DataFrame(similarity_slice, index=list(self.metadata[slice1]["name"]), columns=list(self.metadata["name"]))
				return similarity_slice_df
			else:
				return similarity_slice
		else:
			assert(isinstance(slice2, pd.Series))
			assert(slice2.dtype == bool)
			keep_idxs_2 = list(self.metadata[slice2].index)
			similarity_slice = self.similarity[keep_idxs_1, :][:, keep_idxs_2]
			assert(similarity_slice.shape == (len(keep_idxs_1), len(keep_idxs_2)))
			if with_names:
				similarity_slice_df = pd.DataFrame(similarity_slice, index=list(self.metadata[slice1]["name"]), columns=list(self.metadata[slice2]["name"]))
				return similarity_slice_df
			else:
				return similarity_slice

	def __str__(self):
		return f"Motif Compendium with {len(self)} motifs.\n{self.metadata}"

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, key):
		if isinstance(key, str):
			return self.metadata[key]
		elif isinstance(key, pd.Series) and key.dtype == bool:
			metadata_slice = self.metadata.copy()[key]
			keep_idxs = list(metadata_slice.index)
			metadata_slice = metadata_slice.reset_index(drop=True)
			sims_slice = self.sims.copy()[keep_idxs, :, :]
			logos_slice = self.logos.copy()[keep_idxs, :, :]
			similarity_slice = self.similarity.copy()[keep_idxs, :][:, keep_idxs]
			alignment_fb_slice = self.alignment_fb.copy()[keep_idxs, :][:, keep_idxs]
			alignment_h_slice = self.alignment_h.copy()[keep_idxs, :][:, keep_idxs]
			return MotifCompendium(sims_slice, logos_slice, similarity_slice, alignment_fb_slice, alignment_h_slice, metadata_slice)
		else:
			raise TypeError("MotifCompendium cannot be indexed by this")

	def __eq__(self, other):
		if isinstance(other, MotifCompendium):
			return np.allclose(self.sims, other.sims) and np.allclose(self.logos, other.logos) and np.allclose(self.similarity, other.similarity) and np.allclose(self.alignment_fb, other.alignment_fb) and np.allclose(self.alignment_h, other.alignment_h) and self.metadata.equals(other.metadata)
		return False

