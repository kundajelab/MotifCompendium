import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def motif_to_df(motif):
	return pd.DataFrame(motif, columns=["A", "C", "G", "T"])

def prep_plotting_from_motifs(cwms, clusters, sim_fb, sim_alignments, names):
	motif_plot_dfs = {}
	for c_name, c in clusters.items():
		c_dfs = []
		c_df_index = set()
		first_idx = c[0]
		c_left_alignment = -np.max([sim_alignments[first_idx, idx] for idx in c]) - 0.5
		c_right_alignment = np.abs(np.min([sim_alignments[first_idx, idx] for idx in c])) + 29.5
		for idx in c:
			idx_cwm = cwms[idx, :, :] if sim_fb[first_idx, idx] == 0 else cwms[idx, ::-1, ::-1]
			idx_shift = sim_alignments[first_idx, idx]
			idx_df = pd.DataFrame(idx_cwm, columns=["A", "C", "G", "T"])
			idx_df.index -= idx_shift
			idx_name = names[idx]
			c_dfs.append((idx_name, idx_df))
			c_df_index.update(idx_df.index)
		c_df_index = sorted(c_df_index)
		c_dfs = [(x[0], x[1].reindex(c_df_index, fill_value=0)) for x in c_dfs]
		motif_plot_dfs[c_name] = c_dfs
	return motif_plot_dfs

def create_html(cwms, clusters, sim_fb, sim_alignments, names, html_out_loc, average=True, parallel=True):
	from .make_report import generate_report
	clustering = {}
	for c in sorted(set(clusters)):
		clustering[c] = [i for i, ic in enumerate(clusters) if ic == c]
	motif_plot_dfs = prep_plotting_from_motifs(cwms, clustering, sim_fb, sim_alignments, names)
	generate_report(motif_plot_dfs, html_out_loc, average, parallel)

def plot_motif_on_ax(motif, ax, motif_name=None):
	motif_df = motif_to_df(motif)
	logomaker.Logo(motif_df, ax=ax)
	ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
	ax.set_xticks([])
	ax.set_yticks([])
	if motif_name is not None:
		ax.set_title(motif_name)

def plot_motifs(motifs, motif_names=None):
	if type(motifs) == np.ndarray:
		verify_motif_stack(motifs)
		motifs = [motifs[i, :, :] for i in range(motifs.shape[0])]
	# check_error(type(motifs) == list, "error: motifs must be provided as a list or an np.ndarray.")
	if motif_names is None:
		motif_names = [None for x in range(len(motifs))]
	# check_error(len(motif_names) == len(motifs), "error: number of motifs and names not matching.")
	fig, axs = plt.subplots(len(motifs), 1)
	for i in range(len(motifs)):
		plot_motif_on_ax(motifs[i], axs[i], motif_names[i])
	plt.show()

def plot_heatmap(data, annot=False, labels=None, show=False, save_loc=None):
	if labels is None:
		df = pd.DataFrame(data)
	else:
		df = pd.DataFrame(data, index=labels, columns=labels)
	plt.figure(figsize=(10, 10))
	heatmap = sns.heatmap(df, annot=annot)
	if save_loc is not None:
		heatmap.get_figure().savefig(save_loc)
	if show:
		plt.show()

