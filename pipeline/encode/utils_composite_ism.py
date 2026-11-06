"""
ISM cooperativity helpers for pipeline_encode_composite_ism.py: fold wrappers, summaries across sequences and folds,
HDF5 I/O and the spacing-experiment calculations and plots.

Based on HDMA (https://github.com/GreenleafLab/HDMA/), code/03-chrombpnet/tangermeme_utils, with minor modifications
(BPNet/ZeroControlWrapper, mean-reduced HDF5 I/O, negative spacings and auto-scaled profile plots, from the local
ENCODE-MotifCompendium branch). Its modules are copied here, in this order, so that the pipeline does not import from an
HDMA checkout: constants, utils, wrappers, eval, hdf5_io, syntax_helpers. Each module's code is unchanged; its imports
are merged below.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('white')
import logomaker
import pandas as pd
import torch
from bpnetlite.attribute import deep_lift_shap
import h5py
import matplotlib.ticker as ticker
import plotnine as pn
from tangermeme.plot import plot_logo
import bpnetlite
from bpnetlite import ChromBPNet
from bpnetlite import BPNet
from bpnetlite.bpnet import CountWrapper
import tangermeme
from tangermeme.utils import random_one_hot
from tangermeme.utils import one_hot_encode
from tangermeme.utils import characters
from tangermeme.utils import reverse_complement
from tangermeme.utils import _validate_input
from tangermeme.utils import _cast_as_tensor
from tangermeme.space import space
from tangermeme.ersatz import multisubstitute
from tangermeme.ersatz import substitute
from tangermeme.marginalize import marginalize
from scipy.stats import entropy
from plotnine import (
	ggplot, aes, geom_boxplot, geom_jitter, theme, xlab, ylab, ggtitle, element_blank, element_text, element_rect, scale_fill_manual, coord_cartesian, geom_hline
)


#### constants.py -------------------------------------------------------------------------------------------
# Constants related to ChromBPNet/tangermeme


INPUTLEN = 2114
OUTPUTLEN = 1000
MIDPOINT = INPUTLEN // 2
SHIFT_REL_INPUT = np.int64((2114 - 1000) / 2)
SHIFT_REL_INPUT

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]

LOGO_ALPHABET = 'ACGT'

# finemo colorscheme
LOGO_COLORS = {"A": '#109648', "C": '#255C99', "G": '#F7B32B', "T": '#D62839'}

# logomaker colorscheme
LOGO_COLORS2= {
		'G': [1, .65, 0],
		'T': [1, 0, 0],
		'C': [0, 0, 1],
		'A': [0, .5, 0]
	}

CHAR_IGNORE = ['QWERYUIOPSDFHJKLZXVBNM']


#### utils.py -----------------------------------------------------------------------------------------------
# Author: Selin Jessa
# 2024



INPUTLEN = 2114
OUTPUTLEN = 1000
SHIFT_REL_INPUT = np.int64((2114 - 1000) / 2)




def plot_motif(mat, type="CWM", title=None, ax=None, plot_IC=False, figsize=(6, 2), spines_visible = False):
	"""
	Plot a motif logo.

	Args:
		mat (np.array): The motif matrix, with shape (4, L) where L is the motif length.
	"""

	if ax is None:
		plt.figure(figsize=figsize)
		ax = plt.subplot(111)

	# do IC-scaling if type is PWM
	if type == "PPM" and plot_IC:
		mat_df = logomaker.transform_matrix(mat.T, from_type='probability', to_type='information')
		type = "PPM IC (bits)"
	else:
		mat_df = pd.DataFrame(mat.T, columns=["A", "C", "G", "T"])
	
	# tangermeme.plot.plot_logo(mat, ax=ax)
	logo = logomaker.Logo(mat_df, ax=ax)
	logo.style_spines(visible=spines_visible)
	
	ax.set_title(title)
	ax.set_ylabel(type)
	plt.tight_layout()
	
	if ax is None:
		plt.show()




def trim_cwm(cwm, trim_threshold=0.3, trim_min_length=3, flank=0):
    """
    Trim a CWM based on the contributions of the sequence to the pattern.
    This is adapted from Jacob Schreiber / tf-modiscolite
    (https://github.com/jmschrei/tfmodisco-lite/blob/3c6e38f3ad5df80c55bd4e8c7c2a531ee0a2b316/modiscolite/report.py#L721)
    and Ryan Zhao. The main difference between this and trim_ppm functions used is that here, we 
    trim based on the contributions directly, rather than on the probabilities. 

    Args:
        cwm (np.array): The CWM (contribution scores) of the sequence to the pattern.
        trim_threshold (float): The threshold for trimming; the minimum contribution score to keep a base.
        trim_min_length (int): The minimum length of the trimmed CWM.
    """
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.where(score >= trim_thresh)[0]

    # the start is the start of the passing idx - flank, or if the flank makes this extend into negative positions,
    # then the start is 0
    imp_start = np.max([np.min(pass_inds)-flank, 0])
    
    # the end is the end of the passing idx + flank, or if the flank makes this extend beyond the length of the seq,
    # then the end is the last position
    imp_end = np.min([np.max(pass_inds) + 1 + flank, cwm.shape[0]-1])
    trimmed = cwm[imp_start: imp_end]

    # can be None if no base has prob>t
    return trimmed, imp_start, imp_end



def get_high_affinity_seq(cwm):
	"""
	Function to get the highest affinity sequence from a CWM
	by taking the base with highest absolute contributions at each position.

	Args:
		cwm (np.array, shape L, 4): The CWM (contribution scores) of the sequence
	"""

	# get the highest affinity sequence
	high_affinity_seq = "".join([["A", "C", "G", "T"][np.argmax(np.abs(cwm[i]))] for i in range(cwm.shape[0])])

	return high_affinity_seq


HIT_DTYPE_DICT = {
	"seqnames": "str",
	"start": "int",
	"end": "int",
	"width": "int",
	"strand": "str",
	"start_untrimmed": "int",
	"end_untrimmed": "int",
	"motif_name": "str",
	"source": "str",
	"hit_coefficient": "float64",
	"hit_correlation": "float64",
	"hit_importance": "float64",
	"peak_name": "str",
	"peak_id": "int",
	"motif_name_unlabeled": "str",
	"pattern_class": "str",
	"distToGeneStart": "int",
	"nearestGene": "str",
	"peakType": "str",
	"distToTSS": "int",
	"nearestTSS": "str",
	"GC": "float64",
	"N": "int",
	"distToPeakSummit": "int"
}


def get_rel_hit_positions(hits, peaks, shift=SHIFT_REL_INPUT, flank=0):
	"""
	This function takes in a subset of a hits dataframe (which contain the absolute
	genomic coordinates of each hit), and the peak coordinatess ataframe, and gets the 
	positions of each hit relative to the input window (length 2114) and the output
	window (1000), to make it easy to do experiments on regions centered at hits.

	Returns
	-------
	np.ndarray
		The start positions of the hit relative to the input window (len 2114).
	np.ndarray
		The end positions of the hit relative to the input window (len 2114).
	np.ndarray
		The start positions of the hit relative to the output window (len 1000).
	np.ndarray
		The end positions of the hit relative to the output window (len 1000).

	"""

	print(hits.shape) 

	# get peak idx from hits
	peak_ids = hits['peak_id'].values
	peaks_subset = peaks.loc[peak_ids, :]

	# cast start, end columns in peaks df to int64
	peaks_subset['start'] = peaks_subset['start'].astype('int64')
	peaks_subset['end'] = peaks_subset['end'].astype('int64')

	# the relative start position of the motif within the peak
	hit_starts = hits['start'].values - peaks_subset['start'].values

	# the relative end position of the motif within the peak
	hit_ends = hits['end'].values - peaks_subset['start'].values

	# then get the hit start/end relative to the coordinates of the contribs, since they 
	# are length 2114 while the peaks and predictions are length 1000
	hit_starts2 = hit_starts + shift - flank
	hit_ends2 = hit_ends + shift + flank

	# make a dict
	hit_coords = {
		"start_input": hit_starts2,
		"end_input": hit_ends2,
		"start_output": hit_starts,
		"end_output": hit_ends
	}

	# return coords on the input len (2114), then coords on the output len (1000)
	return 	hit_coords


#### wrappers.py --------------------------------------------------------------------------------------------
# Author: Selin Jessa
# 2024 


class BPNetWrapper(torch.nn.Module):
	"""A wrapper class that returns counts and transformed profiles
	from a BPNet model.

	This class takes in a trained model, specifically expecting a BPNet model
	(i.e. `bpnet.h5`), _not_ the full accessibility model.

	It returns outputs the same shape as the BPNet model, but with the predicted
	profile logits softmaxed and scaled by the exponent of the predicted counts.
	This is for convenience when plotting, since in e.g. bigwigs, we are inspecting
	the scaled profile, not the unnormalized logits.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, X):
		
		y = self.model(X)

		# predicted profile logits
		y_profile = y[0]

		# predicted log counts
		y_counts = y[1]
			  
		# softmax the logits to get profile probabilities
		y_profile = torch.nn.functional.softmax(y_profile, dim = -1)

		# scale the profile by the exponentiated predicted ln(counts)
		y_profile = y_profile * torch.exp(y_counts).unsqueeze(2)

		return y_profile, y_counts


class ZeroControlWrapper(torch.nn.Module):
	"""A wrapper class that supplies an all-zero control track to a model
	trained with control/bias tracks (e.g. via BasePairNet.from_bpnet).

	This mirrors the original TF bpnet pipeline (bpnet/cli/shap_scores.py):
	control/bias inputs are zero-filled whenever no real control bigwig
	applies to the sequence being scored -- e.g. its DeepSHAP background
	distribution always zeros every non-sequence channel, regardless of
	whether the foreground sequence had real control data. This lets callers
	(predict, space, CountWrapper for DeepLIFT/SHAP) treat the wrapped model
	as a single-input model, e.g. for synthetic sequences during in silico
	marginalization where no real experimental control signal exists.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped, expecting an `n_control_tracks` attribute
		(e.g. a BasePairNet model).
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, X, X_ctl=None, **kwargs):
		X_ctl = torch.zeros(X.shape[0], self.model.n_control_tracks, X.shape[2], dtype=X.dtype, device=X.device)
		return self.model(X, X_ctl)


class ChromBPNetWrapper(torch.nn.Module):
	"""A wrapper class that returns counts and transformed profiles
	from a bias-corrected ChromBPNet model.

	This class takes in a trained model, specifically expecting a BPNet model
	from a trained bias-corrected ChromBPNet model (i.e. `chrombpnet_nobias.h5`),
	_not_ the full accessibility model.

	It returns outputs the same shape as the BPNet model, but with the predicted
	profile logits softmaxed and scaled by the exponent of the predicted counts.
	This is for convenience when plotting, since in e.g. bigwigs, we are inspecting
	the scaled profile, not the unnormalized logits.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, X):
        
		y = self.model(X)

		# predicted profile logits
		y_profile = y[0]

		# predicted log counts
		y_counts = y[1]
              
		# softmax the logits to get profile probabilities
		y_profile = torch.nn.functional.softmax(y_profile, dim = -1)

		# scale the profile by the exponentiated predicted ln(counts)
		y_profile = y_profile * torch.exp(y_counts).unsqueeze(2)

		return y_profile, y_counts
	



def reshape_folds(y):
	"""
	Separates a list of [profile, counts] lists, one for each of the model folds, into two separate lists:
	one for profiles and one for counts, with each element corresponding to a fold, then stacks the outputs.

	Args:
		y (list): A list where each element is a list containing two items:
		[profile (torch.Tensor or similar), counts (torch.Tensor or similar)].

	Returns:
		tuple: A tuple containing two lists:
				(profiles (list), counts (list))
				profiles: A list of the profile elements from the input data.
				counts: A list of the counts elements from the input data.
	"""

	profiles = torch.stack([item[0] for item in y], dim=0)
	counts = torch.stack([item[1] for item in y], dim=0)
	
	return profiles, counts


#### eval.py ------------------------------------------------------------------------------------------------
# Author: Selin Jessa
# 2024


def mean_across_sequences(y):
    """
    Compute the mean across sequences (dimension 1) for each fold.

    Parameters
    ----------
    tensor: torch.Tensor
        Input tensor where the first two dimensions are expected to be folds and sequences,
        shape (5, sequences, ...).

    Returns
    -------
    mean_seq: torch.Tensor
        Tensor of means across sequences, shape (folds, positions, ...).
    """
    y_across_seqs = torch.mean(y, dim=1)
    return y_across_seqs






def summarize_across_folds(tensor):
    """
    Compute the mean and standard deviation across folds (dimension 0).

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor where the first dimension is expected to correspond to (folds, ...).

    Returns
    -------
    mean_folds: torch.Tensor
        Tensor of means across folds, shape (positions).

    sd_folds: torch.Tensor
        Tensor of standard deviations across folds, shape (positions).
    """
    
    mean_folds = torch.mean(tensor, dim=0)
    sd_folds = torch.std(tensor, dim=0)
    return mean_folds, sd_folds


#### hdf5_io.py ---------------------------------------------------------------------------------------------
# Deliberately *not* using hdf5plugin's blosc/zstd filter here: it's a
# precompiled native codec that crashed with SIGILL ("Illegal instruction")
# on a Sherlock compute node mid-run, almost certainly a CPU-instruction
# (AVX2/AVX512) mismatch between the wheel's build target and that node's
# CPU -- a well-documented failure class for compiled Python wheels on
# heterogeneous HPC clusters. gzip+shuffle is a core HDF5 filter (pure
# portable C, no SIMD dispatch, no plugin registration needed to read it
# back), at a modest compression cost versus zstd (~7-8x vs ~9x measured).
_GZIP_LEVEL = 9


def _to_numpy(x, dtype=None):
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def _write_dataset(group, name, array):
    group.create_dataset(name, data=array, chunks=True, shuffle=True,
        compression="gzip", compression_opts=_GZIP_LEVEL)


def save_array_h5(path, array, name="data", dtype=None, attrs=None):
    """
    Save a single tensor/array to an HDF5 file with compression.

    `attrs` is an optional dict of scalar/string metadata written as root-level HDF5
    attributes, so a small derived product can carry the coordinates needed to
    interpret it instead of requiring the caller to re-derive them.
    """
    arr = _to_numpy(array, dtype)
    with h5py.File(path, "w") as f:
        _write_dataset(f, name, arr)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value


def load_array_h5(path, name="data", as_torch=True):
    with h5py.File(path, "r") as f:
        arr = f[name][:]
    return torch.from_numpy(arr) if as_torch else arr


def save_pred_dict_h5(path, preds, profile_dtype=np.float16, counts_dtype=None):
    """
    Save a dict[key] -> {"y_before": (profile, counts), "y_after": (profile, counts)}
    structure (as produced by test_spacing_per_fold/reshape_folds) to HDF5,
    one group per key/stage and one dataset per profile/counts array.

    `profile` is the largest artifact here (it carries the full per-spacing
    profile track) and only ever feeds plotting downstream, so it defaults to
    float16. `counts` feeds the effect-size/Wilcoxon statistics, so it stays
    at full precision by default.
    """
    with h5py.File(path, "w") as f:
        for key, stages in preds.items():
            key_group = f.create_group(key)
            for stage, (profile, counts) in stages.items():
                stage_group = key_group.create_group(stage)
                _write_dataset(stage_group, "profile", _to_numpy(profile, profile_dtype))
                _write_dataset(stage_group, "counts", _to_numpy(counts, counts_dtype))


def load_pred_dict_h5(path, as_torch=True):
    preds = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            preds[key] = {}
            for stage in f[key].keys():
                stage_group = f[key][stage]
                profile, counts = stage_group["profile"][:], stage_group["counts"][:]
                if as_torch:
                    profile, counts = torch.from_numpy(profile), torch.from_numpy(counts)
                preds[key][stage] = (profile, counts)
    return preds


def reduce_pred_dict(preds):
    """
    Collapse a full {key: {stage: (profile, counts)}} prediction dict down to the
    only two projections anything downstream actually consumes:

      profile_seqmean  mean over sequences, FOLDS KEPT. Feeds the profile mean
                       lines and, critically, the across-fold SD error bands
                       (fold-level SD is unrecoverable once folds are averaged).
      counts_foldmean  mean over folds, SEQUENCES KEPT. Feeds the per-sequence
                       effect sizes: the paired Wilcoxon test and the effect
                       boxplot both need the individual sequence values.

    Both reductions are exact rather than approximate: the mean is linear, so
    reducing here gives bit-comparable results to reducing at the point of use.
    Nothing downstream ever needs the full (folds x sequences) cross-product.
    """
    reduced = {}
    for key, stages in preds.items():
        reduced[key] = {}
        for stage, (profile, counts) in stages.items():
            reduced[key][stage] = {
                "profile_seqmean": profile.mean(dim=1),
                "counts_foldmean": counts.mean(dim=0),
            }
    return reduced


def save_nested_h5(path, data, dtype=None):
    """
    Save an arbitrarily nested dict of tensors/arrays, one HDF5 group per dict
    level and one dataset per leaf.
    """
    with h5py.File(path, "w") as f:
        _save_nested(f, data, dtype)


def _save_nested(group, data, dtype):
    for name, value in data.items():
        if isinstance(value, dict):
            _save_nested(group.create_group(name), value, dtype)
        else:
            _write_dataset(group, name, _to_numpy(value, dtype))


def load_nested_h5(path, as_torch=True):
    with h5py.File(path, "r") as f:
        return _load_nested(f, as_torch)


def _load_nested(group, as_torch):
    out = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            out[name] = _load_nested(value, as_torch)
        else:
            arr = value[:]
            out[name] = torch.from_numpy(arr) if as_torch else arr
    return out


#### syntax_helpers.py --------------------------------------------------------------------------------------
# plotting






# CALCULATIONS ---------------------------------------------------------------------

def is_palindrome(seq):
	"""
	Check if a sequence is a palindrome.
	"""

	return seq == reverse_complement(seq)



def calc_effect_Z_score(effects):
	"""
	Calculate the Z-scores for the marginal effects of the motif pairs across
	spacings and orientations.

	Z = (X - mean) / std
	"""

	mean = np.mean(effects)
	std = np.std(effects)
	z = (effects - mean) / std

	return z



def calc_effect_entropy(effects):
	"""
	Calculate the entropy of the effects for the motif pairs across
	spacings and orientations.

	We expect *lower* entropy when the uncertainty is low, meaning there is a
	specific orientation and spacing that is favored.

	We expect a *higher* entropy when the uncertainty is high, because the
	distribution is more uniform.
	"""

	# convert log fold changes to probabilities so that they sum to 1

	# shift in case any are below 0
	effects_shifted = effects - np.min(effects)

	# normalize to convert to probabilities
	probabilities = effects_shifted / np.sum(effects_shifted)
		
	# compute entropy
	S = entropy(probabilities)

	return S





# PLOTTING ------------------------------------------------------------------------

def plot_effect_spacing(spacing_dfs, input_motifs, fig_path, n_spaces):
	"""
	Plot the marginal effects for the motif pairs at every spacing in every 
	orientation during in silico marginalization, as a bar plot.
	"""

	n_col = len(input_motifs)
	fig, axes = plt.subplots(n_col, 1, figsize=(10, n_col * 3))
	plt.set_loglevel('WARNING') 

	# iterate over the motif pairs and plot
	for i, key in enumerate(input_motifs.keys()):

		# get the effects from the dataframe, plotted against the actual
		# (possibly negative, for overlapping motifs) spacing values rather
		# than their position in the dataframe, since those diverge once a
		# sweep's spacings don't start at zero
		key_df = spacing_dfs[spacing_dfs.key == key]
		spacings = key_df.spacing.values
		effects = key_df.effect_mean.values
		effects_sd = key_df.effect_sd.values

		# plot bar and errorbars
		axes[i].bar(spacings, effects, alpha=1, label=key, color="red")
		axes[i].errorbar(spacings, effects, yerr=effects_sd, fmt='none', ecolor='black', elinewidth=1, capsize=0)

		# set titles and labels
		axes[i].set_title(f"{key}: {input_motifs[key][0]}, {input_motifs[key][1]}")
		axes[i].set_xlabel("Spacing between motifs (bp)", fontsize=10)
		axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
		axes[i].xaxis.set_minor_locator(ticker.AutoMinorLocator())
		axes[i].tick_params(axis='x', which='both', length=2, direction='out', top=False, bottom=True)
		axes[i].set_xlim([spacings.min(), spacings.max()])
		axes[i].set_ylim([0, 3])
		# axes[i].legend()

	axes[0].set_ylabel("Mean marginal effect (ln counts)")

	# joint title
	plt.suptitle("Marginal effects for motif pairs during in silico marginalization", fontsize=12)
	plt.tight_layout()
	plt.savefig(fig_path, dpi=300);



def plot_effect_boxplot(df_pred_effect, best_orientation, best_spacing, fig_path):
	"""
	Plot the marginale effects for the motif pairs at the best spacing and orientation,
	as well as the fold changes for the solo effects and sum of independent effects.
	"""

	df_pred_effect_long = df_pred_effect.melt(id_vars = "example")
	df_pred_effect_long.head()	

	# boxplot of predicted log counts
	# set order of the variables
	df_pred_effect_long["variable"] = pd.Categorical(
		values = df_pred_effect_long["variable"],
		categories = ['Joint', 'A + B', 'A', 'B'],
		ordered = True)

	p = (	
		ggplot(df_pred_effect_long, aes(x = "variable", y = "value"))
		+ geom_boxplot(aes(fill = "variable"))
		+ ggtitle(f"In silico marginalizations for motifs individually and togther \nBest spacing/orientation: {best_orientation} @ {best_spacing} bp")
		+ ylab("Predicted marginal effect (ln counts)")
		+ xlab("Motifs")
		+ scale_fill_manual(values = {"Joint": 'red',
									"A + B": '#741fb5',
									"A": "blue",
									"B": '#4e54ed'})
		+ geom_hline(yintercept=0, linetype="dashed", color = "black")
		+ theme(axis_text_x=element_text(angle = 45),
				panel_background=element_rect(fill="white"),
				panel_border=element_rect(color = "black", size = 0.5),
				figure_size=(6, 4))
		)

	p.save(filename = fig_path, width=6, height=4, units='in', dpi=300, verbose=False);



def plot_effect_zscores(df_z, input_motifs, fig_path):
	"""
	Plot Z-scored marginal effects for motif pairs across spacings and orientations.
	"""

	palette_blue = seaborn.color_palette("Blues", len(input_motifs)).as_hex()
	palette = { key: palette_blue[i] for i, key in enumerate(input_motifs.keys()) }

	# set order of the levels in orientation as the keys of the input motifs
	df_z["orientation"] = pd.Categorical(df_z["orientation"], categories = list(input_motifs.keys()), ordered = True)

	p = ( ggplot(df_z, aes(x = "orientation", y = "Z"))
		+ geom_jitter(aes(fill = "orientation"), alpha = 0.6, size = 4, width = 0.1)
		+ ggtitle("Z-scores for marginal effects across \nspacings/orientations during ISM")
		+ ylab("Z-score")
		+ xlab("Orientation")
		+ scale_fill_manual(values = palette)
		+ geom_hline(yintercept=[-3, -2, 2, 3], linetype="dashed", color = "black")
		+ theme(axis_text_x=element_text(angle = 45),
				panel_background=element_rect(fill="white"),
				panel_border=element_rect(color = "black", size = 0.5),
				figure_size=(6, 4))
	)

	p.save(filename = fig_path, width=6, height=4, units='in', dpi=300, verbose=False);



def round_up_to_sig_figs(x, sig=1):
	"""
	Round x up to `sig` significant figures: 0.013 -> 0.02, 0.0071 -> 0.008, 1.3 -> 2, 42 -> 50.

	Used to pick a y-axis top that is a clean tick value just above the plotted data, so
	profiles stay legible regardless of the model's absolute signal scale. A fixed axis top
	cannot do this: BPNet TF ChIP per-bp signal over nonpeak background runs orders of
	magnitude below ChromBPNet ATAC, so any floor large enough for one squashes the other
	flat against the x-axis.
	"""

	x = float(x)
	if not np.isfinite(x) or x <= 0:
		return 1.0

	# round before ceil so floating-point dust (3.0000000000000004) doesn't bump a
	# value that is already exactly on a tick up to the next one
	magnitude = 10.0 ** (np.floor(np.log10(x)) - (sig - 1))
	return float(np.ceil(round(x / magnitude, 9)) * magnitude)



def plot_profiles_spacing(profile_preds_summarized, input_motifs, fig_path, spacing_offsets=None, y_max_floor=1, y_max_sig_figs=None):
	"""
	Plot the predicted profile for the background and edited sequence for insertions of
	motif pairs at every orientation, for a window of ten spacings.

	By default (`spacing_offsets=None`), this shows the first ten swept positions (0-9 bp),
	matching prior behavior for sweeps that start at zero spacing. If `spacing_offsets` is
	given -- a dict mapping each key in `input_motifs` to the spacing value at position 0 of
	that key's swept spacing axis, i.e. what `test_spacing_per_fold` calls `spacing_offset` --
	this instead centers the shown window on -5..+4 bp, since "the first ten positions" would
	otherwise land entirely in the deep-overlap region for a sweep with a very negative offset.

	Two ways to set the y-axis top, in both cases shared by every panel in the figure so
	spacings stay visually comparable:

	`y_max_sig_figs` (preferred): fully auto-scale, rounding the largest displayed value up
	to that many significant figures (1 -> a single clean tick above the data). Ignores
	`y_max_floor`. Use this when the absolute signal scale is not known in advance.

	`y_max_floor`: the minimum top of the y-axis, auto-scaling past it whenever the plotted
	signal is larger. Profiles are in predicted counts per bp (softmaxed profile logits scaled
	by exp(predicted log counts)), so a fixed floor is model-dependent -- ~1 for ChromBPNet
	ATAC, ~0.1 for BPNet TF ChIP -- and any floor above a given model's signal flattens its
	profile against the x-axis. Default is 1, matching prior behavior.
	"""

	range_idx = range(250, 750)

	# choose which ten swept positions to show for each key, and what to label them.
	# with no offset info, show the first ten positions as before; with per-key offsets
	# (sweeps that may start at a negative spacing), show a fixed -5..+4 bp window
	# instead, converting each label back to a position via that key's offset.
	# hoisted out of the plotting loop below so the shared y-axis scale can be taken
	# over exactly the spacings that end up on screen
	cols_per_key = {}
	for key in input_motifs.keys():
		n_spaces_key = profile_preds_summarized[key]['y_after_means'].shape[0]
		if spacing_offsets is None:
			col_labels = list(range(10))
			col_idxs = list(col_labels)
		else:
			offset = spacing_offsets[key]
			col_labels = list(range(-5, 5))
			col_idxs = [min(max(label - offset, 0), n_spaces_key - 1) for label in col_labels]
		cols_per_key[key] = (col_labels, col_idxs)

	# one shared y-axis scale for the whole figure, taken over everything actually drawn:
	# the background trace plus the edited traces at the ten displayed spacings, within the
	# displayed x-window. Deliberately not over all swept spacings -- an off-screen spacing
	# with much larger signal would otherwise squash every visible panel flat.
	maxes = []
	for key in input_motifs.keys():
		_, col_idxs = cols_per_key[key]
		maxes.append(float(torch.max(profile_preds_summarized[key]['y_before_means'][:, 250:750])))
		maxes.append(float(torch.max(profile_preds_summarized[key]['y_after_means'][col_idxs][:, :, 250:750])))

	data_max = float(np.max(maxes))
	if y_max_sig_figs is not None:
		y_max = round_up_to_sig_figs(data_max, y_max_sig_figs)
	else:
		y_max = np.max([data_max, y_max_floor])

	# make ten small subplots in a row, for as many rows as input motifs
	fig, ax = plt.subplots(len(input_motifs), 10, figsize=(18, len(input_motifs)*2.5))
	plt.set_loglevel('WARNING') 

	# i indexes the motif pair
	for i, key in enumerate(input_motifs.keys()):

		y_before_mean = profile_preds_summarized[key]['y_before_means']
		y_before_sd = profile_preds_summarized[key]['y_before_sd']
		y_after_mean = profile_preds_summarized[key]['y_after_means']
		y_after_sd = profile_preds_summarized[key]['y_after_sd']

		# get max profiles across all spacings
		# max_profile = torch.max(y_after_mean)
		# y_max = np.max([max_profile, 1])

		# profiles may carry more than one strand/task (e.g. +/- strand bpnet
		# profiles, shape (..., n_strands, position)); plot each strand on the
		# same axes in its own color instead of collapsing them. y_before has
		# no spacing axis, so it's indexed only by strand, not by j.
		n_strands = y_before_mean.shape[0]
		before_colors = ["black", "dimgray", "silver"]
		after_colors = ["red", "blue", "orange"]

		col_labels, col_idxs = cols_per_key[key]

		# j indexes the column; idx indexes the swept spacing position
		for j in range(10):
			idx = col_idxs[j]

			for s in range(n_strands):
				before_color = before_colors[s % len(before_colors)]
				after_color = after_colors[s % len(after_colors)]
				before_label = "background" if n_strands == 1 else f"background (strand {s})"
				after_label = f"edited \n{key}" if n_strands == 1 else f"edited \n{key} (strand {s})"

				ax[i, j].plot(y_before_mean[s][range_idx], label=before_label, color=before_color, alpha=0.2, lw=0.8)
				ax[i, j].fill_between(range(500),
									y_before_mean[s][range_idx] - y_before_sd[s][range_idx],
									y_before_mean[s][range_idx] + y_before_sd[s][range_idx],
									color=before_color, alpha=0.1)

				ax[i, j].plot(y_after_mean[idx, s][range_idx], label=after_label, color=after_color, alpha=0.8, lw=0.8)
				ax[i, j].fill_between(range(500),
									y_after_mean[idx, s][range_idx] - y_after_sd[idx, s][range_idx],
									y_after_mean[idx, s][range_idx] + y_after_sd[idx, s][range_idx],
									color=after_color, alpha=0.1)

			ax[i, j].set_ylim([0, y_max])

			# choose labels for the x-axis, only if it's the last row of plots
			if i!=(len(input_motifs)-1):
				ax[i, j].set_xticks([])
			elif i==(len(input_motifs)-1):
				ax[i, j].set_xticks(ticks = [0, 250, 500], labels = [-250, 0, 250])

			# column titles, if it's the first row of plots
			if i == 0:
				ax[i, j].set_title(f"{col_labels[j]} bp")

			# remove labels for the y-axis, if it's not the first column of plots
			if j != 0:
				ax[i, j].set_yticks([])
				ax[i, j].set_yticks([])

		ax[i, 0].set_ylabel("Avg. predicted profile")
		ax[i, 9].legend()

	# shared title
	fig.suptitle("Predicted profiles for motifs inserted at different spacings", fontsize=12)

	# shared x-axis label
	fig.text(0.5, 0.01, "Position relative to insertion (bp)", ha='center', fontsize=10)

	plt.savefig(fig_path, dpi=300);



def plot_profiles_solo(solo_profile_preds_summarized, profile_preds_summarized, best_orientation, best_spacing, fig_path, spacing_offset=0, y_max_floor=1, y_max_sig_figs=None):
	"""
	Plot the predicted profiles for the best spacing and orientation, as well as the independent effects
	and sum of independent effects.

	`best_spacing` is the true (possibly negative, for overlapping motifs) spacing value, used only for
	the plot label. `spacing_offset` is the spacing value at position 0 of the swept spacing axis (0 for
	a sweep that starts at zero spacing, matching prior behavior); `best_spacing - spacing_offset` is the
	position actually used to index into the swept predictions.

	`y_max_sig_figs` (preferred): fully auto-scale the y-axis, rounding the largest plotted value up
	to that many significant figures (1 -> a single clean tick above the data). Ignores `y_max_floor`.

	`y_max_floor`: the minimum top of the y-axis, auto-scaling past it whenever the plotted signal is
	larger. Profiles are in predicted counts per bp (softmaxed profile logits scaled by exp(predicted
	log counts)), so a fixed floor is model-dependent -- ~1 for ChromBPNet ATAC, ~0.1 for BPNet TF
	ChIP -- and any floor above a given model's signal flattens its profile against the x-axis.
	Default is 1, matching prior behavior.
	"""

	best_position = best_spacing - spacing_offset

	plt.figure(figsize=(7, 4))
	plt.set_loglevel('WARNING')

	# plot the background from one experiment. Profiles with more than one
	# strand/task are all plotted here in the same color as before.
	n_strands = solo_profile_preds_summarized['A']['y_before_means'].shape[0]
	for s in range(n_strands):
		plt.plot(solo_profile_preds_summarized['A']['y_before_means'][s], label="background" if s == 0 else None, color="lightgray", alpha=0.8, lw=0.8)
		plt.fill_between(range(1000),
			solo_profile_preds_summarized['A']['y_before_means'][s] - solo_profile_preds_summarized['A']['y_before_sd'][s],
			solo_profile_preds_summarized['A']['y_before_means'][s] + solo_profile_preds_summarized['A']['y_before_sd'][s], color="lightgray", alpha=0.1)

	# get max across every trace drawn -- the background, the joint profile at the best spacing,
	# and the solo/summed profiles -- since 'A + B' is a sum of two profiles and can exceed the joint
	maxes = [float(torch.max(profile_preds_summarized[best_orientation]['y_after_means'][best_position])),
			float(torch.max(solo_profile_preds_summarized['A']['y_before_means']))]
	for key in ['A', 'B', 'A + B']:
		if key in solo_profile_preds_summarized:
			maxes.append(float(torch.max(solo_profile_preds_summarized[key]['y_after_means'])))

	data_max = float(np.max(maxes))
	if y_max_sig_figs is not None:
		y_max = round_up_to_sig_figs(data_max, y_max_sig_figs)
	else:
		y_max = np.max([data_max, y_max_floor])


	# plot the solo effects
	for key, color in zip(['A', 'B', 'A + B'], ['#4e54ed', 'blue', '#741fb5']):
		for s in range(n_strands):
			plt.plot(solo_profile_preds_summarized[key]['y_after_means'][s], label=key if s == 0 else None, color=color, alpha=0.8, lw=0.8)
			plt.fill_between(range(1000), 
							(solo_profile_preds_summarized[key]['y_after_means'][s] - solo_profile_preds_summarized[key]['y_after_sd'][s]), 
							(solo_profile_preds_summarized[key]['y_after_means'][s] + solo_profile_preds_summarized[key]['y_after_sd'][s]), 
							color=color, alpha=0.1)

	# plot the profile for the best spacing/orientation
	for s in range(n_strands):
		plt.plot(profile_preds_summarized[best_orientation]['y_after_means'][best_position][s], label=f"{best_orientation} @ {best_spacing} bp" if s == 0 else None, color="red", alpha=0.8, lw=0.8)
		plt.fill_between(range(1000),
					(profile_preds_summarized[best_orientation]['y_after_means'][best_position][s] - profile_preds_summarized[best_orientation]['y_after_sd'][best_position][s]),
					(profile_preds_summarized[best_orientation]['y_after_means'][best_position][s] + profile_preds_summarized[best_orientation]['y_after_sd'][best_position][s]),
					color="#ff7077", alpha=0.1)

	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins = 10))
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.tick_params(axis='x', which='both', length=5, direction='out', top=False, bottom=True)

	# put legend outside
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.title("Predicted profiles for joint and independent motifs")
	plt.ylim([0, y_max])
	plt.ylabel("Avg. predicted profile")
	plt.xlabel("Position relative to insertion (bp)")
	plt.xticks(ticks = [0, 500, 1000], labels = [-500, 0, 500])
	plt.savefig(fig_path, dpi=300, bbox_inches='tight');



def plot_effect_solo(solo_dfs, best_orientation, best_spacing, fig_path):
	"""
	Plot the effects for the solo effects and the sum of independent effects
	as bar plots.
	"""

	fig, ax = plt.subplots(1, 1, figsize=(5, 3))
	plt.set_loglevel('WARNING') 

	ax.bar(solo_dfs.key, solo_dfs.effect_mean, alpha=1, label=best_orientation, color = ['#4e54ed', 'blue', '#741fb5'])
	ax.errorbar(solo_dfs.key, solo_dfs.effect_mean, yerr=solo_dfs.effect_sd, fmt='none', ecolor='black', elinewidth=1, capsize=0)

	ax.set_ylabel("Mean marginal effects (ln counts)")
	ax.set_ylim([0, 3])
	plt.title("Marginal effects in silico marginalizations of solo motifs", fontsize=12)

	plt.savefig(fig_path, dpi=300);



def plot_effect_spacing_lines(spacing_df, solo_df, motif_name, fig_path, ymax = 1):

	# plot this as a line plot
	fig, axes = plt.subplots(4, 1, figsize=(13, 11))

	color_pal = {"Joint": 'red',
	"A + B": '#741fb5',
	"A": "blue",
	"B": '#4e54ed'}

	# plot a straight line
	def plot_effect_with_error(ax, x, y, yerr, label, color, alpha=0.1):
		ax.plot(x, y, alpha=1, label=label, color=color)
		ax.fill_between(x, y + yerr, y - yerr, color=color, alpha=alpha)

	# iterate over the motif pairs and plot, against the actual (possibly
	# negative, for overlapping motifs) spacing values rather than their
	# position in the dataframe, since a sweep's per-orientation spacing
	# range need not start at zero
	for i, key in enumerate(spacing_df.key.unique()):

		key_df = spacing_df[spacing_df.key == key]
		spacings = key_df.spacing.values
		effects = key_df.effect_mean.values
		effects_sd = key_df.effect_sd.values

		# plot bar and errorbars
		axes[i].plot(spacings, effects, alpha=1, label=key, color="red")
		axes[i].fill_between(spacings, effects + effects_sd, effects - effects_sd, color = "red", alpha = 0.1)

		# set titles and labels
		axes[i].set_title(f"{key}")
		axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
		axes[i].xaxis.set_minor_locator(ticker.AutoMinorLocator())
		axes[i].tick_params(axis='x', which='both', length=2, direction='out', top=False, bottom=True)
		axes[i].set_xlim([spacings.min(), spacings.max()])
		axes[i].set_ylim([0, ymax])
		# axes[i].legend()

		for solo_key in ["A", "B", "A + B"]:
			solo_row = solo_df[solo_df.key == solo_key]
			plot_effect_with_error(
				axes[i], spacings,
				np.repeat(solo_row.effect_mean.values, len(spacings)),
				np.repeat(solo_row.effect_sd.values, len(spacings)),
				solo_key, color_pal[solo_key],
			)

	axes[1].set_ylabel("Mean marginal effect")
	axes[0].legend()
	axes[3].set_xlabel("Spacing between motifs (bp)", fontsize=10)

	# joint title
	plt.suptitle(f"Mean effects for motif pairs during ISM: {motif_name}", fontsize=12)
	plt.tight_layout()
	plt.savefig(fig_path, dpi = 300);



def plot_mean_contributions(profile_preds_summarized, contributions_mean, coords, input_motifs, best_spacing, best_orientation, fig_path, spacing_offset=0):
	"""
	Plot the mean attributions for the best spacing and orientation. Specifically,
	we plot the average predicted profile across the 1 kb, the average contribution
	scores across the central 1 kb of the input (predicted window), and the zoomed
	in contributions at the center of the predicted window, as a sequence logo.

	This function has some careful handling of starts and end positions, because
	the input window (2,114) and the prediction window (1,000) are not the same.
	So we often want to subset regions _relative_ to one of these windows, for example
	the specific positions where the motifs were inserted.

	`best_spacing` is the true (possibly negative, for overlapping motifs) spacing value, used only for
	the plot title. `spacing_offset` is the spacing value at position 0 of the swept spacing axis (0 for
	a sweep that starts at zero spacing, matching prior behavior); `best_spacing - spacing_offset` is the
	position actually used to index into the swept predictions.
	"""

	best_position = best_spacing - spacing_offset

	fig, ax = plt.subplots(3, 1, figsize = (11, 6))
	plt.set_loglevel('WARNING')

	# plto the background. y_before has no spacing axis, so it is indexed only by
	# strand; profiles with more than one strand/task are all plotted in the same
	# color (background vs. edited coloring is what distinguishes the traces).
	n_strands = profile_preds_summarized[best_orientation]['y_before_means'].shape[0]
	for s in range(n_strands):
		ax[0].plot(profile_preds_summarized[best_orientation]['y_before_means'][s], linewidth=1, label = "mean prediction (background)" if s == 0 else None, color = 'lightgray')
		ax[0].fill_between(range(1000),
						profile_preds_summarized[best_orientation]['y_before_means'][s] - profile_preds_summarized[best_orientation]['y_before_sd'][s],
						profile_preds_summarized[best_orientation]['y_before_means'][s] + profile_preds_summarized[best_orientation]['y_before_sd'][s],
				color="lightgray", alpha=0.1, label = "sd across folds" if s == 0 else None)

	# plot the predicted profile for the top hit
	for s in range(n_strands):
		ax[0].plot(profile_preds_summarized[best_orientation]['y_after_means'][best_position][s], linewidth=1, label = "mean prediction (edited)" if s == 0 else None, color = 'red')
		ax[0].fill_between(range(1000),
						profile_preds_summarized[best_orientation]['y_after_means'][best_position][s] - profile_preds_summarized[best_orientation]['y_after_sd'][best_position][s],
						profile_preds_summarized[best_orientation]['y_after_means'][best_position][s] + profile_preds_summarized[best_orientation]['y_after_sd'][best_position][s],
				color="red", alpha=0.1, label = "sd across folds" if s == 0 else None)
	
	# labels and axes
	ax[0].set_ylabel("Predicted profile")
	ax[0].legend(fontsize = 8)
	ax[0].set_xticks(ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999],
					 labels = [-500, -400, -300, -200, -100, 0, +100, +200, +300, +400, +500])

	# semi-transparent rectangle to indicate the zoom at the bottom
	ax[0].axvspan(coords['start_A'] - SHIFT_REL_INPUT, coords['end_A'] - SHIFT_REL_INPUT, alpha=0.4, color='yellow')
	ax[0].axvspan(coords['start_B'] - SHIFT_REL_INPUT, coords['end_B'] - SHIFT_REL_INPUT, alpha=0.4, color='yellow')

	# contribs
	# plot the sum of contributions across nucleotides, per position
	ax[1].plot(contributions_mean.sum(0)[ MIDPOINT - 500 : MIDPOINT + 500 ], linewidth=1, label = "contrib scores", color = "blue")
	ax[1].set_ylabel("Contrib.")
	ax[1].legend(fontsize = 8)
	ax[1].axvspan(coords['start_A'] - SHIFT_REL_INPUT, coords['end_A'] - SHIFT_REL_INPUT, alpha=0.4, color='yellow')
	ax[1].axvspan(coords['start_B'] - SHIFT_REL_INPUT, coords['end_B'] - SHIFT_REL_INPUT, alpha=0.4, color='yellow')
	ax[1].set_xticks(ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999],
				  	 labels = [-500, -400, -300, -200, -100, 0, +100, +200, +300, +400, +500])

	tangermeme.plot.plot_logo(np.float64(contributions_mean), start = coords['zoom_start'], end = coords['zoom_end'], ax=ax[2])
	# adjust coords by 0.5 to include whole nucleotides in the highlight
	ax[2].axvspan(coords['start_A'] - coords['zoom_start'], coords['end_A'] - coords['zoom_start'], alpha=0.3, color='yellow')
	ax[2].axvspan(coords['start_B'] - coords['zoom_start'], coords['end_B'] - coords['zoom_start'], alpha=0.3, color='yellow')
	ax[2].set_ylabel("Contrib.")
	ax[2].set_xticks(
		ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99],
		labels = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])

	# put a title at the very top
	plt.suptitle(f"Attributions for {best_orientation} @ {best_spacing} bp: {input_motifs[best_orientation][0]}, {input_motifs[best_orientation][1]}", fontsize = 12)
	plt.xlabel("Genomic position")
	plt.tight_layout()

	plt.savefig(fig_path, bbox_inches='tight', dpi = 100);



def plot_contributions_heatmap(contribs_sum_mean_across_folds, coords, best_orientation, best_spacing, fig_path):
	"""
	Plot a heatmap of the contribution scores in the central 100bp, for each sequence.
	"""

	# make a heatmap
	fig, ax = plt.subplots(1, 1, figsize = (5, 7))
	plt.set_loglevel('WARNING') 
	seaborn.heatmap(contribs_sum_mean_across_folds[:, coords['zoom_start']:coords['zoom_end']], cmap = "magma", ax = ax)

	# highlight regions w/ motifs inserted with lines
	ax.axvline(x = coords['start_A'] - coords['zoom_start'], color = "black", lw = 1, ls = "--")
	ax.axvline(x = coords['end_A'] - coords['zoom_start'], color = "black", lw = 1, ls = "--")
	ax.axvline(x = coords['start_B'] - coords['zoom_start'], color = "black", lw = 1, ls = "--")
	ax.axvline(x = coords['end_B'] - coords['zoom_start'], color = "black", lw = 1, ls = "--")

	plt.ylabel("Sequence")
	plt.xlabel("Genomic position (central 100 bp)")
	plt.yticks([])

	plt.xticks(
		ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99],
		labels = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])

	plt.title(f"Contributions for {best_orientation} @ {best_spacing} bp \n(mean across 5 folds per sequence)")
	plt.tight_layout()
	plt.savefig(fig_path, dpi = 300);



def plot_profile_heatmap_z(preds, coords, best_orientation, best_spacing, fig_path):
	"""
	Plot a heatmap of the predicted profiles for the best spacing and orientation,
	z-scored per sequence.
	"""

	# let's do the same for the predictions
	fig, ax = plt.subplots(1, 1, figsize = (8, 7))
	plt.set_loglevel('WARNING') 
	seaborn.heatmap(preds, cmap = "coolwarm", ax = ax, center = 0)

	# dashed lines to highlight edited regions
	ax.axvline(x = coords['start_A'] - SHIFT_REL_INPUT, color = "black", lw = 1, ls = "--")
	ax.axvline(x = coords['end_B'] - SHIFT_REL_INPUT, color = "black", lw = 1, ls = "--")

	plt.xlabel("Genomic position (1 kb)")
	plt.ylabel("Sequence")

	plt.xticks(
		ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999],
		labels = [-500, -400, -300, -200, -100, 0, +100, +200, +300, +400, +500])

	# remove ticks
	plt.yticks([])
	plt.title(f"Predicted profile for {best_orientation} @ {best_spacing} bp (z-score per sequence)")
	plt.savefig(fig_path, dpi = 300);
