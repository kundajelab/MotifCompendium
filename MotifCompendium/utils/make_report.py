from jinja2 import Environment, FileSystemLoader
import logomaker

import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

import base64
import io
import os
import multiprocessing


def plot64_from_motif(plotter_input):
	motif_df, face_color = plotter_input
	fig, ax = plt.subplots(figsize=(6, 2), facecolor=face_color)
	logomaker.Logo(motif_df, ax=ax)
	ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
	ax.set_axis_off()
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	plt.close(fig)
	buf.seek(0)
	return base64.b64encode(buf.read()).decode('utf-8')

def plot64_from_motifs_parallel(motifs, max_parallel):
	max_cpus = min(max_parallel, multiprocessing.cpu_count())
	if max_cpus > 1:
		with multiprocessing.Pool(processes=max_cpus) as p:
			return p.map(plot64_from_motif, motifs)
	else:
		return [plot64_from_motif(x) for x in motifs]

# Function to generate HTML report
def generate_report(cluster_indices, plotter_inputs, plot_names, output_file, max_parallel):
	# Use Agg backend
	current_backend = matplotlib.get_backend()
	matplotlib.use('Agg')

	# Make plots
	plots = plot64_from_motifs_parallel(plotter_inputs, max_parallel)

	# Rearrange data
	plot_dict = dict()
	for c_name, c_idx_start, c_idx_end in cluster_indices:
		plot_dict[c_name] = [(plot_names[i], plots[i]) for i in range(c_idx_start, c_idx_end)]

	# Create Jinja2 environment
	current_dir = os.path.dirname(os.path.abspath(__file__))
	env = Environment(loader=FileSystemLoader(current_dir))

	# Load HTML template
	template = env.get_template('report_template.html')

	# Render HTML with data
	rendered_html = template.render(data=plot_dict, sorted=sorted)

	# Write HTML to file
	with open(output_file, 'w') as f:
		f.write(rendered_html)

	# Switch back to original backend
	matplotlib.use(current_backend)
	
