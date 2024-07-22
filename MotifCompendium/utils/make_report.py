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


def plot64_from_motif(motif_df, face_color="white"):
	fig, ax = plt.subplots(figsize=(6, 2), facecolor=face_color)
	logomaker.Logo(motif_df, ax=ax)
	ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
	ax.set_axis_off()
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	plt.close(fig)
	buf.seek(0)
	return base64.b64encode(buf.read()).decode('utf-8')

def plot64_from_motifs_parallel(motifs, average, parallel):
	plot_names = [m[0] for m in motifs]
	motif_plot_data = [m[1] for m in motifs]
	if parallel:
		with multiprocessing.Pool(processes=None) as p:
			plots = p.map(plot64_from_motif, motif_plot_data)
	else:
		plots = [plot64_from_motif(x) for x in motif_plot_data]

	# Average
	if average:
		motifs_concat = pd.concat(motif_plot_data)
		average_motif = motifs_concat.groupby(motifs_concat.index).mean()
		average_plot = plot64_from_motif(average_motif, face_color="palegreen")
		plot_names.insert(0, "AVERAGE")
		plots.insert(0, average_plot)

	assert(len(plot_names) == len(plots))
	return [(plot_names[i], plots[i]) for i in range(len(plots))]

# Function to generate HTML report
def generate_report(data_dict, output_file, average, parallel):
	# data_dict is cluster name --> motifs in cluster
	# Use Agg backend
	current_backend = matplotlib.get_backend()
	matplotlib.use('Agg')

	# Make plots
	plot_dict = {x: plot64_from_motifs_parallel(m, average, parallel) for x, m in data_dict.items()}

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
	
