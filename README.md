# MotifCompendium

**Please read this README.**

The MotifCompendium package provides a framework for clustering, annotating, and analyzing motifs efficiently at large scales.

**This package is being actively developed.**

**WARNING: If you have an old MotifCompendium object, you may need to load it with `MotifCompendium.load_old_compendium(file_loc)` rather than `MotifCompendium.load(file_loc, safe)`.**

README sections:

- [Installation](#installation)
- [Getting started](#getting-started)
- [Problems with MotifCompendium](#problems)

## Installation

These are instructions for creating a conda environment within which you can run MotifCompendium code. The following instructions assume you have conda installed in a Linux environment.

If you are creating an environment for the first time, follow:

1. Clone this GitHub repository: `git clone https://github.com/kundajelab/MotifCompendium.git`.
2. Move into the MotifCompendium directory: `cd MotifCompendium`.
3. Create a conda environment:
    - If you have GPU access and want to run MotifCompendium with a GPU, run `conda env create -f environment_gpu.yml`.
    - If you do not have GPU access or you want to run MotifCompendium only with a CPU, run `conda env create -f environment.yml`.
4. Activate the environment with `conda activate motifcompendium-gpu` or `conda activate motifcompendium` depending on which environment you built.
5. In the MotifCompendium directory run `pip install -e .`.

If you would like to update an existing conda environment, follow:

1. Run `conda env update -f environment_gpu.yml --prune` or `conda env update -f environment.yml --prune` depending on which environment you want to update.

## Getting started

Please go through the tutorials to learn how to get started. Tutorial 1 can provide you with starter code but **it is highly recommended to at least go through tutorials 2-5, as well**.

## Problems

If you run into an issue with MotifCompendium, please file a GitHub issue.
