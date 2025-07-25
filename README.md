# MotifCompendium

**Please read this README.**

The MotifCompendium package provides a framework for clustering, annotating, and analyzing motifs efficiently at large scales.

**(7/25/2025)** The main branch of MotifCompendium has been updated! It is highly recommended to upgrade to the current version of the package. Please only work off of the main branch, which will contain the most stable build of MotifCompendium.

README sections:

- [Installation](#installation)
- [Getting started](#getting-started)
- [Loading old versions of MotifCompendium](#working-with-old-versions-of-motifcompendium)
- [Documentation](#documentation)
- [Problems with MotifCompendium](#problems)
- [Developers](#developers)

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

Please go through the [tutorials](tutorials/README.md) to learn how to get started. Tutorial 1 can provide you with starter code but **it is highly recommended to at least go through tutorials 2-5, as well**.

## Working with old versions of MotifCompendium

**(7/25/2025)** **Old MotifCompendium objects may not be compatible with the current version of the package!** If possible, rerun your MotifCompendium code with the current version of the package. If that is not possible, you may need to upgrade your MotifCompendium object in order for it to be compatible with the newest version of the package. To check if your MotifCompendium object needs upgrading, try loading the object. If the load fails, try loading with `safe=False` and saving the object; the saved version will be compatible with the package. If that also fails, try loading your object with the `load_old_compendium()` function. And if all of that still fails, please reach out on Slack for assistance.

## Documentation

A Read The Docs will be released when the package is publically released. Until then, please use the [tutorials](tutorials/README.md) as documentation. Or, please feel free to look through the source code directly.

## Problems

If you run into an issue with MotifCompendium, please file a GitHub issue. And concurrently, please post a message to the #motif-compendium channel on Slack.

## Developers

Developed by Salil Deshpande and Chang Yun.
