# PORCELAN: Integrating representation learning, permutation, and optimization to detect lineage-related gene expression patterns
This repository contains code for the paper "Integrating representation learning, permutation, and optimization to detect lineage-related gene expression patterns". We developed **P**ermutation, **O**ptimization, and **R**epresentation learning based single **C**ell gene **E**xpression and **L**ineage **AN**alysis (PORCELAN) to identify lineage-informative genes or subtrees where lineage and expression are tightly coupled:

![porcelan_overview](https://github.com/uhlerlab/porcelan/assets/34376746/bc4ba6c2-00c9-4d34-b4e6-332ba64cf829)

## Repository overview

* `data` contains jupyter notebooks for downloading, simulating, and pre-processing the datasets used in the paper's results. For convenience, we also provide pre-processed data files in `data/preprocessed`. See [`data/README.md`](data/README.md) for further details.
* `figure_notebooks` contains jupyter notebooks to reproduce the paper's main and supplemental figures. Most results can be reproduced in a few seconds or minutes but we also provide the data files for the results displayed in the figures in `results` for convenience. See [`figure_notebooks/README.md`](figure_notebooks/README.md) for further details.

## Dependencies:
**Python:**

This repository was developed using Python 3.8. You can use [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to create a virtual environment for a specific Python version. Additional required packages are listed in [`requirements.txt`](requirements.txt) and can be installed using the following command:
```
pip install -r requirements.txt
```
Installing dependencies can take a few minutes or up to an hour dependending on how many packages need to be downloaded rather than reusing cached versions.

**R:**

We only use R to simulate lineage-resolved gene expression data with TedSim ([installation instructions](https://github.com/Galaxeee/TedSim/tree/main)).

**Operating system and hardware:**

We tested this code on a machine running Ubuntu 20.04.3 and equipped with an NVIDIA RTX A5000 GPU.

## Citation
```
TODO
```
