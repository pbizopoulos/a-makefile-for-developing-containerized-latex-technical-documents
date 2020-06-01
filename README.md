# Reconciler: A Workflow for Certifying Computational Research Reproducibility
This repository contains the code that generates the results of the paper **Reconciler: A Workflow for Certifying Computational Research Reproducibility**.

![](https://github.com/pbizopoulos/reconciler-a-workflow-for-certifying-computational-research-reproducibility/workflows/reproducibility/badge.svg)
![](https://github.com/pbizopoulos/reconciler-a-workflow-for-certifying-computational-research-reproducibility/workflows/arxiv-reproducibility/badge.svg)

ArXiv link: <https://arxiv.org/abs/2005.12660>

# Instructions
The syntax of the `make` command is as follows:

`make [docker] [ARGS="[--full] [--gpu]"]`

where `[...]` denotes an optional argument.

For example you can choose one of the following:
- `make`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes ~1 minute and populates the figures and table.
- `make ARGS="--full --gpu"`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes ~1 minute on an NVIDIA Titan X.
- `make docker`
	- Requires local installation of docker.
	- Takes ~1 minute.
- `make docker ARGS="--full --gpu"`
	- Requires local installation of nvidia-container-toolkit.
	- Takes ~1 minute on an NVIDIA Titan X.
- `make clean`
	- Restores the repo in its initial state by removing all figures, tables and downloaded datasets.

# Citation:
If you use this repository cite the following:
```
@article{bizopoulos2020reconciler,
	title={Reconciler: A Workflow for Certifying Computational Research Reproducibility},
	author={Bizopoulos, Paschalis and Bizopoulos, Dimitris},
	journal={arXiv preprint arXiv:2005.12660},
	year={2020}
}
```
