[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=Reconciler%3A%20A%20Workflow%20for%20Certifying%20Computational%20Research%20Reproducibility.%20arXiv%202020)
[![arXiv](http://img.shields.io/badge/cs.SE-arXiv%3A2005.12660-B31B1B.svg)](https://arxiv.org/abs/2005.12660)

# Reproducible Builds for Computational Research Papers
This repository contains the code that generates the results of the paper **Reproducible Builds for Computational Research Papers**.

## Requirements
- UNIX utilities (cmp, cp, echo, rm, touch)
- docker
- make
- nvidia-container-toolkit [required only when using CUDA]

## Instructions [more info on this template](https://github.com/pbizopoulos/cookiecutter-reproducible-builds-for-computational-research-papers)
1. `git clone https://github.com/pbizopoulos/reproducible-builds-for-computational-research-papers`
2. `cd reproducible-builds-for-computational-research-papers`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove cache, results directories and tex auxiliary files.
