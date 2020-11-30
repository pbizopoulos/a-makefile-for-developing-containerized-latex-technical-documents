[![arXiv](http://img.shields.io/badge/cs.SE-arXiv%3A2005.12660-B31B1B.svg)](https://arxiv.org/abs/2005.12660)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Reconciler%3A%20A%20Workflow%20for%20Certifying%20Computational%20Research%20Reproducibility.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/documenting-results-generation-using-latex-and-docker-template)
[![test-local-reproducibility](https://github.com/pbizopoulos/documenting-results-generation-using-latex-and-docker/workflows/test-local-reproducibility/badge.svg)](https://github.com/pbizopoulos/documenting-results-generation-using-latex-and-docker/actions?query=workflow%3Atest-local-reproducibility)

# Documenting Results Generation using LaTeX and Docker
This repository contains the code that generates **Documenting Results Generation using LaTeX and Docker**.

## Requirements
- [POSIX-oriented operating system](https://en.wikipedia.org/wiki/POSIX#POSIX-oriented_operating_systems)
- [docker](https://docs.docker.com/get-docker/)
- [make](https://www.gnu.org/software/make/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/documenting-results-generation-using-latex-and-docker`
2. `cd documenting-results-generation-using-latex-and-docker`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove tmp directory.
