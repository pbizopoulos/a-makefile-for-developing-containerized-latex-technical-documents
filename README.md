[![arXiv](http://img.shields.io/badge/cs.SE-arXiv%3A2005.12660-B31B1B.svg)](https://arxiv.org/abs/2005.12660)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Reconciler%3A%20A%20Workflow%20for%20Certifying%20Computational%20Research%20Reproducibility.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/latex-technical-documents-with-docker-and-make-template)
[![test-draft-version-document-reproducibility](https://github.com/pbizopoulos/documenting-results-generation-using-latex-and-docker/workflows/test-draft-version-document-reproducibility/badge.svg)](https://github.com/pbizopoulos/latex-technical-documents-with-docker-and-make/actions?query=workflow%3Atest-draft-version-document-reproducibility)

# LaTeX Technical Documents with Docker and Make
This repository contains the code that generates **LaTeX Technical Documents with Docker and Make**.

## Requirements
- [POSIX-oriented operating system](https://en.wikipedia.org/wiki/POSIX#POSIX-oriented_operating_systems)
- [Docker](https://docs.docker.com/get-docker/)
- [Make](https://www.gnu.org/software/make/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/latex-technical-documents-with-docker-and-make`
2. `cd latex-technical-documents-with-docker-and-make/`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate the draft (fast) version document.
    * `make ARG=--full`  # Generate the full (slow) version document.
    * `make clean`       # Remove the tmp/ directory.
