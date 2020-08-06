.POSIX:

ARXIV_ID = 
CACHE_DIR = cache
DOCKER_WORKDIR = /usr/src/app
NAME_CURRENT_DIR = $(notdir $(shell pwd))
PYTHON = python3
RELEASE_NAME = v1
RESULTS_DIR = results
ROOT_CODE = main.py
ROOT_TEX_NO_EXT = ms
SRC_CODE = $(shell find . -maxdepth 1 -name '*.py')
VENV_DIR = venv

$(ROOT_TEX_NO_EXT).pdf: $(ROOT_TEX_NO_EXT).tex $(ROOT_TEX_NO_EXT).bib $(RESULTS_DIR)
	make docker-pdf

$(RESULTS_DIR): $(VENV_DIR) $(SRC_CODE)
	rm -rf $@/
	. $</bin/activate; $(PYTHON) $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)

$(VENV_DIR): requirements.txt
	rm -rf $@/
	$(PYTHON) -m venv $@/
	. $@/bin/activate; $(PYTHON) -m pip install -U pip wheel; $(PYTHON) -m pip install -Ur $<

clean:
	rm -rf __pycache__/ $(CACHE_DIR)/ $(RESULTS_DIR)/ $(VENV_DIR)/ arxiv.tar $(ROOT_TEX_NO_EXT).bbl
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -C -cd /home/latex/$(ROOT_TEX_NO_EXT).tex

docker:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		$(NAME_CURRENT_DIR) \
		$(PYTHON) $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)
	make docker-pdf

docker-gpu:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		--gpus all $(NAME_CURRENT_DIR) \
		$(PYTHON) $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)
	make docker-pdf

docker-pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /home/latex/$(ROOT_TEX_NO_EXT).tex
	sha256sum $(ROOT_TEX_NO_EXT).pdf

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	make docker-pdf
	rm $(ARXIV_ID)

arxiv.tar:
	tar -cvf arxiv.tar $(ROOT_TEX_NO_EXT).{tex,bib,bbl} $(RESULTS_DIR)/*.{pdf,tex}

upload-results:
	hub release create -m 'Results release' $(RELEASE_NAME)
	for f in $(shell ls $(RESULTS_DIR)/*); do hub release edit -m 'Results' -a $$f $(RELEASE_NAME); done

download-results:
	mkdir -p $(RESULTS_DIR); cd $(RESULTS_DIR); hub release download $(RELEASE_NAME); cd ..

delete-results:
	hub release delete $(RELEASE_NAME)
	git push origin :$(RELEASE_NAME)

.gitignore: Makefile
	echo "*.aux" > .gitignore
	echo "*.bbl" >> .gitignore
	echo "*.blg" >> .gitignore
	echo "*.fdb_latexmk" >> .gitignore
	echo "*.fls" >> .gitignore
	echo "*.log" >> .gitignore
	echo "*.out" >> .gitignore
	echo "*.pdf" >> .gitignore
	echo __pycache__/ >> .gitignore
	echo arxiv.tar >> .gitignore
	echo $(CACHE_DIR)/ >> .gitignore
	echo $(RESULTS_DIR)/ >> .gitignore
	echo $(VENV_DIR)/ >> .gitignore

.dockerignore: Makefile
	cat .gitignore > .dockerignore
	echo .git/ >> .dockerignore

.github/workflows/reproducible-build.yml: Makefile
	mkdir -p .github/workflows
	curl -L -o .github/workflows/reproducible-build.yml https://raw.githubusercontent.com/pbizopoulos/reconciler-a-template-for-reproducible-computational-research-papers/master/.github/workflows/reproducible-build.yml

init: .gitignore .dockerignore .github/workflows/reproducible-build.yml
