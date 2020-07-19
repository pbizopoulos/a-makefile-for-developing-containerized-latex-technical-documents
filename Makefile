.POSIX:

ARXIV_ID = 
CACHE_DIR = cache
DOCKER_WORKDIR = /usr/src/app
NAME_CURRENT_DIR = $(notdir $(shell pwd))
RELEASE_NAME = v1
RESULTS_DIR = results
RESULTS_TRAIN_DIR = results-train
ROOT_EVAL = eval.py
ROOT_TEX_NO_EXT = ms
ROOT_TRAIN = train.py
VENV_DIR = venv

$(ROOT_TEX_NO_EXT).pdf: $(ROOT_TEX_NO_EXT).tex $(ROOT_TEX_NO_EXT).bib $(RESULTS_DIR)
	latexmk -gg -pdf -quiet $<

$(RESULTS_DIR): $(RESULTS_TRAIN_DIR) $(ROOT_EVAL)
	rm -rf $@/
	. $(VENV_DIR)/bin/activate; python3 $(ROOT_EVAL) $(ARGS) --cache-dir $(CACHE_DIR) --results-train-dir $(RESULTS_TRAIN_DIR) --results-dir $(RESULTS_DIR)

$(RESULTS_TRAIN_DIR): $(VENV_DIR) $(ROOT_TRAIN)
	rm -rf $@/
	. $(VENV_DIR)/bin/activate; python3 $(ROOT_TRAIN) $(ARGS) --cache-dir $(CACHE_DIR) --results-train-dir $(RESULTS_TRAIN_DIR)

$(VENV_DIR): requirements.txt
	rm -rf $@/
	python3 -m $@ $@/
	. $@/bin/activate; pip install -U pip wheel; pip install -Ur $<

clean:
	rm -rf __pycache__/ $(CACHE_DIR)/ $(RESULTS_DIR)/ $(RESULTS_TRAIN_DIR) $(VENV_DIR)/ arxiv.tar $(ROOT_TEX_NO_EXT).bbl
	latexmk -C $(ROOT_TEX_NO_EXT)

docker:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/$(CACHE_DIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		$(NAME_CURRENT_DIR) \
		/bin/bash -c "python3 $(ROOT_TRAIN) $(ARGS) --cache-dir $(CACHE_DIR) --results-train-dir $(RESULTS_TRAIN_DIR) && python3 $(ROOT_EVAL) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR) --results-train-dir $(RESULTS_TRAIN_DIR)"
	make docker-pdf

docker-gpu:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/$(CACHE_DIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		--gpus all $(NAME_CURRENT_DIR) \
		/bin/bash -c "python3 $(ROOT_TRAIN) $(ARGS) --cache-dir $(CACHE_DIR) --results-train-dir $(RESULTS_TRAIN_DIR) && python3 $(ROOT_EVAL) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR) --results-train-dir $(RESULTS_TRAIN_DIR)"
	make docker-pdf

docker-pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -gg -pdf -quiet -cd /home/latex/$(ROOT_TEX_NO_EXT).tex

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	docker build -t $(NAME_CURRENT_DIR)-arxiv .
	make docker-pdf
	rm $(ARXIV_ID)

arxiv.tar:
	tar -cvf arxiv.tar $(ROOT_TEX_NO_EXT).{tex,bib,bbl} $(RESULTS_TRAIN_DIR)/*.csv $(RESULTS_DIR)/*.{pdf,tex}

upload-results:
	hub release create -m 'Results release' $(RELEASE_NAME)
	for f in $(shell ls $(RESULTS_TRAIN_DIR)/*); do hub release edit -m 'Results' -a $$f $(RELEASE_NAME); done

download-results:
	mkdir -p $(RESULTS_TRAIN_DIR) ; cd $(RESULTS_TRAIN_DIR) ; hub release download $(RELEASE_NAME) ; cd ..

delete-results:
	hub release delete $(RELEASE_NAME)
	git push origin :$(RELEASE_NAME)
