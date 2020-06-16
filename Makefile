all:
	make clean
	make results
	make ms.pdf

clean:
	rm -rf __pycache__/ cache/ venv/ upload_to_arxiv.tar
	make clean-results

clean-results:
	latexmk -C ms.tex
	rm -rf results/ ms.bbl

results: $(shell find . -maxdepth 1 -name '*.py')
	make venv
	. venv/bin/activate; ./main.py $(ARGS)
	touch -c results

venv: requirements.txt
	python3 -m venv venv
	. venv/bin/activate; pip install -U pip wheel; pip install -Ur requirements.txt
	touch -c venv

ms.pdf: results ms.tex ms.bib
	latexmk -gg -pdf -quiet ms.tex

view:
	xdg-open ms.pdf

docker-ms.pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -gg -pdf -quiet -cd /home/latex/ms.tex

GPU != if [[ "$(ARGS)" == *"--gpu"* ]]; then echo "--gpus=all"; fi
PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app
docker:
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=$(WORKDIR)/cache \
		-e TORCH_HOME=$(WORKDIR)/cache \
		-v $(PWD):$(WORKDIR) \
		$(GPU) $(PROJECT) \
		./main.py $(ARGS)
	make docker-ms.pdf

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	docker build -t $(PROJECT)-arxiv .
	make docker-ms.pdf
	rm $(ARXIV_ID)

arxiv-tar:
	tar -cvf upload_to_arxiv.tar ms.tex ms.bib ms.bbl results/*.{pdf,tex}

upload-results:
	hub release create -m 'Results release' results-release
	for f in $(shell ls results/*); do hub release edit -m 'Results release' -a $$f results-release; done

download-results:
	mkdir -p results ; cd results ; hub release download results-release ; cd ..

delete-results:
	hub release delete results-release
	git push origin :results-release
