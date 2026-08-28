# Paper build: placeholders in main.tex → generated fragments → build.tex → PDF
#
#   make          full rebuild (emit + assemble + pdf)
#   make emit     regenerate LaTeX fragments from results/
#   make assemble substitute placeholders into build.tex
#   make pdf      compile build.tex (runs assemble if needed)
#   make clean    remove generated artifacts

PYTHON ?= python3
BUILD  := $(PYTHON) build.py $(if $(KEEP_AUX),--keep-aux,)

.PHONY: all emit assemble pdf clean

all:
	$(BUILD) all

emit:
	$(BUILD) emit

assemble:
	$(BUILD) assemble

pdf: build.tex
	$(BUILD) pdf

build.tex: main.tex build.json $(wildcard generated/*.tex)
	$(BUILD) assemble

clean:
	latexmk -C build.tex 2>/dev/null || true
	rm -f build.tex build.pdf build.aux build.fdb_latexmk build.fls build.log \
	      build.out build.synctex.gz
	rm -rf generated/
