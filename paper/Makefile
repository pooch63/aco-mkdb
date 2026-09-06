# Paper build: placeholders in main.tex → generated fragments → build.tex → PDF
#
#   make          full rebuild (emit + assemble + pdf)
#   make emit     regenerate stale LaTeX fragments from results/ (parallel)
#   make assemble substitute placeholders into build.tex
#   make pdf      compile build.tex (runs assemble if needed)
#   make clean    remove generated artifacts
#
# Emit is incremental (skip up-to-date fragments) and parallel. A fragment is
# stale when results JSON, build.json, or its emit Python sources are newer.
# Override with:
#   make emit FORCE=1          # rebuild all fragments
#   make emit JOBS=4           # cap parallel workers

PYTHON ?= python3
FORCE  ?=
JOBS   ?=
BUILD_FLAGS := $(if $(KEEP_AUX),--keep-aux,) $(if $(FORCE),--force,) $(if $(JOBS),-j$(JOBS),)
BUILD  := $(PYTHON) build.py $(BUILD_FLAGS)

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
