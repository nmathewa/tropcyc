# Project website: content/*.md -> _site/*.html with pandoc and the academic-site-template (theme/).
#   make                 build _site/
#   make THEME=~/academic-site-template    use a local template checkout instead of the submodule
#   make serve           build, then serve on http://0.0.0.0:8010
PANDOC ?= pandoc
THEME  ?= theme
PORT   ?= 8010
PAGES  := $(patsubst content/%.md,_site/%.html,$(wildcard content/*.md))
DEPS   := $(THEME)/templates/page.html content/site.yaml $(THEME)/filters/crossref.lua content/references.bib

all: $(PAGES) _site/assets _site/img

_site/%.html: content/%.md $(DEPS)
	@mkdir -p _site
	$(PANDOC) $< -o $@ --standalone --template $(THEME)/templates/page.html \
	  --metadata-file content/site.yaml --lua-filter $(THEME)/filters/crossref.lua \
	  --citeproc --bibliography content/references.bib \
	  --math-method=mathml --section-divs --toc --toc-depth 2

_site/assets: $(shell find $(THEME)/assets -type f 2>/dev/null)
	@mkdir -p _site && rm -rf _site/assets && cp -r $(THEME)/assets _site/assets

_site/img: $(shell find content/img -type f 2>/dev/null)
	@mkdir -p _site && rm -rf _site/img && cp -r content/img _site/img

serve: all
	python3 -m http.server $(PORT) -d _site --bind 0.0.0.0

clean:
	rm -rf _site

.PHONY: all serve clean
