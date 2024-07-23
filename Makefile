.PHONY: all pytype ruff vulture styler
.ONESHELL: all pytype ruff vulture styler

all: pytype ruff vulture styler

ruff:
	ruff check --fix --no-unsafe-fixes vambn

vulture:
	$(MAKE) ruff
	vulture --min-confidence 60 --sort-by-size vambn
	exit 0

pytype:
	$(MAKE) ruff
	pytype vambn

styler:
	Rscript -e "styler::style_dir('vambn')"

install:
	poetry install
