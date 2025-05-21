.ONESHELL:
SHELL := /bin/bash
EXPERIMENTS := artifacts
CONFIGS_FOLDER := config

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  Install:"
	@echo "  - venv                      : set up the virtual environment for development"
	@echo "  - install                   : installs requirements"

# Installation

venv:
	python3 -m venv .venv

.PHONY: install
install:
	pip install --upgrade pip
	python -m pip install --upgrade torch && \
	python -m pip install --upgrade -r requirements.txt


.PHONY: format
format:
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r src
	black src
	isort src


.PHONY: test
test:
	python -m unittest discover -s test


.PHONY: clear_experiments
clear_experiments:
	rm -r $(EXPERIMENTS)


.PHONY: clean_workspace
clean_workspace:
	mkdir -p .closet
	mv --backup=t *.png .closet/
	mv --backup=t *.pdf .closet/
	mv --backup=t *.csv .closet/

.PHONY: clean_pycache
	find . -name "__pycache__" -type d -exec rm -rf {} +
