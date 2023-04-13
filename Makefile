.PHONY: all lint test test-cov install create-env install-req dev prep-dev-container clean distclean

PYTHON ?= python

all: ;

lint:
	q2lint
	flake8

test: all
	py.test

test-cov: all
	py.test --cov=q2_time

install: all
	$(PYTHON) setup.py install

create-env: install-req
	mamba create -y -n time -c qiime2 -c conda-forge -c bioconda -c defaults $(shell python get_requirements.py ci/recipe/meta.yaml)

install-req:
	@command -v mamba >/dev/null 2>&1 || { echo "Installing mamba..."; conda install -y mamba -n base -c conda-forge; }
	pip install PyYAML jinja2

dev: all
	pip install pre-commit parameterized ruff black pytest flake8 versioneer mypy types-PyYAML
	pip install -e .
	pre-commit install

prep-dev-container: all
	conda install mamba -qy -n base -c conda-forge
	mamba install -p /opt/conda/envs/qiime2-$(QIIME_VERSION) -qy -c conda-forge -c bioconda -c defaults --file requirements.txt flake8 coverage wget pytest-xdist autopep8
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -q https://github.com/qiime2/q2lint/archive/master.zip
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -e .

clean: distclean

distclean: ;
