.PHONY: all lint test test-cov install create-env install-req dev prep-dev-container clean distclean

PYTHON ?= python

all: ;

lint:
	q2lint
	flake8

test: all
	py.test

# todo: adjust coverage threshold once final
test-cov: all
	py.test --cov=q2_ritme --cov-report xml:coverage.xml --cov-fail-under=30

install: all
	$(PYTHON) setup.py install

create-env: install-req
	mamba create -y -n ritme -c qiime2 -c conda-forge -c bioconda -c pytorch -c defaults $(shell python get_requirements.py ci/recipe/meta.yaml conda)

install-req:
	@command -v mamba >/dev/null 2>&1 || { echo "Installing mamba..."; conda install -y mamba -n base -c conda-forge; }
	pip install PyYAML jinja2

dev: all
	$(CONDA_PREFIX)/bin/pip install pre-commit parameterized ruff black pytest pytest-cov flake8 versioneer mypy types-PyYAML ipykernel $(shell python get_requirements.py ci/recipe/meta.yaml pip)
	$(CONDA_PREFIX)/bin/pip install -e .
	pre-commit install

prep-dev-container: all
	conda install mamba -qy -n base -c conda-forge
	mamba install -p /opt/conda/envs/qiime2-$(QIIME_VERSION) -qy -c conda-forge -c bioconda -c defaults --file requirements.txt flake8 coverage wget pytest-xdist autopep8
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -q https://github.com/qiime2/q2lint/archive/master.zip
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -e .

clean: distclean

distclean: ;
