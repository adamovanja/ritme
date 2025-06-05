.PHONY: all lint test test-cov install create-env install-req dev clean distclean

PYTHON ?= python

all: ;

lint:
	ruff check

test: all
	py.test

test-cov: all
	py.test --cov=ritme --cov-report xml:coverage.xml --cov-fail-under=80

# install c-lasso & coral_pytorch
install-pip:
	$(PYTHON) -m pip install \
	https://files.pythonhosted.org/packages/8c/12/b2f7511fbc3563e318754f33c30b2ee5cf095d97896094235202521febc1/c_lasso-1.0.11-py3-none-any.whl \
	https://files.pythonhosted.org/packages/37/16/abebcfe69a8d7b526770ee23832fd6fed7a12afd469611c459f6dd500f81/coral_pytorch-1.4.0-py2.py3-none-any.whl

install: install-pip
	$(PYTHON) -m pip install . --no-deps -vv

create-env: install-req
	mamba create -y -n ritme -c qiime2 -c conda-forge -c bioconda -c pytorch -c anaconda -c defaults $(shell python get_requirements.py ci/recipe/meta.yaml conda)

install-req:
	@command -v mamba >/dev/null 2>&1 || { echo "Installing mamba..."; conda install -y mamba -n base -c conda-forge; }
	$(PYTHON) -m pip install PyYAML jinja2

dev: install-pip
	$(CONDA_PREFIX)/bin/pip install pre-commit parameterized ruff black pytest pytest-cov flake8 versioneer mypy types-PyYAML ipykernel
	$(CONDA_PREFIX)/bin/pip install -e .
	pre-commit install

clean: distclean

distclean: ;
