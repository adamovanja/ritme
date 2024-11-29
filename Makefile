.PHONY: all lint test test-cov install create-env install-req dev prep-dev-container clean distclean

PYTHON ?= python

all: ;

lint:
	ruff check

test: all
	py.test

# todo: adjust coverage threshold once final
test-cov: all
	py.test --cov=ritme --cov-report xml:coverage.xml --cov-fail-under=30

# todo: update c-lasso to this once lfs quota is increased again:
# git+https://github.com/Leo-Simpson/c-lasso.git@82b106c471c70559b1720a5bb79a61a5224cca7f
install-pip:
	git lfs install
	pip install \
	git+https://github.com/adamovanja/c-lasso.git@2d192ed1e645514b849dc0ad4f4a5505687af88a \
	git+https://github.com/Raschka-research-group/coral-pytorch.git@313482f86f50b58d8beb9fb54652e943b06745ef

install: install-pip
	$(PYTHON) setup.py install

create-env: install-req
	mamba create -y -n ritme -c qiime2 -c conda-forge -c bioconda -c pytorch -c defaults $(shell python get_requirements.py ci/recipe/meta.yaml conda)

install-req:
	@command -v mamba >/dev/null 2>&1 || { echo "Installing mamba..."; conda install -y mamba -n base -c conda-forge; }
	pip install PyYAML jinja2

dev: install-pip
	$(CONDA_PREFIX)/bin/pip install pre-commit parameterized ruff black pytest pytest-cov flake8 versioneer mypy types-PyYAML ipykernel
	$(CONDA_PREFIX)/bin/pip install -e .
	pre-commit install

prep-dev-container: all
	conda install mamba -qy -n base -c conda-forge
	mamba install -p /opt/conda/envs/qiime2-$(QIIME_VERSION) -qy -c conda-forge -c bioconda -c defaults --file requirements.txt flake8 coverage wget pytest-xdist autopep8
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -q https://github.com/qiime2/q2lint/archive/master.zip
	/opt/conda/envs/qiime2-$(QIIME_VERSION)/bin/pip install -e .

clean: distclean

distclean: ;
