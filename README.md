# q2-time
Longitudinal modeling approaches accounting for high-dimensional, sparse and compositional nature of microbial time-series.

## Setup
<!-- TODO: replace plugin name "time" with something better-->

To install required dependencies for this package run:
<!-- TODO: verify that conda-build dependencies in ci/recipe/meta.yaml match-->
```shell
mamba create -y -n time \
   -c qiime2 -c conda-forge -c bioconda -c defaults \
  qiime2 q2cli q2-feature-table numpy pandas scipy scikit-learn scikit-bio
conda activate time
make dev

```

For developers additionally install in the above environment:

```shell
mamba install -c conda-forge \
  versioneer pre-commit ruff black pytest flake8 parameterized
```
