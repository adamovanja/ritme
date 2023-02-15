# q2-time
Longitudinal modeling approaches accounting for high-dimensional, sparse and compositional nature of microbial time-series.

## Setup
<!-- TODO: replace plugin name "time" with something better-->

To install required dependencies for this package run:
```shell
conda create -y -n time \
   -c qiime2 -c conda-forge -c bioconda -c defaults \
  qiime2 q2cli numpy pandas scipy scikit-learn scikit-bio

conda activate time

```

For developers run:

```shell
conda create -y -n time \
   -c qiime2 -c conda-forge -c bioconda -c anaconda -c defaults \
  qiime2 q2cli numpy pandas scipy scikit-learn scikit-bio \
  versioneer pre-commit ruff black pytest flake8

conda activate time

```
