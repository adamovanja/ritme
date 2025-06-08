#!/bin/bash

# fetch data
if [[ ! -f data/movpic_metadata.tsv ]]; then
    curl -L -o data/movpic_metadata.tsv \
    "https://data.qiime2.org/2024.10/tutorials/moving-pictures/sample_metadata.tsv"
fi

if [[ ! -f data/movpic_table.qza ]]; then
  curl -L -o data/movpic_table.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/table.qza"
fi

if [[ ! -f data/movpic_taxonomy.qza ]]; then
  curl -L -o data/movpic_taxonomy.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/taxonomy.qza"
fi

if [[ ! -f data/movpic_tree.qza ]]; then
  curl -L -o data/movpic_tree.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/rooted-tree.qza"
fi


# run experiment
# run split-train-test only if train_val.pkl missing
if [[ ! -f data_splits_mlflow/train_val.pkl ]]; then
  ritme split-train-test \
    data_splits_mlflow data/movpic_metadata.tsv data/movpic_table.qza \
    --seed 12
fi

# run find-best-model-config only if logs folder empty
if [[ -z "$(find trials_mlflow -maxdepth 1 -mindepth 1 -print -quit)" ]]; then
  ritme find-best-model-config \
    ../config/trials_mlflow.json data_splits_mlflow/train_val.pkl \
    --path-to-tax data/movpic_taxonomy.qza \
    --path-to-tree-phylo data/movpic_tree.qza \
    --path-store-model-logs trials_mlflow
fi
