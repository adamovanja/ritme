#!/bin/bash

# fetch data
if [[ ! -f data/movpic_metadata.tsv ]]; then
    curl -L -o data/movpic_metadata.tsv \
    "https://data.qiime2.org/2024.10/tutorials/moving-pictures/sample_metadata.tsv"
fi

if [[ ! -f data/movpic_table.tsv ]]; then
  curl -L -o data/movpic_table.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/table.qza"
  unzip -o data/movpic_table.qza -d data/movpic_table_extracted
  cp data/movpic_table_extracted/*/data/feature-table.biom data/
  biom convert -i data/feature-table.biom -o data/movpic_table.tsv --to-tsv
  tail -n +2 data/movpic_table.tsv > data/movpic_table_clean.tsv
  mv data/movpic_table_clean.tsv data/movpic_table.tsv
fi

if [[ ! -f data/movpic_taxonomy.tsv ]]; then
  curl -L -o data/movpic_taxonomy.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/taxonomy.qza"
  unzip -o data/movpic_taxonomy.qza -d data/movpic_taxonomy_extracted
  cp data/movpic_taxonomy_extracted/*/data/taxonomy.tsv data/movpic_taxonomy.tsv
fi

if [[ ! -f data/movpic_tree.nwk ]]; then
  curl -L -o data/movpic_tree.qza \
    "https://docs.qiime2.org/2024.10/data/tutorials/moving-pictures/rooted-tree.qza"
  unzip -o data/movpic_tree.qza -d data/movpic_tree_extracted
  cp data/movpic_tree_extracted/*/data/tree.nwk data/movpic_tree.nwk
fi


# run experiment
# run split-train-test only if train_val.pkl missing
if [[ ! -f data_splits_mlflow/train_val.pkl ]]; then
  ritme split-train-test \
    data_splits_mlflow data/movpic_metadata.tsv data/movpic_table.tsv \
    --seed 12
fi

# run find-best-model-config only if logs folder empty
if [[ -z "$(find ritme_example_logs/trials_mlflow -maxdepth 1 -mindepth 1 -print -quit)" ]]; then
  ritme find-best-model-config \
    ../config/trials_mlflow.json data_splits_mlflow/train_val.pkl \
    --path-to-tax data/movpic_taxonomy.tsv \
    --path-to-tree-phylo data/movpic_tree.nwk \
    --path-store-model-logs ritme_example_logs
else
  echo "trials_mlflow directory is not empty; not running find-best-model-config again."
fi
