#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

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
