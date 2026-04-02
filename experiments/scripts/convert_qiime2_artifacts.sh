#!/usr/bin/env bash
set -euo pipefail

# remove optional QIIME2 directive rows (e.g. #q2:types) from metadata
if [[ -f data/movpic_metadata.tsv ]]; then
  awk 'NR==1 || !/^#q2:/' data/movpic_metadata.tsv > data/movpic_metadata_clean.tsv
  mv data/movpic_metadata_clean.tsv data/movpic_metadata.tsv
fi

command -v biom >/dev/null 2>&1 || {
  echo "biom CLI not found. Install biom-format to run this script."
  exit 1
}

# --- feature table: .qza -> .tsv (samples as rows) ---
unzip -o data/movpic_table.qza -d data/movpic_table_extracted
biom convert -i data/movpic_table_extracted/*/data/feature-table.biom \
  -o data/movpic_table_raw.tsv --to-tsv
# remove biom header comment line, then transpose so rows=samples, cols=features
python3 - <<'PY'
import pandas as pd

path_in = "data/movpic_table_raw.tsv"
path_out = "data/movpic_table.tsv"
table = pd.read_csv(path_in, sep="\t", skiprows=1)
table = table.set_index(table.columns[0]).T
table.index.name = "id"
table.to_csv(path_out, sep="\t")
PY
rm -rf data/movpic_table_extracted data/movpic_table_raw.tsv

# --- taxonomy: .qza -> .tsv ---
unzip -o data/movpic_taxonomy.qza -d data/movpic_taxonomy_extracted
cp data/movpic_taxonomy_extracted/*/data/taxonomy.tsv data/movpic_taxonomy.tsv
rm -rf data/movpic_taxonomy_extracted

# --- rooted phylogeny: .qza -> .nwk ---
unzip -o data/movpic_tree.qza -d data/movpic_tree_extracted
cp data/movpic_tree_extracted/*/data/tree.nwk data/movpic_tree.nwk
rm -rf data/movpic_tree_extracted
