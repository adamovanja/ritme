#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <feature_table.qza> [--metadata <metadata.tsv>] [--taxonomy <taxonomy.qza>] [--tree <tree.qza>]

Convert QIIME2 artifacts (.qza) to plain files usable by ritme.

Required:
  feature_table.qza    FeatureTable artifact -> .tsv (samples as rows)

Optional:
  --metadata FILE      Metadata .tsv to clean (removes #q2: directive rows)
  --taxonomy FILE      Taxonomy artifact (.qza) -> .tsv
  --tree FILE          Phylogeny artifact (.qza) -> .nwk

Output files are written next to the inputs with .tsv / .nwk extensions.
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage

ft_qza="$1"; shift
metadata=""
tax_qza=""
tree_qza=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata) metadata="$2"; shift 2 ;;
    --taxonomy) tax_qza="$2"; shift 2 ;;
    --tree)     tree_qza="$2"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# --- metadata: strip #q2: directive rows ---
if [[ -n "$metadata" ]]; then
  awk 'NR==1 || !/^#q2:/' "$metadata" > "${metadata}.clean"
  mv "${metadata}.clean" "$metadata"
fi

# --- feature table: .qza -> .tsv (samples as rows) ---
command -v biom >/dev/null 2>&1 || {
  echo "Error: biom CLI not found. Install biom-format to convert feature tables."
  exit 1
}

ft_out="${ft_qza%.qza}.tsv"
ft_tmp_dir="${ft_qza%.qza}_extracted"

unzip -o "$ft_qza" -d "$ft_tmp_dir"
biom convert -i "$ft_tmp_dir"/*/data/feature-table.biom \
  -o "${ft_out}.raw" --to-tsv

python3 - "${ft_out}.raw" "$ft_out" <<'PY'
import sys, pandas as pd
path_in, path_out = sys.argv[1], sys.argv[2]
table = pd.read_csv(path_in, sep="\t", skiprows=1)
table = table.set_index(table.columns[0]).T
table.index.name = "id"
table.to_csv(path_out, sep="\t")
PY
rm -rf "$ft_tmp_dir" "${ft_out}.raw"

# --- taxonomy: .qza -> .tsv ---
if [[ -n "$tax_qza" ]]; then
  tax_out="${tax_qza%.qza}.tsv"
  tax_tmp_dir="${tax_qza%.qza}_extracted"
  unzip -o "$tax_qza" -d "$tax_tmp_dir"
  cp "$tax_tmp_dir"/*/data/taxonomy.tsv "$tax_out"
  rm -rf "$tax_tmp_dir"
fi

# --- tree: .qza -> .nwk ---
if [[ -n "$tree_qza" ]]; then
  tree_out="${tree_qza%.qza}.nwk"
  tree_tmp_dir="${tree_qza%.qza}_extracted"
  unzip -o "$tree_qza" -d "$tree_tmp_dir"
  cp "$tree_tmp_dir"/*/data/tree.nwk "$tree_out"
  rm -rf "$tree_tmp_dir"
fi
