#!/bin/bash
# Usage: ./run_experiment_mlflow.sh [regression|classification]
#   Default: regression

TASK_TYPE="${1:-regression}"

if [[ "$TASK_TYPE" == "regression" ]]; then
  CONFIG="../config/trials_mlflow.json"
  EXP_TAG="trials_mlflow"
elif [[ "$TASK_TYPE" == "classification" ]]; then
  CONFIG="../config/trials_mlflow_class.json"
  EXP_TAG="trials_mlflow_class"
else
  echo "Unknown task type: $TASK_TYPE (expected 'regression' or 'classification')"
  exit 1
fi

# fetch and convert data
scripts/fetch_moving_pictures_data.sh
if [[ ! -f data/movpic_table.tsv ]]; then
  scripts/convert_qiime2_artifacts.sh data/movpic_table.qza \
    --metadata data/movpic_metadata.tsv \
    --taxonomy data/movpic_taxonomy.qza \
    --tree data/movpic_tree.qza
fi

# run experiment
# run split-train-test only if train_val.pkl missing
if [[ ! -f data_splits_mlflow/train_val.pkl ]]; then
  ritme split-train-test \
    data_splits_mlflow data/movpic_metadata.tsv data/movpic_table.tsv \
    --seed 12
fi

# run find-best-model-config only if logs folder empty
if [[ -z "$(find "ritme_example_logs/$EXP_TAG" -maxdepth 1 -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  ritme find-best-model-config \
    "$CONFIG" data_splits_mlflow/train_val.pkl \
    --path-to-tax data/movpic_taxonomy.tsv \
    --path-to-tree-phylo data/movpic_tree.nwk \
    --path-store-model-logs ritme_example_logs
else
  echo "$EXP_TAG directory is not empty; not running find-best-model-config again."
fi
