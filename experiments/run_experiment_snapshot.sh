#!/bin/bash
# Snapshot / temporal-mode smoke test for ritme.
#
# Generates a small synthetic dataset (10 hosts x 5 time points x 12
# features), runs ``ritme split-train-test`` in temporal mode
# (``--time-col time --host-col host_id --n-prev 1``), then trains
# linreg + xgb via ``ritme find-best-model-config``. Designed for a
# ~5-7 min wall-clock smoke run.
#
# Re-runs are no-ops if the data, splits, and logs are all already
# present. To force a clean re-run, delete:
#   experiments/data/{md,ft}_dummy_snapshot.tsv
#   experiments/data_splits_snapshot/
#   experiments/ritme_example_logs/trials_snapshot/

set -e

# Generate synthetic snapshot dataset if either file is missing.
if [[ ! -f data/md_dummy_snapshot.tsv || ! -f data/ft_dummy_snapshot.tsv ]]; then
  python data/generate_dummy_snapshot_data.py
fi

# Snapshot-aware split: --time-col / --host-col / --n-prev trigger the
# temporal-snapshot path in split_train_test, producing ``__t-1``-suffixed
# columns on top of the unsuffixed (t0) ones.
if [[ ! -f data_splits_snapshot/train_val.pkl ]]; then
  ritme split-train-test \
    data_splits_snapshot data/md_dummy_snapshot.tsv data/ft_dummy_snapshot.tsv \
    --seed 12 \
    --group-by-column host_id \
    --time-col time \
    --host-col host_id \
    --n-prev 1 \
    --missing-mode exclude
fi

# Run find-best-model-config only if the experiment dir is empty.
if [[ -z "$(find ritme_example_logs/trials_snapshot -maxdepth 1 -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  ritme find-best-model-config \
    ../config/trials_snapshot.json data_splits_snapshot/train_val.pkl \
    --path-store-model-logs ritme_example_logs
else
  echo "trials_snapshot directory is not empty; not running find-best-model-config again."
fi
