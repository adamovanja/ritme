#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# create environment
make create-env

# Activate the Conda environment.
conda activate time

# Run the test commands.
make dev
qiime dev refresh-cache
make test
