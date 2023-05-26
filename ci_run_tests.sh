#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the test commands.
make dev
make test
