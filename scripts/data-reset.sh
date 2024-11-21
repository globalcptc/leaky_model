#!/bin/bash
BASE_DIR=$(dirname "$0")
cd "${BASE_DIR}/../"
echo $(pwd)

# Define array of paths to clean up
cleanup_paths=(
    "${BASE_DIR}/../data/processed/*"
    "${BASE_DIR}/../.progress.json"
)

# Iterate through array and remove each path
for path in "${cleanup_paths[@]}"; do
    echo "rm -f $path"
    rm -f "$path"
    echo "Cleaned up: $path"
done