#!/bin/bash

# Define array of paths to clean up
cleanup_paths=(
    "training_data/markdown/images/*"
    "training_data/markdown/*.md"
    "training_data/tmp/*"
    "training_data/markdown/.processing_progress.json"
)

# Iterate through array and remove each path
for path in "${cleanup_paths[@]}"; do
    rm -f "$path"
    echo "Cleaned up: $path"
done