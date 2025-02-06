#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1

for folder in "$directory"/*; do
    if [ -d "$folder" ]; then
        echo "current folder: $folder"
        python compare_real_sim.py -rd "$folder"
        break
    fi
done