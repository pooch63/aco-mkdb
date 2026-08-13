#!/bin/bash

# Ensure output directories exist before writing to them
mkdir -p k2t5i k5t6i

# Define the categories to process inside data/
categories=("konect" "amazon" "wikipedia")
run_k5t6 = "false"

for cat in "${categories[@]}"; do
  for dir in data/"$cat"/*/; do
    # Skip if no directories are found
    [ -d "$dir" ] || continue

    # Get the folder name (e.g., 'boxes' from 'data/amazon/boxes/')
    name=$(basename "$dir")

    echo "Processing $cat/$name..."

    # Run Configuration 1 (k=2, theta=5)
    julia -t 8 load.jl "$cat/$name" --prefer-smaller-side=false --inject --u=5 --v=5 --k=2 --theta=5 --benchmark=aco,heuristic > "k2t5i/${name}.txt"
  done
done

if [[ "$run_k5t6" == "true" ]]
    for cat in "${categories[@]}"; do
    for dir in data/"$cat"/*/; do
        # Skip if no directories are found
        [ -d "$dir" ] || continue

        # Get the folder name (e.g., 'boxes' from 'data/amazon/boxes/')
        name=$(basename "$dir")

        echo "Processing $cat/$name..."

        # Run Configuration 2 (k=5, theta=6)
        julia -t 8 load.jl "$cat/$name" --prefer-smaller-side=false --inject --u=6 --v=6 --k=5 --theta=6 --benchmark=aco,heuristic > "k5t6i/${name}.txt"
    done
    done
fi