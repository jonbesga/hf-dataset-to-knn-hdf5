#/bin/bash

set -e

train_sizes=()

for i in $(seq 1 20); do
  num=$((i * 1000000))
  train_sizes+=("$num")
done

for size in "${train_sizes[@]}"
do
    echo "Running with train size: $size"
    python main.py "Cohere/wikipedia-22-12-en-embeddings" --train-size="$size" --test-size=10000 --k=100
done
