#/bin/bash

set -e

train_sizes=(1000000 2000000 5000000 10000000 20000000)

for size in "${train_sizes[@]}"
do
    echo "Running with train size: $size"
    python main.py "Cohere/wikipedia-22-12-en-embeddings" --train-size="$size" --test-size=10000 --k=100
done