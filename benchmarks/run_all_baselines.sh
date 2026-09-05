#!/bin/bash
set -e
echo "Starting Equal-Budget Baseline Screening on AMD Instinct MI300X..."

python3 benchmarks/train_wikitext.py --optimizer adamw --epochs 10 --steps_per_epoch 200 --val_steps 50
python3 benchmarks/train_wikitext.py --optimizer sgd --epochs 10 --steps_per_epoch 200 --val_steps 50
python3 benchmarks/train_wikitext.py --optimizer sam --epochs 10 --steps_per_epoch 200 --val_steps 50
python3 benchmarks/train_wikitext.py --optimizer anti_sam --epochs 10 --steps_per_epoch 200 --val_steps 50
python3 benchmarks/train_wikitext.py --optimizer iso_anti_sam --epochs 10 --steps_per_epoch 200 --val_steps 50

python3 benchmarks/plot_benchmark.py
echo "All baseline screening runs completed successfully!"
