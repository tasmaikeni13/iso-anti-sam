#!/bin/bash
set -e
echo "================================================================================"
echo "    Extended Head-to-Head Benchmark: IsoAntiSAM vs AdamW (30.5M Params, 25 Epochs) "
echo "================================================================================"
mkdir -p logs/extended_study analysis/figures

echo "--- Run 1/2: AdamW Baseline (30.5M Params, 25 Epochs, 5,000 Steps) ---"
python3 benchmarks/train_wikitext.py \
    --optimizer adamw \
    --dim 512 --n_layers 8 --n_heads 8 \
    --epochs 25 --steps_per_epoch 200 --val_steps 50 \
    --lr 5e-4 --batch_size 64 --seq_len 256 \
    --exp_name adamw_30m_25ep \
    --log_dir /root/iso-anti-sam/logs/extended_study

echo "--- Run 2/2: IsoAntiSAM (Ours) (30.5M Params, 25 Epochs, 5,000 Steps) ---"
python3 benchmarks/train_wikitext.py \
    --optimizer iso_anti_sam \
    --dim 512 --n_layers 8 --n_heads 8 \
    --epochs 25 --steps_per_epoch 200 --val_steps 50 \
    --lr 5e-4 --rho 0.05 --batch_size 64 --seq_len 256 \
    --exp_name iso_anti_sam_30m_25ep \
    --log_dir /root/iso-anti-sam/logs/extended_study

echo "--- Generating Comprehensive Comparison Figures ---"
python3 benchmarks/plot_extended_comparison.py

echo "Extended study completed successfully!"
