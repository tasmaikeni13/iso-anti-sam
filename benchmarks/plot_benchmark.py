#!/usr/bin/env python3
"""
Generate diagnostic comparison plots for WikiText-103 baseline screening on AMD MI300X.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_dir = '/root/iso-anti-sam/logs/wikitext'
fig_dir = '/root/iso-anti-sam/analysis/figures'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

optimizers = ['adamw', 'sgd', 'sam', 'anti_sam', 'iso_anti_sam']
labels = {
    'adamw': 'AdamW Baseline',
    'sgd': 'SGD Baseline (mom=0.9)',
    'sam': 'Standard SAM (rho=0.05)',
    'anti_sam': 'Raw Anti-SAM (rho=0.05)',
    'iso_anti_sam': 'IsoAntiSAM (Ours, rho=0.05)'
}
colors = {
    'adamw': '#2b5c8f',
    'sgd': '#984ea3',
    'sam': '#7570b3',
    'anti_sam': '#d95f02',
    'iso_anti_sam': '#1b9e77'
}
markers = {
    'adamw': 'o',
    'sgd': 'd',
    'sam': '^',
    'anti_sam': 'v',
    'iso_anti_sam': 's'
}

data = {}
for opt in optimizers:
    path = os.path.join(log_dir, f"{opt}_history.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data[opt] = json.load(f)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for opt in data:
    epochs = range(1, len(data[opt]['val_loss']) + 1)
    axes[0].plot(epochs, data[opt]['val_loss'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)
    axes[1].plot(epochs, data[opt]['val_ppl'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)
    if 'generalization_gap' in data[opt]:
        axes[2].plot(epochs, data[opt]['generalization_gap'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)
    if 'lambda_max' in data[opt]:
        axes[3].plot(epochs, data[opt]['lambda_max'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)

axes[0].set_title('Validation Loss on WikiText-103', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Cross-Entropy Validation Loss', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].set_title('Validation Perplexity (PPL)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Perplexity (PPL)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)

axes[2].set_title('Generalization Divergence Gap $\\Delta(t)$', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=11)
axes[2].set_ylabel('$|L_{\\mathrm{val}} - L_{\\mathrm{train}}|$', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

axes[3].set_title('Hessian Spectral Sharpness ($\\lambda_{\\max}$)', fontsize=11, fontweight='bold')
axes[3].set_xlabel('Epoch', fontsize=11)
axes[3].set_ylabel('Top Curvature $\\lambda_{\\max}(H)$', fontsize=11)
axes[3].grid(True, alpha=0.3)
axes[3].legend(fontsize=8)

plt.tight_layout()
out_log = os.path.join(log_dir, 'benchmark_comparison.png')
out_fig = os.path.join(fig_dir, 'wikitext_benchmark_comparison.png')
plt.savefig(out_log, dpi=200)
plt.savefig(out_fig, dpi=200)
plt.close()
print(f"Comparison plots saved to:\n  {out_log}\n  {out_fig}")
