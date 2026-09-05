#!/usr/bin/env python3
"""
Generate publication-quality comparative diagnostic figures for IsoAntiSAM vs AdamW
on the 30.5M parameter causal Transformer over 25 epochs.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_dir = '/root/iso-anti-sam/logs/extended_study'
fig_dir = '/root/iso-anti-sam/analysis/figures'
os.makedirs(fig_dir, exist_ok=True)

with open(os.path.join(log_dir, 'adamw_30m_25ep_history.json'), 'r') as f:
    adamw = json.load(f)

with open(os.path.join(log_dir, 'iso_anti_sam_30m_25ep_history.json'), 'r') as f:
    iso = json.load(f)

epochs = list(range(1, len(adamw['val_loss']) + 1))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Training & Validation Loss
axes[0, 0].plot(epochs, adamw['train_loss'], label='AdamW Train', color='#2b5c8f', lw=1.8, linestyle='--')
axes[0, 0].plot(epochs, adamw['val_loss'], label='AdamW Val', color='#2b5c8f', lw=2.4, marker='o', ms=4)
axes[0, 0].plot(epochs, iso['train_loss'], label='IsoAntiSAM Train', color='#1b9e77', lw=1.8, linestyle='--')
axes[0, 0].plot(epochs, iso['val_loss'], label='IsoAntiSAM Val', color='#1b9e77', lw=2.4, marker='s', ms=4)
axes[0, 0].set_title('(a) Cross-Entropy Loss Trajectory (30.5M Params)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Loss (nats)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=9)

# Panel 2: Validation Perplexity
axes[0, 1].plot(epochs, adamw['val_ppl'], label='AdamW Baseline', color='#2b5c8f', lw=2.4, marker='o', ms=4)
axes[0, 1].plot(epochs, iso['val_ppl'], label='IsoAntiSAM (Ours)', color='#1b9e77', lw=2.4, marker='s', ms=4)
axes[0, 1].set_title('(b) Validation Perplexity (PPL)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Perplexity (PPL)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=9)

# Panel 3: Generalization Divergence Gap
axes[1, 0].plot(epochs, adamw['generalization_gap'], label='AdamW Gap |L_val - L_train|', color='#2b5c8f', lw=2.4, marker='o', ms=4)
axes[1, 0].plot(epochs, iso['generalization_gap'], label='IsoAntiSAM Gap |L_val - L_train|', color='#1b9e77', lw=2.4, marker='s', ms=4)
axes[1, 0].set_title('(c) Generalization Divergence Gap $\\Delta(t)$', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('$|L_{\\mathrm{val}} - L_{\\mathrm{train}}|$', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=9)

# Panel 4: Hessian Spectral Curvature lambda_max
axes[1, 1].plot(epochs, adamw['lambda_max'], label='AdamW $\\lambda_{\\max}$', color='#2b5c8f', lw=2.4, marker='o', ms=4)
axes[1, 1].plot(epochs, iso['lambda_max'], label='IsoAntiSAM $\\lambda_{\\max}$', color='#1b9e77', lw=2.4, marker='s', ms=4)
axes[1, 1].set_title('(d) Hessian Spectral Sharpness ($\\lambda_{\\max}$)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Top Curvature $\\lambda_{\\max}(H)$', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
out_png = os.path.join(fig_dir, 'extended_study_30m_comparison.png')
plt.savefig(out_png, dpi=200)
plt.close()
print(f"Saved extended comparison figure to: {out_png}")
