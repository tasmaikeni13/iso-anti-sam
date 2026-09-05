import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


log_dir = '/root/iso-anti-sam/logs/wikitext'
optimizers = ['adamw', 'sam', 'anti_sam', 'iso_anti_sam']
labels = {
    'adamw': 'AdamW Baseline',
    'sam': 'Standard SAM (rho=0.05)',
    'anti_sam': 'Raw Anti-SAM (rho=0.05)',
    'iso_anti_sam': 'IsoAntiSAM (Ours, rho=0.05)'
}
colors = {
    'adamw': '#2b5c8f',
    'sam': '#7570b3',
    'anti_sam': '#d95f02',
    'iso_anti_sam': '#1b9e77'
}
markers = {
    'adamw': 'o',
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

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for opt in data:
    epochs = range(1, len(data[opt]['val_loss']) + 1)
    axes[0].plot(epochs, data[opt]['val_loss'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)
    axes[1].plot(epochs, data[opt]['val_ppl'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)
    if 'lambda_max' in data[opt]:
        axes[2].plot(epochs, data[opt]['lambda_max'], label=labels[opt], color=colors[opt], lw=2.2, marker=markers[opt], ms=5)

axes[0].set_title('Validation Loss on WikiText-103 (FlashAttention / AMD MI300X)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Cross-Entropy Validation Loss', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)

axes[1].set_title('Validation Perplexity on WikiText-103 (1x AMD MI300X)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Perplexity (PPL)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

axes[2].set_title('Hessian Spectral Sharpness (Top Eigenvalue $\\lambda_{\\max}$)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=11)
axes[2].set_ylabel('Spectral Norm $\\lambda_{\\max}(H)$', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=9)

plt.tight_layout()
out_path = os.path.join(log_dir, 'benchmark_comparison.png')
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Comparison plot saved to: {out_path}")
