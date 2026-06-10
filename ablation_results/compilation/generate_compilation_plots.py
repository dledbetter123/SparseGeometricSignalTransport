"""Generate compilation plots: CurvBias vs RoPE across three scales."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

# Scale 1: Small (d=256, T=512, ~14M params, 20K steps)
small = {
    'GPT+CurvBias': {'ppl': 48.6, 'acc': 35.4, 'ms': 22},
    'GPT+RoPE':     {'ppl': 53.4, 'acc': 33.7, 'ms': 16},
    'GLA+CurvBias': {'ppl': 50.5, 'acc': 33.8, 'ms': 36},
    'GLA+RoPE':     {'ppl': 48.6, 'acc': 34.1, 'ms': 25},
    'GPT+CARoPE':   {'ppl': 57.2, 'acc': 33.0, 'ms': 19},
    'GLA+CARoPE':   {'ppl': 56.9, 'acc': 33.0, 'ms': 31},
    'GPT':          {'ppl': 62.9, 'acc': 31.7, 'ms': 15},
    'GLA':          {'ppl': 61.8, 'acc': 32.2, 'ms': 23},
}

# Scale 2: Large (d=512, T=1024, ~77M params, 20K steps)
large = {
    'GPT+CurvBias': {'ppl': 37.3, 'acc': 37.5, 'ms': 42},
    'GPT+RoPE':     {'ppl': 38.9, 'acc': 36.8, 'ms': 31},
    'GLA+CurvBias': {'ppl': 39.9, 'acc': 36.4, 'ms': 63},  # best PPL
    'GLA+RoPE':     {'ppl': 40.9, 'acc': 36.1, 'ms': 45},
    'GPT+CARoPE':   {'ppl': 40.7, 'acc': 36.4, 'ms': 36},
    'GLA+CARoPE':   {'ppl': 40.4, 'acc': 36.1, 'ms': 54},  # best PPL
    'GPT':          {'ppl': 52.3, 'acc': 33.3, 'ms': 28},
    'GLA':          {'ppl': 46.0, 'acc': 35.0, 'ms': 41},
}

# Scale 3: v6 (d=512, T=3072, ~77M params, 40K steps)
v6 = {
    'GPT+CurvBias': {'ppl': 24.5, 'acc': 41.6, 'ms': 166},
    'GLA+CurvBias': {'ppl': 25.3, 'acc': 41.2, 'ms': 306},
    'GLA+RoPE':     {'ppl': 25.3, 'acc': 41.1, 'ms': 196},
    'GPT+RoPE':     {'ppl': 25.4, 'acc': 41.2, 'ms': 129},  # from Copy_of_v5_ablation-3
}

# v6 training trajectories (step, PPL)
v6_traj = {
    'GPT+CurvBias': {
        'steps': [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        'ppl':   [60358, 62.3, 40.0, 33.6, 30.2, 27.6, 26.2, 24.9, 24.5],
        'acc':   [0.0, 32.3, 36.6, 38.3, 39.4, 40.3, 41.0, 41.5, 41.6],
    },
    'GPT+RoPE': {
        'steps': [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        'ppl':   [58984, 60.9, 40.5, 34.5, 31.1, 28.8, 26.2, 25.8, 25.4],
        'acc':   [0.0, 32.5, 36.4, 38.0, 39.0, 39.8, 40.9, 41.0, 41.2],
    },
    'GLA+CurvBias': {
        'steps': [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        'ppl':   [60779, 70.3, 47.2, 37.4, 31.9, 29.4, 26.9, 26.1, 25.3],
        'acc':   [0.0, 31.1, 34.3, 37.1, 38.7, 39.6, 40.5, 40.9, 41.2],
    },
    'GLA+RoPE': {
        'steps': [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        'ppl':   [58164, 70.4, 49.0, 38.3, 33.0, 29.0, 26.8, 25.6, 25.3],
        'acc':   [0.0, 31.0, 34.0, 36.6, 38.2, 39.6, 40.5, 41.0, 41.1],
    },
}

colors = {
    'GPT+CurvBias': '#2ca02c',
    'GPT+RoPE':     '#666666',
    'GLA+CurvBias': '#17becf',
    'GLA+RoPE':     '#1f77b4',
    'GPT+CARoPE':   '#e377c2',
    'GLA+CARoPE':   '#bcbd22',
    'GPT':          '#333333',
    'GLA':          '#ff7f0e',
}

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, f'{name}.png'))
    plt.savefig(os.path.join(OUT_DIR, f'{name}.pdf'))
    print(f'Saved: {name}')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Three-Scale PPL Comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('CurvBias vs RoPE: Perplexity Across Three Scales', fontweight='bold', fontsize=14)

models = ['GPT+CurvBias', 'GPT+RoPE', 'GLA+CurvBias', 'GLA+RoPE']
scale_labels = ['Small\n(14M, T=512, 20K)', 'Large\n(77M, T=1024, 20K)', 'v6\n(77M, T=3072, 40K)']
scale_data = [small, large, v6]

x = np.arange(len(scale_labels))
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, model in enumerate(models):
    vals = []
    for sd in scale_data:
        if model in sd:
            vals.append(sd[model]['ppl'])
        else:
            vals.append(0)

    bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                  label=model, color=colors[model], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Value labels
    for j, (bar, v) in enumerate(zip(bars, vals)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(scale_labels)
ax.set_ylabel('Val Perplexity (lower is better)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(small['GPT+RoPE']['ppl'], small['GLA+CurvBias']['ppl']) * 1.15)

plt.tight_layout()
savefig('three_scale_ppl_comparison')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: v6 GPT Convergence Curves (CurvBias vs RoPE)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

# GPT+CurvBias (complete)
cb = v6_traj['GPT+CurvBias']
ax.plot(cb['steps'][1:], cb['ppl'][1:], '-o', color=colors['GPT+CurvBias'],
        label='GPT+CurvBias', markersize=6, linewidth=2)

# GPT+RoPE (complete)
rp = v6_traj['GPT+RoPE']
ax.plot(rp['steps'][1:], rp['ppl'][1:], '--s', color=colors['GPT+RoPE'],
        label='GPT+RoPE', markersize=6, linewidth=2)

# Annotate gap at step 25K (widest)
ax.annotate('', xy=(25000, 27.6), xytext=(25000, 28.8),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(26500, 28.2, '-4.2%', fontsize=10, color='red', fontweight='bold')

# Annotate final gap at 40K
ax.annotate('', xy=(40000, 24.5), xytext=(40000, 25.4),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(37000, 25.7, '-3.5%', fontsize=10, color='red', fontweight='bold')

# Annotate convergence at step 30K then divergence
ax.annotate('Converge\nthen diverge', xy=(30000, 26.2),
            xytext=(32000, 32), fontsize=9, color='#666666', style='italic',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))

ax.set_xlabel('Training Step')
ax.set_ylabel('Val Perplexity')
ax.set_title('v6 GPT Convergence: CurvBias vs RoPE (d=512, T=3072, 77M params)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(2000, 42000)
ax.set_ylim(20, 72)

plt.tight_layout()
savefig('v6_gpt_convergence')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: v6 GLA Convergence Curves (CurvBias vs RoPE)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

gla_cb = v6_traj['GLA+CurvBias']
gla_rp = v6_traj['GLA+RoPE']

ax.plot(gla_cb['steps'][1:], gla_cb['ppl'][1:], '-o', color=colors['GLA+CurvBias'],
        label='GLA+CurvBias', markersize=6, linewidth=2)
ax.plot(gla_rp['steps'][1:], gla_rp['ppl'][1:], '--s', color=colors['GLA+RoPE'],
        label='GLA+RoPE', markersize=6, linewidth=2)

# Annotate mid-training advantage
ax.annotate('', xy=(20000, 31.9), xytext=(20000, 33.0),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(21500, 32.45, '-3.3%', fontsize=10, color='red', fontweight='bold')

# Annotate convergence
ax.annotate('Converge at 25.3', xy=(40000, 25.3),
            xytext=(33000, 28), fontsize=10, color='#666666',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

ax.set_xlabel('Training Step')
ax.set_ylabel('Val Perplexity')
ax.set_title('v6 GLA Convergence: CurvBias vs RoPE (d=512, T=3072, 77M params)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(2000, 42000)
ax.set_ylim(20, 75)

plt.tight_layout()
savefig('v6_gla_convergence')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: CurvBias Advantage Over RoPE Across Scales
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

scale_names = ['Small\n(14M, T=512)', 'Large\n(77M, T=1024)', 'v6\n(77M, T=3072)']
x = np.arange(len(scale_names))

# GPT: CurvBias vs RoPE improvement (negative = CurvBias better)
gpt_adv = [
    -(53.4 - 48.6) / 53.4 * 100,   # Small: -9.0%
    -(38.9 - 37.3) / 38.9 * 100,   # Large: -4.1%
    -(25.4 - 24.5) / 25.4 * 100,   # v6 at 40K: -3.5%
]

# GLA: CurvBias vs RoPE improvement
gla_adv = [
    -(48.6 - 50.5) / 48.6 * 100,   # Small: +3.9% (RoPE wins)
    -(40.9 - 39.9) / 40.9 * 100,   # Large: -2.4% (best PPL)
    0.0,                             # v6: 0% (converge)
]

width = 0.3
bars_gpt = ax.bar(x - width/2, gpt_adv, width, label='GPT (Softmax)',
                   color=colors['GPT+CurvBias'], alpha=0.85, edgecolor='white')
bars_gla = ax.bar(x + width/2, gla_adv, width, label='GLA (Linear)',
                   color=colors['GLA+CurvBias'], alpha=0.85, edgecolor='white')

# Value labels
for bars in [bars_gpt, bars_gla]:
    for bar in bars:
        h = bar.get_height()
        va = 'top' if h < 0 else 'bottom'
        offset = -0.3 if h < 0 else 0.3
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f'{h:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scale_names)
ax.set_ylabel('PPL Improvement vs RoPE (%)\n(negative = CurvBias better)')
ax.set_title('CurvBias Advantage Over RoPE Across Scales')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-12, 6)

# Footnote
ax.text(0.02, -0.12, 'All runs complete at stated step count. v6 GPT+RoPE from v5 framework (architecturally identical).',
        transform=ax.transAxes, fontsize=8, style='italic', color='#666666')

plt.tight_layout()
savefig('curvbias_advantage_across_scales')


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Cost-Performance Frontier
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

markers = {'small': 'o', 'large': 's', 'v6': 'D'}
scale_map = [('small', small, 'Small'), ('large', large, 'Large'), ('v6', v6, 'v6')]

focus_models = ['GPT+CurvBias', 'GPT+RoPE', 'GLA+CurvBias', 'GLA+RoPE']

for scale_key, data, scale_label in scale_map:
    for model in focus_models:
        if model not in data:
            continue
        d = data[model]
        incomplete = d.get('incomplete', False)
        marker = markers[scale_key]
        edge = 'red' if incomplete else colors[model]
        ax.scatter(d['ms'], d['ppl'], c=colors[model], marker=marker, s=100,
                   edgecolors=edge, linewidths=2 if incomplete else 1, zorder=5,
                   alpha=0.6 if incomplete else 0.9)
        # Label
        suffix = '*' if incomplete else ''
        offset_x = 5
        offset_y = 1
        ax.annotate(f'{model}{suffix}\n({scale_label})',
                    xy=(d['ms'], d['ppl']),
                    xytext=(d['ms'] + offset_x, d['ppl'] + offset_y),
                    fontsize=7, color=colors[model])

# Legend for scales
from matplotlib.lines import Line2D
scale_handles = [
    Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=8, label='Small (14M)'),
    Line2D([0], [0], marker='s', color='gray', linestyle='', markersize=8, label='Large (77M)'),
    Line2D([0], [0], marker='D', color='gray', linestyle='', markersize=8, label='v6 (77M, T=3072)'),
]
model_handles = [
    Line2D([0], [0], marker='o', color=colors['GPT+CurvBias'], linestyle='', markersize=8, label='GPT+CurvBias'),
    Line2D([0], [0], marker='o', color=colors['GPT+RoPE'], linestyle='', markersize=8, label='GPT+RoPE'),
    Line2D([0], [0], marker='o', color=colors['GLA+CurvBias'], linestyle='', markersize=8, label='GLA+CurvBias'),
    Line2D([0], [0], marker='o', color=colors['GLA+RoPE'], linestyle='', markersize=8, label='GLA+RoPE'),
]
legend1 = ax.legend(handles=scale_handles, loc='upper left', title='Scale', fontsize=8)
ax.add_artist(legend1)
ax.legend(handles=model_handles, loc='upper center', title='Model', fontsize=8)

ax.set_xlabel('Speed (ms/step)')
ax.set_ylabel('Val Perplexity (lower is better)')
ax.set_title('Cost-Performance Frontier: CurvBias vs RoPE')
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig('cost_performance_frontier')

print(f'\nAll plots saved to: {OUT_DIR}')
