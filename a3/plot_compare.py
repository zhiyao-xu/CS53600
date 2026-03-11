"""
plot_compare.py — Comparison plots for Assignment 3.

Reads CSVs from data/{cc}_run{n}.csv and generates:
  1. results/throughput_comparison.pdf  — bar chart + box plot of goodput
  2. results/timeseries_comparison.pdf  — 3-panel time series (goodput, RTT, loss)

Usage (standalone):
  python3 plot_compare.py --data-dir data/ --results-dir results/

Called automatically by run_experiment.py.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colors per algorithm
_COLORS = {
    'tcp_eta': '#e6393a',   # red
    'cubic':   '#1f77b4',   # blue
    'reno':    '#2ca02c',   # green
}
_DEFAULT_COLOR = '#888888'

ALGORITHMS = ['tcp_eta', 'cubic', 'reno']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(data_dir, algorithms):
    """
    Load all CSVs matching {cc}_run*.csv.
    Returns dict: {cc: [DataFrame, ...]}
    """
    data = {cc: [] for cc in algorithms}
    for path in sorted(Path(data_dir).glob('*.csv')):
        for cc in algorithms:
            if path.stem.startswith(cc + '_run'):
                try:
                    df = pd.read_csv(path)
                    if len(df) >= 3:
                        data[cc].append(df)
                except Exception:
                    pass
    return data


# ---------------------------------------------------------------------------
# Plot 1: Throughput comparison (bar + box)
# ---------------------------------------------------------------------------

def plot_throughput_comparison(data, algorithms, results_dir):
    """
    Side-by-side bar chart (mean ± std) with overlaid box plot.
    Saves: results/throughput_comparison.pdf
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gather per-algorithm goodput arrays (all runs concatenated)
    labels    = []
    all_goods = []
    means     = []
    stds      = []

    for cc in algorithms:
        runs = data.get(cc, [])
        if not runs:
            continue
        vals = np.concatenate([df['goodput_bps'].dropna().values / 1e6
                               for df in runs])
        labels.append(cc)
        all_goods.append(vals)
        means.append(vals.mean())
        stds.append(vals.std())

    if not labels:
        print("[plot] No data to plot")
        return

    x     = np.arange(len(labels))
    colors = [_COLORS.get(cc, _DEFAULT_COLOR) for cc in labels]

    # ── Panel 1: Bar chart ──────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Goodput (Mbps)', fontsize=12)
    ax.set_title('Mean Goodput ± Std Dev', fontsize=13)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate bars with mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # ── Panel 2: Box plot ───────────────────────────────────────────────────
    ax = axes[1]
    bp = ax.boxplot(all_goods, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Goodput (Mbps)', fontsize=12)
    ax.set_title('Goodput Distribution per Algorithm', fontsize=13)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('TCP Congestion Control Comparison — Goodput', fontsize=14, y=1.01)
    fig.tight_layout()

    out = os.path.join(results_dir, 'throughput_comparison.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Time series comparison
# ---------------------------------------------------------------------------

def plot_timeseries_comparison(data, algorithms, results_dir):
    """
    4-panel time series showing one representative run per algorithm:
      Panel 1: snd_cwnd (MSS)
      Panel 2: Goodput (Mbps)
      Panel 3: RTT (ms)
      Panel 4: Cumulative retransmissions
    Saves: results/timeseries_comparison.pdf
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)

    plotted_any = False

    for cc in algorithms:
        runs = data.get(cc, [])
        if not runs:
            continue
        # Pick the run with most samples as representative
        df = max(runs, key=len)
        color = _COLORS.get(cc, _DEFAULT_COLOR)
        t = df['elapsed'].values

        # Panel 1: cwnd
        axes[0].plot(t, df['snd_cwnd'].values,
                     color=color, linewidth=1.5, label=cc, alpha=0.85)

        # Panel 2: Goodput
        axes[1].plot(t, df['goodput_bps'].values / 1e6,
                     color=color, linewidth=1.5, label=cc, alpha=0.85)

        # Panel 3: RTT (ms)
        axes[2].plot(t, df['rtt_us'].values / 1e3,
                     color=color, linewidth=1.5, label=cc, alpha=0.85)

        # Panel 4: Cumulative retransmissions
        axes[3].plot(t, df['total_retrans'].values,
                     color=color, linewidth=1.5, label=cc, alpha=0.85)

        plotted_any = True

    if not plotted_any:
        print("[plot] No data for time series plot")
        plt.close(fig)
        return

    titles  = ['snd_cwnd (MSS)', 'Goodput (Mbps)', 'RTT (ms)', 'Cumul. Retransmissions']
    ylabels = ['cwnd (MSS)',      'Goodput (Mbps)', 'RTT (ms)', 'Retrans count']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('TCP CC Comparison — Time Series (one representative run each)',
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out = os.path.join(results_dir, 'timeseries_comparison.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_all(data_dir='data', results_dir='results', algorithms=None):
    if algorithms is None:
        algorithms = ALGORITHMS
    os.makedirs(results_dir, exist_ok=True)
    data = _load(data_dir, algorithms)

    total = sum(len(v) for v in data.values())
    if total == 0:
        print(f"[plot] No CSV files found in {data_dir}/")
        return

    print(f"[plot] Loaded {total} run(s) across {len(algorithms)} algorithms")
    plot_throughput_comparison(data, algorithms, results_dir)
    plot_timeseries_comparison(data, algorithms, results_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CC comparison (Assignment 3)')
    parser.add_argument('--data-dir',    default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--algos', nargs='+', default=ALGORITHMS)
    args = parser.parse_args()
    generate_all(args.data_dir, args.results_dir, args.algos)
