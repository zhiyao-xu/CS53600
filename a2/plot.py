"""
plot.py — Visualization for Assignment 2.

Generates PDF plots for:
  Q1: Throughput time series + summary table (min/median/avg/p95)
  Q2: TCP stats time series (cwnd, RTT, loss, throughput) + scatter plots

Usage (standalone):
  python3 plot.py --data-dir data/ --results-dir results/ --representative host_port

Called automatically by run_experiment.py.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Consistent colors for destinations
_CMAP = plt.cm.get_cmap('tab20')


def _load_all_csvs(data_dir):
    """Load all CSV files from data_dir. Returns dict: server_key -> DataFrame."""
    data = {}
    for path in sorted(Path(data_dir).glob('*.csv')):
        try:
            df = pd.read_csv(path)
            if df.empty or len(df) < 3:
                continue
            key = path.stem
            data[key] = df
        except Exception:
            continue
    return data


def _mbps(x):
    """Convert bps to Mbps."""
    return x / 1e6


def _ms(x):
    """Convert microseconds to milliseconds."""
    return x / 1e3


# ---------------------------------------------------------------------------
# Q1: Throughput time series
# ---------------------------------------------------------------------------

def plot_throughput_timeseries(data, results_dir):
    """
    Plot goodput time series for all destinations on a single axes.
    Saves: results/throughput_timeseries.pdf
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (key, df) in enumerate(data.items()):
        color = _CMAP(i / max(len(data), 1))
        label = key.replace('_', ':').rsplit(':', 1)
        # Shorten label: last two parts of hostname + port
        label_str = key[:30]
        ax.plot(df['elapsed'], _mbps(df['goodput_bps']),
                label=label_str, color=color, linewidth=1.2, alpha=0.85)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Goodput (Mbps)', fontsize=12)
    ax.set_title('Goodput Time Series — All Destinations', fontsize=14)
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out = os.path.join(results_dir, 'throughput_timeseries.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Q1: Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(data, results_dir):
    """
    Summary table: min/median/mean/p95 goodput per destination.
    Saves: results/summary_table.pdf
    """
    rows = []
    for key, df in data.items():
        g = _mbps(df['goodput_bps'].dropna())
        if g.empty:
            continue
        rows.append({
            'Destination': key[:35],
            'Min (Mbps)':    f"{g.min():.2f}",
            'Median (Mbps)': f"{g.median():.2f}",
            'Avg (Mbps)':    f"{g.mean():.2f}",
            'p95 (Mbps)':    f"{g.quantile(0.95):.2f}",
            'Samples':       len(g),
        })

    if not rows:
        print("[plot] No data for summary table")
        return

    columns = ['Destination', 'Min (Mbps)', 'Median (Mbps)', 'Avg (Mbps)', 'p95 (Mbps)', 'Samples']
    cell_text = [[r[c] for c in columns] for r in rows]

    fig_h = max(3, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(columns))))

    # Style header
    for j in range(len(columns)):
        tbl[(0, j)].set_facecolor('#2c7bb6')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            tbl[(i, j)].set_facecolor(color)

    ax.set_title('Throughput Summary per Destination', fontsize=13, pad=12)

    out = os.path.join(results_dir, 'summary_table.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Q2: TCP stats time series
# ---------------------------------------------------------------------------

def plot_tcp_stats_timeseries(df, server_name, results_dir):
    """
    4-panel time series for a single representative destination:
      Panel 1: snd_cwnd (MSS)
      Panel 2: RTT (ms) with rttvar shading
      Panel 3: Cumulative retransmissions (loss proxy)
      Panel 4: Goodput (Mbps)
    Saves: results/{server_name}_tcp_stats_ts.pdf
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    t = df['elapsed'].values

    # Panel 1: snd_cwnd
    ax = axes[0]
    ax.plot(t, df['snd_cwnd'], color='#1f77b4', linewidth=1.5)
    ax.set_ylabel('snd_cwnd (MSS)', fontsize=11)
    ax.set_title(f'TCP Statistics — {server_name}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 2: RTT (ms) with rttvar band
    ax = axes[1]
    rtt_ms   = _ms(df['rtt_us'].values)
    rttvar_ms = _ms(df['rttvar_us'].values)
    ax.plot(t, rtt_ms, color='#ff7f0e', linewidth=1.5, label='RTT')
    ax.fill_between(t,
                    np.maximum(0, rtt_ms - rttvar_ms),
                    rtt_ms + rttvar_ms,
                    alpha=0.25, color='#ff7f0e', label='±rttvar')
    ax.set_ylabel('RTT (ms)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: Cumulative retransmissions (loss proxy)
    ax = axes[2]
    cum_retrans = df['total_retrans'].values
    ax.plot(t, cum_retrans, color='#d62728', linewidth=1.5)
    ax.set_ylabel('Cumul. Retrans.', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 4: Goodput
    ax = axes[3]
    ax.plot(t, _mbps(df['goodput_bps']), color='#2ca02c', linewidth=1.5)
    ax.set_ylabel('Goodput (Mbps)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    safe = server_name.replace('/', '_').replace(':', '_')
    out = os.path.join(results_dir, f'{safe}_tcp_stats_ts.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Q2: Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(df, server_name, results_dir):
    """
    3 scatter plots for a single representative destination:
      1. snd_cwnd vs goodput
      2. RTT vs goodput
      3. loss signal (per-interval retrans delta) vs goodput
    Saves: results/{server_name}_scatter.pdf
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    goodput = _mbps(df['goodput_bps'].values)
    cwnd    = df['snd_cwnd'].values
    rtt_ms  = _ms(df['rtt_us'].values)

    # Per-interval retransmission delta as loss signal
    retrans = df['total_retrans'].values
    loss_signal = np.diff(retrans, prepend=retrans[0])
    loss_signal = np.maximum(0, loss_signal)

    scatter_kwargs = dict(alpha=0.6, s=20, edgecolors='none')

    # 1: cwnd vs goodput
    ax = axes[0]
    sc = ax.scatter(cwnd, goodput, c=df['elapsed'], cmap='viridis', **scatter_kwargs)
    ax.set_xlabel('snd_cwnd (MSS)', fontsize=11)
    ax.set_ylabel('Goodput (Mbps)', fontsize=11)
    ax.set_title('cwnd vs Goodput', fontsize=12)
    fig.colorbar(sc, ax=ax, label='Time (s)')
    ax.grid(True, alpha=0.3)

    # 2: RTT vs goodput
    ax = axes[1]
    sc = ax.scatter(rtt_ms, goodput, c=df['elapsed'], cmap='viridis', **scatter_kwargs)
    ax.set_xlabel('RTT (ms)', fontsize=11)
    ax.set_ylabel('Goodput (Mbps)', fontsize=11)
    ax.set_title('RTT vs Goodput', fontsize=12)
    fig.colorbar(sc, ax=ax, label='Time (s)')
    ax.grid(True, alpha=0.3)

    # 3: loss signal vs goodput
    ax = axes[2]
    sc = ax.scatter(loss_signal, goodput, c=df['elapsed'], cmap='viridis', **scatter_kwargs)
    ax.set_xlabel('Per-interval Retrans (loss proxy)', fontsize=11)
    ax.set_ylabel('Goodput (Mbps)', fontsize=11)
    ax.set_title('Loss Signal vs Goodput', fontsize=12)
    fig.colorbar(sc, ax=ax, label='Time (s)')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'TCP Metric Correlations — {server_name}', fontsize=13, y=1.02)
    fig.tight_layout()

    safe = server_name.replace('/', '_').replace(':', '_')
    out = os.path.join(results_dir, f'{safe}_scatter.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_all(data_dir, results_dir, representative=None):
    """Generate all required plots."""
    os.makedirs(results_dir, exist_ok=True)
    data = _load_all_csvs(data_dir)

    if not data:
        print(f"[plot] No CSV files found in {data_dir}")
        return

    print(f"[plot] Loaded data for {len(data)} servers")

    # Q1 plots
    plot_throughput_timeseries(data, results_dir)
    plot_summary_table(data, results_dir)

    # Q2 plots: pick representative
    if representative and representative in data:
        rep_key = representative
    else:
        # Use server with most samples as representative
        rep_key = max(data, key=lambda k: len(data[k]))

    print(f"[plot] Using '{rep_key}' as representative server for Q2 plots")
    rep_df = data[rep_key]
    plot_tcp_stats_timeseries(rep_df, rep_key, results_dir)
    plot_scatter(rep_df, rep_key, results_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots for Assignment 2')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--representative', default=None)
    args = parser.parse_args()
    generate_all(args.data_dir, args.results_dir, args.representative)
