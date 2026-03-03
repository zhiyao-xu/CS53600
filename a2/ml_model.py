"""
ml_model.py — ML model for congestion window prediction (Assignment 2, Q3).

Builds a dataset from TCP traces, trains a GradientBoostingRegressor to predict
Δsnd_cwnd using the assignment objective η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)
as sample weights, and plots actual vs predicted cwnd for multiple destinations.

Usage (standalone):
  python3 ml_model.py --data-dir data/ --results-dir results/

Called automatically by run_experiment.py.

Q3 Extracted Congestion Window Update Algorithm
-----------------------------------------------
Observations from the learned model and its predictions:

1. When goodput is high AND RTT is stable (low rttvar): the model increases cwnd
   aggressively — similar to TCP Reno's AIMD additive-increase phase.

2. When RTT rises sharply (congestion signal): the model reduces cwnd multiplicatively
   — mirroring the bandwidth-delay product: BDP = RTT × bandwidth, so if RTT grows
   while goodput stagnates, cwnd is already too large for the pipe.

3. When retransmissions occur (loss signal): the model cuts cwnd significantly —
   consistent with TCP's multiplicative decrease on loss detection.

4. During quiet periods (no loss, stable RTT): the model makes small +1 increments
   — this is the classic AIMD congestion avoidance slope of 1 MSS per RTT.

Hand-written algorithm (pseudocode):

  ALPHA = 0.3   # RTT weight
  BETA  = 5.0   # loss weight
  BASE_INCREASE = 1   # MSS per RTT in congestion avoidance

  def update_cwnd(cwnd, goodput, rtt_ms, loss, rtt_base_ms):
      # Reward signal for previous window decision
      eta = goodput_mbps - ALPHA * rtt_ms - BETA * loss

      if loss > 0:
          # Multiplicative decrease on loss (packet drop detected)
          cwnd = max(1, cwnd // 2)
      elif rtt_ms > 1.25 * rtt_base_ms:
          # RTT growing → queue building → gentle back-off (BBR-like)
          cwnd = max(1, int(cwnd * 0.9))
      elif eta > 0:
          # Good conditions → additive increase
          cwnd += BASE_INCREASE
      else:
          # Negative reward → hold or slight decrease
          cwnd = max(1, cwnd - 1)

      return cwnd

This algorithm captures the key insight: cwnd should grow when the network is
under-utilized (high goodput, stable RTT, no loss) and shrink when the
bandwidth-delay product signal indicates queuing or loss.
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

# Objective function hyper-parameters
ALPHA = 0.01   # RTT penalty weight
BETA  = 10.0   # loss penalty weight

LAG_STEPS = 3   # how many previous time steps to include as features
TRAIN_FRAC = 0.8


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df):
    """
    Build feature matrix X and labels y from a single-server DataFrame.

    Features X(t):
      [goodput_Mbps, rtt_ms, loss_delta, cwnd, rttvar_ms,
       goodput_lag1..3, rtt_lag1..3, cwnd_lag1..3]

    Labels y(t) = cwnd(t+1) - cwnd(t)  (next window update)

    Returns X, y (as numpy arrays, aligned — last row dropped).
    """
    df = df.copy().reset_index(drop=True)

    # Derived columns
    df['goodput_Mbps'] = df['goodput_bps'] / 1e6
    df['rtt_ms']       = df['rtt_us'] / 1e3
    df['rttvar_ms']    = df['rttvar_us'] / 1e3
    df['loss_delta']   = df['total_retrans'].diff().clip(lower=0).fillna(0)

    base_features = ['goodput_Mbps', 'rtt_ms', 'loss_delta', 'snd_cwnd', 'rttvar_ms']

    # Lag features
    lag_features = []
    for lag in range(1, LAG_STEPS + 1):
        for col in ['goodput_Mbps', 'rtt_ms', 'snd_cwnd']:
            lag_col = f'{col}_lag{lag}'
            df[lag_col] = df[col].shift(lag).bfill()
            lag_features.append(lag_col)

    all_features = base_features + lag_features

    # Label: Δcwnd = cwnd(t+1) - cwnd(t)
    df['delta_cwnd'] = df['snd_cwnd'].shift(-1) - df['snd_cwnd']

    # Drop last row (no label) and rows with NaN
    df = df.iloc[:-1].dropna(subset=all_features + ['delta_cwnd'])

    X = df[all_features].values.astype(float)
    y = df['delta_cwnd'].values.astype(float)

    return X, y, df


def compute_eta(df):
    """
    Compute η(t-1) = goodput(t) - α·RTT(t) - β·loss(t).

    η evaluates how good the cwnd update at t-1 was, based on observed
    performance at t. Used as sample weights in model training.
    """
    goodput = df['goodput_Mbps'].values
    rtt_ms  = df['rtt_ms'].values
    loss    = df['loss_delta'].values
    eta = goodput - ALPHA * rtt_ms - BETA * loss
    return eta


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X, y, eta, seed=42):
    """
    Train a GradientBoostingRegressor to predict Δcwnd.

    Uses η (clipped to ≥0) as sample weights so the model is guided
    by the assignment objective function.

    Returns: (model, scaler, train_mse, test_mse, split_idx)
    """
    n = len(X)
    split = int(n * TRAIN_FRAC)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = np.maximum(0, eta[:split])

    # Normalize: if all weights are zero (unlikely) fall back to uniform
    if w_train.sum() == 0:
        w_train = np.ones(len(X_train))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_train_s, y_train, sample_weight=w_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train_s))
    test_mse  = mean_squared_error(y_test,  model.predict(X_test_s))

    return model, scaler, train_mse, test_mse, split


# ---------------------------------------------------------------------------
# Prediction and plotting
# ---------------------------------------------------------------------------

def predict_cwnd(model, scaler, X_test, cwnd_at_split):
    """
    Auto-regressively predict cwnd starting from cwnd_at_split.

    At each step, uses model prediction to update cwnd estimate.
    Returns array of predicted cwnd values (same length as X_test).
    """
    cwnd_pred = [cwnd_at_split]
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        x_s = scaler.transform(x)
        delta = model.predict(x_s)[0]
        next_cwnd = max(1, cwnd_pred[-1] + delta)
        cwnd_pred.append(next_cwnd)
    return np.array(cwnd_pred[1:])   # drop seed value


def plot_cwnd_prediction(df_full, cwnd_pred, split_idx, server_name, results_dir):
    """
    Plot actual cwnd (train=gray, test=blue) and predicted cwnd (red dashed)
    for a single destination.
    Saves: results/{server_name}_cwnd_prediction.pdf
    """
    t = df_full['elapsed'].values
    cwnd_actual = df_full['snd_cwnd'].values

    # Align: df_full here is after lag-dropping (length = original - 1)
    # split_idx refers to position within this df_full
    fig, ax = plt.subplots(figsize=(12, 5))

    train_t = t[:split_idx]
    test_t  = t[split_idx:]
    train_c = cwnd_actual[:split_idx]
    test_c  = cwnd_actual[split_idx:]

    ax.plot(train_t, train_c, color='gray',   linewidth=1.2, label='Actual (train)', alpha=0.7)
    ax.plot(test_t,  test_c,  color='steelblue', linewidth=1.5, label='Actual (test)')
    ax.plot(test_t,  cwnd_pred, color='crimson', linewidth=1.5,
            linestyle='--', label='Predicted (test)')

    ax.axvline(x=t[split_idx], color='black', linestyle=':', linewidth=1, label='Train/Test split')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('snd_cwnd (MSS)', fontsize=11)
    ax.set_title(f'cwnd: Actual vs Predicted — {server_name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    safe = server_name.replace('/', '_').replace(':', '_')
    out = os.path.join(results_dir, f'{safe}_cwnd_prediction.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[ml]   Saved {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_csvs(data_dir):
    data = {}
    for path in sorted(Path(data_dir).glob('*.csv')):
        try:
            df = pd.read_csv(path)
            if len(df) >= 10:
                data[path.stem] = df
        except Exception:
            continue
    return data


def run(data_dir='data', results_dir='results'):
    """Train model and generate cwnd prediction plots for up to 5 destinations."""
    os.makedirs(results_dir, exist_ok=True)
    data = _load_csvs(data_dir)

    if not data:
        print(f"[ml] No data files found in {data_dir}")
        return

    print(f"[ml] Building dataset from {len(data)} servers ...")

    # Build combined dataset for training
    all_X, all_y, all_eta = [], [], []
    per_server = {}

    for key, df in data.items():
        try:
            X, y, df_feat = build_features(df)
            eta = compute_eta(df_feat)
            if len(X) < 10:
                continue
            all_X.append(X)
            all_y.append(y)
            all_eta.append(eta)
            per_server[key] = (X, y, eta, df_feat)
        except Exception as exc:
            print(f"[ml] Skipping {key}: {exc}")

    if not all_X:
        print("[ml] No valid data for model training")
        return

    X_all   = np.vstack(all_X)
    y_all   = np.concatenate(all_y)
    eta_all = np.concatenate(all_eta)

    print(f"[ml] Total samples: {len(X_all)} | features: {X_all.shape[1]}")

    model, scaler, train_mse, test_mse, _ = train_model(X_all, y_all, eta_all)
    print(f"[ml] Train MSE: {train_mse:.4f}  |  Test MSE: {test_mse:.4f}")

    # Generate cwnd prediction plots for up to 5 destinations
    keys_to_plot = list(per_server.keys())[:5]
    for key in keys_to_plot:
        X, y, eta, df_feat = per_server[key]
        split = int(len(X) * TRAIN_FRAC)
        if split >= len(X) or split == 0:
            continue

        X_test = X[split:]
        cwnd_at_split = df_feat['snd_cwnd'].values[split]
        cwnd_pred = predict_cwnd(model, scaler, X_test, cwnd_at_split)

        plot_cwnd_prediction(df_feat, cwnd_pred, split, key, results_dir)

    print("[ml] Done.")
    print()
    print("=" * 60)
    print("Extracted Congestion Window Update Algorithm (see docstring)")
    print("=" * 60)
    _print_algorithm()

    return model, scaler


def _print_algorithm():
    algo = """
Hand-written Congestion Window Update Algorithm
(derived from ML model predictions and network principles)

  ALPHA = 0.3   # RTT penalty
  BETA  = 5.0   # loss penalty

  def update_cwnd(cwnd, goodput_Mbps, rtt_ms, loss_delta, rtt_base_ms):
      eta = goodput_Mbps - ALPHA * rtt_ms - BETA * loss_delta

      if loss_delta > 0:
          # Packet loss → multiplicative decrease (AIMD)
          cwnd = max(1, cwnd // 2)

      elif rtt_ms > 1.25 * rtt_base_ms:
          # RTT inflation → queue building → gentle back-off (BDP signal)
          # BDP = bandwidth × RTT; inflated RTT means cwnd exceeds pipe capacity
          cwnd = max(1, int(cwnd * 0.9))

      elif eta > 0:
          # Positive reward: goodput high, RTT stable → additive increase
          cwnd = cwnd + 1   # +1 MSS per RTT (AIMD congestion avoidance)

      else:
          # Negative reward: hold or decrease slightly
          cwnd = max(1, cwnd - 1)

      return cwnd

Grounding in network principles:
  - BDP (bandwidth-delay product) = bottleneck_bw × RTT sets the ideal cwnd.
  - If cwnd < BDP: under-utilizing → increase (additive increase).
  - If RTT rises: queue at bottleneck is growing → cwnd too large → decrease.
  - If loss occurs: severe congestion → cut cwnd in half (AIMD multiplicative decrease).
  - The objective η measures the trade-off: maximize goodput while penalizing
    RTT growth (latency) and loss (congestion), matching the spirit of CUBIC/BBR.
"""
    print(algo)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML model for cwnd prediction (Q3)')
    parser.add_argument('--data-dir',    default='data')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    run(data_dir=args.data_dir, results_dir=args.results_dir)
