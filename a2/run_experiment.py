"""
run_experiment.py — Main orchestration script for Assignment 2.

Usage:
  python3 run_experiment.py [options]

Options:
  -n, --num-servers    Number of successful servers to test (default: 10)
  --duration           Test duration per server in seconds (default: 60)
  --interval           Sampling interval in seconds (default: 1.0)
  --server-file        Path to file with server list (host:port per line)
                       If omitted, fetches from https://iperf3serverlist.net/
  --data-dir           Directory for CSV output (default: data/)
  --results-dir        Directory for PDF plots (default: results/)
  --skip-plots         Skip plotting step
  --skip-ml            Skip ML model step
  --representative     Server key (host_port) to use for Q2/Q3 plots

The script:
  1. Loads or fetches the iperf3 server list
  2. Picks n random servers, runs experiments (retrying on failure)
  3. Saves per-server CSV files to data/
  4. Calls plot.py to generate all PDFs
  5. Calls ml_model.py to train and plot cwnd predictions
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests

from client import Iperf3Client, Iperf3Error, AccessDeniedError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    'timestamp', 'elapsed', 'goodput_bps',
    'snd_cwnd', 'rtt_us', 'rttvar_us',
    'retrans', 'lost', 'pacing_rate_bps', 'delivery_rate_bps',
    'bytes_acked', 'total_retrans',
]

IPERF3_SERVER_LIST_URL = 'https://export.iperf3serverlist.net/listed_iperf3_servers.json'


# ---------------------------------------------------------------------------
# Server list management
# ---------------------------------------------------------------------------

def fetch_server_list(save_path=None):
    """
    Fetch public iperf3 servers from the JSON API at export.iperf3serverlist.net.
    Returns a list of (host, port) tuples (TCP-capable servers only).
    Saves to save_path if provided.
    """
    logger.info("Fetching server list from %s", IPERF3_SERVER_LIST_URL)
    try:
        resp = requests.get(IPERF3_SERVER_LIST_URL, timeout=15)
        resp.raise_for_status()
        entries = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch server list: %s", exc)
        return []

    servers = []
    for entry in entries:
        host = entry.get('IP/HOST', '').strip()
        if not host:
            continue

        # Skip UDP-only servers (OPTIONS contains -u but not plain TCP)
        options = entry.get('OPTIONS', '')
        if '-u' in options and '-R' not in options and options.strip() == '-u':
            continue

        # PORT may be a single value ("5201") or a range ("9201-9240"); take the first
        port_str = entry.get('PORT', '5201').strip()
        try:
            port = int(port_str.split('-')[0])
        except ValueError:
            port = 5201

        servers.append((host, port))

    logger.info("Found %d servers", len(servers))

    if save_path and servers:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            for host, port in servers:
                f.write(f"{host}:{port}\n")
        logger.info("Saved server list to %s", save_path)

    return servers


def load_server_list(path):
    """Load server list from a file (one host:port per line)."""
    servers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                parts = line.rsplit(':', 1)
                try:
                    port = int(parts[1])
                    servers.append((parts[0], port))
                except ValueError:
                    servers.append((line, 5201))
            else:
                servers.append((line, 5201))
    return servers


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def server_key(host, port):
    """Return a filesystem-safe key for a server."""
    return f"{host.replace(':', '_')}_{port}"


def samples_to_csv(samples, path):
    """Write samples list to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single(host, port, duration, interval, timeout=15):
    """
    Run a single iperf3 experiment.
    Returns list of samples on success, None on failure.
    """
    logger.info("Testing %s:%d (duration=%ds, interval=%.1fs)", host, port, duration, interval)
    client = Iperf3Client(host, port, duration=duration, interval=interval, timeout=timeout)
    try:
        samples = client.run()
        if len(samples) < 3:
            logger.warning("%s:%d — too few samples (%d), skipping", host, port, len(samples))
            return None
        avg_goodput = sum(s['goodput_bps'] for s in samples) / len(samples)
        logger.info("%s:%d — %d samples, avg goodput=%.2f Mbps",
                    host, port, len(samples), avg_goodput / 1e6)
        return samples
    except AccessDeniedError:
        logger.warning("%s:%d — ACCESS_DENIED (rate-limited or blocked)", host, port)
        return None
    except Iperf3Error as exc:
        logger.warning("%s:%d — protocol error: %s", host, port, exc)
        return None
    except Exception as exc:
        logger.warning("%s:%d — connection error: %s", host, port, exc)
        return None


def run_all(servers, n, duration, interval, data_dir, conn_timeout=15):
    """
    Run experiments on n servers, retrying with replacements on failure.

    Returns a dict mapping server_key → list[samples].
    """
    pool = list(servers)
    random.shuffle(pool)

    results = {}
    tried = set()
    idx = 0

    while len(results) < n and idx < len(pool):
        host, port = pool[idx]
        idx += 1
        key = server_key(host, port)

        if key in tried:
            continue
        tried.add(key)

        samples = run_single(host, port, duration, interval, timeout=conn_timeout)
        if samples:
            csv_path = os.path.join(data_dir, f"{key}.csv")
            samples_to_csv(samples, csv_path)
            results[key] = samples
            logger.info("Progress: %d/%d servers done", len(results), n)
        else:
            logger.info("Skipping %s:%d, trying next server", host, port)

        # Brief pause between tests to be a good citizen
        if idx < len(pool):
            time.sleep(1)

    if len(results) < n:
        logger.warning("Only completed %d/%d experiments (not enough working servers)", len(results), n)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run iperf3 experiments and generate plots/ML results')
    p.add_argument('-n', '--num-servers', type=int, default=10,
                   help='Number of servers to test (default: 10)')
    p.add_argument('--duration', type=int, default=60,
                   help='Test duration per server in seconds (default: 60)')
    p.add_argument('--interval', type=float, default=1.0,
                   help='Sampling interval in seconds (default: 1.0)')
    p.add_argument('--server-file', type=str, default=None,
                   help='Path to server list file (host:port, one per line)')
    p.add_argument('--data-dir', type=str, default='data',
                   help='Directory for CSV data files (default: data/)')
    p.add_argument('--results-dir', type=str, default='results',
                   help='Directory for PDF plots (default: results/)')
    p.add_argument('--skip-plots', action='store_true',
                   help='Skip plot generation')
    p.add_argument('--skip-ml', action='store_true',
                   help='Skip ML model training/plotting')
    p.add_argument('--representative', type=str, default=None,
                   help='server_key (host_port) to use for Q2/Q3 detail plots')
    p.add_argument('--timeout', type=int, default=15,
                   help='Connection timeout per server in seconds (default: 15)')
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Step 1: Load or fetch server list
    servers_file = args.server_file or os.path.join(args.data_dir, '..', 'servers.txt')
    if args.server_file and os.path.exists(args.server_file):
        servers = load_server_list(args.server_file)
        logger.info("Loaded %d servers from %s", len(servers), args.server_file)
    else:
        servers = fetch_server_list(save_path=servers_file)
        if not servers:
            logger.error("No servers available. Provide --server-file or check network access.")
            sys.exit(1)

    # Step 2: Run experiments
    if not servers:
        logger.error("Server list is empty")
        sys.exit(1)

    results = run_all(
        servers=servers,
        n=args.num_servers,
        duration=args.duration,
        interval=args.interval,
        data_dir=args.data_dir,
        conn_timeout=args.timeout,
    )

    if not results:
        logger.error("No successful experiments. Check connectivity to iperf3 servers.")
        sys.exit(1)

    # Pick representative server for Q2 detail plots
    representative = args.representative
    if not representative or representative not in results:
        # Choose server with most samples (most stable connection)
        representative = max(results, key=lambda k: len(results[k]))
    logger.info("Representative server for Q2/Q3 plots: %s", representative)

    # Step 3: Generate plots
    if not args.skip_plots:
        logger.info("Generating plots in %s ...", args.results_dir)
        try:
            import plot as plt_module
            plt_module.generate_all(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                representative=representative,
            )
        except Exception as exc:
            logger.error("Plotting failed: %s", exc, exc_info=True)

    # Step 4: ML model
    if not args.skip_ml:
        logger.info("Training ML model ...")
        try:
            import ml_model
            ml_model.run(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
            )
        except Exception as exc:
            logger.error("ML model failed: %s", exc, exc_info=True)

    logger.info("All done. Results in %s/", args.results_dir)


if __name__ == '__main__':
    main()
