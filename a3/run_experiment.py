"""
run_experiment.py — Assignment 3 experiment runner.

Runs 10-second iperf3 TCP experiments for three congestion control algorithms:
  - tcp_eta  (our custom kernel module)
  - cubic    (Linux default)
  - reno     (TCP Reno baseline)

Each algorithm is tested for --runs independent runs against the same server.
Results are saved as CSVs in data/ and compared via plot_compare.py.

Prerequisites:
  sudo insmod tcp_eta.ko   (module must be loaded before running tcp_eta)

Usage:
  python3 run_experiment.py --server <host> --port 5201 [--runs 3]
"""

import argparse
import csv
import logging
import os
import socket
import time
from pathlib import Path

from client import Iperf3Client, Iperf3Error, AccessDeniedError

import plot_compare

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('a3')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TCP_CONGESTION = 13    # socket option number (IPPROTO_TCP level)

CSV_COLUMNS = [
    'timestamp', 'elapsed', 'goodput_bps',
    'snd_cwnd', 'rtt_us', 'rttvar_us',
    'retrans', 'lost', 'pacing_rate_bps', 'delivery_rate_bps',
    'bytes_acked', 'total_retrans',
]

ALGORITHMS = ['tcp_eta', 'cubic', 'reno']


# ---------------------------------------------------------------------------
# CC-aware client subclass
# ---------------------------------------------------------------------------

class CCIperf3Client(Iperf3Client):
    """
    Thin subclass of Iperf3Client that sets TCP_CONGESTION on the data socket
    immediately after it is created (before data transfer begins).

    TCP_CONGESTION must be set before the connection is established for full
    effect, but Linux also accepts it on an already-connected socket to switch
    the CA algorithm mid-connection.  We set it right after TCP_NODELAY.
    """

    def __init__(self, host, port, duration=10, interval=1.0,
                 timeout=15, cc='cubic'):
        super().__init__(host, port, duration=duration,
                         interval=interval, timeout=timeout)
        self.cc = cc

    def run(self):
        """
        Override run() to inject TCP_CONGESTION setsockopt on the data socket.
        Everything else is identical to the parent implementation.
        """
        import socket as _socket
        import struct as _struct
        import json as _json
        import time as _time

        # We call the parent's internal helpers directly.
        # TCP_CONGESTION is set right after the data socket is opened.
        ctrl      = None
        data_sock = None
        samples   = []

        try:
            # Step 1-2: Control connection + cookie
            ctrl = self._connect('ctrl')
            ctrl.sendall(self._cookie)

            # Step 3-5: Param exchange
            from client import (PARAM_EXCHANGE, ACCESS_DENIED, SERVER_ERROR,
                                 CREATE_STREAMS, TEST_START, TEST_RUNNING,
                                 TEST_RUNNING_ALT, TEST_END, AccessDeniedError,
                                 ServerError, Iperf3Error)

            state = self._recv_state(ctrl)
            if state == ACCESS_DENIED:
                raise AccessDeniedError("ACCESS_DENIED")
            if state == SERVER_ERROR:
                raise Iperf3Error("SERVER_ERROR")
            if state != PARAM_EXCHANGE:
                raise Iperf3Error(f"Expected PARAM_EXCHANGE, got {state}")

            params = {
                'tcp':            True,
                'time':           self.duration,
                'parallel':       1,
                'len':            self.SEND_BUF_SIZE,
                'omit':           0,
                'client_version': '3.17',
            }
            self._send_json(ctrl, params)

            # Step 5: CREATE_STREAMS
            state = self._recv_state(ctrl)
            if state == ACCESS_DENIED:
                raise AccessDeniedError("ACCESS_DENIED after params")
            if state == SERVER_ERROR:
                raise Iperf3Error("SERVER_ERROR after params")
            if state != CREATE_STREAMS:
                raise Iperf3Error(f"Expected CREATE_STREAMS, got {state}")

            # Step 6: Data socket — set CC BEFORE connecting
            data_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            data_sock.setsockopt(_socket.IPPROTO_TCP, TCP_CONGESTION,
                                 self.cc.encode() + b'\x00')
            data_sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            data_sock.settimeout(self.timeout)
            data_sock.connect((self.host, self.port))
            data_sock.sendall(self._cookie)
            logger.debug("Data socket connected with CC=%s", self.cc)

            # Step 7-8: TEST_START / TEST_RUNNING
            state = self._recv_state(ctrl)
            if state not in (TEST_START, TEST_RUNNING, TEST_RUNNING_ALT):
                raise Iperf3Error(f"Expected TEST_START, got {state}")
            if state == TEST_START:
                state = self._recv_state(ctrl)
                if state not in (TEST_RUNNING, TEST_RUNNING_ALT):
                    raise Iperf3Error(f"Expected TEST_RUNNING, got {state}")

            # Step 9: Data loop
            data_sock.settimeout(None)
            ctrl.settimeout(0.01)
            samples = self._data_loop(ctrl, data_sock)

            # Step 10-11: Finish
            ctrl.settimeout(self.timeout)
            self._send_state(ctrl, TEST_END)
            self._exchange_results(ctrl, samples)

        finally:
            for s in (data_sock, ctrl):
                if s:
                    try:
                        s.close()
                    except Exception:
                        pass

        return samples


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    logger.info("Saved %d samples → %s", len(samples), path)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(host, port, cc, duration, interval, timeout):
    """
    Run one 10-second experiment with the given CC algorithm.
    Returns list of samples, or None on failure.
    """
    logger.info("Running cc=%-10s  server=%s:%d", cc, host, port)
    client = CCIperf3Client(host, port, duration=duration,
                            interval=interval, timeout=timeout, cc=cc)
    try:
        samples = client.run()
        if len(samples) < 3:
            logger.warning("Too few samples (%d) for cc=%s", len(samples), cc)
            return None
        avg = sum(s['goodput_bps'] for s in samples) / len(samples)
        logger.info("  → %d samples, avg goodput=%.2f Mbps", len(samples), avg / 1e6)
        return samples
    except AccessDeniedError:
        logger.warning("ACCESS_DENIED for cc=%s", cc)
        return None
    except Exception as exc:
        logger.warning("Failed cc=%s: %s", cc, exc)
        return None


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(host, port, algorithms, n_runs, duration, interval,
            timeout, data_dir):
    """
    Run n_runs experiments for each algorithm. Saves CSVs.
    Returns dict: {cc: [list_of_samples_per_run]}
    """
    os.makedirs(data_dir, exist_ok=True)
    results = {cc: [] for cc in algorithms}

    for cc in algorithms:
        for run_idx in range(1, n_runs + 1):
            logger.info("─── %s  run %d/%d ───", cc, run_idx, n_runs)
            samples = run_one(host, port, cc, duration, interval, timeout)
            if samples:
                csv_path = os.path.join(data_dir, f"{cc}_run{run_idx}.csv")
                save_csv(samples, csv_path)
                results[cc].append(samples)
            else:
                logger.warning("Skipping failed run (cc=%s, run=%d)", cc, run_idx)
            # Brief pause between experiments
            if not (cc == algorithms[-1] and run_idx == n_runs):
                time.sleep(2)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Assignment 3: Compare TCP CC algorithms via iperf3'
    )
    parser.add_argument('--server',   required=True,
                        help='iperf3 server hostname or IP')
    parser.add_argument('--port',     type=int, default=5201,
                        help='iperf3 server port (default: 5201)')
    parser.add_argument('--runs',     type=int, default=3,
                        help='Number of runs per algorithm (default: 3)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration per run in seconds (default: 10)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--timeout',  type=int, default=15,
                        help='Connection timeout in seconds (default: 15)')
    parser.add_argument('--algos',    nargs='+', default=ALGORITHMS,
                        help='Algorithms to test (default: tcp_eta cubic reno)')
    parser.add_argument('--data-dir',    default='data',
                        help='Directory for CSV output (default: data/)')
    parser.add_argument('--results-dir', default='results',
                        help='Directory for plots (default: results/)')
    args = parser.parse_args()

    logger.info("Experiment: server=%s:%d  runs=%d  duration=%ds  algos=%s",
                args.server, args.port, args.runs, args.duration, args.algos)

    results = run_all(
        host=args.server,
        port=args.port,
        algorithms=args.algos,
        n_runs=args.runs,
        duration=args.duration,
        interval=args.interval,
        timeout=args.timeout,
        data_dir=args.data_dir,
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Algorithm':<12} {'Runs OK':<10} {'Avg Goodput (Mbps)'}")
    print("=" * 60)
    for cc, runs in results.items():
        if runs:
            all_samples = [s for run in runs for s in run]
            avg = sum(s['goodput_bps'] for s in all_samples) / len(all_samples)
            print(f"{cc:<12} {len(runs):<10} {avg/1e6:.2f}")
        else:
            print(f"{cc:<12} {'0':<10} N/A")
    print("=" * 60)

    # Generate comparison plots
    logger.info("Generating comparison plots …")
    plot_compare.generate_all(args.data_dir, args.results_dir, args.algos)
    logger.info("Done. Results in %s/", args.results_dir)


if __name__ == '__main__':
    main()
