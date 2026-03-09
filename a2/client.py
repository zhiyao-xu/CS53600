"""
iperf3-compatible TCP client with TCP stats collection.

Implements the iperf3 control protocol to communicate with standard public
iperf3 servers (https://iperf3serverlist.net/). Collects TCP socket statistics
(snd_cwnd, RTT, retransmits, bytes_acked, etc.) via getsockopt(TCP_INFO).

Protocol state machine (state bytes on wire):
  PARAM_EXCHANGE   = 9
  CREATE_STREAMS   = 10
  TEST_START       = 1
  TEST_RUNNING     = 2 (some servers use 3)
  TEST_END         = 4
  EXCHANGE_RESULTS = 13
  DISPLAY_RESULTS  = 14
  IPERF_DONE       = 16
  ACCESS_DENIED    = 0xFF
  SERVER_ERROR     = 0xFE
"""

import json
import logging
import random
import select
import socket
import struct
import time

logger = logging.getLogger(__name__)

# iperf3 protocol state constants
PARAM_EXCHANGE   = 9
CREATE_STREAMS   = 10
TEST_START       = 1
TEST_RUNNING     = 2
TEST_RUNNING_ALT = 3   # some server versions use 3
TEST_END         = 4
EXCHANGE_RESULTS = 13
DISPLAY_RESULTS  = 14
IPERF_DONE       = 16
SERVER_TERMINATE = 11
ACCESS_DENIED    = 0xFF
SERVER_ERROR     = 0xFE

# TCP_INFO struct format for Linux kernel
# Matches struct tcp_info in /usr/include/linux/tcp.h
# Fields up through tcpi_snd_wnd (all we need are available in first ~232 bytes)
_TCP_INFO_FMT = (
    'B'   # tcpi_state
    'B'   # tcpi_ca_state
    'B'   # tcpi_retransmits
    'B'   # tcpi_probes
    'B'   # tcpi_backoff
    'B'   # tcpi_options
    'B'   # tcpi_snd_wscale:4 + tcpi_rcv_wscale:4
    'B'   # tcpi_delivery_rate_app_limited:1 + tcpi_fastopen_client_fail:2
    'I'   # tcpi_rto
    'I'   # tcpi_ato
    'I'   # tcpi_snd_mss
    'I'   # tcpi_rcv_mss
    'I'   # tcpi_unacked
    'I'   # tcpi_sacked
    'I'   # tcpi_lost
    'I'   # tcpi_retrans
    'I'   # tcpi_fackets
    'I'   # tcpi_last_data_sent
    'I'   # tcpi_last_ack_sent
    'I'   # tcpi_last_data_recv
    'I'   # tcpi_last_ack_recv
    'I'   # tcpi_pmtu
    'I'   # tcpi_rcv_ssthresh
    'I'   # tcpi_rtt            (microseconds)
    'I'   # tcpi_rttvar         (microseconds)
    'I'   # tcpi_snd_ssthresh
    'I'   # tcpi_snd_cwnd       (in MSS units)
    'I'   # tcpi_advmss
    'I'   # tcpi_reordering
    'I'   # tcpi_rcv_rtt
    'I'   # tcpi_rcv_space
    'I'   # tcpi_total_retrans
    'Q'   # tcpi_pacing_rate    (bytes/s)
    'Q'   # tcpi_max_pacing_rate
    'Q'   # tcpi_bytes_acked    (RFC4898 tcpEStatsAppHCThruOctetsAcked)
    'Q'   # tcpi_bytes_received
    'I'   # tcpi_segs_out
    'I'   # tcpi_segs_in
    'I'   # tcpi_notsent_bytes
    'I'   # tcpi_min_rtt
    'I'   # tcpi_data_segs_in
    'I'   # tcpi_data_segs_out
    'Q'   # tcpi_delivery_rate  (bytes/s)
    'Q'   # tcpi_busy_time
    'Q'   # tcpi_rwnd_limited
    'Q'   # tcpi_sndbuf_limited
    'I'   # tcpi_delivered
    'I'   # tcpi_delivered_ce
    'Q'   # tcpi_bytes_sent
    'Q'   # tcpi_bytes_retrans
    'I'   # tcpi_dsack_dups
    'I'   # tcpi_reord_seen
    'I'   # tcpi_rcv_ooopack
    'I'   # tcpi_snd_wnd
)
_TCP_INFO_SIZE = struct.calcsize(_TCP_INFO_FMT)

# Field index mapping
_IDX = {name: i for i, name in enumerate([
    'state', 'ca_state', 'retransmits', 'probes', 'backoff', 'options',
    'wscale', 'dl_flags',
    'rto', 'ato', 'snd_mss', 'rcv_mss',
    'unacked', 'sacked', 'lost', 'retrans', 'fackets',
    'last_data_sent', 'last_ack_sent', 'last_data_recv', 'last_ack_recv',
    'pmtu', 'rcv_ssthresh',
    'rtt', 'rttvar', 'snd_ssthresh', 'snd_cwnd', 'advmss', 'reordering',
    'rcv_rtt', 'rcv_space', 'total_retrans',
    'pacing_rate', 'max_pacing_rate', 'bytes_acked', 'bytes_received',
    'segs_out', 'segs_in', 'notsent_bytes', 'min_rtt',
    'data_segs_in', 'data_segs_out',
    'delivery_rate', 'busy_time', 'rwnd_limited', 'sndbuf_limited',
    'delivered', 'delivered_ce',
    'bytes_sent', 'bytes_retrans',
    'dsack_dups', 'reord_seen', 'rcv_ooopack', 'snd_wnd',
])}


class Iperf3Error(Exception):
    pass


class AccessDeniedError(Iperf3Error):
    pass


class ServerError(Iperf3Error):
    pass


class Iperf3Client:
    """
    Minimal iperf3-compatible TCP client.

    Performs the full iperf3 handshake with a standard iperf3 server,
    sends data for `duration` seconds, and collects TCP statistics at
    every `interval` seconds using TCP_INFO getsockopt.

    Returns a list of measurement dicts with keys:
      timestamp, elapsed, goodput_bps, snd_cwnd, rtt_us, rttvar_us,
      retrans, lost, pacing_rate_bps, delivery_rate_bps, bytes_acked
    """

    SEND_BUF_SIZE = 128 * 1024  # 128 KB send buffer

    def __init__(self, host, port=5201, duration=60, interval=1.0, timeout=15):
        self.host = host
        self.port = port
        self.duration = duration
        self.interval = interval
        self.timeout = timeout
        self._cookie = self._make_cookie()

    # ------------------------------------------------------------------
    # Cookie
    # ------------------------------------------------------------------

    def _make_cookie(self):
        """Generate a 37-byte cookie matching iperf3's make_cookie().
        Official charset: abcdefghijklmnopqrstuvwxyz234567 (base-32).
        Format: 36 printable chars + null terminator = 37 bytes total.
        """
        chars = 'abcdefghijklmnopqrstuvwxyz234567'
        return (''.join(random.choices(chars, k=36)) + '\x00').encode('ascii')

    # ------------------------------------------------------------------
    # Low-level socket I/O helpers
    # ------------------------------------------------------------------

    def _recv_exactly(self, sock, n):
        """Receive exactly n bytes, raising ConnectionError on EOF."""
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"Connection closed while expecting {n} bytes (got {len(buf)})")
            buf += chunk
        return buf

    def _recv_state(self, sock):
        """Read one state byte from the control socket. Returns int (signed)."""
        data = self._recv_exactly(sock, 1)
        val = data[0]
        # Interpret 0xFF as -1 (ACCESS_DENIED), 0xFE as -2 (SERVER_ERROR)
        if val >= 0xFE:
            val = val - 256
        return val

    def _send_state(self, sock, state):
        """Send one state byte on the control socket."""
        sock.sendall(bytes([state & 0xFF]))

    def _send_json(self, sock, obj):
        """Send a length-prefixed JSON message (4-byte big-endian length + JSON bytes)."""
        data = json.dumps(obj).encode('utf-8')
        sock.sendall(struct.pack('!I', len(data)) + data)

    def _recv_json(self, sock):
        """Receive a length-prefixed JSON message."""
        length_bytes = self._recv_exactly(sock, 4)
        length = struct.unpack('!I', length_bytes)[0]
        if length == 0 or length > 1_000_000:
            raise Iperf3Error(f"Suspicious JSON length: {length}")
        json_bytes = self._recv_exactly(sock, length)
        return json.loads(json_bytes)

    # ------------------------------------------------------------------
    # TCP_INFO extraction
    # ------------------------------------------------------------------

    def _get_tcp_info(self, sock):
        """
        Extract TCP socket statistics via getsockopt(TCP_INFO).
        Returns a dict of TCP metrics, or None on failure.
        """
        try:
            raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, _TCP_INFO_SIZE)
            # Pad if kernel returns fewer bytes than expected
            if len(raw) < _TCP_INFO_SIZE:
                raw = raw + b'\x00' * (_TCP_INFO_SIZE - len(raw))
            fields = struct.unpack(_TCP_INFO_FMT, raw[:_TCP_INFO_SIZE])
            return {
                'snd_cwnd':      fields[_IDX['snd_cwnd']],
                'rtt_us':        fields[_IDX['rtt']],
                'rttvar_us':     fields[_IDX['rttvar']],
                'retrans':       fields[_IDX['retrans']],
                'lost':          fields[_IDX['lost']],
                'pacing_rate':   fields[_IDX['pacing_rate']],
                'delivery_rate': fields[_IDX['delivery_rate']],
                'bytes_acked':   fields[_IDX['bytes_acked']],
                'total_retrans': fields[_IDX['total_retrans']],
            }
        except Exception as exc:
            logger.debug("TCP_INFO failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Socket factory
    # ------------------------------------------------------------------

    def _connect(self, label='ctrl'):
        """Create a connected TCP socket with timeout."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
        except Exception:
            sock.close()
            raise
        logger.debug("[%s] connected to %s:%d", label, self.host, self.port)
        return sock

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Run the full iperf3 test against self.host:self.port.

        Returns a list of sample dicts collected every self.interval seconds.
        Raises Iperf3Error subclasses on protocol errors.
        Raises socket.timeout / OSError on network failures.
        """
        ctrl = None
        data_sock = None
        samples = []

        try:
            # Step 1: Control connection
            ctrl = self._connect('ctrl')

            # Step 2: Send cookie on control connection
            ctrl.sendall(self._cookie)
            logger.debug("Sent cookie: %s", self._cookie.decode())

            # Step 3: Expect PARAM_EXCHANGE from server
            state = self._recv_state(ctrl)
            logger.debug("State after cookie: %d", state)
            if state == ACCESS_DENIED:
                raise AccessDeniedError("Server returned ACCESS_DENIED")
            if state == SERVER_ERROR:
                raise ServerError("Server returned SERVER_ERROR")
            if state != PARAM_EXCHANGE:
                raise Iperf3Error(f"Expected PARAM_EXCHANGE ({PARAM_EXCHANGE}), got {state}")

            # Step 4: Send test parameters (field names match iperf3 source send_parameters())
            params = {
                'tcp':            True,
                'time':           self.duration,
                'parallel':       1,
                'len':            self.SEND_BUF_SIZE,
                'omit':           0,
                'client_version': '3.17',
            }
            self._send_json(ctrl, params)
            logger.debug("Sent params: %s", params)

            # Step 5: Expect CREATE_STREAMS
            state = self._recv_state(ctrl)
            logger.debug("State after params: %d", state)
            if state == ACCESS_DENIED:
                raise AccessDeniedError("Server rejected params: ACCESS_DENIED")
            if state == SERVER_ERROR:
                raise ServerError("Server rejected params: SERVER_ERROR")
            if state != CREATE_STREAMS:
                raise Iperf3Error(f"Expected CREATE_STREAMS ({CREATE_STREAMS}), got {state}")

            # Step 6: Open data connection and send cookie
            data_sock = self._connect('data')
            data_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            data_sock.sendall(self._cookie)
            logger.debug("Data socket connected, cookie sent")

            # Step 7: Expect TEST_START
            state = self._recv_state(ctrl)
            logger.debug("State after data connect: %d", state)
            if state not in (TEST_START, TEST_RUNNING, TEST_RUNNING_ALT):
                raise Iperf3Error(f"Expected TEST_START, got {state}")

            # Step 8: Expect TEST_RUNNING (some servers send both TEST_START then TEST_RUNNING)
            if state == TEST_START:
                state = self._recv_state(ctrl)
                logger.debug("State (TEST_RUNNING?): %d", state)
                if state not in (TEST_RUNNING, TEST_RUNNING_ALT):
                    raise Iperf3Error(f"Expected TEST_RUNNING, got {state}")

            # Step 9: Send data loop with TCP stats collection
            data_sock.settimeout(None)  # blocking for data
            ctrl.settimeout(0.01)       # non-blocking poll for server signals
            samples = self._data_loop(ctrl, data_sock)

            # Step 10: Send TEST_END
            ctrl.settimeout(self.timeout)
            self._send_state(ctrl, TEST_END)
            logger.debug("Sent TEST_END")

            # Step 11: Exchange results
            self._exchange_results(ctrl, samples)

        finally:
            if data_sock:
                try:
                    data_sock.close()
                except Exception:
                    pass
            if ctrl:
                try:
                    ctrl.close()
                except Exception:
                    pass

        return samples

    def _data_loop(self, ctrl, data_sock):
        """
        Send data continuously for self.duration seconds.
        Collect TCP_INFO samples every self.interval seconds.
        Returns list of sample dicts.
        """
        send_buf = b'\x00' * self.SEND_BUF_SIZE
        samples = []
        t_start = time.monotonic()
        t_next_sample = t_start + self.interval
        prev_bytes_acked = 0
        prev_sample_time = t_start

        while True:
            now = time.monotonic()
            elapsed = now - t_start

            if elapsed >= self.duration:
                break

            # Check for server termination signal (non-blocking)
            try:
                state = self._recv_state(ctrl)
                if state in (TEST_END, SERVER_TERMINATE, ACCESS_DENIED, SERVER_ERROR):
                    logger.warning("Server sent terminate signal: %d", state)
                    break
            except (socket.timeout, BlockingIOError):
                pass
            except Exception:
                break

            # Send data chunk
            try:
                data_sock.sendall(send_buf)
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.warning("Data connection broken during send")
                break

            # Collect TCP stats at each interval boundary
            if time.monotonic() >= t_next_sample:
                now_sample = time.monotonic()
                info = self._get_tcp_info(data_sock)
                if info:
                    dt = now_sample - prev_sample_time
                    bytes_acked = info['bytes_acked']
                    delta_acked = max(0, bytes_acked - prev_bytes_acked)
                    goodput_bps = (delta_acked * 8 / dt) if dt > 0 else 0.0

                    sample = {
                        'timestamp':        time.time(),
                        'elapsed':          now_sample - t_start,
                        'goodput_bps':      goodput_bps,
                        'snd_cwnd':         info['snd_cwnd'],
                        'rtt_us':           info['rtt_us'],
                        'rttvar_us':        info['rttvar_us'],
                        'retrans':          info['retrans'],
                        'lost':             info['lost'],
                        'pacing_rate_bps':  info['pacing_rate'] * 8,
                        'delivery_rate_bps':info['delivery_rate'] * 8,
                        'bytes_acked':      bytes_acked,
                        'total_retrans':    info['total_retrans'],
                    }
                    samples.append(sample)
                    prev_bytes_acked = bytes_acked
                    prev_sample_time = now_sample

                t_next_sample = now_sample + self.interval

        return samples

    def _exchange_results(self, ctrl, samples):
        """Handle EXCHANGE_RESULTS state: send our stats JSON, receive server's."""
        try:
            state = self._recv_state(ctrl)
            if state != EXCHANGE_RESULTS:
                logger.debug("Expected EXCHANGE_RESULTS (%d), got %d", EXCHANGE_RESULTS, state)
                return

            # Build minimal results JSON
            total_bytes = samples[-1]['bytes_acked'] if samples else 0
            result = {
                'end': {
                    'sum_sent': {
                        'bytes': total_bytes,
                        'bits_per_second': (total_bytes * 8 / self.duration) if self.duration else 0,
                    }
                }
            }
            self._send_json(ctrl, result)

            # Receive server results (optional — just drain)
            try:
                server_result = self._recv_json(ctrl)
                logger.debug("Server results received: %d bytes total",
                             server_result.get('end', {}).get('sum_sent', {}).get('bytes', 0))
            except Exception:
                pass

            # Drain DISPLAY_RESULTS and IPERF_DONE
            for _ in range(3):
                try:
                    state = self._recv_state(ctrl)
                    logger.debug("Drain state: %d", state)
                    if state == IPERF_DONE:
                        break
                except Exception:
                    break

        except Exception as exc:
            logger.debug("Result exchange error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG)
    host = sys.argv[1] if len(sys.argv) > 1 else 'bouygues.iperf.fr'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5201
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    client = Iperf3Client(host, port, duration=duration, interval=1.0)
    samples = client.run()
    print(f"Collected {len(samples)} samples from {host}:{port}")
    for s in samples:
        print(f"  t={s['elapsed']:.1f}s  goodput={s['goodput_bps']/1e6:.2f} Mbps"
              f"  cwnd={s['snd_cwnd']}  rtt={s['rtt_us']/1000:.1f}ms")
