#!/usr/bin/env python3
import csv, json, math, os, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import matplotlib.pyplot as plt

PING_RE = re.compile(r"min/avg/max/(?:stddev|mdev)\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)")
MS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms")

def sh(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return ""

def ipinfo_json(target=None, token=None):
    url = "https://ipinfo.io/json" if target is None else f"https://ipinfo.io/{target}/json"
    params = {"token": token} if token else None
    return requests.get(url, params=params, timeout=10).json()

def loc_of(target, token):
    try:
        loc = ipinfo_json(target, token).get("loc")
        if not loc: return None
        a, b = loc.split(",")
        return float(a), float(b)
    except Exception:
        return None

def hav_km(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = math.radians(a1), math.radians(a2)
    dp = math.radians(a2 - a1)
    dl = math.radians(o2 - o1)
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(x))

def ping_stats(t, n=100, interval="0.1", w_ms=800):
    out = sh([
        "/sbin/ping",
        "-c", str(n),
        "-i", interval,     # interval between pings
        "-W", str(w_ms),
        t
    ], timeout=max(15, int(n*float(interval)) + 5))
    m = PING_RE.search(out)
    return (None, None, None) if not m else tuple(map(float, m.groups()))

def traceroute_min_rtts(t, max_hops=30, hop_timeout=2):
    out = sh(["/usr/sbin/traceroute", "-n", "-m", str(max_hops), "-q", "3", "-w", str(hop_timeout), t],
             timeout=max_hops * hop_timeout + 10)
    hops = []
    for line in out.splitlines():
        times = [float(x) for x in MS_RE.findall(line)]
        if times: hops.append(min(times))
    return hops

def read_targets_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames: raise SystemExit("CSV missing header.")
        keys = {k.lower(): k for k in r.fieldnames}
        col = keys.get("ip") or keys.get("host") or keys.get("hostname") or r.fieldnames[0]
        targets = [(row.get(col) or "").strip() for row in r]
    out, seen = [], set()
    for t in targets:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out

def scatter(xs, ys, xlabel, ylabel, title, outpdf):
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(outpdf, format="pdf")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("usage: python assignment1.py <parsed_servers.csv> [outdir]", file=sys.stderr)
        sys.exit(2)

    servers = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("IPINFO_TOKEN") or None

    me = ipinfo_json(token=token)  # one call: your IP + loc
    my_ip, my_loc = me.get("ip"), me.get("loc")
    if not (my_ip and my_loc):
        raise SystemExit("ipinfo.io/json did not return ip/loc (rate limit or network).")
    my_lat, my_lon = map(float, my_loc.split(","))

    targets = read_targets_csv(servers)
    if my_ip not in targets:
        targets = [my_ip] + targets

    # parallel ping (fast)
    ping_map = {}
    WORKERS = 60
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(ping_stats, t): t for t in targets}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            ping_map[t] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(targets):
                print(f"ping: {done}/{len(targets)}", flush=True)

    # write results incrementally (so outdir never stays empty)
    rows = []
    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "rtt_min_ms", "rtt_avg_ms", "rtt_max_ms", "lat", "lon", "distance_km"])
        for t in targets:
            rmin, ravg, rmax = ping_map.get(t, (None, None, None))
            loc = loc_of(t, token) if ravg is not None else None
            dist = hav_km(my_lat, my_lon, loc[0], loc[1]) if loc else None
            row = (t, rmin, ravg, rmax, None if not loc else loc[0], None if not loc else loc[1], dist)
            rows.append(row)
            w.writerow(row)
            f.flush()

    xs = [d for (_, _, a, _, _, _, d) in rows if a is not None and d is not None]
    ys = [a for (_, _, a, _, _, _, d) in rows if a is not None and d is not None]
    scatter(xs, ys, "Distance (km)", "Avg RTT (ms)", "Distance vs Avg RTT (ping)", outdir / "dist_vs_rtt.png")

    rtt_avg = {t: a for (t, _, a, _, _, _, _) in rows} # filter out non-responsive hops
    pool = [t for t in targets if t != my_ip and rtt_avg.get(t) is not None]
    random.seed(2205)
    chosen = random.sample(pool, k=min(5, len(pool)))

    traces = []
    for t in chosen:
        hops = traceroute_min_rtts(t)
        traces.append({"target": t, "hop_count": len(hops), "hop_rtts_ms": hops})

    (outdir / "traceroutes.jsonl").write_text("\n".join(json.dumps(x) for x in traces) + "\n", encoding="utf-8")
    with (outdir / "traceroute_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target", "hop_count"])
        w.writerows([(x["target"], x["hop_count"]) for x in traces])

    hx, hy = [], []
    for x in traces:
        a = rtt_avg.get(x["target"])
        if a is not None:
            hx.append(x["hop_count"]); hy.append(a)
    scatter(hx, hy, "Hop count (responsive)", "Avg RTT (ms)", "Hop Count vs Avg RTT", outdir / "hops_vs_rtt.png")

    maxlen = max((len(x["hop_rtts_ms"]) for x in traces), default=0)
    if maxlen:
        labels = [x["target"] for x in traces]
        incs = []
        for x in traces:
            prev, inc = 0.0, []
            for r in x["hop_rtts_ms"]:
                inc.append(max(0.0, r - prev)); prev = r
            inc += [0.0] * (maxlen - len(inc))
            incs.append(inc)

        plt.figure(figsize=(max(8, 1.6 * len(labels)), 5))
        bottom = [0.0] * len(labels)
        for i in range(maxlen):
            h = [incs[j][i] for j in range(len(labels))]
            plt.bar(labels, h, bottom=bottom)
            bottom = [bottom[j] + h[j] for j in range(len(labels))]
        plt.ylabel("Incremental RTT (ms)")
        plt.title("Traceroute hop-by-hop incremental latency")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "hop_latency_stacked.png", dpi=100)
        plt.close()

if __name__ == "__main__":
    main()