#!/usr/bin/env python3
"""
Assignment 5: Collective Communication
CS 536, Spring 2026

Implements and benchmarks five collective algorithms on the PyTorch gloo backend:

  AllGather:
    - Ring              — N-1 steps, one chunk per step circulates the ring
    - Recursive doubling — log2(N) steps, each rank XOR-partners double their buffer
    - Swing             — log2(N) steps with alternating distances (-2)^s; send and
                          receive targets differ (unlike recursive doubling)

  Broadcast:
    - Binary tree       — each node receives from parent, forwards to ≤2 children
    - Binomial tree     — at step s, ranks 0..2^s-1 send to rank+2^s

All algorithms use isend/irecv to avoid deadlock. Requires world_size = power of 2.

Swing reference: De Sensi et al., "Swing: Short-cutting Rings for Higher Bandwidth
Allreduce," EuroSys 2023.

Usage:
    conda activate CS53600
    python a5.py

Outputs: collective_benchmarks.png
"""
import math
import socket
import statistics
import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ── AllGather ──────────────────────────────────────────────────────────────────

def allgather_ring(tensor, rank, world_size):
    """
    Ring AllGather: N-1 steps.
    Each step: send the current "live" chunk to (rank+1), receive a new chunk
    from (rank-1). After N-1 steps every rank holds all N chunks.
    """
    chunks = [None] * world_size
    chunks[rank] = tensor.clone()
    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size

    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        recv_buf = torch.empty_like(tensor)

        req_s = dist.isend(chunks[send_idx].contiguous(), dst=send_to)
        req_r = dist.irecv(recv_buf, src=recv_from)
        req_s.wait(); req_r.wait()
        chunks[recv_idx] = recv_buf

    return torch.cat(chunks)


def allgather_recursive_doubling(tensor, rank, world_size):
    """
    Recursive Doubling AllGather: log2(N) steps.
    At step s, rank r exchanges its entire accumulated buffer with rank r XOR 2^s.
    Lower-rank side goes first so the concatenated result stays sorted.
    Requires world_size to be a power of 2.
    """
    buf = tensor.clone()
    for s in range(int(math.log2(world_size))):
        partner = rank ^ (1 << s)
        recv = torch.empty_like(buf)
        req_s = dist.isend(buf.contiguous(), dst=partner)
        req_r = dist.irecv(recv, src=partner)
        req_s.wait(); req_r.wait()
        # lower rank holds the lower block, so its data comes first
        buf = torch.cat([buf, recv] if rank < partner else [recv, buf])
    return buf


def _swing_holdings(world_size):
    """
    Precompute which chunk indices each rank holds *before* each Swing step.
    Returns a list of length (steps+1), where entry s is a list of sets,
    one set per rank.  Entry 0 is the initial state ({rank}).
    """
    holdings = [{r} for r in range(world_size)]
    snapshots = [list(holdings)]
    for s in range(int(math.log2(world_size))):
        d = int((-2) ** s)
        next_h = []
        for r in range(world_size):
            recv_from = (r - d) % world_size
            next_h.append(holdings[r] | holdings[recv_from])
        holdings = next_h
        snapshots.append(list(holdings))
    return snapshots


def allgather_swing(tensor, rank, world_size):
    """
    Swing AllGather: log2(N) steps with alternating distances d_s = (-2)^s.
      step s:  rank r  sends to   (r + d_s) % N
                       receives from (r - d_s) % N
    Unlike recursive doubling, the send and receive targets are different ranks,
    forming a chain (step 0: ring distance 1; step 1: distance -2; step 2: +4 …).
    _swing_holdings() precomputes which chunks each rank holds at each step so
    received data can be placed into the right output slots.
    Requires world_size to be a power of 2.
    """
    snapshots = _swing_holdings(world_size)
    output = [None] * world_size
    output[rank] = tensor.clone()

    for s in range(int(math.log2(world_size))):
        d = int((-2) ** s)
        send_to   = (rank + d) % world_size
        recv_from = (rank - d) % world_size

        my_idx   = sorted(snapshots[s][rank])
        peer_idx = sorted(snapshots[s][recv_from])

        send_flat = torch.stack([output[i] for i in my_idx]).contiguous().flatten()
        recv_flat = torch.empty_like(send_flat)   # same size: 2^s chunks each

        req_s = dist.isend(send_flat, dst=send_to)
        req_r = dist.irecv(recv_flat, src=recv_from)
        req_s.wait(); req_r.wait()

        recv_chunks = recv_flat.view(len(peer_idx), tensor.numel())
        for idx, chunk_idx in enumerate(peer_idx):
            output[chunk_idx] = recv_chunks[idx].clone()

    return torch.cat([output[i] for i in range(world_size)])


# ── Broadcast ──────────────────────────────────────────────────────────────────

def broadcast_binary_tree(tensor, rank, world_size, root=0):
    """
    Binary Tree Broadcast.
    Tree structure: parent of r = (r-1)//2, children = 2r+1 and 2r+2.
    Each node blocks on recv from its parent, then isends to its children in
    parallel. No explicit barrier needed — the recv naturally gates forwarding.
    """
    buf = tensor.clone()
    parent   = (rank - 1) // 2 if rank > 0 else -1
    children = [c for c in (2 * rank + 1, 2 * rank + 2) if c < world_size]

    if rank != root:
        dist.recv(buf, src=parent)

    reqs = [dist.isend(buf.contiguous(), dst=c) for c in children]
    for req in reqs:
        req.wait()
    return buf


def broadcast_binomial_tree(tensor, rank, world_size, root=0):
    """
    Binomial Tree Broadcast: ceil(log2(N)) steps.
    At step s (step_size = 2^s):
      - Ranks 0..step_size-1 (which already have the data) send to rank+step_size.
      - Ranks step_size..2*step_size-1 receive from rank-step_size.
    After log2(N) steps every rank has the data.
    """
    buf = tensor.clone()
    steps = math.ceil(math.log2(world_size)) if world_size > 1 else 0
    for s in range(steps):
        step_size = 1 << s
        if rank < step_size:
            dest = rank + step_size
            if dest < world_size:
                dist.isend(buf.contiguous(), dst=dest).wait()
        elif rank < 2 * step_size:
            dist.recv(buf, src=rank - step_size)
    return buf


# ── Benchmark infrastructure ───────────────────────────────────────────────────

ALLGATHER_ALGOS = {
    "ring":          allgather_ring,
    "rec. doubling": allgather_recursive_doubling,
    "swing":         allgather_swing,
}
BROADCAST_ALGOS = {
    "binary tree":   broadcast_binary_tree,
    "binomial tree": broadcast_binomial_tree,
}

N_WARMUP = 3
N_TRIALS = 7

MSG_SIZES   = [1 << k for k in range(10, 25, 2)]  # 1 KB → 16 MB (log2 steps)
FIXED_SIZE  = 1 << 20                              # 1 MB (included in MSG_SIZES)
WORLD_SIZES = [2, 4, 8]


def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, algo_names, is_allgather, msg_sizes, result_queue):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    algos = ALLGATHER_ALGOS if is_allgather else BROADCAST_ALGOS
    results = {}

    for name in algo_names:
        fn = algos[name]
        for msg_size in msg_sizes:
            n = msg_size // 4  # float32 = 4 bytes
            # AllGather: each rank contributes a distinct chunk.
            # Broadcast: rank 0 holds the data; others hold a zero buffer.
            if is_allgather:
                inp = torch.full((n,), float(rank))
            else:
                inp = torch.ones(n) if rank == 0 else torch.zeros(n)

            for _ in range(N_WARMUP):
                dist.barrier()
                fn(inp.clone(), rank, world_size)

            times = []
            for _ in range(N_TRIALS):
                dist.barrier()
                t0 = time.perf_counter()
                fn(inp.clone(), rank, world_size)
                times.append(time.perf_counter() - t0)

            # Take median locally, then MAX across ranks (slowest rank = wall time).
            med = torch.tensor([statistics.median(times)])
            dist.all_reduce(med, op=dist.ReduceOp.MAX)
            if rank == 0:
                results[(name, msg_size)] = med.item()

    if rank == 0:
        result_queue.put(results)
    dist.destroy_process_group()


def run_benchmarks(world_size, algo_names, is_allgather, msg_sizes):
    """Spawn world_size processes, run all (algo × msg_size) combos, return {(name,size): seconds}."""
    port = _find_free_port()
    q = mp.Queue()
    mp.spawn(
        _worker,
        args=(world_size, port, algo_names, is_allgather, msg_sizes, q),
        nprocs=world_size,
        join=True,
    )
    return q.get()


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_vs_size(ax, results, algo_names, world_size, title):
    for name in algo_names:
        ys = [results.get((name, s), float("nan")) * 1e3 for s in MSG_SIZES]
        ax.plot(MSG_SIZES, ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Per-rank message size (bytes)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{title} [{world_size} ranks]")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def _plot_vs_ranks(ax, results_by_ws, algo_names, title):
    for name in algo_names:
        ys = [results_by_ws[ws].get((name, FIXED_SIZE), float("nan")) * 1e3
              for ws in WORLD_SIZES]
        ax.plot(WORLD_SIZES, ys, marker="o", label=name)
    ax.set_xlabel("Number of ranks")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{title} [1 MB per rank]")
    ax.set_xticks(WORLD_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ag_names = list(ALLGATHER_ALGOS)
    bc_names = list(BROADCAST_ALGOS)

    ag_results, bc_results = {}, {}
    for ws in WORLD_SIZES:
        print(f"[AllGather]  {ws} ranks …")
        ag_results[ws] = run_benchmarks(ws, ag_names, True,  MSG_SIZES)
        print(f"[Broadcast]  {ws} ranks …")
        bc_results[ws] = run_benchmarks(ws, bc_names, False, MSG_SIZES)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    _plot_vs_size( axes[0, 0], ag_results[max(WORLD_SIZES)], ag_names,
                   max(WORLD_SIZES), "AllGather: time vs message size")
    _plot_vs_ranks(axes[0, 1], ag_results, ag_names,
                   "AllGather: time vs ranks")
    _plot_vs_size( axes[1, 0], bc_results[max(WORLD_SIZES)], bc_names,
                   max(WORLD_SIZES), "Broadcast: time vs message size")
    _plot_vs_ranks(axes[1, 1], bc_results, bc_names,
                   "Broadcast: time vs ranks")

    plt.tight_layout()
    out = "collective_benchmarks.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
