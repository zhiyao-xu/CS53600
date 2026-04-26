"""
Assignment 4 - Part 2: Maximum Concurrent Flow via MILP (Gurobi)

Problem
-------
n=8 nodes, d=4 directed links in/out per node, each of capacity 1.
Given a traffic matrix T in the hose model (T_ii=0, row/col sums ≤ d=4),
find the d-regular directed topology and flow routing that maximize θ,
where we route θ·T_ij flow for every commodity (i,j).

MILP formulation
----------------
Variables:
  x[i,j]       ∈ {0,1}  — edge i→j exists
  f[i,j,s,t]   ≥ 0      — flow of commodity (s,t) on edge (i,j)
  θ             ≥ 0      — concurrent-flow scaling factor

Maximize θ subject to:
  Σ_j x[i,j] = d          ∀i          (out-degree)
  Σ_i x[i,j] = d          ∀j          (in-degree)
  Σ_v f[u,v,s,t] - Σ_v f[v,u,s,t]
    = θ·T[s,t]  if u=s
    = -θ·T[s,t] if u=t
    = 0         otherwise              (flow conservation per commodity)
  Σ_{s,t} f[i,j,s,t] ≤ x[i,j]        ∀(i,j)  (aggregate capacity)
  f[i,j,s,t] ≤ x[i,j]                 ∀(i,j),(s,t)  (per-commodity tightening)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve(T, n=8, d=4, time_limit=1000, verbose=True):
    """
    Solve the MILP for a given traffic matrix T.

    Parameters
    ----------
    T          : (n, n) array-like, T[i,i]=0, row/col sums ≤ d
    n, d       : network parameters
    time_limit : Gurobi time limit in seconds
    verbose    : whether to print Gurobi log

    Returns
    -------
    dict with keys:
      'theta'    : best θ found (float)
      'obj_bound': upper bound on optimal θ
      'gap'      : MIP gap (|bound - obj| / |obj|)
      'topology' : list of (i,j) edges in the chosen topology
      'status'   : Gurobi status code
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (n, n), "T must be n×n"
    assert np.allclose(np.diag(T), 0), "T must have zero diagonal"

    nodes = list(range(n))
    edges = [(i, j) for i in nodes for j in nodes if i != j]
    # Only create flow variables for commodities with positive demand
    commodities = [(s, t) for s in nodes for t in nodes
                   if s != t and T[s, t] > 1e-8]

    m = gp.Model("max_concurrent_flow")
    if not verbose:
        m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    x   = m.addVars(edges, vtype=GRB.BINARY, name="x")
    f   = m.addVars(edges, commodities, lb=0.0, name="f")
    theta = m.addVar(lb=0.0, ub=1.0, name="theta")

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    m.setObjective(theta, GRB.MAXIMIZE)

    # ------------------------------------------------------------------
    # Topology: exactly d out-edges and d in-edges per node
    # ------------------------------------------------------------------
    for i in nodes:
        m.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == d,
                    name=f"out_{i}")
        m.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == d,
                    name=f"in_{i}")

    # ------------------------------------------------------------------
    # Flow conservation for each commodity (s,t)
    # ------------------------------------------------------------------
    for s, t in commodities:
        demand = T[s, t]
        for u in nodes:
            out_f = gp.quicksum(f[u, v, s, t] for v in nodes if v != u)
            in_f  = gp.quicksum(f[v, u, s, t] for v in nodes if v != u)
            if u == s:
                rhs = theta * demand       # net outflow = θ·T[s,t]
            elif u == t:
                rhs = -theta * demand      # net inflow  = θ·T[s,t]
            else:
                rhs = 0.0
            m.addConstr(out_f - in_f == rhs, name=f"fc_{s}_{t}_{u}")

    # ------------------------------------------------------------------
    # Link capacity constraints
    # ------------------------------------------------------------------
    for i, j in edges:
        # Aggregate: total flow on (i,j) ≤ x[i,j]
        m.addConstr(
            gp.quicksum(f[i, j, s, t] for s, t in commodities) <= x[i, j],
            name=f"cap_{i}_{j}",
        )
        # Per-commodity: tightens LP relaxation (valid since each f ≤ total ≤ x)
        for s, t in commodities:
            m.addConstr(f[i, j, s, t] <= x[i, j],
                        name=f"indcap_{i}_{j}_{s}_{t}")

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    m.optimize()

    if m.SolCount == 0:
        return {"theta": None, "obj_bound": m.ObjBound,
                "gap": None, "topology": None, "status": m.status}

    topology = sorted((i, j) for i, j in edges if x[i, j].X > 0.5)
    return {
        "theta":    theta.X,
        "obj_bound": m.ObjBound,
        "gap":       m.MIPGap,
        "topology":  topology,
        "status":    m.status,
    }


# ---------------------------------------------------------------------------
# Traffic matrix helpers
# ---------------------------------------------------------------------------

def uniform_hose(n=8, d=4):
    """
    Uniform matrix at the hose-model boundary: T[i,j] = d/(n-1) for i≠j.
    Row and column sums = d.
    """
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0)
    return T


def ring_permutation(n=8, d=4):
    """
    All traffic follows the ring pattern: node i → (i+1) mod n, volume d.
    Single permutation, row/col sums = d (tight hose bound).
    """
    T = np.zeros((n, n))
    for i in range(n):
        T[i, (i + 1) % n] = d
    return T


def random_hose(n=8, d=4, rng=None):
    """
    Random traffic matrix inside T_hose via iterative row/column clipping.
    Not guaranteed to be at the boundary, but satisfies all constraints.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    T = rng.random((n, n))
    np.fill_diagonal(T, 0)
    # Iteratively scale rows and columns until sums ≤ d
    for _ in range(200):
        row_sums = T.sum(axis=1, keepdims=True)
        T = np.where(row_sums > d, T * d / row_sums, T)
        col_sums = T.sum(axis=0, keepdims=True)
        T = np.where(col_sums > d, T * d / col_sums, T)
    np.fill_diagonal(T, 0)
    return T


def print_result(name, T, res):
    print(f"\n{'='*60}")
    print(f"Traffic matrix: {name}")
    print(f"  Row sums: {T.sum(axis=1).round(3)}")
    if res["theta"] is None:
        print("  No feasible solution found.")
        return
    print(f"  θ (best found) : {res['theta']:.6f}")
    print(f"  θ (upper bound): {res['obj_bound']:.6f}")
    print(f"  MIP gap        : {res['gap']:.4%}" if res["gap"] is not None else "")
    print(f"  Topology edges ({len(res['topology'])}):")
    adj = np.zeros((8, 8), dtype=int)
    for i, j in res["topology"]:
        adj[i, j] = 1
    print(adj)


# ---------------------------------------------------------------------------
# Main: run on three representative traffic matrices
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n, d = 8, 4

    test_cases = [
        ("Uniform (T[i,j]=4/7)",   uniform_hose(n, d)),
        # ("Ring permutation",        ring_permutation(n, d)),
        # ("Random from T_hose",      random_hose(n, d)),
    ]

    results = {}
    for name, T in test_cases:
        print(f"\nSolving: {name}")
        res = solve(T, n=n, d=d, time_limit=600, verbose=True)
        print_result(name, T, res)
        results[name] = (T, res)
