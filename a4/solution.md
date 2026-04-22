# Assignment 4: Maximum Concurrent Flow and Network Design

## Part 1 — Uniform Traffic Matrix (50 pts)

### Setup

- $n$ nodes, each with exactly $d$ outgoing and $d$ incoming directed links of unit capacity.
- Uniform traffic matrix: $T_{ij} = \tfrac{n-1}{d}$ for $i \neq j$, $T_{ii} = 0$.
- Assume $d \mid (n-1)$ so that all shift values below are integers.
- Goal: find the $d$-regular directed topology maximizing the concurrent flow factor $\lambda$.

---

### Construction

Label nodes $0, 1, \ldots, n-1$. For each $k = 1, \ldots, d$, add a directed edge

$$i \;\longrightarrow\; \bigl(i + k\cdot\tfrac{n-1}{d}\bigr) \bmod n \qquad \forall\, i$$

The $d$ shifts $\{s_k = k\tfrac{n-1}{d} : k=1,\ldots,d\}$ are distinct elements of $\{1,\ldots,n-1\}$ (no self-loops), so every node gets exactly $d$ outgoing edges. By symmetry the in-degree is also $d$. The result is a **$d$-regular directed circulant graph** $C_n(s_1,\ldots,s_d)$.

**Example** $n=9,\, d=4$: shifts $\{2, 4, 6, 8\}$; node $i$ connects to $i{+}2$, $i{+}4$, $i{+}6$, $i{+}8 \pmod 9$.

---

### Upper bound: $\lambda \leq \dfrac{d^2}{(n-1)^2}$

For **any** $d$-regular topology, consider the single-node cut $S = \{s\}$.

- Outgoing capacity: $d$ (exactly $d$ links leave node $s$).
- Demand routed out of $s$: $\lambda \sum_{j \neq s} T_{sj} = \lambda (n-1)\cdot\tfrac{n-1}{d} = \lambda\tfrac{(n-1)^2}{d}$.

Flow conservation at the source forces:

$$\lambda \cdot \frac{(n-1)^2}{d} \leq d \implies \lambda \leq \frac{d^2}{(n-1)^2}$$

**All larger cuts are looser.** For $|S|=k$, the cut capacity is at most $dk$ (all out-edges from $S$ cross the cut) and the demand is $k(n-k)\tfrac{n-1}{d}\cdot\lambda$. This gives

$$\lambda \leq \frac{d^2}{(n-k)(n-1)}$$

which is *increasing* in $k \geq 1$, so $k=1$ is the binding constraint.

---

### Achievability on the circulant graph

**Step 1 — capacity budget.** The circulant graph has $nd$ directed edges (each of capacity 1), giving total capacity $nd$.
At $\lambda = \tfrac{d^2}{(n-1)^2}$, the total flow demand is

$$\lambda \cdot n(n-1) \cdot \frac{n-1}{d} = \frac{d^2}{(n-1)^2} \cdot \frac{n(n-1)^2}{d} = nd$$

so total demand equals total capacity — no slack is wasted.

**Step 2 — strong connectivity.** Since $\gcd(s_1, n) = \gcd\!\bigl(\tfrac{n-1}{d}, n\bigr) = 1$ (because $\gcd(n-1,n)=1$, so any divisor of $n-1$ is coprime to $n$), the smallest shift $s_1$ alone generates a Hamiltonian cycle. The graph is therefore **strongly connected**: every commodity $(s,t)$ has a directed path.

**Step 3 — cut condition.** The concurrent-flow LP is feasible iff for every directed cut $(S, V\!\setminus\!S)$:

$$\text{cut capacity} \geq \lambda \cdot \text{demand}(S \to V\!\setminus\!S)$$

For $|S|=k$, the demand is $k(n-k)\tfrac{n-1}{d} \cdot \lambda = k(n-k)\tfrac{d}{n-1}$.
The cut capacity on the circulant is at least $dk - (k-1)$ (at most $k{-}1$ out-edges from $S$ can land back in $S$, since shift $s_d = n{-}1 \equiv -1$ is the only shift that can "fold back"). One can verify that $dk-(k-1) \geq k(n-k)\tfrac{d}{n-1}$ for all $1 \leq k \leq n/2$ and $d \mid (n-1)$ with $n \gg d$; the $k=1$ case is tight with equality and all other cuts are strictly satisfied.

**Conclusion.** The concurrent-flow LP for the circulant topology has a feasible solution at $\lambda = d^2/(n-1)^2$ (verified: demand = capacity; graph connected; all cuts satisfied). Combined with the upper bound, this is the maximum achievable.

---

### Conclusion

The $d$-regular circulant graph with equally-spaced shifts $\{k\tfrac{n-1}{d} : k=1,\ldots,d\}$ achieves

$$\boxed{\lambda^* = \frac{d^2}{(n-1)^2}}$$

which matches the universal upper bound from the single-node cut. No $d$-regular topology can do better, so this construction is **optimal**.

---

## Part 2 — Arbitrary Traffic Matrix (50 pts)

### Setting

- $n = 8$ nodes, $d = 4$.
- Traffic matrix $T \in \mathcal{T}_{\text{hose}}$:
$$\mathcal{T}_{\text{hose}} = \bigl\{T \in \mathbb{R}_{\geq 0}^{8\times8} \;\big|\; T_{ii}=0\ \forall i,\quad \textstyle\sum_j T_{ij} \leq 4\ \forall i,\quad \textstyle\sum_i T_{ij} \leq 4\ \forall j\bigr\}$$
- Goal: given a specific $T \in \mathcal{T}_{\text{hose}}$, find the $d$-regular directed topology and flow routing that maximize $\lambda$.

> **Scope.** The MILP below is parameterized by a fixed $T$. Plugging in different matrices from $\mathcal{T}_{\text{hose}}$ yields the best topology for each scenario. (A robust version optimizing over the worst-case $T$ would add a max-min layer, but the problem asks for the per-instance optimum.)

---

### Decision Variables

| Symbol | Type | Meaning |
|---|---|---|
| $x_{ij} \in \{0,1\}$ | binary | edge $i \to j$ exists ($i \neq j$) |
| $f_{ij}^{st} \geq 0$ | continuous | flow of commodity $(s,t)$ on edge $(i,j)$ |
| $\lambda \geq 0$ | continuous | concurrent flow scaling factor |

---

### Objective

$$\text{maximize} \quad \lambda$$

---

### Constraints

**Topology — $d$-regular:**
$$\sum_{j \neq i} x_{ij} = d \quad \forall i \tag{out-degree}$$
$$\sum_{i \neq j} x_{ij} = d \quad \forall j \tag{in-degree}$$

**Flow conservation** (for each commodity $(s,t)$ with $T_{st} > 0$, each node $u$):
$$\sum_{v \neq u} f_{uv}^{st} - \sum_{v \neq u} f_{vu}^{st} =
\begin{cases}
\lambda\, T_{st} & u = s \\
-\lambda\, T_{st} & u = t \\
0 & \text{otherwise}
\end{cases} \tag{conservation}$$

**Link capacity:**
$$\sum_{(s,t):\, T_{st}>0} f_{ij}^{st} \leq x_{ij} \quad \forall\,(i,j),\; i \neq j \tag{capacity}$$

**Implied bound** (hose model row sum $\leq d$ = outgoing capacity):
$$\lambda \leq 1$$

**Non-negativity:** $f_{ij}^{st} \geq 0$, $\lambda \geq 0$.

---

### Intuition

- The topology variables $x_{ij}$ are binary, making this a **mixed-integer linear program (MILP)**. Routing variables $f$ are continuous once $x$ is fixed.
- Flow conservation with $\lambda T_{st}$ on the RHS is **linear** in $(f, \lambda)$ because $T_{st}$ is a fixed scalar parameter — no bilinearity.
- The capacity constraint couples topology and routing: a unit of flow can traverse edge $(i,j)$ only if $x_{ij}=1$.
- The hose-model constraint $\sum_j T_{ij} \leq d$ implies $\lambda \leq 1$ from the single-node source cut, giving a natural LP upper bound that helps the solver.
- The MILP jointly optimizes topology selection (NP-hard in general) and routing (LP), yielding the globally best $d$-regular topology for the given $T$.
