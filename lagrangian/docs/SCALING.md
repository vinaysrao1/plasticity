# PlasticityFEM.jl — Scaling Design (Route A: CG + AMG, single 64 GB node)

Status: design only (no implementation code). This document is the specification
that implementers and validators rely on for scaling `PlasticityFEM` from the
current direct-solver, fully-cached v1 to **~10 million DOFs on a single 64 GB
multicore CPU node**. It is self-contained and implementation-ready, and it
grounds every change in the existing source (`src/Solver.jl`, `src/Assembly.jl`,
`src/Elements.jl`, `src/Model.jl`, `src/Mesh.jl`, `src/BoundaryConditions.jl`,
`src/Materials.jl`).

The physics kernels (`return_map` in `Materials.jl`, `element_force_tangent!` in
`Elements.jl`) are **reused unchanged**. This work touches the *linear solver*,
the *assembly / data structures*, and adds *threading*. The v1 API for small
problems must keep working.

---

## 0. Target, constraints, and the shape of the problem

### 0.1 The 10M-DOF problem, sized

We target a structured box mesh near `n³` elements with `n ≈ 150` (`150³ =
3.375M` elements). With Hex8 / 3 DOF per node:

| Quantity | Symbol | Value at the target |
|---|---|---|
| Elements | `nelem` | `3.375e6` (≈ 3.4M) |
| Nodes | `nnodes` | `(151)³ ≈ 3.443e6` |
| DOFs | `N = 3·nnodes` | `≈ 1.033e7` (≈ 10.3M) |
| Gauss points | `ngp = 8·nelem` | `2.70e7` (≈ 27M) |
| Nonzeros in K | `nnz` | see below |

**nnz estimate.** In a structured Hex8 grid an interior node couples to the
`3×3×3 = 27` nodes in its element neighborhood. Each node-node coupling is a 3×3
block ⇒ `27 · 9 = 243` scalar nonzeros per interior DOF-row-group, i.e.
`nnz ≈ N · 81` (81 = 27 neighbor nodes × 3 coupled components per row, since each
of the 3 rows of a node couples to 3 components of each of 27 neighbors =
`3·27·3·... ` → per scalar row `27·3 = 81` entries). So:

```
nnz ≈ 81 · N ≈ 81 · 1.033e7 ≈ 8.4e8   (≈ 840M nonzeros)   ✓ matches the ~800M target
```

This is the key structural number: **K has ~80–84 nonzeros per row**, and that
ratio is *mesh-independent* (it is a property of the 27-point Hex8 stencil, not
of `N`). Everything below — memory, flops, AMG cost — is driven by `nnz = O(N)`.

### 0.2 Fixed constraints

- **One node, 64 GB RAM, multicore** (target). Dev sandbox is 15 GB ⇒ the 10M
  run cannot execute here; we **design for 64 GB and validate by scaling-law +
  extrapolation** (Section 6).
- **Route A is fixed:** assemble sparse SPD/symmetric `K`, solve `K δU = −R` with
  **preconditioned Conjugate Gradient (Krylov.jl `cg`) + Algebraic Multigrid
  preconditioner (AlgebraicMultigrid.jl `ruge_stuben` / smoothed aggregation via
  `aspreconditioner`)**. K is SPD in the elastic regime and symmetric for
  associative J2 (the consistent tangent `D_alg` is symmetric — `DESIGN.md` §2.5,
  test T6), so CG is the correct Krylov method.
- **Dependency policy:** add only `Krylov` and `AlgebraicMultigrid` (both
  confirmed installable, pure-Julia). Do **not** pull in `LinearSolve.jl` — it
  drags a heavy dependency tree (Requires/extensions, many backends) for no
  benefit here; we call `Krylov.cg` directly. Keep `SparseArrays`, `StaticArrays`,
  `LinearAlgebra`.

### 0.3 Why v1 cannot reach 10M — the four walls (quantified in Sections 1–2)

1. **Direct LU fill-in** (`Solver.jl:124` `K \ -R`): infeasible in 3D at 10M
   (Section 1.1).
2. **Per-element B cache** (`Elements.jl` `ElementCache.B`): **≈ 31 GB**
   (Section 2.2).
3. **`Vector{Vector{Int}}` scatter map** (`Assembly.jl` `SparsityPattern.map`):
   **≈ 15 GB** (Section 2.3).
4. **COO triplet transient** in `build_sparsity` (`Assembly.jl`): **≈ 47 GB**
   just to build the pattern (Section 2.4).

Any one of (2)(3)(4) alone exceeds or nearly exhausts 64 GB. All four must go.

---

## 1. Solver redesign (Phase 1): inexact Newton + CG + AMG

### 1.1 Why UMFPACK LU is infeasible at 10M in 3D

For a 3D PDE discretized on an `n×n×n` grid (`N ≈ n³` DOFs), a sparse direct
factorization with even an optimal nested-dissection ordering has well-known
asymptotics:

```
LU fill (nonzeros in factors):  O(N^{4/3})   = O(n⁴)
LU flops:                       O(N²)        = O(n⁶)
```

Plug in the v1 baseline and the target:

| Mesh | N | LU fill ~ N^{4/3} | LU flops ~ N² |
|---|---|---|---|
| 40³ (`v1-ish`) | 2.0e5 | ~1.2e7 nz | ~4e10 |
| 100³ | 3.0e6 | ~4.4e8 nz | ~9e12 |
| 150³ (**target**) | 1.0e7 | ~**2.2e9 nz** | ~**1e14** |

`2.2e9` factor nonzeros × (8 B value + 4–8 B index) ≈ **25–40 GB just for the L/U
factors**, *on top of* K, and the peak during factorization (frontal/update
matrices) is higher still — UMFPACK routinely needs 2–4× the final factor size as
working memory. Combined with K (~13 GB) and state (~8 GB) this is **well over 64
GB**, and `~1e14` flops makes a single factorization take minutes to hours
single-node. It also *refactorizes every Newton iteration* in the current
`Solver.jl`. **Direct solve is ruled out.** This is the primary motivation for
Route A.

By contrast (Section 4): **CG + AMG is `O(N)` memory and `O(N)` flops per linear
solve** with a mesh-independent iteration count. That is the entire game.

### 1.2 The new linear solve: PCG with an AMG preconditioner

Replace `Solver.jl:124`

```julia
δU = K \ (-R)              # UMFPACK LU  — REMOVE
```

with a preconditioned CG call (sketch, not final code):

```julia
# ml :: AlgebraicMultigrid.MultiLevel       (the AMG hierarchy, built from K)
# Pl :: preconditioner = AlgebraicMultigrid.aspreconditioner(ml)   (one V-cycle)
δU, stats = Krylov.cg(K, -R;
                      M = Pl,                 # SPD preconditioner (left, symmetric)
                      atol = 0.0,
                      rtol = η_k,             # inexact-Newton forcing term (Section 1.4)
                      itmax = cg_itmax,
                      ldiv = true)            # Pl supports ldiv!(y, Pl, x)
```

- **K stays the assembled `SparseMatrixCSC`** (same object the assembler fills).
  CG only needs `mul!(y, K, x)` (SpMV) and `ldiv!(z, Pl, r)` (one AMG V-cycle).
  Both are `O(nnz) = O(N)`.
- **Preconditioner `M = Pl`.** `AlgebraicMultigrid.ruge_stuben(K)` (classical AMG)
  or `smoothed_aggregation(K)` builds a multilevel hierarchy `ml`;
  `aspreconditioner(ml)` wraps it as an operator whose `ldiv!` applies one
  symmetric V-cycle (symmetric Gauss–Seidel pre/post-smoothing ⇒ SPD operator ⇒
  valid CG preconditioner). For 3D elasticity, **smoothed aggregation** is usually
  the more robust default for the vector (3-DOF-per-node) Laplacian-like operator;
  **classical Ruge–Stüben** is the simpler fallback. We default to smoothed
  aggregation and keep Ruge–Stüben as a config switch (Section 7 risk R1).
- Krylov.jl preallocates its CG workspace; reuse a `CgSolver` (or
  `Krylov.cg!` with a persistent workspace held in the `Model`/solver state) so
  the inner solve allocates `O(1)`, not `O(N)`, per Newton iteration.

### 1.3 AMG hierarchy: when to (re)build vs reuse — the central correctness question

The AMG *setup* (coarsening + building interpolation/restriction and coarse
operators) costs roughly **1–3 SpMV-equivalents-worth of work per level, summed
over the hierarchy ≈ a few× a single V-cycle**, but in practice setup is the
expensive part of AMG and we must *not* rebuild it every Newton iteration.

Key physical fact (from `Materials.jl` / `DESIGN.md` §2): **the tangent `K`
changes only where material is yielding.** In elastic regions `D_alg = ℂ`
(constant); only Gauss points with `f_trial > 0` contribute a different `D_alg`.
Across Newton iterations within one load step, and across load steps, the *bulk*
of K is unchanged and the spectral character (an elliptic, elasticity-like
operator) is preserved even in plastic zones (the consistent tangent stays SPD/
symmetric and well-conditioned for linear hardening, `H_iso,H_kin ≥ 0`).

**Preconditioner reuse policy (this is a preconditioner, not the operator — so
reuse is always *safe*, only *efficiency* is at stake):**

- A *stale* AMG preconditioner never makes CG converge to the wrong answer. CG
  always uses the **true current K** in its `mul!`; `M` only shapes convergence
  speed. So freezing the hierarchy can only cost extra CG iterations, never
  correctness. This is the single most important safety statement of Phase 1.
- **Build the hierarchy once per load step**, from the *first* Newton iterate's K
  (or reuse the previous step's hierarchy and only rebuild on a trigger), and
  **reuse it for all Newton iterations of that step.**
- **Refresh trigger.** Rebuild the hierarchy when CG efficiency degrades, detected
  cheaply by the inner iteration count: if `cg_iters` for a Newton step exceeds
  `f_refresh × (rolling median cg_iters)` (e.g. `f_refresh = 1.5`) or exceeds an
  absolute cap, rebuild from the current K before the next solve. Also force a
  rebuild at the **first load step that develops new plasticity** (detected by a
  rising count of yielding Gauss points). This adaptively rebuilds exactly when
  the operator has drifted, and otherwise amortizes one setup over many solves.
- **Default schedule (simple, robust):** rebuild once at the start of every load
  step; reuse within the step. This is the recommended default; the count-based
  refresh above is the optimization for many-Newton-iteration steps.

Because reuse is only an efficiency question, the implementer can start with
"rebuild every load step" and add the trigger later without any correctness risk.

### 1.4 Inexact Newton (Eisenstat–Walker) — don't over-solve early iterations

Solving the linear system to tight tolerance when the Newton iterate is still far
from the root is wasted work. **Inexact Newton** solves

```
‖ K δU + R ‖ ≤ η_k · ‖R‖          (η_k = forcing term, the CG rtol for iter k)
```

and still converges **superlinearly/quadratically** provided `η_k → 0` as `‖R‖ →
0`. We use the **Eisenstat–Walker choice 2** forcing term (cheap, robust):

```
η_k = γ · ( ‖R_k‖ / ‖R_{k−1}‖ )^α ,     γ ∈ (0,1], α ∈ (1,2]   (e.g. γ=0.9, α=1.5)
η_0 = η_max  (loose, e.g. 0.1)
η_k = clamp(η_k, η_min, η_max)           (η_min ~ 1e-6, η_max ~ 0.1)
# safeguard against oversolving oscillation:
η_k = max(η_k, γ·η_{k−1}^α)  when  γ·η_{k−1}^α > 0.1
```

Effect: early Newton iterations (large residual) use a **loose** CG tolerance
(`~1e-1`) ⇒ very few CG iterations; as `‖R‖` drops, `η_k` tightens toward `η_min`
⇒ the final Newton steps are solved accurately, preserving the quadratic tail
(test T15 in `DESIGN.md` §8.3 must still pass — see Section 6.2). A simpler fixed
schedule (`η = 0.1` until `‖R‖ < tol·100`, then `η = 1e-4`) is an acceptable
fallback and is easier to test deterministically; we ship Eisenstat–Walker with
the simple schedule as a documented fallback.

**Why it composes correctly.** The outer Newton convergence test is unchanged
from v1 (`Solver.jl` lines 112–122): relative residual `‖R‖ ≤ tol·ref`. The inner
CG only needs to reduce the linear residual by `η_k`; since `η_k → η_min ≪ 1` near
convergence, the inexactness is asymptotically negligible and Newton retains its
rate. The two tolerances are independent and *nested*: outer drives `‖R(U)‖→0`,
inner drives `‖KδU+R‖ ≤ η_k‖R‖` for the *current* linear model.

### 1.5 Convergence criteria — inner CG and outer Newton, and how they compose

- **Outer Newton (unchanged):** keep the existing relative test from `Solver.jl`:
  `ref = max(‖R₁‖, ‖Fext‖, 1)` set on iteration 1; converged when
  `‖R‖ ≤ tol·ref` (`tol = 1e-8` default). Cap `maxiter` (default 25).
- **Inner CG:** `rtol = η_k` (Section 1.4), `atol = 0`, `itmax = cg_itmax`.
  Choose `cg_itmax` generously (e.g. `200`) — with a good AMG preconditioner CG
  should need **~5–30 iterations regardless of N** (Section 4); hitting `itmax`
  signals preconditioner trouble (handled in 1.7).
- **Composition:** inexact Newton guarantees the outer sequence still converges
  q-superlinearly with `η_k→0`. The cost per Newton step is
  `O(cg_iters · nnz)`; with mesh-independent `cg_iters` this is `O(N)`.

### 1.6 Dirichlet BCs with an iterative solver — keep K SPD/symmetric (already done)

The v1 `impose_dirichlet!` (`BoundaryConditions.jl`) does **symmetric row/column
elimination**: it moves known columns to the RHS, zeros the constrained rows *and*
columns, and puts `1.0` on the constrained diagonal, with `R[d] = −δg[d]`. This is
**exactly what CG+AMG needs** and requires **no change**:

- The resulting K is still **symmetric** (rows and columns zeroed symmetrically,
  unit diagonal) ⇒ CG is valid.
- It is **SPD** on the elastic problem once enough constraints remove rigid-body
  modes (test T16). The unit-diagonal constrained rows are decoupled `1·δU[d] =
  −δg[d]` equations — they contribute isolated eigenvalue `1` (benign for CG; AMG
  treats them as trivial fine-grid points).
- AMG compatibility: the symmetric elimination leaves the constrained DOFs as
  diagonal-only rows. AMG coarsening handles these as F-points with no strong
  connections; no special handling needed. (If AMG setup is ever unhappy about the
  scale mismatch between unit-diagonal BC rows and physical rows ~`E`, an optional
  refinement is to scale BC diagonal entries to the mean physical diagonal — a
  one-line, symmetry-preserving tweak; not required by default.)

**Conclusion:** Dirichlet handling stays as-is. Do not switch to a penalty method
(penalty inflates the condition number and *hurts* CG/AMG). Symmetric elimination
is the right choice for iterative solves and is already implemented.

### 1.7 Fallback if CG stalls

A robust solver must degrade gracefully:

1. **Refresh the preconditioner.** If CG hits `itmax` (or `stats` shows stagnant
   residual), rebuild the AMG hierarchy from the *current* K and retry the same
   linear solve once. (Cheap, common cure after plasticity spreads.)
2. **Loosen then re-tighten Newton.** If still failing, accept a looser inexact
   tolerance for that Newton step (larger `η_k`) and let the outer loop take an
   extra iteration — globally cheaper than chasing a stalled solve.
3. **Cut the load step.** If a load step's Newton loop fails to converge
   (`maxiter` reached), bisect the load increment (halve `λ` step) and retry — a
   standard plasticity safeguard. This is a small extension of the existing
   `solve!` load loop (`Solver.jl` lines 148–166), which currently just `@warn`s
   and breaks.
4. **Diagonal/Jacobi preconditioner as last resort.** If AMG setup itself fails
   (rare; near-singular blocks), fall back to a Jacobi-preconditioned CG so the
   run still completes (slower, `O(N^{4/3})` — Section 4 — but correct). Emit a
   warning; this is a safety net, not a path to 10M.

---

## 2. Assembly & data-structure redesign (Phase 2): the three memory fixes

This phase deletes the 31 GB cache, the 15 GB map, and the 47 GB COO transient,
and cleanly **separates the one-time symbolic pattern build from the
per-iteration numeric assembly**. The element kernel
(`element_force_tangent!`) and `return_map` are reused unchanged.

### 2.1 Steady-state memory budget (what must coexist at 10M) — and the fixes

| Consumer | v1 form | v1 bytes @10M | New form | New bytes @10M |
|---|---|---|---|---|
| Element B cache | `Vector{SVector{8,SMatrix{6,24}}}` | **≈ 31 GB** | one reference element (uniform box) | **≈ 0.01 MB** |
| Scatter map | `Vector{Vector{Int}}` | **≈ 15 GB** | derived from CSC + edofs (Section 2.5) | **0 (none stored)** |
| Pattern build | COO `I,J,V` len `nelem·576` | **≈ 47 GB transient** | count-then-fill CSC from adjacency | **≈ 0.5 GB transient** |
| K (CSC) | `SparseMatrixCSC{Float64,Int}` | ~13.5 GB | `SparseMatrixCSC{Float64,Int32}` | **~10.4 GB** |
| GaussState ×2 | committed + trial SoA | ~8.4 GB | keep ×2 (Section 2.7) | ~8.4 GB |
| AMG hierarchy | — | — | ~1.3–2× nnz(K) values | **~7–13 GB** |
| CG work vectors | — | — | ~5 × N × 8 B | ~0.4 GB |

The three big deletions (31+15+47 GB) are the heart of Phase 2.

### 2.2 Fix 1 — Element B cache: exploit the uniform box (reference element)

**The arithmetic.** `Elements.jl` caches `B` as
`Vector{SVector{8,SMatrix{6,24,Float64,144}}}`: per element, 8 GPs × (6×24 = 144
Float64) = `1152` Float64 = `9216` B, plus `detJw` 8×8 B. At `nelem = 3.375e6`:

```
3.375e6 · (9216 + 64) B ≈ 3.375e6 · 9280 B ≈ 3.13e10 B ≈ 31 GB     ✗ untenable
```

**Two options.**

- **(A) On-the-fly recompute.** Drop the cache entirely; recompute `dN`, `J`,
  `J⁻¹`, `B`, `detJ` inside the element loop per GP from node coordinates. Memory:
  **0 extra**. Cost: ~a few hundred flops/GP for the trilinear Jacobian + inverse,
  i.e. tens of flops/DOF added — negligible next to the return map and the SpMV
  (Section 4). Works for *any* mesh (general meshes later).

- **(B) Reference-element cache (recommended for the box).** In `box_mesh`
  (`Mesh.jl`) **every element is geometrically identical up to translation**:
  same edge lengths `(lx/nx, ly/ny, lz/nz)`, axis-aligned, so the Jacobian `J` is
  the *same diagonal matrix* for every element and every corresponding GP. Hence
  `B(gp)` and `detJ·w(gp)` are **identical across all elements**. Cache **one**
  set: `B_ref :: SVector{8,SMatrix{6,24}}` and `detJw_ref :: SVector{8}`.

```
1 · (9216 + 64) B ≈ 9.3 kB     ✓ (a factor 3.375e6 smaller)
```

**Recommendation.** Ship **(B) reference-element** as the default for the
structured box mesh (the only generator in v1), and provide **(A) on-the-fly** as
the general fallback for non-uniform meshes (zero memory, modest compute). This is
a clean, small change: `ElementCache` gains a `uniform::Bool` flag and stores
either one reference set or (fallback) recomputes. The element kernel signature
(`element_force_tangent!(mat, Bs, detJw, ue, …)`) is **unchanged** — for the
uniform case the caller simply passes the same `B_ref`, `detJw_ref` for every `e`.

> Compute-vs-memory tradeoff: (B) saves 31 GB and adds *zero* compute (it is the
> v1 math, just not replicated). (A) saves 31 GB and adds ~Jacobian-recompute
> compute per GP. For the box, (B) strictly dominates. Keep (A) for generality.

`ElementCache` (new, minimal):

```julia
struct ElementCache
    uniform::Bool
    Bref::SVector{8, SMatrix{6,24,Float64,144}}   # valid iff uniform
    detJwref::SVector{8, Float64}                  # valid iff uniform
    # fallback (non-uniform): store nothing; recompute from mesh.nodes on the fly
end
```

### 2.3 Fix 2 — Scatter map: stop storing `Vector{Vector{Int}}`

**The arithmetic.** `SparsityPattern.map :: Vector{Vector{Int}}` holds, per
element, 576 `Int` (64-bit) nzval indices:

```
3.375e6 · 576 · 8 B ≈ 1.55e10 B ≈ 15 GB   (+ 3.375e6 vector headers)   ✗
```

**Fix.** Do **not** store a per-element index list at all. There are two
allocation-free ways to scatter `Ke[r,c]` into `K.nzval`, both `O(1)` memory:

- **(Recommended) On-the-fly column search during assembly.** For element `e`,
  for each local column `c` (global col `gcol = edofs[c,e]`), binary-search the
  CSC column once for each of the 24 local rows' global row, exactly as
  `build_sparsity` already does — but do it *in the numeric assembly* instead of
  caching the result. Cost: `24·24` binary searches per element over columns of
  length ≤ ~81 ⇒ `~576·log₂81 ≈ 576·6.3 ≈ 3.6k` comparisons/element ⇒ `~1.2e10`
  comparisons total per assembly. This adds compute but **zero memory**. Since
  assembly is **not** the bottleneck (the CG solve is), this is an acceptable
  trade and is the simplest correct option.

- **(Optimization) Compact `Int32` map for the uniform box.** Because the box is
  structured, the *relative* position of an element's 576 entries within the CSC
  is **periodic** — interior elements share the same local→nzval offset pattern
  (shifted by node index). One can store the map compactly (a handful of stencil
  classes: interior + boundary/edge/corner element types) rather than per element.
  This recovers near-cached speed at `O(1)` memory but adds bookkeeping. **Defer**
  unless assembly profiling demands it; the on-the-fly search is the default.

Either way: **the 15 GB `Vector{Vector{Int}}` is deleted.** If a per-element map
is ever wanted for speed on general meshes, store it as a **single flat
`Matrix{Int32}` of size `576 × nelem`** (`3.375e6·576·4 B ≈ 7.8 GB`) — half the
v1 size and one contiguous allocation (no 3.4M vector headers, cache-friendly).
We do not store this by default.

### 2.4 Fix 3 — Pattern build without the 47 GB COO transient

**The arithmetic.** `build_sparsity` allocates `I,J :: Vector{Int}` and `V ::
Vector{Float64}` each of length `nelem·576`:

```
nelem·576 = 3.375e6·576 ≈ 1.944e9 entries
I + J + V = 1.944e9 · (8 + 8 + 8) B ≈ 4.67e10 B ≈ 47 GB transient   ✗ blows 64 GB
```

`sparse(I,J,V)` then *also* allocates sort/coalesce scratch on top. This single
step is fatal.

**Fix — build the CSC pattern symbolically from node adjacency, count-then-fill,
never materializing COO.** Two-pass construction directly into `colptr/rowval`:

1. **Pass 0 (adjacency).** For the structured box, the column-`j` row set is the
   3×3×3 node neighborhood of node `j` (≤27 nodes ⇒ ≤81 DOF-rows), computable
   from the lexicographic node numbering in `Mesh.jl` *analytically* — no element
   loop needed. (General-mesh fallback: build node→node adjacency by looping
   elements once and inserting into per-node `Set`/sorted small vectors.)
2. **Pass 1 (count).** Compute `nnz` per column and fill `K.colptr` by prefix sum.
   `colptr` is `length N+1` (`~10M·sizeof(Int32) ≈ 40 MB`).
3. **Pass 2 (fill rows).** For each column, write the sorted global DOF rows of
   its neighbor stencil into `K.rowval`; set `nzval .= 0`. `rowval` is `nnz`
   long (`~840M·4 B ≈ 3.4 GB` with Int32).

Peak transient is then just the adjacency scratch (`O(N)` small vectors, ~hundreds
of MB) plus the final `colptr/rowval/nzval` (the steady-state K itself). **No
`nelem·576` array ever exists.** Peak ≈ `K + O(N)` scratch ≈ **~11 GB**, not 47 GB.

This pass is the **symbolic pattern build, run once** (Section 2.6).

### 2.5 Fix the index type: `Int32` columns/rows

`SparseMatrixCSC{Float64,Int}` uses 64-bit `colptr`/`rowval`. At `nnz ≈ 8.4e8`:

```
Int64 indices: rowval 8.4e8·8 B = 6.7 GB, colptr 10M·8 B = 80 MB
Int32 indices: rowval 8.4e8·4 B = 3.4 GB, colptr 10M·4 B = 40 MB     ⇒ saves ~3.3 GB
nzval (Float64, unavoidable): 8.4e8·8 B = 6.7 GB
```

`N ≈ 1.03e7 < 2.15e9 = typemax(Int32)` and `nnz ≈ 8.4e8 < typemax(Int32)`, so
**`Int32` indices are safe** at 10M and save ~3.3 GB. Use
`SparseMatrixCSC{Float64,Int32}`. Krylov.jl, AlgebraicMultigrid.jl, and
`SparseArrays.mul!` all work with `Ti=Int32`. (Document this as the large-problem
path; small problems can keep `Int` — the assembler is written generically over
`Ti`.)

### 2.6 Symbolic vs numeric split — the clean contract

This is the central structural change to `Assembly.jl`.

- **Symbolic (built once, at `Model` construction):**
  `build_sparsity(mesh) -> SparsityPattern` produces the CSC skeleton
  (`colptr/rowval`, `nzval=0`) by the count-then-fill of Section 2.4, plus the
  `edofs :: Matrix{Int32}` (24 × nelem) DOF map (already present). **No per-element
  index map is stored** (Section 2.3). Complexity `O(N)` time, `O(N)` memory.

- **Numeric (every Newton iteration, `O(nelem)=O(N)`, ~0 alloc/element):**
  `assemble!` keeps its current structure (`Assembly.jl` `_assemble!`) but
  scatters via the on-the-fly column search (Section 2.3) instead of `sp.map[e]`,
  and for the uniform box passes the single `cache.Bref/detJwref` to the unchanged
  element kernel. It still does `fill!(nzval,0)`, loops elements, calls
  `element_force_tangent!`, scatters `Fe` into `R` and `Ke` into `nzval`. The
  per-element body remains **allocation-free** (StaticArrays), and the whole
  `assemble!` remains **`O(1)` heap allocation** independent of `nelem` (existing
  test T20 / `DESIGN.md` §7.1 invariant preserved).

New `SparsityPattern` (minimal):

```julia
struct SparsityPattern{Ti<:Integer}
    K::SparseMatrixCSC{Float64,Ti}     # CSC skeleton (colptr,rowval; nzval scratch)
    edofs::Matrix{Ti}                  # 24 × nelem element DOF maps
    # NOTE: no `map::Vector{Vector{Int}}` — scatter does an on-the-fly CSC search.
end
```

### 2.7 GaussState: keep two copies (committed + trial), sized

`GaussState` (`Mesh.jl`) is SoA: `εp,β,σ` are `6×ngp`, `ᾱ` is `ngp`. Per GP that
is `6+6+6+1 = 19` Float64 = `152 B`. At `ngp = 2.7e7`:

```
one GaussState  ≈ 2.7e7 · 152 B ≈ 4.1 GB
committed+trial ≈ 8.2 GB
```

The Newton loop in `Solver.jl` keeps both (path-dependent plasticity needs the
committed state to restart each iteration — line 94 `copyto!(trial, committed)`;
commit copies trial→committed on convergence). **Keep both copies** — 8.2 GB fits
the budget (Section 5) and single-copy schemes would complicate the restart-each-
iteration semantics for no memory headroom we actually need. (If memory ever got
tight, `σ` is output-only and could be dropped from the *trial* copy, saving ~1.4
GB; not needed at 64 GB.) **Optimization, deferred:** `assemble!` currently
re-reads committed and re-runs the return map every Newton iteration anyway, so we
do not need to store trial `σ` during iterations — but the saving is small; keep
the simple two-copy design.

---

## 3. Parallelism (Phase 3): thread-parallel assembly + threaded CG

Target node is multicore; assembly and SpMV must use threads. Order: land Phase 1
(solver) and Phase 2 (memory) first, then thread.

### 3.1 Race-free threaded assembly: mesh coloring (recommended) vs thread-local buffers

The hazard is the scatter `nzval[idx] += Ke[r,c]` (and `R[dof] += Fe[i]`): two
threads assembling elements that share a node write the same `nzval`/`R` entries —
a data race.

**Option A — Mesh coloring (recommended).** Partition elements into **colors**
such that no two elements of the same color share a node. Then all elements of one
color write **disjoint** DOF sets ⇒ scatter them fully in parallel with **no
atomics, no locks, no per-thread buffers**. Loop colors serially, `@threads` over
each color's element list.

- For a structured box, a perfect coloring is **analytic and free**: the standard
  3D "red-black"-style `2×2×2 = 8-color` scheme by element parity
  `(i%2, j%2, k%2)` guarantees same-color elements are ≥2 apart in every axis ⇒
  share no node. **8 colors, computed in `O(nelem)` from the lexicographic index,
  stored as `Vector{Vector{Int32}}` of element ids (~`nelem·4 B ≈ 13.5 MB`).**
- Parallel efficiency: 8 sequential color-sweeps, each ~`nelem/8` elements done in
  parallel. Load is near-perfectly balanced (equal color sizes). Sync overhead = 8
  thread barriers per assembly — negligible.

**Option B — Thread-local accumulation buffers.** Each thread accumulates into a
private `nzval`-shaped (or COO) buffer, then a reduction sums them. Cost: `nthreads
× nnz × 8 B` extra memory — at 16 threads that is `16 · 6.7 GB ≈ 107 GB`. **This
alone blows 64 GB.** A partial-buffer variant (only touched entries) is complex
and still memory-heavy. Rejected on memory grounds.

**Recommendation: mesh coloring (Option A).** It is race-free by construction,
needs ~13.5 MB (vs ~100 GB for buffers), is trivial to generate for the box, and
keeps the element kernel and scatter unchanged (just reordered). Generate colors
once in `build_sparsity`/`Model` and store `colors :: Vector{Vector{Int32}}`.
For general meshes, a greedy graph coloring of the element-adjacency graph gives a
handful of colors at `O(nelem)` cost.

### 3.2 Threaded SpMV / CG

The CG inner loop is dominated by `mul!(y, K, x)` (SpMV) over `nnz ≈ 840M`.

- `SparseArrays.mul!` for CSC `A*x` parallelizes awkwardly (CSC is column-major;
  `A*x` accumulates across columns into shared `y`). Two clean options:
  (i) since K is **symmetric**, store/treat it as CSR-equivalent and do a
  **row-parallel** SpMV (`@threads` over rows, each thread writes disjoint `y[i]`)
  — race-free and cache-friendly; or
  (ii) provide K to Krylov.jl as a symmetric operator and supply a custom threaded
  `mul!` that does the row-parallel product.
- AMG V-cycle (`ldiv!`) smoothing (Gauss–Seidel) is partly sequential; use a
  **colored / hybrid Jacobi–GS** smoother (AlgebraicMultigrid.jl supports Jacobi
  and symmetric GS; Jacobi is fully parallel) on fine levels and exact solve on
  the small coarse level. This keeps the preconditioner threaded where it matters
  (fine-level smoothing dominates).
- CG axpy/dot operations are trivially threaded (`@threads` / `@simd`), `O(N)`.

### 3.3 Expected speedup and Amdahl bottlenecks

- **Assembly:** ~linear speedup in cores up to memory-bandwidth saturation (the
  scatter is bandwidth-bound, not compute-bound). Expect 6–10× on a 16-core node
  before bandwidth caps it.
- **CG/SpMV:** **memory-bandwidth bound** (SpMV reads `nnz` Float64 + indices per
  iteration ≈ 10 GB/iteration of traffic). Speedup tracks usable memory bandwidth,
  not core count — typically saturates at a fraction of the cores. This is the
  dominant Amdahl limit: the linear solve is the long pole and it is
  bandwidth-limited.
- **Serial residue:** AMG coarse-level solve and hierarchy setup have a serial
  tail (small coarse problems); the V-cycle's coarsest levels are tiny so the
  serial fraction is small but nonzero. Setup, if rebuilt every step, can become
  Amdahl-significant — another reason for the reuse policy (Section 1.3).

---

## 4. Complexity & optimality targets (the validator will check these)

State the asymptotics explicitly; these are the contract Section 6 verifies.

| Stage | Target complexity | Why |
|---|---|---|
| Symbolic pattern build | **`O(N)`** time & mem | count-then-fill from adjacency, ≤81 nz/row (Section 2.4) |
| Numeric assembly / Newton iter | **`O(nelem) = O(N)`** | loop elements, `O(1)` work/element; ~0 alloc/element |
| SpMV (per CG iter) | **`O(nnz) = O(N)`** | 81 nz/row, fixed |
| AMG V-cycle (per CG iter) | **`O(nnz) = O(N)`** | optimal AMG: work ∝ Σ nnz over levels = `c·nnz`, geometric |
| **CG iteration count** | **`≈ const` (mesh-independent)** | AMG gives an `O(1)`-condition preconditioned operator ⇒ flat iters |
| Linear solve / Newton iter | **`O(N)`** | `(const iters) × O(N)` |
| Newton iters / load step | **`O(1)`** | quadratic local convergence (consistent tangent, T15) |
| **Total / load step** | **`O(N)`** | product of the above |

**What breaks `O(N)`** (call-outs the validator checks):

- **Jacobi/diagonal preconditioner instead of AMG.** CG iteration count for a 3D
  elliptic operator with Jacobi scales like `κ^{1/2} ∼ h⁻¹ ∼ N^{1/3}` ⇒ linear
  solve becomes `O(N·N^{1/3}) = O(N^{4/3})`, and total `O(N^{4/3})`. **AMG is what
  buys flat iteration count and overall `O(N)`.** Losing AMG (the 1.7 fallback) is
  a correctness-preserving but complexity-breaking degradation.
- **Direct LU.** `O(N²)` flops / `O(N^{4/3})` memory (Section 1.1) — ruled out.
- **Storing the per-element map / COO** (Sections 2.3–2.4) — `O(N)` memory with a
  huge constant (15 / 47 GB) that breaks the *budget* even though it is formally
  `O(N)`.

**Quantitative per-DOF targets (the budget Section 5 must hit):**

- **bytes/DOF (steady state):** target **≤ ~4.0 kB/DOF** total RSS.
  - K (Int32): `nnz/N · (8 + 4) B + (colptr) ≈ 81·12 ≈ 972 B + ~4 B ≈ ~0.98 kB`.
  - GaussState ×2: `2·152 B/GP · 8 GP/elem / 3 DOF/... ` → `8.2 GB / 10.3M ≈ 0.8
    kB/DOF`.
  - AMG hierarchy: `~1.5× nnz values ≈ 1.5·0.65 kB ≈ ~1.0–1.5 kB/DOF`.
  - CG vectors + misc: `~0.1 kB/DOF`.
  - **Sum ≈ 3.0–3.6 kB/DOF ⇒ ~31–37 GB at 10M** (Section 5 table).
- **flops/DOF per Newton iter:** assembly `~few×10³ flops/DOF` (return map + 24×24
  element products, GP loop) + linear solve `cg_iters·(SpMV+Vcycle) ≈ 20 ·
  (≈ 2·81 + ≈ 5·81) flops/DOF ≈ 20·~560 ≈ ~1.1e4 flops/DOF`. **Target order
  `~10⁴ flops/DOF per Newton iteration`**, mesh-independent.

---

## 5. The 10M-on-64GB memory budget (line-item, the binding table)

Steady-state resident memory during a Newton solve at the target
(`N=1.033e7`, `nnz=8.4e8`, `ngp=2.7e7`, **Int32 indices**):

| Line item | Formula | Bytes | Notes |
|---|---|---|---|
| K `nzval` (Float64) | `nnz·8` | **6.7 GB** | unavoidable; the operator |
| K `rowval` (Int32) | `nnz·4` | **3.4 GB** | Int32 saves 3.3 GB vs Int64 |
| K `colptr` (Int32) | `(N+1)·4` | 0.04 GB | |
| **K total** | | **≈ 10.1 GB** | |
| GaussState committed | `ngp·152` | 4.1 GB | εp,β,σ (6 each) + ᾱ |
| GaussState trial | `ngp·152` | 4.1 GB | restart-each-iter semantics |
| **State total** | | **≈ 8.2 GB** | |
| AMG hierarchy | `~1.3–2.0 × nnz·(8+4)` | **≈ 7–13 GB** | coarse operators + P/R + smoother data; **the swing factor & likely binding constraint** |
| CG work vectors | `~5 · N · 8` | 0.4 GB | r, p, Ap, z, x (Krylov workspace) |
| `edofs` (Int32) | `24·nelem·4` | 0.32 GB | element DOF map |
| Mesh nodes/elements | `3·nnodes·8 + 8·nelem·8` | 0.30 GB | |
| Coloring lists (Int32) | `nelem·4` | 0.014 GB | 8-color element ids |
| Element B reference | `~9.3 kB` | ~0 | uniform-box single ref element |
| Misc (Fext, U, Rbuf, scratch) | `~4·N·8` | 0.33 GB | |
| **Steady-state total (AMG≈1.5×)** | | **≈ 29–32 GB** | comfortably < 64 GB |
| **Steady-state total (AMG≈2.0×)** | | **≈ 32–36 GB** | still < 64 GB |

**Transient peaks (must also fit 64 GB):**

- **Symbolic pattern build (Section 2.4):** `K` + `O(N)` adjacency scratch ≈
  **~11 GB**. (vs **47 GB** for the deleted COO path — the whole point.)
- **AMG setup:** transiently allocates the next-level operators while building;
  peak ≈ steady AMG + one extra level ≈ **+1–3 GB** over steady. Bounded.

**Binding constraint:** the **AMG hierarchy size** (7–13 GB) is the largest single
swing and the most uncertain line. If smoothed-aggregation hierarchies come out at
the high end (operator complexity ~2), total ~36 GB — still well inside 64 GB,
leaving comfortable headroom. **The design fits 64 GB with ~2× margin.** The
deletions in Phase 2 are what make this true: without them the budget is
31 (B cache) + 15 (map) + 47 (COO transient) GB over.

---

## 6. Verification & validation plan (built for the 15 GB sandbox)

The 10M run cannot execute in 15 GB. We validate by **(a) correctness on moderate
meshes vs the existing direct solver, (b) physics regression, and (c) measuring
the scaling laws on a refinement sweep up to ~1–2M DOFs, then extrapolating to
10M with the explicit budget table of Section 5.**

### 6.1 Correctness: CG+AMG vs direct solver (medium mesh)

- **C1 — Identical solution.** On a mesh that *both* solvers handle (e.g.
  `30³–40³`, `~0.08–0.2M` DOFs), solve the same load step with (i) v1
  `K \ -R` and (ii) PCG+AMG. Assert nodal displacements and Gauss stresses agree
  to the CG tolerance: `‖U_cg − U_lu‖/‖U_lu‖ ≤ 10·rtol` (e.g. ≤ 1e-7 for
  `rtol=1e-8`). This proves the iterative path computes the same answer.
- **C2 — SPD/symmetry preserved through BC imposition.** Re-assert T16: after
  `impose_dirichlet!`, K is symmetric and CG (which assumes symmetry) converges;
  AMG setup succeeds. Confirms Section 1.6.
- **C3 — Inexact Newton preserves the Newton rate.** Re-run T15 (quadratic
  convergence) with inexact CG tolerances: assert the last 2–3 Newton residuals
  still satisfy `‖R_{k+1}‖ ≤ C‖R_k‖²` and the step converges in ≤ ~6 Newton
  iterations. Confirms Section 1.4 composition.

### 6.2 Physics regression (existing suite must still pass)

All existing tests in `test/` (T1–T20, `hard_validation.jl`, `test_material.jl`,
`test_solver.jl`, `test_assembly.jl`, `test_element.jl`) must pass **unchanged**
with the new solver and data structures swapped in for moderate meshes:

- Material kernel tests (T1–T8) — untouched (`return_map` unchanged).
- Uniaxial tension (T2/T13), perfect plasticity (T3), Bauschinger (T4), cantilever
  (T14), reaction balance (T17), load-path consistency (T18), unload (T19) — must
  reproduce the v1 results to tolerance with PCG+AMG.
- Assembly allocation (T20): `@allocated assemble!` stays `O(1)` (bounded
  constant, independent of `nelem`) with the new on-the-fly scatter — **assert
  this explicitly** (Section 6.4).

### 6.3 Scaling sweep (prove the laws up to ~1–2M DOFs)

Refinement sweep `n ∈ {20, 40, 60, 80, ~100}` elements/side (DOFs ≈ `0.03M,
0.2M, 0.68M, 1.6M, 3.0M` — cap at what fits 15 GB; ~80³–90³ is the practical
ceiling in the sandbox). For each mesh, **measure and assert trends**:

| Quantity | Measured | Asserted scaling |
|---|---|---|
| peak RSS / DOF | bytes resident / N | **flat (linear total memory)**: RSS/DOF within ±20% across the sweep |
| nnz / DOF | `nnz(K)/N` | **flat ≈ 81** (constant, ±5%) |
| CG iterations / Newton step | from Krylov `stats.niter` | **≈ flat** (mesh-independent under AMG): max iters within a small band (e.g. ≤ 1.5× the coarsest-mesh count), **not** growing like `N^{1/3}` |
| wall-time / DOF (assembly) | assemble time / N | **flat (linear)** within ±25% |
| wall-time / DOF (linear solve) | solve time / N | **near-flat**; slope in log–log of total time vs N ≈ **1.0** (assert ≤ ~1.15) |
| flops / DOF / Newton iter | derived (iters·nnz) | **flat ~10⁴** |

**The decisive assertion is flat CG-iteration-count**: fit `cg_iters` vs `N`; the
exponent must be `≈ 0` (AMG optimality), and a control run with **Jacobi**
preconditioning must visibly grow (`∼N^{1/3}`) to demonstrate that AMG is what
delivers `O(N)` (Section 4). This is the single most important scaling check.

### 6.4 Allocation & per-iteration checks

- `@allocated assemble!(...)` bounded by a small constant independent of `nelem`
  (on-the-fly scatter must not allocate per element). Hard gate.
- `@allocated` of one CG iteration's `mul!` and AMG `ldiv!` ~0 (persistent
  workspaces). Hard gate on the SpMV; AMG may have minor internal allocation —
  bound it.
- Element kernel `@allocated == 0` (unchanged from v1 T8/T20).

### 6.5 Extrapolated 10M budget (the deliverable assertion)

From the measured `RSS/DOF`, `nnz/DOF`, and AMG-hierarchy-bytes/DOF on the
≤2M sweep, **extrapolate linearly to `N=10.3M`** and emit the Section 5 table with
*measured* per-DOF constants substituted in. **Assert the extrapolated total <
64 GB** (with the expected ~30–36 GB landing well under). Also extrapolate
wall-time/load-step = `(time/DOF)·N` to report the projected 10M solve time. This
extrapolation, backed by the proven flat trends, is the validation that the design
meets the 10M-on-64GB target without executing it in the sandbox.

---

## 7. Phasing, risks, decision points

### 7.1 Phase order and gates

- **Phase 1 — Solver (CG + AMG + inexact Newton).** Smallest, highest-value
  change; touches `Solver.jl` (replace `K\-R`), adds `Krylov`+`AlgebraicMultigrid`
  deps, threads nothing yet. **Gate:** C1 (matches direct solver), C3 (Newton rate
  preserved), full physics suite (6.2) passes on moderate meshes. Delivers: the
  algorithmic path to large N, runnable on meshes the *memory* still allows.
- **Phase 2 — Memory (reference element, scatter-search, count-then-fill CSC,
  Int32).** Touches `Elements.jl` (`ElementCache`), `Assembly.jl`
  (`build_sparsity`, `_assemble!`, `SparsityPattern`), `Model.jl` (wiring). **Gate:**
  same physics suite passes; T20 allocation bound holds; scaling sweep (6.3) shows
  flat RSS/DOF and nnz/DOF; peak transient during pattern build is `O(N)` (~11 GB
  trend), **not** the 47 GB COO path. Delivers: the budget that fits 64 GB.
- **Phase 3 — Threads (8-color assembly, row-parallel SpMV, parallel smoother).**
  Touches assembly loop ordering + a custom threaded `mul!`. **Gate:** identical
  results to single-thread (race-free check: assembled K bitwise/loosely equal
  across thread counts), measured speedup reported. Delivers: wall-clock at scale.

Each phase is independently shippable and independently gated; correctness is
locked by C1/C3 + the physics suite at every phase.

### 7.2 Risks

- **R1 — AMG robustness for elasticity near the incompressible limit / vector
  block structure.** 3D elasticity is a *system* PDE (3 DOF/node); scalar AMG can
  underperform if it ignores the near-null-space (rigid-body modes). Mitigations:
  prefer **smoothed aggregation** (better for systems) over classical RS; if
  iteration counts grow with N (failing 6.3), supply the **rigid-body-mode
  near-null-space** to smoothed aggregation (6 modes: 3 translations + 3
  rotations) — AlgebraicMultigrid.jl's SA accepts a near-null-space `B`. Trilinear
  Hex8 also locks near `ν→0.5` (`DESIGN.md` §0); for `ν=0.3` test cases this is
  not triggered. **Decision point:** if flat iterations fail with default SA,
  switch on rigid-body near-null-space before considering anything heavier.
- **R2 — AMG setup cost / preconditioner-reuse correctness.** Setup is the
  expensive AMG phase; rebuilding every Newton iteration would dominate. The reuse
  policy (Section 1.3) is **safe by construction** (a stale preconditioner cannot
  corrupt the answer — CG always uses the true K), so the only risk is *efficiency*
  (too-stale ⇒ more CG iters). Mitigation: the count-based refresh trigger (1.3).
  **Decision point:** if rebuild-every-step setup time dominates the sweep, enable
  the trigger and reuse across steps.
- **R3 — AMG hierarchy memory (the budget swing).** Operator complexity ~1.3–2.0×
  nnz (Section 5). If it lands at 2× and other items creep, headroom shrinks (still
  >25 GB free at the worst case). Mitigation: cap coarsening aggressiveness / use a
  more aggressive coarsening to shrink the hierarchy if needed; drop trial-`σ`
  (saves 1.4 GB) only if ever required.
- **R4 — Coloring overhead / load balance (Phase 3).** For the box, the 8-color
  scheme is analytic, balanced, and ~13.5 MB — minimal risk. For general meshes,
  greedy coloring quality varies; not a v1-box concern.
- **R5 — Int32 overflow.** Safe at 10M (`nnz,N < typemax(Int32)`); a larger run
  would need Int64. Guard: assert `nnz < typemax(Int32)` at pattern build and fall
  back to Int64 with a warning.

### 7.3 Where Route B (matrix-free / GPU) would later take over — NOT designed here

Beyond ~tens of millions of DOFs, or to use a GPU, the binding constraint flips
from *flops* to *memory bandwidth/footprint of storing K and the AMG hierarchy*.
The natural successor is **Route B: a matrix-free operator** — never assemble K;
implement `mul!(y, K, x)` by an on-the-fly element loop (the element kernel
already produces `Ke·ue` cheaply), preconditioned by matrix-free smoothing or a
geometric-multigrid / low-order auxiliary AMG. This eliminates the 10 GB K and the
7–13 GB AMG hierarchy and maps naturally to GPUs (the element loop is embarrassingly
parallel; the reference-element B of Section 2.2 makes it cheap). **We explicitly
do not design Route B here** — Route A (assembled CG+AMG) is sufficient for
10M-on-64GB and is the smaller, lower-risk step. Route B is the documented next
horizon when assembled storage becomes the wall.

---

## 8. Summary of changes by file (implementation map)

| File | Change | Phase |
|---|---|---|
| `Project.toml` | add `Krylov`, `AlgebraicMultigrid` deps | 1 |
| `Solver.jl` | replace `K \ -R` with PCG+AMG; inexact-Newton forcing term; preconditioner build/reuse policy; CG-stall + load-cut fallbacks | 1 |
| `Materials.jl`, element kernel | **unchanged** (reused) | — |
| `Elements.jl` | `ElementCache` → reference-element (uniform box) + on-the-fly fallback; **delete 31 GB per-element B cache** | 2 |
| `Assembly.jl` | `build_sparsity` → count-then-fill CSC from adjacency (**delete 47 GB COO**); `SparsityPattern` → drop `map` (**delete 15 GB**), Int32 indices; `_assemble!` → on-the-fly scatter search | 2 |
| `Model.jl` | wire new `ElementCache`/`SparsityPattern`; store coloring; persistent CG/AMG workspace | 2/3 |
| `BoundaryConditions.jl` | **unchanged** (symmetric elimination already CG/AMG-correct) | — |
| `Assembly.jl` / SpMV | 8-color threaded assembly; row-parallel threaded `mul!`; parallel smoother | 3 |
| `test/` | C1–C3, scaling sweep with trend assertions, allocation gates, extrapolated 10M budget | all |

The design keeps new structs minimal (`ElementCache` and `SparsityPattern` shrink
rather than grow), reuses the physics kernels verbatim, leaves the v1 small-problem
API and Dirichlet handling intact, and fits 10M DOFs in ~30–36 GB of a 64 GB node
with provably `O(N)` work per load step under CG+AMG.

---

## 9. Implementation revisions (driven by validation)

Two changes to §1–§5 were forced by measured behaviour during implementation/review
and are reflected in the code:

1. **Rigid-body near-null-space is mandatory, not optional (revises §4, R1).**
   Default smoothed-aggregation AMG *without* the near-null-space gives a CG
   iteration count growing ~`N^0.3` (measured 24→33→44→54 over N≈2.2k→28k) — i.e.
   `O(N^1.3)` flops, not `O(N)`. Supplying the 6 rigid-body modes (3 translations +
   3 rotations) as the SA near-null-space flattens it to ~`N^0.09` (measured
   8→9→10→11). The solver now *always* builds this near-null-space for `amg=:sa`
   (`_rigid_body_modes`), so flat-iteration `O(N)` is achieved by default. A test
   (`test_scaling.jl` "C2 mesh-independent CG count") asserts the log-log slope
   ≤ 0.15 and gates this.

2. **Default index type is `Int` (Int64), not `Int32` (revises §2.5, §5).**
   AlgebraicMultigrid's SA-with-near-null-space (and Ruge–Stüben) build Int64
   prolongation operators and cannot form an Int32 hierarchy. An Int32 K would
   therefore force a *retained* Int64 copy for AMG — `~10 GB (Int32 K) + ~13 GB
   (Int64 AMG copy)` — whereas one shared Int64 K is `~13.4 GB`. Int64 is thus both
   simpler and ~10 GB leaner once the (required) near-null-space is in play. The
   ~3.3 GB the Int32 micro-optimization saved is moot. Updated steady-state budget:
   **K ≈ 13.4 GB**, total **≈ 33–39 GB** at 10M — still well under 64 GB. `Int32`
   remains forceable via `build_sparsity(mesh; Ti=Int32)` for the `:direct`/`:rs`/
   Jacobi paths.

3. **`_is_uniform` made allocation-free (revises §2.2 build cost).** The uniformity
   check must not compute per-element B-matrices: doing so allocated ~87 kB/element
   (~295 GB transient at 10M, an OOM at `Model` construction). It now compares each
   element's node offsets against element 1's with scalar arithmetic — alloc-free
   and exact ("identical up to translation"), so build-time allocation is a small
   constant (~87 kB total).

### 9.1 Measured 10M budget (revises the §5 estimate)

`test/scaling_validation.jl` sweeps n=8…48 and extrapolates the **measured**
per-DOF constants to N=10.3M. All 14 validation gates pass; the measured budget
is heavier than the §5 estimate because the **6-mode rigid-body near-null-space
inflates the AMG hierarchy** (the §5 estimate assumed a scalar hierarchy):

| Line item | Measured @10M |
|---|---|
| K (nzval + rowval + colptr, Int64) | 12.0 GB |
| GaussState ×2 | 7.6 GB |
| **AMG hierarchy** (measured 2587 B/DOF, operator complexity 1.33) | **24.9 GB** ← binding |
| CG vectors, edofs, nullspace, mesh/misc | 2.1 GB |
| **TOTAL (steady-state)** | **46.7 GB** (< 64 GB; **17.3 GB headroom**) |

Pattern-build transient peak ≈ 12.7 GB (vs the 47 GB COO path that was deleted).

**Measured scaling (the optimality claims, confirmed):** CG-iters vs N slope
**+0.13** (flat ⇒ O(N) flops — *decisive*; the no-near-null-space control grows at
**+0.31**, proving the near-null-space is what buys O(N)); assembly time and
CG-time/iter both ~**+1.07** (O(N)); bytes/DOF tail slope **+0.026** (flat ⇒ linear
memory); nnz/N → 81; flops/DOF/Newton-iter ≈ **1.5×10⁴** (mesh-independent).

**Projected serial wall-time @10M ≈ 2500 s/load step (~42 min)**, of which AMG
*setup* (~740 s, superlinear ~N^1.36) is the long pole — amortized once per load
step by the reuse policy (§1.3) and the prime target for the threaded/parallel
coarsening follow-on. The Newton-loop portion (assembly + CG) is O(N) and is what
the §3 threading accelerates. Headroom and O(N) work both hold; AMG hierarchy
memory and setup cost are the levers if a future revision needs to push past 10M.
