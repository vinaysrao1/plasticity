# PlasticityFEM.jl

A small, fast, modular **3D finite element solver for computational
elastoplasticity** in Julia — in both **small-strain** and **finite-strain
(large-deformation)** kinematics, with an iterative solver that **scales to ~10
million degrees of freedom** on a single workstation. It is built to make it
*easy to build test models and change loading / boundary conditions*, while
keeping the numerical core clean and allocation-free.

| | |
|---|---|
| **Element** | 8-node hexahedron (Hex8), trilinear, 2×2×2 Gauss |
| **Material** | J2 / von Mises, rate-independent, **combined linear isotropic + kinematic hardening** |
| **Kinematics** | small strain *(default)* **and** finite strain (`F = Fᵉ·Fᵖ`, Hencky log-strain) + F-bar |
| **Integration** | radial-return mapping with the **consistent algorithmic tangent** (quadratic Newton) |
| **Global solve** | incremental Newton–Raphson; **direct** (small) or **CG + Algebraic Multigrid** (large), symmetry-aware auto-selection |
| **Scale** | validated to **~10M DOFs** with linear `O(N)` memory & flops; threaded assembly + SpMV |
| **Mesh** | structured box generator + predicate-based node/face selection |
| **BCs / loads** | Dirichlet (fixed & prescribed) and nodal/face forces, one line each |
| **Output** | dependency-free VTK (`.vtu`) export for ParaView/VisIt |

### Documentation map

| Doc | What it is |
|---|---|
| [`docs/EXPLAINER.md`](docs/EXPLAINER.md) | **Start here for intuition** — plasticity and the method in plain language |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Small-strain spec: governing equations, radial return, consistent tangent, conventions, verification plan |
| [`docs/FINITE_STRAIN.md`](docs/FINITE_STRAIN.md) | Finite-strain spec: multiplicative split, log-strain wrap, two-point tangent, F-bar, objectivity |
| [`docs/SCALING.md`](docs/SCALING.md) | Scaling to 10M DOFs: CG+AMG, memory layout, threading, the measured `O(N)` budget |

---

## Quick start (small strain)

```julia
using PlasticityFEM

# 1) Mesh — a 1×1×1 cube as a single Hex8 element (refine with box_mesh(...,n,n,n))
mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)

# 2) Material — steel-like, with linear isotropic hardening (consistent units: N, mm, MPa)
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)

# 3) Model
model = Model(mesh, steel)

# 4) Boundary conditions by face predicates — easy to read and change
fix!(model, on_face(mesh, :xmin), :x)    # roller on x = 0 face (fix x)
fix!(model, on_face(mesh, :ymin), :y)    # roller on y = 0 face (fix y)
fix!(model, on_face(mesh, :zmin), :z)    # roller on z = 0 face (fix z)

# 5) Loading — pull the x = L face to 1% nominal strain (displacement control)
prescribe!(model, on_face(mesh, :xmax), :x, 0.01)

# 6) Solve with load stepping
result = solve!(model; nsteps = 20, tol = 1e-8)

# 7) Postprocess
σ = gauss_stress(model)                  # 6 × ngp Voigt stresses [xx,yy,zz,xy,yz,zx]
println("σ_xx = ", σ[1, 1])              # ≈ 258.77 MPa (matches the analytical curve)
println("eq. plastic strain = ", maximum(equivalent_plastic_strain(model)))
```

This exact model is in [`examples/tension_cube.jl`](examples/tension_cube.jl).

---

## Building models, BCs, and loads

Everything is driven by **node selection predicates**, so changing the setup is a
one-line edit.

```julia
mesh = box_mesh(lx, ly, lz, nx, ny, nz)   # nx·ny·nz Hex8 elements in [0,lx]×[0,ly]×[0,lz]

# Select nodes…
on_face(mesh, :xmax)                       # nodes on a bounding-box face
                                           # :xmin/:xmax/:ymin/:ymax/:zmin/:zmax
select_nodes(mesh, (x,y,z) -> x ≈ 0 && z > 0.5)   # arbitrary coordinate predicate
```

| Call | Meaning |
|---|---|
| `Model(mesh, material; element=:small)` | build the problem; `element ∈ {:small, :finite, :finite_fbar}` |
| `fix!(model, nodes, comp=:all)` | homogeneous Dirichlet `u = 0` on `comp ∈ {:x,:y,:z,:all}` |
| `prescribe!(model, nodes, comp, value; ramp=true)` | inhomogeneous Dirichlet (prescribed displacement) |
| `load!(model, nodes, comp, value; distribute=false)` | nodal force; `distribute=true` splits a **total** face load across the nodes |
| `solve!(model; nsteps, tol=1e-8, maxiter=25, linsolve=:auto, threaded=…)` | load-stepped Newton; returns a `SolveResult` (per-step iterations, residual & CG-iteration histories) |
| `reset!(model)` | clear the solution and history (called automatically by `solve!`) |

**Postprocessing:** `nodal_displacements(model)` (3 × nnodes),
`gauss_stress(model)` (6 × ngp Cauchy stress), `gauss_kirchhoff(model)` (Kirchhoff
τ; finite strain), `gauss_strain(model)` (6 × ngp total strain),
`equivalent_plastic_strain(model)` (ngp), `von_mises(σ)` (scalar from a Voigt
6-vector).

`solve!` is idempotent — it resets the model on entry, so you can change loads or
BCs and re-solve the same `model` object safely. Loading is ramped over `nsteps`
load steps because plasticity is path dependent.

### Visualizing stress/strain distributions

Export a ParaView/VisIt file (dependency-free `.vtu` writer):

```julia
solve!(model; nsteps = 20)
write_vtu("result", model)          # -> result.vtu
```

The file carries per-node **Displacement** (use *Warp By Vector* for the deformed
shape — exact for finite strain) and per-element (Gauss-point-averaged) **Stress**
and **Strain** (Voigt tensors), **VonMises**, **MeanStress**, and
**EqPlasticStrain** — color by any of these in ParaView.

---

## Finite strain (large deformation)

A finite-strain J2 element family runs the *same* model under large-deformation
kinematics, selected by one keyword:

```julia
model = Model(mesh, steel)                        # small strain (default)
model = Model(mesh, steel; element = :finite)      # finite strain, standard F
model = Model(mesh, steel; element = :finite_fbar) # finite strain, F-bar
```

`fix!`, `prescribe!`, `load!`, `solve!`, `reset!`, and all postprocessing work
unchanged. The method (see [`docs/FINITE_STRAIN.md`](docs/FINITE_STRAIN.md)):

- **Multiplicative split** `F = Fᵉ·Fᵖ` with **Hencky (logarithmic)** hyperelasticity
  and an **exponential-map** plastic update (`det Fᵖ = 1` exactly). By the Simo
  (1992) equivalence, the stress update in elastic *log-strain* space is
  algebraically identical to the small-strain radial return — so the verified
  `return_map` kernel is **reused verbatim**, wrapped by a geometric pre-/post-processor.
- **Consistent tangent** in the two-point first-Piola form `K = ∫ Gᵀ (∂P/∂F) G dV`,
  containing both the material and geometric (initial-stress) parts, FD-verified
  (`∂P/∂F` to ~2e-9, the assembled element tangent to <1e-6, so Newton stays quadratic).
- **`:finite_fbar`** applies the **F-bar** volumetric correction (de Souza Neto et
  al. 1996) — the large-strain analogue of B-bar — relieving the volumetric
  locking of the trilinear Hex8 in the near-incompressible plastic limit.
- **Objectivity:** with kinematic hardening the back-stress is stored in the
  rotation-neutralized reference frame and pushed forward by the polar rotation of
  `F`, so the response is exactly frame-indifferent under large rotations.
- `gauss_stress` reports **Cauchy** stress `σ = τ/J`; `gauss_kirchhoff` returns the
  raw Kirchhoff τ.

**Tangent symmetry & the solver.** Small strain, and finite strain with
isotropic/perfect plasticity, give a **symmetric** tangent (CG + AMG). With
**kinematic** hardening (the objective back-stress rotation) or **F-bar** (the
centroid-`J₀` coupling) the tangent is genuinely **non-symmetric** — an inherent
property of those formulations, not an approximation. `solve!` is **symmetry-aware**:
`linsolve=:auto` (default) picks CG for the symmetric cases and the direct UMFPACK
solver for the non-symmetric ones (forcing `:cg` on a non-symmetric configuration
warns and overrides to `:direct`).

Examples: [`examples/finite_necking_bar.jl`](examples/finite_necking_bar.jl) (the
classic tensile-necking benchmark, with an imperfection to localize the neck) and
[`examples/finite_large_rotation_cantilever.jl`](examples/finite_large_rotation_cantilever.jl).

---

## Scaling to large problems

For models beyond ~10⁵–10⁶ DOFs the direct factorization in 3D becomes infeasible
(fill-in). `solve!` then uses a **preconditioned Conjugate Gradient** solver with
an **Algebraic Multigrid** preconditioner (smoothed aggregation seeded with the
rigid-body near-null-space), which gives a **mesh-independent CG iteration count**
and overall **`O(N)` work and memory**. See [`docs/SCALING.md`](docs/SCALING.md).

```julia
mesh  = box_mesh(10, 1, 1, 160, 16, 16)      # ~140k DOFs
model = Model(mesh, steel)
solve!(model; nsteps = 20)                    # CG+AMG chosen automatically for large N
```

```bash
# use threads for the 8-color parallel assembly + row-parallel sparse mat-vec:
julia -t auto --project=. examples/cantilever_80x8x8.jl
```

Highlights, all validated by a scaling sweep extrapolated to 10M:

- **No memory blow-up:** reference-element geometry cache, a CSC pattern built by
  count-then-fill from node adjacency (no giant COO transient), and no per-element
  scatter map. A 10M-DOF model fits in **~47 GB** on a 64 GB node.
- **`O(N)` flops:** the near-null-space keeps CG iterations flat in `N`
  (measured slope ≈ 0.13, vs ≈ 0.3 without it).
- **Threaded:** race-free 8-color assembly and a row-parallel symmetric SpMV,
  enabled automatically when Julia is started with `>1` thread.

`solve!` keywords: `linsolve ∈ {:auto, :cg, :direct}`, `amg ∈ {:sa, :rs}`,
`smoother ∈ {:gs, :jacobi}`, `threaded`, `cg_itmax`.

---

## Installation

Requires Julia ≥ 1.10. Dependencies: **StaticArrays**, **Krylov**,
**AlgebraicMultigrid** (plus the `SparseArrays` / `LinearAlgebra` standard
libraries).

```bash
git clone <this-repo>
cd experimental
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run an example or the test suite:

```bash
julia --project=. examples/tension_cube.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Conventions

- **Voigt ordering** (everywhere): `[xx, yy, zz, xy, yz, zx]`.
- **Strain** uses engineering shear (`γ_xy = 2 ε_xy`); **stress** uses physical
  shear components.
- **DOF numbering**: node-major, global DOF of (node `n`, component `c`) is
  `3(n−1)+c`, with `c = 1,2,3` ↦ `x,y,z`.
- **Units** are unit-agnostic; supply a consistent set (examples use N, mm, MPa).
- Sign convention: tension positive. Finite strain uses the Kirchhoff stress
  internally and reports Cauchy.

See [`docs/DESIGN.md` §9](docs/DESIGN.md) and [`docs/FINITE_STRAIN.md` §1](docs/FINITE_STRAIN.md).

---

## Project layout

```
src/
  PlasticityFEM.jl       top module + curated public API
  Materials.jl           J2Material, return_map (radial return + consistent tangent)
  Elements.jl            Hex8 shape functions, B-matrix, Gauss rule, cached geometry,
                         small- and finite-strain element force/tangent kernels
  FiniteStrain.jl        finite-strain kinematics, log-strain wrap, two-point ∂P/∂F
                         tangent, F-bar, objective hardening, element-kind dispatch
  Mesh.jl                Mesh, box_mesh, select_nodes / on_face, dof, GaussState
  BoundaryConditions.jl  Dirichlet/Neumann structs, symmetric BC imposition
  Assembly.jl            count-then-fill CSC pattern, on-the-fly scatter,
                         8-color threaded assembly, ElementKind dispatch
  Solver.jl              Newton–Raphson; direct & CG+AMG; symmetry-aware selection;
                         row-parallel SpMV; inexact-Newton; load stepping
  Model.jl               Model + high-level fix!/prescribe!/load!/reset! + postprocessing
  Visualization.jl       dependency-free VTK (.vtu) export for ParaView/VisIt
examples/                tension cube, cantilever (bending/torsion), cylinder torsion,
                         mesh-refinement sweep, finite-strain necking & large rotation
test/                    unit, system, hard-validation, scaling, and finite-strain tests
docs/                    EXPLAINER.md, DESIGN.md, FINITE_STRAIN.md, SCALING.md
```

---

## Performance

- **Type-stable, allocation-free hot kernels.** `return_map` and the small- and
  finite-strain element force/tangent kernels allocate **0 bytes** per call
  (verified by tests); element-level math uses `StaticArrays`.
- **O(1) assembly.** The CSC pattern is built once; numeric assembly scatters
  on-the-fly, so `assemble!` allocation is bounded and independent of mesh size.
- **`O(N)` at scale.** Mesh-independent CG iteration counts under AMG ⇒ linear
  flops and memory; threaded assembly and SpMV.
- **Consistent tangent ⇒ quadratic Newton** in both kinematics, FD-verified.

---

## Verification

The test suite (`julia --project=. -e 'using Pkg; Pkg.test()'`) covers:

- **Small strain** (`docs/DESIGN.md §8`): consistent tangent vs finite differences;
  uniaxial post-yield closed form; Bauschinger effect; patch tests; rigid-body
  modes; plastic incompressibility; reaction balance; cantilever vs beam theory;
  allocation gates.
- **Finite strain** (`docs/FINITE_STRAIN.md §7`, F1–F10): small-displacement limit
  matches small strain; two-point tangent vs FD (`<1e-6`); `det Fᵖ = 1`; objectivity
  under large rotation; tensile necking; F-bar locking relief; large-rotation
  cantilever; tangent symmetry where expected; 0-allocation kernels.
- **Scaling** (`docs/SCALING.md §6`): CG+AMG vs direct agreement; flat CG-iteration
  count (`O(N)` flops); linear memory; threaded == serial.

---

## Scope

Hex8 elements, J2 plasticity with linear (isotropic + kinematic) hardening,
quasi-static loading, in **both small-strain and finite-strain** kinematics,
scaling to ~10M DOFs. Intentionally **not** included: contact/dynamics, nonlinear
hardening laws, other element types, anisotropic plasticity, thermomechanics. See
`docs/DESIGN.md §0, §10` and `docs/FINITE_STRAIN.md §0`.
