# PlasticityFEM.jl — Design Document

A 3D finite element solver for **small-strain elastoplasticity** in Julia.

Status: design only (no implementation). This document is the specification that
implementers and validators rely on. It is intended to be self-contained and
mathematically precise.

---

## 0. Design philosophy & scope

**Goal.** A clean, modular, high-performance Julia package for 3D FEM analysis
of elastoplastic solids under the small-strain assumption. Easy to build test
models (meshes), and easy to change loading / boundary conditions.

**Deliberately tight scope (v1):**

| Concern            | Choice (v1)                                                          |
|--------------------|---------------------------------------------------------------------|
| Kinematics         | Small strain (infinitesimal); additive split ε = εᵉ + εᵖ            |
| Element            | **Hex8** (8-node trilinear hexahedron), 2×2×2 Gauss (8 points)       |
| Material           | **J2 / von Mises** rate-independent plasticity                       |
| Hardening          | Combined **linear isotropic** + **linear kinematic** (Ziegler/Prager)|
| Stress integration | **Radial return mapping** (closed form for linear hardening)         |
| Tangent            | **Consistent (algorithmic) tangent** for quadratic Newton           |
| Global solve       | Incremental-iterative **Newton–Raphson** with load stepping         |
| Linear algebra     | `SparseArrays`, COO triplets → `sparse()`, cached sparsity pattern   |
| Mesh               | Structured **box generator** `(nx,ny,nz)` + predicate node selection |
| BCs                | Dirichlet (displacement) + Neumann (nodal/face forces)              |

**Non-goals for v1 (explicitly out of scope; noted to bound the design):**
finite strain, contact, dynamics/inertia, nonlinear hardening laws (power-law /
Voce), tetrahedral or higher-order elements, parallel/distributed assembly,
adaptive mesh refinement, plane-stress/plane-strain reductions. The architecture
leaves clean extension points (a `Material` interface and an `Element`
interface) but we do **not** build the abstractions until a second model exists.

**Why Hex8 only (and not also a tet/simpler element).** A single, well-tested
element keeps the verification surface small. Hex8 is the workhorse for
structured box meshes, integrates cleanly with 2×2×2 Gauss, and is sufficient
for every validation case in this document. Trilinear Hex8 is mildly susceptible
to volumetric/shear locking in bending and near-incompressibility; we accept
this in v1 (it does not affect the J2 verification suite, which uses uniform
stress states or mesh-refinable bending), and we record selective reduced
integration / B-bar as a documented future extension rather than building it
speculatively. We do **not** add a second element type in v1 — it would double
the test matrix for no validation benefit.

---

## 1. Overview & governing equations

### 1.1 Strong form (quasi-static equilibrium)

Find the displacement field **u**(**x**) on domain Ω ⊂ ℝ³ such that

```
div σ + b = 0           in Ω        (balance of linear momentum, no inertia)
u = ū                   on Γ_D      (prescribed displacement)
σ · n = t̄               on Γ_N      (prescribed traction)
```

with the small-strain tensor

```
ε = ½ (∇u + ∇uᵀ).
```

### 1.2 Constitutive law (small-strain elastoplasticity)

Additive decomposition of strain:

```
ε = εᵉ + εᵖ.
```

Linear isotropic elasticity relates stress to **elastic** strain:

```
σ = ℂ : εᵉ = ℂ : (ε − εᵖ),
```

where ℂ is the fourth-order isotropic elasticity tensor (parameters E, ν, or
equivalently bulk modulus K and shear modulus G). The evolution of εᵖ and the
internal hardening variables is governed by the J2 flow rule (Section 2).

### 1.3 Weak form (principle of virtual work)

For admissible virtual displacements δ**u** (δu = 0 on Γ_D):

```
∫_Ω  ε(δu) : σ  dV  =  ∫_Ω δu · b dV  +  ∫_{Γ_N} δu · t̄ dA.
```

Define internal and external virtual work:

```
δW_int(u; δu) = ∫_Ω ε(δu) : σ(u) dV,
δW_ext(δu)    = ∫_Ω δu · b dV + ∫_{Γ_N} δu · t̄ dA.
```

Equilibrium ⇔ δW_int = δW_ext for all admissible δu.

### 1.4 Discretization (FEM)

Displacement is interpolated within each element by trilinear shape functions
(Section 3). Collecting nodal displacements into the global vector **U** ∈ ℝ^{ndof}
(ndof = 3·nnodes), the discrete equilibrium is the **residual** system

```
R(U) = F_int(U) − F_ext = 0,
```

where

```
F_int(U) = ⋃_e ∫_{Ω_e} Bᵀ σ dV        (assembled element internal forces),
F_ext    = ⋃_e ∫_{Ω_e} Nᵀ b dV + ⋃_e ∫_{Γ_N^e} Nᵀ t̄ dA + F_nodal.
```

`B` is the strain-displacement matrix (6×24 per element, Section 3.4) and σ is
the Voigt stress 6-vector.

### 1.5 Linearization (Newton–Raphson)

Because σ depends nonlinearly on ε through the return mapping, R(U) is nonlinear.
Newton's method: given iterate Uᵏ, solve for the correction δU,

```
K_T(Uᵏ) δU = −R(Uᵏ),     Uᵏ⁺¹ = Uᵏ + δU,
```

with the **consistent (algorithmic) tangent stiffness**

```
K_T = ∂R/∂U = ⋃_e ∫_{Ω_e} Bᵀ Dᵃˡᵍ B dV,
```

where Dᵃˡᵍ (6×6, Voigt) is the consistent material tangent ∂σ/∂ε produced by the
return-mapping algorithm (Section 2.5). Using Dᵃˡᵍ (not the continuum elastoplastic
tangent) is what yields **quadratic** asymptotic convergence of the global Newton
iteration.

For **load stepping**, F_ext is ramped over load steps `n = 1…N` via a scalar
factor λ_n ∈ (0,1]; at each step we Newton-iterate to equilibrium, then commit
(history variables advance). This is required because plasticity is path
dependent: the converged state of step n is the initial state of step n+1.

---

## 2. J2 radial-return algorithm (the core)

This section is written out fully. It is the single most important part of the
document for implementers and validators.

### 2.1 Conventions and material parameters

- **Voigt ordering (fixed throughout):** `[xx, yy, zz, xy, yz, zx]`.
  Index map: 1→xx, 2→yy, 3→zz, 4→xy, 5→yz, 6→zx.
- Strains in Voigt use **engineering shear**: γ_xy = 2 ε_xy, etc.
  So ε_voigt = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx].
- Stresses in Voigt are the physical components: σ_voigt = [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].
- Elastic moduli:
  ```
  G = E / (2(1+ν))            (shear modulus)
  K = E / (3(1−2ν))           (bulk modulus)
  λ = E ν / ((1+ν)(1−2ν))     (Lamé first parameter)
  ```
- Hardening:
  - Isotropic: yield radius grows as `σ_y(ᾱ) = σ_y0 + H_iso · ᾱ`,
    where ᾱ ≥ 0 is the accumulated (equivalent) plastic strain and
    H_iso ≥ 0 is the linear isotropic hardening modulus.
  - Kinematic: back stress **β** evolves linearly (Prager/linear Ziegler),
    `β̇ = (2/3) H_kin · ε̇ᵖ`, H_kin ≥ 0 the kinematic hardening modulus.
- Sign convention: tension positive; pressure p = −(1/3) tr σ such that
  σ = s − p·I with deviator s.

### 2.2 Elastic stress–strain in Voigt

The isotropic elasticity matrix `ℂ` (6×6) in the above Voigt ordering:

```
        | λ+2G   λ     λ    0   0   0 |
        |  λ   λ+2G    λ    0   0   0 |
ℂ  =    |  λ     λ   λ+2G   0   0   0 |
        |  0     0     0    G   0   0 |
        |  0     0     0    0   G   0 |
        |  0     0     0    0   0   G |
```

Note the shear diagonal entries are `G` (not 2G), because engineering shear
strain γ already carries the factor 2. This is the standard, self-consistent
Voigt convention with `σ = ℂ ε_voigt`.

Equivalent split into volumetric + deviatoric using K and G is used inside the
return map and the consistent tangent.

### 2.3 Trial (elastic predictor) state

Given at a Gauss point the committed state of the previous (converged) load step
`{εᵖₙ, βₙ, ᾱₙ}` and the current total strain εₙ₊₁ (from the current displacement
iterate), compute the **trial elastic** state assuming no plastic flow in the
step:

```
εᵉ,trial = εₙ₊₁ − εᵖₙ
σ_trial  = ℂ : εᵉ,trial                         (Voigt: σ_trial = ℂ ε_voigt^{e,trial})
```

Deviatoric trial stress and relative (shifted) stress:

```
p_trial = (1/3) (σ_trial,xx + σ_trial,yy + σ_trial,zz)      (mean stress)
s_trial = dev(σ_trial) = σ_trial − p_trial · [1,1,1,0,0,0]ᵀ
ξ_trial = s_trial − βₙ                          (relative stress; β is deviatoric)
```

Von Mises equivalent of the relative stress:

```
‖ξ_trial‖   = sqrt( ξ_xx² + ξ_yy² + ξ_zz² + 2(ξ_xy² + ξ_yz² + ξ_zx²) )
q_trial     = sqrt(3/2) · ‖ξ_trial‖            (von Mises effective relative stress)
```

> Implementation note: the factor 2 on the shear terms in `‖ξ‖²` is the correct
> tensor (Frobenius) norm of a symmetric deviator written in Voigt with physical
> (not engineering) shear stress components. Because ξ is a **stress** deviator,
> its shear components are physical, so the double-contraction ξ:ξ =
> Σ ξ_ij ξ_ij = ξ_xx²+ξ_yy²+ξ_zz² + 2ξ_xy²+2ξ_yz²+2ξ_zx².

### 2.4 Yield check and plastic corrector (closed-form for linear hardening)

Yield function (von Mises with combined hardening):

```
f_trial = q_trial − ( σ_y0 + H_iso · ᾱₙ )
```

- If `f_trial ≤ 0`: **elastic step.** Accept the trial state:
  ```
  σₙ₊₁ = σ_trial,  εᵖₙ₊₁ = εᵖₙ,  βₙ₊₁ = βₙ,  ᾱₙ₊₁ = ᾱₙ,
  Dᵃˡᵍ = ℂ  (elastic tangent).
  ```
- If `f_trial > 0`: **plastic step.** Perform the radial return.

**Flow direction** (unit deviatoric normal, constant during the return for J2):

```
n̂ = ξ_trial / ‖ξ_trial‖      (6-vector; physical shear components)
```

**Plastic multiplier.** The consistency condition for the updated state with the
linear hardening laws gives a *linear* equation in the increment of accumulated
plastic strain Δγ (= Δᾱ). Using the standard result

```
q_trial − 3G·Δγ − H_kin·Δγ − (σ_y0 + H_iso·(ᾱₙ + Δγ)) = 0
```

solved in closed form:

```
            q_trial − (σ_y0 + H_iso · ᾱₙ)            f_trial
   Δγ  =  ─────────────────────────────────  =  ─────────────────────.
                3G + H_iso + H_kin                3G + H_iso + H_kin
```

(Δγ ≥ 0 is guaranteed because we are in the branch f_trial > 0 and the
denominator is positive. For J2, Δγ equals the increment of equivalent plastic
strain Δᾱ.)

**State update.** Updated relative-stress magnitude and stress:

```
σₙ₊₁ = σ_trial − 2G · Δγ · n̂                                    (Voigt, deviatoric correction only;
                                                                  mean stress p unchanged → plastic incompressibility)
```

Plastic strain increment (engineering-shear Voigt; note factor 2 on shear so the
*tensorial* plastic strain matches the flow rule Δεᵖ = Δγ · n̂):

```
Δεᵖ_voigt = Δγ · [ n̂_xx, n̂_yy, n̂_zz, 2 n̂_xy, 2 n̂_yz, 2 n̂_zx ]ᵀ
εᵖₙ₊₁     = εᵖₙ + Δεᵖ_voigt
```

> Consistency check: εᵉ = ε − εᵖ, and σ = ℂ εᵉ. After updating εᵖ, recomputing
> σ via ℂ(ε − εᵖ) must reproduce σₙ₊₁ above (good unit test on the kernel).

Back stress update (kinematic hardening, deviatoric):

```
βₙ₊₁ = βₙ + (2/3) H_kin · Δγ · n̂            (stored as a 6-vector deviator, physical shears)
```

Accumulated plastic strain:

```
ᾱₙ₊₁ = ᾱₙ + Δγ
```

### 2.5 Consistent (algorithmic) tangent — 6×6 Voigt

The radial return for linear J2 admits a closed-form algorithmic tangent. Define
the deviatoric projector `P_dev` (6×6) and the rank-1 outer product on n̂. With

```
q_trial = sqrt(3/2)‖ξ_trial‖,
θ   = 1 − (3G · Δγ) / q_trial,                       (0 < θ ≤ 1)
θ̄   = 1 / (1 + (H_iso + H_kin)/(3G)) − (1 − θ)
    = (3G)/(3G + H_iso + H_kin) − (1 − θ),
```

the consistent tangent (elastic when not yielding) is

```
Dᵃˡᵍ = K · (1⊗1) + 2G·θ · (P_dev − (1/3) 1⊗1 corrected)  − 2G·θ̄ · (n̂ ⊗ n̂),
```

written explicitly in Voigt as the standard Simo–Hughes form:

```
Dᵃˡᵍ = ℂ − (2G)·(2G·Δγ / q_trial) · ( I_dev − n̂⊗n̂ ) − (2G) · ( (2G)/(3G+H_iso+H_kin) − (2G·Δγ/q_trial) ) · (n̂⊗n̂)
```

To avoid ambiguity, the **implementation-ready** componentwise recipe is:

```
Let  β0 = 2G·Δγ / q_trial             (note q_trial uses sqrt(3/2)‖ξ_trial‖)
     γ0 = (2G) / (3G + H_iso + H_kin)
     β1 = 2G · (1 − β0)               (effective shear-ish multiplier on I_dev part)
     β2 = 2G · (β0 − γ0)              (correction along n̂⊗n̂)

Dᵃˡᵍ =  K · (𝟙 ⊗ 𝟙)                       (volumetric part, 𝟙 = [1,1,1,0,0,0]ᵀ)
      + β1 · I_dev_voigt                    (deviatoric isotropic part)
      + β2 · (n̂ ⊗ n̂),                       (radial correction)
```

where:

- `𝟙 ⊗ 𝟙` is the 6×6 matrix with the upper-left 3×3 block all ones, zero
  elsewhere.
- `I_dev_voigt` is the deviatoric part of the symmetric 4th-order identity in
  Voigt. Concretely, the symmetric identity `I_sym_voigt = diag(1,1,1,½,½,½)`
  (the ½ on shear rows accounts for engineering shear), and
  `I_dev_voigt = I_sym_voigt − (1/3)(𝟙 ⊗ 𝟙)`.
- `n̂ ⊗ n̂` uses the physical-shear n̂ (6-vector) directly as an outer product.

> Verification of the tangent: in the limit Δγ → 0 (just at first yield) and with
> the elastic branch, Dᵃˡᵍ → ℂ. A finite-difference check `Dᵃˡᵍ ≈ ∂σ/∂ε`
> (perturb each of the 6 strain components by ε ~ 1e-7, recompute σ via the full
> return map, form the difference quotient) must match to ~1e-5 relative. This
> FD check is a **required** unit test (Section 8).

> Symmetry: Dᵃˡᵍ is symmetric for associative J2 with these hardening laws. The
> assembled K_T is therefore symmetric; we may exploit this in the linear solve
> (e.g. `cholesky` when SPD, else `ldlt`/`lu`). v1 uses `lu` for robustness and
> notes Cholesky as an optimization.

### 2.6 Return-map kernel interface (pure function, allocation-free)

The kernel operates on `SVector`/`SMatrix` and returns updated state + tangent
with **zero heap allocation**:

```
(σ_new, εp_new, β_new, ᾱ_new, D_alg) =
    return_map(mat, ε_total::SVector{6}, εp_n::SVector{6}, β_n::SVector{6}, ᾱ_n::Float64)
```

All intermediate quantities (`σ_trial`, `s_trial`, `ξ_trial`, `n̂`, …) are
`SVector{6,Float64}`; `D_alg` is `SMatrix{6,6,Float64,36}`. This is the hot
kernel; it must pass an `@allocated == 0` test.

---

## 3. Hex8 element

### 3.1 Node ordering and reference element

Reference (natural) coordinates ξ = (ξ, η, ζ) ∈ [−1, 1]³. Standard hexahedron
node ordering (matches typical FEM / VTK_HEXAHEDRON convention):

```
node : (ξ,    η,    ζ)
 1   : (−1, −1, −1)
 2   : (+1, −1, −1)
 3   : (+1, +1, −1)
 4   : (−1, +1, −1)
 5   : (−1, −1, +1)
 6   : (+1, −1, +1)
 7   : (+1, +1, +1)
 8   : (−1, +1, +1)
```

Local DOF ordering per element: node-major, `[u1x,u1y,u1z, u2x,u2y,u2z, …, u8x,u8y,u8z]`
→ 24 DOFs.

### 3.2 Shape functions (trilinear)

```
N_a(ξ,η,ζ) = (1/8)(1 + ξ_a ξ)(1 + η_a η)(1 + ζ_a ζ),     a = 1…8,
```

with (ξ_a, η_a, ζ_a) the natural coordinates of node a above. Derivatives:

```
∂N_a/∂ξ = (1/8) ξ_a (1 + η_a η)(1 + ζ_a ζ)
∂N_a/∂η = (1/8) η_a (1 + ξ_a ξ)(1 + ζ_a ζ)
∂N_a/∂ζ = (1/8) ζ_a (1 + ξ_a ξ)(1 + η_a η)
```

Stored as `SVector{8}` (N) and `SMatrix{8,3}` (dN/dξ).

### 3.3 Isoparametric mapping and Jacobian

Element node coordinates `X_e ∈ SMatrix{8,3}`. Geometry interpolation:

```
x(ξ) = Σ_a N_a(ξ) X_a.
```

Jacobian (3×3):

```
J = (dN/dξ)ᵀ · X_e            (J_ij = Σ_a ∂N_a/∂ξ_i  X_a,j),    J ∈ SMatrix{3,3}.
detJ = det(J)                 (must be > 0; checked once at mesh build for the box mesh)
```

Spatial shape-function gradients (8×3):

```
dN/dx = (dN/dξ) · J⁻¹          (∂N_a/∂x_j),     SMatrix{8,3}.
```

### 3.4 B-matrix (6×24)

For node a, the 6×3 block (Voigt ordering `[xx,yy,zz,xy,yz,zx]`,
engineering shear):

```
        | N_a,x    0      0   |
        |  0     N_a,y    0   |
B_a =   |  0      0     N_a,z |
        | N_a,y  N_a,x    0   |
        |  0     N_a,z   N_a,y|
        | N_a,z   0     N_a,x |
```

`B = [B_1 B_2 … B_8] ∈ SMatrix{6,24}`. Strain at a Gauss point:
`ε_voigt = B · u_e`, with `u_e ∈ SVector{24}`.

### 3.5 Gauss quadrature (2×2×2)

8 points, each coordinate ∈ {−1/√3, +1/√3}, weight = 1 each (product of 1D
weights all equal to 1). Tensor-product point list and weights stored as compile-
time constants (`SVector`s). Integration of any field g:

```
∫_{Ω_e} g dV ≈ Σ_{gp=1}^{8} g(ξ_gp) · detJ(ξ_gp) · w_gp.
```

### 3.6 Element internal force and tangent

For element e, looping the 8 Gauss points:

```
F_int^e = Σ_gp  Bᵀ(gp) · σ(gp) · detJ(gp) · w_gp            ∈ SVector{24}
K_T^e   = Σ_gp  Bᵀ(gp) · Dᵃˡᵍ(gp) · B(gp) · detJ(gp) · w_gp  ∈ SMatrix{24,24}
```

where σ(gp) and Dᵃˡᵍ(gp) come from `return_map` evaluated with the strain
`ε = B(gp) u_e` and that Gauss point's committed history. The element routine is
allocation-free (all `SMatrix`/`SVector`). It returns both F_int^e and K_T^e (and,
on a "commit" pass, writes back updated per-GP state).

> Note: B and detJ depend only on geometry, not on U. For a structured box mesh
> all elements share the same shape (up to translation), so B and detJ at each GP
> can be **precomputed once per element** (or once globally if the mesh is
> uniform) and cached. v1 caches per-element `B` and `detJ·w` arrays to keep the
> hot Newton loop free of Jacobian recomputation. (For a uniform box this is even
> a single cached set — a documented optimization.)

---

## 4. Data structures

All structs are concrete-typed (no abstract fields), immutable where possible,
with `Float64` and `Int` element types. State that mutates across load steps is
held in mutable container arrays, not in immutable structs, to preserve type
stability.

### 4.1 Material

```julia
struct J2Material
    E::Float64        # Young's modulus
    ν::Float64        # Poisson ratio
    σy0::Float64      # initial yield stress
    Hiso::Float64     # linear isotropic hardening modulus (≥ 0)
    Hkin::Float64     # linear kinematic hardening modulus (≥ 0)
    # derived (precomputed in constructor for the hot loop):
    G::Float64        # shear modulus
    K::Float64        # bulk modulus
    λ::Float64        # Lamé λ
    Cmat::SMatrix{6,6,Float64,36}   # elastic Voigt matrix ℂ
end
```

A constructor `J2Material(; E, ν, σy0, Hiso=0.0, Hkin=0.0)` computes G, K, λ, Cmat.

### 4.2 Mesh

```julia
struct Mesh
    nodes::Matrix{Float64}       # 3 × nnodes  (column = node coords)  [contiguous, cache-friendly]
    elements::Matrix{Int}        # 8 × nelem   (column = node ids of an element, Hex8 ordering)
    nnodes::Int
    nelem::Int
end
```

(We store `nodes` as `3 × nnodes` and `elements` as `8 × nelem` so each
element's/node's data is a contiguous column — fast `@view` access and good
cache behavior in the assembly loop.)

Box generator:

```julia
"""
    box_mesh(lx, ly, lz, nx, ny, nz) -> Mesh

Structured grid of nx×ny×nz Hex8 elements filling [0,lx]×[0,ly]×[0,lz].
Node count = (nx+1)(ny+1)(nz+1). Lexicographic node numbering (x fastest).
"""
box_mesh(lx, ly, lz, nx, ny, nz)
```

Node/face selection helpers (predicate-based, return `Vector{Int}` of node ids):

```julia
select_nodes(mesh, pred)            # pred(x,y,z)::Bool
on_face(mesh, :xmin)                # convenience: :xmin,:xmax,:ymin,:ymax,:zmin,:zmax (uses bounding box + tol)
select_faces(mesh, :zmax)          # returns list of (elem, local_face) for Neumann surface integrals
```

`on_face` uses a coordinate tolerance derived from the mesh bounding box
(`tol = 1e-8 * characteristic_length`) so floating-point coordinates select
cleanly.

### 4.3 Per-Gauss-point state (struct of arrays, flat & contiguous)

State is stored in a **struct-of-arrays** layout sized `[6 (or 1) , ngp_total]`,
where `ngp_total = nelem × 8`. This is contiguous, allocation-friendly, and
SoA-vectorizable.

```julia
struct GaussState
    εp::Matrix{Float64}     # 6 × ngp_total  (plastic strain, engineering-shear Voigt)
    β::Matrix{Float64}      # 6 × ngp_total  (back stress deviator, physical shear)
    ᾱ::Vector{Float64}      # ngp_total      (accumulated plastic strain)
    σ::Matrix{Float64}      # 6 × ngp_total  (stress, for output/postprocessing)
end
```

Two copies are maintained: `committed` (state at the last converged load step)
and `trial`/working (updated during Newton iterations; copied into committed only
when a load step converges). Indexing for element e, local gp g:
`idx = (e-1)*8 + g`.

### 4.4 Cached element geometry

```julia
struct ElementCache
    B::Array{Float64,3}        # 6 × 24 × 8 per element? -> stored as Vector over elements
    detJw::Matrix{Float64}     # 8 × nelem   (detJ * w at each gp)
    # In practice: Vector{NTuple{8, SMatrix{6,24,Float64,144}}} for B, and
    #              Vector{SVector{8,Float64}} for detJw, to keep StaticArray types.
end
```

(Implementation detail: store B as a `Vector{SVector{8, SMatrix{6,24,Float64,144}}}`
or similar StaticArray-of-StaticArrays so element math stays allocation-free. The
exact container is an implementation choice; the contract is "B and detJ·w for
every (element, gp) are precomputed once and reused every Newton iteration.")

### 4.5 Boundary conditions and DOFs

```julia
struct DirichletBC
    dofs::Vector{Int}        # constrained global DOF indices
    values::Vector{Float64}  # prescribed values (per load step these are scaled by λ if `ramp=true`)
    ramp::Bool               # ramp with load factor (true) or apply fully at step 1 (false)
end

struct NeumannBC
    dofs::Vector{Int}        # loaded global DOF indices (nodal forces)
    values::Vector{Float64}  # force magnitudes (ramped by λ)
end
```

Global DOF numbering: DOF for node `n`, component `c∈{1,2,3}` is
`3*(n-1) + c`. Helper `dof(node, comp)` and vectorized variants are provided.

### 4.6 Model (the assembled problem)

```julia
mutable struct Model
    mesh::Mesh
    material::J2Material            # v1: single material; extensible to per-element later
    cache::ElementCache
    dirichlet::DirichletBC
    neumann::NeumannBC
    state_committed::GaussState
    state_trial::GaussState
    # assembly scratch (preallocated):
    sparsity::SparsityPattern       # cached COO → CSC mapping (see §5/§7)
    Kbuf::SparseMatrixCSC{Float64,Int}
    Rbuf::Vector{Float64}
    U::Vector{Float64}              # current displacement solution
end
```

`Model` is `mutable` only so `U` and the buffers can be reassigned; the heavy
arrays are preallocated once.

---

## 5. Module / file breakdown

Small number of well-factored modules. One top module includes the submodules.

```
Project.toml
src/
  PlasticityFEM.jl        # top module; `using`/`include` submodules; re-exports public API
  Materials.jl            # J2Material, return_map (radial return + consistent tangent)
  Elements.jl             # Hex8 shape funcs, Jacobian, B-matrix, Gauss quadrature,
                          #   element_internal_force_and_tangent!, geometry caching
  Mesh.jl                 # Mesh struct, box_mesh, select_nodes/on_face/select_faces
  BoundaryConditions.jl   # DirichletBC, NeumannBC, dof helpers, apply!/impose!
  Assembly.jl             # SparsityPattern, build pattern, assemble_K_and_R!
  Solver.jl               # Newton–Raphson, load stepping, convergence, line-search hook (off by default)
  Model.jl                # Model struct + high-level driver API (add_bc!, add_load!, solve!)
test/
  runtests.jl
  test_material.jl        # return-map unit tests, FD tangent check, allocation test
  test_element.jl         # patch test, rigid body, B-matrix, Jacobian
  test_assembly.jl        # symmetry, sparsity, assembly allocation
  test_solver.jl          # uniaxial tension, cantilever, Newton convergence rate
  test_allocations.jl     # @allocated on hot kernels
examples/
  tension_cube.jl         # single-/multi-element uniaxial tension into plastic regime
  cantilever.jl           # elastic + elastoplastic cantilever
docs/
  DESIGN.md               # (this file)
```

### 5.1 Module responsibilities & public API

- **Materials** — `J2Material(; …)`, `return_map(mat, ε, εp_n, β_n, ᾱ_n)`.
  Pure, allocation-free constitutive kernel. No FEM knowledge.
- **Elements** — `hex8_shape(ξ)`, `hex8_dshape(ξ)`, `jacobian(Xe, dN)`,
  `bmatrix(dNdx)`, `GAUSS_PTS`, `GAUSS_WTS`, `precompute_cache(mesh)`,
  `element_force_tangent!(...)`. No global/assembly knowledge.
- **Mesh** — `box_mesh`, `select_nodes`, `on_face`, `select_faces`, `dof`.
- **BoundaryConditions** — builders for `DirichletBC`/`NeumannBC` from node lists;
  `impose_dirichlet!(K, R, bc, λ)` (symmetric elimination / penalty-free row–col
  zeroing with 1 on diagonal), `assemble_neumann!(F_ext, bc, λ)`.
- **Assembly** — `build_sparsity(mesh)` (COO pattern from element connectivity,
  cached index map), `assemble!(model, U)` filling `Kbuf`, `Rbuf` from element
  contributions + per-GP return maps.
- **Solver** — `newton!(model; tol, maxiter)` for one load step; `solve!(model;
  nsteps, …)` driving the load ramp; returns convergence history.
- **Model** — high-level UX (Section 6): `Model(mesh, material)`,
  `fix!`, `prescribe!`, `load!`, `solve!`.

**Public exports (curated, small):**
`box_mesh, on_face, select_nodes, J2Material, Model, fix!, prescribe!, load!,
solve!, nodal_displacements, gauss_stress`.

---

## 6. "Easy model building" UX (target API)

The driving requirement is that building a model and changing BCs/loads is a few
readable lines. Target example code (this is the API we commit to implementing):

### 6.1 Uniaxial tension cube into the plastic regime

```julia
using PlasticityFEM

# 1) Mesh: a 1×1×1 cube, single element (or refine with (n,n,n))
mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)

# 2) Material: steel-like, with isotropic hardening
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)   # MPa, mm units

# 3) Model
model = Model(mesh, steel)

# 4) Boundary conditions by face predicates — easy to read & change
fix!(model, on_face(mesh, :xmin), :x)     # roller: x=0 face fixed in x
fix!(model, on_face(mesh, :ymin), :y)     # roller: y=0 face fixed in y
fix!(model, on_face(mesh, :zmin), :z)     # roller: z=0 face fixed in z

# 5) Loading: prescribe displacement on x=lx face (strain control), ramped
prescribe!(model, on_face(mesh, :xmax), :x, 0.01)   # pull to 1% nominal strain

# 6) Solve with load stepping
result = solve!(model; nsteps = 20, tol = 1e-8, maxiter = 25)

# 7) Postprocess
σ = gauss_stress(model)             # 6 × ngp stresses
@show σ[1, 1]                       # σ_xx at first Gauss point
```

### 6.2 Force-controlled cantilever

```julia
mesh   = box_mesh(10.0, 1.0, 1.0, 40, 4, 4)        # slender beam
mat    = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hkin = 2000.0)
model  = Model(mesh, mat)

fix!(model, on_face(mesh, :xmin))                  # clamp the x=0 face (all 3 components)

# distributed tip load as nodal forces on the x=lx face, total Fz = −P
load!(model, on_face(mesh, :xmax), :z, -100.0; distribute = true)

result = solve!(model; nsteps = 10)
δtip   = maximum(abs, nodal_displacements(model)[3, :])   # tip deflection (z)
```

### 6.3 API surface summary

- `fix!(model, nodes, comp=:all)` — homogeneous Dirichlet (u = 0) on selected
  nodes / component(s). `comp ∈ {:x,:y,:z,:all}`.
- `prescribe!(model, nodes, comp, value; ramp=true)` — inhomogeneous Dirichlet.
- `load!(model, nodes, comp, value; distribute=false, ramp=true)` — nodal force;
  `distribute=true` splits `value` equally across the selected nodes (so the
  user specifies a *total* face load).
- `solve!(model; nsteps, tol=1e-8, maxiter=25, verbose=false)` — load-stepped
  Newton; returns a `SolveResult` with per-step iteration counts and residual
  histories (used by tests to assert convergence rate).
- Postprocessing: `nodal_displacements(model)` (3 × nnodes),
  `gauss_stress(model)`, `equivalent_plastic_strain(model)`.

This keeps "build a model / change BCs / change loads" to single-line calls with
coordinate/face predicates — the stated top priority.

---

## 7. Performance plan

Principles, applied concretely:

1. **Type stability everywhere.** Concrete struct fields (`Float64`, `Int`,
   `SMatrix`/`SVector` with full type params). No abstract-typed fields, no
   `Any`, no non-const globals. `@code_warntype` clean on `return_map`,
   `element_force_tangent!`, and `assemble!`.

2. **StaticArrays in the hot path.** Element math (N, dN, J, J⁻¹, B 6×24,
   Dᵃˡᵍ 6×6, σ/ε 6-vectors, K_T^e 24×24, F_int^e 24) uses `SMatrix`/`SVector`.
   These live on the stack → zero heap allocation per Gauss point.

3. **Preallocation & cached sparsity.**
   - Build the global sparsity pattern **once** from element connectivity:
     gather all (row, col) index pairs from element DOF maps, build the CSC
     structure, and cache a mapping from each element's 24×24 local entry to its
     position in the CSC `nzval` array. During every Newton iteration we
     `fill!(Kbuf.nzval, 0)` and scatter element contributions directly into
     `nzval` (no `sparse()` rebuild per iteration, no COO re-sort).
   - `Rbuf`, element scratch, and per-GP state arrays preallocated.

4. **Factorization reuse.** The sparsity pattern is constant across iterations and
   load steps, so the symbolic factorization can be reused (`lu` with
   `klu`/UMFPACK symbolic reuse, or refactor numerically each iteration). v1:
   numeric `lu` each iteration on the cached pattern; symbolic-reuse noted as an
   optimization. K_T is symmetric → Cholesky/LDLᵀ possible (optimization).

5. **SoA state, flat arrays.** `GaussState` columns are contiguous; commit is a
   single `copyto!`. No per-GP heap objects.

6. **Allocation-free hot loops.** The kernels `return_map` and
   `element_force_tangent!` allocate **zero** bytes; the per-iteration
   `assemble!` allocates O(1) (not O(nelem)). Enforced by tests.

### 7.1 What the tests must measure (performance)

- `@allocated return_map(...) == 0` after warmup.
- `@allocated element_force_tangent!(...) == 0` after warmup.
- `@allocated assemble!(model, U)` is bounded by a small constant (independent of
  `nelem`); assert `< C` bytes.
- Assembly throughput: time per Newton iteration scales ~linearly in `nelem`
  (sanity benchmark, not a hard gate).
- Memory footprint of `Model` scales linearly in `nelem` and `nnz(K)` as
  expected; no quadratic blowup.
- BenchmarkTools `@btime` recorded for: one return map, one element tangent, one
  global assemble, one Newton iteration on a 10×10×10 mesh (regression
  baselines, stored in test output, not asserted as absolute times).

---

## 8. Verification & validation test plan

Each test lists the setup and the **analytical expected value** so the validator
can assert. Tolerances given are suggestions.

### 8.1 Material kernel (unit tests)

**T1 — Elastic round-trip.** Random strain below yield → `return_map` must give
`σ = ℂ ε`, `εp` unchanged, `D_alg == ℂ`, `Δγ = 0`. Assert rel. error ≤ 1e-12.

**T2 — Uniaxial tension, analytical stress–strain (isotropic hardening).**
Drive a single Gauss point in uniaxial stress (impose ε_xx, hold the lateral
stresses to zero by the cube BCs in the element test). For uniaxial monotonic
loading with linear isotropic hardening modulus H_iso the post-yield response is
```
σ = σy0 + H_t · (ε − ε_y),    ε_y = σy0 / E,
H_t = E·H_iso / (E + H_iso)          (tangent modulus in uniaxial σ–ε space).
```
Hence at total strain ε beyond ε_y, expected axial stress
```
σ_xx = σy0 + (E·H_iso/(E+H_iso)) · (ε − σy0/E).
```
Example: E=210000, σy0=250, H_iso=1000 ⇒ ε_y=250/210000≈1.190476e-3,
H_t=210000·1000/211000≈995.26. At ε=0.01:
σ_xx ≈ 250 + 995.26·(0.01−1.190476e-3) ≈ 250 + 995.26·8.8095e-3 ≈ **258.77** (MPa).
Assert rel. error ≤ 1e-6 against the closed form. (Note: the **uniaxial** tangent
H_t differs from the 3D H_iso; the test uses H_t for the analytical line.)

**T3 — Perfect plasticity cap.** H_iso=H_kin=0: under increasing uniaxial strain,
σ_xx must asymptote to σy0 and never exceed it (within FP). Assert
σ_xx ≤ σy0(1+1e-10) and σ_xx → σy0.

**T4 — Kinematic hardening / Bauschinger.** Load into plastic range in +x, then
reverse. With pure kinematic hardening (H_iso=0) the reverse yield occurs when the
stress has traversed `2·σy0` from the unload point (Bauschinger effect): the elastic
span on reversal is 2σy0. Assert the reverse-yield onset is at Δσ = 2σy0 (±tol) and
the back stress magnitude grew consistently with (2/3)H_kin·ᾱ.

**T5 — Consistent tangent vs finite difference.** At several plastic states
(varying Δγ, with iso and kin hardening on), compute `D_alg` and compare to the
central finite-difference Jacobian of σ(ε) (perturbation h≈1e-7 per component).
Assert ‖D_alg − D_FD‖ / ‖D_alg‖ ≤ 1e-5. **This guards the quadratic convergence.**

**T6 — Symmetry of D_alg.** Assert `D_alg ≈ D_algᵀ` to 1e-12 (associative J2).

**T7 — Plastic incompressibility.** tr(Δεᵖ) = 0 (Δεᵖ_xx+Δεᵖ_yy+Δεᵖ_zz = 0) to
1e-12; mean stress p unaffected by the plastic correction.

**T8 — Allocation.** `@allocated return_map(...) == 0` after warmup.

### 8.2 Element tests

**T9 — Partition of unity.** Σ_a N_a = 1 and Σ_a ∂N_a/∂x_j = 0 at every Gauss
point (assert ≤ 1e-13).

**T10 — Jacobian / volume.** For `box_mesh`, detJ at each GP equals
(element volume)/8 with positive sign; Σ_gp detJ·w over an element equals element
volume. Assert against analytic box element volume.

**T11 — Rigid body motion.** Apply a uniform translation and a small rotation to
element node coords (as displacement); element strain ε = B u_e must be ~0
(translation: exactly 0; infinitesimal rotation: 0 to linear order). Assert
‖ε‖ ≤ 1e-10. Element internal force from rigid translation = 0.

**T12 — Single-element patch test (constant strain).** Prescribe displacements on
all boundary nodes consistent with a uniform strain field (e.g. ε_xx = c); solve
for any interior DOFs (none for a 1-element patch, several for a multi-element
patch). The recovered stress must be the exact constant `ℂ : ε` at **every** Gauss
point of **every** element, and reactions must be in equilibrium. Assert ≤ 1e-10.
(Standard FEM patch test — guarantees convergence/consistency of the element.)

### 8.3 System / solver tests

**T13 — Uniaxial tension cube, full pipeline (elastic + plastic).** The §6.1
model. In the elastic range assert σ_xx = E·ε_xx and lateral strains
ε_yy=ε_zz=−ν·ε_xx (roller BCs give a uniaxial stress state). Past yield assert the
T2 analytical curve at several load steps. Mesh-independence: 1, 2³, 4³ elements
give the same uniform stress (≤ 1e-8).

**T14 — Elastic cantilever tip deflection vs beam theory.** §6.2 geometry, fully
elastic load (P small enough to stay elastic). Euler–Bernoulli tip deflection for
an end-loaded cantilever:
```
δ = P L³ / (3 E I),     I = b h³ / 12  (rectangular section).
```
For L=10, b=h=1 ⇒ I=1/12, with E=210000 and P=100:
δ = 100·1000 / (3·210000·(1/12)) = 100000 / (52500) ≈ **1.905** (length units).
Hex8 with a reasonably refined mesh (≥ 40×4×4) plus shear contribution will give a
slightly larger value (Timoshenko correction `+ P L /(G A κ)`); assert agreement
with Euler–Bernoulli to within ~5–10% and convergence toward the Timoshenko value
under refinement. (This is a convergence-trend assertion, with the analytic value
as the target, acknowledging Hex8 shear/locking behavior.)

**T15 — Newton quadratic convergence.** On a plastic load step using the
consistent tangent, the residual norm must drop quadratically: log–log of
‖R^{k+1}‖ vs ‖R^k‖ has slope ≈ 2 in the asymptotic regime. Concretely assert
‖R^{k+1}‖ ≤ C ‖R^k‖² for the last 2–3 iterations, and that the step converges in
a small number of iterations (≤ ~6) to tol=1e-8. (If a non-consistent/continuum
tangent were used by mistake, convergence degrades to linear — this test catches
that regression.)

**T16 — Stiffness symmetry & SPD-ness.** Assembled K_T (before BC imposition) is
symmetric to 1e-10; after imposing sufficient Dirichlet BCs to remove rigid-body
modes, K_T is positive definite (Cholesky succeeds) in the elastic regime.

**T17 — Reaction equilibrium / global force balance.** Sum of reaction forces on
Dirichlet DOFs equals minus the applied external load (∑F_react + ∑F_ext = 0) to
1e-8 — conservation check at every converged step.

**T18 — Load-path consistency (commit semantics).** Loading to a state in N steps
vs 2N steps must give the same final stress/plastic strain to a small tolerance
for this rate-independent model (linear hardening makes it path-step-insensitive
for monotonic loading). Validates that history commit and stepping are correct.

**T19 — Unload elastic check.** Load into plasticity, then unload: the unloading
slope in σ–ε must be the elastic modulus E (not the tangent H_t), and residual
plastic strain matches ᾱ. Assert unloading stiffness = E to 1e-6.

**T20 — Assembly allocation / scaling.** `@allocated assemble!` bounded constant
(§7.1); nnz(K) and assemble time scale ~linearly with nelem on a refinement
sweep.

### 8.4 Summary of "hard" analytical targets

| Test | Quantity                    | Closed-form / target                                  |
|------|-----------------------------|-------------------------------------------------------|
| T2   | uniaxial post-yield stress  | σ = σy0 + (E·H_iso/(E+H_iso))(ε − σy0/E)              |
| T3   | perfect plasticity          | σ → σy0 (cap)                                          |
| T4   | Bauschinger reverse yield   | elastic span on reversal = 2σy0                       |
| T13  | uniaxial elastic            | σ_xx = E ε_xx, ε_lat = −ν ε_xx                        |
| T14  | cantilever tip deflection   | δ = P L³ / (3 E I), I = b h³/12 (≈1.905 for the case) |
| T15  | Newton convergence          | ‖R^{k+1}‖ ≤ C‖R^k‖² (quadratic)                       |
| T17  | global balance              | ∑F_react + ∑F_ext = 0                                  |

---

## 9. Conventions (single source of truth)

- **Voigt ordering:** `[xx, yy, zz, xy, yz, zx]` — used identically for stress,
  strain, B-matrix rows, ℂ, and Dᵃˡᵍ everywhere in the code.
- **Strain shear:** **engineering** shear in Voigt strain vectors
  (γ_xy = 2ε_xy). The ℂ shear-diagonal entries are `G` (Section 2.2) to be
  consistent. Stress Voigt uses physical shear components.
- **Norm of deviatoric stress:** `ξ:ξ = ξ_xx²+ξ_yy²+ξ_zz² + 2(ξ_xy²+ξ_yz²+ξ_zx²)`
  (physical shear components; factor 2 from the two off-diagonal tensor entries).
- **von Mises stress:** `q = sqrt(3/2)·‖dev σ‖` with the norm above.
- **Sign convention:** tension positive; mean stress p = (1/3) tr σ (so
  σ = s + p·𝟙); pressure = −p.
- **DOF numbering:** node-major; global DOF of (node n, component c) = `3(n−1)+c`,
  c=1→x, 2→y, 3→z. Local element DOFs ordered node 1..8, each x,y,z.
- **Node ordering (Hex8):** as in Section 3.1 (VTK_HEXAHEDRON-compatible).
- **Units:** unit-agnostic / consistent units. Examples use N, mm, MPa
  (E in MPa, lengths in mm, stress in MPa, forces in N). The library does no unit
  conversion; the user supplies a consistent set.
- **Load factor λ:** ramped linearly 1/N … 1 over N steps by default; Dirichlet
  and Neumann magnitudes are scaled by λ when `ramp=true`.
- **Committed vs trial state:** `return_map` reads committed history of the
  *previous converged step* and the *current* total strain; per-GP state is
  committed (copied trial→committed) only after a load step's Newton loop
  converges.

---

## 10. Open extension points (intentionally deferred, not built in v1)

These are noted so the v1 interfaces don't paint us into a corner, but are
**not** implemented in v1 (avoiding over-engineering):

- `Material` as an interface (`return_map(::AbstractMaterial, …)`) to add Voce /
  power-law isotropic hardening or other yield surfaces.
- `Element` interface to add Tet4/Tet10/Hex20, B-bar / selective reduced
  integration for incompressible-limit robustness.
- Per-element material assignment (`Vector{Int}` material id per element).
- Symbolic-factorization reuse / Cholesky for the SPD elastic predictor.
- Surface traction (true face integral) Neumann BCs (v1 uses consistent nodal
  forces / `distribute` for face loads).

Each extension touches exactly one module, by design.
