# PlasticityFEM.jl — Finite-Strain Plasticity Design Spec

An extension of `PlasticityFEM.jl` to **finite-strain (large-deformation) J2
elastoplasticity**, added as a new element family alongside the existing
small-strain `Hex8`. This document is the implementation contract: it is
self-contained, mathematically precise, and written to the same standard as
[`DESIGN.md`](DESIGN.md). Read `DESIGN.md` first; this spec only states what
*changes* and what is *added*.

Status: specification. The small-strain path (`DESIGN.md`) is unchanged and
remains the default.

---

## 0. Scope, philosophy, and the central idea

**Goal.** Let a user run the *same* model (mesh, BCs, loads, material) under
either small-strain or finite-strain kinematics, selected by one keyword, and see
genuine large deformations (finite rotations, necking, large bending/torsion).

**The central idea (why this is tractable).** For isotropic elasticity with the
multiplicative split `F = Fᵉ·Fᵖ`, if the elastic response is the **Hencky
(logarithmic) hyperelastic** law and the plastic flow is integrated with the
**exponential map**, then the stress-update problem expressed in the
**elastic logarithmic strain** is *algebraically identical* to the small-strain
radial-return problem (Simo 1992; Eterovic–Bathe 1990; Weber–Anand 1990;
Miehe–Apel–Lambrecht 2002). Concretely:

```
            ┌─────────────────────── per Gauss point ───────────────────────┐
  F  ──►  geometric  ──► εᵉ_tr (log strain, 6-Voigt) ──► [ return_map ] ──► τ, D
        pre-processor                                    (UNCHANGED v1 kernel)
            └──────────────────────────────────────────────────────────────┘
                                          │
  Fᵖ_{n+1}, σ, fᵉ, Kᵉ  ◄──  geometric post-processor  ◄──────────────────────┘
                            (Kirchhoff stress, exp-map plastic update,
                             material + geometric consistent tangent)
```

So the verified, allocation-free `return_map` (DESIGN §2) is **reused verbatim**;
finite strain is a *geometric wrapper* around it, implemented as a new element.
The solver (Newton + CG/AMG, SCALING §1), the assembler's scatter machinery
(SCALING §2–3), and the BC/Model UX are **unchanged** — they consume `(fᵉ, Kᵉ)`
and don't care how those were produced.

**Scope (this extension):**

| Concern              | Choice                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Kinematics           | Finite strain, multiplicative `F = Fᵉ·Fᵖ`, **det Fᵖ = 1**            |
| Elastic law          | **Hencky** (quadratic in log strain) — degenerates to v1 ℂ exactly    |
| Plasticity           | J2 / von Mises, combined linear iso+kin hardening (same as v1)        |
| Integration          | **Exponential map** of the flow rule (preserves plastic incompressibility) |
| Stress measure       | **Kirchhoff** `τ = Jσ` internally; Cauchy `σ = τ/J` reported          |
| Formulation          | **Updated-Lagrangian / spatial** (reuses the v1 `bmatrix`)            |
| Tangent              | Consistent **material + geometric (initial-stress)** spatial tangent  |
| Element              | `Hex8Finite` (+ `Hex8FiniteFbar` for the near-incompressible limit)   |
| Locking cure         | **F-bar** (de Souza Neto et al. 1996), the finite-strain B-bar        |

**Non-goals (unchanged from v1):** contact, dynamics/inertia, nonlinear
hardening, non-hex elements, anisotropic plasticity, thermomechanics.

**Why updated-Lagrangian/spatial (not total-Lagrangian).** The spatial form lets
us reuse the existing 6×24 `bmatrix` (built from *spatial* gradients `∂N/∂x`),
the existing 6-Voigt stress/tangent data layout, and the existing
`∫ Bᵀ(·) dV` assembly almost verbatim. The total-Lagrangian (Green–Lagrange +
2nd-PK + material tangent, Miehe 1998) is mathematically equivalent and is noted
where relevant, but it would require a different (nonlinear) B-operator.

---

## 1. Conventions (delta from DESIGN §9)

Everything in `DESIGN.md §9` holds. Additions / clarifications:

- **Reference vs current.** `X` = reference (material) coordinates (the mesh
  node coordinates, undeformed). `x = X + u` = current (spatial) coordinates.
- **Deformation gradient** `F = ∂x/∂X = I + ∂u/∂X = I + Gradᵤ`. `J = det F > 0`.
- **Tensor storage.** `F`, `Fᵖ`, `bᵉ` are full 3×3 `SMatrix{3,3}` (9 components;
  `F`, `Fᵖ` are generally non-symmetric). Symmetric tensors (`εᵉ`, `τ`, `Cᵖ⁻¹`)
  use the existing 6-Voigt `[xx,yy,zz,xy,yz,zx]` ordering. **Strain Voigt uses
  engineering shear** (γ = 2ε); **stress Voigt uses physical shear** — identical
  to v1, so `return_map` consumes/produces the same layout.
- **Work-conjugacy.** The Hencky elastic strain `εᵉ = ½ ln bᵉ` is work-conjugate
  to the **Kirchhoff** stress `τ` (in the rotated/principal frame). This is why
  `return_map`, whose `elastic_matrix(λ,G)` maps strain→stress, yields `τ` (not
  `σ`) when fed log strain — see §3.2.
- **Plastic state stored in the reference frame** as `Cᵖ⁻¹ = Fᵖ⁻¹·Fᵖ⁻ᵀ`
  (symmetric, 6-Voigt) so the trial elastic tensor `bᵉ_tr = F·Cᵖ⁻¹·Fᵀ` is a
  simple push-forward each iteration. Initial (unloaded) value `Cᵖ⁻¹ = I`.
- **Hardening variables** `β` (back-stress deviator) and `ᾱ` (accumulated
  equivalent plastic strain) carry over **unchanged** — they live in the same
  log/principal space and are updated by the same `return_map` arithmetic.

---

## 2. Kinematics (the geometric pre-processor)

Per element, per Gauss point, per Newton iteration:

### 2.1 Deformation gradient
With reference shape-function gradients `∂Nₐ/∂X` (8×3, **cacheable** — they
depend only on the undeformed mesh, exactly like the v1 reference geometry) and
element nodal displacements `uₐ` (a=1..8):

```
F = I + Σₐ uₐ ⊗ (∂Nₐ/∂X)        (3×3 SMatrix)
J = det F                        (> 0; abort the step if J ≤ 0)
```

The spatial gradients needed for the v1 `bmatrix` are
`∂Nₐ/∂x = F⁻ᵀ · ∂Nₐ/∂X` (equivalently `(∂Nₐ/∂X)ᵀ F⁻¹`). The spatial B-matrix is
then **the existing** `bmatrix(∂N/∂x)`.

### 2.2 Trial elastic left Cauchy–Green and its spectral form
With the committed plastic state `Cᵖ⁻¹_n` (frozen over the Newton loop):

```
bᵉ_tr = F · Cᵖ⁻¹_n · Fᵀ          (symmetric positive-definite 3×3)
```

Spectral decomposition (symmetric 3×3 eigenproblem):

```
bᵉ_tr = Σ_{A=1}^{3} (λᵉ_A)² · n_A ⊗ n_A
```

where `(λᵉ_A)²` are eigenvalues (elastic principal stretches squared) and `n_A`
the orthonormal spatial principal directions.

### 2.3 Trial elastic logarithmic (Hencky) strain
```
εᵉ_tr = ½ ln bᵉ_tr = Σ_A ln(λᵉ_A) · n_A ⊗ n_A
```
Assemble into 6-Voigt **engineering-shear** form `εᵉ_tr_voigt` (off-diagonal
entries multiplied by 2) so it matches what `return_map` expects for a strain.

> **Numerical note (eigensolver).** Use a robust symmetric-3×3 eigendecomposition
> (closed-form via the analytic formula, or `StaticArrays`/`LinearAlgebra`
> `eigen` on the `SMatrix{3,3}`). Must be allocation-free in the hot loop. Handle
> near-degenerate eigenvalues (repeated stretches) — see §4.3.

---

## 3. Constitutive update in log space (reuse `return_map`)

### 3.1 The equivalence theorem
Feed the trial Hencky strain through the **unchanged** small-strain kernel:

```
(τ, εᵖ_log_{n+1}, β_{n+1}, ᾱ_{n+1}, D) = return_map(mat, εᵉ_tr_voigt, εᵖ_log_n, β_n, ᾱ_n)
```

This is exact: the J2 radial-return equations in principal logarithmic strains
have identical algebraic structure to the infinitesimal theory (Simo 1992). The
returned 6-Voigt "stress" **is the Kirchhoff stress** `τ` (principal/rotated
frame); the returned 6×6 `D = ∂τ/∂εᵉ_tr` is the algorithmic modulus used to build
the spatial tangent in §4.

> **Why `return_map` already encodes Hencky hyperelasticity.** `elastic_matrix(λ,G)`
> is `τ = K·tr(εᵉ)·1 + 2G·dev(εᵉ)`. With `εᵉ` the Hencky strain this is exactly
> the isotropic Hencky stored-energy response. No change to `Materials.jl` is
> required. (The v1 `J2Material` is reused as-is.)

### 3.2 Stress outputs
```
τ (Kirchhoff, 6-Voigt physical shear)   — work-conjugate to εᵉ, used for fᵉ, Kᵉ
σ = τ / J  (Cauchy, reported / VTK)
```

### 3.3 Plastic update by exponential map (preserves det Fᵖ = 1)
The flow is **coaxial** with `bᵉ_tr` (associative J2 ⇒ the plastic corrector is a
deviatoric scaling along the same principal axes `n_A`). Hence the converged
elastic log strain shares the trial principal directions:

```
εᵉ_A^{n+1} = εᵉ_tr_A − Δεᵖ_A          (Δεᵖ from return_map, deviatoric ⇒ Σ_A Δεᵖ_A = 0)
bᵉ_{n+1}   = Σ_A exp(2 εᵉ_A^{n+1}) · n_A ⊗ n_A
Cᵖ⁻¹_{n+1} = F⁻¹ · bᵉ_{n+1} · F⁻ᵀ      (push back to reference; the stored state)
```

**Plastic incompressibility is exact.** `Δεᵖ` is deviatoric (`tr Δεᵖ = 0`), so
`tr εᵉ_{n+1} = tr εᵉ_tr` ⇒ the elastic volume ratio `Jᵉ = exp(tr εᵉ)` is
unchanged by the corrector ⇒ `det Cᵖ` (hence `det Fᵖ`) is preserved. Started at
`det Fᵖ = 1`, it stays `1` to machine precision — the exponential-map payoff that
the additive small-strain split only approximates.

> Equivalent (and often cheaper) statement avoiding a re-exponentiation:
> reconstruct `bᵉ_{n+1}` directly from the corrected principal values. Either
> form is acceptable provided the `det Fᵖ = 1` test (§7) passes to ~1e-12.

### 3.4 Objective kinematic hardening (rotation-neutralized back-stress)
Frame-indifference (objectivity) requires that under a superposed spatial rotation
`Q` (`F → QF`) the Kirchhoff stress rotate as `τ → Q τ Qᵀ`. The trial Hencky
strain already does: eigenvalues of `bᵉ_tr = F Cᵖ⁻¹ Fᵀ` are `Q`-invariant and the
eigenvectors co-rotate. **But a back-stress stored in the global spatial frame does
NOT co-rotate**, so `ξ = s − β` uses `β` in the wrong frame and the model is
non-objective — up to ~17% Cauchy-stress error at large `Q` with `Hkin > 0` (an
adversarial F4 check exposes this).

Fix (de Souza Neto §14 / Simo & Hughes): store the back-stress in the
**rotation-neutralized (reference) configuration** `β_ref`, which — like `Cᵖ⁻¹` —
is `Q`-invariant. Inside the stress update, push it forward to the spatial frame
with the **polar rotation** `R` of `F` (`F = R U`):

```
β_sp = R · β_ref · Rᵀ      (spatial back-stress for the return map)
... return_map(εᵉ_tr, β_sp) → τ, β_sp_{n+1} ...
β_ref_{n+1} = Rᵀ · β_sp_{n+1} · R    (pull the update back to reference, stored)
```

Under `F → QF`, `R → QR`, so `β_sp → Q β_sp Qᵀ` and hence `τ → Q τ Qᵀ` — exact
objectivity (error ~1e-15 even at 120°, F4b). For isotropic / perfect plasticity
(`Hkin = 0`) `β_ref ≡ 0`, so `β_sp = 0` for any `R` and the polar decomposition is
skipped entirely (the path reduces to the original symmetric one). The consistent
tangent for this term is given in §4.6.

---

## 4. Element internal force and consistent tangent

### 4.1 Internal force
Because `∫_Ω Bᵀσ dv = ∫_{Ω₀} Bᵀ(Jσ) dV = ∫_{Ω₀} Bᵀτ dV`, integrate the
**Kirchhoff** stress over the **reference** volume using the **cached reference**
weights `detJ₀·w` and the **spatial** B-matrix:

```
fᵉ = Σ_gp  B_spatialᵀ · τ_voigt · (detJ₀·w)            (SVector{24})
```

This mirrors the v1 `Fe += (Bᵀσ)·w` exactly, with `σ→τ` and spatial `B`.

### 4.2 Tangent structure
The element tangent has two parts (this is the essential new physics vs v1):

```
Kᵉ = Kᵉ_material + Kᵉ_geometric
```

- **Material** `Kᵉ_material = Σ_gp B_spatialᵀ · a · B_spatial · (detJ₀·w)`, where
  `a` is the **spatial algorithmic modulus** (6×6 Voigt) obtained by pushing the
  log-space modulus `D` through the derivative of the tensor log/exp map (§4.3).
- **Geometric / initial-stress** `Kᵉ_geometric` couples the current Kirchhoff
  stress with the gradient of the test/trial functions (§4.4). It is what makes
  finite rotations and buckling correct and Newton quadratic.

> **Tangent symmetry — when K = Kᵀ, and when it is not.** Symmetry is *not*
> universal here; it depends on the element kind and the hardening:
>
> | configuration | tangent | linear solver (auto) |
> |---|---|---|
> | small strain (`:small`) | **symmetric** | CG + AMG |
> | finite strain (`:finite`), isotropic / perfect (Hkin = 0) | **symmetric** | CG + AMG |
> | finite strain (`:finite`), **kinematic hardening** (Hkin > 0) | **non-symmetric** | direct (UMFPACK) |
> | **F-bar** (`:finite_fbar`), any material | **non-symmetric** (always) | direct (UMFPACK) |
>
> For the symmetric cases the production two-point form `K = ∫ Gᵀ A G dV` is
> symmetric (the §4.3 material part + §4.4 initial-stress part both preserve
> symmetry for the coaxial associative-J2 response), and the CG + AMG solver and
> the row-parallel `SymThreadedK` SpMV (SCALING §3.2) apply unchanged.
>
> Two effects break symmetry. (1) **F-bar**: the `∂F̄/∂uₑ` centroid-J₀ coupling
> (`_fbar_Geff`, de Souza Neto Box 15.2) is a rank-one non-symmetric operator, so
> the F-bar tangent is non-symmetric for *any* material. (2) **Objective kinematic
> hardening** (§4.6): the back-stress is stored in the reference configuration and
> pushed to the spatial frame by the polar rotation `R(F)` (the fix that makes the
> law frame-indifferent, §3.4); the resulting `∂τ/∂β_sp · ∂β_sp/∂F` term is
> non-symmetric. We **prioritize objectivity over symmetry** — the non-symmetric
> tangent is fully *consistent* (FD-verified <1e-6, Newton stays quadratic).
>
> Because the tangent can be non-symmetric, `solve!` is **symmetry-aware**: it
> inspects the element kind + `Hkin` and auto-selects CG (symmetric) or the direct
> UMFPACK solve (non-symmetric). Forcing `linsolve=:cg` on a non-symmetric
> configuration warns and overrides to `:direct` (CG/AMG/SymThreadedK all assume
> K = Kᵀ). Tests assert symmetry only for the symmetric configurations (F9) and
> assert consistency-but-non-symmetry for the kinematic case (F9b).

### 4.3 Spatial material modulus `a` (principal-axis form)
With principal Kirchhoff stresses `τ_A`, trial eigenvalues `b_A = (λᵉ_tr_A)²`,
and the **principal block** `D_AB = ∂τ_A/∂εᵉ_tr_B` (3×3, extracted from the
return-map modulus `D` in the principal frame), the spatial elasticity tensor in
principal dyads `m_A = n_A⊗n_A`, `m_AB = n_A⊗n_B` is (Simo & Hughes 1998, Box 8.2;
de Souza Neto–Perić–Owen 2008, Box 14.3):

```
a = Σ_A Σ_B  (D_AB − 2 τ_A δ_AB) · (m_A ⊗ m_B)
  + Σ_A Σ_{B≠A}  g_AB · (m_AB ⊗ m_AB + m_AB ⊗ m_BA)
```

with the eigenvalue-coupling coefficient

```
        τ_A b_B − τ_B b_A
g_AB = ───────────────────          (b_A ≠ b_B)
            b_A − b_B
```

> **Sign convention (verified).** This numerator is `τ_A b_B − τ_B b_A` (NOT
> `τ_B b_A − τ_A b_B`). The order was confirmed by finite-differencing the
> Kirchhoff stress under symmetric spatial velocity-gradient perturbations
> (Lie-derivative definition `a:H = dτ/dε − Hτ − τH`): the form above reproduces
> the ground-truth `a` to machine precision (~1e-10) and converges continuously
> to the degenerate limit below; the opposite sign gives ~78% error. Match
> indices carefully against Simo & Hughes Box 8.2 / de Souza Neto Box 14.3.

and the **degenerate limit** `b_A → b_B` (repeated stretch), required for
numerical robustness:

```
g_AB → ½ (D_BB − D_AB) − τ_A          (use when |b_A − b_B| < tol)
```

The `−2 τ_A δ_AB` term arises from `d(½ ln b)/d(b)` (the nonlinearity of the log
strain measure) — it is the spectral form of the geometric contribution of the
strain measure and is easy to drop by mistake; the FD tangent check (§7) is the
backstop.

> **Authoritative source + mandatory gate.** Implement `a` from the cited boxes,
> then **finite-difference-verify the assembled element tangent `Kᵉ` against `fᵉ`**
> (central differences, `‖K_FD − K‖/‖K‖ < 1e-6`) across many *deformed* and
> *plastic* states. This single test guards every factor above. Do not consider
> §4.3–4.4 correct until it passes (it is the finite-strain analogue of the v1 T7
> tangent check).

### 4.4 Geometric (initial-stress) stiffness
Standard form with the current Cauchy/Kirchhoff stress and spatial gradients:

```
Kᵉ_geometric[ (a,i),(b,k) ] = δ_ik · Σ_gp (∂Nₐ/∂x)ᵀ · τ · (∂N_b/∂x) · (detJ₀·w)
```

(`a,b` node indices; `i,k` spatial components; `τ` the 3×3 Kirchhoff stress).
This adds the same scalar to all three diagonal component-blocks of each
node-pair.

### 4.5 Equivalent two-point (P–F) implementation (optional, informative)
An algebraically equivalent route assembles `Kᵉ = Σ_gp Gᵀ A G (detJ₀·w)` and
`fᵉ = Σ_gp Gᵀ P_9 (detJ₀·w)`, where `P = τ·F⁻ᵀ` is the first Piola–Kirchhoff
stress, `G` (9×24) maps `uₑ → F` via the **reference** gradients (cacheable), and
`A = ∂P/∂F` (9×9). This automatically contains both material and geometric parts.
It is a valid alternative; the spatial form §4.1–4.4 is the **primary**
specification because it maximizes reuse of the v1 `bmatrix`/Voigt code. Whichever
is implemented, the FD gate §7 applies.

> **Implementation note.** The production element kernel uses this two-point
> (P–F) form (`dPdF`), not the spatial form, because it is also valid for the
> non-coaxial corrector of kinematic hardening. The spatial form §4.3/§4.4
> (`spatial_modulus` + `geometric_stiffness`) is retained as the reference
> derivation and is unit-tested to reproduce `dPdF` to machine precision for the
> isotropic coaxial case (F12).

### 4.6 Tangent term for objective kinematic hardening
With the rotation-neutralized back-stress (§3.4) the spatial back-stress
`β_sp = R(F)·β_ref·Rᵀ` depends on `F` through the polar rotation, so the consistent
tangent gains

```
∂τ/∂F |_β = (∂τ/∂β_sp) · (∂β_sp/∂F)
```

- `∂τ/∂β_sp` (`dtau_dbeta`): at fixed trial strain the plastic corrector is
  `c(ξ) = a·ξ`, `ξ = s_tr − β_sp`, `a = 2G√(3/2)Δγ/‖ξ‖`. Since `τ = σ_tr − c` and
  `∂ξ/∂β = −I`, `∂τ/∂β = ∂c/∂ξ = a·I + ξ⊗(∂a/∂β)` (zero in the elastic branch).
  FD-verified ~1e-10.
- `∂β_sp/∂F`: from `β_sp = R β_ref Rᵀ` with the **polar-rotation derivative**
  `Ṙ` (`dR_polar`), obtained by solving the Sylvester equation `Ω U + U Ω =
  RᵀḞ − ḞᵀR` for the skew `Ω = RᵀṘ` in `U`'s eigenbasis.

This term is non-symmetric (see §4.2) but makes the full tangent consistent
(`dPdF` FD-verified <1e-6, F9b). It is added inside `dPdF` and is inactive when
`Hkin = 0`.

---

## 5. F-bar (near-incompressible limit)

Trilinear Hex8 locks volumetrically; J2 plastic flow is isochoric, so the locking
worsens at large plastic strain. The finite-strain cure is **F-bar** (de Souza
Neto et al. 1996), the large-strain analogue of B-bar:

```
F̄ = (J₀ / J)^{1/3} · F           (replace F by F̄ everywhere in §2–4)
```

where `J₀ = det F₀` is evaluated at the **element centroid** (natural coords
`(0,0,0)`). The volumetric part is taken from the centroid, relaxing the
incompressibility constraint per element. The deviatoric part is unchanged.

The consistent tangent gains an **F-bar coupling term** from `∂F̄/∂F` (the
centroid `J₀` depends on all nodal displacements). Implement per de Souza Neto
Box 15.2; FD-verify as in §4.3.

`Hex8FiniteFbar` is a distinct element kind selecting this path; `Hex8Finite`
uses the standard `F`. F-bar is required for the necking benchmark (§7) and any
fully-developed plastic flow demo.

---

## 6. Data structures, dispatch, and API

### 6.1 Element kind (the dispatch seam)
Introduce a **type-level element kind** so the assembly hot loop dispatches
statically (no runtime branch in the kernel, mirroring `Val{COMMIT}` /
`Val{UNIFORM}` in `Assembly.jl`):

```julia
abstract type ElementKind end
struct Hex8Small      <: ElementKind end   # current v1 path (default)
struct Hex8Finite     <: ElementKind end   # finite strain, standard F
struct Hex8FiniteFbar <: ElementKind end   # finite strain, F-bar
```

`Model` carries the kind as a type parameter (`Model{Ti,EK<:ElementKind}`), set at
construction. The assembler selects the element kernel by dispatch on a
zero-size `EK` instance. The v1 kernel `element_force_tangent!` is the
`Hex8Small` method; new kernels `element_force_tangent_finite!` (and the F-bar
variant) implement §2–5. **Allocation-free, StaticArrays, same `commit::Val`
convention.**

### 6.2 Per-Gauss-point state (extend `GaussState`)
Add the plastic deformation history. `εᵖ` (additive log plastic strain) is kept
for the return-map call; the *tensorial* plastic configuration is `Cᵖ⁻¹`:

```
Cp_inv :: Matrix{Float64}   # 6 × ngp, symmetric Voigt; initialized to I = [1,1,1,0,0,0]
```

- `reset!` initializes `Cp_inv` columns to `[1,1,1,0,0,0]` (**identity, not zero**).
- `copyto!`/commit semantics extended to copy `Cp_inv`.
- Small-strain models leave `Cp_inv` unused (or omit via the element kind).
- Memory: +6 floats/GP committed + trial (negligible vs existing `εp,β,σ`).

### 6.3 Cache (`ElementCache`)
Reference shape gradients `∂N/∂X` and `detJ₀·w` are geometry of the *undeformed*
mesh ⇒ the existing uniform-mesh caching (`SCALING §2.2`) **still applies** and is
reused: one reference set for a uniform box. What cannot be cached is the
*spatial* push-forward (`F`, `F⁻¹`, spatial B), recomputed per element per
iteration — this is unavoidable for finite strain and is `O(nelem)` compute with
`O(1)` extra memory (same structure as the v1 non-uniform path).

### 6.4 Public API (Model.jl / PlasticityFEM.jl)
Select the kind at model construction; everything else is identical:

```julia
model = Model(mesh, steel)                       # small strain (default, unchanged)
model = Model(mesh, steel; element = :finite)     # finite strain, standard F
model = Model(mesh, steel; element = :finite_fbar) # finite strain, F-bar
```

`:small` (default), `:finite`, `:finite_fbar` map to the three kinds. `fix!`,
`prescribe!`, `load!`, `solve!`, `reset!`, and all postprocessing
(`nodal_displacements`, `gauss_stress`, `equivalent_plastic_strain`,
`write_vtu`) work unchanged. `gauss_stress` reports **Cauchy** stress for finite
elements (document this). VTK warp-by-vector shows the true deformed shape.

---

## 7. Verification & validation plan (the correctness gates)

Mirrors `DESIGN §8` rigor. All are added to the test suite; the suite must stay
green and allocation gates must hold.

| ID  | Test | Target |
|-----|------|--------|
| F1  | **Small-displacement limit**: `:finite` reproduces `:small` on `tension_cube` & `cantilever` at small load | match to ~1e-8 (relative) |
| F2  | **Consistent tangent vs FD** (§4.3) across many deformed+plastic GP states | `‖K_FD−K‖/‖K‖ < 1e-6`; quadratic Newton observed |
| F3  | **Plastic incompressibility** `det Fᵖ = 1` after finite plastic flow | `< 1e-12` |
| F4  | **Objectivity**: superpose a finite rigid rotation `Q` on a stressed state | Cauchy stress rotates as `QσQᵀ`; **no** spurious stress/dissipation |
| F5  | **Frame-indifference under cyclic simple shear** (Jaumann pathology check) | no spurious stress oscillation / energy drift |
| F6  | **Necking of a tension bar** (3D, `:finite_fbar`) | load–displacement & neck profile vs published de Souza Neto / Simo results |
| F7  | **Volumetric-locking relief**: `:finite` vs `:finite_fbar` on near-incompressible plastic bending | F-bar markedly softer / converged where standard locks |
| F8  | **Large-rotation patch / cantilever**: 90° bending or large torsion | physically sensible; energy-consistent; matches refined reference |
| F9  | **Tangent symmetry**: `‖Kᵉ − Kᵉᵀ‖` | `< 1e-10·‖Kᵉ‖` (guards CG/AMG validity) |
| F10 | **Allocation gates**: finite element kernel + assembly | `0` bytes / `O(1)` like v1 (T20-style) |

F2 is the master gate for §4; F1 is a free regression that any factor error in
§2–4 will break.

---

## 8. Module / file plan

| File | Change |
|------|--------|
| `Materials.jl` | **none** (`return_map`, `J2Material` reused as-is) |
| `FiniteStrain.jl` | **new**: kinematics (§2), log pre/post-processor (§3), spatial tangent `a` (§4.3), geometric stiffness (§4.4), F-bar (§5) — pure, allocation-free |
| `Elements.jl` | add `element_force_tangent_finite!` (+ F-bar) kernels; cache `∂N/∂X` (reference gradients) |
| `Mesh.jl` | extend `GaussState` with `Cp_inv` (§6.2); init/`copyto!` |
| `Assembly.jl` | static dispatch on `ElementKind` (§6.1); finite path recomputes spatial geometry per element/iter |
| `Model.jl` | `element=` keyword → `ElementKind` type parameter; Cauchy reporting for finite |
| `Solver.jl` | **none** (consumes `fᵉ,Kᵉ`; add `J ≤ 0` step-cut safeguard only if needed) |
| `Visualization.jl` | report Cauchy stress on deformed config (already warps) |
| `PlasticityFEM.jl` | export the element-kind selector / new public surface |
| `test/test_finite_strain.jl` | **new**: F1–F10 |
| `examples/` | **new**: necking bar, large-rotation cantilever/torsion |
| `README.md` | document `element=` selection and the finite-strain scope |

Each change is local; the solver/assembler scatter core is untouched.

---

## 9. References

- E. A. de Souza Neto, D. Perić, D. R. J. Owen, *Computational Methods for
  Plasticity: Theory and Applications*, Wiley (2008) — log-strain framework
  (Ch. 14), F-bar (Ch. 15). **Primary implementation reference.**
- J. C. Simo, T. J. R. Hughes, *Computational Inelasticity*, Springer (1998) —
  multiplicative theory; principal-axis spatial tangent (Box 8.2).
- J. C. Simo, "Algorithms for static and dynamic multiplicative plasticity that
  preserve the classical return mapping schemes of the infinitesimal theory,"
  *CMAME* 99 (1992) 61–112 — the equivalence theorem.
- G. Weber, L. Anand, *CMAME* 79 (1990); A. L. Eterovic, K.-J. Bathe, *IJNME* 30
  (1990) — log-strain finite-strain plasticity.
- C. Miehe, N. Apel, M. Lambrecht, *CMAME* 191 (2002) — modular log-strain space
  formulation; C. Miehe, *IJNME* 1998 — exponential-map algorithmic tangent.
- E. A. de Souza Neto, D. Perić, M. Dutko, D. R. J. Owen, *Int. J. Solids
  Struct.* 33 (1996) — F-bar method.
- J. Korelc, S. Stupkiewicz, *IJNME* (2014) — closed-form matrix exponential and
  its differentiation for finite-strain plasticity.
