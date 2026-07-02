# ParticlePlasticity.jl — Design Document

A 3D **Material Point Method (MPM)** solver for J2 / von Mises elastoplasticity at
**finite and very large deformation**, built to reuse the verified finite-strain
constitutive kernel of the sibling [`../lagrangian/`](../lagrangian/) FEM package.

Status: design only (no implementation yet). This document is the specification
implementers and validators rely on. It is self-contained and mathematically
precise. It follows the recommendations settled in
[`PROPOSAL.md`](./PROPOSAL.md) (read that first for the *why*; this is the *how*).

**Revision note.** This design was hardened by an expert-programmer review before
any code was written. The review confirmed the reuse contract (§3) against the
actual kernel and found one critical spec bug now fixed: damping was applied
*after* grid BCs (§4/§7), which would have silently under-driven every
prescribed-velocity boundary in the quasi-static examples — reordered so BCs are
always enforced last. It also found the F-update should use the APIC affine matrix
`Cₚ` directly (§4) rather than a separately-computed grid gradient (simpler and the
standard choice), pinned down that mass scaling must use one consistent density for
both `Δt` and particle mass (§6), and trimmed day-one complexity: the exp/mid-point
F-update and dynamic relaxation are deferred to be added only if a dedicated
measurement (gate **G1b**, §12) shows they're needed, rather than built speculatively.

---

## 0. Mandate (the two goals, in order)

1. **Replicate the `lagrangian` finite-strain outcomes.** The first job is *not*
   novelty — it is to reproduce, with an entirely different spatial
   discretization, the results the FEM solver already produces for finite-strain
   J2: the same stress on the same deformation, the same reaction on a tension
   block, the same qualitative bend of a cantilever, the same neck localization on
   the moderate necking bar. If MPM and FEM disagree where both are valid, MPM is
   wrong until proven otherwise. This is the correctness anchor.
2. **Then expand to very large deformation.** Once (1) holds, push past where the
   Lagrangian mesh inverts (`ElementInversionError`): extreme necking toward
   rupture, a rod bent through large rotation, deformation where a fixed mesh
   entangles. This is the regime that justifies the method.

Everything below is chosen to serve (1) first — because the reuse of the verified
kernel makes (1) *achievable and checkable* — and (2) as the payoff.

---

## 1. Scope

**In scope (v1):**

| Concern | Choice |
|---|---|
| Method | **Explicit** MPM (symplectic Euler), single-grid |
| Grid | Uniform Cartesian background, reset every step |
| Basis | **Quadratic B-spline** (C¹, 27-node stencil in 3D) |
| Transfer | **APIC** (affine PIC; Jiang et al. 2015), single transfer |
| Constitutive | **reuse** `lagrangian` finite-strain log-strain radial return (§3) |
| Material | J2 / von Mises, combined **linear** isotropic + kinematic hardening |
| Kinematics | multiplicative finite strain `F = FᵉFᵖ`, per-particle `F`, `Cᵖ⁻¹` |
| Locking cure | **F̄ at particles** (cell-averaged J; Coombs et al. 2018), flag-gated |
| Quasi-static | mass scaling (single density for `Δt` and `mₚ`) + scalar viscous damping (§6) |
| BCs | grid-node velocity BCs via predicates; rigid symmetry planes |
| Output | per-particle `.vtu` (VTK point cloud) for ParaView |

**Non-goals (v1, explicit):** implicit/quasi-static-Newton MPM, CPDI/GIMP (B-spline
is the baseline), MLS-fused transfers, GPU, multi-body frictional contact,
nonlinear/saturation hardening, damage/fracture/erosion, adaptive particle
splitting, thermomechanical coupling. Each is a documented extension point, not
built speculatively. (Nonlinear hardening is the *first* planned post-v1 kernel
upgrade — it is required for the published Simo necking curve; see PROPOSAL §8 V3b.)

**Non-goal that matters:** we do **not** re-derive the constitutive physics. The
return map, the finite-strain log-strain wrap, the exponential-map plastic update,
the objectivity treatment, and their verification all come from `lagrangian`
unchanged (§3). If we find ourselves editing that kernel, we have left scope.

---

## 2. Method in one paragraph (recap)

MPM carries mass, momentum, `F`, and plastic history on material **points**. Each
step: (P2G) scatter particle mass/momentum to a background grid and assemble nodal
internal forces from particle stress; (grid) integrate momentum explicitly and
apply BCs on grid nodes; (G2P) gather the updated grid velocity back to particles
(APIC), advect particles, and form each particle's velocity gradient; (update)
integrate `F`, run the reused stress update, reset the grid. The grid never
deforms, so mesh entanglement — the wall that stops FEM — cannot occur. See
PROPOSAL §1–§4 for the literature basis of every choice.

---

## 3. The reuse contract (the heart of this design)

This is what makes the project tractable. The constitutive update is **imported,
not written**.

### 3.1 What we import from `lagrangian`

From `PlasticityFEM.Materials`:
- `J2Material(; E, ν, σy0, Hiso=0, Hkin=0)` — the material (verbatim).
- `return_map(mat, ε, εp_n, β_n, ᾱ_n)` — small-strain radial return (verbatim; not
  called directly by MPM — it is called *inside* `finite_stress_update`).

From `PlasticityFEM.FiniteStrain`:
- `finite_kinematics(F, Cp_inv_n) -> FiniteKin` — forms `bᵉ_tr = F·Cᵖ⁻¹·Fᵀ`,
  spectral log, trial Hencky strain, `J`, and an `ok` flag (`false` if `J ≤ 0`).
- `finite_stress_update(mat, kin, F, εp_n, β_ref_n, ᾱ_n)` — runs the return map on
  the trial log strain and does the exponential-map plastic update. Returns (we use
  a subset): `τ_voigt` (Kirchhoff stress, 6-Voigt physical shear), `εp_new`,
  `β_ref_new`, `ᾱ_new`, and `Cp_inv_new` (updated `Cᵖ⁻¹`, 6-Voigt).
- `FiniteKin`, `voigt_to_sym3`, `sym3_to_voigt`, `det_Fp_from_Cpinv` — helpers.

These are `@inline`, allocation-free, `StaticArrays`-only, and already
finite-difference-verified in `lagrangian/test/test_finite_strain.jl`. **We call
them; we do not modify them.**

### 3.2 Why the reuse is valid for explicit MPM (verified against the code)

- **Stress only, no tangent.** Explicit MPM assembles nodal forces from stress and
  advances momentum by `v ← v + Δt f/m`. There is **no global stiffness and no
  Newton iteration**, so the consistent tangent (`dPdF`, `spatial_modulus`) is
  never needed. We call exactly the *stress* half of the kernel. Confirmed: the
  returned `D`/tangent outputs are simply ignored.
- **Total-F interface matches.** `finite_kinematics` takes the *total* `F` and the
  committed `Cᵖ⁻¹`. MPM carries total `F` and `Cᵖ⁻¹` per particle, so the call is
  direct (`FiniteStrain.jl:148`).
- **Call-once-then-commit.** FEM calls the kernel on *trial* `F` each Newton
  iteration and commits `Cᵖ⁻¹` only at convergence. Explicit MPM calls **once per
  step** from the previous committed `Cᵖ⁻¹ₙ` and immediately stores the returned
  `Cᵖ⁻¹ₙ₊₁`. No double evaluation, no iteration.

### 3.3 The one guard the kernel does *not* do for us

`finite_stress_update` does **not** itself throw on `J ≤ 0`. In FEM that check
lives in the assembly (`Elements.jl:447`, `kin.ok || throw(ElementInversionError)`).
If `finite_kinematics` returns `ok=false` and we proceed, the fallback in
`FiniteStrain.jl:151-153` sets `εe_tr=0` and — critically — **force-sets `Finv=I3`
rather than the true inverse**. Tracing this through `finite_stress_update`: the
zero trial stress takes the elastic branch (`τ_voigt=0`), and because `Finv=I3` was
substituted, `Cp_inv_new` collapses to the **identity** — i.e. the particle's
entire plastic history is silently *reset*, not merely corrupted with "garbage."
This is a clean-looking but wrong state that could mask itself in diagnostics, which
is precisely why it must never be allowed to happen silently. The MPM
particle-update loop **must replicate the guard**: check `kin.ok`; on failure, raise
a typed `ParticleInversionError` (carrying the particle index and `J`). This is a
per-particle event and — unlike a whole FEM mesh — can be handled later by particle
splitting; for v1 it is a clear, loud failure, never swallowed.

### 3.4 Particle state initialization (get this wrong → NaN)

At `t = 0`, every particle: `F = I`, `Cᵖ⁻¹ = I` (Voigt `[1,1,1,0,0,0]`), `ᾱ = 0`,
`β_ref = 0`, `εp = 0`, `V = V₀`, `C = 0` (APIC affine), `v = v₀`. A zero `Cᵖ⁻¹`
makes `bᵉ_tr = 0` and the log singular — the single most common init bug.

---

## 4. The explicit MPM step (exact specification)

Notation: particle `p` at `xₚ` with mass `mₚ`, velocity `vₚ`, affine matrix `Cₚ`
(3×3), deformation gradient `Fₚ`, reference volume `Vₚ⁰`, current volume
`Vₚ = Jₚ Vₚ⁰`, Cauchy stress `σₚ = τₚ / Jₚ`. Grid node `i` at fixed `xᵢ` on a
uniform grid of spacing `h`. `Sᵢ(xₚ) = wᵢₚ` is the quadratic-B-spline weight,
`∇Sᵢ(xₚ) = ∇wᵢₚ` its gradient (§5). Sums over `i` run over the 27 nodes in the
particle's stencil.

```
--- P2G (particles → grid) : accumulate, then finalize ---------------------
for each particle p, for each stencil node i:
    mᵢ    += wᵢₚ mₚ
    pᵢ    += wᵢₚ mₚ (vₚ + Cₚ (xᵢ − xₚ))          # APIC affine momentum
    fᵢ    += − Vₚ σₚ ∇wᵢₚ                          # internal force  (−∫σ:∇S dV)
grid velocity:  vᵢ = pᵢ / mᵢ            (only where mᵢ > mₜₒₗ; else vᵢ = 0)

--- grid momentum update (explicit) ---------------------------------------
fᵢ    += mᵢ g                                       # body force (gravity, optional)
vᵢ*   = vᵢ + Δt fᵢ / mᵢ
[quasi-static only] vᵢ* *= (1 − α_damp)             # simple viscous damping, §6
apply grid BCs to vᵢ*   (§7: fixed / symmetry / prescribed-velocity nodes)
                                    # BCs applied LAST: damping must never touch a
                                    # constrained/prescribed node's velocity (a
                                    # damped-then-overwritten node is correct; a
                                    # damped-AFTER-BC node silently under-drives
                                    # every displacement-controlled example)

--- G2P (grid → particles) ------------------------------------------------
for each particle p:
    vₚ    = Σᵢ wᵢₚ vᵢ*                               # PIC velocity
    Cₚ    = (4/h²) Σᵢ wᵢₚ vᵢ* (xᵢ − xₚ)ᵀ            # APIC affine
    xₚ   += Δt vₚ                                    # advect

--- particle constitutive update (§3, reused kernel) ----------------------
    Lₚ    = Cₚ                                        # velocity gradient = APIC affine
                                                       # (standard APIC/MLS-MPM; no
                                                       # separate grid-gradient term)
    Fₚ    ← (I + Δt Lₚ) Fₚ                           # F-update (see 4.1)
    [F̄]   Fₚ ← F̄ₚ   if locking cure on               # §4.2
    kin   = finite_kinematics(Fₚ, Cᵖ⁻¹ₚ)
    kin.ok || throw(ParticleInversionError(p, kin.J))            # §3.3
    τv, εpₚ, β_refₚ, ᾱₚ, _, _, Cᵖ⁻¹ₚ, _, _, _ = finite_stress_update(mat, kin, Fₚ,
                                              εpₚ, β_refₚ, ᾱₚ)
                                    # real return is (τ_voigt, εp_new, β_ref_new,
                                    # ᾱ_new, D, τ_princ, Cp_inv_new, β_sp, R, U);
                                    # D/τ_princ/β_sp/R/U are tangent-only, unused
    Jₚ    = det Fₚ ;  σₚ = voigt_to_sym3(τv) / Jₚ   ;  Vₚ = Jₚ Vₚ⁰

reset the grid (zero mᵢ, pᵢ, fᵢ, vᵢ)
```

**Notes tying to the reuse contract:**
- **`Lₚ = Cₚ`, not a separately-computed grid gradient.** Standard APIC/MLS-MPM
  (Jiang 2015; Hu 2018) drives the F-update directly from the affine matrix `Cₚ`
  that P2G/G2P already compute — there is no separate `Σᵢ vᵢ*(∇wᵢₚ)ᵀ` reduction.
  (An earlier draft of this design used a separate grid-velocity-gradient term for
  the F-update and called the combination "textbook APIC"; that claim didn't hold
  up — the grid-gradient functional and the `Cₚ` moment-matched functional are
  numerically different quantities for a non-affine nodal velocity field, and only
  `Cₚ` matches the cited references. Using `Cₚ` is both the standard choice and
  the cheaper one, since it's already computed for the momentum transfer.)
- Internal force uses **current** `Vₚ` and **Cauchy** `σₚ`; equivalently
  `−Σ Vₚ⁰ τₚ ∇wᵢₚ`. These are identical (no `J` double-count) — a checked invariant.
- `σₚ = τₚ/Jₚ` is the only stress conversion; the kernel returns Kirchhoff `τ`.
- **`m` in the CFL formula (§4.3) and the stored particle mass `mₚ` must be the
  SAME (possibly scaled) density.** Mass scaling (§6) means literally setting
  `mₚ = ρ_eff·Vₚ⁰` on the particle at init — not just plugging a larger `ρ` into
  the `Δt` formula while leaving `mₚ` at the true density. Doing the latter breaks
  CFL and is unconditionally unstable; this is spelled out again in §6.

### 4.1 The F-update (accuracy note, from the design review)

The baseline `Fₚ ← (I + Δt Lₚ) Fₚ` is first-order and **not exactly isochoric**:
for plastic (traceless-`L`) flow `det(I + Δt L) ≠ 1`, injecting an `O(Δt²)` volume
error that *accumulates* over many thousands of explicit steps — exactly where
pressure/locking is delicate.
- **Use this total-F form** (store total `Fₚ`, feed it to `finite_kinematics`) —
  the kernel is built for total `F`; there is no incremental `bᵉ`-only variant.
- **v1 ships the forward form only, with a dedicated drift measurement (G-test
  below), not a second F-update variant.** An earlier draft of this design
  pre-built an exponential/mid-point (Hughes–Winget) update behind a `Val`-dispatch
  as a day-one option — that is premature: it adds a branch and an `exp(3×3)`
  routine to defend against a risk that hasn't been *measured* yet, for a
  many-step (necking) case that is Phase 5, not Phase 0–4 work. Instead, Phase 2
  adds a targeted unit test — drive one particle through many steps of prescribed
  pure shear/rotation through the actual MPM forward-Euler F-update loop and plot
  `det Fₚ` vs. step count — so the decision to add the accurate update is made from
  a real number, not a citation. If the drift measured there is unacceptable for
  the necking step counts in §13, add the exponential update **then**, as a Phase-5
  change, as a runtime flag (not `Val`-dispatched — see §9's dispatch-style note).

### 4.2 F̄ at particles (volumetric locking cure)

Isochoric J2 plastic flow locks trilinear/low-order MPM (checkerboard pressure),
just as it locks Hex8 (hence `lagrangian`'s F-bar). The MPM cure (Coombs et al.
2018) mirrors it with a *cell-averaged* Jacobian instead of the element-centroid
`J₀`:

1. Bin particles into background cells (each particle knows its cell from `⌊x/h⌋`).
2. Per cell `c`, volume-weighted mean dilation
   `J̄_c = (Σ_{p∈c} Jₚ Vₚ⁰) / (Σ_{p∈c} Vₚ⁰)`.
3. Replace each particle's gradient before the stress update:
   `F̄ₚ = (J̄_c / Jₚ)^{1/3} Fₚ`  (so `det F̄ₚ = J̄_c`, deviatoric part untouched).
4. Use `F̄ₚ` **consistently** in `finite_kinematics`, `finite_stress_update`, and in
   the current volume `Vₚ = J̄_c Vₚ⁰` for the internal force — matching how
   `lagrangian` feeds `F̄` to *both* kinematics and stress update (`Elements.jl:437–464`).

This is the one genuinely new constitutive-adjacent piece (the rest is reuse). It
is flag-gated (`fbar=true`) and validated by the Phase-3 no-checkerboard gate
before any large-plastic result is trusted.

### 4.3 Time step (CFL)

`Δt = C · h / c`, `c = √((K + 4G/3)/ρ)` (elastic dilatational/P-wave speed — the
fastest and correct one; conservative since the plastic tangent speed is lower),
`C ∈ [0.1, 0.5]` (default `0.2`). **`ρ` here is whatever density is actually
stored on the particles** (`mₚ = ρ·Vₚ⁰`) — if mass scaling (§6) is on, it is the
*scaled* `ρ_eff`, computed once and used consistently for both `Δt` and every
particle's `mₚ` at init. Using a scaled `ρ` in this formula while leaving `mₚ` at
the true density breaks CFL and is unconditionally unstable — there is exactly one
`ρ` per run, not two. Fixed `Δt` for v1; a per-step global min is a trivial
extension if densities become heterogeneous.

---

## 5. Quadratic B-spline shape functions

On a uniform grid with spacing `h`, per axis, a particle at coordinate `x` has base
node `b = ⌊x/h − 1/2⌋` and local coordinate `ξ = x/h − b ∈ [1/2, 3/2)`. The three
1-D weights over nodes `b, b+1, b+2` are (standard MLS-MPM / taichi form):

```
d0 = 3/2 − ξ ;   w0 = 1/2 d0²
d1 = ξ − 1   ;   w1 = 3/4 − d1²
d2 = ξ − 1/2 ;   w2 = 1/2 d2²          (note w0+w1+w2 = 1)
```

with derivatives (w.r.t. `x`, so `× 1/h`): `w0' = −d0/h`, `w1' = −2d1/h`,
`w2' = d2/h`. The 3-D weight is the tensor product `wᵢₚ = wx·wy·wz` and its gradient
`∇wᵢₚ = (wx'·wy·wz, wx·wy'·wz, wx·wy·wz')`. The stencil is the 3×3×3 = 27 nodes
`(bx+a, by+b, bz+c)`, `a,b,c ∈ {0,1,2}`.

**Why quadratic B-spline (not linear or GIMP).** C¹ ⇒ no cell-crossing force
spikes; the APIC affine inertia tensor is the constant `Dₚ = ¼h² I` (so
`Cₚ = 4/h² · Bₚ`, no per-particle inverse); and it reduces quadrature error and
eases locking — one basis choice retires three of PROPOSAL §4's failure modes.
This is the deliberate baseline; GIMP/CPDI remain documented upgrades for extreme
stretch/folding not reached by the v1 benchmarks.

**Two different "which cell" questions, intentionally.** The B-spline stencil base
`⌊x/h − 1/2⌋` (above) picks the 27 *nodes* a particle weights against. F-bar's cell
binning (§4.2) uses the plain background-cell index `⌊x/h⌋` to group particles for
volume-averaging. These are different partitions for different purposes — do not
unify them.

---

## 6. Quasi-static loading (how we match implicit FEM)

The `lagrangian` benchmarks are **quasi-static** (no inertia). Explicit MPM is
dynamic, so to reproduce them we must reach a near-equilibrium, rate-independent
response. Three knobs, used together, with an honesty check:

- **Slow, smooth loading.** Drive prescribed boundary velocities with a smooth ramp
  (e.g. a `sin²` ease-in) over enough steps that the loading time ≫ the bar's
  natural period.
- **Mass scaling.** For rate-independent J2 the stress depends on the deformation
  *path*, not the rate, so we may inflate the density used to build each
  particle's mass — **`mₚ = ρ_eff·Vₚ⁰` at particle init, the same `ρ_eff` that
  feeds the §4.3 CFL formula; there is exactly one density per run, not a
  CFL-only fiction.** Raising `ρ_eff` raises the stable `Δt`, reaching the target
  strain in far fewer steps. Scale only as far as the **inertia check** allows.
- **Light damping (v1: one mechanism only).** A single scalar `α_damp` applied to
  `vᵢ*` each step (§4, applied *before* BCs are (re-)imposed) bleeds kinetic energy
  so the path tracks the static equilibrium curve. Dynamic relaxation (adaptive,
  per-node damping) is a documented **reserve** upgrade (§11) if scalar damping
  proves insufficient — not a day-one option; shipping two damping mechanisms
  before either is validated only widens the search space when G3 disagrees with
  FEM (see §15).
- **Inertia check (the gate).** Report `KE / IE` (kinetic / internal energy) each
  step; a run only counts as "quasi-static" where `KE/IE ≲ 1%`. This is a
  first-class diagnostic, printed by the necking/tension examples, not an
  afterthought — it is how we *prove* the comparison to FEM is fair. Note this
  bounds *instantaneous* kinetic energy, not cumulative path deviation from a
  transient overshoot before damping settles; if G3 shows a small systematic
  offset from FEM despite `KE/IE` passing, check for early-transient ringing
  before suspecting the constitutive kernel.

For genuinely dynamic large-deformation cases (Taylor bar, a future example) none
of this applies — the dynamics *are* the physics.

---

## 7. Boundary conditions (on the grid)

BCs are imposed on **grid-node velocities** after the momentum update *and after
damping* (§4, §6 — order matters: damping must never be applied after a BC, or it
silently under-drives every prescribed-velocity node), using the same
predicate-selection UX as `lagrangian` (`on_face`, `select_nodes` analogues on grid
coordinates):

- **Fixed / symmetry plane:** zero the normal (or all) components of `vᵢ*` for nodes
  on the plane. A quarter/eighth model uses symmetry planes exactly as the FEM
  necking bar does (min-x/min-y/min-z rollers).
- **Prescribed velocity (displacement control):** set `vᵢ*` to the ramp velocity on
  a face — the MPM analogue of `prescribe!`. Reaction is recovered as
  `−Σ_{i∈face} fᵢ` (the grid internal force on the driven nodes).
- **Rigid wall / symmetry with separation (dynamic examples):** zero the inward
  normal velocity but allow outward (no-tension release) — needed for a future
  Taylor bar; noted, not built in v1.

Traction/body loads enter as grid forces (`fᵢ += ...`). This mirrors `lagrangian`'s
`fix!`/`prescribe!`/`load!` so example scripts read almost identically.

---

## 8. Data structures

Struct-of-arrays for cache-friendly, allocation-free hot loops (mirrors the
`lagrangian` `StaticArrays` style).

```julia
struct Particles
    x   ::Vector{SVector{3,Float64}}   # position
    v   ::Vector{SVector{3,Float64}}   # velocity
    C   ::Vector{SMatrix{3,3,Float64,9}}  # APIC affine velocity
    F   ::Vector{SMatrix{3,3,Float64,9}}  # deformation gradient
    Cp_inv ::Vector{SVector{6,Float64}}   # plastic state Cᵖ⁻¹  (reused-kernel state)
    εp  ::Vector{SVector{6,Float64}}   # accumulated plastic strain (diagnostic)
    β_ref ::Vector{SVector{6,Float64}} # back-stress, reference frame (kinematic hardening)
    ᾱ   ::Vector{Float64}              # equivalent plastic strain
    τ   ::Vector{SVector{6,Float64}}   # last Kirchhoff stress (for output/force)
    m   ::Vector{Float64}              # mass = ρ_eff·V0 (fixed; ρ_eff is the ONE
                                        # density used for both this and CFL, §6)
    V0  ::Vector{Float64}              # reference volume (fixed)
    J   ::Vector{Float64}              # det F (cached)
end

struct Grid                      # uniform Cartesian, reset every step
    origin ::SVector{3,Float64}
    h      ::Float64
    n      ::NTuple{3,Int}        # nodes per axis
    m  ::Vector{Float64}          # nodal mass          (length prod(n))
    p  ::Vector{SVector{3,Float64}}  # nodal momentum
    f  ::Vector{SVector{3,Float64}}  # nodal force
    v  ::Vector{SVector{3,Float64}}  # nodal velocity
end
```

Node linear index `i = ix + nx*((iy-1) + ny*(iz-1))` (1-based, column-major),
matching Julia's memory order.

**Grid extent and out-of-bounds particles.** The grid is fixed-size (`origin`,
`h`, `n`), sized generously at model-build time to cover the full range of motion
the example expects — this matters especially for the large-deformation flagship
examples (§13), where a bending or necking rod moves far from its initial box. If a
particle's 27-node stencil ever falls outside `[origin, origin + n·h]`, that is a
setup error (grid too small), not a state to silently index around: the P2G/G2P
loop must bounds-check the stencil and raise a typed `ParticleOutOfBoundsError`
(carrying the particle index and position) — the same "loud failure, never
swallowed" philosophy as `ParticleInversionError` (§3.3). Example authors pad the
grid (a multiple of the expected displacement) rather than the solver silently
tolerating an escaped particle.

**Particle sampling.** Fill a geometry (box, cylinder, or a predicate region) with
`PPC = 2³ = 8` particles per cell (baseline), each carrying `Vₚ⁰ = h³/PPC`,
`mₚ = ρ_eff Vₚ⁰` (§6 — the one, possibly-scaled, density for the run). This is the
"mesh generator" analogue; a `sample_box` and a generic `sample_region(pred)` cover
every v1 example.

**Single-threaded v1.** The P2G scatter (multiple particles writing the same grid
node) is not thread-safe as written; v1 is single-threaded. Parallelizing P2G is a
documented future extension (coloring or atomics), not a v1 concern.

---

## 9. Module layout & public API

Small, flat, mirrors `lagrangian` (no premature abstraction):

```
particle_method/
  Project.toml            # [sources] path-dep on ../lagrangian (§10)
  src/
    ParticlePlasticity.jl # top module: includes + curated exports
    Grid.jl               # uniform grid + quadratic B-spline weights/gradients
    Particles.jl          # SoA container + sampling (sample_box, sample_region)
    Transfer.jl           # p2g!, g2p!  (APIC)
    Constitutive.jl       # particle_stress_update!  (F-update + reused kernel + J-guard + F̄)
    BoundaryConditions.jl # grid predicates + apply_bcs!  (fix/symmetry/prescribe)
    Step.jl               # step!, run!  (orchestration, mass scaling, damping, KE/IE diagnostics)
    Visualization.jl      # write_particles_vtu  (VTK point cloud)
  test/
    runtests.jl + unit/validation tests (§12)
  examples/
    bent_rod.jl, necking_extreme.jl, tension_block.jl, ...
  docs/  PROPOSAL.md, DESIGN.md
```

Curated public exports (keep it as small as `lagrangian`'s):
`Grid, Particles, sample_box, sample_region, J2Material` (re-export),
`MPMModel, fix!, symmetry!, prescribe!, gravity!, run!, step!`,
`equivalent_plastic_strain, particle_cauchy, kinetic_energy, write_particles_vtu`.

A thin `MPMModel` bundles `grid + particles + material + BC list + settings`
(`Δt`, `fbar`, `damping`, `mass_scale`) — the single object examples drive,
analogous to `lagrangian`'s `Model`.

**Dispatch style: runtime flags, not `Val`.** `fbar` and any future F-update
option are plain `Bool`/`enum` fields on `MPMModel`, branched on inside `step!` —
**not** `Val`-dispatched compile-time specialization. The branch sits once per step
outside the 27-node P2G/G2P inner loops, where its cost is immaterial next to the
constitutive kernel call; `Val`-dispatch would only earn its complexity if
profiling showed otherwise, which is not expected and not worth designing for
up front.

---

## 10. Reusing the kernel without duplication

`particle_method` is a **separate Julia package** that path-depends on
`PlasticityFEM`, so the constitutive kernel has one source of truth (no copy, no
drift). Using Julia ≥ 1.11 `[sources]`:

```toml
# particle_method/Project.toml
name = "ParticlePlasticity"
[deps]
PlasticityFEM = "4f518d9e-bd7c-43b7-aad4-7f44139b9b61"
StaticArrays  = "90137ffa-7385-5640-81b9-e52037218182"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
[compat]
julia = "1.11"
[sources]
PlasticityFEM = { path = "../lagrangian" }
```

`[sources]` requires Julia ≥ 1.11 — note `particle_method`'s `[compat]` floor is
**stricter** than `lagrangian`'s own (`julia = "1.10"`, `lagrangian/Project.toml`),
since only `particle_method` uses this mechanism; pin it explicitly so a 1.10
contributor gets a clear resolver message rather than a confusing failure.

Then `import PlasticityFEM.FiniteStrain: finite_kinematics, finite_stress_update,
FiniteKin, voigt_to_sym3, sym3_to_voigt` and `import PlasticityFEM.Materials:
J2Material, return_map`. `Pkg.instantiate()` resolves the path with no committed
`Manifest.toml` (Manifests stay git-ignored — this differs from `lagrangian`, which
does commit one; `particle_method`'s path-dependency makes an environment-specific
Manifest less portable to commit, so the two packages deliberately follow different
conventions here). No change to `lagrangian` is required — the submodules are
reachable as `PlasticityFEM.FiniteStrain` / `PlasticityFEM.Materials`.

---

## 11. Numerical safeguards (baseline → reserve)

Per PROPOSAL §4, resolved to concrete v1 choices:

| Failure mode | v1 choice | Reserve |
|---|---|---|
| Cell-crossing noise | quadratic B-spline (C¹) | GIMP/CPDI |
| PIC dissipation / FLIP noise | APIC | MLS-MPM |
| Ringing / null-space | APIC suppresses; monitor | local-SVD null-space filter |
| Volumetric locking (isochoric J2) | **F̄ at particles** (§4.2) | higher-order B-spline; mixed u–p |
| `J ≤ 0` at a particle | typed `ParticleInversionError` (§3.3) | particle splitting |
| Particle escapes grid extent | typed `ParticleOutOfBoundsError` (§8) | adaptive/growing grid |
| Empty/low-mass grid node | skip nodes with `mᵢ ≤ mₜₒₗ` | — |
| F-update volume drift | total-F form; **measure via a dedicated drift test (Phase 2)** before adding a cure (§4.1) | exp/mid-point (Hughes–Winget) update, if measured drift demands it |
| Quasi-static energy bleed | scalar viscous damping (§6) | dynamic relaxation |
| Explicit instability | CFL `Δt` (§4.3), single density for `Δt` and `mₚ` (§6) | precise Δt (Ni & Zhang 2020) |

---

## 12. Verification & validation (the whole point — gated, against `lagrangian`)

Each gate must pass before the next. Tolerances are explicit. "vs FEM" means the
`lagrangian` solver run on the matching problem.

- **G0 — B-spline partition of unity.** `Σᵢ wᵢₚ = 1` and `Σᵢ ∇wᵢₚ = 0` for random
  particle positions (to `1e-12`); linear field reproduction `Σᵢ wᵢₚ xᵢ = xₚ`. Pure
  unit test, no physics.

- **G1 — single material point == FEM kernel (EXACT).** Drive one particle through a
  prescribed homogeneous `F(t)` (uniaxial, then simple shear to large γ) and call
  the *same* `finite_stress_update`. Assert the stress path equals a direct call of
  the `lagrangian` kernel on the same `F` **to machine precision** (it is literally
  the same function) and `det Fᵖ = 1` throughout. This nails the reuse. The
  simple-shear-to-large-γ case doubles as the objectivity / no-Jaumann-oscillation
  check. This variant prescribes `F(t)` directly (bypassing the MPM integrator) —
  it isolates the *kernel*, not the F-update.

- **G1b — F-update volumetric drift (isolates §4.1, not bundled into G1).** Drive
  one particle through many steps of prescribed pure shear/rotation through the
  *actual MPM forward-Euler F-update loop* (`Fₚ ← (I+ΔtLₚ)Fₚ`, not a prescribed
  `F(t)`) and plot `det Fₚ` vs. step count. This produces the number that decides
  whether the exp/mid-point update (§4.1) is ever needed — a measurement, not a
  guess — and localizes any drift to the F-update specifically rather than letting
  it hide inside G6/G7's full necking runs.

- **G2 — vibrating / stretched elastic bar.** Elastic-only 1-D bar: standing-wave
  frequency and energy behavior vs analytic; confirms P2G/G2P, APIC low
  dissipation, CFL. Also a spinning elastic body to *verify* APIC angular-momentum
  conservation (not assume it).

- **G3 — homogeneous tension/compression block vs FEM (TIGHT).** A cube pulled
  uniformly (symmetry rollers + prescribed face velocity, quasi-static per §6).
  Because deformation is homogeneous, every particle sees the same `F` as every FEM
  Gauss point at the same nominal strain, so **stress and reaction must match FEM to
  ~1e-3 (relative)** once `KE/IE ≲ 1%`. This is the first true cross-solver check
  and isolates transfers + loading from discretization error.

- **G4 — F̄ anti-locking gate.** Fully-plastic compression of the block: with
  `fbar=false` expect checkerboard pressure; with `fbar=true` the pressure field is
  smooth and the mean stress matches FEM's F-bar result. Must pass before any
  necking result is trusted.

- **G5 — bent cantilever vs FEM (finite strain, moderate).** The
  `lagrangian/examples/finite_large_rotation_cantilever.jl` geometry/material under a
  transverse tip load or prescribed rotation, into the finite-strain (large-ish
  rotation) regime both solvers handle. Compare **tip deflection within a few %** and
  the qualitative bent shape / plastic-hinge location. Discretization differs
  (particles vs mesh), so this is an agreement-in-trend + integrated-quantity check,
  not machine precision — the tolerance is stated per run and justified.

- **G6 — moderate necking vs FEM.** Reproduce
  `lagrangian/examples/finite_necking_bar.jl` (10×1×1 bar, mid-length imperfection,
  steel, symmetry rollers, ~2.5% elongation): confirm the neck localizes at
  mid-length (thinnest section and `ᾱ` peak at `x≈L/2`, as the FEM diagnostic
  reports) and compare neck contraction and peak `ᾱ` to FEM within a stated
  tolerance. This is goal (1) fully met.

- **G7 — beyond FEM (large deformation, MPM-only).** Push the necking bar past the
  elongation where the FEM solver raises `ElementInversionError` / diverges; show
  MPM continues to a deep neck (large area reduction). No FEM reference exists here —
  the check is qualitative correctness (monotone reaction drop, plastic strain
  concentrating in the neck, `det Fᵖ=1` maintained, no blow-up) plus energy
  bookkeeping. This is goal (2).

Unit tests additionally cover: grid indexing round-trip, particle→cell binning,
`ParticleInversionError` on a deliberately inverted `F`, APIC exactness on affine
velocity fields, and F̄ dilation algebra (`det F̄ₚ = J̄_c`).

---

## 13. Examples (deliverables)

1. **`tension_block.jl`** — the G3 cross-check, kept as a runnable example (prints
   the `KE/IE` diagnostic and the MPM-vs-FEM stress).
2. **`bent_rod.jl`** — a slender rod clamped at one end, bent through large rotation
   by a prescribed tip motion or gravity, into the plastic hinge regime. ParaView:
   warp by displacement, color by `EqPlasticStrain`. Demonstrates large-rotation
   objectivity on a real structure.
3. **`necking_extreme.jl`** — the G6→G7 stretched rod: start matching the FEM
   necking bar, then keep pulling to **extreme necking** (deep area reduction toward
   rupture) — the flagship demonstration of the regime FEM cannot reach. Uses
   `fbar=true`; adds the exp/mid-point F-update only if G1b's measured drift over
   this example's step count warrants it. Prints neck-radius history and `KE/IE`.

Each example mirrors the terse, self-documenting style of the `lagrangian`
examples (geometry, material, BCs, run, a printed diagnostic, a `.vtu`).

---

## 14. Development phases (build order = validation order)

- **Phase 0 — package scaffold + reuse hook.** `Project.toml` with `[sources]`,
  module skeleton, `import` the kernel, G0/G1 pass. *Decision locked here:*
  **full 3D** (matches the FEM box benchmarks and the requested examples) and
  quasi-static via mass-scaling+damping (§6) — both architectural, chosen now.
- **Phase 1 — elastodynamics.** Grid, B-spline, APIC (`Lₚ = Cₚ` for the F-update,
  §4), explicit step with the corrected damping-then-BC order (§4, §7), elastic
  material path. Gate: **G2**.
- **Phase 2 — plug in J2 (reuse).** `Constitutive.jl` calls the kernel; J-guard;
  `ParticleOutOfBoundsError` guard; state init. Gates: **G1, G1b, G3**. G1b's
  measured drift decides whether Phase 5 needs the exp/mid-point F-update.
- **Phase 3 — F̄ + quasi-static machinery.** Anti-locking, mass scaling (single
  `ρ_eff` for `Δt` and `mₚ`, §6), scalar damping, KE/IE diagnostic. Gate: **G4**.
- **Phase 4 — validation vs FEM.** Gates **G5, G6** (goal 1 met).
- **Phase 5 — large deformation.** Gate **G7**; examples §13 (goal 2 met). Add the
  exp/mid-point F-update here only if G1b showed it's warranted.

---

## 15. Risks & explicit decisions

- **Quasi-static match is the subtle risk**, not the physics. If G3/G6 disagree with
  FEM it is almost certainly the loading/inertia/damping, not the (reused, verified)
  stress update — the `KE/IE` gate and G1 exactness are designed to localize blame
  there. Root-cause, don't paper over with fudged tolerances. (An earlier draft of
  this design had a latent self-inflicted instance of exactly this failure mode — a
  damping-before-BC ordering bug, §4/§7 — that would have under-driven every
  prescribed-velocity boundary and shown up as a spurious "G3 disagrees with FEM."
  Fixed at the spec level before any code was written.)
- **F̄-in-MPM correctness** is the top numerical risk for metal J2 — G4 gates it
  before any necking claim.
- **Explicit-only** means quasi-static necking costs many steps even with mass
  scaling. Accepted for v1; implicit MPM is the documented escape hatch if step
  counts become impractical.
- **3D chosen over axisymmetric.** More particles, but a *direct* match to the FEM
  box benchmarks and to the bent-rod/necking examples, and no separate axisymmetric
  kinematics to verify. This is the deliberate simplicity choice.
- **Linear hardening only in v1.** The published Simo necking *curve* (PROPOSAL §8
  V3b) needs saturation hardening = a local Newton in `return_map` — the first
  post-v1 kernel upgrade, out of scope here; G6 uses the FEM's linear-hardening
  material so the cross-check is apples-to-apples.

**First implementation step:** Phase 0 — stand up the package with the `[sources]`
reuse hook and pass G0 (B-spline partition of unity) and G1 (single-point ==
kernel), proving the reuse before any grid machinery exists.
