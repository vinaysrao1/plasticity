# particle_method — Development Proposal

**J2 / von Mises elastoplasticity under very large deformation with a particle
method (Material Point Method).**

Status: proposal (research synthesis + development plan). No code yet. This
document reviews the most relevant and highest-cited literature and proposes a
concrete way to build the code, reusing the verified constitutive core of the
[`../lagrangian/`](../lagrangian/) finite element solver.

**Revision note.** This proposal was hardened after an adversarial review by a
computational-plasticity reviewer. The central bet — reuse of the verified
log-strain radial-return kernel, *stress-only*, in explicit MPM — was **confirmed
sound and verified in the code**. The review corrected several items now folded in:
(i) the Simo necking benchmark needs *saturation* hardening the current kernel
lacks, and adding it is a real change (§8 V3, §2.3, §10); (ii) APIC makes MUSL
redundant — use a single transfer (§0, §3, §4); (iii) prefer a **B-spline** grid
basis with APIC over GIMP (§0, §4); (iv) the forward F-update is only first-order
and not exactly isochoric (§2.1); (v) per-particle J≤0 guards, particle-state
initialization, and grid contact/rigid-walls for Taylor/upsetting are on the
critical path (§2.3, §3, §7).

---

## 0. Executive summary

**Method.** Use the **Material Point Method (MPM)**. For confined, isochoric,
history-dependent *metal* plasticity at very large deformation (necking to
fracture, upsetting/forging, Taylor-bar impact), MPM is the mainstream
particle/meshfree choice: the background grid is reset every step, so there is no
mesh entanglement — the exact failure mode that stops the Lagrangian FEM solver —
while plastic history rides on the particles. SPH (specifically Total-Lagrangian
SPH) is the credible alternative but is generally less robust for confined
isochoric metal flow (tensile instability, hourglassing); it wins mainly for
fracture/fragmentation and fluid coupling. (de Vaucorbeil et al. 2020; Sulsky et
al. 1994.)

**The key reuse win.** The recommended per-particle stress update for metals is
the **multiplicative, logarithmic (Hencky) elastic strain, exponential-map return
mapping** of Simo (1992) — which is *algebraically identical to the small-strain
radial return performed in principal log-strain space*. **This is exactly what
`lagrangian/src/FiniteStrain.jl` already implements and verifies**
(`finite_kinematics(F, Cᵖ⁻¹)` → `return_map` → Kirchhoff τ + updated `Cᵖ⁻¹`).
Because explicit MPM needs only the **stress** (no global tangent — there is no
global linear solve), we reuse the *stress* half of that kernel almost verbatim
and skip the two-point `∂P/∂F` tangent entirely. The hard, verified physics is
already ours; MPM only changes the *spatial discretization*.

**Recommended baseline stack** (build incrementally; defer the rest):

| Concern | Baseline (build first) | Defer / upgrade to |
|---|---|---|
| Method | Explicit MPM | implicit/quasi-static MPM (Guilkey–Weiss 2003; iGIMP) |
| Basis / shape functions | **quadratic B-spline** (also removes cell-crossing, cuts quadrature error, eases locking) | GIMP as alternative; **CPDI/CPDI2** for extreme stretch/folding |
| Particle↔grid transfer | **APIC** (affine PIC), a *single* symplectic transfer | MLS-MPM; PolyPIC |
| Stress-update ordering | USF/USL *timing* with the single APIC transfer | MUSL only on a fallback PIC/FLIP (non-APIC) path |
| Constitutive update | reuse `lagrangian` log-strain radial return — **linear** hardening | **saturation/nonlinear hardening (local Newton on Δγ)** for the true Simo necking curve; kinematic hardening |
| Volumetric locking | cell/patch-averaged **F̄ at particles** (Coombs 2018) | higher-order B-spline; nodal-pressure smoothing; mixed u–p |
| Time step | CFL `Δt ≤ C·h/c`, `c=√((K+4G/3)/ρ)` (elastic P-wave; conservative under plastic softening) | precise Δt (Ni & Zhang 2020) |
| Dimensions | **decide in Phase 0**: axisymmetric (natural for necking/Taylor) vs full-3D w/ symmetry-plane grid BCs | — |

**Build approach.** Two viable routes (§6): (A) write our **own minimal explicit
MPM** from scratch, reusing our constitutive kernel directly — matches this
project's ethos, maximizes reuse of the verified core, and the MPM discretization
around it is small; or (B) build on **Tesserae.jl** (MIT; GIMP/CPDI/MLS/APIC +
CUDA/Metal + AD tangents) to get advanced features immediately. **Recommendation:
start with (A)** to reach a validated Taylor-bar/necking result with full control
and clean, tested code, then adopt Tesserae for CPDI/GPU if/when needed.

---

## 1. Why MPM (method landscape)

MPM (Sulsky, Chen & Schreyer 1994; Sulsky, Zhou & Schreyer 1995) carries mass,
momentum and constitutive **history** on material points that move through a fixed
background grid used only as a scratch space to solve momentum each step. It
combines the Lagrangian advantage (history stays on particles — no advection) with
the Eulerian advantage (no mesh distortion). This is why it dominates the
very-large-deformation metal-plasticity literature (Taylor impact, forging).

**Flavors and what each fixes** (all directly relevant):
- **GIMP** (Bardenhagen & Kober 2004): particle characteristic functions give C¹
  effective shape functions → removes *cell-crossing noise* (the spurious
  force spikes from the C⁰ linear grid gradient when a particle crosses a cell).
- **CPDI / CPDI2** (CPDI: Sadeghirad, Brannon & Burghardt 2011; CPDI2:
  Sadeghirad, Brannon & **Guilkey** 2013): particle domains *convect* with F →
  tracks massive stretch/rotation where GIMP domains degrade. The enabler for
  extreme-deformation MPM.
- **APIC** (Jiang, Schroeder, Selle, Teran & Stomakhin 2015; JCP 2017): an affine
  per-particle velocity mode → recovers FLIP's low dissipation *without* its noise
  and **conserves linear and angular momentum** with a lumped mass. Resolves the
  classic PIC-dissipation / FLIP-noise dilemma (Brackbill & Ruppel 1986/88).
- **MLS-MPM** (Hu et al. 2018): MLS weak-form recast unifying/accelerating APIC
  (~2× faster); the modern high-throughput choice.
- **Snow MPM** (Stomakhin et al. 2013): the landmark finite-strain *elastoplastic*
  MPM (multiplicative F = Fᵉ·Fᵖ, return mapping) — the template we follow.

**SPH** for solids (Gray, Monaghan & Swift 2001; Total-Lagrangian SPH, Vignjevic
et al. 2006) needs explicit stabilization for the *tensile instability* (Swegle et
al. 1995) and is generally less robust than MPM for confined isochoric metal flow
— hence MPM is our choice, with SPH noted as the route if fracture/fragmentation
later dominates.

---

## 2. The constitutive core — reuse `lagrangian`, almost verbatim

This is the heart of the reuse case, so it is spelled out precisely.

### 2.1 What the literature says to do at a particle
For metals, current best practice is **Algorithm B** (Simo 1992; Weber & Anand
1990; Eterovic & Bathe 1990; Cuitiño & Ortiz 1992), preferred over the older
hypoelastic Jaumann-rate + radial return (Belytschko–Liu–Moran) because it is
*exactly* incrementally objective (no Jaumann spurious-shear-stress artifact),
*exactly* isochoric (exp of a traceless update has det 1), and reduces the J2
return map to the identical small-strain algebra. Per particle, per step:

```
1. F update:          Fₙ₊₁ = (I + Δt ∇v) Fₙ            (∇v from the grid, §3)
2. elastic trial:     bᵉ_tr = Fₙ₊₁ · Cᵖ⁻¹ₙ · Fₙ₊₁ᵀ      (or f·bᵉₙ·fᵀ, f = I+Δt∇v)
3. spectral log:      εᵉ_tr = ½ ln bᵉ_tr  (principal Hencky strain)
4. radial return:     small-strain J2 return map on εᵉ_tr → τ (Kirchhoff), Δεᵖ
5. exp-map update:    bᵉₙ₊₁ = Σ exp(2εᵉ_A) n_A⊗n_A ;  Cᵖ⁻¹ₙ₊₁ = F⁻¹ bᵉₙ₊₁ F⁻ᵀ
6. Cauchy stress:     σ = τ / J,   J = det Fₙ₊₁
```

**On step 1 — accuracy vs objectivity (do not conflate them).** Steps 2–6 are
*exactly* incrementally objective *given* F — but that is a property of the stress
map, not of the F-update. The forward form `Fₙ₊₁ = (I+Δt∇v)Fₙ` is only first-order,
and for isochoric plastic flow (`tr ∇v = 0`) `det(I+Δt∇v) ≠ 1`, so it injects an
O(Δt²) volume error that *accumulates* over the many thousands of explicit steps —
exactly where locking/pressure is already delicate. Prefer the **total-F** storage
form `bᵉ_tr = Fₙ₊₁ Cᵖ⁻¹ₙ Fₙ₊₁ᵀ` (as the kernel does) over the incremental
`f·bᵉₙ·fᵀ`, and offer an exponential / mid-point (Hughes–Winget) F-update as the
accurate option for quasi-static, many-step cases (necking).

### 2.2 What we already have
`lagrangian/src/FiniteStrain.jl` implements steps 2–5 exactly:
`finite_kinematics(F, Cᵖ⁻¹)` does 2–3, `return_map` (from `Materials.jl`) does 4,
`finite_stress_update` does 5 and returns `τ_voigt`, the updated `Cᵖ⁻¹`, `ᾱ`, and
the back-stress — **already storing `Cp_inv` per integration point** (which becomes
per *particle*). It is allocation-free, `StaticArrays`-based, and FD-verified.

### 2.3 What changes for MPM (small)
- **No tangent.** Explicit MPM assembles no global stiffness, so we do **not**
  need `dPdF`/`spatial_modulus` — only `τ`/`σ`. This *removes* the most complex
  part of the finite-strain kernel from the hot path.
- **Total-F form is compatible.** `finite_kinematics` takes the *total* `F` and
  `Cᵖ⁻¹`; MPM carries total `F` and `Cᵖ⁻¹` per particle, so the call is direct.
- **F̄ for locking must be re-derived for MPM.** `lagrangian` F-bar uses the
  *element-centroid* J₀; MPM F̄ uses a *cell-averaged* J (Coombs et al. 2018).
  Same idea, different averaging operator — a small new piece, not a reuse.
- **Cauchy reporting** (`σ = τ/J`) and internal force (§3) are the MPM-side glue.
- **Per-particle J≤0 guard is the MPM loop's job.** The `kin.ok` check lives in
  the FEM *assembly* (`lagrangian/src/Elements.jl:447`), **not** in the kernel: if
  `finite_kinematics` fails, `finite_stress_update` silently returns *zero stress*
  and garbage `Cᵖ⁻¹`. The particle update must replicate that guard. MPM removes
  *grid* entanglement, but each particle still integrates its own F and can still
  hit J≤0 under extreme stretch — hence CPDI / particle splitting / J-guards.
- **Particle state must be initialized:** `Cᵖ⁻¹ = I` (Voigt `[1,1,1,0,0,0]`),
  `F = I`, `ᾱ = 0`, `β = 0`; a zero `Cᵖ⁻¹` makes `bᵉ_tr = 0` and the log singular.
- **Call-once-then-commit.** The FEM calls the kernel on *trial* F each Newton
  iteration, committing `Cᵖ⁻¹` only at convergence; explicit MPM calls once per
  step from the previous committed `Cᵖ⁻¹ₙ` and stores the returned `Cᵖ⁻¹ₙ₊₁`.
- **Decide which F feeds the F̄ pull-back.** Under F̄, the exp-map step 5 uses
  `F⁻¹`; choose consistently whether that is the modified `F̄` or the true `F`
  (the FEM feeds `F̄` to *both* kinematics and stress update, `Elements.jl:437–464`).

**Consequence:** the verification we already trust (small-displacement limit,
`det Fᵖ = 1`, objectivity) transfers with the kernel; a particle-level single-point
test is byte-for-byte the small-strain radial-return check.

---

## 3. The explicit MPM step (algorithm skeleton)

Per time step (single APIC transfer; USF/USL stress-update timing):

```
P2G  (particles → grid):
  mᵢ      = Σₚ Sᵢ(xₚ) mₚ                                   (lumped mass)
  (mv)ᵢ   = Σₚ Sᵢ(xₚ) mₚ (vₚ + Cₚ(xᵢ − xₚ))                (APIC affine transfer)
  fᵢⁱⁿᵗ  = − Σₚ Vₚ σₚ ∇Sᵢ(xₚ)                              (internal force, current V)
  fᵢᵉˣᵗ  = body/traction loads
Grid solve (explicit):
  vᵢ*     = vᵢ + Δt (fᵢⁱⁿᵗ + fᵢᵉˣᵗ) / mᵢ                    (+ BCs on grid nodes)
G2P  (grid → particles):
  vₚ      ← Σᵢ Sᵢ vᵢ*        (PIC part)   ;  Cₚ ← APIC affine reconstruction
  xₚ     += Δt Σᵢ Sᵢ vᵢ*
  ∇vₚ     = Σᵢ vᵢ* ⊗ ∇Sᵢ(xₚ)   (single APIC transfer — no MUSL second remap)
Update (per particle, §2):
  Fₚ     ← (I + Δt ∇vₚ) Fₚ ;  (σₚ, Cᵖ⁻¹ₚ, ᾱₚ) ← finite stress update ;  Vₚ = Jₚ Vₚ⁰
Reset the grid.
```

- **Stress measure / internal force** (de Vaucorbeil 2020; Jiang et al. 2016
  course): `σₚ` is Cauchy; `fᵢⁱⁿᵗ = −Σₚ Vₚ σₚ ∇Sᵢ` is the discrete `∫σ:∇S dV`.
- **BCs** are imposed on grid nodes (velocity/traction) — the same predicate-style
  UX as `lagrangian` can wrap this.
- **Contact / rigid walls are separate machinery, on the critical path for V4/V5.**
  Taylor impact needs a symmetry-plane wall with a *no-tension separation* release
  (else the bar sticks); upsetting needs die contact (friction). Plain nodal
  velocity BCs are **not** sufficient — use MPM nodal multi-body contact
  (Bardenhagen et al. 2001) or an explicit rigid-wall algorithm; Sulsky & Schreyer
  (1996), cited below, did exactly this through the grid.

---

## 4. Numerical safeguards (the choices, and why)

MPM's known failure modes and the accepted cures (Bardenhagen 2002; Steffen et al.
2008; de Vaucorbeil 2020) — pick the baseline, keep the upgrade in reserve:

- **Cell-crossing noise →** **quadratic B-spline basis** (baseline; C¹, and it
  simultaneously cuts quadrature error and eases locking — one change, three
  fixes); GIMP is the alternative; **CPDI/CPDI2** for extreme stretch/folding.
- **Ringing / null-space instability →** APIC transfers largely suppress it; a
  local-SVD **null-space filter** (Gritton et al. 2017) is the fallback. (APIC's
  affine moment matrix `Dₚ` is analytic/constant for B-splines — another reason to
  prefer B-spline over GIMP, where `Dₚ` must be assembled and inverted per particle.)
- **Quadrature error (particle clustering) →** covered by the **B-spline** grid
  basis baseline (Steffen, Kirby & Berzins 2008).
- **PIC dissipation vs FLIP noise →** **APIC** (the resolution) — and because APIC
  already resolves this and conserves angular momentum, do **not** also stack MUSL.
- **Volumetric locking (isochoric J2) →** cell/patch-averaged **F̄ at particles**
  (Coombs et al. 2018); the single most important numerical caveat for metal J2 in
  MPM — plain linear MPM/GIMP *will* lock and show checkerboard pressure. Weigh
  higher-order B-spline MPM and nodal-pressure smoothing as alternatives/complements
  (mixed u–p is the heavier fallback); F̄ is the baseline, not the only option.
- **Time step →** explicit CFL `Δt ≤ C·h/c`, `c = √((K+4G/3)/ρ)` (elastic P-wave,
  the fastest/correct speed; conservative since the plastic tangent speed is lower),
  `C ≈ 0.1–0.5`, global min over particles; refine with Ni & Zhang (2020) if needed.
- **Stress-update ordering →** with **APIC**, use a *single* transfer: MUSL's
  second remap is redundant and would re-average the affine state APIC exists to
  preserve. Choose USF vs USL *timing* only (Bardenhagen 2002); keep **MUSL** solely
  for a fallback PIC/FLIP (non-APIC) path.

---

## 5. Build approach: our own MPM vs Tesserae.jl

**Julia ecosystem (all MIT unless noted):**

| Package | Role | Notes |
|---|---|---|
| **Tesserae.jl** (K. Nakamura) | scaffold to *build on* | tensor-native constitutive interface (`Tensorial.jl`) with **AD tangents**; B-spline/GIMP/CPDI/MLS; `@P2G`/`@G2P`; multithread + **CUDA/Metal**. Neo-Hookean example shipped; J2 is a natural extension you author. |
| **MaterialPointSolver.jl** (LandslideSIM) | turnkey solver | backend-agnostic GPU (KernelAbstractions: CUDA/ROCm/Metal/oneAPI); USL/USF/MUSL, GIMP/MLS. Native models are Drucker–Prager/Mohr–Coulomb (geomechanics) — J2 must be added. |
| `taichi_mpm` (C++/Taichi) | compact reference | the 88-line MLS-MPM (MIT) — best terse APIC/return-map reference. |
| CB-Geo `mpm` (C++) | architecture reference | clean modular MIT MPM. |
| Karamelo (C++) | algorithm reference | finite-strain elastoplastic MPM, but **GPL** — read, don't copy into a permissive project. |

**Recommendation: Route A — our own minimal explicit MPM**, for the first
validated milestone, because:
1. the hard part (the constitutive kernel) is already ours and verified — the MPM
   layer around it (P2G/grid/G2P/F-update, GIMP shapes, APIC) is a few hundred
   lines and educational;
2. it keeps the clean, tested, `StaticArrays`, predicate-BC style of `lagrangian`
   and lets us reuse `return_map`/`finite_kinematics`, the Voigt conventions, and
   the `.vtu` writer directly;
3. full control over the exact algorithm we validate.
Then **evaluate Tesserae.jl** once physics is validated, if we want CPDI + GPU
without writing them ourselves. (Route B — start on Tesserae — is the pragmatic
choice if GPU/CPDI are needed on day one; the cost is adapting our Voigt kernel to
Tesserae's `Tensorial.jl` tensor interface and learning the framework.)

---

## 6. Data structures (sketch, mirroring `lagrangian` style)

```
Particles (struct-of-arrays, one entry per material point):
  x, v          position, velocity        (Vec3 / 3×np)
  m, V0, V      mass, reference & current volume
  F             deformation gradient        (3×3 per particle; SMatrix)
  Cp_inv        plastic state Cᵖ⁻¹          (6-Voigt per particle) — reused kernel state
  ᾱ, β_ref      eq. plastic strain, back-stress (reused history)
  C             APIC affine velocity matrix (3×3)
Grid (fixed background, reset each step):
  regular Cartesian nodes; lumped mass mᵢ, momentum, force; GIMP/B-spline weights.
Material: reuse J2Material (E, ν, σy0, Hiso, Hkin) verbatim.
```

Particle sampling (fill a geometry with material points, **baseline 8 PPC = 2³ per
cell** in 3D) is the new "mesh generator" analogue; `MaterialPointGenerator.jl` is a
reference. **Initialize** `F = I`, `Cᵖ⁻¹ = I` (`[1,1,1,0,0,0]`), `ᾱ = 0`, `β = 0`,
`V = V₀`, `C = 0` at t=0 (§2.3).

---

## 7. Development phases (each gated by a benchmark)

- **Phase 0 — design note + architectural decisions.** Pin the baseline (this doc),
  lay out the package (`ParticlePlasticity`?), particle/grid structs, reuse hooks
  into `lagrangian` kernels, `.vtu` output. **Decide now, not later:** axisymmetric
  vs full-3D discretization (dictates hoop-strain/`1/r` weighting and BCs), and the
  quasi-static loading strategy — mass scaling / dynamic relaxation with a
  KE ≪ internal-energy check — for necking. These are architectural, not add-ons.
- **Phase 1 — explicit elastodynamics.** B-spline shapes, a single APIC transfer,
  Neo-Hookean or St-Venant elastic. **Gate:** vibrating-bar dispersion/energy vs
  analytic, with explicit energy/momentum monitors (Bardenhagen 2002; Steffen et
  al. 2008) — and a spinning-body test to *verify* APIC angular-momentum conservation.
- **Phase 2 — plug in J2.** Reuse `finite_kinematics`+`return_map` at particles;
  add the per-particle J≤0 guard and state init (§2.3). **Gate:** single-particle
  uniaxial return-map == the small-strain closed form (exact), `det Fᵖ = 1`, **and**
  a large-simple-shear test showing no Jaumann-type stress oscillation (stresses
  the objectivity claim *and* the F-update accuracy of §2.1).
- **Phase 3 — anti-locking + robustness.** F̄-at-particles (Coombs 2018); confirm
  APIC/GIMP suppress ringing/cell-crossing. **Gate:** no checkerboard pressure in
  fully-plastic flow.
- **Phase 4 — large-deformation validation.** The benchmarks in §8; cross-check
  against the `lagrangian` FEM solver where they overlap (moderate necking), then
  push past where the FEM mesh inverts. **Prerequisite:** V3b (the true Simo curve)
  needs saturation hardening (local Newton on Δγ) added to `return_map` first — a
  real kernel upgrade, scheduled here, not in Phase 5. V3a (linear hardening) and
  V4 (Taylor, perfectly plastic) need only the current kernel.
- **Phase 5 (optional) — CPDI and/or GPU**, or migrate to Tesserae.jl.

---

## 8. Verification & validation benchmarks (with reference data)

- **V1 — single material point, uniaxial J2.** Exact vs the small-strain radial
  return closed form (this is the `lagrangian` T2 test at a particle). *Exact.*
- **V2 — elastic vibrating bar.** Standing-wave dispersion + energy behavior;
  distinguishes USF (energy-conserving) from USL/MUSL (dissipative), per
  Bardenhagen (2002).
- **V3 — necking of a round bar** (finite-strain J2). Geometry (both variants):
  **R₀ = 6.413 mm, L₀ = 53.334 mm**, ~1.8% mid-length radius imperfection
  (0.982 R₀), total elongation **14 mm**; compare reaction–elongation and neck-radius
  reduction. **Two distinct materials — do not conflate them:**
  - **V3a (baseline, linear hardening).** Cross-check against the FEM solver using
    its *linear* constants (`lagrangian/examples/necking_titanium_bar.jl`,
    "linearized hardening"). This the reused kernel does exactly, today.
  - **V3b (the published Simo curve — deferred to the hardening upgrade).**
    **E = 206.9 GPa, ν = 0.29, σ_y0 = 450 MPa** with **saturation hardening
    σ_y(ᾱ) = σ_y0 + (σ_∞ − σ_y0)(1 − e^{−δᾱ}) + Hᾱ,  σ_∞ = 715 MPa, δ = 16.93,
    H = 129.24 MPa** (Simo 1988; Simo & Armero 1992 — *verify vs primary source*).
    This requires adding nonlinear hardening to `return_map` (a **local Newton on
    Δγ**, replacing the closed form) — a real kernel change, not free reuse.
- **V4 — Taylor bar impact** (canonical large-strain J2 dynamic test — a
  *verification* / code-to-code benchmark, **not** physical validation).
  **Case A (Abaqus/Wilkins–Guinan):** copper, **L₀ = 32.4 mm, R₀ = 3.2 mm,
  v = 227 m/s, E = 110 GPa, ν = 0.3, ρ = 8970 kg/m³, σ_y = 314 MPa (elastic–
  perfectly plastic)**; reference deformed **final length ≈ 21.4–21.5 mm**,
  **mushroom foot radius ≈ 7 mm** (confirm against the primary Wilkins–Guinan 1973
  table). "Pass" means *agreement with other perfectly-plastic solvers* — physical
  copper at 227 m/s strain-hardens and heats (Johnson–Cook + thermal coupling, out
  of baseline scope). Requires the grid rigid-wall/contact of §3. MPM references:
  Sulsky & Schreyer (1996); Ma, Zhang & Huang (2009) — *verify Ma et al. vs source*.
- **V5 — cylinder upsetting** (forging). Validated in Sulsky & Schreyer (1996);
  demonstrates die/workpiece contact via the grid — a capability the FEM solver
  lacks.

The Taylor bar and upsetting are chosen deliberately: they are exactly the
mesh-entangling problems the Lagrangian solver *cannot* reach, so they justify the
particle method.

---

## 9. Key references (curated; full lists in the research appendix)

**MPM foundations & flavors:** Sulsky, Chen & Schreyer, *CMAME* 118 (1994);
Bardenhagen & Kober, *CMES* 5 (2004, GIMP); Sadeghirad, Brannon & Burghardt,
*IJNME* 86 (2011, CPDI); Jiang, Schroeder, Selle, Teran & Stomakhin, *TOG* 34
(2015, APIC) + *JCP* 338 (2017); Hu et al., *TOG* 37 (2018, MLS-MPM); Stomakhin et
al., *TOG* 32 (2013, elastoplastic snow MPM); de Vaucorbeil, Nguyen, Sinaie & Wu,
*Adv. Appl. Mech.* 53 (2020, review); Nguyen, de Vaucorbeil & Bordas, Springer
(2023, book).

**Finite-strain J2 at a point (the reused algorithm):** Simo, *CMAME* 99 (1992);
Weber & Anand, *CMAME* 79 (1990); Eterovic & Bathe, *IJNME* 30 (1990); Cuitiño &
Ortiz, *Eng. Comput.* 9 (1992); Simo & Hughes, *Computational Inelasticity*
(1998). Hypoelastic alternative: Belytschko, Liu, Moran & Elkhodary (2014).

**Time integration & stability:** Bardenhagen, *JCP* 180 (2002, USF/USL/MUSL
energy); Wallstedt & Guilkey, *JCP* 227 (2008); Steffen, Kirby & Berzins, *IJNME*
76 (2008, quadrature/B-spline); Brackbill, *JCP* 75 (1988, ringing); Gritton,
Berzins & Kirby, *Comput. Part. Mech.* 4 (2017, null-space filter); Berzins (2018)
and Ni & Zhang, *IJNME* 121 (2020, critical Δt); Guilkey & Weiss, *IJNME* 57
(2003, implicit); Charlton, Coombs & Augarde, *Comput. Struct.* 190 (2017, iGIMP).

**Volumetric locking:** Coombs et al., *CMAME* 333 (2018, F̄ at particles);
Iaconeta, Larese, Rossi & Oñate, *Comput. Mech.* 63 (2019, mixed u–p).

**Benchmarks:** Taylor, *Proc. R. Soc. A* 194 (1948); Sulsky & Schreyer, *CMAME*
139 (1996, MPM Taylor + upsetting); Simo (1988) & Simo & Armero, *IJNME* 33 (1992,
necking); Abaqus Benchmarks 1.3.10 (copper rod, Case A parameters).

**Julia:** Tesserae.jl (Nakamura, MIT); MaterialPointSolver.jl (Huo et al.,
*Computers & Geotechnics* 183, 2025); `taichi_mpm` (Hu et al. 2018).

---

## 10. Risks, open questions, decisions to make

- **Explicit vs implicit.** Explicit is simplest and natural for dynamic Taylor
  impact, but quasi-static forming/necking needs many tiny steps → consider mass
  scaling / dynamic relaxation, or an implicit MPM later (Guilkey–Weiss; iGIMP).
  *Decision: start explicit; revisit if quasi-static cases dominate.*
- **Own code vs Tesserae.jl.** §5 — recommend own minimal MPM first; the main risk
  of "own" is reimplementing CPDI/GPU, which we defer.
- **F̄ correctness in MPM** is the top numerical risk for metal J2 — validate the
  no-locking gate (Phase 3) before trusting large-plastic results.
- **Taylor-bar reference numbers** (final length/foot radius) should be confirmed
  against the primary Wilkins–Guinan (1973) source before using as pass/fail
  targets; the parameter *inputs* (Case A) are confirmed.
- **Scope discipline** (as in `lagrangian`): grid-based MPM, J2 + **linear**
  hardening, isotropic for the baseline; defer contact-heavy forming, fracture/
  erosion, and GPU. **Exception:** nonlinear/saturation hardening is *not* fully
  deferrable — the published Simo necking curve (V3b) needs it (local Newton on
  Δγ); schedule it as the first post-baseline kernel upgrade, ahead of Phase-4 V3b.
- **Grid contact & axisymmetry are architectural.** Taylor/upsetting need grid
  rigid-wall/multi-body contact (§3, M-list), and the axisymmetric-vs-3D choice
  (§7 Phase 0) sets the discretization — neither can be bolted on late.

**Proposed first step:** approve this baseline, then write a short
`particle_method/docs/DESIGN.md` (the implementation contract, mirroring
`lagrangian/docs/DESIGN.md`) and stand up Phase 1 (explicit elastodynamics) to the
vibrating-bar gate.
