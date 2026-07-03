# Matching a finite-strain neck: MPM vs. FEM

This note documents a controlled comparison between the two solvers in this repo —
the Lagrangian finite-strain FEM (`lagrangian/`) and the Material Point Method
(`particle_method/`) — on the same necking problem, and the tuning it took to bring
the MPM neck into quantitative agreement with the FEM.

Both solvers share the one J2 return-mapping kernel, so any disagreement here is
**discretization, not physics**. The exercise was to find and remove the numerical
dissipation that makes the explicit MPM under-localize a sharp neck.

## The specimen

A round titanium bar with a **smooth parabolic ("spindle") profile** — the center is
thinner than the ends, so the mid-length is the unambiguous global-minimum section and
localization has nowhere to go but there.

- End radius `R0 = 5`, length `L = 50`; radius `R(x) = R0·(1 − taper·(1 − s²))`,
  `s = (x−L/2)/(L/2)`, with `taper = 0.12` (center 12% thinner than the ends).
- Ti-6Al-4V (linearized): `E = 113.8 GPa, ν = 0.342, σy0 = 880 MPa, Hiso = 1000`.
- Pulled to 9% nominal elongation.

**Why a 12% spindle and not a slight notch.** A weak imperfection (a 1.5% notch, even a
5% notch) does *not* localize the neck in MPM: the driven grip's edge is a stronger
stress raiser than the notch, so the neck forms at the grips instead. The FEM has no
such trouble (it prescribes displacement on a single node layer). Making the center the
global-min section over a broad region — the spindle — is what lets *both* methods
localize at the same place. The MPM threshold is a taper of ~10–12%; below ~6% it still
necks at the grips.

## Boundary conditions (perfectly symmetric)

Tension is applied symmetrically so there is no dynamic bias and no lateral-grip stress
raiser:

- Both ends pulled apart by `±δ/2` (uniform axial motion per end).
- In-plane rigid-body modes removed by **symmetry-plane rollers**: `u_y = 0` on the
  `y=0` plane, `u_z = 0` on the `z=0` plane. These also keep the neck axisymmetric with
  zero over-constraint.
- FEM additionally grips the end faces laterally (stabilizes the coarse tapered corner;
  the neck is far away at the center so it is unaffected). MPM drives thin symmetric grip
  bands and is otherwise free to contract.
- Displacement control, not load control — necking is a softening instability, so a
  prescribed displacement traverses the post-peak branch where load control snaps through.

## The FEM result (reference, converged)

The FEM necks cleanly at the center and is **resolution-converged**: a graded mesh with
0.13 mm elements at the neck and a uniform 0.42 mm mesh give essentially the same answer.

| FEM mesh | neck radius reduction | peak ᾱ |
|---|---|---|
| graded, 0.13 mm at center (12×12×160) | 24.6% | 0.41 |
| uniform, 0.42 mm (12×12×120) | 25.3% | 0.41 |

So ~25% reduction with the peak equivalent plastic strain **at the center** is the target.

## Why MPM under-localizes

At matched resolution the MPM initially disagreed badly — a shallower, more diffuse neck
with the plastic strain smeared onto the *shoulders* instead of peaked at the center, and
(worse) it got *more* diffuse under naive refinement. Three sources of numerical
dissipation, all absent from the quasi-static FEM:

1. **Quasi-static damping.** The explicit solver uses viscous grid-velocity damping to
   settle toward equilibrium. That same term resists the fast inward motion of a forming
   neck. Critically it is **not normalized by `dt`** (a fixed fraction per step), so
   refining the grid — which shrinks `dt` and adds steps — makes it bite *harder*. This
   is why h=0.83 regressed vs. h=1.25.
2. **Transfer dissipation.** APIC recovers the affine velocity field losslessly but still
   discards the sub-affine part every particle→grid→particle round trip — a diffusion of
   the velocity field over the stencil.
3. **Shape-function smoothing.** The quadratic B-spline support is `3h` wide; it spreads
   the peak strain over a few cells. This is the last, irreducible smoothing length.

## The levers (and how to push MPM toward FEM)

| Lever | Where | Direction to sharpen | Notes |
|---|---|---|---|
| Grid `h` (master — sets support, F-bar cell, spacing) | `examples/…:H` | decrease | work ∝ h⁻⁴ |
| Viscous damping | `MPMModel(...; damping=)` → `Step.jl` | decrease / scale with `dt` | watch KE/IE gate |
| **PIC/FLIP blend** | `MPMModel(...; flip=)` → `Transfer.jl g2p!` | increase toward 1 | added for this study |
| F-bar | `MPMModel(...; fbar=)` | keep **on** | off relocks → worse |
| `ppc` | `sample_region(...; ppc=)` | 2→3 | secondary |

The **FLIP/APIC blend** (`flip`) is new code: `g2p!` now blends the PIC gather with a
FLIP increment (`vₚ = (1−flip)·vₚᴾᴵᶜ + flip·(vₚᵒˡᵈ + Σw(vᵢ − vᵢᴾ²ᴳ))`), keeping APIC's
affine `Cₚ` and PIC advection. `flip = 0` reproduces the original solver exactly (all
6624 tests pass).

## Convergence of MPM toward FEM (12% spindle)

| MPM config | neck reduction | peak ᾱ | ᾱ location |
|---|---|---|---|
| h=0.83, damp=0.02 | 15.8% | 0.14 | shoulders |
| h=0.625, damp=0.005 | 22.3% | 0.24 | center |
| h=0.625, damp=0.003, flip=0.9 | 24.2% | 0.30 | center |
| h=0.5, damp=0.002, flip=0.9 | 23.7% | 0.34 | center |
| **FEM (converged)** | **~25%** | **0.41** | center |

**Neck depth, location, and profile shape converge onto the FEM.** In the neck region the
ᾱ profiles nearly overlie each other; only the pointwise peak remains ~15% low, smeared by
the `3h` B-spline support — it climbs monotonically (0.14 → 0.24 → 0.30 → 0.34) as the
grid is refined. Closing it fully needs much finer `h` (each halving is ~16× cost) or a
linear-basis option, both diminishing returns against an already-solid match.

## Reproducing

```bash
export PATH="$HOME/.juliaup/bin:$PATH"

# FEM reference (uniform 0.42 mm, 12% spindle)
cd lagrangian
TAPER=0.12 NCROSS=12 NAXIAL=120 GRADE=1.0 NSTEPS=160 \
  julia -t auto --project=. examples/necking_titanium_symmetric.jl

# MPM, tuned toward the FEM (finer grid + reduced damping + FLIP blend)
cd ../particle_method
TAPER=0.12 H=0.5 DAMP=0.002 FLIP=0.9 \
  julia --project=. examples/necking_titanium_bar.jl
```

Env knobs: `TAPER` (spindle amplitude), `H` (grid spacing), `DAMP` (viscous damping),
`FLIP` (PIC/FLIP blend), plus `NCROSS/NAXIAL/GRADE/NSTEPS` on the FEM side.

## Files

- `lagrangian/examples/necking_titanium_symmetric.jl` — FEM, symmetric spindle.
- `particle_method/examples/necking_titanium_bar.jl` — MPM, symmetric spindle.
- `particle_method/src/Transfer.jl` — `g2p!` with the PIC/FLIP blend.
- `particle_method/src/Step.jl` — `MPMModel.flip` field and plumbing.

## Takeaway

Once the extra numerical dissipation is accounted for — dt-consistent damping, a FLIP
blend to stop bleeding the velocity field, and a fine enough grid — the particle neck
matches the finite-element neck in depth, location, and profile, with only the very peak
strain still slightly rounded by the shape functions. The constitutive kernel never
changed. The disagreement was entirely in the discretization.
