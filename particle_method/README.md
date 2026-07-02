# ParticlePlasticity.jl — MPM for very large deformation

An explicit **Material Point Method (MPM)** solver for J2/von Mises
elastoplasticity in Julia, for the deformation regime where the mesh-based
[`../lagrangian/`](../lagrangian/) solver runs out of road: elements distort
until a Jacobian goes non-positive, a hard ceiling that only remeshing can push
past. MPM carries material state on **particles** and resets a background grid
every step, so mesh distortion is not a failure mode.

The constitutive core (finite-strain, log-strain radial-return J2 plasticity)
is **reused, unmodified**, from `lagrangian/src/FiniteStrain.jl` and
`Materials.jl` via a `[sources]` path-dependency — the physics is
discretization-agnostic; only how gradients/momentum are computed changes.
See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the literature survey and
[`docs/DESIGN.md`](docs/DESIGN.md) for the full implementation spec.

| | |
|---|---|
| **Method** | Explicit MPM, symplectic Euler, single background grid reset every step |
| **Grid basis** | Quadratic B-spline (C¹, 27-node stencil in 3D) |
| **Transfer** | APIC (affine PIC), single transfer, `Lₚ = Cₚ` for the F-update |
| **Constitutive** | Reused `lagrangian` finite-strain log-strain radial return (unmodified) |
| **Locking cure** | F̄ at particles (cell-averaged `J`), flag-gated |
| **Quasi-static loading** | Mass scaling + scalar viscous damping, `KE/IE` diagnostic |
| **Output** | Dependency-free `.vtu` point-cloud export for ParaView |

## Status

Implemented and validated against `lagrangian` (gates G0–G6, `test/runtests.jl`):
the reused kernel matches bit-exactly through the MPM particle path (G1); a
homogeneous tension block matches FEM stress to ~5e-4 relative (G3); F̄ cuts
pressure checkerboarding >7x under fully-plastic compression (G4); a bent
cantilever and a moderately necked bar both match FEM's deflection/contraction
within a few–ten percent (G5, G6) once driven-boundary predicates are set up to
track the material through large motion, not just its initial position (see
`test_G5_cantilever.jl`'s root-cause note — a real, fixed bug: a Dirichlet
velocity BC selected once at t=0 on a small, moving-material footprint detaches
once the material travels more than the B-spline support radius).

Known open item: MPM's peak-equivalent-plastic-strain *location* in the necking
gate sits slightly off the true neck, at the driven boundary — a smaller,
distinct boundary-stencil artifact from the one above, root-caused but not
eliminated (see `test_G6_necking.jl`). Geometric localization (where the neck
forms) is unaffected.

## Quick start

```julia
using ParticlePlasticity

grid = Grid(SVector(-1.0, -1.0, -1.0), 0.25, (44, 12, 12))  # origin, spacing, node counts
pts  = sample_box(SVector(0.0,0.0,0.0), SVector(10.0,1.0,1.0), 0.25; ppc=2, ρ=7.85e-9)
mat  = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)

model = MPMModel(grid, pts, mat; dt=1e-7, fbar=true, damping=0.02)
fix!(model, x -> x[1] < 1e-9, :all)
prescribe!(model, x -> x[1] > 10.0 - 1e-9, :x, t -> 5000.0)  # ramp velocity

for _ in 1:10000
    step!(model)
end
write_particles_vtu("out", model)
```

See `examples/` for the bent-rod and extreme-necking demonstrations.
