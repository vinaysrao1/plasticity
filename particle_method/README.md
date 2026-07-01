# particle_method — plasticity under very large deformation

**Status: exploratory / starting point.** Nothing implemented yet — this is the
home for a particle / meshfree approach to computational plasticity, aimed at the
deformation regime where the mesh-based [`../lagrangian/`](../lagrangian/) solver
runs out of road.

## Motivation

The `lagrangian/` finite element solver is a total-Lagrangian formulation: the
mesh is attached to the material and deforms with it. That is accurate and fast,
but at **extreme** deformation (bulk forming, extrusion, crushing/folding,
deformation to fracture, hundreds of % strain) the elements distort until a
Jacobian goes non-positive — a hard ceiling that only remeshing/rezoning can push
past. Particle / meshfree methods sidestep this by carrying the material state on
**points** instead of a fixed connectivity, so mesh distortion is not a failure
mode.

## Candidate approaches to explore

- **MPM (Material Point Method).** Material points carry mass/stress/history and
  are mapped to a background grid each step to solve momentum, then the grid
  solution updates the points. The background grid is reset every step, so it
  never distorts. Strong for very large deformation, self-contact, and
  fragmentation. Leading candidate.
- **SPH / Total-Lagrangian SPH.** Fully meshfree; TLSPH mitigates the tensile
  instability of classic SPH for solids.
- **Optimal-transport / other meshfree particle schemes.** Secondary.

## What can be reused from `lagrangian/`

The physics is method-agnostic. The J2 return-mapping constitutive kernel
(`return_map`) and the finite-strain log-strain wrap are **independent of the
spatial discretization** — they map a strain increment + history to stress +
updated history. A particle method needs the same constitutive update at each
material point, so that kernel (and its verification) can carry over directly; only
the *discretization* (how gradients/momentum are computed and integrated) changes.

## First milestones (proposed)

1. Pick the method (MPM first) and write a short design note (mirroring
   `lagrangian/docs/`).
2. 1D/2D elastic bar to validate the grid↔particle transfer and time integration.
3. Add the shared J2 plasticity constitutive update at the material points.
4. A large-deformation benchmark that the Lagrangian mesh solver *can't* reach
   (e.g. deep necking to fracture, upsetting/forging), compared where they overlap.
