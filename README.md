# experimental — computational plasticity under large deformation

Two complementary approaches to 3D computational elastoplasticity, especially at
large deformation, live side by side in this repository:

| Sub-project | Approach | Status |
|---|---|---|
| [`lagrangian/`](lagrangian/) | **Mesh-based finite element method** (Lagrangian). Small- and finite-strain J2 plasticity, Hex8 elements, F-bar, CG+AMG scaling to ~10M DOFs. | Complete, tested |
| [`particle_method/`](particle_method/) | **Particle / meshfree method** (e.g. MPM/SPH) for plasticity under *very* large deformation, where a fixed mesh distorts and fails. | Exploratory |

## Why two methods

Mesh-based FEM (the `lagrangian/` solver) is accurate and efficient, but a
total-Lagrangian formulation is limited by **element distortion**: at extreme
strains (bulk forming, crushing, deformation to fracture) the elements degrade
until a Jacobian goes non-positive. Particle / meshfree methods carry the material
on points rather than a fixed mesh, so the mesh-distortion barrier disappears —
which is why `particle_method/` is the natural place to explore the very-large-
deformation regime. The two share the same physics (J2 return-mapping
constitutive law), so ideas and kernels can cross over.

## Getting started

- The finite element solver: see [`lagrangian/README.md`](lagrangian/README.md)
  (run examples with `julia --project=lagrangian lagrangian/examples/…`).
- The particle-method exploration: see
  [`particle_method/README.md`](particle_method/README.md).
