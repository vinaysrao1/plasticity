"""
    ParticlePlasticity

A 3D Material Point Method (MPM) solver for J2 / von Mises elastoplasticity at
finite and very large deformation, reusing the verified finite-strain
constitutive kernel of the sibling `PlasticityFEM` (`../lagrangian`) package.
See `docs/DESIGN.md` (implementation spec) and `docs/PROPOSAL.md` (rationale).

Public API (curated, DESIGN §9):
    Grid, Particles, sample_box, sample_region, J2Material,
    MPMModel, fix!, symmetry!, prescribe!, gravity!, run!, step!,
    equivalent_plastic_strain, particle_cauchy, kinetic_energy,
    write_particles_vtu

One deliberate, small addition beyond §9's literal bullet list: `load!`
(nodal force / traction BC). §7's prose explicitly calls for it ("Traction/
body loads enter as grid forces... mirrors lagrangian's fix!/prescribe!/
load!") — the §9 bullet list omits it, evidently an oversight, not a
decision to cut it. It was implemented to try a force-controlled G5 cantilever
(matching `lagrangian`'s tip-traction example exactly); the *final* G5 gate
instead uses `prescribe!` (displacement control, far fewer steps to a
quasi-static state for that problem) — `load!` remains a small, tested,
correctly-behaving part of the public API per §7, kept as minimal as the
existing `fix!`/`prescribe!` builders.
"""
module ParticlePlasticity

using PlasticityFEM.Materials: J2Material

include("Grid.jl")
include("Particles.jl")
include("Transfer.jl")
include("Constitutive.jl")
include("BoundaryConditions.jl")
include("Step.jl")
include("Visualization.jl")

using .GridMod
using .ParticlesMod
using .Transfer
using .Constitutive
using .BoundaryConditions
using .Step
using .Visualization

# --- curated public exports (DESIGN §9) ---
export Grid, Particles, sample_box, sample_region, J2Material
export MPMModel, fix!, symmetry!, prescribe!, load!, gravity!, run!, step!
export equivalent_plastic_strain, particle_cauchy, kinetic_energy, write_particles_vtu

end # module
