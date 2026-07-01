"""
    PlasticityFEM

A 3D finite element solver for small-strain elastoplasticity (J2 / von Mises,
combined linear isotropic + kinematic hardening, Hex8 elements). See docs/DESIGN.md.

Public API (curated, small):
    box_mesh, on_face, select_nodes, J2Material, Model, fix!, prescribe!, load!,
    solve!, reset!, nodal_displacements, gauss_stress, gauss_strain,
    equivalent_plastic_strain, von_mises, write_vtu
"""
module PlasticityFEM

include("Materials.jl")
include("Mesh.jl")
include("FiniteStrain.jl")
include("Elements.jl")
include("BoundaryConditions.jl")
include("Assembly.jl")
include("Model.jl")
include("Solver.jl")
include("Visualization.jl")

using .Materials
using .MeshMod
using .FiniteStrain
using .Elements
using .BoundaryConditions
using .Assembly
using .ModelMod
using .Solver
using .Visualization

# --- curated public exports (DESIGN §5.1) ---
export box_mesh, on_face, select_nodes
export J2Material
export Model, fix!, prescribe!, load!, solve!, reset!
export nodal_displacements, gauss_stress, gauss_kirchhoff, equivalent_plastic_strain
export write_vtu, gauss_strain, von_mises
export SolveResult
# finite-strain element-kind selectors (FINITE_STRAIN §6.1, §6.4)
export ElementKind, Hex8Small, Hex8Finite, Hex8FiniteFbar

# Re-export selected internals useful for tests / advanced use.
export return_map, precompute_cache, element_geometry, element_force_tangent!,
       build_sparsity, assemble!, GaussState, DirichletBC, NeumannBC,
       impose_dirichlet!, dof, Mesh, LinearSolveState

end # module
