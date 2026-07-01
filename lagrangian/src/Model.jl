"""
    ModelMod

The assembled problem (`Model`) and the high-level UX builders
(`fix!`, `prescribe!`, `load!`) plus postprocessing. See DESIGN.md §4.6, §6.
"""
module ModelMod

using SparseArrays
using StaticArrays
using LinearAlgebra: det
using ..MeshMod: Mesh, GaussState, dof, reset_state!
using ..Materials: J2Material
using ..Elements: ElementCache, precompute_cache, element_geometry, element_ref_grads
using ..FiniteStrain: ElementKind, Hex8Small, Hex8Finite, Hex8FiniteFbar, voigt_to_sym3
using ..BoundaryConditions: DirichletBC, NeumannBC
using ..Assembly: SparsityPattern, build_sparsity

export Model, fix!, prescribe!, load!, reset!,
       nodal_displacements, gauss_stress, gauss_kirchhoff, equivalent_plastic_strain,
       Hex8Small, Hex8Finite, Hex8FiniteFbar

"""
    Model(mesh, material)

Assembled elastoplastic problem. Heavy arrays (cache, sparsity, state) are
preallocated once; `U` and buffers are reassignable (DESIGN §4.6).

BCs/loads are accumulated in growable lists by the builder API and frozen into
`DirichletBC`/`NeumannBC` at solve time.
"""
mutable struct Model{Ti<:Integer,EK<:ElementKind}
    mesh::Mesh
    material::J2Material
    kind::EK                         # element kind (zero-size; static dispatch seam)
    cache::ElementCache
    sparsity::SparsityPattern{Ti}
    state_committed::GaussState
    state_trial::GaussState
    Rbuf::Vector{Float64}
    U::Vector{Float64}
    δU::Vector{Float64}             # Newton correction buffer (reused each iter)
    # BC accumulators (DOF-indexed; deduplicated at solve)
    dir_dofs::Vector{Int}
    dir_vals::Vector{Float64}
    dir_ramp::Vector{Bool}
    neu_dofs::Vector{Int}
    neu_vals::Vector{Float64}
end

"""
    Model(mesh, material; element=:small, Ti=nothing)

Build the assembled problem. `element` selects the kinematics / element family
(FINITE_STRAIN §6.4):
- `:small`       — small-strain Hex8 (default, v1 path, behaviorally unchanged);
- `:finite`      — finite-strain Hex8 (standard F, Hencky/J2);
- `:finite_fbar` — finite-strain Hex8 with F-bar (near-incompressible limit).

`Ti` selects the sparse index type. The returned `Model` is parametric on the
index type and the `ElementKind`, so the assembly hot loop dispatches statically.
"""
function Model(mesh::Mesh, material::J2Material;
               element::Symbol=:small, Ti::Union{Type{<:Integer},Nothing}=nothing)
    ndof = 3 * mesh.nnodes
    ngp = mesh.nelem * 8
    cache = precompute_cache(mesh.nodes, mesh.elements)
    sparsity = build_sparsity(mesh; Ti=Ti)
    Tidx = eltype(sparsity.K.colptr)
    kind = _element_kind(element)
    return Model{Tidx,typeof(kind)}(mesh, material, kind, cache, sparsity,
                       GaussState(ngp), GaussState(ngp),
                       zeros(ndof), zeros(ndof), zeros(ndof),
                       Int[], Float64[], Bool[], Int[], Float64[])
end

function _element_kind(element::Symbol)
    element === :small && return Hex8Small()
    element === :finite && return Hex8Finite()
    element === :finite_fbar && return Hex8FiniteFbar()
    error("unknown element kind $element (use :small, :finite, :finite_fbar)")
end

# True for finite-strain element kinds (Kirchhoff stored, Cauchy reported).
@inline _is_finite(::Model{Ti,Hex8Small}) where {Ti<:Integer} = false
@inline _is_finite(::Model{Ti,EK}) where {Ti<:Integer,EK<:ElementKind} = true

# component symbol -> list of component indices
function _comps(comp::Symbol)
    comp === :x && return (1,)
    comp === :y && return (2,)
    comp === :z && return (3,)
    comp === :all && return (1, 2, 3)
    error("unknown component $comp (use :x,:y,:z,:all)")
end

"""
    fix!(model, nodes, comp=:all)

Homogeneous Dirichlet (u = 0) on selected nodes/components (DESIGN §6.3).
"""
function fix!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol=:all)
    for n in nodes, c in _comps(comp)
        push!(model.dir_dofs, dof(n, c))
        push!(model.dir_vals, 0.0)
        push!(model.dir_ramp, false)
    end
    return model
end

"""
    prescribe!(model, nodes, comp, value; ramp=true)

Inhomogeneous Dirichlet: prescribe `value` on selected nodes/component
(DESIGN §6.3).
"""
function prescribe!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol,
                    value::Real; ramp::Bool=true)
    for n in nodes, c in _comps(comp)
        push!(model.dir_dofs, dof(n, c))
        push!(model.dir_vals, Float64(value))
        push!(model.dir_ramp, ramp)
    end
    return model
end

"""
    load!(model, nodes, comp, value; distribute=false, ramp=true)

Nodal force on selected nodes/component. `distribute=true` splits `value`
equally across the nodes (user gives a *total* face load) (DESIGN §6.3).
`ramp` is honored via the Neumann load factor at solve time.
"""
function load!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol,
               value::Real; distribute::Bool=false, ramp::Bool=true)
    nn = length(nodes)
    per = distribute ? Float64(value) / nn : Float64(value)
    for n in nodes, c in _comps(comp)
        push!(model.neu_dofs, dof(n, c))
        push!(model.neu_vals, per)
    end
    # `ramp` always true in v1 Neumann; kept in signature for API completeness.
    ramp || @warn "load! ramp=false not supported in v1; load is ramped" maxlog=1
    return model
end

"""
    reset!(model) -> model

Return the model to its just-built state: zero the displacement solution and the
committed/trial per-Gauss-point history (plastic strain, back stress, ᾱ, stress).
BCs and loads are kept. `solve!` calls this on entry so that re-solving a model
(e.g. after changing the load magnitude) starts from the undeformed, unhardened
state rather than silently continuing to accumulate plastic strain from a
previous solve.
"""
function reset!(model::Model)
    fill!(model.U, 0.0)
    reset_state!(model.state_committed)
    reset_state!(model.state_trial)
    return model
end

# --- postprocessing (DESIGN §6.3) ---

"""
    nodal_displacements(model) -> Matrix (3 × nnodes)
"""
function nodal_displacements(model::Model)
    U = model.U
    out = Matrix{Float64}(undef, 3, model.mesh.nnodes)
    @inbounds for n in 1:model.mesh.nnodes
        out[1, n] = U[dof(n, 1)]
        out[2, n] = U[dof(n, 2)]
        out[3, n] = U[dof(n, 3)]
    end
    return out
end

"""
    gauss_stress(model) -> Matrix (6 × ngp)  committed stresses

For small-strain models this is the stored Cauchy/engineering stress. For finite-
strain models the stored quantity is the **Kirchhoff** stress τ; this function
reports the **Cauchy** stress σ = τ/J (J = det F at the committed displacement),
per FINITE_STRAIN §6.4. Use `gauss_kirchhoff` for the raw τ.
"""
function gauss_stress(model::Model)
    _is_finite(model) || return model.state_committed.σ
    τ = model.state_committed.σ
    σ = similar(τ)
    edofs = model.sparsity.edofs
    U = model.U
    @inbounds for e in 1:model.mesh.nelem
        ue = SVector{24,Float64}(ntuple(i -> U[edofs[i, e]], Val(24)))
        dNdXs = element_ref_grads(model.cache, e)
        for g in 1:8
            gp = (e - 1) * 8 + g
            F = _defgrad(ue, dNdXs[g])
            J = det(F)
            invJ = J > 0 ? 1.0 / J : 1.0
            for k in 1:6
                σ[k, gp] = τ[k, gp] * invJ
            end
        end
    end
    return σ
end

# local deformation gradient (avoids importing the kernel's helper namespace)
@inline function _defgrad(ue::SVector{24,Float64}, dNdX)
    H = zero(SMatrix{3,3,Float64,9})
    @inbounds for a in 1:8
        ux = ue[3(a - 1) + 1]; uy = ue[3(a - 1) + 2]; uz = ue[3(a - 1) + 3]
        gx = dNdX[a, 1]; gy = dNdX[a, 2]; gz = dNdX[a, 3]
        H += SMatrix{3,3,Float64,9}(ux * gx, uy * gx, uz * gx,
                                    ux * gy, uy * gy, uz * gy,
                                    ux * gz, uy * gz, uz * gz)
    end
    return SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1) + H
end

"""
    gauss_kirchhoff(model) -> Matrix (6 × ngp)  committed Kirchhoff stress τ
(finite-strain models). For small strain this equals `gauss_stress`.
"""
gauss_kirchhoff(model::Model) = model.state_committed.σ

"""
    equivalent_plastic_strain(model) -> Vector (ngp)  committed ᾱ
"""
equivalent_plastic_strain(model::Model) = model.state_committed.ᾱ

end # module
