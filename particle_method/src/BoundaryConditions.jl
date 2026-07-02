"""
    BoundaryConditions

Grid-node velocity boundary conditions (DESIGN §7): fixed/symmetry planes and
prescribed (displacement-control) velocities, selected by predicates on grid
node coordinates — the same predicate-selection UX as `lagrangian`'s
`on_face`/`select_nodes`. Operates on plain `Grid` + `Vector{VelocityBC}` (not
on `MPMModel`, to avoid a module-inclusion cycle with `Step.jl`); `Step.jl`
adds thin `MPMModel`-taking methods of the same generic functions.

**Applied after damping** (DESIGN §4/§7 — the corrected order): a
damped-then-overwritten node is fine; a damped-after-BC node would silently
under-drive every prescribed-velocity boundary.
"""
module BoundaryConditions

using StaticArrays
using ..GridMod: Grid, node_coords

export VelocityBC, ForceBC, select_nodes, fix!, symmetry!, prescribe!, apply_bcs!,
       load!, apply_loads!

"""
    select_nodes(grid, pred) -> Vector{Int}

Grid node linear indices whose position satisfies `pred(x::SVector{3})::Bool`
(DESIGN §7's `on_face`/`select_nodes` analogue on grid coordinates).
"""
function select_nodes(grid::Grid, pred)
    N = grid.n[1] * grid.n[2] * grid.n[3]
    idx = Int[]
    for i in 1:N
        pred(node_coords(grid, i)) && push!(idx, i)
    end
    return idx
end

@inline function _axis_mask(comp::Symbol)
    comp === :x && return SVector{3,Bool}(true, false, false)
    comp === :y && return SVector{3,Bool}(false, true, false)
    comp === :z && return SVector{3,Bool}(false, false, true)
    comp === :all && return SVector{3,Bool}(true, true, true)
    throw(ArgumentError("comp must be :x, :y, :z, or :all (got $comp)"))
end

"""
    VelocityBC(nodes, mask, value)

A grid-node velocity BC: `nodes` the (precomputed) selected linear indices,
`mask::SVector{3,Bool}` which components it overwrites, `value(t)` a function
of simulation time returning the target `SVector{3,Float64}` (masked
components only).
"""
struct VelocityBC
    nodes::Vector{Int}
    mask::SVector{3,Bool}
    value::Function
end

"""
    fix!(bcs, grid, pred, comp=:all)

Zero the `comp` velocity component(s) (`:x`,`:y`,`:z`,`:all`) of every grid
node satisfying `pred` (DESIGN §7 fixed/symmetry plane), for every step.
Pushes a `VelocityBC` onto `bcs`.
"""
function fix!(bcs::Vector{VelocityBC}, grid::Grid, pred, comp::Symbol=:all)
    nodes = select_nodes(grid, pred)
    push!(bcs, VelocityBC(nodes, _axis_mask(comp), t -> zero(SVector{3,Float64})))
    return bcs
end

"""
    symmetry!(bcs, grid, pred, axis)

Alias for `fix!` (DESIGN §7): a roller / symmetry-plane BC zeroing the normal
velocity component `axis` (`:x`,`:y`,`:z`) of the selected nodes.
"""
symmetry!(bcs::Vector{VelocityBC}, grid::Grid, pred, axis::Symbol) =
    fix!(bcs, grid, pred, axis)

"""
    prescribe!(bcs, grid, pred, comp, v)

Set the `comp` (`:x`/`:y`/`:z`) velocity of every grid node satisfying `pred`
to `v` (DESIGN §7, displacement control) — `v` is either a constant
`Real` or a ramp `t -> Float64`. Pushes a `VelocityBC` onto `bcs`.
"""
function prescribe!(bcs::Vector{VelocityBC}, grid::Grid, pred, comp::Symbol, v)
    comp === :all && throw(ArgumentError("prescribe! needs a single axis :x/:y/:z"))
    nodes = select_nodes(grid, pred)
    mask = _axis_mask(comp)
    valf = v isa Function ? v : (t -> Float64(v))
    vec = t::Float64 -> SVector{3,Float64}(mask[1] ? valf(t) : 0.0,
                                           mask[2] ? valf(t) : 0.0,
                                           mask[3] ? valf(t) : 0.0)
    push!(bcs, VelocityBC(nodes, mask, vec))
    return bcs
end

"""
    apply_bcs!(grid, bcs, t)

Enforce every BC's target velocity on its nodes (DESIGN §7). Called **after**
damping (DESIGN §4) — the one ordering fix this design was hardened around.
"""
function apply_bcs!(grid::Grid, bcs::Vector{VelocityBC}, t::Float64)
    @inbounds for bc in bcs
        val = bc.value(t)
        m = bc.mask
        for i in bc.nodes
            v = grid.v[i]
            grid.v[i] = SVector{3,Float64}(m[1] ? val[1] : v[1],
                                           m[2] ? val[2] : v[2],
                                           m[3] ? val[3] : v[3])
        end
    end
    return grid
end

"""
    ForceBC(nodes, force)

A grid-node external nodal force (traction/point load, DESIGN §7: "Traction/
body loads enter as grid forces `fᵢ += ...`", mirroring `lagrangian`'s
`load!`). `force(t)` is a function of simulation time returning the
`SVector{3,Float64}` force added to **each** of `nodes` every step.
"""
struct ForceBC
    nodes::Vector{Int}
    force::Function
end

"""
    load!(loads, grid, pred, comp, value; distribute=true)

Add a nodal force BC (DESIGN §7): `value` (constant or `t -> Float64`) is the
**total** force in direction `comp` (`:x`/`:y`/`:z`) applied over every grid
node satisfying `pred`; if `distribute` (default), it is divided evenly across
the selected nodes (matching `lagrangian`'s `load!(...; distribute=true)`
convention for a face/traction load), otherwise `value` is applied to *each*
selected node. Pushes a `ForceBC` onto `loads`.
"""
function load!(loads::Vector{ForceBC}, grid::Grid, pred, comp::Symbol, value;
               distribute::Bool=true)
    comp === :all && throw(ArgumentError("load! needs a single axis :x/:y/:z"))
    nodes = select_nodes(grid, pred)
    mask = _axis_mask(comp)
    valf = value isa Function ? value : (t -> Float64(value))
    nn = length(nodes)
    scale = distribute && nn > 0 ? 1.0 / nn : 1.0
    fvec = t::Float64 -> (scale * valf(t)) * SVector{3,Float64}(mask[1], mask[2], mask[3])
    push!(loads, ForceBC(nodes, fvec))
    return loads
end

"""
    apply_loads!(grid, loads, t)

Add every `ForceBC`'s nodal force to `grid.f` (DESIGN §7: `fᵢ += ...`).
Called during P2G's grid-force accumulation, **before** the explicit momentum
update (so it participates in `vᵢ* = vᵢ + Δt fᵢ/mᵢ` exactly like the internal
and gravity forces).
"""
function apply_loads!(grid::Grid, loads::Vector{ForceBC}, t::Float64)
    @inbounds for bc in loads
        f = bc.force(t)
        for i in bc.nodes
            grid.f[i] += f
        end
    end
    return grid
end

end # module
