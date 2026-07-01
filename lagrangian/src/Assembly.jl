"""
    Assembly

Global sparsity pattern and assembly of the global tangent and residual by
scattering element contributions into `nzval`. See DESIGN.md §5.1, §7 and the
scaling redesign SCALING.md §2.

Scaling changes (SCALING.md §2):
- `build_sparsity` builds the CSC skeleton by **count-then-fill from node
  adjacency** — it never materializes the `nelem·576` COO triplet arrays
  (≈47 GB transient at 10M).
- `SparsityPattern` drops the per-element scatter `map` (≈15 GB at 10M); numeric
  `assemble!` scatters via an **on-the-fly CSC column binary search**.
- The pattern/assembler are **generic over the index type `Ti`**; large problems
  default to `Int32` (≈3.3 GB saved at 10M), with an `Int64` fallback when
  `nnz ≥ typemax(Int32)`.
- An analytic **8-color** element partition (box parity `(i%2,j%2,k%2)`) is
  generated here for the race-free threaded assembly of Phase 3.
"""
module Assembly

using SparseArrays
using StaticArrays
using ..MeshMod: Mesh, dof
using ..Elements: ElementCache, element_force_tangent!, element_force_tangent_finite!,
    element_geometry, element_ref_grads, element_coords
using ..FiniteStrain: ElementKind, Hex8Small, Hex8Finite, Hex8FiniteFbar
using ..Materials: J2Material

export SparsityPattern, build_sparsity, assemble!, assemble_threaded!

"""
    SparsityPattern{Ti}

CSC skeleton (`K`, with `nzval` used as the assembly scratch) plus the element
DOF map (`edofs`, 24×nelem) and the 8-color element partition (`colors`) for
threaded assembly. No per-element scatter map is stored — numeric assembly does
an on-the-fly CSC column search (SCALING.md §2.3, §2.6).
"""
struct SparsityPattern{Ti<:Integer}
    K::SparseMatrixCSC{Float64,Ti}
    edofs::Matrix{Ti}                 # 24 × nelem element DOF maps
    colors::Vector{Vector{Int}}       # element-id partition; no two share a node
end

"""
    element_dofs(elements, e) -> NTuple{24,Int}

Global DOFs of element e in local order (node-major: u1x,u1y,u1z,...,u8z).
"""
@inline function element_dofs(elements::Matrix{Int}, e::Int)
    return ntuple(24) do c
        a = (c - 1) ÷ 3 + 1
        comp = (c - 1) % 3 + 1
        dof(elements[a, e], comp)
    end
end

# Node→node adjacency (each node lists itself + every node sharing an element
# with it), as sorted, deduplicated vectors. O(nelem) time, O(N) memory; ≤27
# neighbors/node on a structured Hex8 grid. This replaces the COO transient.
function _node_adjacency(mesh::Mesh)
    nnodes = mesh.nnodes
    elements = mesh.elements
    # gather (unsorted, with duplicates) neighbor lists
    adj = [Int[] for _ in 1:nnodes]
    @inbounds for e in 1:mesh.nelem
        for a in 1:8
            na = elements[a, e]
            la = adj[na]
            for b in 1:8
                push!(la, elements[b, e])
            end
        end
    end
    @inbounds for n in 1:nnodes
        v = adj[n]
        sort!(v)
        unique!(v)
    end
    return adj
end

"""
    build_sparsity(mesh; Ti=nothing) -> SparsityPattern

Build the global CSC skeleton (zero values) by count-then-fill from node
adjacency (SCALING.md §2.4) — no COO triplet array is ever allocated. Also builds
the element DOF map and the 8-color element partition.

The index type `Ti` defaults to `Int` (Int64). Int32 indices would save ~3.3 GB
at 10M, but the smoothed-aggregation AMG preconditioner with a near-null-space
(the rigid-body modes that flatten the CG iteration count — SCALING.md §1.3, R1)
builds Int64 prolongation operators internally and cannot form an Int32 hierarchy,
so an Int32 K would force a retained Int64 copy for AMG (net *more* memory). One
shared Int64 K is the leaner, simpler choice. Pass `Ti=Int32` to force Int32 (only
sound with `amg=:rs` via an internal copy, or the `:direct`/Jacobi paths).
"""
function build_sparsity(mesh::Mesh; Ti::Union{Type{<:Integer},Nothing}=nothing)
    ndof = 3 * mesh.nnodes
    nnodes = mesh.nnodes

    adj = _node_adjacency(mesh)

    # nnz per scalar column = 3 * (#neighbor nodes of that column's node).
    # Each node owns 3 consecutive columns (DOFs) with identical row structure.
    nnz_total = 0
    @inbounds for n in 1:nnodes
        nnz_total += 3 * length(adj[n])   # one column's nnz
    end
    nnz_total *= 3   # three columns per node share the same nnz count

    Tidx = Ti === nothing ? Int : Ti

    colptr = Vector{Tidx}(undef, ndof + 1)
    rowval = Vector{Tidx}(undef, nnz_total)
    nzval = zeros(Float64, nnz_total)

    # Pass 1 (count) → colptr by prefix sum.
    colptr[1] = 1
    @inbounds for jnode in 1:nnodes
        ncol = 3 * length(adj[jnode])
        base = 3 * (jnode - 1)
        for comp in 1:3
            col = base + comp
            colptr[col + 1] = colptr[col] + ncol
        end
    end

    # Pass 2 (fill) → rowval: for every column of a node, write the sorted DOF
    # rows of its neighbor stencil (each neighbor node contributes 3 DOFs).
    @inbounds for jnode in 1:nnodes
        nbrs = adj[jnode]
        base = 3 * (jnode - 1)
        for comp in 1:3
            col = base + comp
            p = colptr[col]
            for nb in nbrs              # nbrs sorted → rows written ascending
                r0 = 3 * (nb - 1)
                rowval[p]     = r0 + 1
                rowval[p + 1] = r0 + 2
                rowval[p + 2] = r0 + 3
                p += 3
            end
        end
    end

    K = SparseMatrixCSC{Float64,Tidx}(ndof, ndof, colptr, rowval, nzval)

    # element DOF map (Tidx)
    nelem = mesh.nelem
    edofs = Matrix{Tidx}(undef, 24, nelem)
    @inbounds for e in 1:nelem
        ed = element_dofs(mesh.elements, e)
        for i in 1:24
            edofs[i, e] = ed[i]
        end
    end

    colors = element_colors(mesh)
    return SparsityPattern{Tidx}(K, edofs, colors)
end

"""
    element_colors(mesh) -> Vector{Vector{Int}}

Analytic 8-color partition of the box elements by parity `(i%2,j%2,k%2)`
(SCALING.md §3.1). Same-color elements are ≥2 apart in every axis ⇒ share no
node ⇒ their scatters touch disjoint DOFs ⇒ race-free `@threads` assembly with no
atomics. Falls back to a single color (serial) for a non-box element count.
"""
function element_colors(mesh::Mesh)
    # recover (nx,ny,nz) from the structured numbering, if possible
    nxyz = _box_dims(mesh)
    if nxyz === nothing
        return [collect(1:mesh.nelem)]
    end
    nx, ny, nz = nxyz
    colors = [Int[] for _ in 1:8]
    e = 0
    @inbounds for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
        e += 1
        c = (i & 1) + 2 * (j & 1) + 4 * (k & 1) + 1
        push!(colors[c], e)
    end
    # drop empty colors (e.g. nx=ny=nz=1 ⇒ only 1 nonempty)
    return [c for c in colors if !isempty(c)]
end

# Recover (nx,ny,nz) from a box_mesh by inspecting node coordinates. Returns
# nothing if the mesh does not look like a structured axis-aligned box grid.
function _box_dims(mesh::Mesh)
    nodes = mesh.nodes
    nnodes = mesh.nnodes
    nelem = mesh.nelem
    nelem == 0 && return nothing
    xs = unique(round.(@view(nodes[1, :]); digits=12))
    ys = unique(round.(@view(nodes[2, :]); digits=12))
    zs = unique(round.(@view(nodes[3, :]); digits=12))
    nx = length(xs) - 1; ny = length(ys) - 1; nz = length(zs) - 1
    if nx >= 1 && ny >= 1 && nz >= 1 &&
       nx * ny * nz == nelem && (nx + 1) * (ny + 1) * (nz + 1) == nnodes
        return (nx, ny, nz)
    end
    return nothing
end

# Binary-search a CSC column [colstart,colend] for global row `grow`; returns the
# nzval index (or 0 if not present, which never happens for a correct pattern).
@inline function _find_nz(rowval, colstart::Integer, colend::Integer, grow::Integer)
    lo = colstart; hi = colend
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        rv = rowval[mid]
        if rv == grow
            return mid
        elseif rv < grow
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return zero(colstart)
end

"""
    assemble!(sp, mat, cache, U, εp, β, ᾱ, σ, R; commit=false, threaded=false)
        -> (K, R)

Assemble global tangent (into `sp.K.nzval`) and internal-force residual `R` from
element contributions + per-GP return maps (DESIGN §5.1, §7; SCALING.md §2.6).
O(1) heap allocation independent of nelem (the on-the-fly scatter does not
allocate per element). `R` is filled with F_int (the caller subtracts F_ext).

`threaded=true` uses the 8-color partition for race-free multithreaded assembly
(SCALING.md §3.1).
"""
function assemble!(sp::SparsityPattern, mat::J2Material, cache::ElementCache,
                   U::Vector{Float64},
                   εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                   σ::Matrix{Float64}, R::Vector{Float64};
                   commit::Bool=false,
                   kind::ElementKind=Hex8Small(),
                   Cp_inv::Union{Matrix{Float64},Nothing}=nothing)
    # Resolve `commit` to a *statically typed* Val via an explicit branch (not
    # `Val(commit)`, which would be a runtime-typed Union forcing a dynamic
    # dispatch that boxes the SparseMatrixCSC return — ~10 kB). The explicit
    # branch keeps both call sites fully inferred and the assembly O(1)-alloc,
    # preserving the v1 T20 `@allocated` bound. `kind` (a zero-size ElementKind)
    # statically selects the small- vs finite-strain element kernel.
    Cpi = Cp_inv === nothing ? σ : Cp_inv   # σ is a harmless placeholder for small
    if commit
        return _assemble!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cpi, R, Val(true))
    else
        return _assemble!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cpi, R, Val(false))
    end
end

function _assemble!(sp::SparsityPattern, mat::J2Material, kind::ElementKind,
                    cache::ElementCache, U::Vector{Float64},
                    εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                    σ::Matrix{Float64}, Cp_inv::Matrix{Float64}, R::Vector{Float64},
                    cv::Val{COMMIT}) where {COMMIT}
    fill!(sp.K.nzval, 0.0)
    fill!(R, 0.0)
    # Resolve the uniform/non-uniform geometry choice ONCE (outside the hot loop)
    # to a compile-time Val, so the inner loop has no geometry branch / type union
    # and stays allocation-free per element (SCALING.md §2.2, §2.6).
    if cache.uniform
        _assemble_loop!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cp_inv, R, cv, Val(true))
    else
        _assemble_loop!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cp_inv, R, cv, Val(false))
    end
    return sp.K, R
end

@inline function _elem_geom(cache::ElementCache, e::Integer, ::Val{true})
    return cache.Bref, cache.detJwref
end
@inline function _elem_geom(cache::ElementCache, e::Integer, ::Val{false})
    return element_geometry(cache, e)
end

# Per-element (Fe, Ke) dispatched on the ElementKind (static; no runtime branch).
# Small-strain reuses the v1 spatial B-matrix kernel; finite strain uses the
# reference shape gradients + the two-point P–F kernel (FINITE_STRAIN §4–5).
@inline function _elem_contribution(::Hex8Small, mat::J2Material, cache::ElementCache,
                                    e::Int, ue::SVector{24,Float64},
                                    εp, β, ᾱ, σ, Cp_inv, ::Val{COMMIT}, uv) where {COMMIT}
    Bs, detJw = _elem_geom(cache, e, uv)
    return element_force_tangent!(mat, Bs, detJw, ue, εp, β, ᾱ, e, σ, Val(COMMIT))
end
@inline function _elem_contribution(kind::ElementKind, mat::J2Material, cache::ElementCache,
                                    e::Int, ue::SVector{24,Float64},
                                    εp, β, ᾱ, σ, Cp_inv, ::Val{COMMIT}, uv) where {COMMIT}
    dNdXs = element_ref_grads(cache, e)
    _, detJw = _elem_geom(cache, e, uv)
    Xe = element_coords(cache.nodes, cache.elements, e)
    return element_force_tangent_finite!(kind, mat, dNdXs, detJw, ue, Xe,
                                         εp, β, ᾱ, Cp_inv, e, σ, Val(COMMIT))
end

function _assemble_loop!(sp::SparsityPattern, mat::J2Material, kind::ElementKind,
                         cache::ElementCache, U::Vector{Float64},
                         εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                         σ::Matrix{Float64}, Cp_inv::Matrix{Float64}, R::Vector{Float64},
                         cv::Val{COMMIT}, uv::Val{UNIFORM}) where {COMMIT,UNIFORM}
    nzval = sp.K.nzval
    colptr = sp.K.colptr
    rowval = sp.K.rowval
    edofs = sp.edofs
    nelem = size(edofs, 2)
    @inbounds for e in 1:nelem
        ue = SVector{24,Float64}(ntuple(i -> U[edofs[i, e]], Val(24)))
        Fe, Ke = _elem_contribution(kind, mat, cache, e, ue, εp, β, ᾱ, σ, Cp_inv, cv, uv)
        # scatter K via on-the-fly CSC column binary search (SCALING.md §2.3)
        for c in 1:24
            gcol = edofs[c, e]
            colstart = colptr[gcol]
            colend = colptr[gcol + 1] - 1
            for r in 1:24
                idx = _find_nz(rowval, colstart, colend, edofs[r, e])
                nzval[idx] += Ke[r, c]
            end
        end
        for i in 1:24
            R[edofs[i, e]] += Fe[i]
        end
    end
    return nothing
end

"""
    assemble_threaded!(sp, mat, cache, U, εp, β, ᾱ, σ, R; commit=false) -> (K, R)

Race-free 8-color multithreaded assembly (SCALING.md §3.1). Identical result to
the serial `assemble!`; uses `sp.colors` so same-color elements write disjoint
DOFs (no atomics). Run with `JULIA_NUM_THREADS > 1`.
"""
function assemble_threaded!(sp::SparsityPattern, mat::J2Material, cache::ElementCache,
                            U::Vector{Float64},
                            εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                            σ::Matrix{Float64}, R::Vector{Float64};
                            commit::Bool=false,
                            kind::ElementKind=Hex8Small(),
                            Cp_inv::Union{Matrix{Float64},Nothing}=nothing)
    # Static Val branch (see `assemble!`): avoids dynamic-dispatch boxing.
    Cpi = Cp_inv === nothing ? σ : Cp_inv
    if commit
        return _assemble_threaded_entry!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cpi, R, Val(true))
    else
        return _assemble_threaded_entry!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cpi, R, Val(false))
    end
end

function _assemble_threaded_entry!(sp::SparsityPattern, mat::J2Material, kind::ElementKind,
                                   cache::ElementCache, U::Vector{Float64},
                                   εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                                   σ::Matrix{Float64}, Cp_inv::Matrix{Float64}, R::Vector{Float64},
                                   cv::Val{COMMIT}) where {COMMIT}
    fill!(sp.K.nzval, 0.0)
    fill!(R, 0.0)
    _assemble_threaded!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cp_inv, R, cv)
    return sp.K, R
end

# Race-free threaded assembly: for each color (whose elements share no node, so
# they write disjoint nzval/R entries) thread over its element list. R is also
# race-free because same-color elements own disjoint DOFs (SCALING.md §3.1).
function _assemble_threaded!(sp::SparsityPattern, mat::J2Material, kind::ElementKind,
                             cache::ElementCache, U::Vector{Float64},
                             εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                             σ::Matrix{Float64}, Cp_inv::Matrix{Float64}, R::Vector{Float64},
                             ::Val{COMMIT}) where {COMMIT}
    # Resolve uniform/non-uniform ONCE to a compile-time Val (as the serial path
    # does), so the per-element call is fully inferred and allocation-free.
    uv = cache.uniform ? Val(true) : Val(false)
    for color in sp.colors
        Threads.@threads for ci in eachindex(color)
            e = color[ci]
            _assemble_one!(sp, mat, kind, cache, U, εp, β, ᾱ, σ, Cp_inv, R, Val(COMMIT), uv, e)
        end
    end
    return nothing
end

# Assemble a single element e (used by the threaded path). Allocation-free.
@inline function _assemble_one!(sp::SparsityPattern, mat::J2Material, kind::ElementKind,
                                cache::ElementCache, U::Vector{Float64},
                                εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                                σ::Matrix{Float64}, Cp_inv::Matrix{Float64}, R::Vector{Float64},
                                ::Val{COMMIT}, uv::Val{UNIFORM}, e::Integer) where {COMMIT,UNIFORM}
    nzval = sp.K.nzval
    colptr = sp.K.colptr
    rowval = sp.K.rowval
    edofs = sp.edofs
    @inbounds begin
        ue = SVector{24,Float64}(ntuple(i -> U[edofs[i, e]], Val(24)))
        Fe, Ke = _elem_contribution(kind, mat, cache, e, ue, εp, β, ᾱ, σ, Cp_inv,
                                    Val(COMMIT), uv)
        for c in 1:24
            gcol = edofs[c, e]
            colstart = colptr[gcol]
            colend = colptr[gcol + 1] - 1
            for r in 1:24
                grow = edofs[r, e]
                idx = _find_nz(rowval, colstart, colend, grow)
                nzval[idx] += Ke[r, c]
            end
        end
        for i in 1:24
            R[edofs[i, e]] += Fe[i]
        end
    end
    return nothing
end

end # module
