"""
    GridMod

Uniform Cartesian background grid and the quadratic B-spline shape functions
(DESIGN.md §5, §8). The grid is fixed-size and reset every step; nodes are
1-based, column-major linear indices `i = ix + nx*((iy-1) + ny*(iz-1))`
(DESIGN §8).
"""
module GridMod

using StaticArrays

export Grid, node_index, node_coords, Stencil, bspline_stencil, ParticleOutOfBoundsError

"""
    ParticleOutOfBoundsError(p, x) <: Exception

Thrown when particle `p`'s 27-node B-spline stencil would need a grid node
outside `[origin, origin + n·h]` (DESIGN §8) — a setup error (grid too small),
never silently indexed around.
"""
struct ParticleOutOfBoundsError <: Exception
    p::Int
    x::SVector{3,Float64}
end

function Base.showerror(io::IO, err::ParticleOutOfBoundsError)
    print(io, "ParticleOutOfBoundsError: particle $(err.p) at x = $(err.x) has a ",
          "B-spline stencil that falls outside the grid extent. Pad the grid ",
          "(DESIGN §8) — this is a setup error, not a runtime state to tolerate.")
end

"""
    Grid(origin, h, n)

Uniform Cartesian background grid (DESIGN §8): `origin::SVector{3}` is the
position of node `(1,1,1)`, `h` the (isotropic) spacing, `n::NTuple{3,Int}`
nodes per axis. Nodal buffers (`m`, `p`, `f`, `v`) are preallocated once and
zeroed every step by `Step.step!`.
"""
struct Grid
    origin::SVector{3,Float64}
    h::Float64
    n::NTuple{3,Int}
    m::Vector{Float64}
    p::Vector{SVector{3,Float64}}
    f::Vector{SVector{3,Float64}}
    v::Vector{SVector{3,Float64}}
end

function Grid(origin::SVector{3,Float64}, h::Float64, n::NTuple{3,Int})
    N = n[1] * n[2] * n[3]
    return Grid(origin, h, n,
                zeros(Float64, N),
                fill(zero(SVector{3,Float64}), N),
                fill(zero(SVector{3,Float64}), N),
                fill(zero(SVector{3,Float64}), N))
end

"""
    node_index(grid, ix, iy, iz) -> Int

1-based column-major linear node index `i = ix + nx*((iy-1) + ny*(iz-1))`
(DESIGN §8). `ix,iy,iz` are 1-based node coordinates.
"""
@inline function node_index(g::Grid, ix::Int, iy::Int, iz::Int)
    nx, ny, _ = g.n
    return ix + nx * ((iy - 1) + ny * (iz - 1))
end

"""
    node_coords(grid, i) -> SVector{3,Float64}

Inverse of `node_index`: the spatial position of linear node index `i`.
"""
@inline function node_coords(g::Grid, i::Int)
    nx, ny, _ = g.n
    i0 = i - 1
    ix = i0 % nx
    iy = (i0 ÷ nx) % ny
    iz = i0 ÷ (nx * ny)
    return g.origin + g.h * SVector{3,Float64}(ix, iy, iz)
end

# --- quadratic B-spline weights (DESIGN §5) ---
#
# Per axis: base node b = floor(x/h - 1/2) (0-based node number), local
# coordinate ξ = x/h - b ∈ [1/2, 3/2). Weights over nodes b, b+1, b+2:
#   d0 = 3/2-ξ; w0 = 1/2 d0^2      d1 = ξ-1  ; w1 = 3/4-d1^2
#   d2 = ξ-1/2; w2 = 1/2 d2^2      (w0+w1+w2 = 1)
# Derivatives w.r.t. x (chain rule × 1/h): w0'=-d0/h, w1'=-2d1/h, w2'=d2/h.

@inline function _bspline_axis(xrel::Float64, h::Float64)
    b = floor(Int, xrel - 0.5)
    ξ = xrel - b
    d0 = 1.5 - ξ
    d1 = ξ - 1.0
    d2 = ξ - 0.5
    w = SVector{3,Float64}(0.5 * d0 * d0, 0.75 - d1 * d1, 0.5 * d2 * d2)
    dw = SVector{3,Float64}(-d0 / h, -2.0 * d1 / h, d2 / h)
    return b, w, dw
end

"""
    Stencil

The 27-node quadratic-B-spline stencil of a particle: 1-based grid linear
node indices `idx`, weights `w`, weight gradients `gradw`, and node positions
`pos` — all `SVector{27}` (fixed-size, stack-allocated, no heap alloc).
"""
struct Stencil
    idx::SVector{27,Int}
    w::SVector{27,Float64}
    gradw::SVector{27,SVector{3,Float64}}
    pos::SVector{27,SVector{3,Float64}}
end

"""
    bspline_stencil(grid, xp, p=0) -> Stencil

The 27-node (3×3×3) quadratic-B-spline stencil of particle at position `xp`
(DESIGN §5): weights `Sᵢ(xp)`, gradients `∇Sᵢ(xp)`, grid linear indices, and
node positions. Throws `ParticleOutOfBoundsError(p, xp)` if any stencil node
falls outside the grid (DESIGN §8). `p` is the particle index, used only for
the error message (pass `0` if unknown/not applicable, e.g. in unit tests).
Allocation-free.
"""
@inline function bspline_stencil(g::Grid, xp::SVector{3,Float64}, p::Int=0)
    h = g.h
    bx, wx, dwx = _bspline_axis((xp[1] - g.origin[1]) / h, h)
    by, wy, dwy = _bspline_axis((xp[2] - g.origin[2]) / h, h)
    bz, wz, dwz = _bspline_axis((xp[3] - g.origin[3]) / h, h)
    nx, ny, nz = g.n
    if bx < 0 || bx + 2 > nx - 1 || by < 0 || by + 2 > ny - 1 || bz < 0 || bz + 2 > nz - 1
        throw(ParticleOutOfBoundsError(p, xp))
    end
    idx = MVector{27,Int}(undef)
    w = MVector{27,Float64}(undef)
    gradw = MVector{27,SVector{3,Float64}}(undef)
    pos = MVector{27,SVector{3,Float64}}(undef)
    k = 0
    @inbounds for c in 0:2, b in 0:2, a in 0:2
        k += 1
        ix = bx + a + 1
        iy = by + b + 1
        iz = bz + c + 1
        wv = wx[a+1] * wy[b+1] * wz[c+1]
        gv = SVector{3,Float64}(dwx[a+1] * wy[b+1] * wz[c+1],
                                wx[a+1] * dwy[b+1] * wz[c+1],
                                wx[a+1] * wy[b+1] * dwz[c+1])
        idx[k] = node_index(g, ix, iy, iz)
        w[k] = wv
        gradw[k] = gv
        pos[k] = g.origin + h * SVector{3,Float64}(ix - 1, iy - 1, iz - 1)
    end
    return Stencil(SVector(idx), SVector(w), SVector(gradw), SVector(pos))
end

end # module
