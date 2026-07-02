"""
    ParticlesMod

Struct-of-arrays material-point container (DESIGN §8) and particle sampling
(`sample_box`, `sample_region`, DESIGN §8 "Particle sampling"). Cache-friendly,
`StaticArrays`-only fields, mirroring the `lagrangian` style.
"""
module ParticlesMod

using StaticArrays

export Particles, sample_box, sample_region

const I3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
const CPINV0 = SVector{6,Float64}(1, 1, 1, 0, 0, 0)   # Cᵖ⁻¹ = I, Voigt

"""
    Particles

Struct-of-arrays material points (DESIGN §8). State initialization
(DESIGN §3.4) at construction: `F = I`, `Cᵖ⁻¹ = I` (`[1,1,1,0,0,0]`), `ᾱ = 0`,
`β_ref = 0`, `εp = 0`, `J = 1`, `C = 0`, `v = 0` — a zero `Cᵖ⁻¹` would make
`bᵉ_tr = 0` and the trial log strain singular, so this is never allowed.
"""
struct Particles
    x::Vector{SVector{3,Float64}}
    v::Vector{SVector{3,Float64}}
    C::Vector{SMatrix{3,3,Float64,9}}
    F::Vector{SMatrix{3,3,Float64,9}}
    Cp_inv::Vector{SVector{6,Float64}}
    εp::Vector{SVector{6,Float64}}
    β_ref::Vector{SVector{6,Float64}}
    ᾱ::Vector{Float64}
    τ::Vector{SVector{6,Float64}}
    m::Vector{Float64}
    V0::Vector{Float64}
    J::Vector{Float64}
end

Base.length(pts::Particles) = length(pts.x)

"""
    Particles(xs, V0, ρ)

Build a `Particles` container from a vector of positions `xs`, a common
reference volume `V0` per particle, and density `ρ` (DESIGN §3.4, §8):
`m = ρ·V0` for every particle. All history/kinematic state is initialized to
its DESIGN §3.4 rest value.
"""
function Particles(xs::Vector{SVector{3,Float64}}, V0::Float64, ρ::Float64)
    np = length(xs)
    return Particles(
        copy(xs),
        fill(zero(SVector{3,Float64}), np),
        fill(zero(SMatrix{3,3,Float64,9}), np),
        fill(I3, np),
        fill(CPINV0, np),
        fill(zero(SVector{6,Float64}), np),
        fill(zero(SVector{6,Float64}), np),
        zeros(Float64, np),
        fill(zero(SVector{6,Float64}), np),
        fill(ρ * V0, np),
        fill(V0, np),
        ones(Float64, np),
    )
end

"""
    sample_box(lo, hi, h; ppc=2, ρ=1.0) -> Particles

Fill the axis-aligned box `[lo, hi]` with a regular lattice of material
points, `ppc^3` particles per background cell of size `h` (baseline `ppc=2`,
i.e. `PPC=8`, DESIGN §8). Each particle carries reference volume
`V0 = (h/ppc)^3` and mass `m = ρ·V0`. `lo`, `hi` are `SVector{3,Float64}`.
"""
function sample_box(lo::SVector{3,Float64}, hi::SVector{3,Float64}, h::Float64;
                     ppc::Int=2, ρ::Float64=1.0)
    return sample_region(x -> true, lo, hi, h; ppc=ppc, ρ=ρ)
end

"""
    sample_region(pred, lo, hi, h; ppc=2, ρ=1.0) -> Particles

Like `sample_box`, but keep only lattice points satisfying `pred(x)::Bool`
(DESIGN §8's `sample_region(pred)` — the generic geometry-generator analogue
of `lagrangian`'s mesh generator). `lo`, `hi` bound the candidate lattice.
"""
function sample_region(pred, lo::SVector{3,Float64}, hi::SVector{3,Float64},
                        h::Float64; ppc::Int=2, ρ::Float64=1.0)
    dx = h / ppc
    nx = round(Int, (hi[1] - lo[1]) / dx)
    ny = round(Int, (hi[2] - lo[2]) / dx)
    nz = round(Int, (hi[3] - lo[3]) / dx)
    xs = SVector{3,Float64}[]
    for k in 0:(nz-1), j in 0:(ny-1), i in 0:(nx-1)
        x = lo + dx * SVector{3,Float64}(i + 0.5, j + 0.5, k + 0.5)
        pred(x) && push!(xs, x)
    end
    V0 = dx^3
    return Particles(xs, V0, ρ)
end

end # module
