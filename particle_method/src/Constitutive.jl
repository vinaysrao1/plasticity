"""
    Constitutive

Particle-level constitutive update (DESIGN §3, §4, §4.2): the F-update
(`Lₚ = Cₚ`), the optional F̄-at-particles anti-locking correction, and the
call into the **reused, unmodified** `lagrangian` finite-strain J2 kernel
(`PlasticityFEM.FiniteStrain.finite_kinematics` /
`PlasticityFEM.FiniteStrain.finite_stress_update`). This module owns the one
guard the kernel does not provide itself (DESIGN §3.3): `kin.ok || throw`.
"""
module Constitutive

using StaticArrays
using LinearAlgebra: det
using ..ParticlesMod: Particles
using ..GridMod: Grid
using PlasticityFEM.FiniteStrain: finite_kinematics, finite_stress_update
using PlasticityFEM.Materials: J2Material

export particle_stress_update!, update_particles!, ParticleInversionError,
       cell_Jbar!, ncells

const I3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

"""
    ParticleInversionError(p, J) <: Exception

Thrown when particle `p`'s deformation gradient is inverted (`J = det F ≤ 0`,
DESIGN §3.3): `finite_kinematics` reports `ok = false`. The kernel itself does
not throw on this (that guard lives in the FEM element assembly, not in
`FiniteStrain.jl`); the MPM particle loop must replicate it — silently
proceeding would (per DESIGN §3.3's trace through the kernel's `J ≤ 0`
fallback) reset the particle's entire plastic history to identity, a
clean-looking but wrong state.
"""
struct ParticleInversionError <: Exception
    p::Int
    J::Float64
end

function Base.showerror(io::IO, err::ParticleInversionError)
    print(io, "ParticleInversionError: particle $(err.p) is inverted ",
          "(det F = J = $(err.J) ≤ 0). The log-strain stress update requires ",
          "J > 0; reduce Δt, refine the grid, or add particle splitting.")
end

# --- F̄-at-particles cell binning (DESIGN §4.2, §5's "two different cells") ---
#
# Uses the plain background-cell index ⌊x/h⌋ — distinct from the B-spline
# stencil base ⌊x/h − 1/2⌋ (Grid.jl) — to group particles for volume-averaging.

"""
    ncells(grid) -> Int

Number of background cells `(nx-1)(ny-1)(nz-1)` used for F̄ cell binning
(DESIGN §4.2).
"""
ncells(grid::Grid) = (grid.n[1] - 1) * (grid.n[2] - 1) * (grid.n[3] - 1)

@inline function _cell_id(grid::Grid, x::SVector{3,Float64})
    nx, ny, nz = grid.n
    r = (x - grid.origin) / grid.h
    ix = clamp(floor(Int, r[1]), 0, nx - 2)
    iy = clamp(floor(Int, r[2]), 0, ny - 2)
    iz = clamp(floor(Int, r[3]), 0, nz - 2)
    return 1 + ix + (nx - 1) * (iy + (ny - 1) * iz)
end

"""
    cell_Jbar!(Jnum, Jden, grid, pts, Ftrial) -> nothing

Accumulate, per background cell, the volume-weighted mean dilation numerator
and denominator (DESIGN §4.2): `Jnum_c = Σ_{p∈c} Jₚ Vₚ⁰`, `Jden_c = Σ_{p∈c} Vₚ⁰`
so `J̄_c = Jnum_c / Jden_c`. `Ftrial` is each particle's F **after** this
step's `(I+ΔtCₚ)` update but **before** the F̄ correction. `Jnum`/`Jden` must
be preallocated with length `ncells(grid)`; zeroed here.
"""
function cell_Jbar!(Jnum::Vector{Float64}, Jden::Vector{Float64}, grid::Grid,
                     pts::Particles, Ftrial::Vector{SMatrix{3,3,Float64,9}})
    fill!(Jnum, 0.0)
    fill!(Jden, 0.0)
    @inbounds for p in eachindex(pts.x)
        c = _cell_id(grid, pts.x[p])
        V0 = pts.V0[p]
        Jnum[c] += det(Ftrial[p]) * V0
        Jden[c] += V0
    end
    return nothing
end

# Shared core: F is the FINAL deformation gradient for this step (already
# F̄-corrected if fbar is on). Runs the reused kernel and commits state.
@inline function _commit_stress!(pts::Particles, p::Int, mat::J2Material,
                                  F::SMatrix{3,3,Float64,9})
    kin = finite_kinematics(F, pts.Cp_inv[p])
    kin.ok || throw(ParticleInversionError(p, kin.J))
    τv, εp_new, β_ref_new, ᾱ_new, _, _, Cp_inv_new, _, _, _ =
        finite_stress_update(mat, kin, F, pts.εp[p], pts.β_ref[p], pts.ᾱ[p])
    pts.F[p] = F
    pts.εp[p] = εp_new
    pts.β_ref[p] = β_ref_new
    pts.ᾱ[p] = ᾱ_new
    pts.Cp_inv[p] = Cp_inv_new
    pts.τ[p] = τv
    pts.J[p] = det(F)
    return nothing
end

"""
    particle_stress_update!(pts, p, mat, dt; fbar=false, Jbar=NaN)

Single-particle constitutive update (DESIGN §4): `Lₚ = Cₚ` (the APIC affine
matrix already computed by `Transfer.g2p!` — **not** a separately computed
grid velocity gradient, per the design review, DESIGN §4), `Fₚ ← (I+ΔtLₚ)Fₚ`,
optionally the F̄ correction `Fₚ ← (J̄_c/Jₚ)^{1/3} Fₚ` (DESIGN §4.2, `fbar=true`
with a precomputed cell average `Jbar`), then the reused kernel. This is the
per-particle primitive used both by `update_particles!` (the full-grid batch
path, `Step.step!`) and by isolated tests (G1b: drive one particle through the
actual F-update loop with no grid at all).
"""
function particle_stress_update!(pts::Particles, p::Int, mat::J2Material, dt::Float64;
                                  fbar::Bool=false, Jbar::Float64=NaN)
    Ftrial = (I3 + dt * pts.C[p]) * pts.F[p]
    F = if fbar
        Jtrial = det(Ftrial)
        cbrt(Jbar / Jtrial) * Ftrial
    else
        Ftrial
    end
    _commit_stress!(pts, p, mat, F)
    return nothing
end

"""
    update_particles!(pts, mat, dt, grid; fbar=false, Ftrial=..., Jnum=..., Jden=...)

Batch particle constitutive update for the full grid step (DESIGN §4): the
F-update for every particle, the F̄ cell-averaging pass (if `fbar`), then the
per-particle kernel call. `Ftrial`, `Jnum`, `Jden` are reusable preallocated
buffers (owned by `Step.MPMModel`) so no heap allocation happens per step.
"""
function update_particles!(pts::Particles, mat::J2Material, dt::Float64, grid::Grid;
                            fbar::Bool=false,
                            Ftrial::Vector{SMatrix{3,3,Float64,9}},
                            Jnum::Vector{Float64},
                            Jden::Vector{Float64})
    np = length(pts.x)
    @inbounds for p in 1:np
        Ftrial[p] = (I3 + dt * pts.C[p]) * pts.F[p]
    end
    if fbar
        cell_Jbar!(Jnum, Jden, grid, pts, Ftrial)
        @inbounds for p in 1:np
            c = _cell_id(grid, pts.x[p])
            Jbar = Jnum[c] / Jden[c]
            Jtrial = det(Ftrial[p])
            F = cbrt(Jbar / Jtrial) * Ftrial[p]
            _commit_stress!(pts, p, mat, F)
        end
    else
        @inbounds for p in 1:np
            _commit_stress!(pts, p, mat, Ftrial[p])
        end
    end
    return pts
end

end # module
